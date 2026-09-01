//! Lookup-backed Tip5 constraints for one complete call per main row.
//!
//! A split-and-lookup word contributes either one width-two
//! `(input byte, output byte)` query per byte or one width-four query per
//! adjacent pair. The query is what range-binds every coordinate *and* what
//! evaluates the byte permutation, so it replaces the baseline's bit
//! decomposition, its `(x + 1)^3 - 1 = y + 257 q` relation and its sixteen-bit
//! quotient in one move.
//!
//! What it does **not** replace is the recomposition argument. Two things are
//! still ours:
//!
//! * `recompose(input bytes) == mont_R * x` is equality in the AIR field, so it
//!   accepts every 64-bit integer congruent to `mont_R * x`, not only the
//!   canonical representative the reference decomposes. Since the bytes are the
//!   lookup keys, an accepted second encoding would query different rows and
//!   return a different "S-box output". [`harness::gadgets::canonical_word`]'s
//!   two-cell Goldilocks argument is what rules the second encodings out.
//! * The output side needs no counterpart. Its bytes are a function of the
//!   queried input bytes, and the reference's own definition reduces the
//!   recomposed 64-bit output into the field, so `F::from_u64(transformed)` *is*
//!   the specification. There is no prover choice left to pin.
//!
//! Everything else is the baseline's, reused rather than restated: the input
//! and output boundary cells, the five-round schedule, the MDS matrix, the
//! round constants, and [`crate::air::eval_power_word`] for the seventh-power
//! words — the same evaluator the lookup-free AIR calls, so the two
//! arithmetizations cannot disagree about the part the lookup does not touch.
//!
//! `air.rs` and `generation.rs` will drift. Keep them line-for-line parallel —
//! same order, same helper names, same per-round split (POLICY §6).

use core::borrow::Borrow;

use harness::gadgets::canonical_word::{eval_canonicity, recompose};
use harness::lookup::{FractionPacking, LookupGranularity};
use harness::permutation::{add_round_constants, assert_state_eq};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, Field, PrimeCharacteristicRing, PrimeField64};
use p3_lookup::{InteractionBuilder, LookupBus};
use p3_mds::util::mds_multiply;

use crate::air::eval_power_word;
use crate::columns::assert_layout;
use crate::logup::columns::{KIND, Round, SplitWord, Tip5LogupCols, num_cols};
use crate::logup::tables::{BYTE_BUS, BYTE_PAIR_BUS};
use crate::params::{BYTES, MONT_R, ROUNDS, SPLIT_WORDS, Tip5Params};

/// Constants for one lookup-backed Tip5 call.
#[derive(Clone, Debug)]
pub struct Tip5LogupAir<F, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize> {
    pub(crate) params: Tip5Params<F, WIDTH>,
    granularity: LookupGranularity,
    packing: FractionPacking,
}

/// Maximum constraint degree of one register × fraction-packing point.
///
/// Two independent sources of degree meet here. The seventh-power words give 7
/// unsplit and 3 with one register ([`crate::air::max_constraint_degree`]); a
/// LogUp fraction column folding `n` same-bus denominators gives `n + 1`. The
/// AIR's degree is the larger, which is why the registered frontier is not the
/// full product: at packing 8 the split and unsplit variants both land on 9,
/// and below packing 2 the split variant's own 3 already dominates.
#[must_use]
pub const fn max_constraint_degree(registers: usize, packing: FractionPacking) -> usize {
    let power = crate::air::max_constraint_degree(registers);
    let lookup = packing.max_constraint_degree();
    if power > lookup { power } else { lookup }
}

impl<F: PrimeField64, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>
    Tip5LogupAir<F, WIDTH, POWER_WORDS, REGISTERS>
{
    /// Build one register × granularity × packing variant from exact exported
    /// parameters — the same [`Tip5Params`] the baseline AIR is built from.
    #[must_use]
    pub fn from_params(
        params: &Tip5Params<F, WIDTH>,
        granularity: LookupGranularity,
        packing: FractionPacking,
    ) -> Self {
        assert_layout(WIDTH, POWER_WORDS, REGISTERS);
        assert_eq!(params.alpha, 7);
        // The byte layout belongs to the field, and the gadget owns it; a Tip5
        // split word is exactly one Goldilocks word's worth of table keys.
        assert_eq!(F::ORDER_U64, KIND.modulus());
        assert_eq!(BYTES, KIND.num_chunks());
        if granularity == LookupGranularity::AdjacentPair {
            assert_eq!(BYTES % 2, 0, "adjacent lookup bytes must pair exactly");
        }
        Self {
            params: params.clone(),
            granularity,
            packing,
        }
    }

    /// Lookup message granularity selected for this AIR variant.
    #[must_use]
    pub const fn granularity(&self) -> LookupGranularity {
        self.granularity
    }

    /// Same-bus fraction-column packing selected for this AIR variant.
    #[must_use]
    pub const fn packing(&self) -> FractionPacking {
        self.packing
    }

    /// Call-side lookup messages emitted by one permutation row, before
    /// same-bus fraction packing: `5 * 4 * 8 = 160`, or 80 when paired.
    #[must_use]
    pub const fn raw_queries_per_call(&self) -> usize {
        ROUNDS * SPLIT_WORDS * BYTES / self.granularity.chunks_per_query()
    }

    /// Fraction columns after the selected same-bus packing.
    ///
    /// Both query counts are divisible by every supported packing factor, so
    /// no partially filled terminal column is hidden by this total.
    #[must_use]
    pub const fn fraction_columns_per_call_air(&self) -> usize {
        self.raw_queries_per_call() / self.packing.denominators_per_column()
    }

    /// Once-per-proof fixed-table rows required by this granularity.
    #[must_use]
    pub const fn fixed_table_rows(&self) -> usize {
        match self.granularity {
            LookupGranularity::Byte => 1 << 8,
            LookupGranularity::AdjacentPair => 1 << 16,
        }
    }

    /// Calls at which paired bytes break even with single bytes in the raw
    /// `fixed table rows + call-side queries` model.
    ///
    /// This is an analytic crossover, not a prover-time measurement: table
    /// commitments, auxiliary columns, quotient degree and openings are not
    /// interchangeable units in the actual proof system.
    #[must_use]
    pub const fn raw_granularity_crossover_calls(&self) -> usize {
        ((1 << 16) - (1 << 8)) / (ROUNDS * SPLIT_WORDS * BYTES / 2)
    }
}

impl<F: PrimeField64 + Sync, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>
    BaseAir<F> for Tip5LogupAir<F, WIDTH, POWER_WORDS, REGISTERS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, POWER_WORDS, REGISTERS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        // `p3-batch-stark` uses this degree budget to fold same-bus
        // interactions without crossing the selected quotient bucket.
        Some(max_constraint_degree(REGISTERS, self.packing))
    }
}

/// Pin one split-and-lookup word and return its S-box output as an affine
/// expression, exactly as the baseline's `eval_split_word` does.
fn eval_split_word<AB>(
    input: AB::Expr,
    cols: &SplitWord<AB::Var>,
    granularity: LookupGranularity,
    builder: &mut AB,
) -> AB::Expr
where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    // Pins the eight input bytes to the word the reference decomposes.
    let recomposed_input = recompose::<AB>(&cols.input_bytes, KIND);
    builder.assert_eq(recomposed_input, input * AB::F::from_u64(MONT_R));
    // Recomposition is equality in the field; this rules out the second
    // encodings in [p, 2^64) it accepts, on bytes the lookups below range-bind.
    eval_canonicity(&cols.input_bytes, &cols.input_canonicity, KIND, builder);

    match granularity {
        LookupGranularity::Byte => {
            for (&input_byte, &output_byte) in cols.input_bytes.iter().zip(&cols.output_bytes) {
                // This one relation pins both prover-chosen bytes to their
                // ranges and to the reference's byte permutation.
                LookupBus::new(BYTE_BUS).lookup_key(
                    builder,
                    [input_byte.into(), output_byte.into()],
                    1,
                );
            }
        }
        LookupGranularity::AdjacentPair => {
            for byte in (0..BYTES).step_by(2) {
                // Keep all four coordinates separate. A packed field-element
                // key would have collisions unless another constraint
                // independently range-bound each coordinate.
                LookupBus::new(BYTE_PAIR_BUS).lookup_key(
                    builder,
                    [
                        cols.input_bytes[byte].into(),
                        cols.input_bytes[byte + 1].into(),
                        cols.output_bytes[byte].into(),
                        cols.output_bytes[byte + 1].into(),
                    ],
                    1,
                );
            }
        }
    }

    // No output canonicity: the bytes are pinned by the queries above, and the
    // reference's own definition reduces this 64-bit integer into the field.
    recompose::<AB>(&cols.output_bytes, KIND) * AB::F::from_u64(MONT_R).inverse()
}

/// One round, kept in the same order as `generation::generate_round`.
fn eval_round<AB, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, POWER_WORDS, REGISTERS>,
    params: &Tip5Params<AB::F, WIDTH>,
    round_index: usize,
    granularity: LookupGranularity,
    builder: &mut AB,
) where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    for (word, cols) in state.iter_mut().take(SPLIT_WORDS).zip(&round.split) {
        *word = eval_split_word(word.dup(), cols, granularity, builder);
    }
    for (word, cols) in state.iter_mut().skip(SPLIT_WORDS).zip(&round.powers) {
        *word = eval_power_word(word.dup(), cols, builder);
    }
    mds_multiply(state, &params.m);
    add_round_constants(state, &params.rcons[round_index]);
}

/// Evaluate one call. Keeping this a free function makes the layout reusable.
pub fn eval<AB, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>(
    air: &Tip5LogupAir<AB::F, WIDTH, POWER_WORDS, REGISTERS>,
    builder: &mut AB,
    cols: &Tip5LogupCols<AB::Var, WIDTH, POWER_WORDS, REGISTERS>,
) where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    let mut state = cols.inputs.map(Into::into);
    for round in 0..ROUNDS {
        eval_round(
            &mut state,
            &cols.rounds[round],
            &air.params,
            round,
            air.granularity,
            builder,
        );
    }
    // The final boundary cells are constrained, not merely exposed to tests.
    assert_state_eq(builder, &state, &cols.outputs);
}

impl<AB, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize> Air<AB>
    for Tip5LogupAir<AB::F, WIDTH, POWER_WORDS, REGISTERS>
where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &Tip5LogupCols<_, WIDTH, POWER_WORDS, REGISTERS> = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
