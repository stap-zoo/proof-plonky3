//! Lookup-backed Monolith constraints for one complete call per main row.
//!
//! A Bar contributes either one width-two `(input, output)` query per chunk or
//! one width-four query per adjacent pair. Thus the lookup itself range-binds
//! every coordinate and replaces both the baseline's input-bit range proof and
//! its bit-level chi computation. Recomposition is still only equality in the
//! AIR field, so every recomposed word additionally carries the canonicity
//! argument of [`harness::gadgets::canonical_word`], which rules out the second
//! integer encodings congruent to a field element.
//!
//! What is Monolith's here is the round structure and which bus answers which
//! chunk. The chunk widths, the recomposition and the canonicity argument
//! belong to the field, and the granularity and packing axes belong to the
//! measurement; all three come from `harness`.

use core::borrow::Borrow;

use harness::gadgets::canonical_word::{WordKind, eval_canonicity, recompose};
use harness::lookup::{FractionPacking, LookupGranularity};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_lookup::{InteractionBuilder, LookupBus};
use p3_mds::{MdsPermutation, util::mds_multiply};
use p3_monolith::{Monolith, MonolithBars};

use crate::logup::columns::{MonolithLogupBar, MonolithLogupCols, MonolithLogupRound, num_cols};
use crate::logup::tables::{BYTE_BUS, BYTE_PAIR_BUS, BYTE_SEVEN_PAIR_BUS, SEVEN_BIT_BUS};

/// Bus answering the `index`-th chunk of a word. Which chunk widths exist is
/// the field's business ([`WordKind`]); which bus carries them is Monolith's.
const fn chunk_bus(kind: WordKind, index: usize) -> &'static str {
    match kind {
        WordKind::Mersenne31 if index == 3 => SEVEN_BIT_BUS,
        WordKind::Goldilocks | WordKind::Mersenne31 => BYTE_BUS,
    }
}

/// Bus answering the `pair`-th adjacent chunk pair of a word.
const fn pair_bus(kind: WordKind, pair: usize) -> &'static str {
    match kind {
        WordKind::Mersenne31 if pair == 1 => BYTE_SEVEN_PAIR_BUS,
        WordKind::Goldilocks | WordKind::Mersenne31 => BYTE_PAIR_BUS,
    }
}

/// Constants for one lookup-backed Monolith call.
#[derive(Debug, Clone)]
pub struct MonolithLogupAir<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> {
    pub(crate) round_constants: [[F; WIDTH]; NUM_FULL_ROUNDS],
    pub(crate) mds_matrix: [[F; WIDTH]; WIDTH],
    pub(crate) kind: WordKind,
    granularity: LookupGranularity,
    packing: FractionPacking,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> MonolithLogupAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>
{
    /// Construct from the same upstream native object used by the wrapped AIR.
    pub(crate) fn from_native<B, Mds>(
        native: Monolith<F, B, Mds, WIDTH, NUM_FULL_ROUNDS>,
        kind: WordKind,
        granularity: LookupGranularity,
        packing: FractionPacking,
    ) -> Self
    where
        B: MonolithBars<F, WIDTH>,
        Mds: MdsPermutation<F, WIDTH>,
    {
        const { assert!(NUM_BARS <= WIDTH) };
        // The chunk layout belongs to the field, and the gadget owns it; only
        // the Bars count per round is Monolith's own.
        assert_eq!(F::ORDER_U64, kind.modulus());
        assert_eq!(NUM_CHUNKS, kind.num_chunks());
        assert_eq!(CANONICAL_CELLS, kind.canonical_cells());
        match kind {
            WordKind::Goldilocks => assert_eq!(NUM_BARS, 4),
            WordKind::Mersenne31 => assert_eq!(NUM_BARS, 8),
        }
        assert_eq!(B::NUM_BARS, NUM_BARS);
        if granularity == LookupGranularity::AdjacentPair {
            assert_eq!(
                NUM_CHUNKS % 2,
                0,
                "adjacent lookup chunks must pair exactly"
            );
        }
        let mds_matrix = extract_mds_matrix(&native.mds);
        Self {
            round_constants: native.round_constants,
            mds_matrix,
            kind,
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
    /// same-bus fraction packing.
    #[must_use]
    pub const fn raw_queries_per_call(&self) -> usize {
        (NUM_FULL_ROUNDS + 1) * NUM_BARS * NUM_CHUNKS / self.granularity.chunks_per_query()
    }

    /// Fraction columns after the selected same-bus packing.
    ///
    /// The current Monolith instances split Mersenne-31's high-chunk bus into
    /// a count divisible by every supported packing factor, so no partially
    /// filled terminal column is hidden by this total.
    #[must_use]
    pub const fn fraction_columns_per_call_air(&self) -> usize {
        self.raw_queries_per_call() / self.packing.denominators_per_column()
    }

    /// Once-per-proof fixed-table rows required by this granularity.
    #[must_use]
    pub const fn fixed_table_rows(&self) -> usize {
        match (self.kind, self.granularity) {
            (WordKind::Goldilocks, LookupGranularity::Byte) => 1 << 8,
            (WordKind::Goldilocks, LookupGranularity::AdjacentPair) => 1 << 16,
            (WordKind::Mersenne31, LookupGranularity::Byte) => (1 << 8) + (1 << 7),
            (WordKind::Mersenne31, LookupGranularity::AdjacentPair) => (1 << 16) + (1 << 15),
        }
    }

    /// Calls at which paired chunks break even with byte chunks in the raw
    /// `fixed table rows + call-side queries` model.
    ///
    /// This is an analytic crossover, not a prover-time measurement: table
    /// commitments, auxiliary columns, quotient degree and openings are not
    /// interchangeable units in the actual proof system.
    #[must_use]
    pub const fn raw_granularity_crossover_calls(&self) -> usize {
        match self.kind {
            WordKind::Goldilocks => ((1 << 16) - (1 << 8)) / (192 - 96),
            WordKind::Mersenne31 => ((1 << 16) + (1 << 15) - (1 << 8) - (1 << 7)) / (192 - 96),
        }
    }
}

fn extract_mds_matrix<F, Mds, const WIDTH: usize>(mds: &Mds) -> [[F; WIDTH]; WIDTH]
where
    F: PrimeCharacteristicRing + Copy,
    Mds: MdsPermutation<F, WIDTH>,
{
    let columns: [[F; WIDTH]; WIDTH] = core::array::from_fn(|column| {
        let mut basis = [F::ZERO; WIDTH];
        basis[column] = F::ONE;
        mds.permute_mut(&mut basis);
        basis
    });
    core::array::from_fn(|row| core::array::from_fn(|column| columns[column][row]))
}

impl<
    F: PrimeField64 + Sync,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> BaseAir<F>
    for MonolithLogupAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        // `p3-batch-stark` uses this degree budget to fold same-bus
        // interactions without crossing the selected quotient bucket.
        Some(self.packing.max_constraint_degree())
    }
}

fn eval_bar<AB, const NUM_CHUNKS: usize, const CANONICAL_CELLS: usize>(
    input: AB::Expr,
    cols: &MonolithLogupBar<AB::Var, NUM_CHUNKS, CANONICAL_CELLS>,
    kind: WordKind,
    granularity: LookupGranularity,
    builder: &mut AB,
) -> AB::Expr
where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    let recomposed_input = recompose::<AB>(&cols.input_chunks, kind);
    let recomposed_output = recompose::<AB>(&cols.output_chunks, kind);
    builder.assert_eq(recomposed_input, input);
    // Recomposition is equality in the field; these rule out the second
    // integer encodings it accepts, on chunks the lookups below range-bind.
    eval_canonicity(&cols.input_chunks, &cols.input_canonicity, kind, builder);
    eval_canonicity(&cols.output_chunks, &cols.output_canonicity, kind, builder);

    match granularity {
        LookupGranularity::Byte => {
            for (index, (&input_chunk, &output_chunk)) in cols
                .input_chunks
                .iter()
                .zip(&cols.output_chunks)
                .enumerate()
            {
                // This one relation pins both prover-chosen chunks to their
                // ranges and to the upstream fixed Bars map.
                LookupBus::new(chunk_bus(kind, index)).lookup_key(
                    builder,
                    [input_chunk.into(), output_chunk.into()],
                    1,
                );
            }
        }
        LookupGranularity::AdjacentPair => {
            for index in (0..NUM_CHUNKS).step_by(2) {
                // Keep all four coordinates separate. A packed field-element
                // key would have collisions unless another constraint
                // independently range-bound each coordinate.
                LookupBus::new(pair_bus(kind, index / 2)).lookup_key(
                    builder,
                    [
                        cols.input_chunks[index].into(),
                        cols.input_chunks[index + 1].into(),
                        cols.output_chunks[index].into(),
                        cols.output_chunks[index + 1].into(),
                    ],
                    1,
                );
            }
        }
    }
    recomposed_output
}

fn eval_round<
    AB,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &MonolithLogupRound<AB::Var, WIDTH, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
    mds_matrix: &[[AB::F; WIDTH]; WIDTH],
    round_constants: Option<&[AB::F; WIDTH]>,
    kind: WordKind,
    granularity: LookupGranularity,
    builder: &mut AB,
) where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    for (word, cols) in state.iter_mut().take(NUM_BARS).zip(&round.bars) {
        *word = eval_bar(word.clone(), cols, kind, granularity, builder);
    }

    let mut post_bricks = core::array::from_fn(|i| {
        if i == 0 {
            state[0].clone()
        } else {
            state[i].clone() + state[i - 1].clone().square()
        }
    });
    mds_multiply(&mut post_bricks, mds_matrix);
    if let Some(constants) = round_constants {
        for (word, constant) in post_bricks.iter_mut().zip(constants) {
            *word += *constant;
        }
    }
    for (computed, &committed) in post_bricks.into_iter().zip(&round.post) {
        builder.assert_eq(computed, committed);
    }
    *state = round.post.map(Into::into);
}

/// Evaluate one call. Keeping this free function makes the row layout reusable.
pub fn eval<
    AB,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    air: &MonolithLogupAir<AB::F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
    builder: &mut AB,
    cols: &MonolithLogupCols<
        AB::Var,
        WIDTH,
        NUM_FULL_ROUNDS,
        NUM_BARS,
        NUM_CHUNKS,
        CANONICAL_CELLS,
    >,
) where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    let mut state = cols.inputs.map(Into::into);
    mds_multiply(&mut state, &air.mds_matrix);
    for round in 0..NUM_FULL_ROUNDS {
        eval_round(
            &mut state,
            &cols.full_rounds[round],
            &air.mds_matrix,
            Some(&air.round_constants[round]),
            air.kind,
            air.granularity,
            builder,
        );
    }
    eval_round(
        &mut state,
        &cols.final_round,
        &air.mds_matrix,
        None,
        air.kind,
        air.granularity,
        builder,
    );
}

impl<
    AB,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> Air<AB> for MonolithLogupAir<AB::F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>
where
    AB: AirBuilder + InteractionBuilder,
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &MonolithLogupCols<
            _,
            WIDTH,
            NUM_FULL_ROUNDS,
            NUM_BARS,
            NUM_CHUNKS,
            CANONICAL_CELLS,
        > = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
