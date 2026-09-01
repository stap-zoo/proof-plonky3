//! The AIR: constants only, plus a free `eval` over one call's columns.
//!
//! POLICY §6: the AIR struct holds constants and nothing else; `Air::eval` is a
//! thin wrapper around a **free** `eval(air, builder, cols)` taking one call's
//! columns. That free function is what `vectorized.rs` reuses, and the reuse is
//! the reason the two layouts cannot diverge.
//!
//! `air.rs` and `generation.rs` will drift. Keep them line-for-line parallel —
//! same order, same helper names, same per-round split — and paired per gadget.
//!
//! **Name the constraint that pins each witnessed value, where it is used**
//! (POLICY §9). Every trace cell is prover-chosen: a committed `x^3` is free
//! money until the `assert_eq` ties it to `x`, and so is every limb, every
//! inverse-power output, every canonicity flag. This layout keeps a complete
//! call in one row, so inputs and outputs are explicit boundary cells and no
//! transition can be silently omitted.

use core::borrow::Borrow;

use harness::gadgets::power_map::eval_power_map;
use harness::permutation::{add_round_constants, assert_state_eq};
use p3_air::utils::pack_bits_le;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, Field, PrimeCharacteristicRing, PrimeField64};
use p3_mds::util::mds_multiply;

use crate::columns::{MATCH_FLAGS, PowerWord, Round, SplitWord, Tip5Cols, assert_layout, num_cols};
use crate::params::{BYTES, MONT_R, ROUNDS, SPLIT_WORDS, Tip5Params};

/// Constants for one independently constrained call.
#[derive(Clone, Debug)]
pub struct Tip5Air<F, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize> {
    pub(crate) params: Tip5Params<F, WIDTH>,
}

impl<F: PrimeField64, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>
    Tip5Air<F, WIDTH, POWER_WORDS, REGISTERS>
{
    /// Build one degree/register variant from exact exported parameters.
    #[must_use]
    pub fn from_params(params: &Tip5Params<F, WIDTH>) -> Self {
        assert_layout(WIDTH, POWER_WORDS, REGISTERS);
        assert_eq!(params.alpha, 7);
        Self {
            params: params.clone(),
        }
    }
}

/// Maximum constraint degree of the seventh-power register variant.
#[must_use]
pub const fn max_constraint_degree(registers: usize) -> usize {
    match registers {
        0 => 7,
        1 => 3,
        _ => panic!("Tip5 supports zero or one seventh-power register"),
    }
}

impl<F: PrimeField64 + Sync, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>
    BaseAir<F> for Tip5Air<F, WIDTH, POWER_WORDS, REGISTERS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, POWER_WORDS, REGISTERS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(REGISTERS))
    }
}

/// Enforce that a 64-bit decomposition is the canonical Goldilocks encoding.
///
/// This is the same MSB-to-LSB paired-prefix walk used by
/// `p3-monolith-air`. Goldilocks has 33 one-bits, hence sixteen committed pair
/// flags and the final bit folded into the closing assertion.
fn eval_canonical_bits<AB: AirBuilder>(
    bits: &[AB::Expr; 64],
    flags: &[AB::Var; MATCH_FLAGS],
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    debug_assert_eq!(AB::F::ORDER_U64, 0xffff_ffff_0000_0001);
    let mut prev = AB::Expr::ONE;
    let mut flag = 0;
    let mut pending: Option<AB::Expr> = None;
    for i in (0..64).rev() {
        let bit = bits[i].dup();
        if ((AB::F::ORDER_U64 >> i) & 1) == 1 {
            if let Some(first) = pending.take() {
                let next: AB::Expr = flags[flag].into();
                builder.assert_eq(next.dup(), prev * first * bit);
                prev = next;
                flag += 1;
            } else {
                pending = Some(bit);
            }
        } else {
            builder.assert_zero(prev.dup() * bit);
        }
    }
    debug_assert_eq!(flag, MATCH_FLAGS);
    builder.assert_zero(prev * pending.expect("Goldilocks has odd Hamming weight"));
}

/// Pin one split-and-lookup word and return its S-box output as an affine expression.
fn eval_split_word<AB: AirBuilder>(
    input: AB::Expr,
    cols: &SplitWord<AB::Var>,
    builder: &mut AB,
) -> AB::Expr
where
    AB::F: PrimeField64,
{
    let flat_input: [AB::Var; 64] = core::array::from_fn(|i| cols.input_bits[i / 8][i % 8]);
    builder.assert_bools(flat_input);
    let input_bits = flat_input.map(Into::into);
    let reconstructed: AB::Expr = pack_bits_le(input_bits.iter().cloned());
    builder.assert_eq(reconstructed, input * AB::F::from_u64(MONT_R));
    eval_canonical_bits(&input_bits, &cols.match_flags, builder);

    let mut transformed = AB::Expr::ZERO;
    for byte in 0..BYTES {
        builder.assert_bools(cols.output_bits[byte]);
        builder.assert_bools(cols.quotient_bits[byte]);
        let digit: AB::Expr = pack_bits_le(cols.input_bits[byte].map(Into::into).into_iter());
        let output: AB::Expr = pack_bits_le(cols.output_bits[byte].map(Into::into).into_iter());
        let quotient: AB::Expr = pack_bits_le(cols.quotient_bits[byte].map(Into::into).into_iter());
        let shifted = digit + AB::Expr::ONE;
        builder.assert_eq(
            shifted.dup() * shifted.dup() * shifted - AB::Expr::ONE,
            output.dup() + quotient * AB::F::from_u64(257),
        );
        transformed += output * AB::F::from_u64(1u64 << (8 * byte));
    }
    transformed * AB::F::from_u64(MONT_R).inverse()
}

/// Pin one seventh-power word. `pub(crate)` because the lookup-backed
/// arithmetization ([`crate::logup`]) leaves these words untouched and calls
/// this very function, rather than restating a round-function rule twice.
pub(crate) fn eval_power_word<AB: AirBuilder, const REGISTERS: usize>(
    input: AB::Expr,
    cols: &PowerWord<AB::Var, REGISTERS>,
    builder: &mut AB,
) -> AB::Expr {
    let power = eval_power_map::<AB, 7, REGISTERS>(input, &cols.registers, builder);
    builder.assert_eq(power, cols.output);
    cols.output.into()
}

/// One round, kept in the same order as `generation::generate_round`.
fn eval_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, POWER_WORDS, REGISTERS>,
    params: &Tip5Params<AB::F, WIDTH>,
    round_index: usize,
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    for (word, cols) in state.iter_mut().take(SPLIT_WORDS).zip(&round.split) {
        *word = eval_split_word(word.dup(), cols, builder);
    }
    for (word, cols) in state.iter_mut().skip(SPLIT_WORDS).zip(&round.powers) {
        *word = eval_power_word(word.dup(), cols, builder);
    }
    mds_multiply(state, &params.m);
    add_round_constants(state, &params.rcons[round_index]);
}

/// Evaluate one call's columns. The vectorized AIR reuses this function.
pub fn eval<AB: AirBuilder, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>(
    air: &Tip5Air<AB::F, WIDTH, POWER_WORDS, REGISTERS>,
    builder: &mut AB,
    cols: &Tip5Cols<AB::Var, WIDTH, POWER_WORDS, REGISTERS>,
) where
    AB::F: PrimeField64,
{
    let mut state = cols.inputs.map(Into::into);
    for round in 0..ROUNDS {
        eval_round(&mut state, &cols.rounds[round], &air.params, round, builder);
    }
    // The final boundary cells are constrained, not merely exposed to tests.
    assert_state_eq(builder, &state, &cols.outputs);
}

impl<AB: AirBuilder, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize> Air<AB>
    for Tip5Air<AB::F, WIDTH, POWER_WORDS, REGISTERS>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &Tip5Cols<_, WIDTH, POWER_WORDS, REGISTERS> = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_air::check_constraints;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;

    #[derive(Debug)]
    struct CanonicalAir;

    impl BaseAir<Goldilocks> for CanonicalAir {
        fn width(&self) -> usize {
            64 + MATCH_FLAGS
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            vec![]
        }
    }

    impl<AB: AirBuilder<F = Goldilocks>> Air<AB> for CanonicalAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();
            let bits: [AB::Expr; 64] = core::array::from_fn(|i| row[i].into());
            let flags: [AB::Var; MATCH_FLAGS] = core::array::from_fn(|i| row[64 + i]);
            builder.assert_bools(core::array::from_fn::<_, 64, _>(|i| row[i]));
            // Both zero and the integer p represent zero in the field. This
            // equation cannot distinguish them; only the prefix walk can.
            let reconstructed: AB::Expr = pack_bits_le(bits.iter().cloned());
            builder.assert_zero(reconstructed);
            eval_canonical_bits(&bits, &flags, builder);
        }
    }

    fn trace(value: u64) -> RowMajorMatrix<Goldilocks> {
        let mut row = (0..64)
            .map(|i| Goldilocks::from_bool(((value >> i) & 1) != 0))
            .collect::<Vec<_>>();
        let mut prev = true;
        let mut pending = None;
        for i in (0..64).rev() {
            if ((Goldilocks::ORDER_U64 >> i) & 1) == 1 {
                let bit = ((value >> i) & 1) != 0;
                if let Some(first) = pending.take() {
                    prev = prev && first && bit;
                    row.push(Goldilocks::from_bool(prev));
                } else {
                    pending = Some(bit);
                }
            }
        }
        assert_eq!(row.len(), 64 + MATCH_FLAGS);
        RowMajorMatrix::new(row, 64 + MATCH_FLAGS)
    }

    /// `p` is arithmetically the same field element as zero; the canonical
    /// walk must nevertheless reject that alternate 64-bit representation.
    #[test]
    fn noncanonical_modulus_encoding_is_rejected() {
        check_constraints(&CanonicalAir, &trace(0), &[]);
        assert!(
            std::panic::catch_unwind(|| {
                check_constraints(&CanonicalAir, &trace(Goldilocks::ORDER_U64), &[]);
            })
            .is_err()
        );
    }
}
