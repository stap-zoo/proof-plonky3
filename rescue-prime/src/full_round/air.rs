//! The AIR: constants only, plus a free `eval` over one call's columns.
//!
//! POLICY §6, kept line-for-line parallel with `generation.rs` in this module:
//! `eval_round` against `generate_round`, the forward half then the inverse
//! half in both, the two constant rows read in the same order.
//!
//! # What pins what (POLICY §9)
//!
//! Per round `r`, with `a` the incoming committed state (`inputs` at `r = 0`)
//! and `post` this round's:
//!
//! | cell | pinned by |
//! |---|---|
//! | `forward[i]` | `register == a_i^2` (or `^3`), inside `harness::gadgets::power_map` |
//! | `inverse[i]` | `register == w_i^2` (or `^3`), same gadget on the affine `w_i` |
//! | `post[i]` | `w_i^alpha == (M * a^alpha + c_forward)_i`, where `w = M^{-1}(post - c_inverse)` |
//!
//! The last row of that table is the whole layout. `post` is pinned *through*
//! `M^{-1}`, so the constraint that ties it down is a statement about a linear
//! combination of the round's committed cells rather than about one of them —
//! and because `M^{-1}` is invertible, pinning every combination pins every
//! cell. `inputs` is pinned by being the right-hand side of round 0.
//!
//! There is no boundary constraint on the output: the last round's `post` *is*
//! the output, already pinned by that round's equation.
//!
//! # Degree
//!
//! `alpha` unsplit, three with one register — the same as the half-round layout,
//! which is the point of this layout existing. The left side is a power map of a
//! degree-1 form and the right a degree-`alpha` combination; they are compared,
//! never multiplied.

use core::borrow::Borrow;

use harness::gadgets::inverse_power_map::eval_inverse_power_map;
use harness::gadgets::power_map::eval_power_map;
use harness::permutation::add_round_constants;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeField64};
use p3_mds::util::mds_multiply;

use crate::full_round::columns::{FullRoundCols, Round, assert_layout, num_cols};
use crate::params::RescuePrimeParams;

/// Constants for one independently constrained Rescue-Prime call, full-round
/// layout.
#[derive(Clone, Debug)]
pub struct FullRoundAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> {
    pub(crate) params: RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> FullRoundAir<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>
{
    /// Build one degree/register variant from an instance.
    ///
    /// # Panics
    ///
    /// If the variant is unsupported, if `HALF_ROUNDS != 2 * ROUNDS`, or if
    /// `ALPHA` disagrees with the parameters.
    #[must_use]
    pub fn from_params(params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>) -> Self {
        assert_layout(WIDTH, ALPHA, REGISTERS, HALF_ROUNDS, ROUNDS);
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        Self {
            params: params.clone(),
        }
    }
}

/// Maximum constraint degree of a variant — the power map's, as in the
/// half-round layout.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, registers: usize) -> usize {
    match (alpha, registers) {
        (3, 0) => 3,
        (5 | 7, 0) => alpha as usize,
        (5 | 7, 1) => 3,
        _ => panic!("unsupported Rescue-Prime degree/register variant"),
    }
}

impl<
    F: PrimeField64 + Sync,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> BaseAir<F> for FullRoundAir<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, REGISTERS, ROUNDS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(ALPHA, REGISTERS))
    }
}

/// One full round: one equation per word, tying this round's committed state to
/// the next through both S-boxes at once.
#[inline]
fn eval_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, WIDTH, REGISTERS>,
    forward_constants: &[AB::F; WIDTH],
    inverse_constants: &[AB::F; WIDTH],
    params: &RescuePrimeParams<AB::F, WIDTH, HALF_ROUNDS>,
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    // Forward half, evaluated *forwards*: `u = M * a^alpha + c`.
    let mut forward: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
        eval_power_map::<AB, ALPHA, REGISTERS>(state[i].dup(), &round.forward[i].0, builder)
    });
    mds_multiply(&mut forward, &params.m);
    add_round_constants(&mut forward, forward_constants);

    // Inverse half, evaluated *backwards*: the value the inverse S-box produced
    // is `w = M^{-1}(post - c)`, affine in this round's committed cells. This is
    // the step that costs `m_inv` and buys the halved cell count.
    let mut inverse: [AB::Expr; WIDTH] =
        core::array::from_fn(|i| round.post[i].into() - inverse_constants[i]);
    mds_multiply(&mut inverse, &params.m_inv);

    // The one constraint per word: the inverse half's output, raised back
    // through the forward power map, is the forward half's output. Without it
    // every `post` cell is free (POLICY §9).
    for ((root, target), registers) in inverse.into_iter().zip(forward).zip(&round.inverse) {
        eval_inverse_power_map::<AB, ALPHA, REGISTERS>(root, target, &registers.0, builder);
    }

    // The committed state becomes the next round's input.
    *state = core::array::from_fn(|i| round.post[i].into());
}

/// Evaluate one call's columns. `vectorized.rs` reuses this exact function.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    air: &FullRoundAir<AB::F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>,
    builder: &mut AB,
    cols: &FullRoundCols<AB::Var, WIDTH, REGISTERS, ROUNDS>,
) where
    AB::F: PrimeField64,
{
    let mut state: [AB::Expr; WIDTH] = cols.inputs.map(Into::into);

    for round in 0..ROUNDS {
        eval_round::<AB, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
            &mut state,
            &cols.rounds[round],
            &air.params.rcons[2 * round],
            &air.params.rcons[2 * round + 1],
            &air.params,
            builder,
        );
    }

    // No output boundary constraint: the last round's `post` is the permutation
    // output, and that round's equation is what pins it (see the module docs).
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> Air<AB> for FullRoundAir<AB::F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &FullRoundCols<_, WIDTH, REGISTERS, ROUNDS> = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
