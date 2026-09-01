//! The AIR: constants only, plus a free `eval` over one call's columns.
//!
//! POLICY §6: the AIR struct holds constants and nothing else; `Air::eval` is a
//! thin wrapper around a **free** [`eval`] taking one call's columns. That free
//! function is what `vectorized.rs` reuses, and the reuse is the reason the two
//! layouts cannot diverge.
//!
//! `air.rs` and `generation.rs` will drift. Keep them line-for-line parallel —
//! same order, same helper names, same per-half-round split: `eval_half_round`
//! here against `generate_half_round` there, with the constant row added by the
//! caller in both.
//!
//! # What pins what (POLICY §9)
//!
//! Every cell is prover-chosen, including the ones `generation.rs` writes. Per
//! half-round `h`, with `x = M * post[h-1] + rcons[h-1]` (and `x = inputs` at
//! `h = 0`):
//!
//! | cell | pinned by |
//! |---|---|
//! | `post[i]`, forward half-round | `post[i] == x_i^alpha`, via `harness::gadgets::power_map` |
//! | `post[i]`, inverse half-round | `post[i]^alpha == x_i`, via `harness::gadgets::inverse_power_map` |
//! | `powers[i]` | the power map inside whichever of the two ran |
//! | `outputs` | `outputs == M * post_last + rcons_last` |
//!
//! `inputs` is pinned by being the argument of the first half-round's
//! constraints. Nothing is left over: the negative test in `tests/air.rs`
//! corrupts every cell in turn and expects each one to be rejected.
//!
//! **The inverse half-round is where a reader should look twice.** Its committed
//! cell is a root, and a root is free money until `root^alpha == x` is asserted
//! (POLICY §9). The assertion lives in `harness::gadgets::inverse_power_map`, and it is
//! the same assertion the forward half-round makes with its two sides swapped —
//! which is exactly why the expensive direction costs nothing here.
//!
//! # Degree
//!
//! `x` is affine in the previous half-round's cells and the S-box is the only
//! non-linear step, so the AIR's degree is the power map's: `alpha` unsplit,
//! three with one register at alpha 5 or 7. Unlike Griffin's Horst layer or
//! Neptune's Lai--Massey, nothing here sets a floor above the S-box, so **alpha
//! 3 gives a degree-3 AIR with no register at all**.

use core::borrow::Borrow;

use harness::gadgets::inverse_power_map::eval_inverse_power_map;
use harness::gadgets::power_map::eval_power_map;
use harness::permutation::{add_round_constants, assert_state_eq};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeField64};
use p3_mds::util::mds_multiply;

use crate::half_round::columns::{HalfRound, RescuePrimeCols, assert_layout, num_cols};
use crate::params::RescuePrimeParams;

/// Constants for one independently constrained Rescue-Prime call.
#[derive(Clone, Debug)]
pub struct RescuePrimeAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> {
    pub(crate) params: RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> RescuePrimeAir<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>
{
    /// Build one degree/register variant from an instance.
    ///
    /// # Panics
    ///
    /// If the variant is unsupported, or `ALPHA` disagrees with the parameters.
    #[must_use]
    pub fn from_params(params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>) -> Self {
        assert_layout(WIDTH, ALPHA, REGISTERS, HALF_ROUNDS);
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        Self {
            params: params.clone(),
        }
    }
}

/// Maximum constraint degree of a variant.
///
/// The power map's, with nothing above it: no product of committed cells
/// survives the round function, so the S-box is the whole story.
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
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> BaseAir<F> for RescuePrimeAir<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, REGISTERS, HALF_ROUNDS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(ALPHA, REGISTERS))
    }
}

/// One half-round: pin this half-round's committed S-box output, then advance
/// the state expression through the linear layer.
///
/// `forward` selects the direction, and the two branches are the same power map
/// with its arguments exchanged — which is the arithmetization's whole point.
/// The constant row is the caller's, because the half-round index is.
#[inline]
fn eval_half_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    state: &mut [AB::Expr; WIDTH],
    half_round: &HalfRound<AB::Var, WIDTH, REGISTERS>,
    forward: bool,
    params: &RescuePrimeParams<AB::F, WIDTH, HALF_ROUNDS>,
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    for ((word, &post), registers) in state
        .iter()
        .zip(half_round.post.iter())
        .zip(half_round.powers.iter())
    {
        if forward {
            // `post` is the forward power of the state word.
            let power = eval_power_map::<AB, ALPHA, REGISTERS>(word.dup(), &registers.0, builder);
            builder.assert_eq(power, post);
        } else {
            // `post` is a witnessed alpha-th root, and this is the constraint
            // that makes it a value rather than a free cell.
            eval_inverse_power_map::<AB, ALPHA, REGISTERS>(
                post.into(),
                word.dup(),
                &registers.0,
                builder,
            );
        }
    }

    // The committed cells become the next half-round's input, through the
    // linear layer.
    *state = core::array::from_fn(|i| half_round.post[i].into());
    mds_multiply(state, &params.m);
}

/// Evaluate one call's columns. `vectorized.rs` reuses this exact function.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    air: &RescuePrimeAir<AB::F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>,
    builder: &mut AB,
    cols: &RescuePrimeCols<AB::Var, WIDTH, REGISTERS, HALF_ROUNDS>,
) where
    AB::F: PrimeField64,
{
    // No leading layer: the input state is the first S-box's argument.
    let mut state: [AB::Expr; WIDTH] = cols.inputs.map(Into::into);

    for half_round in 0..HALF_ROUNDS {
        eval_half_round::<AB, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
            &mut state,
            &cols.half_rounds[half_round],
            half_round.is_multiple_of(2),
            &air.params,
            builder,
        );
        add_round_constants(&mut state, &air.params.rcons[half_round]);
    }

    // The final boundary cells are constrained, not merely exposed to tests.
    assert_state_eq(builder, &state, &cols.outputs);
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
> Air<AB> for RescuePrimeAir<AB::F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &RescuePrimeCols<_, WIDTH, REGISTERS, HALF_ROUNDS> =
            main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
