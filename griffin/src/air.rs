//! The AIR: constants only, plus a free `eval` over one call's columns.
//!
//! POLICY §6: the AIR struct holds constants and nothing else; `Air::eval` is a
//! thin wrapper around a **free** `eval(air, builder, cols)` taking one call's
//! columns. That free function is what `vectorized.rs` reuses, and the reuse is
//! the reason the two layouts cannot diverge.
//!
//! `air.rs` and `generation.rs` will drift. Keep them line-for-line parallel —
//! same order, same helper names, same per-round split: `eval_round` here
//! against `generate_round` there, and the leading `mds_multiply` at the same
//! place in both.
//!
//! # What pins what (POLICY §9)
//!
//! Every cell is prover-chosen, including the ones `generation.rs` writes. Per
//! round, in this order:
//!
//! | cell | pinned by |
//! |---|---|
//! | `post[0]`, the witnessed alpha-th root | `post[0]^alpha == x_0`, via `harness::gadgets::inverse_power_map` |
//! | `powers[0]` | the power map inside that gadget |
//! | `post[1]` | `post[1] == x_1^alpha`, via `harness::gadgets::power_map` |
//! | `powers[1]` | the power map inside it |
//! | `post[i]`, `i >= 2` | `post[i] == x_i * G_i(L_i)` |
//! | `outputs` | `outputs == M * post_last`, after the last round |
//!
//! `inputs` is pinned by being the argument of the first round's constraints.
//! Nothing is left over: the negative test in `tests/air.rs` corrupts every cell
//! in turn and expects each one to be rejected.
//!
//! # Degree
//!
//! `x` is affine in the previous round's cells, `L_i` is affine in this one's,
//! so `G_i(L_i)` is degree two and the Horst product degree three. The Horst
//! layer therefore sets a **floor of three** on the whole AIR, exactly as
//! Neptune's Lai--Massey layer sets a floor of four on the variants that commit
//! nothing inside it: the unsplit variant costs `max(alpha, 3)` and one register
//! brings alpha 5 or 7 down to three, but no variant offered here goes below
//! three. As Neptune's register variants show, such a floor is a property of
//! what a layout commits rather than of the round function: committing
//! `G_i(L_i)` would take the Horst product to two, at a cell per word per round.

use core::borrow::Borrow;

use harness::gadgets::inverse_power_map::eval_inverse_power_map;
use harness::gadgets::power_map::eval_power_map;
use harness::permutation::{add_round_constants, assert_state_eq};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing, PrimeField64};
use p3_mds::util::mds_multiply;

use crate::columns::{GriffinCols, Round, assert_layout, num_cols};
use crate::params::GriffinParams;

/// Constants for one independently constrained Griffin call.
#[derive(Clone, Debug)]
pub struct GriffinAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> {
    pub(crate) params: GriffinParams<F, WIDTH, ROUNDS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> GriffinAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    /// Build one degree/register variant from an instance.
    ///
    /// # Panics
    ///
    /// If the variant is unsupported, or `ALPHA` disagrees with the parameters.
    #[must_use]
    pub fn from_params(params: &GriffinParams<F, WIDTH, ROUNDS>) -> Self {
        assert_layout(WIDTH, ALPHA, REGISTERS);
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        Self {
            params: params.clone(),
        }
    }
}

/// Maximum constraint degree of a variant.
///
/// The `3` is the Horst layer's floor, not the power map's: `x_i * G_i(L_i)` is
/// degree three however cheap the S-box is, which is why alpha 3 gains nothing
/// from a register and is not offered one.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, registers: usize) -> usize {
    match (alpha, registers) {
        (3, 0) => 3,
        (5 | 7, 0) => alpha as usize,
        (5 | 7, 1) => 3,
        _ => panic!("unsupported Griffin degree/register variant"),
    }
}

impl<
    F: PrimeField64 + Sync,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> BaseAir<F> for GriffinAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
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

/// One round: pin this round's committed non-linear output, then advance the
/// state expression through the linear layer and the round constants.
#[inline]
fn eval_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, WIDTH, REGISTERS>,
    params: &GriffinParams<AB::F, WIDTH, ROUNDS>,
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    let y_0: AB::Expr = round.post[0].into();
    let y_1: AB::Expr = round.post[1].into();

    // `y_0` is the witnessed alpha-th root of `x_0`; this is the constraint that
    // makes it a value rather than a free cell.
    eval_inverse_power_map::<AB, ALPHA, REGISTERS>(
        y_0.dup(),
        state[0].dup(),
        &round.powers[0].0,
        builder,
    );
    // `y_1` is the forward power, committed so that `L_i` stays affine.
    let power = eval_power_map::<AB, ALPHA, REGISTERS>(state[1].dup(), &round.powers[1].0, builder);
    builder.assert_eq(power, y_1.dup());

    for i in 2..WIDTH {
        // `x_{i-1}`, the round *input* word — and none at i = 2.
        let z = if i == 2 {
            AB::Expr::ZERO
        } else {
            state[i - 1].dup()
        };
        let l = y_0.dup() * AB::F::from_usize(i - 1) + y_1.dup() + z;
        builder.assert_eq(round.post[i], state[i].dup() * params.quadratic(i, l));
    }

    // The committed cells become the next round's input, through the linear
    // layer. The caller adds this round's constants, because the round index is
    // the caller's — `generate_round` is split the same way.
    *state = core::array::from_fn(|i| round.post[i].into());
    mds_multiply(state, &params.m);
}

/// Evaluate one call's columns. `vectorized.rs` reuses this exact function.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    air: &GriffinAir<AB::F, WIDTH, REGISTERS, ROUNDS, ALPHA>,
    builder: &mut AB,
    cols: &GriffinCols<AB::Var, WIDTH, REGISTERS, ROUNDS>,
) where
    AB::F: PrimeField64,
{
    let mut state: [AB::Expr; WIDTH] = cols.inputs.map(Into::into);

    // `_pre_rounds`: one linear layer before the first round.
    mds_multiply(&mut state, &air.params.m);

    for round in 0..ROUNDS {
        eval_round::<AB, WIDTH, REGISTERS, ROUNDS, ALPHA>(
            &mut state,
            &cols.rounds[round],
            &air.params,
            builder,
        );
        add_round_constants(&mut state, &air.params.rcons[round]);
    }

    // The final boundary cells are constrained, not merely exposed to tests.
    assert_state_eq(builder, &state, &cols.outputs);
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> Air<AB> for GriffinAir<AB::F, WIDTH, REGISTERS, ROUNDS, ALPHA>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &GriffinCols<_, WIDTH, REGISTERS, ROUNDS> = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
