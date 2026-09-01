//! One-call Anemoi AIR using the closed Flystel verification equations.
//!
//! For an open Flystel output `(u,v)` and its input `(x,y)`, let
//! `e = (y-v)^alpha`. The AIR checks
//!
//! ```text
//! x = e + beta*y^2 + gamma
//! u = e + beta*v^2 + delta
//! ```
//!
//! The first of those is `harness::gadgets::inverse_power_map` in Anemoi's clothing: the
//! open Flystel's inverse power is `y - v`, the committed cell it is built from
//! is `v`, and `x - beta*y^2 - gamma` is the target it is pinned against. So the
//! AIR never evaluates the inverse power that native evaluation uses — it
//! evaluates the forward one, once, and that is what makes the two equations
//! degree `alpha`, or three with one register.

use core::borrow::Borrow;

use harness::gadgets::inverse_power_map::eval_inverse_power_map;
use harness::permutation::assert_state_eq;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing};

use crate::columns::{AnemoiCols, Round, assert_layout, num_cols};
use crate::native::linear_layer;
use crate::params::AnemoiParams;

/// Constants for one independently constrained Anemoi call.
#[derive(Clone, Debug)]
pub struct AnemoiAir<
    F,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> {
    pub(crate) beta: F,
    pub(crate) gamma: F,
    pub(crate) delta: F,
    pub(crate) m_x: [[F; COLUMNS]; COLUMNS],
    pub(crate) m_y: [[F; COLUMNS]; COLUMNS],
    pub(crate) c: [[F; COLUMNS]; ROUNDS],
    pub(crate) d: [[F; COLUMNS]; ROUNDS],
}

impl<
    F: Copy,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> AnemoiAir<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>
{
    /// Build one degree/register variant from an instance.
    #[must_use]
    pub fn from_params(params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>) -> Self {
        assert_layout(WIDTH, COLUMNS, ALPHA, REGISTERS);
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        Self {
            beta: params.beta,
            gamma: params.gamma,
            delta: params.delta,
            m_x: params.m_x,
            m_y: params.m_y,
            c: params.c,
            d: params.d,
        }
    }
}

/// Maximum constraint degree of a variant.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, registers: usize) -> usize {
    match (alpha, registers) {
        (3 | 5 | 7, 0) => alpha as usize,
        (5 | 7, 1) => 3,
        _ => panic!("unsupported Anemoi degree/register variant"),
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> BaseAir<F> for AnemoiAir<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, COLUMNS, REGISTERS, ROUNDS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(ALPHA, REGISTERS))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ALPHA: u64,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, WIDTH, COLUMNS, REGISTERS>,
    c: &[AB::F; COLUMNS],
    d: &[AB::F; COLUMNS],
    m_x: &[[AB::F; COLUMNS]; COLUMNS],
    m_y: &[[AB::F; COLUMNS]; COLUMNS],
    beta: AB::F,
    gamma: AB::F,
    delta: AB::F,
    builder: &mut AB,
) {
    for i in 0..COLUMNS {
        state[i] += c[i].dup();
        state[COLUMNS + i] += d[i].dup();
    }
    linear_layer(state, m_x, m_y);

    for i in 0..COLUMNS {
        let x = state[i].dup();
        let y = state[COLUMNS + i].dup();
        let u = round.post[i];
        let v = round.post[COLUMNS + i];
        let v_expr: AB::Expr = v.into();

        // `v` is the witnessed cell; `y - v` is the inverse power the native
        // open Flystel computes, and this pins it against the value it must be
        // the alpha-th root of. That assertion is what makes `v` a constrained
        // cell rather than a free one.
        let e = eval_inverse_power_map::<AB, ALPHA, REGISTERS>(
            y.dup() - v_expr.dup(),
            x - y.square() * beta.dup() - gamma.dup(),
            &round.powers[i].0,
            builder,
        );

        // The second closed-Flystel equation pins the other committed cell.
        builder.assert_eq(u, e + v_expr.square() * beta.dup() + delta.dup());
        state[i] = u.into();
        state[COLUMNS + i] = v.into();
    }
}

/// Evaluate one call's columns. `vectorized.rs` reuses this exact function.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    air: &AnemoiAir<AB::F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>,
    builder: &mut AB,
    cols: &AnemoiCols<AB::Var, WIDTH, COLUMNS, REGISTERS, ROUNDS>,
) {
    let mut state: [AB::Expr; WIDTH] = cols.inputs.map(Into::into);
    for round in 0..ROUNDS {
        eval_round::<AB, WIDTH, COLUMNS, REGISTERS, ALPHA>(
            &mut state,
            &cols.rounds[round],
            &air.c[round],
            &air.d[round],
            &air.m_x,
            &air.m_y,
            air.beta.dup(),
            air.gamma.dup(),
            air.delta.dup(),
            builder,
        );
    }
    linear_layer(&mut state, &air.m_x, &air.m_y);
    // The final boundary cells are constrained, not merely exposed to tests.
    assert_state_eq(builder, &state, &cols.outputs);
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> Air<AB> for AnemoiAir<AB::F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &AnemoiCols<_, WIDTH, COLUMNS, REGISTERS, ROUNDS> = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
