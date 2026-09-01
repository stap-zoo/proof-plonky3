//! One-call Neptune AIR. Its external layer uses Plonky3's generic dense MDS
//! multiply; its `J + diag` internal layer reuses Poseidon2's linear-time path.
//!
//! # Degree, and the two independent places it comes from
//!
//! A round ends in [`commit_state`], so the AIR's maximum constraint degree is
//! the maximum degree of *one round function in its input*, and Neptune has two
//! kinds of round pulling in different directions:
//!
//! | round | unsplit | with registers |
//! |---|---|---|
//! | external (Lai--Massey) | 4 | 2, committing `(p0 - p1)²` per pair |
//! | internal (`x^alpha`) | `alpha` | 3 with one register, 2 with three |
//!
//! Cutting one and not the other cuts nothing: the maximum is over both. That
//! is why `LM` and `PREGS` are separate parameters that
//! [`assert_layout`](crate::columns::assert_layout) only admits together.
//!
//! There is no variant at degree three that splits the pair map differently:
//! the external round is four without its register and two with it, because the
//! register makes `u - v` affine and `(u - v)²` is the round's only remaining
//! multiplication.

use core::borrow::Borrow;

use harness::gadgets::power_map::eval_power_map;
use harness::permutation::{add_round_constants, commit_state};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeField64};
use p3_mds::util::mds_multiply;
use p3_poseidon2::matmul_internal;

use crate::columns::{ExternalRound, InternalRound, NeptuneCols, assert_layout, num_cols};
use crate::native::{external_sbox, external_sbox_first_square, external_sbox_pair};
use crate::params::NeptuneParams;

/// The degree the internal rounds reach for one power-map variant.
///
/// The gadget's own table (`harness::gadgets::power_map`), mirrored here so the
/// AIR can declare its degree in a `const` context.
#[must_use]
pub const fn internal_degree(alpha: u64, pregs: usize) -> usize {
    match (alpha, pregs) {
        (3, 0) => 3,
        (5, 0) => 5,
        (7, 0) => 7,
        (5 | 7, 1) => 3,
        (7, 3) => 2,
        _ => panic!("unsupported Neptune power-map variant"),
    }
}

/// The maximum constraint degree of one variant.
///
/// The external Lai--Massey map has degree four unless its first square is
/// committed, in which case it has degree two; the internal power map has
/// whatever [`internal_degree`] says. The AIR's degree is the larger, and a
/// variant that cuts only one of the two gains nothing.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, lm: usize, pregs: usize) -> usize {
    let external = if lm == 0 { 4 } else { 2 };
    let internal = internal_degree(alpha, pregs);
    if external > internal {
        external
    } else {
        internal
    }
}

/// One independently constrained Neptune permutation call.
#[derive(Clone, Debug)]
pub struct NeptuneAir<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
> {
    /// The reference-derived round constants and linear layers.
    pub params: NeptuneParams<F, WIDTH, EXT, INT>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
> NeptuneAir<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>
{
    /// Build the AIR. `HALF_EXT` repeats `EXT / 2` because stable Rust cannot
    /// express that quotient in an array length at this generic boundary.
    ///
    /// # Panics
    ///
    /// If the layout or the degree/register variant is not one Neptune admits.
    #[must_use]
    pub const fn new(params: NeptuneParams<F, WIDTH, EXT, INT>) -> Self {
        assert_layout(WIDTH, EXT, HALF_EXT, DEGREE, LM, PREGS);
        Self { params }
    }
}

impl<
    F: PrimeField64 + Sync,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
> BaseAir<F> for NeptuneAir<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, HALF_EXT, INT, LM, PREGS>()
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(DEGREE, LM, PREGS))
    }
}

/// One external round: the pair map, the dense MDS, the round constants, the
/// commitment.
///
/// POLICY §9, per pair: `cells.lm[j]` is prover-chosen and is pinned by the
/// `assert_eq` below before anything reads it. Skip that assertion and the pair
/// map becomes arbitrary, because every later use of the register is affine in
/// it — which is exactly what makes the round degree two.
fn eval_external_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const EXT: usize,
    const INT: usize,
    const LM: usize,
>(
    params: &NeptuneParams<AB::F, WIDTH, EXT, INT>,
    builder: &mut AB,
    state: &mut [AB::Expr; WIDTH],
    cells: &ExternalRound<AB::Var, WIDTH, LM>,
    round: usize,
) where
    AB::F: PrimeField64,
{
    let gamma: AB::Expr = params.gamma.dup().into();
    if LM == 0 {
        external_sbox(state, gamma);
    } else {
        for (pair, &register) in state.chunks_exact_mut(2).zip(cells.lm.iter()) {
            // This is the constraint that pins the witnessed first square.
            builder.assert_eq(register, external_sbox_first_square(&pair[0], &pair[1]));
            // Read back as the committed cell, never as the expression: that
            // substitution is the whole degree reduction, four down to two.
            let (out0, out1) =
                external_sbox_pair(pair[0].dup(), pair[1].dup(), register.into(), gamma.dup());
            pair[0] = out0;
            pair[1] = out1;
        }
    }
    mds_multiply(state, &params.m_ext);
    add_round_constants(state, &params.rcons[round + 1]);
    // POLICY §9: `post` is prover-chosen, and the assertion inside
    // `commit_state` is what pins it before the next round continues from it.
    commit_state(builder, state, &cells.post);
}

/// One internal round: the single S-box, the `J + diag` layer, the round
/// constants, the commitment.
///
/// POLICY §9: `cells.powers` is pinned inside the power-map gadget, which owns
/// every assertion tying the chain to `state[0]`.
fn eval_internal_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const PREGS: usize,
>(
    params: &NeptuneParams<AB::F, WIDTH, EXT, INT>,
    builder: &mut AB,
    state: &mut [AB::Expr; WIDTH],
    cells: &InternalRound<AB::Var, WIDTH, PREGS>,
    round: usize,
) where
    AB::F: PrimeField64,
{
    state[0] = eval_power_map::<AB, DEGREE, PREGS>(state[0].dup(), &cells.powers, builder);
    matmul_internal(state, params.m_int_diag_m_1);
    add_round_constants(state, &params.rcons[round + 1]);
    commit_state(builder, state, &cells.post);
}

/// Evaluate one call's independent lane.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
>(
    air: &NeptuneAir<AB::F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>,
    builder: &mut AB,
    cols: &NeptuneCols<AB::Var, WIDTH, HALF_EXT, INT, LM, PREGS>,
) where
    AB::F: PrimeField64,
{
    let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| cols.inputs[i].into());
    mds_multiply(&mut state, &air.params.m_ext);
    let mut round = 0;
    for cells in &cols.first {
        eval_external_round(&air.params, builder, &mut state, cells, round);
        round += 1;
    }
    for cells in &cols.internal {
        eval_internal_round::<AB, WIDTH, EXT, INT, DEGREE, PREGS>(
            &air.params,
            builder,
            &mut state,
            cells,
            round,
        );
        round += 1;
    }
    for cells in &cols.last {
        eval_external_round(&air.params, builder, &mut state, cells, round);
        round += 1;
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
> Air<AB> for NeptuneAir<AB::F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &NeptuneCols<_, WIDTH, HALF_EXT, INT, LM, PREGS> = main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
