//! The AIR: constants only, plus a free `eval` over one call's columns.
//!
//! POLICY §6: the AIR struct holds constants and nothing else; `Air::eval` is a
//! thin wrapper around a **free** `eval(air, builder, cols)` taking one call's
//! columns. That free function is what `vectorized.rs` reuses, and the reuse is
//! the reason the two layouts cannot diverge.
//!
//! `air.rs` and `generation.rs` will drift. Keep them line-for-line parallel —
//! same order, same helper names, same per-cycle split — and paired per gadget.
//!
//! **Name the constraint that pins each witnessed value, where it is used**
//! (POLICY §9). Every trace cell is prover-chosen: a committed `x^3` is free
//! money until the `assert_eq` ties it to `x`, and so is every limb, every
//! inverse-power output, every canonicity flag. First row, last row, and the
//! transitions into and out of a call in a multi-row layout are asserted, not
//! assumed.
//!
//! # What pins what
//!
//! Per cycle, in this order:
//!
//! | cell | pinned by |
//! |---|---|
//! | `forward_powers` | `register == x^{(alpha-1)/2}`, inside `harness::gadgets::power_map` |
//! | `backward`, the witnessed root | `backward^alpha == target`, via `harness::gadgets::inverse_power_map` |
//! | `backward_powers` | the forward power map inside that gadget |
//! | `extension_powers` | one equality per register against the P3 input, below |
//! | `post` | `post == P3(input)`, coordinate by coordinate |
//! | `outputs` | the final `M`-plus-constant boundary |
//!
//! `inputs` is pinned by being the argument of the first cycle's constraints.

use core::borrow::Borrow;

use harness::gadgets::inverse_power_map::eval_inverse_power_map;
use harness::gadgets::power_map::eval_power_map;
use harness::permutation::{add_round_constants, assert_state_eq};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeField64};
use p3_mds::util::mds_multiply;

use crate::columns::{Cycle, XHashCols, assert_layout, num_cols};
use crate::native::{
    extension_half_power, extension_power_from_cpolys, extension_power_from_half,
    extension_power_from_quadratics, extension_quadratics,
};
use crate::params::XHashParams;

/// Constants for one independently constrained XHash call.
#[derive(Clone, Debug)]
pub struct XHashAir<
    F,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
> {
    pub(crate) params: XHashParams<F, WIDTH, CONSTANT_ROWS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
> XHashAir<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>
{
    /// Build one exact variant and one degree/register choice.
    ///
    /// # Panics
    ///
    /// If the layout is unsupported, if the instance disagrees with `ALPHA` or
    /// `ACTIVE` — or if `P3_BLOCKS == 1` for an instance whose exported
    /// coordinate table is **not** the power map of its declared quotient. That
    /// last one is not a formality: the three-cell P3 basis computes
    /// `(x^{(alpha-1)/2})^2 * x` in `F_p[X]/(f)`, so on a table that is not that
    /// map the AIR would constrain a different permutation than the oracle
    /// evaluates, and no known-answer test on the *native* side would notice.
    #[must_use]
    pub fn from_params(params: &XHashParams<F, WIDTH, CONSTANT_ROWS>) -> Self {
        assert_layout(WIDTH, ACTIVE, ALPHA, REGISTERS, P3_BLOCKS, CYCLES);
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        assert_eq!(params.skip_middle, ACTIVE != WIDTH);
        assert!(
            P3_BLOCKS != 1 || params.structured,
            "{}: the structured P3 basis needs the exported coordinate table to \
             be x^alpha in the declared quotient, and it is not (see ERROR.md)",
            params.name
        );
        Self {
            params: params.clone(),
        }
    }
}

/// Maximum constraint degree of the direct and register layouts.
///
/// The P3 layer and the base-field S-box reach the same degree in both bases:
/// unsplit it is `alpha`, and one register brings either to three.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, registers: usize) -> usize {
    match (alpha, registers) {
        (5 | 7, 0) => alpha as usize,
        (5 | 7, 1) => 3,
        _ => panic!("unsupported XHash degree/register variant"),
    }
}

impl<
    F: PrimeField64 + Sync,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
> BaseAir<F> for XHashAir<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(ALPHA, REGISTERS))
    }
}

/// The P3 layer over one triple, in whichever register basis the variant uses.
///
/// Both bases pin every register they consume before it enters the output
/// expression; an unpinned register would make the whole layer arbitrary
/// (POLICY §9).
#[inline]
fn eval_p3_triple<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    input: [AB::Expr; 3],
    registers: &[crate::columns::ExtensionPower<AB::Var, REGISTERS, P3_BLOCKS>],
    params: &XHashParams<AB::F, WIDTH, CONSTANT_ROWS>,
    builder: &mut AB,
) -> [AB::Expr; 3]
where
    AB::F: PrimeField64,
{
    if REGISTERS == 0 {
        // Degree alpha, straight from the reference's own coordinate table.
        return extension_power_from_cpolys(input, &params.cpolys);
    }
    if P3_BLOCKS == 1 {
        // Three cells: the committed `x^{(alpha-1)/2}` of the declared quotient,
        // pinned coordinate-wise before its square multiplies the input.
        let expected = extension_half_power::<_, _, ALPHA>(
            input.each_ref().map(Dup::dup),
            &params.reduction,
            &params.reduction_x4,
        );
        let half: [AB::Expr; 3] = core::array::from_fn(|coordinate| {
            let register = registers[coordinate].0[0][0];
            builder.assert_eq(expected[coordinate].dup(), register);
            register.into()
        });
        extension_power_from_half(input, half, &params.reduction, &params.reduction_x4)
    } else {
        // Six cells: the quadratic monomial fallback, which needs no modulus
        // and remains available for rejecting or diagnosing a malformed table.
        let expected = extension_quadratics(input.each_ref().map(Dup::dup));
        let quadratics: [AB::Expr; 6] = core::array::from_fn(|index| {
            let register = registers[index / 2].0[index % 2][0];
            builder.assert_eq(expected[index].dup(), register);
            register.into()
        });
        extension_power_from_quadratics(input, quadratics, &params.cpolys)
    }
}

#[inline]
fn eval_cycle<
    AB: AirBuilder,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    state: &mut [AB::Expr; WIDTH],
    cycle: &Cycle<AB::Var, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS>,
    cycle_index: usize,
    params: &XHashParams<AB::F, WIDTH, CONSTANT_ROWS>,
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    let first = 3 * cycle_index;

    // F: add, mix, then apply x^alpha to every word. Every optional
    // register is pinned inside `eval_power_map` before it enters the B-step
    // expression.
    add_round_constants(state, &params.rcons[first]);
    mds_multiply(state, &params.m);
    for (word, powers) in state.iter_mut().zip(&cycle.forward_powers) {
        *word = eval_power_map::<AB, ALPHA, REGISTERS>(word.dup(), &powers.0, builder);
    }

    // B: mix and add, then witness the inverse power. The root is not trusted:
    // `eval_inverse_power_map` pins `backward^alpha == target`; skipped
    // coordinates are pinned by equality instead.
    mds_multiply(state, &params.m);
    add_round_constants(state, &params.rcons[first + 1]);
    let mut active = 0;
    for (i, word) in state.iter().enumerate() {
        if params.skip_middle && i % 3 == 1 {
            builder.assert_eq(word.dup(), cycle.backward[i]);
        } else {
            eval_inverse_power_map::<AB, ALPHA, REGISTERS>(
                cycle.backward[i].into(),
                word.dup(),
                &cycle.backward_powers[active].0,
                builder,
            );
            active += 1;
        }
    }
    debug_assert_eq!(active, ACTIVE);
    *state = cycle.backward.map(Into::into);

    // P3: add, then take x^alpha in the degree-three quotient, triple by triple.
    add_round_constants(state, &params.rcons[first + 2]);
    for triple in 0..WIDTH / 3 {
        let base = 3 * triple;
        let input = [
            state[base].dup(),
            state[base + 1].dup(),
            state[base + 2].dup(),
        ];
        let output = eval_p3_triple::<AB, WIDTH, REGISTERS, P3_BLOCKS, CONSTANT_ROWS, ALPHA>(
            input,
            &cycle.extension_powers[base..base + 3],
            params,
            builder,
        );
        for (coordinate, value) in output.iter().enumerate() {
            builder.assert_eq(value.dup(), cycle.post[base + coordinate]);
        }
    }
    *state = cycle.post.map(Into::into);
}

/// Evaluate one call's columns. `vectorized.rs` reuses this exact function.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    air: &XHashAir<AB::F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>,
    builder: &mut AB,
    cols: &XHashCols<AB::Var, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>,
) where
    AB::F: PrimeField64,
{
    let mut state: [AB::Expr; WIDTH] = cols.inputs.map(Into::into);
    for cycle in 0..CYCLES {
        eval_cycle::<AB, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CONSTANT_ROWS, ALPHA>(
            &mut state,
            &cols.cycles[cycle],
            cycle,
            &air.params,
            builder,
        );
    }
    mds_multiply(&mut state, &air.params.m);
    add_round_constants(&mut state, &air.params.rcons[CONSTANT_ROWS - 1]);
    // The final boundary cells are constrained, not merely exposed to tests.
    assert_state_eq(builder, &state, &cols.outputs);
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
> Air<AB> for XHashAir<AB::F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &XHashCols<_, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES> =
            main.current_slice().borrow();
        eval(self, builder, cols);
    }
}
