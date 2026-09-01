//! The AIR: constants only, plus a free `eval` over one call's columns.
//!
//! POLICY §6: the AIR struct holds constants and nothing else; `Air::eval` is a
//! thin wrapper around a **free** `eval(air, builder, cols)` taking one call's
//! columns. That free function is what `vectorized.rs` reuses, and the reuse is
//! the reason the two layouts cannot diverge.
//!
//! `air.rs` and `generation.rs` will drift. Keep them line-for-line parallel —
//! same order, same helper names, same per-round split.
//!
//! # The recurrence
//!
//! `native.rs` runs the permutation the oracle's way: a `t`-element state, one
//! S-box on branch 0, `t - 1` additions, a rotation, and `M_IO` at each end. This
//! file never forms a state, and the derivation is `gmimc`'s
//! ([`gmimc::air`](../../../gmimc/src/air.rs)) with the two changes GMiMC2 makes.
//!
//! Only branch 0 is read nonlinearly, so a branch is a running sum of S-box
//! outputs and only *matters* on the round it reaches position 0. Write `b_r` for
//! its value there — **after** the round constant, which this design adds into the
//! state — and `y_r = b_r^alpha`. A branch leaving position 0 at round `r` is back
//! for round `r+t`, having received `y` in the `t-1` rounds between, and picks up
//! `rc_{r+t}` on arrival:
//!
//! ```text
//! b_r = b_{r-t} + rc_r + sum_{j=r-t+1}^{r-1} y_j
//! ```
//!
//! That extra `rc_r` is the whole arithmetic difference between the two crates,
//! and it is exactly what the specification means by "permits slightly more
//! efficient circuits": the constant merges into an addition the round already
//! performs, instead of being a separate input to the S-box.
//!
//! The rule holds at the boundaries too, under conventions that cost nothing:
//! `y_j = 0` and `rc_j = 0` outside `0 .. R`, and `b_{r-t} = M_IO(input)[r]` for
//! `r < t`. And after the last round the state is fixed, so `b_{R+k}` for
//! `k < t` is the branch sitting at position `k` — which is what the trailing
//! `M_IO` is applied to.
//!
//! # The telescoped constraint
//!
//! Written out, the window relation is a constraint with `t-1` S-box terms.
//! Subtracting consecutive copies cancels all but the two ends:
//!
//! ```text
//! b_{r+1} - b_r = (b_{r-t+1} - b_{r-t}) + (rc_{r+1} - rc_r) + y_r - y_{r-t+1}
//! ```
//!
//! so a constraint carries **four committed cells, two constants and two S-box
//! terms**, at every width. Telescoping is an induction and needs its base:
//! `b_0 = M_IO(input)[0] + rc_0`, the one constraint [`eval`] writes before the
//! loop. Dropping it would leave the chain free to slide by a constant.
//!
//! # The two `M_IO` brackets, priced
//!
//! The leading one is free: `M_IO(inputs)` is three cells and two doublings per
//! link, degree 1, and it appears only in the first `2t` constraints — the ones
//! whose window reaches before round 0.
//!
//! The trailing one is not. The chain's committed block stops at `b_{R-1}`, so
//! each `b_{R+k}` the output needs is an *expression*, and the telescoped form
//! cannot reach it — differencing needs a committed cell at both ends. What
//! [`eval`] writes is the window relation itself, which is bounded by `t-1` S-box
//! terms and reaches no further than one trip around the state:
//!
//! ```text
//! b_{R+k}   = b_{R+k-t} + sum_{j=R+k-t+1}^{R-1} y_j        k = 0 .. t-1
//! output    = M_IO(b_R, .., b_{R+t-1})
//! ```
//!
//! Three of those per output cell, so the `t` output constraints carry about
//! `3t(t-1)/2` S-box terms between them against the `2R` of the whole round loop —
//! a little over half as much again at `t = 24, R = 264`, at `alpha = 2` where an
//! S-box term is one squaring.
//!
//! It buys the property [`columns`](crate::columns) argues for: the `outputs`
//! cells are the permutation's own output, so POLICY §10's second layer compares
//! them against the reference's directly and the trailing `M_IO` is a constraint
//! rather than an expression a test recomputes. A cheaper layout exists — commit
//! the *pre*-`M_IO` tail instead, and the whole call becomes one uniform chain as
//! in `gmimc` — and it is not taken, because it would prove the permutation
//! relation with its output in a different basis and leave nothing in the trace to
//! compare with a vector.
//!
//! # Cost, and the floor
//!
//! `R + t` constraints and `R + 2t` cells: 108 and 120 at Goldilocks `t=12`, 288
//! and 312 at every 31-bit `t=24`. Max degree `alpha`, or 2 with the register
//! alpha 4 admits.
//!
//! **This attains POLICY §11's floor**, for the reason `gmimc::air` gives: a round
//! performs one `alpha`-th power, a degree-`D` constraint absorbs `log2(D)` chained
//! multiplications, and at `D = alpha` that is the one cell per round this layout
//! commits — attainable because the committed cell is the one the next round reads
//! through its only nonlinear input.
//!
//! At `alpha = 2` the floor is reached at degree 2 with no register at all, which
//! is why the three 31-bit points have a single variant: a register would commit
//! `y = head^2` and leave the recurrence affine, buying nothing for one more cell
//! per round. That is a finding, and [`columns::assert_layout`](crate::columns::assert_layout)
//! is where it is enforced rather than remembered.

use core::borrow::Borrow;

use harness::gadgets::power_map::eval_power_map;
use harness::permutation::assert_state_eq;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing};

use crate::columns::{GMiMC2Cols, assert_layout, num_cols};
use crate::params::{GMiMC2Params, m_io};

/// Constants for one independently constrained GMiMC2 call.
///
/// The round constants and nothing else (POLICY §6). Neither matrix is here: `M`
/// is the rotation this arithmetization has absorbed into the chain's indexing,
/// and `M_IO` is the specification's circulant, applied as the addition program
/// [`m_io`] rather than stored.
#[derive(Clone, Debug)]
pub struct GMiMC2Air<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> {
    /// `rc_0 .. rc_{R-1}`, one per round, added *into* branch 0 before the power
    /// map.
    pub(crate) constants: [F; ROUNDS],
}

impl<F, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize, const ALPHA: u64>
    GMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    /// Forces the layout assertions at monomorphization: an associated const of a
    /// generic type is evaluated when it is used, so an illegal parameter set is a
    /// compile error rather than a trace whose cells mean something else.
    const LAYOUT: () = assert_layout(WIDTH, ROUNDS, ALPHA, REGISTERS);

    /// Build the AIR from one instance's round constants.
    #[must_use]
    pub const fn new(constants: [F; ROUNDS]) -> Self {
        let () = Self::LAYOUT;
        Self { constants }
    }
}

impl<F: Copy, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize, const ALPHA: u64>
    GMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    /// Build one degree/register variant of an instance.
    ///
    /// # Panics
    ///
    /// If `ALPHA` disagrees with the parameters' own exponent. It matters more
    /// here than in most crates: `alpha` is part of the constant seed, so a
    /// mismatch means the constants belong to a different instance than the S-box
    /// does.
    #[must_use]
    pub fn from_params(params: &GMiMC2Params<F, WIDTH, ROUNDS>) -> Self {
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        Self::new(params.rcons)
    }
}

/// Maximum constraint degree of a variant.
///
/// Every constraint is affine in committed cells apart from its S-box terms, so
/// the AIR's degree *is* the power map's: this construction sets no floor of its
/// own. `M_IO` does not raise it — it is a linear combination of three terms of
/// the same degree.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, registers: usize) -> usize {
    match (alpha, registers) {
        (2 | 4, 0) => alpha as usize,
        // `x^2` witnessed at degree 2, then `register^2`: both halves are 2.
        (4, 1) => 2,
        _ => panic!("unsupported GMiMC2 degree/register variant"),
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> BaseAir<F> for GMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, REGISTERS, ROUNDS>()
    }

    /// One row is one call, so nothing is read from the next row. Declaring it
    /// lets the prover skip opening the shifted trace entirely (POLICY §6).
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(ALPHA, REGISTERS))
    }
}

/// Evaluate one call's columns. `vectorized.rs` reuses this exact function.
///
/// # What pins what (POLICY §9)
///
/// | cell | pinned by |
/// |---|---|
/// | `rounds[0].head` | the base constraint, `b_0 == M_IO(input)[0] + rc_0` |
/// | `rounds[r].head`, `r >= 1` | the difference rule at chain index `r + t - 1` |
/// | `rounds[r].registers` | `register == head^2`, inside the power-map gadget |
/// | `outputs` | `outputs == M_IO(b_R, .., b_{R+t-1})`, the trailing bracket |
///
/// `inputs` is the statement rather than a witness — it is the call's input — and
/// every input cell still reaches the constraints through `M_IO`, three at a time,
/// so corrupting one is rejected like any other cell. Nothing is left over.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    air: &GMiMC2Air<AB::F, WIDTH, REGISTERS, ROUNDS, ALPHA>,
    builder: &mut AB,
    cols: &GMiMC2Cols<AB::Var, WIDTH, REGISTERS, ROUNDS>,
) {
    // One S-box per round, evaluated once. Each `y_r` is read by the two
    // constraints whose windows end at `b_{r+1}` and `b_{r+t}`, and by the output
    // boundary for the last `t-1` rounds; a register may only be asserted once, so
    // this loop and not the constraint loop is where the power map runs.
    //
    // No round constant appears here: `head` already carries it (`columns::Round`).
    let mut sbox: Vec<AB::Expr> = Vec::with_capacity(ROUNDS);
    for round in &cols.rounds {
        sbox.push(eval_power_map::<AB, ALPHA, REGISTERS>(
            round.head.into(),
            &round.registers,
            builder,
        ));
    }

    // `y` at chain index `i`, which is round `i - WIDTH`: zero before the round
    // block, which is what makes the input boundary the same rule as every round
    // rather than a special case.
    let y = |i: usize| -> AB::Expr {
        if (WIDTH..WIDTH + ROUNDS).contains(&i) {
            sbox[i - WIDTH].dup()
        } else {
            AB::Expr::ZERO
        }
    };

    // The chain's first `WIDTH` links: `M_IO` of the input cells, affine and
    // committed nowhere. The same function the native permutation and the
    // generator apply (`params::m_io`), over `AB::Expr` instead of over `F`.
    let pre: [AB::Expr; WIDTH] = m_io(&cols.inputs.map(Into::into));
    let chain = |i: usize| -> AB::Expr {
        if i < WIDTH {
            pre[i].dup()
        } else {
            cols.rounds[i - WIDTH].head.into()
        }
    };

    // The base of the induction: `b_0 = b_{-t} + rc_0`, the empty window. Without
    // it the differences below pin every *step* of the chain and none of its
    // position.
    builder.assert_eq(chain(WIDTH), chain(0) + air.constants[0].dup());

    // The telescoped window relation, over the committed block. `i` is the link
    // the step starts from, so it runs from the first round cell to the
    // second-to-last: `R - 1` constraints, `R` with the base.
    for i in WIDTH..WIDTH + ROUNDS - 1 {
        let carried = chain(i) + chain(i + 1 - WIDTH) - chain(i - WIDTH);
        let constants = air.constants[i + 1 - WIDTH].dup() - air.constants[i - WIDTH].dup();
        builder.assert_eq(chain(i + 1), carried + constants + y(i) - y(i + 1 - WIDTH));
    }

    // The output boundary: the `t` links past the committed block, each by the
    // window relation itself, then the trailing `M_IO`. See the module docs for
    // why this is the one place the telescoped form cannot be used and what the
    // untelescoped one costs.
    let tail: [AB::Expr; WIDTH] = core::array::from_fn(|k| {
        // `b_{R+k} = b_{R+k-t} + sum_{j=R+k-t+1}^{R-1} y_j`, with no `rc` term:
        // there is no round `R+k` to add one.
        let mut value = chain(ROUNDS + k);
        for term in &sbox[ROUNDS + k + 1 - WIDTH..ROUNDS] {
            value += term.dup();
        }
        value
    });
    assert_state_eq(builder, &m_io(&tail), &cols.outputs);
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> Air<AB> for GMiMC2Air<AB::F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &GMiMC2Cols<_, WIDTH, REGISTERS, ROUNDS> = main.current_slice().borrow();
        eval::<AB, WIDTH, REGISTERS, ROUNDS, ALPHA>(self, builder, cols);
    }
}

#[cfg(test)]
mod tests {
    use p3_air::symbolic::{AirLayout, get_max_constraint_degree, get_symbolic_constraints};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;

    use super::*;

    /// The declared degree against the **evaluator's** symbolic degree, and the
    /// constraint count against the `R + t` this file derives.
    ///
    /// Neither is visible to a known-answer vector: undo the telescoping, drop the
    /// base constraint, or fold `M_IO` to the wrong side of a boundary, and every
    /// KAT still passes or fails for a different reason. The degree is also what
    /// buys the blowup (POLICY §7), so a wrong declaration is a silently cheaper
    /// configuration rather than a failure.
    ///
    /// Constant *values* are irrelevant to a degree, so this builds the AIR from
    /// zeros: what is checked is the shape of the expressions.
    #[test]
    fn the_symbolic_shape_is_the_one_this_file_derives() {
        // Goldilocks t = 12, R = 96, alpha 4.
        let unsplit = GMiMC2Air::<Goldilocks, 12, 0, 96, 4>::new([Goldilocks::ZERO; 96]);
        let layout = AirLayout::from_air(&unsplit);
        layout.validate_against_air(&unsplit);

        // No periodic columns and no transition selector — one row is one call —
        // so none of these depend on the trace length.
        assert_eq!(get_max_constraint_degree(&unsplit, layout, 1 << 10), 4);
        assert_eq!(
            BaseAir::<Goldilocks>::max_constraint_degree(&unsplit),
            Some(get_max_constraint_degree(&unsplit, layout, 1 << 10))
        );
        assert_eq!(
            get_symbolic_constraints::<Goldilocks, _>(&unsplit, layout).len(),
            96 + 12
        );
        assert_eq!(BaseAir::<Goldilocks>::width(&unsplit), 120);

        // One register: one more cell and one more constraint per round, and the
        // degree drops from 4 to 2.
        let split = GMiMC2Air::<Goldilocks, 12, 1, 96, 4>::new([Goldilocks::ZERO; 96]);
        let layout = AirLayout::from_air(&split);
        layout.validate_against_air(&split);
        assert_eq!(get_max_constraint_degree(&split, layout, 1 << 10), 2);
        assert_eq!(
            get_symbolic_constraints::<Goldilocks, _>(&split, layout).len(),
            96 + 12 + 96
        );
        assert_eq!(BaseAir::<Goldilocks>::width(&split), 216);
    }

    /// The 31-bit point, where the S-box is a squaring and the whole AIR is
    /// degree 2 with nothing committed beyond the chain.
    ///
    /// This is the row the register variant would have to beat, and the reason it
    /// is not offered: 312 cells at degree 2 already, so a register could only add
    /// 264 more for the same degree.
    #[test]
    fn the_squaring_instances_are_degree_two_at_one_cell_a_round() {
        let air = GMiMC2Air::<KoalaBear, 24, 0, 264, 2>::new([KoalaBear::ZERO; 264]);
        let layout = AirLayout::from_air(&air);
        layout.validate_against_air(&air);

        assert_eq!(get_max_constraint_degree(&air, layout, 1 << 10), 2);
        assert_eq!(
            get_symbolic_constraints::<KoalaBear, _>(&air, layout).len(),
            264 + 24
        );
        assert_eq!(BaseAir::<KoalaBear>::width(&air), 312);
    }
}
