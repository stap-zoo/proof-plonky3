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
//! `native.rs` runs the permutation the reference's way: a `t`-element state, one
//! S-box on branch 0, `t - 1` additions, a rotation. This file never forms a
//! state at all, and the reason is worth the derivation, because everything below
//! rests on it.
//!
//! Only branch 0 is read nonlinearly. Every other branch accumulates the same
//! `y` and is never mixed with another branch, so a branch's value is a running
//! sum of S-box outputs and only *matters* on the round it reaches position 0.
//! Write `b_r` for that value — branch 0 entering round `r`, before the round
//! constant — and `y_r = (b_r + rc_r)^alpha`.
//!
//! The rotation moves position 0 to position `t-1` and every other position down
//! by one, so a branch leaving position 0 at round `r` sits at positions
//! `t-1, t-2, .., 1` during rounds `r+1 .. r+t-1` — receiving `y` in each — and is
//! back at position 0 for round `r+t`:
//!
//! ```text
//! b_{r+t} = b_r + sum_{j=r+1}^{r+t-1} y_j
//! ```
//!
//! At the ends the same rule holds under two conventions that cost nothing:
//! `y_j = 0` outside `0 .. R`, and `b_{r-t} = input[r]` for `r < t` — the branch
//! entering round `r` at position 0 is the one that started at position `r`. And
//! `out[i] = b_{R+i}`: once the rounds stop the state is fixed, so `t` further
//! steps of the recurrence rotate each branch past position 0 once and read the
//! output off in order. One rule, over the whole chain `b_{-t} .. b_{R+t-1}`
//! ([`columns`](crate::columns)).
//!
//! # Why the committed cell is `b_r` and not `y_r`
//!
//! `y_r` is the round's *output* and looks like the natural thing to commit — it
//! is what every other construction here commits. It is the wrong choice, and by
//! a wide margin: with `y` committed, `b_r` unrolls to `b_{r-t} + (t-1)` cells,
//! then `b_{r-t}` unrolls again, and the last round's S-box input is a sum of
//! `O(R)` terms — 334 of them at `R = 335`. Committing `b_r` reaches back exactly
//! `t` rounds and stops, because `b_{r-t}` is a cell.
//!
//! The same choice is what caps the *degree*. `b_r` is a committed cell, so
//! `y_r = (b_r + rc_r)^alpha` is degree `alpha` and nothing compounds: the
//! committed cell is the one the next S-box reads, which is POLICY §11's
//! condition for a floor to be attainable rather than merely countable.
//!
//! # The telescoped constraint
//!
//! Written out, `b_r = b_{r-t} + sum_{j=r-t+1}^{r-1} y_j` is a constraint with
//! `t-1` S-box terms. Subtracting consecutive copies of it cancels all but the
//! two ends:
//!
//! ```text
//! b_{r+1} - b_r = (b_{r-t+1} - b_{r-t}) + y_r - y_{r-t+1}
//! ```
//!
//! so a constraint carries **four committed cells and two S-box terms**, at every
//! width — 2 instead of 23 at `t = 24`. Telescoping is an induction and needs its
//! base: `b_0 = b_{-t} = input[0]`, the one constraint [`eval`] writes outside the
//! loop. Given that, the differences reconstruct every window relation, and
//! dropping it would leave the whole chain free to slide by a constant.
//!
//! The boundaries need no special case. The difference rule holds across them
//! under the two conventions above, so `eval` writes it over the whole chain and
//! the input and output blocks are pinned by the same rule as every round —
//! including the last row POLICY §9 asks to be asserted rather than assumed.
//!
//! # Cost, and the floor
//!
//! `R + t` constraints and `R + 2t` cells: 105 and 117 at Goldilocks `t=12`, 359
//! and 383 at every 31-bit `t=24`. Max degree `alpha` unsplit; with the power
//! map's one register, 3 at alpha 5 and 7 and 2 at alpha 3, for one more cell and
//! one more constraint per round.
//!
//! **This attains POLICY §11's floor.** A round performs one `alpha`-th power,
//! i.e. `ceil(log2(alpha))` chained multiplications, and a degree-`D` constraint
//! absorbs `log2(D)` of them; at `D = alpha` that is one cell per round, which is
//! what this layout commits. The claim POLICY §11 requires is that the floor is
//! *attained*, and the derivation is the recurrence above: what makes it
//! attainable is that the committed cell is the one the next round reads through
//! its only nonlinear input, so the degree settles at `alpha` instead of
//! compounding.
//!
//! # The two other variant axes, and why neither is here
//!
//! POLICY §11 names two axes beyond the power map, and this construction has
//! neither to offer.
//!
//! * **How much of a round is flattened.** A GMiMC round *is* one power map, so
//!   the flattening axis and the register axis are the same axis, and
//!   `REGISTERS` already walks it.
//! * **How many rounds between state commitments.** Committing every `SPAN`-th
//!   `b` leaves the skipped `b`s as expressions, and each skipped round raises
//!   the S-box input's degree by a factor of `alpha` — `alpha^SPAN` at the
//!   constraint — to save `1/SPAN` of a width that is already one cell per round.
//!   At `alpha = 7` one skipped round costs degree 49 for 47 fewer cells at
//!   `t = 12`. It is not a trade worth a variant, and saying so here is cheaper
//!   than measuring it.

use core::borrow::Borrow;

use harness::gadgets::power_map::eval_power_map;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing};

use crate::columns::{GMiMCCols, assert_layout, chain_len, num_cols};
use crate::params::GMiMCParams;

/// Constants for one independently constrained GMiMC call.
///
/// The round constants and nothing else (POLICY §6): the width, the round count
/// and the exponent are const parameters, and the linear layer is a rotation this
/// arithmetization has already absorbed into the chain's indexing — there is no
/// matrix to store.
#[derive(Clone, Debug)]
pub struct GMiMCAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> {
    /// `rc_0 .. rc_{R-1}`, one per round, an argument to that round's S-box.
    pub(crate) constants: [F; ROUNDS],
}

impl<F, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize, const ALPHA: u64>
    GMiMCAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    /// Forces the layout assertions at monomorphization: an associated const of a
    /// generic type is evaluated when it is used, so an illegal parameter set is a
    /// compile error rather than a trace whose cells mean something else.
    const LAYOUT: () = assert_layout(WIDTH, ALPHA, REGISTERS);

    /// Build the AIR from one instance's round constants.
    #[must_use]
    pub const fn new(constants: [F; ROUNDS]) -> Self {
        let () = Self::LAYOUT;
        Self { constants }
    }
}

impl<F: Copy, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize, const ALPHA: u64>
    GMiMCAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    /// Build one degree/register variant of an instance.
    ///
    /// # Panics
    ///
    /// If `ALPHA` disagrees with the parameters' own exponent. That is the one
    /// way the const parameter and the reference-derived value could drift apart,
    /// and the result would be a permutation that is merely not the reference's.
    #[must_use]
    pub fn from_params(params: &GMiMCParams<F, WIDTH, ROUNDS>) -> Self {
        assert_eq!(params.alpha, ALPHA, "AIR alpha must match its parameters");
        Self::new(params.rcons)
    }
}

/// Maximum constraint degree of a variant.
///
/// Every constraint is affine in committed cells apart from its two S-box terms,
/// so the AIR's degree *is* the power map's — this construction sets no floor of
/// its own, unlike Griffin's Horst layer or Neptune's Lai--Massey layer. That is
/// why alpha 3 is offered a register here where Griffin does not offer one: with
/// nothing else at degree 3, the register is the difference between 3 and 2.
#[must_use]
pub const fn max_constraint_degree(alpha: u64, registers: usize) -> usize {
    match (alpha, registers) {
        (3 | 5 | 7, 0) => alpha as usize,
        // `x^2` witnessed: `register*x` at alpha 3, `register^2*x` at 5 and 7 —
        // and the witnessing constraint itself is degree 2 (alpha 3, 5) or 3
        // (alpha 7, where the register holds `x^3`).
        (3, 1) => 2,
        (5 | 7, 1) => 3,
        _ => panic!("unsupported GMiMC degree/register variant"),
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> BaseAir<F> for GMiMCAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA>
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
/// | `rounds[0].head` | the base constraint, `b_0 == input[0]` |
/// | `rounds[r].head`, `r >= 1` | the difference rule at chain index `r + t - 1` |
/// | `rounds[r].registers` | `register == (head + rc_r)^2` (or `^3`), inside the power-map gadget |
/// | `outputs[i]` | the difference rule at chain index `R + t + i - 1` |
///
/// `inputs` is the statement rather than a witness — it is the call's input — and
/// every input cell still appears in two constraints, so corrupting one is
/// rejected like any other cell. Nothing is left over.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    air: &GMiMCAir<AB::F, WIDTH, REGISTERS, ROUNDS, ALPHA>,
    builder: &mut AB,
    cols: &GMiMCCols<AB::Var, WIDTH, REGISTERS, ROUNDS>,
) {
    // One S-box per round, evaluated once. Each `y_r` is read by exactly two
    // constraints — the one that ends at `b_{r+1}` and the one that ends at
    // `b_{r+t}` — and a register may only be asserted once, so this loop and not
    // the constraint loop is where the power map runs.
    let mut sbox: Vec<AB::Expr> = Vec::with_capacity(ROUNDS);
    for (round, constant) in cols.rounds.iter().zip(&air.constants) {
        // The constant is an argument to the S-box and never enters the state:
        // `native.rs`'s `nonlinear_layer`, and the one line GMiMC2 changes.
        let value = round.head + constant.dup();
        sbox.push(eval_power_map::<AB, ALPHA, REGISTERS>(
            value,
            &round.registers,
            builder,
        ));
    }

    // `y` at chain index `i`, which is round `i - WIDTH`: zero outside the round
    // block, which is what makes the input and output boundaries the same rule as
    // every round rather than two special cases.
    let y = |i: usize| -> AB::Expr {
        if (WIDTH..WIDTH + ROUNDS).contains(&i) {
            sbox[i - WIDTH].dup()
        } else {
            AB::Expr::ZERO
        }
    };

    // The base of the induction: `b_0 = b_{-t}`, the empty window. Without it the
    // differences below pin every *step* of the chain and none of its position.
    builder.assert_eq(cols.chain(WIDTH), cols.chain(0));

    // The telescoped window relation, over the whole chain. `i` is the link the
    // step starts from, so it runs from the first round cell to the second-to-last
    // output cell: `R + t - 1` constraints, `R + t` with the base.
    for i in WIDTH..chain_len::<WIDTH, ROUNDS>() - 1 {
        let carried = cols.chain(i) + cols.chain(i + 1 - WIDTH) - cols.chain(i - WIDTH);
        builder.assert_eq(cols.chain(i + 1), carried + y(i) - y(i + 1 - WIDTH));
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
> Air<AB> for GMiMCAir<AB::F, WIDTH, REGISTERS, ROUNDS, ALPHA>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cols: &GMiMCCols<_, WIDTH, REGISTERS, ROUNDS> = main.current_slice().borrow();
        eval::<AB, WIDTH, REGISTERS, ROUNDS, ALPHA>(self, builder, cols);
    }
}

#[cfg(test)]
mod tests {
    use p3_air::symbolic::{AirLayout, get_max_constraint_degree, get_symbolic_constraints};
    use p3_goldilocks::Goldilocks;

    use super::*;

    /// The declared degree against the **evaluator's** symbolic degree, and the
    /// constraint count against the `R + t` this file derives.
    ///
    /// Neither is visible to a known-answer vector: undo the telescoping, commit
    /// `y` instead of `b`, or drop the base constraint, and every KAT still
    /// passes. The degree is also what buys the blowup (POLICY §7), so a wrong
    /// declaration is a silently cheaper configuration rather than a failure —
    /// hence the cross-check here, at the cheapest grid point, as early as the
    /// evaluator exists, and again per instance in `tests/numbers.rs`.
    ///
    /// Constant *values* are irrelevant to a degree, so this builds the AIR from
    /// zeros: what is checked is the shape of the expressions.
    #[test]
    fn the_symbolic_shape_is_the_one_this_file_derives() {
        // Goldilocks t = 12, R = 93, alpha 7.
        let unsplit = GMiMCAir::<Goldilocks, 12, 0, 93, 7>::new([Goldilocks::ZERO; 93]);
        let layout = AirLayout::from_air(&unsplit);
        layout.validate_against_air(&unsplit);

        // No periodic columns and no transition selector — one row is one call —
        // so none of these depend on the trace length.
        assert_eq!(get_max_constraint_degree(&unsplit, layout, 1 << 10), 7);
        assert_eq!(
            BaseAir::<Goldilocks>::max_constraint_degree(&unsplit),
            Some(get_max_constraint_degree(&unsplit, layout, 1 << 10))
        );
        assert_eq!(
            get_symbolic_constraints::<Goldilocks, _>(&unsplit, layout).len(),
            93 + 12
        );
        assert_eq!(BaseAir::<Goldilocks>::width(&unsplit), 117);

        // One register: one more cell and one more constraint per round, and the
        // degree drops from 7 to 3.
        let split = GMiMCAir::<Goldilocks, 12, 1, 93, 7>::new([Goldilocks::ZERO; 93]);
        let layout = AirLayout::from_air(&split);
        layout.validate_against_air(&split);
        assert_eq!(get_max_constraint_degree(&split, layout, 1 << 10), 3);
        assert_eq!(
            get_symbolic_constraints::<Goldilocks, _>(&split, layout).len(),
            93 + 12 + 93
        );
        assert_eq!(BaseAir::<Goldilocks>::width(&split), 210);
    }

    /// The width is a round count, not a state width — the claim
    /// [`columns`](crate::columns) makes and the reason 335 rounds are
    /// affordable. Stated against the *evaluator* rather than against `size_of`,
    /// because it is the constraint count that would give it away if some
    /// per-branch constraint had crept in.
    #[test]
    fn neither_the_width_nor_the_constraints_grow_with_t() {
        fn shape<const WIDTH: usize>() -> (usize, usize) {
            let air = GMiMCAir::<Goldilocks, WIDTH, 0, 40, 5>::new([Goldilocks::ZERO; 40]);
            let layout = AirLayout::from_air(&air);
            (
                BaseAir::<Goldilocks>::width(&air),
                get_symbolic_constraints::<Goldilocks, _>(&air, layout).len(),
            )
        }
        // Doubling `t` at a fixed `R` adds `2t` cells and `t` constraints, and
        // nothing else: no term of either is a multiple of `R * t`.
        assert_eq!(shape::<12>(), (40 + 24, 40 + 12));
        assert_eq!(shape::<24>(), (40 + 48, 40 + 24));
    }
}
