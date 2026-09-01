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
//! (POLICY §9). Here that matters more than in any other file of this crate: the
//! round's *even* output cells are written in terms of the committed odd ones, so
//! an odd cell the prover picked freely would make **both** halves of the round's
//! output arbitrary. `eval_round`'s `commit_state` call is the assertion the
//! whole layout rests on.
//!
//! # The fused round
//!
//! The same round as [`full_commit::air`](crate::full_commit::air)'s, and `M` is
//! not written out twice: the prologue, the half swap and the `M_IO` brackets are
//! `linear.rs`, shared with `full_commit` and with both generators. What is
//! restated here is the *nonlinear* part and the writes — which is exactly where
//! the two variants differ, so there is nothing left to share and nothing that
//! could be shared without a branch inside the one function whose degree
//! behaviour the crate rests on.
//! `tests/air.rs::every_variant_computes_the_same_permutation` is the guard on
//! what remains duplicated.
//!
//! With `h = t/2`, `p` ranging over the `t/4` Feistels, and `U[j] = x[h+j]` the
//! round's *upper* half:
//!
//! ```text
//! z       = (2·x[h-2] + x[h-1],  x[h-2] + x[h-1])
//! extra   = (Σ x[2q],  Σ x[2q+1])          for q = 1 .. t/4 - 2
//!
//! raw out[2p]     = U[2p]   + y4(pair h-2-2p) + [p > 0] z.0 + [p = t/4-1] extra.0
//! raw out[2p+1]   = U[2p+1] + y5(pair h-2-2p) + [p > 0] z.1 + [p = t/4-1] extra.1
//! out[h + i]      = x[i]
//! ```
//!
//! `z` and `extra` read the *old* lower half, which the round does not modify, so
//! the order of the writes is free.
//!
//! # What this file does differently: the even half is never written
//!
//! Only `raw out[2p+1]` becomes a constraint and a column. The even output is
//! recovered by *differencing* the committed odd one, and the raw degree-4 form
//! above never appears in the builder. Two identities make that exact —
//! `y5 − y4 = y3` and `z.0 − z.1 = x[h−2]`:
//!
//! ```text
//! out[2p] = out[2p+1] + (U[2p] − U[2p+1]) − y3
//!                     + [p > 0]      x[h-2]
//!                     + [p = t/4-1] (extra.0 − extra.1)
//! ```
//!
//! and `y3 = y1 + y2 + c1` is one squaring shallower than `y4` and `y5`. So the
//! even half costs a degree-2 expression instead of a degree-4 one, and no cell.
//!
//! **The two forms are not equal as polynomials in the committed cells** — the raw
//! one is degree 4 in the round's input, the differenced one degree 2 — they agree
//! only on the variety the committing constraint cuts out. That is precisely what
//! flattening is (POLICY §9): the `commit_state` call is not bookkeeping that
//! could be moved or dropped, it is the hypothesis the differenced form is derived
//! under, and the whole degree claim rests on it. `eval_round` argues it again at
//! the write site, because that is where a reader will be tempted to "simplify"
//! the expression back to the natural one.
//!
//! # Degree accounting
//!
//! The Feistel is asymmetric in its input pair and the committed slot is the steep
//! one; [`crate::half_commit::columns`] carries that argument, since it is a
//! statement about *which* cell the layout commits. What belongs here is the
//! recurrence, because [`max_constraint_degree`] replays it.
//!
//! Track four degrees: the lower half's even and odd cells `(e, o)`, and the upper
//! half's `(e_prev, o_prev)` — the upper half *is* the previous round's lower half,
//! verbatim, so it is not reset by this round's commitment. Entering round 1 all
//! four are 1, the state being `M_IO` of the `inputs` columns. The Feistel reads
//! pair `(x[i], x[i+1])` with `i` even, so `x0` is an even cell and `x1` an odd
//! one, and:
//!
//! ```text
//! y3         = max(e, 2·o)                     y2 = x0 + y1², y3 affine in it
//! constraint = max(o_prev, 2·y3, e, o)         U[odd], y5, z, extra
//! new_even   = max(y3, e_prev, o_prev, e, o)   a column, minus y3, plus affine
//! new_odd    = 1                               committed
//! ```
//!
//! then `(e_prev, o_prev) ← (e, o)` and `(e, o) ← (new_even, 1)`. The trailing
//! `M_IO` and its `outputs` assert add `max(e, o, e_prev, o_prev)`, affine in both
//! halves. Round 1 gives `constraint = 4` and `lower = [2, 1]`, round 2 reproduces
//! `lower = [2, 1]`, and from there it is a **fixed point**: degree 4 at every
//! round count, not a number that happens to be 4 at `R = 52`.
//!
//! The committed parity is what makes it a fixed point rather than a geometric
//! series, and choosing it wrong fails silently — every known-answer test still
//! passes and only the degree moves. [`degree_of_schedule`] is therefore
//! parameterized by [`CommittedSlot`] and the tests below assert what the
//! alternatives cost: `2⁵³` for the even slot and `2²⁸` for alternating, against 4
//! here. That is the same failure mode `full_commit::air` documents for
//! `SPAN ≥ 2` — a commitment covering only part of the state leaves the rest to
//! compound.
//!
//! `harness::measure` cross-checks [`max_constraint_degree`] against the symbolic
//! degree on every measured row, and `the_differenced_form_evaluates_to_degree_four`
//! below checks the *evaluator* against it — which is what catches the raw form
//! being written by mistake, the one error the recurrence cannot see.

use core::borrow::Borrow;

use harness::permutation::{assert_state_eq, commit_state};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing};

use crate::half_commit::columns::{HalfCommitCols, Round, assert_layout, num_cols};
use crate::linear;

/// The AIR for one pSquareHash instance at the half-commitment layout. Constants
/// only.
#[derive(Debug, Clone)]
pub struct HalfCommitAir<F, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize> {
    /// `ROUNDS` rows of `PAIRS` constant pairs; pair `p` belongs to the Feistel
    /// writing output pair `p`. See `crate::params::PSquareHashParams::rcons`.
    ///
    /// The same array [`PSquareHashAir`](crate::full_commit::air::PSquareHashAir)
    /// takes, and flat for the same reason: the round constants belong to the
    /// *permutation* and any grouping belongs to the arithmetization. This layout
    /// has no grouping at all.
    pub(crate) constants: [[[F; 2]; PAIRS]; ROUNDS],
}

impl<F, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>
    HalfCommitAir<F, WIDTH, PAIRS, ROUNDS>
{
    /// Forces the layout assertions at monomorphization: an associated const of
    /// a generic type is evaluated when it is used, so an illegal parameter set
    /// is a compile error rather than a trace whose cells mean something else.
    const LAYOUT: () = assert_layout(WIDTH, PAIRS);

    /// Build the AIR for one instance.
    #[must_use]
    pub const fn new(constants: [[[F; 2]; PAIRS]; ROUNDS]) -> Self {
        let () = Self::LAYOUT;
        Self { constants }
    }
}

/// Which output slot of each Feistel a schedule commits.
///
/// Only [`CommittedSlot::Odd`] is this layout; the other two exist because the
/// choice is the one thing the scheme rests on and it fails *silently*. Giving
/// them to [`degree_of_schedule`] records the decision as arithmetic the tests
/// check rather than as prose a future reader can talk themselves out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommittedSlot {
    /// `out[2p+1]`, every round — the layout this module implements.
    ///
    /// The odd slot is the one the next round squares twice, so committing it is
    /// what caps the degree; see [`crate::half_commit::columns`].
    Odd,
    /// `out[2p]`, every round.
    ///
    /// Leaves the steep input uncommitted, so the odd track doubles every round.
    Even,
    /// Flipping every round, starting from the slot `odd_first` names in round 1.
    ///
    /// Not a way out: the linear layer adds `z` into *both* outputs of every
    /// Feistel with `p > 0`, so there is no independent even/odd degree track to
    /// alternate between. It only halves the doubling rate.
    Alternating {
        /// Whether round 1 commits the odd slot.
        odd_first: bool,
    },
}

/// `max` over `usize`, as a `const fn`.
///
/// `Ord::max` is not const, and the recurrence below is a `const fn` because a
/// wrong declared degree buys a silently cheaper blowup (POLICY §7) and the
/// declaration is therefore checked at compile time in `instances.rs`.
const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// The maximum constraint degree of this layout, from the recurrence rather than
/// from a table.
///
/// Four, at every round count — see the module docs for why it is a fixed point.
/// Derived rather than written down because `harness::measure` cross-checks the
/// declared value against `get_max_constraint_degree` on every measured row, and
/// because the *evaluator* is what the value has to describe: see
/// `the_differenced_form_evaluates_to_degree_four`.
#[must_use]
pub const fn max_constraint_degree(rounds: usize) -> usize {
    degree_of_schedule(CommittedSlot::Odd, rounds)
}

/// The recurrence of the module docs, over any committed-slot schedule.
///
/// The parameter is what stops the parity being "simplified" away: with
/// [`CommittedSlot::Odd`] this is a fixed point at 4, and the other two schedules
/// compound. Only the `Odd` answer is used by an AIR; the others are asserted by
/// the tests, which is where the design decision is recorded.
///
/// No overflow guard, in the style of `full_commit::air::max_constraint_degree`:
/// the compounding schedules reach `2⁵³` at `R = 52` and are called nowhere but the
/// tests, which stay at or below that round count.
#[must_use]
pub const fn degree_of_schedule(slot: CommittedSlot, rounds: usize) -> usize {
    // `[even, odd]` degrees of the lower half, and of the upper half — which is
    // the previous round's lower half. Entering round 1 the state is `M_IO` of
    // the `inputs` columns, affine, so all four are 1.
    let mut lower = [1usize, 1];
    let mut upper = [1usize, 1];
    let mut max = 1;

    let mut r = 1;
    while r <= rounds {
        let commit_odd = match slot {
            CommittedSlot::Odd => true,
            CommittedSlot::Even => false,
            CommittedSlot::Alternating { odd_first } => odd_first == (r % 2 == 1),
        };
        // The committed slot's parity, as an index into `lower` / `upper`.
        let c = if commit_odd { 1 } else { 0 };

        // `y1 = x1 + c0` is the odd input and is squared into `y2 = x0 + y1²`;
        // `y3 = y1 + y2 + c1` is affine in `y2`. `y4` and `y5` are both one
        // further squaring deep, hence the `2 * y3` below.
        let y3 = max2(lower[0], 2 * lower[1]);

        // The committing constraint, `raw out[c] == cell`: the upper half's slot
        // of the *same* parity, the Feistel's output, and `z` and `extra` off the
        // old lower half. The last two are dominated here, and are listed anyway
        // because a schedule this comment did not anticipate may not dominate
        // them.
        let constraint = max2(max2(upper[c], 2 * y3), max2(lower[0], lower[1]));
        max = max2(max, constraint);

        // The differenced slot: a column (degree 1), minus `y3`, plus terms
        // affine in the upper half and in the old lower half.
        let other = max2(max2(y3, max2(upper[0], upper[1])), max2(lower[0], lower[1]));

        upper = lower;
        lower = if commit_odd { [other, 1] } else { [1, other] };
        r += 1;
    }

    // The trailing `M_IO` and its `outputs` assert: affine in both halves.
    max2(
        max,
        max2(max2(lower[0], lower[1]), max2(upper[0], upper[1])),
    )
}

impl<F: PrimeCharacteristicRing + Sync, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>
    BaseAir<F> for HalfCommitAir<F, WIDTH, PAIRS, ROUNDS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, PAIRS, ROUNDS>()
    }

    /// One row is one call, so nothing is read from the next row. Declaring it
    /// lets the prover skip opening the shifted trace entirely (POLICY §6).
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(ROUNDS))
    }
}

/// One round: the nonlinear layer and `M`, fused, committing one cell per
/// Feistel. See the module docs.
///
/// POLICY §9: `round.odd` is prover-chosen and the `commit_state` below is what
/// pins it. This is the strongest form of that requirement in the crate — the
/// *even* half of the round's output is an expression in these cells, so a free
/// `odd` would leave the whole output state arbitrary rather than just half of it.
#[inline]
fn eval_round<AB: AirBuilder, const WIDTH: usize, const PAIRS: usize>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, PAIRS>,
    constants: &[[AB::F; 2]; PAIRS],
    builder: &mut AB,
) {
    let h = WIDTH / 2;

    // `M`'s contributions to the new lower half that no Feistel supplies, read
    // off the old lower half — which this round leaves alone. Shared with
    // `full_commit`'s `eval_round` and with both generators (`linear.rs`): `M`
    // is the same matrix in every variant, and `z.0 − z.1 = x[h−2]` — the identity
    // the even half below is differenced by — is a property of that function.
    let (z, extra) = linear::prologue(state.as_slice());

    // The odd half of the round's output, raw — and each Feistel's `y3`, which is
    // the correction the even half needs and is one squaring shallower.
    let mut odd = core::array::from_fn::<_, PAIRS, _>(|_| AB::Expr::ZERO);
    let mut y3s = core::array::from_fn::<_, PAIRS, _>(|_| AB::Expr::ZERO);
    for p in 0..PAIRS {
        // Output pair p is written by the Feistel over input pair h-2-2p, and
        // reads round-constant pair p. See `PSquareHashParams::rcons`.
        let i = h - 2 - 2 * p;
        let (y5, y3) = eval_feistel::<AB>(state[i].dup(), state[i + 1].dup(), &constants[p]);

        let mut out = state[h + 2 * p + 1].dup() + y5;
        if p > 0 {
            out += z[1].dup();
        }
        if p == PAIRS - 1 {
            out += extra[1].dup();
        }
        odd[p] = out;
        y3s[p] = y3;
    }

    // **The commitment, and everything below depends on it having happened.**
    // `commit_state` asserts `odd[p] == round.odd[p]` and then continues from the
    // column, which is what drops the odd half from degree 4 to degree 1 — the
    // same mechanism `full_commit::air` uses for `Group::post`, applied to a
    // strided half of the state instead of a contiguous one. Collecting into a
    // contiguous array first is what lets the gadget be reused rather than
    // re-inlined.
    commit_state(builder, &mut odd, &round.odd);

    let mut next = core::array::from_fn::<_, WIDTH, _>(|_| AB::Expr::ZERO);
    for p in 0..PAIRS {
        next[2 * p + 1] = odd[p].dup();

        // ------------------------------------------------------------------
        // The one place in this crate where the AIR deliberately does not write
        // the natural expression. Three things a reader needs before touching it:
        //
        // 1. THE ALGEBRA. Subtracting the two raw outputs of the module docs,
        //
        //      raw out[2p] − raw out[2p+1]
        //        = (U[2p] − U[2p+1]) − y3 + [p>0](z.0 − z.1)
        //                                 + [last](extra.0 − extra.1)
        //
        //    because `y4 − y5 = −y3` and `z.0 − z.1 = x[h−2]`. Rearranged with
        //    the committed cell on the right, that is the expression below.
        //
        // 2. WHY IT IS SOUND. The two forms are *not* equal as polynomials in
        //    the committed cells: the raw form is degree 4 in the round's input
        //    and this one is degree 2. They agree on the variety cut out by the
        //    `commit_state` assertion above, which is exactly what flattening is
        //    (POLICY §9). Move that call after this loop, or drop it, and this
        //    expression stops computing the round function.
        //
        // 3. WHY NOT THE RAW FORM. Writing `y4` here instead would make the even
        //    half degree 4 in the round's input, the next round's odd output
        //    degree 8, and the whole call `4^r` — a degree nothing in this file
        //    would notice, because the round is still *correct*. Only
        //    `max_constraint_degree` against the symbolic degree catches it, in
        //    this module's tests and again in `tests/numbers.rs`.
        // ------------------------------------------------------------------
        let mut even =
            odd[p].dup() - y3s[p].dup() + state[h + 2 * p].dup() - state[h + 2 * p + 1].dup();
        if p > 0 {
            // `z.0 − z.1`, the whole of it.
            even += state[h - 2].dup();
        }
        if p == PAIRS - 1 {
            even += extra[0].dup() - extra[1].dup();
        }
        next[2 * p] = even;
    }

    // After the loops, so that `z`, `extra` and the Feistels read the old values.
    linear::write_upper_half(&mut next, state.as_slice());
    *state = next;
}

/// One Feistel, with nothing committed inside it.
///
/// Returns `(y5, y3)`: the odd output the round commits, and the correction the
/// even output is differenced by. `y4` is never formed — `y5 = y3 + y2 + y3²`
/// needs it only as an intermediate, and the even output comes from differencing —
/// so there is no register array, no `Feistel` cell struct and no `REGISTERS`
/// parameter. The flattening this variant spends is all at the round's output.
///
/// POLICY §9: nothing here is witnessed, so nothing here needs pinning. The one
/// prover-chosen value in a round is `round.odd`, and [`eval_round`] pins it.
#[inline]
fn eval_feistel<AB: AirBuilder>(
    x0: AB::Expr,
    x1: AB::Expr,
    constants: &[AB::F; 2],
) -> (AB::Expr, AB::Expr) {
    let y1 = x1 + constants[0].dup();
    let y2 = x0 + y1.dup().square();
    let y3 = y1 + y2.dup() + constants[1].dup();
    let y5 = y3.dup() + y2 + y3.dup().square();
    (y5, y3)
}

/// The whole call, over one row's columns.
///
/// Flat: no groups, because every round commits. The boundary conditions POLICY
/// §9 asks to be asserted rather than assumed are both here and both cheap:
/// `inputs` is where the state starts, and `outputs` is asserted equal to `M_IO`
/// of the last round's state. There is no padding row to worry about — tables are
/// full (POLICY §6) — and no next row, since one row is one call.
pub fn eval<AB: AirBuilder, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>(
    air: &HalfCommitAir<AB::F, WIDTH, PAIRS, ROUNDS>,
    builder: &mut AB,
    local: &HalfCommitCols<AB::Var, WIDTH, PAIRS, ROUNDS>,
) {
    let mut state: [AB::Expr; WIDTH] = local.inputs.map(Into::into);

    linear::m_io(state.as_mut_slice());
    // The constants are indexed by round and so are the cells, so this zip is the
    // only place the two indexings meet — `tests/air.rs`'s
    // `every_round_constant_is_read_by_the_constraints` is what checks it.
    for (round, constants) in local.rounds.iter().zip(&air.constants) {
        eval_round::<AB, WIDTH, PAIRS>(&mut state, round, constants, builder);
    }
    linear::m_io(state.as_mut_slice());

    assert_state_eq(builder, &state, &local.outputs);
}

impl<AB: AirBuilder, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize> Air<AB>
    for HalfCommitAir<AB::F, WIDTH, PAIRS, ROUNDS>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &HalfCommitCols<_, WIDTH, PAIRS, ROUNDS> = main.current_slice().borrow();

        eval::<AB, WIDTH, PAIRS, ROUNDS>(self, builder, local);
    }
}

#[cfg(test)]
mod tests {
    use p3_air::symbolic::{AirLayout, get_max_constraint_degree, get_symbolic_constraints};
    use p3_mersenne_31::Mersenne31;

    use super::*;

    /// Degree 4 at every round count, which is the point.
    ///
    /// A recurrence that compounds would still read 4 at some round counts, so
    /// asserting `R = 52` alone would not distinguish a fixed point from a
    /// coincidence. `200` is there to say that nothing about `52` is load-bearing.
    #[test]
    fn the_degree_is_a_fixed_point_at_four() {
        for rounds in [1, 2, 3, 4, 52, 200] {
            assert_eq!(
                max_constraint_degree(rounds),
                4,
                "the committed-odd schedule must be a fixed point at degree 4, \
                 but round count {rounds} reached something else"
            );
        }
    }

    /// What the other two parities cost, as arithmetic rather than as prose.
    ///
    /// This is the test that records the design decision. The committed slot is
    /// the whole scheme and getting it wrong fails *silently* — every
    /// known-answer test still passes, the round function is still computed
    /// correctly, and only the degree moves. In the style of
    /// `full_commit::air::tests::a_longer_period_does_not_converge`, and for the
    /// same reason: it is what stops a future reader simplifying the parity away.
    #[test]
    fn the_other_parities_do_not_converge() {
        // The even slot: the odd track doubles every round, so the constraint
        // degree is `2^(r+1)`.
        assert_eq!(degree_of_schedule(CommittedSlot::Even, 1), 4);
        assert_eq!(degree_of_schedule(CommittedSlot::Even, 6), 1 << 7);
        assert_eq!(degree_of_schedule(CommittedSlot::Even, 12), 1 << 13);
        assert_eq!(degree_of_schedule(CommittedSlot::Even, 52), 1 << 53);
        assert!(degree_of_schedule(CommittedSlot::Even, 52) > 1 << 32);

        // Alternating: `M`'s `z` reaches both outputs of every Feistel, so there
        // is no independent track to alternate between and this only halves the
        // doubling rate. The phase is immaterial to the verdict.
        assert_eq!(
            degree_of_schedule(CommittedSlot::Alternating { odd_first: false }, 52),
            1 << 28
        );
        assert_eq!(
            degree_of_schedule(CommittedSlot::Alternating { odd_first: true }, 52),
            1 << 27
        );

        // Growth in the round count is what "does not converge" means: the same
        // schedule over more rounds reaches a strictly higher degree, where the
        // committed-odd one does not.
        for slot in [
            CommittedSlot::Even,
            CommittedSlot::Alternating { odd_first: false },
        ] {
            assert!(
                degree_of_schedule(slot, 52) > degree_of_schedule(slot, 30),
                "{slot:?} must still be growing at R = 52"
            );
        }
        assert_eq!(max_constraint_degree(52), max_constraint_degree(30));
    }

    /// Against `full_commit`'s two comparable variants, at `R = 52`.
    ///
    /// The dominance claims of [`crate::half_commit`], as arithmetic: `StateOnly*`
    /// is degree 4 at twice the cells per round, `Spaced*` is the same cells per
    /// round at degree 16. Both numbers come out of a different recurrence in a
    /// different file, so a change to either one that broke the comparison would
    /// otherwise only show up in a table nobody recomputes.
    #[test]
    fn it_matches_state_only_and_beats_spaced_on_degree() {
        // StateOnly*: REGISTERS = 0, commit every round. `t/2` cells per round.
        assert_eq!(
            crate::full_commit::air::max_constraint_degree(0, 0, 0, 8, 52),
            4
        );
        assert_eq!(max_constraint_degree(52), 4);
        // Spaced*: REGISTERS = 0, commit every second round. `t/4` cells per
        // round, the same width as here.
        assert_eq!(
            crate::full_commit::air::max_constraint_degree(0, 0, 1, 8, 52),
            16
        );
    }

    /// The declared degree against the **evaluator's** symbolic degree.
    ///
    /// The recurrence above cannot see the one mistake that matters most here:
    /// writing the raw degree-4 form in [`eval_round`] instead of the differenced
    /// one. That would leave every known-answer test passing, leave
    /// `max_constraint_degree` reading 4, and buy a blowup the arithmetization has
    /// not earned (POLICY §7). Only the symbolic pass over the real evaluator
    /// notices, so it is checked here — at one grid point, as early as the
    /// evaluator exists — and pinned per instance in `tests/numbers.rs`.
    ///
    /// The constant *values* are irrelevant to a degree, so this builds the AIR
    /// from zeros rather than from an instance: what is being checked is the shape
    /// of the expressions, not the permutation.
    #[test]
    fn the_differenced_form_evaluates_to_degree_four() {
        // t = 16, PAIRS = 4, R = 52.
        let air = HalfCommitAir::<Mersenne31, 16, 4, 52>::new([[[Mersenne31::ZERO; 2]; 4]; 52]);
        let layout = AirLayout::from_air(&air);
        layout.validate_against_air(&air);

        // No periodic columns and no transition selector — one row is one call —
        // so the degree does not depend on the trace length.
        assert_eq!(get_max_constraint_degree(&air, layout, 1 << 10), 4);
        assert_eq!(
            BaseAir::<Mersenne31>::max_constraint_degree(&air),
            Some(get_max_constraint_degree(&air, layout, 1 << 10))
        );

        // One constraint per committed cell, plus the `outputs` assert: the count
        // `tests/numbers.rs` pins at `VECTOR_LEN` scale. Here it is the guard that
        // the differencing did not quietly become a second assertion — a
        // `commit_state` over the even half too would still be degree 4 and would
        // cost `t/4` more constraints and `t/4` more cells.
        assert_eq!(
            get_symbolic_constraints::<Mersenne31, _>(&air, layout).len(),
            52 * 4 + 16
        );
        assert_eq!(BaseAir::<Mersenne31>::width(&air), 240);
    }
}
