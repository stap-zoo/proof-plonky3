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
//! inverse-power output, every canonicity flag. First row, last row, and the
//! transitions into and out of a call in a multi-row layout are asserted, not
//! assumed.
//!
//! # The fused round
//!
//! `native.rs` applies the nonlinear layer and then `M`, as two steps over a
//! materialized intermediate state, exactly as `hash.py` does. This file does
//! not: `M` is a half-swap plus a handful of additions, so it is folded into the
//! Feistel's write and the intermediate state never exists. Reading the two side
//! by side is how the fusion gets checked — and the KAT is what settles it.
//!
//! Written out, with `h = t/2` and `p` ranging over the `t/4` Feistels:
//!
//! ```text
//! z       = (2·x[h-2] + x[h-1],  x[h-2] + x[h-1])
//! extra   = (Σ x[2q],  Σ x[2q+1])          for q = 1 .. t/4 - 2
//!
//! out[2p]     = x[h+2p]     + y0(pair h-2-2p)  + [p > 0] z.0  + [p = t/4-1] extra.0
//! out[2p+1]   = x[h+2p+1]   + y1(pair h-2-2p)  + [p > 0] z.1  + [p = t/4-1] extra.1
//! out[h + i]  = x[i]
//! ```
//!
//! `z` and `extra` read the *old* lower half, which the round does not modify, so
//! the order of the writes is free. Everything in that display except the two `y`
//! terms is `linear.rs`, shared with `generation.rs` and with the other
//! arithmetization: `M` is the same matrix in all four files, and which `y` a
//! variant writes is the only thing that distinguishes them.
//!
//! # Degree accounting
//!
//! Let `d` be the algebraic degree of the round's input state in the committed
//! cells. One Feistel is
//!
//! ```text
//! y1 = x1 + c0        deg d
//! y2 = x0 + y1²       deg 2d   (deg d if y1² is committed)
//! y3 = y1 + y2 + c1   same as y2
//! y4 = y2 + y3²       deg 2·deg(y3)   (deg of y2 if y3² is committed)
//! y5 = y3 + y4        deg max(deg y3, deg y4)
//! ```
//!
//! so the round's output degree is `4d`, `2d` or `d` for `REGISTERS` 0, 1, 2, and
//! a committed square's own constraint is degree `2·deg` of what it squares. With
//! the state committed each round (`d = 1` at every round entry) that gives:
//!
//! | `REGISTERS` | register constraint | `post` constraint | max degree |
//! |---|---|---|---|
//! | 0 | — | 4 | **4** |
//! | 1 | 2 | 2 | **2** |
//! | 2 | 2 | none needed | **2** |
//!
//! `REGISTERS = 2` is the interesting one: committing *both* squarings leaves
//! `y4` and `y5` affine, so the state never leaves degree 1 and the state columns
//! stop paying for themselves. Two cells per Feistel is also the floor for a
//! degree-2 arithmetization of this round — one committed cell per
//! multiplication, and a Feistel performs exactly two.
//!
//! The expressions do keep *growing* in that variant — round 52's state is an
//! affine combination of some 400 cells — but not in a way anything charges for.
//! `AB::Expr` is accumulated as a value, so the prover and verifier folders do
//! `O(t)` work per round either way, and `SymbolicExpression` shares subtrees
//! through `Arc` and caches its own degree, so the symbolic pass stays linear too.
//!
//! Two cells per Feistel is also the floor at degree 2. `y4` has algebraic degree
//! 4 in the input pair, one multiplication at most doubles degree, and the `t/4`
//! Feistels of a round read disjoint pairs — so a round needs `t/2`
//! multiplications and, at degree 2, `t/2` cells to hold them. Only the `2t`
//! boundary cells have any slack.
//!
//! # Rounds per state commitment
//!
//! Committing the state every `k` rounds instead of every round is the second
//! variant axis of POLICY §11, and it is `SPAN` — a group is `SPAN` rounds that
//! commit nothing, then one that does. A round's upper half is the previous
//! round's lower half, so with `a_r = deg(lower half after round r)`:
//!
//! ```text
//! a_{r+1} = max(a_{r-1}, mult · a_r),   mult = 4 | 2 | 1  for REGISTERS 0 | 1 | 2
//! ```
//!
//! a register constraint in round `r+1` has degree `2·a_r`, and a commit costs
//! `h` cells, constrains at the pre-commit degree, and resets `a` to 1. The
//! frontier that falls out:
//!
//! | period | cells / round | max degree | min `log_blowup` |
//! |---|---|---|---|
//! | flatten every round, never commit | `h` | 2 | 1 |
//! | (0 reg); (1 reg + commit) | `3h/4` | 8 | 3 |
//! | (0 reg); (0 reg + commit) | `h/2` | 16 | 4 |
//! | `k-1 ×` (0 reg); (0 reg + commit), `k ≥ 3` | `h/k` | **unbounded in `R`** | — |
//!
//! `3h/4` is minimal at degree ≤ 9; degree 4 buys nothing over degree 2.
//!
//! **The last row is why this axis stops at period 2, and it is a correction to
//! the `4^k` this table used to claim.** A commitment pins the new *lower* half,
//! `h` cells; the upper half is the previous round's lower half, which at period
//! `k ≥ 3` was itself never committed. So a group starts from `(lower = 1,
//! upper = a)` with `a` the degree two rounds back, and that `a` — not the reset
//! — is what the next group multiplies. At `k = 2` the surviving term is the
//! immediately preceding round's, the recurrence has a fixed point, and the
//! degree is 16. At `k = 3` it compounds instead: `4^19` by round 52 at
//! `R = 52`, which no blowup serves. Reaching the `h/k` widths would mean
//! committing the *whole* state, `t` cells rather than `t/2` — a different
//! layout with a different cost, not a longer period on this one.
//!
//! `R = 52 = 2 · 26` divides evenly, so the period-2 schedules fit with no ragged
//! group, and both of them are registered (`instances::Spaced*`). Longer periods
//! stay expressible — `SPAN = k - 1`, `GROUPS = R / k` — and unregistered.
//!
//! **The two readings of POLICY §11 pick different winners here, and only one of
//! them is the primary reading.** At each AIR's own minimum blowup — which is
//! what a measured row runs at — `width × 2^log_blowup` runs `h`, `3h`, `4h`,
//! `10.7h` down the table, so flattening wins outright and the spacing axis is a
//! width saving that the code rate more than takes back. At the common
//! `log_blowup = 3` every degree up to 9 pays blowup 8 regardless, so the
//! degree-8 period looks 25% narrower for free — which is exactly the reading
//! that prices what a high degree saves and none of what it costs. The numbers
//! settle it rather than this comment: these are the rows to compare.
//!
//! [`max_constraint_degree`] replays the recurrence above rather than reading a
//! table, and `harness::measure` cross-checks its answer against the symbolic
//! degree on every measured row — so a schedule this comment never anticipated
//! still reports the degree it actually has.

use core::borrow::Borrow;

use harness::gadgets::power_map::eval_power_map;
use harness::permutation::{assert_state_eq, commit_state};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing};

use crate::full_commit::columns::{
    Feistel, Group, PSquareHashCols, Round, assert_layout, assert_rounds, num_cols,
};
use crate::linear;

/// The AIR for one pSquareHash instance at one variant. Constants only.
#[derive(Debug, Clone)]
pub struct PSquareHashAir<
    F,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
> {
    /// `ROUNDS` rows of `PAIRS` constant pairs; pair `p` belongs to the Feistel
    /// writing output pair `p`. See `crate::params::PSquareHashParams::rcons`.
    ///
    /// Flat rather than grouped: the round constants belong to the *permutation*
    /// and the grouping belongs to the arithmetization, so a variant that
    /// regroups the rounds must not be able to regroup the constants with them.
    pub(crate) constants: [[[F; 2]; PAIRS]; ROUNDS],
}

impl<
    F,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
> PSquareHashAir<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>
{
    /// Forces the layout assertions at monomorphization: an associated const of
    /// a generic type is evaluated when it is used, so an illegal parameter set
    /// is a compile error rather than a trace whose cells mean something else.
    const LAYOUT: () = {
        assert_layout(WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST);
        assert_rounds(SPAN, GROUPS, ROUNDS);
    };

    /// Build the AIR for one instance at one variant.
    #[must_use]
    pub const fn new(constants: [[[F; 2]; PAIRS]; ROUNDS]) -> Self {
        let () = Self::LAYOUT;
        Self { constants }
    }
}

/// The degree a round multiplies its input state's degree by.
///
/// `y4 = y2 + y3²` is where it comes from: with neither squaring committed `y3`
/// is degree 2 in the round's input, so `y4` is degree 4; committing `y1²` makes
/// `y3` degree 1 and `y4` degree 2; committing `y3²` as well leaves the round
/// affine.
const fn multiplier(registers: usize) -> usize {
    match registers {
        0 => 4,
        1 => 2,
        2 => 1,
        _ => panic!("REGISTERS must be 0, 1 or 2"),
    }
}

/// The maximum constraint degree of a variant, from the recurrence rather than
/// from a table.
///
/// A `const fn` replaying the module docs' `a_{r+1} = max(a_{r-1}, mult · a_r)`
/// over all `ROUNDS`, because `harness::measure` cross-checks the declared value
/// against `get_max_constraint_degree` and a wrong declaration would otherwise
/// buy a silently cheaper blowup (POLICY §7). Three kinds of constraint compete
/// for the maximum:
///
/// * a **register** constraint in a round reading a state of degree `a`, which
///   is `2a` — `y1²` squares the input, and `y3` is affine in it once `y1²` is a
///   column;
/// * a **commitment**, which constrains at the degree the group's last round
///   reached;
/// * the **output** constraint, `M_IO` of the final state, which is affine in
///   the last two rounds' halves.
///
/// The upper half is the previous round's lower half and is *not* reset by a
/// commitment, which is exactly why the recurrence carries two terms: with
/// `SPAN > 0` a group's first round reads a committed lower half and an
/// uncommitted upper one.
#[must_use]
pub const fn max_constraint_degree(
    registers: usize,
    registers_last: usize,
    span: usize,
    post: usize,
    rounds: usize,
) -> usize {
    let m = multiplier(registers);
    let m_last = multiplier(registers_last);
    let period = span + 1;

    // Entering round 1 both halves are the committed input state, degree 1.
    let mut upper = 1;
    let mut lower = 1;
    let mut max = 1;

    let mut r = 1;
    while r <= rounds {
        let committing = r % period == 0;
        let (m_r, registers_r) = if committing {
            (m_last, registers_last)
        } else {
            (m, registers)
        };

        // The registers this round commits are pinned against its input state.
        if registers_r > 0 && 2 * lower > max {
            max = 2 * lower;
        }

        let feistel = m_r * lower;
        let new_lower = if feistel > upper { feistel } else { upper };
        upper = lower;
        lower = new_lower;

        if committing && post != 0 {
            if lower > max {
                max = lower;
            }
            lower = 1;
        }
        r += 1;
    }

    let output = if lower > upper { lower } else { upper };
    if output > max { output } else { max }
}

/// The Feistel's algebraic degree in its input pair, for `Labels::sbox_degree`.
///
/// pSquareHash has no power map, so there is no `alpha`. Four is the honest
/// answer: `y4 = x0 + (x1+c0)² + ((x1+c0) + x0 + (x1+c0)² + c1)²` has degree 4 in
/// `(x0, x1)`, and it is what the round's degree multiplier is when nothing is
/// committed.
pub const FEISTEL_DEGREE: u64 = 4;

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
> BaseAir<F>
    for PSquareHashAir<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>()
    }

    /// One row is one call, so nothing is read from the next row. Declaring it
    /// lets the prover skip opening the shifted trace entirely (POLICY §6).
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(max_constraint_degree(
            REGISTERS,
            REGISTERS_LAST,
            SPAN,
            POST,
            ROUNDS,
        ))
    }
}

/// One group: `SPAN` rounds, the committing round, and the commitment.
///
/// The commitment is the only place a degree is reset, and it is *one* place per
/// group rather than one per round — which is the whole content of the spacing
/// axis (POLICY §11).
///
/// POLICY §9: every `post` cell is prover-chosen and the `assert_eq` below is
/// what pins it. A `post` the prover picked freely would let the group's output
/// state be anything at all.
#[inline]
fn eval_group<
    AB: AirBuilder,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    group: &Group<AB::Var, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST>,
    constants: &[[[AB::F; 2]; PAIRS]],
    builder: &mut AB,
) {
    debug_assert_eq!(constants.len(), SPAN + 1);

    for (round, constants) in group.rounds.iter().zip(constants) {
        eval_round::<AB, WIDTH, PAIRS, REGISTERS>(state, round, constants, builder);
    }
    eval_round::<AB, WIDTH, PAIRS, REGISTERS_LAST>(state, &group.last, &constants[SPAN], builder);

    // The state commitment, when this variant has one. Only the *lower* half
    // needs it: the upper half is the previous round's lower half, which the
    // group's own rounds already expressed.
    if POST != 0 {
        commit_state(builder, &mut state[..WIDTH / 2], &group.post);
    }
}

/// One round: the nonlinear layer and `M`, fused. See the module docs.
#[inline]
fn eval_round<AB: AirBuilder, const WIDTH: usize, const PAIRS: usize, const REGISTERS: usize>(
    state: &mut [AB::Expr; WIDTH],
    round: &Round<AB::Var, PAIRS, REGISTERS>,
    constants: &[[AB::F; 2]; PAIRS],
    builder: &mut AB,
) {
    let h = WIDTH / 2;

    // `M`'s contributions to the new lower half that no Feistel supplies, read
    // off the old lower half — which this round leaves alone.
    let (z, extra) = linear::prologue(state.as_slice());

    let mut next = core::array::from_fn::<_, WIDTH, _>(|_| AB::Expr::ZERO);
    for p in 0..PAIRS {
        // Output pair p is written by the Feistel over input pair h-2-2p, and
        // reads round-constant pair p. See `PSquareHashParams::rcons`.
        let i = h - 2 - 2 * p;
        let y = eval_feistel(
            state[i].dup(),
            state[i + 1].dup(),
            &constants[p],
            &round.feistels[p],
            builder,
        );
        for k in 0..2 {
            let mut out = state[h + 2 * p + k].dup() + y[k].dup();
            if p > 0 {
                out += z[k].dup();
            }
            if p == PAIRS - 1 {
                out += extra[k].dup();
            }
            next[2 * p + k] = out;
        }
    }

    // After the loop, so that `z`, `extra` and the Feistels read the old values.
    linear::write_upper_half(&mut next, state.as_slice());
    *state = next;
}

/// One Feistel, at whatever flattening the variant asks for.
///
/// Returns `(y4, y5)`, the pair the round adds into the upper half.
///
/// POLICY §9, per witnessed value:
///
/// * `feistel.0[0]` is `y1²`, pinned by `assert_eq(reg, (x1 + c0)²)`. Without
///   that assert the prover picks `y1²` freely and the Feistel computes an
///   arbitrary function.
/// * `feistel.0[1]` is `y3²`, pinned by `assert_eq(reg, y3²)`, where `y3` is
///   itself affine in `x0`, `x1` and the first register.
#[inline]
fn eval_feistel<AB: AirBuilder, const REGISTERS: usize>(
    x0: AB::Expr,
    x1: AB::Expr,
    constants: &[AB::F; 2],
    feistel: &Feistel<AB::Var, REGISTERS>,
    builder: &mut AB,
) -> [AB::Expr; 2] {
    let y1 = x1 + constants[0].dup();

    let y2 = match REGISTERS {
        0 => x0 + y1.square(),
        _ => {
            // Committed y1². Degree 2 in the state; degree 1 from here on.
            let y1_sq = eval_power_map::<AB, 2, 1>(y1.dup(), &[feistel.0[0]], builder);
            x0 + y1_sq
        }
    };
    let y3 = y1.dup() + y2.dup() + constants[1].dup();

    let y3_sq = match REGISTERS {
        2 => {
            // Committed y3². This is the second and last multiplication a Feistel
            // performs, so with it committed the whole round map is affine in
            // committed cells — which is why this variant needs no `post`.
            eval_power_map::<AB, 2, 1>(y3.dup(), &[feistel.0[1]], builder)
        }
        _ => y3.square(),
    };

    let y4 = y2 + y3_sq;
    let y5 = y3 + y4.dup();
    [y4, y5]
}

/// The whole call, over one row's columns.
///
/// The boundary conditions POLICY §9 asks to be asserted rather than assumed are
/// both here and both cheap: `inputs` is where the state starts, and `outputs` is
/// asserted equal to `M_IO` of the last round's state. There is no padding row to
/// worry about — tables are full (POLICY §6) — and no next row, since one row is
/// one call.
pub fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
>(
    air: &PSquareHashAir<
        AB::F,
        WIDTH,
        PAIRS,
        REGISTERS,
        REGISTERS_LAST,
        SPAN,
        POST,
        GROUPS,
        ROUNDS,
    >,
    builder: &mut AB,
    local: &PSquareHashCols<AB::Var, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>,
) {
    let mut state: [AB::Expr; WIDTH] = local.inputs.map(Into::into);

    linear::m_io(state.as_mut_slice());
    for (g, group) in local.groups.iter().enumerate() {
        // The constants are indexed by round and the groups partition the rounds
        // in order, so this slice is the group's own rounds and nothing else.
        let first = g * (SPAN + 1);
        eval_group::<AB, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST>(
            &mut state,
            group,
            &air.constants[first..first + SPAN + 1],
            builder,
        );
    }
    linear::m_io(state.as_mut_slice());

    assert_state_eq(builder, &state, &local.outputs);
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
> Air<AB>
    for PSquareHashAir<AB::F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &PSquareHashCols<
            _,
            WIDTH,
            PAIRS,
            REGISTERS,
            REGISTERS_LAST,
            SPAN,
            POST,
            GROUPS,
        > = main.current_slice().borrow();

        eval::<AB, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>(
            self, builder, local,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frontier of the module docs, replayed by the recurrence.
    ///
    /// This is the table that decides which schedules are worth registering, and
    /// it is arithmetic rather than measurement — so it is asserted here, once,
    /// instead of being restated per instance. `harness::measure` then checks the
    /// same function against the symbolic degree of every row it measures.
    #[test]
    fn the_degree_frontier() {
        // Commit every round (SPAN = 0), all three register choices.
        assert_eq!(max_constraint_degree(0, 0, 0, 8, 52), 4);
        assert_eq!(max_constraint_degree(1, 1, 0, 8, 52), 2);
        assert_eq!(max_constraint_degree(2, 2, 0, 0, 52), 2);

        // Commit every second round.
        assert_eq!(max_constraint_degree(0, 0, 1, 8, 52), 16);
        assert_eq!(max_constraint_degree(0, 1, 1, 8, 52), 8);
        assert_eq!(max_constraint_degree(1, 1, 1, 8, 52), 4);
    }

    /// At period 2 the two register placements are the same trade.
    ///
    /// One register per group either way, one factor of two off the round the
    /// commitment reads either way. `REGISTERS_LAST` is a separate parameter
    /// because the *schedule* distinguishes them in general, not because this
    /// period does — and asserting the equality is what stops a future reader
    /// assuming an asymmetry the arithmetic does not have.
    #[test]
    fn the_two_register_placements_cost_the_same_at_period_two() {
        assert_eq!(max_constraint_degree(0, 1, 1, 8, 52), 8);
        assert_eq!(max_constraint_degree(1, 0, 1, 8, 52), 8);
    }

    /// Why the axis stops at period 2: only half the state is committed, so at
    /// period 3 the uncommitted half compounds instead of resetting.
    ///
    /// This is the corrected version of the `4^k` row the module docs used to
    /// carry. `4^3 = 64` would be measurable; `4^19` is not, and the difference
    /// is the whole reason `instances` registers `SPAN = 1` and stops.
    #[test]
    fn a_longer_period_does_not_converge() {
        assert_eq!(max_constraint_degree(0, 0, 2, 8, 6), 256);
        assert_eq!(max_constraint_degree(0, 0, 2, 8, 12), 4096);
        assert!(max_constraint_degree(0, 0, 2, 8, 52) > 1 << 32);
        // Growth in the round count is what "unbounded" means here: the same
        // schedule over more rounds reaches a strictly higher degree.
        assert!(
            max_constraint_degree(0, 0, 2, 8, 52) > max_constraint_degree(0, 0, 2, 8, 30),
            "the period-3 degree must still be growing at R = 52"
        );
        // Period 2, for contrast: a fixed point, whatever the round count.
        assert_eq!(max_constraint_degree(0, 0, 1, 8, 10), 16);
        assert_eq!(max_constraint_degree(0, 0, 1, 8, 52), 16);
    }
}
