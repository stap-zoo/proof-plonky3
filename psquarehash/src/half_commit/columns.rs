//! One call's cells, declared once, as a type.
//!
//! POLICY §6: a `#[repr(C)]` struct for **one call's** cells, `num_cols()` via
//! `size_of`, `Borrow`/`BorrowMut` for `[T]`. Nothing anywhere else computes an
//! offset by hand, and the `prefix.is_empty()` / `suffix.is_empty()` assertions
//! that stand between a layout edit and silent misalignment come with
//! `harness::permutation::impl_call_columns!`.
//!
//! First of the four files (POLICY §2, step 4) — `air.rs` and `generation.rs`
//! both read the layout off this type, which is what stops them drifting apart
//! silently.
//!
//! # What this layout commits: one cell per Feistel, and it is the odd one
//!
//! Per round, `PAIRS = t/4` cells: for each Feistel `p`, the **odd** element of
//! the output pair it writes, `out[2p+1]`, and nothing else. Not the state
//! ([`Group::post`], `t/2` cells), not the squarings ([`Feistel`], `t/2` cells at
//! `REGISTERS = 2`) — half of the round's output state, from which the AIR
//! reconstructs the other half as a degree-2 expression.
//!
//! [`Group::post`]: crate::full_commit::columns::Group::post
//! [`Feistel`]: crate::full_commit::columns::Feistel
//!
//! Two facts about the round function are what make half a state enough. Both
//! are read off `full_commit`'s Feistel, whose outputs are `(y4, y5)` written
//! into `out[2p]` and `out[2p+1]`:
//!
//! ```text
//! y1 = x1 + c0      y2 = x0 + y1²      y3 = y1 + y2 + c1
//! y4 = y2 + y3²     y5 = y3 + y4
//! ```
//!
//! 1. **The two outputs differ by an expression one squaring shallower.**
//!    `y5 − y4 = y3`, and `y3` is not squared where `y4` is. So committing
//!    `out[2p+1]` pins `out[2p]` up to a correction of degree 2 rather than
//!    leaving it at degree 4: the AIR writes the *differenced* form,
//!    `out[2p] = out[2p+1] − y3 + (affine in the upper half)`, and the raw
//!    degree-4 form never appears. That the two forms are equal only on the
//!    variety the committing constraint cuts out is exactly what flattening is
//!    (POLICY §9); `air.rs` is where it is argued at the write site.
//!
//! 2. **The Feistel is asymmetric in its input pair, and the odd slot is the
//!    steep one.** `x1` enters through `y1`, which is squared twice; `x0` enters
//!    only through `y2`, which is squared once. So an output's degree is
//!    `max(2·deg x0, 4·deg x1)`. The next round reads pair `(state[i],
//!    state[i+1])` with `i` even, so `out[2p+1]` — the committed cell — is what
//!    becomes the next round's `x1`, and the factor of four multiplies a degree-1
//!    column. With the lower half at `[even, odd] = [2, 1]` the next round's
//!    outputs are `max(2·2, 4·1) = 4` at the constraint and `2` again in the even
//!    slot: a fixed point, reached at round 1 and stable at every round count.
//!
//! **The parity is the whole scheme, and choosing it wrong fails silently.**
//! Committing the even slot leaves the steep input uncommitted, and the linear
//! layer offers no escape: `M`'s `z = (2·x[h−2] + x[h−1], x[h−2] + x[h−1])` is
//! added into *both* outputs of every Feistel with `p > 0`, so there is no
//! independent even/odd degree track to alternate between. Over `R = 52` rounds
//! at `t = 16`, odd-only gives degree 4, even-only ≈ 2⁵³ and alternating ≈ 2²⁸ —
//! the same failure mode `full_commit::air` documents for `SPAN ≥ 2`, where a
//! commitment covering only part of the state leaves the rest to compound.
//! Nothing downstream of a wrong parity fails loudly: every KAT still passes and
//! only the degree moves, so `air::max_constraint_degree` asserts the fixed point
//! and `harness::measure` cross-checks it against the symbolic degree on every
//! measured row.
//!
//! # Cost
//!
//! `2·WIDTH + ROUNDS·PAIRS` cells per call: the two boundaries, plus `t/4` per
//! round. At `R = 52` that is 240 cells at `t = 16` and 360 at `t = 24`.
//!
//! It is also the floor at degree 4. A round performs `t/2` multiplications and a
//! degree-4 constraint absorbs two *chained* ones, so `t/4` cells is the minimum
//! — the same argument by which `t/2` is the floor at degree 2
//! (`full_commit::air`, "Two cells per Feistel is also the floor at degree 2"),
//! and the general statement POLICY §11 now carries: one cell per `log₂(D)`
//! chained multiplications is the floor at degree `D`.

use harness::permutation::impl_call_columns;

/// One permutation call's cells.
///
/// `WIDTH` is `t` and `PAIRS` is `t/4`, a separate const parameter because stable
/// Rust cannot write `[T; WIDTH / 4]`; [`assert_layout`] is what stops the two
/// disagreeing. There is no `REGISTERS`, no `SPAN`, no `POST` and no `GROUPS`:
/// this layout commits the same one cell per Feistel in every round, which is the
/// axis it exists to sit on.
#[repr(C)]
pub struct HalfCommitCols<T, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize> {
    /// The permutation input, *before* `M_IO`.
    ///
    /// The input rather than `M_IO(input)`: `M_IO` is affine, so committing its
    /// output would spend `t` cells to save nothing, and a KAT compares against
    /// the permutation's input, not against an internal state.
    pub inputs: [T; WIDTH],

    /// The rounds, in order. Flat — this layout has no grouping, because every
    /// round commits.
    pub rounds: [Round<T, PAIRS>; ROUNDS],

    /// The permutation output, *after* the trailing `M_IO`.
    ///
    /// Committed rather than left as an expression for the same three reasons as
    /// in [`full_commit`](crate::full_commit::columns): it is what POLICY §10's
    /// layer 2 reads off a KAT-input trace, it makes the trailing `M_IO` a
    /// constraint rather than an expression a test happens to recompute the same
    /// way, and it is the only place a call's result exists as a cell — the round
    /// cells hold half a state each.
    pub outputs: [T; WIDTH],
}

/// One round's cells: the **odd** element of each Feistel's output pair, and
/// nothing else.
///
/// Block `p` is `out[2p+1]` of the round's new lower half — the Feistel that
/// reads *input* pair `t/2 − 2 − 2p` and round-constant pair `p`, in the indexing
/// [`full_commit::air`](crate::full_commit::air)'s fused round uses. The even
/// element `out[2p]` is never a column: the AIR recovers it by differencing
/// against this cell, and the module docs say why it must be this parity and not
/// the other.
///
/// Every cell in here is prover-chosen and worth nothing until an `assert_eq`
/// ties it to the state (POLICY §9) — and here it is worth *less* than nothing,
/// because the even half is written in terms of it: a free `odd` would make both
/// halves of the round's output arbitrary. `air.rs`'s `commit_state` call is
/// where that assert lives.
#[repr(C)]
pub struct Round<T, const PAIRS: usize> {
    /// The odd element of each Feistel's output pair, one per Feistel.
    ///
    /// Length exactly `PAIRS`: this layout commits one cell per Feistel and the
    /// soundness of the even half rests on there being one for every Feistel. A
    /// shorter array would leave some Feistel's output pair unpinned at both
    /// slots; a longer one would put cells in the trace that no constraint reads,
    /// which is POLICY §9's other failure mode.
    pub odd: [T; PAIRS],
}

/// The layout invariant every type here is parameterized on.
///
/// `WIDTH = 4 * PAIRS` because a round consumes the lower half two elements at a
/// time and the linear layer singles out the last lower-half *pair*; `PAIRS ≥ 2`
/// because `t < 8` leaves that pair undefined. Both are
/// [`full_commit`](crate::full_commit::columns)'s, and they are the only
/// invariants that survive here — the variant parameters this layout drops take
/// their invariants with them, and what replaces them is [`Round::odd`]'s length,
/// which the type system already fixes at `PAIRS`.
///
/// Called from every `new`, so a mismatched instantiation fails at
/// monomorphization instead of building a trace whose cells mean something other
/// than what `air.rs` reads.
pub const fn assert_layout(width: usize, pairs: usize) {
    assert!(width == 4 * pairs, "WIDTH must be 4 * PAIRS");
    assert!(
        pairs >= 2,
        "t < 8 leaves the linear layer's z-pair undefined"
    );
}

impl_call_columns!(
    HalfCommitCols,
    WIDTH: usize,
    PAIRS: usize,
    ROUNDS: usize,
);

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;

    /// The width formula, spelled out, against `size_of`. This is the one number
    /// every other cost column is downstream of, and POLICY §11 pins it — but a
    /// pin only catches a *change*, not a layout that was wrong from the start.
    #[test]
    fn cell_count_is_inputs_plus_rounds_plus_outputs() {
        // t = 16, PAIRS = 4, R = 52: 16 + 52 * 4 + 16 = 240.
        assert_eq!(num_cols::<16, 4, 52>(), 16 + 52 * 4 + 16);
        assert_eq!(num_cols::<16, 4, 52>(), 240);

        // t = 24, PAIRS = 6, R = 52: 24 + 52 * 6 + 24 = 360.
        assert_eq!(num_cols::<24, 6, 52>(), 24 + 52 * 6 + 24);
        assert_eq!(num_cols::<24, 6, 52>(), 360);

        // One round's cells are one per Feistel, which is what makes the round
        // term `t/4` rather than the `t/2` every crate-root variant pays.
        assert_eq!(size_of::<Round<u8, 4>>(), 4);
        assert_eq!(size_of::<Round<u8, 6>>(), 6);
    }

    /// Against `full_commit`'s five variants at the same instance, because the
    /// width is the reason this layout exists.
    ///
    /// The `Spaced*` collision is the one to notice: identical cell counts, told
    /// apart only by their degree — 4 here against 16 there. The numbers alone do
    /// not distinguish the two rows, which is why `tests/numbers.rs` pins the
    /// degree beside the width and why a copy-paste between those rows would
    /// survive everything except the degree pin.
    #[test]
    fn it_is_half_the_state_variants_and_the_width_of_the_spaced_one() {
        use crate::full_commit::columns::num_cols as root;

        // StateOnly16 and Flattened16: t/2 per round, 448 cells.
        assert_eq!(root::<16, 4, 0, 0, 0, 8, 52>(), 448);
        assert_eq!(root::<16, 4, 2, 2, 0, 0, 52>(), 448);
        // Spaced16: t/4 per round at degree 16 — the same 240 cells.
        assert_eq!(root::<16, 4, 0, 0, 1, 8, 26>(), 240);
        assert_eq!(num_cols::<16, 4, 52>(), 240);

        // And the same story at t = 24: 672 against 360.
        assert_eq!(root::<24, 6, 0, 0, 0, 12, 52>(), 672);
        assert_eq!(root::<24, 6, 0, 0, 1, 12, 26>(), 360);
        assert_eq!(num_cols::<24, 6, 52>(), 360);
    }

    /// The invariant, at the two registered widths and at the two ways to break
    /// it. `assert_layout` runs at monomorphization in `air.rs`, so a `should_panic`
    /// test is the only place its *failure* is reachable at runtime.
    #[test]
    fn the_layout_invariant_holds_at_the_grid_widths() {
        assert_layout(16, 4);
        assert_layout(24, 6);
    }

    #[test]
    #[should_panic(expected = "WIDTH must be 4 * PAIRS")]
    fn a_pairs_count_that_is_not_a_quarter_of_the_width_is_rejected() {
        assert_layout(16, 8);
    }

    #[test]
    #[should_panic(expected = "z-pair")]
    fn a_state_too_small_for_the_linear_layer_is_rejected() {
        assert_layout(4, 1);
    }
}
