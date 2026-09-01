//! One call's cells, declared once, as a type.
//!
//! POLICY §6: a `#[repr(C)]` struct for **one call's** cells, `num_cols()` via
//! `size_of`, `Borrow`/`BorrowMut` for `[T]`. Nothing anywhere else computes an
//! offset by hand.
//!
//! Keep the `prefix.is_empty()` / `suffix.is_empty()` asserts in the borrow
//! impls: with `#[repr(C)]` they are all that stands between a layout edit and
//! silent misalignment.
//!
//! First of the four files (POLICY §2, step 4) — `air.rs` and `generation.rs`
//! both read the layout off this type, which is what stops them drifting apart
//! silently.
//!
//! # The layout, and why it has this shape
//!
//! One row is one call (times `VECTOR_LEN` lanes, see `vectorized.rs`). Per
//! round the cells are whatever that variant needs to keep the constraint degree
//! down, and nothing else — in particular **the round's output state is not
//! automatically a column.**
//!
//! Three facts about the round function set the whole layout:
//!
//! 1. **A round's upper half is the previous round's lower half, verbatim.**
//!    `M`'s first job is the Feistel swap, so committing "the state after round
//!    r" costs `t/2` cells, not `t`: the other half is already a column one round
//!    back. That is why [`Group::post`] is `t/2` wide.
//! 2. **A Feistel is two squarings and nothing else nonlinear.** `y5 = y3 + y4`
//!    is *affine* in the two squarings' outputs, so committing both squarings
//!    makes the Feistel — and therefore the whole round map — affine in committed
//!    cells. That is [`Feistel`]'s `REGISTERS = 2` variant, and it is why that
//!    variant needs no state commitment at all.
//! 3. **A state commitment need not happen every round.** Skipping one multiplies
//!    the degree and divides the width, which is POLICY §11's second variant
//!    axis. [`Group`] is that axis made a type: `SPAN` rounds that commit
//!    nothing, then one round that does.
//!
//! # The two axes, as const parameters
//!
//! Per round, in cells, with `h = t/2`, `PAIRS = t/4` and `PERIOD = SPAN + 1`:
//!
//! | `REGISTERS` | `REGISTERS_LAST` | `SPAN` | `POST` | cells / round | max degree |
//! |---|---|---|---|---|---|
//! | 0 | 0 | 0 | `h` | `h`      | 4 |
//! | 1 | 1 | 0 | `h` | `3h/2`   | 2 |
//! | 2 | 2 | 0 | `0` | `h`      | 2 |
//! | 0 | 1 | 1 | `h` | `3h/4`   | 8 |
//! | 0 | 0 | 1 | `h` | `h/2`    | 16 |
//!
//! The first three are the commit-every-round row of POLICY §11's frontier; the
//! last two skip every second commitment, at `PERIOD = 2`. The degree column is
//! not a table anyone maintains by hand — [`super::air::max_constraint_degree`]
//! replays the recurrence and `harness::measure` cross-checks it against the
//! symbolic degree on every measured row.
//!
//! `REGISTERS = 2` remains the design to reach for at `SPAN = 0`: same width as
//! `REGISTERS = 0` at half the degree, and narrower than `REGISTERS = 1` at the
//! same degree. What the spacing axis buys is width below that floor, and what
//! it costs is degree — and since a degree buys a code rate (POLICY §11), the
//! two are measured against each other rather than argued about.
//!
//! See [`super::air`] for the degree accounting each row rests on.

use harness::permutation::impl_call_columns;

/// One permutation call's cells.
///
/// `WIDTH` is `t`; `PAIRS` is `t/4`; `POST` is `t/2` or `0`; `GROUPS` is
/// `ROUNDS / (SPAN + 1)`. Each is a separate const parameter because stable Rust
/// cannot write `[T; WIDTH / 4]`, and [`assert_layout`] is what stops them
/// disagreeing.
#[repr(C)]
pub struct PSquareHashCols<
    T,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
> {
    /// The permutation input, *before* `M_IO`.
    ///
    /// The input rather than `M_IO(input)`: `M_IO` is affine, so committing its
    /// output would spend `t` cells to save nothing, and a KAT compares against
    /// the permutation's input, not against an internal state.
    pub inputs: [T; WIDTH],

    /// The rounds, in order, grouped by state commitment.
    pub groups: [Group<T, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST>; GROUPS],

    /// The permutation output, *after* the trailing `M_IO`.
    ///
    /// Committed rather than left as an expression, for three reasons: it is what
    /// POLICY §10's layer 2 reads off a KAT-input trace; it is what makes the
    /// trailing `M_IO` a constraint rather than an unchecked expression the test
    /// happens to recompute the same way; and in the `REGISTERS = 2` variant it
    /// is the *only* committed state cell, so without it a call's result would
    /// exist nowhere in the trace.
    pub outputs: [T; WIDTH],
}

/// `SPAN` rounds that commit no state, then one that does.
///
/// At `SPAN = 0` a group is one round and this is the commit-every-round layout,
/// byte for byte: `[Round; 0]` is a zero-sized field under `#[repr(C)]`, so
/// `Group` is then exactly `last` followed by `post`. Every pinned number for
/// those variants is therefore the number it was before this axis existed, which
/// is what `tests/numbers.rs` checks.
///
/// The committing round carries its own `REGISTERS_LAST`, because the interesting
/// point on the frontier spends registers only where they buy the most: a
/// register on the *committing* round halves the degree the commitment has to
/// absorb, while a register on a round whose result is never committed only
/// halves a degree that the next round multiplies again.
#[repr(C)]
pub struct Group<
    T,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
> {
    /// The rounds whose output state is never committed.
    pub rounds: [Round<T, PAIRS, REGISTERS>; SPAN],

    /// The round the commitment follows.
    pub last: Round<T, PAIRS, REGISTERS_LAST>,

    /// The group's new lower half, `t/2` cells — the other half being the
    /// previous round's lower half, which is an expression rather than a column
    /// whenever `SPAN > 0`.
    ///
    /// Empty when the whole round map is affine in committed cells
    /// (`REGISTERS = REGISTERS_LAST = 2`): a state column would then reset a
    /// degree that never grew.
    pub post: [T; POST],
}

/// One round's cells: its Feistels' flattening registers, and nothing else.
#[repr(C)]
pub struct Round<T, const PAIRS: usize, const REGISTERS: usize> {
    /// One block per Feistel, in the order the round applies them: block `p`
    /// belongs to the Feistel writing **output** pair `p`, which reads *input*
    /// pair `t/2 - 2 - 2p`.
    pub feistels: [Feistel<T, REGISTERS>; PAIRS],
}

/// A Feistel's flattening cells: the squarings it commits instead of expanding.
///
/// `REGISTERS = 0` commits neither and multiplies the round's degree by four.
/// `REGISTERS = 1` commits `y1²`, which halves that multiplier to two.
/// `REGISTERS = 2` commits `y1²` and `y3²`, which are *every* multiplication the
/// Feistel performs — leaving the round affine in committed cells.
///
/// Every cell in here is prover-chosen and worth nothing until an `assert_eq`
/// ties it to the state (POLICY §9). `super::air::eval_feistel` is where those
/// asserts live; `tests/air.rs` sweeps every cell for deadness and
/// `tests/numbers.rs` pins the constraint count, which is the half of the pair
/// that notices a missing assertion.
#[repr(C)]
pub struct Feistel<T, const REGISTERS: usize>(pub [T; REGISTERS]);

/// The layout invariant every type here is parameterized on.
///
/// `WIDTH = 4 * PAIRS` because a round consumes the lower half two elements at a
/// time and the linear layer singles out the last lower-half *pair*. `POST` is
/// `t/2` exactly when the variant commits its state, and dropping the commitment
/// entirely is only sound arithmetically — not for soundness, for *degree* —
/// when every round is already affine in committed cells.
///
/// Called from every `new`, so a mismatched instantiation fails at
/// monomorphization instead of building a trace whose cells mean something other
/// than what `air.rs` reads.
pub const fn assert_layout(
    width: usize,
    pairs: usize,
    registers: usize,
    registers_last: usize,
    span: usize,
    post: usize,
) {
    assert!(width == 4 * pairs, "WIDTH must be 4 * PAIRS");
    assert!(
        pairs >= 2,
        "t < 8 leaves the linear layer's z-pair undefined"
    );
    assert!(registers <= 2, "REGISTERS must be 0, 1 or 2");
    assert!(registers_last <= 2, "REGISTERS_LAST must be 0, 1 or 2");
    if post == 0 {
        // Without a commitment the round map's degree is whatever the recurrence
        // makes it, and over 52 rounds that is only bounded when every round is
        // affine in committed cells.
        assert!(
            registers == 2 && registers_last == 2 && span == 0,
            "a variant with no state commitment must commit both squarings every round"
        );
    } else {
        assert!(
            post == 2 * pairs,
            "a committing variant commits the new lower half, t/2 cells"
        );
    }
}

/// The rounds a group covers, and the check that `GROUPS` divides them.
///
/// Separate from [`assert_layout`] because `ROUNDS` belongs to the AIR — the
/// constants array is indexed by round — while the column layout only needs to
/// know how many groups there are.
pub const fn assert_rounds(span: usize, groups: usize, rounds: usize) {
    assert!(
        rounds == groups * (span + 1),
        "GROUPS * (SPAN + 1) must be exactly ROUNDS: a ragged final group would \
         leave a round's state uncommitted"
    );
}

impl_call_columns!(
    PSquareHashCols,
    WIDTH: usize,
    PAIRS: usize,
    REGISTERS: usize,
    REGISTERS_LAST: usize,
    SPAN: usize,
    POST: usize,
    GROUPS: usize,
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
        // t = 16, PAIRS = 4, R = 52 rounds committed one at a time.
        // REGISTERS = 2, POST = 0: 16 + 52 * (4 * 2 + 0) + 16 = 448.
        assert_eq!(num_cols::<16, 4, 2, 2, 0, 0, 52>(), 448);
        // REGISTERS = 0, POST = 8: 16 + 52 * (0 + 8) + 16 = 448.
        assert_eq!(num_cols::<16, 4, 0, 0, 0, 8, 52>(), 448);
        // REGISTERS = 1, POST = 8: 16 + 52 * (4 + 8) + 16 = 656.
        assert_eq!(num_cols::<16, 4, 1, 1, 0, 8, 52>(), 656);

        // t = 24, PAIRS = 6.
        assert_eq!(num_cols::<24, 6, 2, 2, 0, 0, 52>(), 24 + 52 * 12 + 24);
        assert_eq!(num_cols::<24, 6, 0, 0, 0, 12, 52>(), 24 + 52 * 12 + 24);
        assert_eq!(num_cols::<24, 6, 1, 1, 0, 12, 52>(), 24 + 52 * 18 + 24);
    }

    /// The spacing axis is a width reduction, and this is where the size of it
    /// is visible: 26 groups of two rounds, committing once per group.
    #[test]
    fn spacing_halves_the_committed_state() {
        // t = 16: 16 + 26 * (0 + 0 + 8) + 16 = 240 — h/2 cells per round.
        assert_eq!(num_cols::<16, 4, 0, 0, 1, 8, 26>(), 16 + 26 * 8 + 16);
        // One register on the committing round only: + PAIRS per group.
        assert_eq!(num_cols::<16, 4, 0, 1, 1, 8, 26>(), 16 + 26 * 12 + 16);

        // Against the commit-every-round layouts of the same instance: 240 and
        // 344 cells against 448.
        assert_eq!(num_cols::<16, 4, 0, 0, 1, 8, 26>(), 240);
        assert_eq!(num_cols::<16, 4, 0, 1, 1, 8, 26>(), 344);
        assert_eq!(num_cols::<16, 4, 2, 2, 0, 0, 52>(), 448);
    }

    /// A zero-length array inside `#[repr(C)]` must not pad. If it did, the
    /// `SPAN = 0` layouts would stop being byte-identical to the ones that
    /// existed before the spacing axis, and every pinned number would move.
    #[test]
    fn empty_arrays_cost_nothing() {
        assert_eq!(size_of::<Feistel<u8, 0>>(), 0);
        assert_eq!(size_of::<Round<u8, 4, 0>>(), 0);
        assert_eq!(size_of::<Group<u8, 4, 0, 0, 0, 8>>(), 8);
        assert_eq!(size_of::<Group<u8, 4, 2, 2, 0, 0>>(), 8);
        // One skipped commitment: two rounds' registers and one `post`.
        assert_eq!(size_of::<Group<u8, 4, 0, 1, 1, 8>>(), 12);
    }
}
