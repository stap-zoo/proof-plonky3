//! Trace columns for one Neptune call. Every round commits its post-ARK state,
//! and the degree-reduced variants commit the intermediates that get it there.
//!
//! # Why the rounds are three arrays and not one
//!
//! Neptune's rounds are not interchangeable: `HALF_EXT` external rounds, then
//! `INT` internal ones, then `HALF_EXT` external again. The unsplit variant can
//! ignore that — every round commits the same `WIDTH` cells — but the register
//! variants cannot, because the two kinds witness different things: an external
//! round commits one Lai--Massey square per *pair*, an internal round commits
//! the power-map chain of its single S-box. One flat array of a union of the
//! two would leave dead cells in every round, and a dead cell is a cell nothing
//! pins (POLICY §9) — `tests/air.rs`'s `no_committed_cell_is_dead` is what would
//! catch it. Splitting the array by round kind is what makes the layout carry
//! Neptune's actual schedule instead of an approximation of it, and it removes
//! the round-index-to-register-index mapping that would otherwise be computed
//! by hand at two call sites (POLICY §6).
//!
//! At `LM = PREGS = 0` this is byte-for-byte the layout it replaces: `inputs`,
//! then one `post` per round in round order, and the permutation's output is
//! still the trailing `WIDTH` cells of the call.

use harness::permutation::impl_call_columns;

/// All cells of one independent call.
#[repr(C)]
pub struct NeptuneCols<
    T,
    const WIDTH: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const LM: usize,
    const PREGS: usize,
> {
    /// State before the leading external linear layer.
    pub inputs: [T; WIDTH],
    /// The external rounds before the internal block.
    pub first: [ExternalRound<T, WIDTH, LM>; HALF_EXT],
    /// The internal, single-S-box rounds.
    pub internal: [InternalRound<T, WIDTH, PREGS>; INT],
    /// The external rounds after the internal block.
    pub last: [ExternalRound<T, WIDTH, LM>; HALF_EXT],
}

/// Cells committed by one external (Lai--Massey) round.
#[repr(C)]
pub struct ExternalRound<T, const WIDTH: usize, const LM: usize> {
    /// One Lai--Massey first square `(p0 - p1)²` per pair, or none.
    ///
    /// Empty in the unsplit variant, where the pair map is evaluated as one
    /// degree-four expression instead.
    pub lm: [T; LM],
    /// The fully constrained state at the end of this round.
    pub post: [T; WIDTH],
}

/// Cells committed by one internal round.
#[repr(C)]
pub struct InternalRound<T, const WIDTH: usize, const PREGS: usize> {
    /// The power map's registers: none, `x³`, or `x²`, `x³`, `x⁶`.
    pub powers: [T; PREGS],
    /// The fully constrained state at the end of this round.
    pub post: [T; WIDTH],
}

/// Layout and variant invariant, checked where an AIR is built.
///
/// The admissible `(alpha, LM, PREGS)` triples and the degree each buys:
///
/// | variant | LM | PREGS | degree | why |
/// |---|---|---|---|---|
/// | unsplit | 0 | 0 | `max(alpha, 4)` | the pair map is one degree-four expression |
/// | split | `WIDTH/2` | 1 | 3 | pair map at two, `x⁷` through one register |
/// | flattened | `WIDTH/2` | 3 | 2 | pair map at two, `x⁷` through the three-cell chain |
///
/// Splitting the power map without splitting the pair map is refused rather
/// than allowed: it would pay a cell per internal round for a degree the
/// external rounds immediately give back.
pub const fn assert_layout(
    width: usize,
    ext: usize,
    half_ext: usize,
    alpha: u64,
    lm: usize,
    pregs: usize,
) {
    assert!(width.is_multiple_of(2), "Neptune width must be even");
    assert!(
        ext == 2 * half_ext,
        "Neptune's external rounds split evenly around the internal block"
    );
    assert!(
        lm == 0 || lm == width / 2,
        "one Lai--Massey register per pair, or none at all"
    );
    match (alpha, pregs) {
        (3 | 5 | 7, 0) => {}
        (7, 1 | 3) => assert!(
            lm != 0,
            "splitting the power map below the Lai--Massey floor buys nothing \
             unless the pair map is split too"
        ),
        _ => panic!("Neptune supports the unsplit power map, or one or three registers at alpha 7"),
    }
}

impl_call_columns!(
    NeptuneCols,
    WIDTH: usize,
    HALF_EXT: usize,
    INT: usize,
    LM: usize,
    PREGS: usize,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The width formula, spelled out, against `size_of`.
    #[test]
    fn the_cell_count_is_the_layout_it_declares() {
        // Unsplit: exactly the pre-register layout, `WIDTH + ROUNDS * WIDTH`.
        assert_eq!(num_cols::<12, 3, 42, 0, 0>(), 12 + 48 * 12);
        assert_eq!(num_cols::<8, 3, 38, 0, 0>(), 8 + 44 * 8);
        // Split: one register per pair per external round, one per internal.
        assert_eq!(
            num_cols::<12, 3, 42, 6, 1>(),
            12 + 6 * (6 + 12) + 42 * (1 + 12)
        );
        // Flattened: three registers per internal round.
        assert_eq!(
            num_cols::<12, 3, 42, 6, 3>(),
            12 + 6 * (6 + 12) + 42 * (3 + 12)
        );
        assert_eq!(num_cols::<8, 3, 38, 4, 3>(), 8 + 6 * (4 + 8) + 38 * (3 + 8));
    }

    /// An empty register array must cost nothing, or the unsplit variant pays
    /// for cells it does not have and `align_to` stops landing on row
    /// boundaries.
    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<ExternalRound<u8, 8, 0>>(), 8);
        assert_eq!(core::mem::size_of::<InternalRound<u8, 8, 0>>(), 8);
    }

    /// The unsplit layout is the one it replaces, cell for cell: `inputs` then
    /// one `post` per round, output last.
    #[test]
    fn the_unsplit_layout_is_unchanged() {
        assert_eq!(num_cols::<8, 3, 38, 0, 0>(), 360);
        assert_eq!(num_cols::<12, 3, 42, 0, 0>(), 588);
    }
}
