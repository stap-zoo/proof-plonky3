//! One call's cells, declared once, as a type.
//!
//! POLICY §6, for the full-round layout: `t` input cells and `t` cells per
//! round, against the half-round layout's `2t` input/output cells and `2t` per
//! round. The register blocks come in pairs, because a round now pins two power
//! maps per word rather than one.
//!
//! # Cost
//!
//! `WIDTH * (1 + ROUNDS * (1 + 2 * REGISTERS))` cells per call. At `t = 24` and
//! `R = 8` that is 216 unsplit and 600 split, against the half-round layout's
//! 432 and 816. The unsplit saving is exactly a factor of two on the round
//! cells; the split saving is smaller, because a round that pins two power maps
//! needs two register blocks per word where the half-round layout needed one per
//! half-round — the same count — while the state cells it saves are halved.

use harness::permutation::impl_call_columns;

/// One independent permutation call.
#[repr(C)]
pub struct FullRoundCols<T, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize> {
    /// Permutation input, and the first round's forward S-box argument.
    pub inputs: [T; WIDTH],
    /// One entry per full round. The last one's `post` is the permutation
    /// output — there is no trailing layer, so no separate output block.
    pub rounds: [Round<T, WIDTH, REGISTERS>; ROUNDS],
}

/// Cells for one full round.
#[repr(C)]
pub struct Round<T, const WIDTH: usize, const REGISTERS: usize> {
    /// Registers pinning the forward half's `a_i^alpha`, one block per word.
    pub forward: [Power<T, REGISTERS>; WIDTH],
    /// Registers pinning the inverse half's `w_i^alpha`, one block per word.
    ///
    /// `w_i` is not a cell — it is the affine form `M^{-1}(post - c)` — so this
    /// block pins a power map of an *expression*, which the gadget supports and
    /// which is the only structural difference from the forward side.
    pub inverse: [Power<T, REGISTERS>; WIDTH],
    /// The state after both halves of this round.
    pub post: [T; WIDTH],
}

/// One power map's flattening cells.
#[repr(C)]
pub struct Power<T, const REGISTERS: usize>(pub [T; REGISTERS]);

/// Layout and variant invariant.
///
/// `HALF_ROUNDS == 2 * ROUNDS` is checked here rather than assumed, because this
/// layout carries both as const parameters: the column count needs `ROUNDS` and
/// the parameter type needs `HALF_ROUNDS`, and stable Rust cannot derive one
/// from the other in a const-generic position. A mismatch would index the wrong
/// constant row and still produce a plausible trace.
pub const fn assert_layout(
    width: usize,
    alpha: u64,
    registers: usize,
    half_rounds: usize,
    rounds: usize,
) {
    assert!(width > 0, "a permutation needs a state");
    assert!(
        half_rounds == 2 * rounds,
        "a Rescue-Prime round is two half-rounds, and each has its own constant row"
    );
    match (alpha, registers) {
        (3 | 5 | 7, 0) | (5 | 7, 1) => {}
        _ => panic!("Rescue-Prime supports unsplit alpha 3/5/7 or one register for alpha 5/7"),
    }
}

impl_call_columns!(FullRoundCols, WIDTH: usize, REGISTERS: usize, ROUNDS: usize);

#[cfg(test)]
mod tests {
    use super::*;

    /// The width formula, spelled out, against `size_of`.
    #[test]
    fn the_cell_count_is_the_layout_it_declares() {
        assert_eq!(num_cols::<8, 0, 8>(), 8 + 8 * 8);
        assert_eq!(num_cols::<8, 1, 8>(), 8 + 8 * (8 + 8 + 8));
        assert_eq!(num_cols::<24, 0, 8>(), 24 + 8 * 24);
        assert_eq!(num_cols::<24, 1, 8>(), 24 + 8 * (24 + 24 + 24));
    }

    /// Against the layout this replaces: half the cells unsplit, and a smaller
    /// saving split, for the reason in the module docs.
    #[test]
    fn it_is_narrower_than_the_half_round_layout() {
        use crate::half_round::columns::num_cols as half_round_cols;
        assert_eq!(num_cols::<24, 0, 8>(), 216);
        assert_eq!(half_round_cols::<24, 0, 16>(), 432);
        assert_eq!(num_cols::<24, 1, 8>(), 600);
        assert_eq!(half_round_cols::<24, 1, 16>(), 816);
    }

    /// An empty register array must cost nothing, or the unsplit variant pays
    /// for cells it does not have.
    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<Power<u8, 0>>(), 0);
        assert_eq!(core::mem::size_of::<Round<u8, 8, 0>>(), 8);
    }
}
