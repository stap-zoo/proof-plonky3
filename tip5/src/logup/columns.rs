//! One lookup-backed Tip5 call's cells, declared once as a type.
//!
//! The lookup-free baseline commits, per split-and-lookup word, 64 input bits,
//! sixteen canonical-prefix flags, 64 output bits and eight sixteen-bit
//! quotients — 272 cells. This layout commits eight input bytes, eight output
//! bytes and the shared two-cell input-canonicity witness — eighteen. What
//! replaces the missing 254 cells is one `(input byte, output byte)` query per
//! byte on a Tip5-only bus: the query is what range-binds both coordinates and
//! what evaluates the byte permutation, so neither the bit decomposition nor
//! the division-by-257 quotient has anything left to do.
//!
//! Everything else is the lookup-free layout unchanged, including
//! [`crate::columns::PowerWord`], which is reused rather than restated: the
//! seventh-power words are not touched by the lookup at all.
//!
//! POLICY §6's rules are the baseline's: `#[repr(C)]`, `num_cols()` via
//! `size_of`, `Borrow`/`BorrowMut` for `[T]` with their alignment asserts, and
//! nothing anywhere else computing an offset by hand.

use harness::gadgets::canonical_word::WordKind;
use harness::permutation::impl_call_columns;

use crate::columns::PowerWord;
use crate::params::{BYTES, ROUNDS, SPLIT_WORDS};

/// The prime whose integer structure a split word is decomposed against.
///
/// Tip5 is defined over Goldilocks alone (`params.rs`), so this is a constant
/// here rather than an AIR parameter as it is in `monolith::logup`.
pub const KIND: WordKind = WordKind::Goldilocks;

/// Canonicity cells one recomposed input word carries: `(high_is_max, inverse)`.
pub const CANONICAL_CELLS: usize = KIND.canonical_cells();

/// One independent lookup-backed permutation call.
#[repr(C)]
pub struct Tip5LogupCols<T, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize> {
    /// Permutation input.
    pub inputs: [T; WIDTH],
    /// Five complete Tip5 rounds.
    pub rounds: [Round<T, POWER_WORDS, REGISTERS>; ROUNDS],
    /// Permutation output, explicitly bound after the last round.
    pub outputs: [T; WIDTH],
}

/// One Tip5 round's non-linear witnesses.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Round<T, const POWER_WORDS: usize, const REGISTERS: usize> {
    /// The first four split-and-lookup words.
    pub split: [SplitWord<T>; SPLIT_WORDS],
    /// The remaining `WIDTH - 4` seventh-power words, unchanged.
    pub powers: [PowerWord<T, REGISTERS>; POWER_WORDS],
}

/// Witnesses for one split-and-lookup word.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SplitWord<T> {
    /// Little-endian bytes of `mont_R * x`. Their lookup keys range-bind them.
    pub input_bytes: [T; BYTES],
    /// Little-endian substituted bytes. The same keys range-bind them.
    pub output_bytes: [T; BYTES],
    /// `(high_is_max, high_gap_inverse)` for the recomposed input word.
    ///
    /// There is deliberately no output counterpart: the output bytes are
    /// uniquely determined by the queried input bytes, and the reference
    /// reduces the recomposed 64-bit output into the field by definition, so
    /// no second encoding of it is a prover choice (`air::eval_split_word`).
    pub input_canonicity: [T; CANONICAL_CELLS],
}

impl_call_columns!(
    Tip5LogupCols,
    WIDTH: usize,
    POWER_WORDS: usize,
    REGISTERS: usize,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// A split word is one byte tuple, its image, and the canonicity witness —
    /// the lookup is what removed everything else.
    #[test]
    fn the_cell_count_is_the_layout_it_declares() {
        assert_eq!(BYTES, KIND.num_chunks());
        let split_word = BYTES + BYTES + CANONICAL_CELLS;
        assert_eq!(split_word, 18);
        let round_t12_unsplit = SPLIT_WORDS * split_word + 8;
        let round_t16_split = SPLIT_WORDS * split_word + 12 * 2;
        assert_eq!(num_cols::<12, 8, 0>(), 12 + ROUNDS * round_t12_unsplit + 12);
        assert_eq!(num_cols::<16, 12, 1>(), 16 + ROUNDS * round_t16_split + 16);
    }

    /// The whole point of the second arithmetization, as a number.
    #[test]
    fn the_lookup_layout_is_far_narrower_than_the_in_air_one() {
        assert_eq!(crate::columns::num_cols::<12, 8, 0>(), 5_504);
        assert_eq!(num_cols::<12, 8, 0>(), 424);
    }
}
