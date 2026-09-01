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

use harness::permutation::impl_call_columns;

use crate::params::{BYTES, ROUNDS, SPLIT_WORDS};

/// Bits in the quotient of the byte S-box's integer division by 257.
pub const QUOTIENT_BITS: usize = 16;
/// Canonical-prefix flags for Goldilocks: `popcount(p) / 2 = 16`.
pub const MATCH_FLAGS: usize = 16;

/// One independent permutation call.
#[repr(C)]
pub struct Tip5Cols<T, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize> {
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
    /// The remaining `WIDTH - 4` seventh-power words.
    pub powers: [PowerWord<T, REGISTERS>; POWER_WORDS],
}

/// Witnesses for one split-and-lookup word.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SplitWord<T> {
    /// Canonical little-endian bits of `mont_R * x`.
    pub input_bits: [[T; BYTES]; BYTES],
    /// Goldilocks canonical-prefix walk flags.
    pub match_flags: [T; MATCH_FLAGS],
    /// Bits of the eight substituted bytes.
    pub output_bits: [[T; BYTES]; BYTES],
    /// Sixteen-bit quotient for each equation
    /// `(input + 1)^3 - 1 = output + 257 * quotient`.
    pub quotient_bits: [[T; QUOTIENT_BITS]; BYTES],
}

/// One seventh-power word and its optional degree-three register.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PowerWord<T, const REGISTERS: usize> {
    /// Power-map flattening registers.
    pub registers: [T; REGISTERS],
    /// Committed S-box output.
    pub output: T,
}

/// Shared layout invariant.
pub const fn assert_layout(width: usize, power_words: usize, registers: usize) {
    assert!(width == SPLIT_WORDS + power_words);
    assert!(matches!(width, 12 | 16));
    assert!(matches!(registers, 0 | 1));
}

impl_call_columns!(
    Tip5Cols,
    WIDTH: usize,
    POWER_WORDS: usize,
    REGISTERS: usize,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs and outputs plus five repetitions of the complete round layout.
    #[test]
    fn the_cell_count_is_the_layout_it_declares() {
        let split_word = 64 + MATCH_FLAGS + 64 + BYTES * QUOTIENT_BITS;
        let round_t12_unsplit = SPLIT_WORDS * split_word + 8;
        let round_t16_split = SPLIT_WORDS * split_word + 12 * 2;
        assert_eq!(num_cols::<12, 8, 0>(), 12 + ROUNDS * round_t12_unsplit + 12);
        assert_eq!(num_cols::<16, 12, 1>(), 16 + ROUNDS * round_t16_split + 16);
    }
}
