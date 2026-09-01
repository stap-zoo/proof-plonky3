//! One lookup-backed Monolith call's cells, declared once as a type.
//!
//! The wrapped baseline commits bits, chi products, match flags and a Bar
//! output.  This layout instead commits the input/output chunks queried on the
//! fixed Bars buses and only the field-specific witnesses needed to prove that
//! each recomposed word is canonical.  The round `post` states are unchanged.

use harness::permutation::impl_call_columns;

/// One complete, independently constrained Monolith permutation.
#[repr(C)]
pub struct MonolithLogupCols<
    T,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> {
    /// Permutation input, before the initial Concrete layer.
    pub inputs: [T; WIDTH],
    /// Rounds carrying an addition of round constants.
    pub full_rounds:
        [MonolithLogupRound<T, WIDTH, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>; NUM_FULL_ROUNDS],
    /// Final Bars/Bricks/Concrete round, without round constants.
    pub final_round: MonolithLogupRound<T, WIDTH, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
}

/// Witnesses for one Monolith round.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MonolithLogupRound<
    T,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> {
    /// Chunk decompositions for the first `NUM_BARS` state words.
    pub bars: [MonolithLogupBar<T, NUM_CHUNKS, CANONICAL_CELLS>; NUM_BARS],
    /// State after Bricks, Concrete and the optional round constants.
    pub post: [T; WIDTH],
}

/// One Bar's lookup keys and whole-word canonicity witnesses.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MonolithLogupBar<T, const NUM_CHUNKS: usize, const CANONICAL_CELLS: usize> {
    /// Little-endian input chunks. Their lookup keys range-bind them.
    pub input_chunks: [T; NUM_CHUNKS],
    /// Little-endian output chunks. Their lookup keys range-bind them.
    pub output_chunks: [T; NUM_CHUNKS],
    /// Goldilocks: `(high_is_max, high_gap_inverse)`; M31: `(word_gap_inverse,)`.
    pub input_canonicity: [T; CANONICAL_CELLS],
    /// The same field-specific witness for the recomposed output word.
    pub output_canonicity: [T; CANONICAL_CELLS],
}

impl_call_columns!(
    MonolithLogupCols,
    WIDTH: usize,
    NUM_FULL_ROUNDS: usize,
    NUM_BARS: usize,
    NUM_CHUNKS: usize,
    CANONICAL_CELLS: usize,
);

#[cfg(test)]
mod tests {
    use super::num_cols;

    #[test]
    fn cell_count_is_exactly_the_declared_layout() {
        // inputs + six repetitions of (Bars witnesses + committed post state).
        assert_eq!(
            num_cols::<8, 5, 4, 8, 2>(),
            8 + 6 * (4 * (8 + 8 + 2 + 2) + 8)
        );
        assert_eq!(
            num_cols::<16, 5, 8, 4, 1>(),
            16 + 6 * (8 * (4 + 4 + 1 + 1) + 16)
        );
    }
}
