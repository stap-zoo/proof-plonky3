//! One Anemoi call's committed cells.
//!
//! Every round commits its output state. The split variant additionally commits
//! one power-map register per Flystel: `(y-v)^2` for alpha 5 and `(y-v)^3` for
//! alpha 7. The register turns the closed Flystel's degree into three.

use harness::permutation::impl_call_columns;

/// One independent permutation call.
#[repr(C)]
pub struct AnemoiCols<
    T,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
> {
    /// Permutation input.
    pub inputs: [T; WIDTH],
    /// Round witnesses and committed states.
    pub rounds: [Round<T, WIDTH, COLUMNS, REGISTERS>; ROUNDS],
    /// Permutation output after the final linear layer.
    pub outputs: [T; WIDTH],
}

/// Cells for one round.
#[repr(C)]
pub struct Round<T, const WIDTH: usize, const COLUMNS: usize, const REGISTERS: usize> {
    /// One optional power-map register per Flystel column.
    pub powers: [Power<T, REGISTERS>; COLUMNS],
    /// Output of the open Flystel layer.
    pub post: [T; WIDTH],
}

/// A Flystel power-map's flattening cells.
#[repr(C)]
pub struct Power<T, const REGISTERS: usize>(pub [T; REGISTERS]);

/// Layout and variant invariant.
pub const fn assert_layout(width: usize, columns: usize, alpha: u64, registers: usize) {
    assert!(width == 2 * columns, "WIDTH must equal 2 * COLUMNS");
    match (alpha, registers) {
        (3 | 5 | 7, 0) | (5 | 7, 1) => {}
        _ => panic!("Anemoi supports unsplit alpha 3/5/7 or one register for alpha 5/7"),
    }
}

impl_call_columns!(
    AnemoiCols,
    WIDTH: usize,
    COLUMNS: usize,
    REGISTERS: usize,
    ROUNDS: usize,
);
