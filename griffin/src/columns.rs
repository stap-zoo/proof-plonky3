//! One call's cells, declared once, as a type.
//!
//! POLICY §6: a `#[repr(C)]` struct for **one call's** cells, `num_cols()` via
//! `size_of`, `Borrow`/`BorrowMut` for `[T]`. Nothing anywhere else computes an
//! offset by hand, and the alignment assertions that stand between a layout edit
//! and silent misalignment come with `harness::permutation::impl_call_columns!`.
//!
//! First of the four files (POLICY §2, step 4) — `air.rs` and `generation.rs`
//! both read the layout off this type, which is what stops them drifting apart
//! silently.
//!
//! # What one round commits, and why it is the non-linear layer's output
//!
//! A round is `y = NL(x)`, then `x' = M*y + rc`. Committing `y` — rather than
//! the round's input, or its post-linear-layer state — is what makes every value
//! the round function *multiplies* a committed cell:
//!
//! ```text
//! y_0                       committed, pinned by  y_0^alpha == x_0
//! y_1                       committed, pinned by  y_1 == x_1^alpha
//! y_i = x_i * G_i(L_i)      committed, pinned by that equation
//! ```
//!
//! where `x = M*y_prev + rc` is affine in the *previous* round's cells and
//! `L_i = (i-1)*y_0 + y_1 + x_{i-1}` is affine in this round's. So `G_i(L_i)` is
//! degree two, the Horst product is degree three, and the whole layer costs `t`
//! cells and `t` constraints per round.
//!
//! The alternative — committing the round input `x` as well — would need two
//! extra cells per round for `y_0` and `y_1`, which have to be committed
//! whatever else is: `y_0` because no AIR can evaluate an inverse power, and
//! `y_1` because leaving it as a degree-`alpha` expression inside `L_i` would
//! make the Horst product degree `2*alpha + 1`, fifteen at alpha 7.
//!
//! Both S-box pins go through the power-map gadget, so `REGISTERS` buys the same
//! trade here as everywhere: one committed cell each, and degree three instead
//! of degree `alpha`.

use harness::permutation::impl_call_columns;

/// One independent permutation call.
#[repr(C)]
pub struct GriffinCols<T, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize> {
    /// Permutation input, before the leading linear layer.
    pub inputs: [T; WIDTH],
    /// One entry per round.
    pub rounds: [Round<T, WIDTH, REGISTERS>; ROUNDS],
    /// Permutation output, after the final linear layer.
    pub outputs: [T; WIDTH],
}

/// Cells for one round.
#[repr(C)]
pub struct Round<T, const WIDTH: usize, const REGISTERS: usize> {
    /// Registers flattening the two S-boxes: `powers[0]` pins `y_0`'s power map
    /// and `powers[1]` pins `y_1`'s. Empty in the unsplit variant.
    pub powers: [Power<T, REGISTERS>; 2],
    /// The non-linear layer's output `y`.
    pub post: [T; WIDTH],
}

/// One power map's flattening cells.
#[repr(C)]
pub struct Power<T, const REGISTERS: usize>(pub [T; REGISTERS]);

/// Layout and variant invariant.
///
/// The register variants are the power-map gadget's: alpha 3 has none, because a
/// register would raise the width and buy no degree — the Horst layer already
/// sets a floor of three.
pub const fn assert_layout(width: usize, alpha: u64, registers: usize) {
    assert!(width >= 4, "Griffin needs at least two Horst words");
    assert!(
        width.is_multiple_of(4),
        "Griffin's block-circulant matrix needs t a multiple of 4"
    );
    match (alpha, registers) {
        (3 | 5 | 7, 0) | (5 | 7, 1) => {}
        _ => panic!("Griffin supports unsplit alpha 3/5/7 or one register for alpha 5/7"),
    }
}

impl_call_columns!(GriffinCols, WIDTH: usize, REGISTERS: usize, ROUNDS: usize);

#[cfg(test)]
mod tests {
    use super::*;

    /// The width formula, spelled out, against `size_of`. Inputs, then `ROUNDS`
    /// rounds of two register blocks and a state, then outputs.
    #[test]
    fn the_cell_count_is_the_layout_it_declares() {
        assert_eq!(num_cols::<16, 0, 15>(), 16 + 15 * 16 + 16);
        assert_eq!(num_cols::<16, 1, 15>(), 16 + 15 * (2 + 16) + 16);
        assert_eq!(num_cols::<24, 1, 15>(), 24 + 15 * (2 + 24) + 24);
        assert_eq!(num_cols::<8, 0, 8>(), 8 + 8 * 8 + 8);
    }

    /// An empty register array must cost nothing, or the unsplit variant pays
    /// for cells it does not have and `align_to` stops landing on row
    /// boundaries.
    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<Power<u8, 0>>(), 0);
        assert_eq!(core::mem::size_of::<Round<u8, 8, 0>>(), 8);
    }
}
