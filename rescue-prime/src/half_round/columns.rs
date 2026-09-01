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
//! # What a half-round commits, and why the inverse S-box is free
//!
//! A half-round is `y = S(x)`, then `x' = M*y + c`, where `S` is `x^alpha` on
//! the forward half-rounds and `x^(1/alpha)` on the inverse ones. Committing `y`
//! — the non-linear layer's *output* — is what makes the expensive direction
//! cost nothing:
//!
//! ```text
//! forward  half-round:   y_i == x_i^alpha     a power map on the previous cells
//! inverse  half-round:   y_i^alpha == x_i     the same power map, mirrored
//! ```
//!
//! and `x = M*y_prev + c` is affine in the previous half-round's cells in both
//! cases. So neither direction needs a witness cell beyond the state itself, and
//! both cost one constraint of degree `alpha` per word. On the inverse
//! half-rounds the committed state cell *is* the alpha-th root, and
//! `harness::gadgets::inverse_power_map` is what pins it in place.
//!
//! Committing once per round instead is `crate::full_round`: a separate module
//! rather than a flag here, because it commits different cells.
//!
//! # Cost
//!
//! `WIDTH * (2 + HALF_ROUNDS * (1 + REGISTERS))` cells per call: the input and
//! output boundaries, plus one state and `REGISTERS` register cells per word per
//! half-round. At `t = 24` and `R = 8` that is 432 unsplit and 816 with one
//! register — the register variant is not cheap here, because *every* word runs
//! an S-box in *every* half-round. Whether the degree it buys pays for that is
//! the measurement, not a prediction.

use harness::permutation::impl_call_columns;

/// One independent permutation call.
#[repr(C)]
pub struct RescuePrimeCols<T, const WIDTH: usize, const REGISTERS: usize, const HALF_ROUNDS: usize>
{
    /// Permutation input. Rescue-Prime has no leading layer, so this is the
    /// first forward S-box's argument.
    pub inputs: [T; WIDTH],
    /// One entry per half-round: forward at even indices, inverse at odd ones.
    pub half_rounds: [HalfRound<T, WIDTH, REGISTERS>; HALF_ROUNDS],
    /// Permutation output, after the last half-round's linear layer and
    /// constant row. Rescue-Prime has no trailing layer either.
    pub outputs: [T; WIDTH],
}

/// Cells for one half-round.
#[repr(C)]
pub struct HalfRound<T, const WIDTH: usize, const REGISTERS: usize> {
    /// Registers flattening each word's power map — one block per word, empty
    /// in the unsplit variant.
    pub powers: [Power<T, REGISTERS>; WIDTH],
    /// The non-linear layer's output `y`, before the linear layer.
    pub post: [T; WIDTH],
}

/// One power map's flattening cells.
#[repr(C)]
pub struct Power<T, const REGISTERS: usize>(pub [T; REGISTERS]);

/// Layout and variant invariant.
///
/// The register variants are the power-map gadget's, with one difference from
/// Griffin's list: **alpha 3 is offered no register here and needs none**, since
/// nothing in this round function sets a floor above the S-box itself — the
/// linear layer is affine and the constants are additive, so an alpha-3
/// instance is a degree-3 AIR outright.
pub const fn assert_layout(width: usize, alpha: u64, registers: usize, half_rounds: usize) {
    assert!(width > 0, "a permutation needs a state");
    assert!(
        half_rounds.is_multiple_of(2),
        "a Rescue-Prime round is two half-rounds"
    );
    match (alpha, registers) {
        (3 | 5 | 7, 0) | (5 | 7, 1) => {}
        _ => panic!("Rescue-Prime supports unsplit alpha 3/5/7 or one register for alpha 5/7"),
    }
}

impl_call_columns!(
    RescuePrimeCols,
    WIDTH: usize,
    REGISTERS: usize,
    HALF_ROUNDS: usize,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The width formula, spelled out, against `size_of`. Inputs, then
    /// `HALF_ROUNDS` half-rounds of `WIDTH` register blocks and a state, then
    /// outputs.
    #[test]
    fn the_cell_count_is_the_layout_it_declares() {
        assert_eq!(num_cols::<8, 0, 16>(), 8 + 16 * 8 + 8);
        assert_eq!(num_cols::<8, 1, 16>(), 8 + 16 * (8 + 8) + 8);
        assert_eq!(num_cols::<24, 0, 16>(), 24 + 16 * 24 + 24);
        assert_eq!(num_cols::<24, 1, 16>(), 24 + 16 * (24 + 24) + 24);
    }

    /// An empty register array must cost nothing, or the unsplit variant pays
    /// for cells it does not have and `align_to` stops landing on row
    /// boundaries.
    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<Power<u8, 0>>(), 0);
        assert_eq!(core::mem::size_of::<HalfRound<u8, 8, 0>>(), 8);
    }
}
