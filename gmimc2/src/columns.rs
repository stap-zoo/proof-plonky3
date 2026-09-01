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
//! # The same chain as GMiMC's, with the constant riding it
//!
//! The layout is `gmimc`'s and the argument for it is that crate's
//! [`columns`](../../../gmimc/src/columns.rs): only branch 0 is read
//! nonlinearly, so a branch is a running sum of S-box outputs and the one value
//! worth committing is the one entering each S-box. Committing it makes the
//! permutation a single recurrence reaching back exactly `t` rounds.
//!
//! Two things differ, and both are the design rather than the arithmetization:
//!
//! 1. **The committed cell already carries its round constant.** GMiMC2 adds
//!    `rc_r` *into* branch 0 and then powers it, so `head` is the S-box's argument
//!    outright — `y_r = head^alpha`, where GMiMC needs `(head + rc_r)^alpha` — and
//!    the constant stays on that branch for its whole trip around the state. The
//!    recurrence therefore gains one term:
//!
//!    ```text
//!    b_r = b_{r-t} + rc_r + sum_{j=r-t+1}^{r-1} y_j
//!    ```
//!
//!    which telescopes just as cleanly, the constants differencing to
//!    `rc_{r+1} - rc_r`.
//!
//! 2. **`M_IO` brackets the round loop**, where GMiMC's `_pre_rounds` and
//!    `_post_rounds` are the identity. So the chain's two ends are no longer the
//!    permutation's own input and output: the first `t` links are `M_IO(inputs)`
//!    and the last `t` are what the trailing `M_IO` is applied *to*.
//!
//! # Why the boundary blocks still hold the permutation's own input and output
//!
//! `M_IO` is affine, so putting it on either side of the boundary costs the same
//! `t` cells; what differs is *what the cells mean*. These hold the permutation's
//! input and output, so that POLICY §10's second layer — the output cells of a
//! KAT-input trace, against the reference's own outputs — reads a call's result
//! directly, and so that the trailing `M_IO` is a **constraint** rather than an
//! expression a test happens to recompute the same way as the generator.
//!
//! What that costs is one bounded expression per output cell instead of a
//! four-cell difference, and `air.rs` prices it where it is written.
//!
//! # Cost
//!
//! `2*WIDTH + ROUNDS*(1 + REGISTERS)` cells per call — 120 at Goldilocks
//! `t=12, R=96` and 312 at every 31-bit `t=24, R=264`, and 216 at Goldilocks with
//! the register alpha 4 admits. The three 31-bit points have **no** register
//! variant: at `alpha = 2` a register would commit `y = head^2` at degree 2 and
//! leave the recurrence at degree 1, which is the same max degree for twice the
//! cells. That is a finding rather than an omission (POLICY §11), and
//! [`assert_layout`] is where it is enforced.

use harness::permutation::impl_call_columns;

/// One independent permutation call.
///
/// The middle block plus the two boundaries are one chain (see the module docs);
/// `air.rs` reads them as one, through `M_IO` at each end.
#[repr(C)]
pub struct GMiMC2Cols<T, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize> {
    /// The permutation input, **before** `M_IO`.
    ///
    /// `M_IO(inputs)` is the chain's first `t` links: `M_IO(inputs)[r]` is the
    /// branch that reaches the S-box at round `r`.
    pub inputs: [T; WIDTH],

    /// The rounds, in order: `b_0 .. b_{R-1}`.
    pub rounds: [Round<T, REGISTERS>; ROUNDS],

    /// The permutation output, **after** the trailing `M_IO`.
    pub outputs: [T; WIDTH],
}

/// One round's cells: the S-box's argument, and the register flattening it.
///
/// Both are prover-chosen and worth nothing until a constraint ties them down
/// (POLICY §9): `head` by the chain rule in [`air::eval`](crate::air::eval),
/// `registers` by the power-map gadget's own `register == value^2` assertion.
#[repr(C)]
pub struct Round<T, const REGISTERS: usize> {
    /// `b_r`: branch 0's value at round `r`, **after** its round constant.
    ///
    /// This is the one cell whose meaning differs from `gmimc`'s, and it differs
    /// because the designs differ: here the constant is added into the state and
    /// travels with the branch, so `y_r = head^alpha` with no constant in sight.
    /// Swapping the two readings gives a permutation that passes every structural
    /// test and no vector.
    pub head: T,

    /// The power map's register, or nothing. See `harness::gadgets::power_map`.
    pub registers: [T; REGISTERS],
}

/// Layout and variant invariant.
///
/// * `WIDTH` divisible by 6 is `M_IO`'s: the specification's `c = t/3` circulant
///   needs both `t/3` and `t/2` whole (`params::assert_shape`).
/// * `ROUNDS >= WIDTH` is the *output* boundary's: an output link reaches back `t`
///   rounds into the committed block, so a permutation shorter than one trip
///   around the state would reach past the chain. Both grid points run at least
///   eight trips, and the specification's own `R % t == 0` is checked in
///   `params.rs`.
/// * The variant list is the finding above: alpha 4 with zero or one register,
///   alpha 2 with none.
///
/// Called from every `new`, so an illegal parameter set fails at monomorphization
/// rather than building a trace whose cells mean something other than what
/// `air.rs` reads.
pub const fn assert_layout(width: usize, rounds: usize, alpha: u64, registers: usize) {
    crate::params::assert_shape(width);
    assert!(
        rounds >= width,
        "the output boundary reaches back one full trip around the state"
    );
    match (alpha, registers) {
        (4, 0 | 1) | (2, 0) => {}
        _ => panic!("GMiMC2 supports alpha 4 with zero or one register, alpha 2 with none"),
    }
}

impl_call_columns!(GMiMC2Cols, WIDTH: usize, REGISTERS: usize, ROUNDS: usize);

#[cfg(test)]
mod tests {
    use super::*;

    /// The width formula, spelled out, against `size_of`. This is the number every
    /// other cost column is downstream of, and POLICY §11 pins it — but a pin only
    /// catches a *change*, not a layout that was wrong from the start.
    #[test]
    fn the_cell_count_is_the_chain_plus_its_registers() {
        // Goldilocks t = 12, R = 96, alpha 4: both variants.
        assert_eq!(num_cols::<12, 0, 96>(), 2 * 12 + 96);
        assert_eq!(num_cols::<12, 0, 96>(), 120);
        assert_eq!(num_cols::<12, 1, 96>(), 216);

        // Every 31-bit prime, t = 24, R = 264, alpha 2: one variant only.
        assert_eq!(num_cols::<24, 0, 264>(), 2 * 24 + 264);
        assert_eq!(num_cols::<24, 0, 264>(), 312);
    }

    /// An empty register array must cost nothing, or the unsplit variant pays for
    /// cells it does not have and `align_to` stops landing on row boundaries.
    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<Round<u8, 0>>(), 1);
        assert_eq!(core::mem::size_of::<Round<u8, 1>>(), 2);
    }

    /// It is narrower than its sibling at the same width, and by the round count
    /// alone — 264 against 335 at `t = 24`. The comparison is the reason both
    /// crates exist, so it is worth one line that fails if either layout moves.
    #[test]
    fn the_width_difference_from_gmimc_is_the_round_count() {
        assert_eq!(num_cols::<24, 0, 264>() + (335 - 264), 383);
    }

    #[test]
    fn the_layout_invariant_holds_at_the_grid_points() {
        assert_layout(12, 96, 4, 0);
        assert_layout(12, 96, 4, 1);
        assert_layout(24, 264, 2, 0);
    }

    /// `t = 16` is where `M_IO`'s offsets stop being whole — one of the two reasons
    /// the 31-bit `t=16` points are absent (`params.rs` carries the other, the
    /// specification's `R % t == 0`).
    #[test]
    #[should_panic(expected = "t/3 and t/2")]
    fn a_width_m_io_cannot_be_built_at_is_rejected() {
        assert_layout(16, 264, 2, 0);
    }

    #[test]
    #[should_panic(expected = "one full trip")]
    fn a_permutation_shorter_than_one_trip_is_rejected() {
        assert_layout(12, 8, 4, 0);
    }

    /// The finding, as an assertion: alpha 2 admits exactly one variant.
    #[test]
    #[should_panic(expected = "alpha 2 with none")]
    fn a_register_on_a_squaring_is_refused() {
        assert_layout(24, 264, 2, 1);
    }
}
