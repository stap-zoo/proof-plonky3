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
//! # `P3_BLOCKS`
//!
//! A layout parameter retained so the generic code can reject a malformed
//! coordinate table without silently constraining a different permutation.
//!
//! | `P3_BLOCKS` | P3 register basis | cells per triple | needs |
//! |---|---|---|---|
//! | 1 | `x^{(alpha-1)/2}` in `F_p[X]/(f)` | 3 | the exported table to *be* that power map |
//! | 2 | the six quadratic monomials | 6 | nothing |
//!
//! Both reach degree three. Every current instance uses `P3_BLOCKS = 1` after
//! the Mersenne-31 export repair; [`crate::air::XHashAir::from_params`] still
//! refuses that basis if a coordinate table does not match its quotient.

use harness::permutation::impl_call_columns;

/// One independent XHash call.
#[repr(C)]
pub struct XHashCols<
    T,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
> {
    /// Permutation input, used by the first F step.
    pub inputs: [T; WIDTH],
    /// Complete F/B/P3 groups.
    pub cycles: [Cycle<T, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS>; CYCLES],
    /// Final MDS-plus-constant boundary.
    pub outputs: [T; WIDTH],
}

/// Cells committed by one F/B/P3 group.
#[repr(C)]
pub struct Cycle<
    T,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
> {
    /// Optional flattening cells for the full F-step base-field power maps.
    pub forward_powers: [Power<T, REGISTERS>; WIDTH],
    /// Optional flattening cells pinning the active B-step root witnesses.
    pub backward_powers: [Power<T, REGISTERS>; ACTIVE],
    /// B-step outputs; active words are witnessed `alpha`-th roots.
    pub backward: [T; WIDTH],
    /// Optional P3 registers, `P3_BLOCKS` blocks beside each input coordinate.
    ///
    /// Per coordinate rather than per triple so that the array length stays free
    /// of const-generic arithmetic; a triple therefore carries
    /// `3 * P3_BLOCKS * REGISTERS` cells.
    pub extension_powers: [ExtensionPower<T, REGISTERS, P3_BLOCKS>; WIDTH],
    /// P3 output and next group's state.
    pub post: [T; WIDTH],
}

/// One base-field power map's optional flattening cells.
#[repr(C)]
pub struct Power<T, const REGISTERS: usize>(pub [T; REGISTERS]);

/// The optional P3 registers assigned to one input coordinate.
#[repr(C)]
pub struct ExtensionPower<T, const REGISTERS: usize, const P3_BLOCKS: usize>(
    pub [[T; REGISTERS]; P3_BLOCKS],
);

/// Assert every const-generic layout choice before a trace is allocated.
pub const fn assert_layout(
    width: usize,
    active: usize,
    alpha: u64,
    registers: usize,
    p3_blocks: usize,
    cycles: usize,
) {
    assert!(width > 0 && width.is_multiple_of(3));
    assert!(cycles == 3, "XHash has three F/B/P3 groups");
    assert!(
        active == width || active * 3 == width * 2,
        "the full or one-in-three-skipped variant is required"
    );
    assert!(
        matches!(alpha, 5 | 7),
        "XHash supports the fifth and seventh power maps"
    );
    assert!(
        registers == 0 || registers == 1,
        "XHash supports the direct or one-register power map"
    );
    // The six-quadratic basis factors a degree-five monomial as `2 + 2 + 1`;
    // degree seven would need ten cubics and no instance asks for it.
    assert!(
        p3_blocks == 1 || (p3_blocks == 2 && alpha == 5),
        "the modulus-free P3 basis is defined for alpha five only"
    );
}

impl_call_columns!(
    XHashCols,
    WIDTH: usize,
    ACTIVE: usize,
    REGISTERS: usize,
    P3_BLOCKS: usize,
    CYCLES: usize,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The assertion guarding the implementation's mechanical layout: every
    /// field in every cycle appears in this formula, so adding, removing or
    /// silently padding a block changes the test before it changes a headline
    /// number.
    ///
    /// One formula covers both families: F contributes `width` power-map
    /// register blocks, while B contributes only `active` in aggressive B'.
    #[test]
    fn the_cell_count_is_the_declared_layout() {
        let expected = |width: usize, active: usize, registers: usize, blocks: usize| {
            2 * width + 3 * (2 * width + registers * (width + active + width * blocks))
        };

        // Goldilocks t = 12, the structured P3 basis.
        assert_eq!(num_cols::<12, 8, 0, 1, 3>(), expected(12, 8, 0, 1));
        assert_eq!(num_cols::<12, 8, 1, 1, 3>(), expected(12, 8, 1, 1));
        assert_eq!(num_cols::<12, 12, 0, 1, 3>(), expected(12, 12, 0, 1));
        assert_eq!(num_cols::<12, 12, 1, 1, 3>(), expected(12, 12, 1, 1));

        // Mersenne-31 t = 24, also the structured P3 basis after the repair.
        assert_eq!(num_cols::<24, 16, 0, 1, 3>(), expected(24, 16, 0, 1));
        assert_eq!(num_cols::<24, 16, 1, 1, 3>(), expected(24, 16, 1, 1));
        assert_eq!(num_cols::<24, 24, 0, 1, 3>(), expected(24, 24, 0, 1));
        assert_eq!(num_cols::<24, 24, 1, 1, 3>(), expected(24, 24, 1, 1));
    }

    /// What the Mersenne-31 repair saves, as a checked difference between the
    /// legacy fallback and the now-active structured basis.
    #[test]
    fn the_structured_p3_basis_is_narrower_at_the_same_degree() {
        assert_eq!(num_cols::<24, 16, 1, 2, 3>(), 456);
        assert_eq!(num_cols::<24, 16, 1, 1, 3>(), 384);
        assert_eq!(num_cols::<24, 24, 1, 2, 3>(), 480);
        assert_eq!(num_cols::<24, 24, 1, 1, 3>(), 408);
    }

    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<Power<u8, 0>>(), 0);
        assert_eq!(core::mem::size_of::<ExtensionPower<u8, 0, 1>>(), 0);
        assert_eq!(core::mem::size_of::<ExtensionPower<u8, 0, 2>>(), 0);
        assert_eq!(core::mem::size_of::<Cycle<u8, 12, 8, 0, 1>>(), 24);
        assert_eq!(core::mem::size_of::<Cycle<u8, 24, 16, 0, 1>>(), 48);
    }
}
