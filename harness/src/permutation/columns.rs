//! The column-layout boilerplate every written construction repeats.
//!
//! POLICY §6 fixes the shape: one `#[repr(C)]` struct for one call's cells,
//! `num_cols()` via `size_of`, and `Borrow`/`BorrowMut` for `[T]` so that
//! nothing computes an offset by hand. What that costs is roughly twenty lines
//! of `unsafe` per construction, identical every time apart from the struct's
//! name and its const parameters — and the three `debug_assert`s inside it are
//! the only thing standing between a layout edit and silent misalignment.
//!
//! [`impl_call_columns`] writes them, so that a construction declares the
//! layout and nothing else.

/// Implement `num_cols`, `Borrow<Cols<T, ..>>` and `BorrowMut<Cols<T, ..>>` for
/// `[T]`.
///
/// The struct must be `#[repr(C)]` and its fields must all be `T`, arrays of
/// `T`, or `#[repr(C)]` structs that are themselves only `T`. That is what makes
/// `size_of::<Cols<u8, ..>>()` the cell count and what makes the reinterpretation
/// sound.
///
/// ```ignore
/// #[repr(C)]
/// pub struct GriffinCols<T, const WIDTH: usize, const ROUNDS: usize> { .. }
///
/// impl_call_columns!(GriffinCols, WIDTH: usize, ROUNDS: usize);
/// ```
///
/// The generated `num_cols` is a free function in the calling module, generic
/// over the same const parameters in the same order:
/// `num_cols::<WIDTH, ROUNDS>()`.
///
/// The `prefix.is_empty()` / `suffix.is_empty()` assertions POLICY §6 requires
/// are part of the expansion, so they cannot be dropped in one construction and
/// kept in another. They are `debug_assert`s, matching upstream's own column
/// borrows: `cargo test` builds with `debug_assertions` on, which is where a
/// misalignment shows up.
#[macro_export]
macro_rules! impl_call_columns {
    ($cols:ident $(, $param:ident : $ty:ty)* $(,)?) => {
        /// Number of cells in one call, from the layout type itself.
        ///
        /// `size_of` over the `u8` instantiation: one byte per cell, so the
        /// byte size *is* the cell count, and no second definition of the
        /// layout can drift from the struct.
        #[must_use]
        pub const fn num_cols<$(const $param: $ty),*>() -> usize {
            ::core::mem::size_of::<$cols<u8 $(, $param)*>>()
        }

        impl<T $(, const $param: $ty)*> ::core::borrow::Borrow<$cols<T $(, $param)*>> for [T] {
            fn borrow(&self) -> &$cols<T $(, $param)*> {
                debug_assert_eq!(self.len(), num_cols::<$($param),*>());
                let (prefix, structs, suffix) =
                    unsafe { self.align_to::<$cols<T $(, $param)*>>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert!(suffix.is_empty(), "Alignment should match");
                debug_assert_eq!(structs.len(), 1);
                &structs[0]
            }
        }

        impl<T $(, const $param: $ty)*> ::core::borrow::BorrowMut<$cols<T $(, $param)*>> for [T] {
            fn borrow_mut(&mut self) -> &mut $cols<T $(, $param)*> {
                debug_assert_eq!(self.len(), num_cols::<$($param),*>());
                let (prefix, structs, suffix) =
                    unsafe { self.align_to_mut::<$cols<T $(, $param)*>>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert!(suffix.is_empty(), "Alignment should match");
                debug_assert_eq!(structs.len(), 1);
                &mut structs[0]
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use core::borrow::{Borrow, BorrowMut};

    #[repr(C)]
    struct ToyCols<T, const WIDTH: usize, const ROUNDS: usize> {
        inputs: [T; WIDTH],
        rounds: [Round<T, WIDTH>; ROUNDS],
        outputs: [T; WIDTH],
    }

    #[repr(C)]
    struct Round<T, const WIDTH: usize> {
        post: [T; WIDTH],
    }

    impl_call_columns!(ToyCols, WIDTH: usize, ROUNDS: usize);

    #[test]
    fn the_cell_count_is_the_declared_layout() {
        assert_eq!(num_cols::<4, 3>(), 4 + 3 * 4 + 4);
        assert_eq!(num_cols::<0, 0>(), 0);
    }

    /// The borrow is a reinterpretation, not a copy: the struct's fields must
    /// land on the cells the layout order says they do, which is the property
    /// `generation.rs` and `air.rs` both rely on to name the same cell.
    #[test]
    fn borrowing_maps_fields_onto_cells_in_declaration_order() {
        let mut cells: Vec<u32> = (0..12).collect();
        {
            let cols: &ToyCols<u32, 2, 4> = cells.as_slice().borrow();
            assert_eq!(cols.inputs, [0, 1]);
            assert_eq!(cols.rounds[0].post, [2, 3]);
            assert_eq!(cols.rounds[3].post, [8, 9]);
            assert_eq!(cols.outputs, [10, 11]);
        }
        let cols: &mut ToyCols<u32, 2, 4> = cells.as_mut_slice().borrow_mut();
        cols.rounds[2].post = [100, 101];
        assert_eq!(cells[6..8], [100, 101]);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn a_slice_of_the_wrong_length_is_rejected() {
        let cells: Vec<u32> = (0..11).collect();
        let _: &ToyCols<u32, 2, 4> = cells.as_slice().borrow();
    }
}
