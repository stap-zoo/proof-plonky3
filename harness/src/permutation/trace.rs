//! The trace allocation every written construction repeats.
//!
//! POLICY §6 fixes this too: `MaybeUninit` columns via `align_to_mut`,
//! `par_chunks_mut` over `F::Packing::WIDTH`, and `extra_capacity_bits`
//! forwarded so the prover's LDE extends this allocation instead of
//! reallocating a trace we could have sized correctly.
//!
//! None of that is construction-specific. What *is* construction-specific is
//! the round function — the two closures [`fill_trace`] takes, one filling a
//! single call and one filling a packed batch of `F::Packing::WIDTH` calls at
//! once. Everything around them is here, once, including the alignment
//! assertions.
//!
//! # Why one function serves both generators
//!
//! `VECTOR_LEN` calls side by side in one row is the same byte layout as
//! `VECTOR_LEN` rows of one call, so `align_to_mut` reinterprets the buffer as a
//! flat run of per-call column structs either way. The scalar and vectorized
//! generators differ only in the `ncols` they pass, which is what keeps
//! `vectorized.rs` free of round logic (POLICY §6).

use core::mem::MaybeUninit;

use p3_field::{Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

/// Allocate a trace of `ncols` columns holding one call per `C`, and fill it.
///
/// `C` is the construction's column struct over `MaybeUninit<F>` cells — e.g.
/// `GriffinCols<MaybeUninit<F>, WIDTH, ROUNDS>`. `cells_per_call` is that
/// struct's `num_cols()`, and `state_width` is the permutation's `t`, checked
/// against every input.
///
/// `fill_call` writes one call's cells from one input state. `fill_batch`
/// writes `F::Packing::WIDTH` calls from as many inputs, so that the round
/// function runs once over a packed value rather than once per lane; it is used
/// only when the call count is a multiple of the packing width, and the scalar
/// path is otherwise equivalent.
///
/// # Safety
///
/// The caller must write **every** cell of every `C` handed to `fill_call` and
/// `fill_batch`. This function then declares the buffer initialized, so a field
/// the closures forget is read as uninitialized memory rather than as a wrong
/// number. Keeping the closures a straight walk over the layout struct's own
/// fields — inputs, each round, outputs — is what makes that checkable by
/// reading them.
///
/// # Panics
///
/// If an input is not `state_width` long, or if the calls do not fill whole
/// rows of `ncols` cells.
pub unsafe fn fill_trace<F, C>(
    inputs: &[Vec<F>],
    state_width: usize,
    cells_per_call: usize,
    ncols: usize,
    extra_capacity_bits: usize,
    fill_call: impl Fn(&mut C, &[F]) + Sync,
    fill_batch: impl Fn(&mut [C], &[Vec<F>]) + Sync,
) -> RowMajorMatrix<F>
where
    F: Field,
    C: Send,
{
    assert!(
        inputs.iter().all(|input| input.len() == state_width),
        "every input is one call's state, of length t"
    );
    let cells = inputs.len() * cells_per_call;
    assert!(
        cells.is_multiple_of(ncols),
        "the call count does not fill whole rows"
    );

    let mut values = Vec::with_capacity(cells << extra_capacity_bits);
    let spare: &mut [MaybeUninit<F>] = &mut values.spare_capacity_mut()[..cells];

    // SAFETY: `C` is `#[repr(C)]` over `MaybeUninit<F>` cells and
    // `cells_per_call` of them, so the buffer is a whole number of `C`s; the
    // assertions below are what check that claim rather than trusting it.
    let (prefix, calls, suffix) = unsafe { spare.align_to_mut::<C>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(calls.len(), inputs.len());

    let packing_width = F::Packing::WIDTH;
    if packing_width > 1 && inputs.len().is_multiple_of(packing_width) {
        calls
            .par_chunks_mut(packing_width)
            .zip(inputs.par_chunks(packing_width))
            .for_each(|(calls, inputs)| fill_batch(calls, inputs));
    } else {
        calls
            .par_iter_mut()
            .zip(inputs.par_iter())
            .for_each(|(call, input)| fill_call(call, input));
    }

    // SAFETY: the caller's contract — every cell of every call was written.
    unsafe { values.set_len(cells) };
    RowMajorMatrix::new(values, ncols)
}

/// Assert the row count a scalar generator was handed.
///
/// Tables are always full: no padding, no selector (POLICY §6), so a caller
/// hands over exactly `2^k` calls and this is an assertion rather than a
/// rounding step.
///
/// # Panics
///
/// If `num_calls` is not a power of two.
pub fn assert_full_table(num_calls: usize) {
    assert!(
        num_calls.is_power_of_two(),
        "tables are full (POLICY §6): the call count is a power of two"
    );
}

/// Assert the row count a vectorized generator was handed.
///
/// # Panics
///
/// If `num_calls` is not `vector_len` times a power of two.
pub fn assert_full_vectorized_table(num_calls: usize, vector_len: usize) {
    assert!(
        num_calls.is_multiple_of(vector_len) && (num_calls / vector_len).is_power_of_two(),
        "inputs must fill VECTOR_LEN lanes over a power-of-two number of rows"
    );
}

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;
    use core::mem::MaybeUninit;

    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;

    use super::{assert_full_table, assert_full_vectorized_table, fill_trace};

    /// A two-cell "call": the input word and its double.
    #[repr(C)]
    struct ToyCols<T> {
        input: T,
        doubled: T,
    }

    crate::impl_call_columns!(ToyCols);

    fn fill(call: &mut ToyCols<MaybeUninit<Mersenne31>>, input: &[Mersenne31]) {
        call.input.write(input[0]);
        call.doubled.write(input[0].double());
    }

    fn fill_batch(calls: &mut [ToyCols<MaybeUninit<Mersenne31>>], inputs: &[Vec<Mersenne31>]) {
        for (call, input) in calls.iter_mut().zip(inputs) {
            fill(call, input);
        }
    }

    fn inputs(n: usize) -> Vec<Vec<Mersenne31>> {
        (0..n)
            .map(|i| vec![Mersenne31::from_u32(i as u32 + 1)])
            .collect()
    }

    /// One call per row and several calls per row are the same buffer read with
    /// two different `ncols`, which is the identity `vectorized.rs` depends on.
    #[test]
    fn the_scalar_and_vectorized_layouts_are_the_same_bytes() {
        let inputs = inputs(8);
        let scalar = unsafe { fill_trace(&inputs, 1, num_cols(), 2, 0, fill, fill_batch) };
        let vectorized = unsafe { fill_trace(&inputs, 1, num_cols(), 8, 0, fill, fill_batch) };
        assert_eq!(scalar.height(), 8);
        assert_eq!(vectorized.height(), 2);
        assert_eq!(scalar.values, vectorized.values);

        let first: &ToyCols<Mersenne31> = scalar.values[..2].borrow();
        assert_eq!(first.input, Mersenne31::ONE);
        assert_eq!(first.doubled, Mersenne31::TWO);
    }

    /// `extra_capacity_bits` must reach the allocation, or the prover's LDE
    /// reallocates a trace this function could have sized correctly.
    #[test]
    fn extra_capacity_is_reserved_and_not_filled() {
        let inputs = inputs(4);
        let trace = unsafe { fill_trace(&inputs, 1, num_cols(), 2, 2, fill, fill_batch) };
        assert_eq!(trace.values.len(), 8);
        assert!(trace.values.capacity() >= 32);
    }

    #[test]
    #[should_panic(expected = "one call's state")]
    fn an_input_of_the_wrong_width_is_rejected() {
        let inputs = vec![vec![Mersenne31::ONE, Mersenne31::TWO]];
        let _ = unsafe { fill_trace(&inputs, 1, num_cols(), 2, 0, fill, fill_batch) };
    }

    #[test]
    fn full_table_assertions_accept_exactly_full_tables() {
        assert_full_table(8);
        assert_full_vectorized_table(16, 8);
        assert!(std::panic::catch_unwind(|| assert_full_table(6)).is_err());
        assert!(std::panic::catch_unwind(|| assert_full_vectorized_table(12, 8)).is_err());
    }
}
