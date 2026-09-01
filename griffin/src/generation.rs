//! The witness producer — never evidence.
//!
//! POLICY §6: a free `generate_trace_rows(inputs, params, extra_capacity_bits)`
//! and its vectorized twin, over `MaybeUninit` columns, with
//! `extra_capacity_bits` forwarded so the prover's LDE happens in place. The
//! allocation, the alignment assertions and the packing-width dispatch are
//! `harness::permutation::trace::fill_trace`'s; what is Griffin's is the round function
//! below.
//!
//! POLICY §9: nothing written here is evidence of anything. Every cell it writes
//! is a cell a malicious prover could have written differently, so each one
//! needs a constraint in `air.rs` naming it — the table in that file's docs is
//! that list, and `tests/air.rs`'s per-cell negative test is what checks it.
//!
//! # Kept parallel to `air.rs`
//!
//! Same order, same names, same per-round split: `generate_round` against
//! `eval_round`, the leading `mds_multiply` at the same place, the round
//! constants added by the caller in both. The one structural difference is that
//! the arithmetic here is generic over the ring rather than over the builder,
//! which is what lets the same code run over `F` and over `F::Packing` — the
//! scalar path and the SIMD path are the same round function, and only the cell
//! *writing* differs.

use core::mem::MaybeUninit;

use harness::gadgets::inverse_power_map::generate_inverse_power_map;
use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{Algebra, Field, PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::columns::{GriffinCols, Round, assert_layout, num_cols};
use crate::native::horst_layer;
use crate::params::GriffinParams;

/// One row per call.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate Griffin trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &GriffinParams<F, WIDTH, ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, REGISTERS, ROUNDS>();
    fill_calls::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(inputs, params, ncols, extra_capacity_bits)
}

/// `VECTOR_LEN` calls per row.
///
/// # Panics
///
/// If `inputs.len()` is not `VECTOR_LEN` times a power of two, or an input is
/// not `WIDTH` long.
#[instrument(name = "generate vectorized Griffin trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    params: &GriffinParams<F, WIDTH, ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, REGISTERS, ROUNDS>() * VECTOR_LEN;
    fill_calls::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(inputs, params, ncols, extra_capacity_bits)
}

fn fill_calls<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &GriffinParams<F, WIDTH, ROUNDS>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, ALPHA, REGISTERS);
    assert_eq!(params.alpha, ALPHA, "generator alpha must match parameters");

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `GriffinCols` — `inputs`, every round's two register blocks and `post`,
    // and `outputs` — in layout order, so the buffer is fully initialized.
    unsafe {
        fill_trace::<F, GriffinCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, REGISTERS, ROUNDS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(call, input, params);
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(calls, inputs, params);
            },
        )
    }
}

/// One round over any algebra, returning the two S-boxes' register values.
///
/// Mirrors `air::eval_round`: the same two witnessed S-boxes in the same order,
/// then the Horst words, then the linear layer — and, like there, the round
/// constants are the caller's to add.
#[inline]
fn generate_round<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    state: &mut [A; WIDTH],
    params: &GriffinParams<F, WIDTH, ROUNDS>,
) -> ([A; WIDTH], [[A; REGISTERS]; 2]) {
    // Pinned by `eval_inverse_power_map`'s `y_0^alpha == x_0`.
    let (y_0, registers_0) =
        generate_inverse_power_map::<A, ALPHA, REGISTERS>(state[0].dup(), params.alpha_inv);
    // Pinned by `eval_power_map`'s assertion inside `y_1 == x_1^alpha`.
    let (y_1, registers_1) = generate_power_map::<A, ALPHA, REGISTERS>(state[1].dup());

    let post = horst_layer(params, state, &y_0, &y_1);
    *state = core::array::from_fn(|i| post[i].dup());
    mds_multiply(state, &params.m);
    (post, [registers_0, registers_1])
}

/// One call, scalar.
fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    call: &mut GriffinCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>,
    input: &[F],
    params: &GriffinParams<F, WIDTH, ROUNDS>,
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }

    mds_multiply(&mut state, &params.m);
    for round in 0..ROUNDS {
        let (post, powers) =
            generate_round::<F, F, WIDTH, REGISTERS, ROUNDS, ALPHA>(&mut state, params);
        write_round(&mut call.rounds[round], &powers, &post, |value| *value);
        add_round_constants(&mut state, &params.rcons[round]);
    }

    for (cell, value) in call.outputs.iter_mut().zip(state) {
        cell.write(value);
    }
}

/// `F::Packing::WIDTH` calls at once, running the round function over
/// `F::Packing`. Mirrors [`generate_call`] step for step; only the writes
/// differ, because each lane owns its own column struct.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    calls: &mut [GriffinCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>],
    inputs: &[Vec<F>],
    params: &GriffinParams<F, WIDTH, ROUNDS>,
) {
    debug_assert_eq!(calls.len(), F::Packing::WIDTH);
    debug_assert_eq!(inputs.len(), F::Packing::WIDTH);
    for (call, input) in calls.iter_mut().zip(inputs) {
        for (cell, value) in call.inputs.iter_mut().zip(input) {
            cell.write(*value);
        }
    }

    let mut state: [F::Packing; WIDTH] =
        core::array::from_fn(|i| F::Packing::from_fn(|lane| inputs[lane][i]));
    mds_multiply(&mut state, &params.m);
    for round in 0..ROUNDS {
        let (post, powers) =
            generate_round::<F, F::Packing, WIDTH, REGISTERS, ROUNDS, ALPHA>(&mut state, params);
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.rounds[round], &powers, &post, |value| {
                value.extract(lane)
            });
        }
        add_round_constants(&mut state, &params.rcons[round]);
    }

    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, value) in call.outputs.iter_mut().zip(&state) {
            cell.write(value.extract(lane));
        }
    }
}

/// Write one round's cells in layout order: both register blocks, then the
/// non-linear layer's output.
#[inline]
fn write_round<F: Field, A, const WIDTH: usize, const REGISTERS: usize>(
    round: &mut Round<MaybeUninit<F>, WIDTH, REGISTERS>,
    powers: &[[A; REGISTERS]; 2],
    post: &[A; WIDTH],
    project: impl Fn(&A) -> F,
) {
    for (cells, values) in round.powers.iter_mut().zip(powers) {
        for (cell, value) in cells.0.iter_mut().zip(values) {
            cell.write(project(value));
        }
    }
    for (cell, value) in round.post.iter_mut().zip(post) {
        cell.write(project(value));
    }
}
