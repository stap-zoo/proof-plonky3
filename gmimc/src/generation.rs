//! The witness producer — never evidence.
//!
//! POLICY §6: a free `generate_trace_rows(inputs, constants, extra_capacity_bits)`
//! and its vectorized twin — `MaybeUninit` columns via `align_to_mut`,
//! `par_chunks_mut` over `F::Packing::WIDTH`. It takes inputs and never invents
//! them, and it forwards `extra_capacity_bits` so the prover's LDE happens in
//! place instead of reallocating a trace this file could have sized correctly.
//! The allocation, the alignment assertions and the packing dispatch are
//! [`harness::permutation::trace::fill_trace`]'s; what is GMiMC's is the round
//! function below.
//!
//! POLICY §9: nothing written here is evidence of anything. Every cell is one a
//! malicious prover could have written differently, so each needs a constraint in
//! `air.rs` naming it — the table in [`air::eval`](crate::air::eval)'s docs is
//! that list, and `tests/air.rs`'s per-cell negative test is what checks it.
//!
//! # Kept parallel to `air.rs`
//!
//! Same order, same names, same per-round split: `generate_round` against the
//! S-box loop of `eval`, and the same `WIDTH`-indexed chain underneath. The one
//! structural difference is that the arithmetic here is generic over the ring
//! rather than over the builder, which is what lets the same code run over `F` and
//! over `F::Packing` — the scalar and SIMD paths are the same round function, and
//! only the cell *writing* differs.
//!
//! # This file runs the permutation; it does not run the recurrence
//!
//! `air.rs` rests on an algebraic claim about the round loop: that branch 0's
//! value satisfies `b_r = b_{r-t} + sum y_j`. A generator that wrote the cells
//! *from that recurrence* would check the claim against itself. So this file does
//! what `native.rs` does — a real state, `t-1` additions, a rotation — and simply
//! records branch 0 on the way past. The recurrence is then a property of the
//! trace rather than a rule that produced it, and the known-answer vectors and
//! `check_constraints` are what say the two agree.
//!
//! The rotation is the one thing worth noticing: it is a rotation *here* and an
//! index offset *there*, and that is the whole of the linear layer in both.

use core::mem::MaybeUninit;

use harness::gadgets::power_map::generate_power_map;
use harness::permutation::trace::{
    assert_full_table, assert_full_vectorized_table, fill_trace as fill_calls,
};
use p3_field::{Algebra, Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::columns::{GMiMCCols, Round, assert_layout, num_cols};
use crate::params::shift_source;

/// One row per call.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate GMiMC trace", skip_all)]
pub fn generate_trace_rows<
    F: Field,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    constants: &[F; ROUNDS],
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, REGISTERS, ROUNDS>();
    fill_trace::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(inputs, constants, ncols, extra_capacity_bits)
}

/// `VECTOR_LEN` calls per row.
///
/// # Panics
///
/// If `inputs.len()` is not `VECTOR_LEN` times a power of two, or an input is not
/// `WIDTH` long.
#[instrument(name = "generate vectorized GMiMC trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: Field,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    constants: &[F; ROUNDS],
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, REGISTERS, ROUNDS>() * VECTOR_LEN;
    fill_trace::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(inputs, constants, ncols, extra_capacity_bits)
}

/// Allocate a trace of `ncols` columns holding `inputs.len()` calls, and fill it.
///
/// The two public generators differ only in the `ncols` they pass, which is the
/// identity `vectorized.rs` rests on: `VECTOR_LEN` calls side by side in one row
/// is the same byte layout as `VECTOR_LEN` rows of one call.
fn fill_trace<
    F: Field,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    constants: &[F; ROUNDS],
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, ALPHA, REGISTERS);

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `GMiMCCols` — `inputs`, every round's `head` and `registers`, and
    // `outputs` — in layout order, so the buffer is fully initialized. The
    // register array is empty in the unsplit variant, which writes zero cells and
    // is the only conditional field.
    unsafe {
        fill_calls::<F, GMiMCCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, REGISTERS, ROUNDS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(call, input, constants);
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, REGISTERS, ROUNDS, ALPHA>(calls, inputs, constants);
            },
        )
    }
}

/// One round over any algebra, returning the cells it commits.
///
/// Mirrors the S-box loop of `air::eval`: the round constant is an argument to
/// the power map and never enters the state, and the linear layer is the same
/// rotation [`shift_source`] gives `native.rs`.
///
/// The returned `head` is branch 0 *before* the constant — the value `air.rs`
/// calls `b_r` and reads as a committed cell.
#[inline]
fn generate_round<
    F: Field,
    A: Algebra<F>,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ALPHA: u64,
>(
    state: &mut [A; WIDTH],
    constant: F,
) -> (A, [A; REGISTERS]) {
    let head = state[0].dup();
    // Pinned by `eval_power_map`'s `register == value^2` (or `^3`) assertion.
    let (y, registers) = generate_power_map::<A, ALPHA, REGISTERS>(head.dup() + constant);

    for word in state.iter_mut().skip(1) {
        *word += y.dup();
    }
    let shifted = core::array::from_fn(|i| state[shift_source(i, WIDTH)].dup());
    *state = shifted;

    (head, registers)
}

/// One call, scalar.
fn generate_call<
    F: Field,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    call: &mut GMiMCCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>,
    input: &[F],
    constants: &[F; ROUNDS],
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }

    for (round, constant) in constants.iter().enumerate() {
        let (head, registers) =
            generate_round::<F, F, WIDTH, REGISTERS, ALPHA>(&mut state, *constant);
        write_round(&mut call.rounds[round], &head, &registers, |value| *value);
    }

    // `out[i] = b_{R+i}`: the last `WIDTH` links of the chain are the final state
    // itself, read off in position order. No trailing layer — `_post_rounds` is
    // the identity in `hash.py`.
    for (cell, value) in call.outputs.iter_mut().zip(state) {
        cell.write(value);
    }
}

/// `F::Packing::WIDTH` calls at once, running the round function over
/// `F::Packing`. Mirrors [`generate_call`] step for step; only the writes differ,
/// because each lane owns its own column struct.
fn generate_call_batch<
    F: Field,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    calls: &mut [GMiMCCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>],
    inputs: &[Vec<F>],
    constants: &[F; ROUNDS],
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

    for (round, constant) in constants.iter().enumerate() {
        let (head, registers) =
            generate_round::<F, F::Packing, WIDTH, REGISTERS, ALPHA>(&mut state, *constant);
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.rounds[round], &head, &registers, |value| {
                value.extract(lane)
            });
        }
    }

    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, value) in call.outputs.iter_mut().zip(&state) {
            cell.write(value.extract(lane));
        }
    }
}

/// Write one round's cells in layout order: the committed branch value, then the
/// power map's register.
///
/// POLICY §9: both are prover-chosen. `head` is pinned by the difference rule at
/// chain index `r + t - 1` — or, for round 0, by the base constraint — and
/// `registers` by the power-map gadget's own assertion. Writing `head` *after*
/// the round would still produce a self-consistent trace and a permutation that
/// is off by one round; the KAT is what catches that, and `tests/air.rs` replays
/// the rounds independently to say where.
#[inline]
fn write_round<F: Field, A, const REGISTERS: usize>(
    round: &mut Round<MaybeUninit<F>, REGISTERS>,
    head: &A,
    registers: &[A; REGISTERS],
    project: impl Fn(&A) -> F,
) {
    round.head.write(project(head));
    for (cell, value) in round.registers.iter_mut().zip(registers) {
        cell.write(project(value));
    }
}
