//! The witness producer — never evidence.
//!
//! POLICY §6: a free `generate_trace_rows(inputs, params, extra_capacity_bits)`
//! and its vectorized twin, over `MaybeUninit` columns, with
//! `extra_capacity_bits` forwarded so the prover's LDE happens in place. The
//! allocation, the alignment assertions and the packing-width dispatch are
//! `harness::permutation::trace::fill_trace`'s; what is Rescue-Prime's is the half-round
//! below.
//!
//! POLICY §9: nothing written here is evidence of anything. Every cell it writes
//! is a cell a malicious prover could have written differently, so each one
//! needs a constraint in `air.rs` naming it — the table in that file's docs is
//! that list, and `tests/air.rs`'s per-cell negative test is what checks it.
//!
//! # Kept parallel to `air.rs`
//!
//! Same order, same names, same per-half-round split: `generate_half_round`
//! against `eval_half_round`, the same `forward` flag deciding the same branch,
//! the constant row added by the caller in both. The one structural difference
//! is that the arithmetic here is generic over the ring rather than over the
//! builder, which is what lets the same code run over `F` and over `F::Packing`.
//!
//! # This is the only place the inverse power is computed
//!
//! `x^(1/alpha)` is an exponentiation by `alpha^{-1} mod (p-1)` — a full-width
//! exponent, and by far the most expensive thing in this file, `WIDTH` times per
//! inverse half-round. The AIR never does it; it checks the cheap direction of
//! the same relation. That asymmetry is Rescue's whole design and it is also why
//! trace generation here costs more than the constraint count suggests.

use core::mem::MaybeUninit;

use harness::gadgets::inverse_power_map::generate_inverse_power_map;
use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{Algebra, Field, PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::half_round::columns::{HalfRound, RescuePrimeCols, assert_layout, num_cols};
use crate::params::RescuePrimeParams;

/// One row per call.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate Rescue-Prime trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, REGISTERS, HALF_ROUNDS>();
    fill_calls::<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
        inputs,
        params,
        ncols,
        extra_capacity_bits,
    )
}

/// `VECTOR_LEN` calls per row.
///
/// # Panics
///
/// If `inputs.len()` is not `VECTOR_LEN` times a power of two, or an input is
/// not `WIDTH` long.
#[instrument(name = "generate vectorized Rescue-Prime trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, REGISTERS, HALF_ROUNDS>() * VECTOR_LEN;
    fill_calls::<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
        inputs,
        params,
        ncols,
        extra_capacity_bits,
    )
}

fn fill_calls<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, ALPHA, REGISTERS, HALF_ROUNDS);
    assert_eq!(params.alpha, ALPHA, "generator alpha must match parameters");

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `RescuePrimeCols` — `inputs`, every half-round's register blocks and
    // `post`, and `outputs` — in layout order, so the buffer is fully
    // initialized.
    unsafe {
        fill_trace::<F, RescuePrimeCols<MaybeUninit<F>, WIDTH, REGISTERS, HALF_ROUNDS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, REGISTERS, HALF_ROUNDS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(call, input, params);
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
                    calls, inputs, params,
                );
            },
        )
    }
}

/// One half-round over any algebra, returning the committed S-box outputs and
/// their register values.
///
/// Mirrors `air::eval_half_round`: the same direction flag, the same power map,
/// then the linear layer — and, like there, the constant row is the caller's.
#[inline]
fn generate_half_round<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    state: &mut [A; WIDTH],
    forward: bool,
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
) -> ([A; WIDTH], [[A; REGISTERS]; WIDTH]) {
    let mut registers: [[A; REGISTERS]; WIDTH] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));

    let post: [A; WIDTH] = core::array::from_fn(|i| {
        let (value, block) = if forward {
            // Pinned by `eval_power_map`'s assertion inside `post == x^alpha`.
            generate_power_map::<A, ALPHA, REGISTERS>(state[i].dup())
        } else {
            // Pinned by `eval_inverse_power_map`'s `post^alpha == x`.
            generate_inverse_power_map::<A, ALPHA, REGISTERS>(state[i].dup(), params.alpha_inv)
        };
        registers[i] = block;
        value
    });

    *state = core::array::from_fn(|i| post[i].dup());
    mds_multiply(state, &params.m);
    (post, registers)
}

/// One call, scalar.
fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    call: &mut RescuePrimeCols<MaybeUninit<F>, WIDTH, REGISTERS, HALF_ROUNDS>,
    input: &[F],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }

    for half_round in 0..HALF_ROUNDS {
        let (post, registers) = generate_half_round::<F, F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
            &mut state,
            half_round.is_multiple_of(2),
            params,
        );
        write_half_round(&mut call.half_rounds[half_round], &registers, &post, |v| *v);
        add_round_constants(&mut state, &params.rcons[half_round]);
    }

    for (cell, value) in call.outputs.iter_mut().zip(state) {
        cell.write(value);
    }
}

/// `F::Packing::WIDTH` calls at once, running the half-round over `F::Packing`.
/// Mirrors [`generate_call`] step for step; only the writes differ, because each
/// lane owns its own column struct.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    calls: &mut [RescuePrimeCols<MaybeUninit<F>, WIDTH, REGISTERS, HALF_ROUNDS>],
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
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
    for half_round in 0..HALF_ROUNDS {
        let (post, registers) =
            generate_half_round::<F, F::Packing, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
                &mut state,
                half_round.is_multiple_of(2),
                params,
            );
        for (lane, call) in calls.iter_mut().enumerate() {
            write_half_round(&mut call.half_rounds[half_round], &registers, &post, |v| {
                v.extract(lane)
            });
        }
        add_round_constants(&mut state, &params.rcons[half_round]);
    }

    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, value) in call.outputs.iter_mut().zip(&state) {
            cell.write(value.extract(lane));
        }
    }
}

/// Write one half-round's cells in layout order: every word's register block,
/// then the non-linear layer's output.
#[inline]
fn write_half_round<F: Field, A, const WIDTH: usize, const REGISTERS: usize>(
    half_round: &mut HalfRound<MaybeUninit<F>, WIDTH, REGISTERS>,
    registers: &[[A; REGISTERS]; WIDTH],
    post: &[A; WIDTH],
    project: impl Fn(&A) -> F,
) {
    for (cells, values) in half_round.powers.iter_mut().zip(registers) {
        for (cell, value) in cells.0.iter_mut().zip(values) {
            cell.write(project(value));
        }
    }
    for (cell, value) in half_round.post.iter_mut().zip(post) {
        cell.write(project(value));
    }
}
