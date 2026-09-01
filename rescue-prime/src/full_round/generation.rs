//! The witness producer — never evidence.
//!
//! POLICY §6, kept line-for-line parallel with `air.rs` in this module:
//! `generate_round` against `eval_round`, the forward half then the inverse
//! half, the same two constant rows in the same order.
//!
//! # The one asymmetry worth naming
//!
//! `air.rs` walks the inverse half **backwards**, through `m_inv`, because that
//! is what makes the constraint a forward power map. This file walks it
//! forwards, because that is what computing the permutation means. The two
//! meet at `w`: the value this file writes into the state and the value that
//! file reconstructs as `M^{-1}(post - c)` are the same vector, and the round's
//! constraint is exactly the claim that they are.
//!
//! POLICY §9: nothing written here is evidence of anything. Every cell needs a
//! constraint in `air.rs` naming it, and `tests/full_round.rs`'s per-cell
//! negative test is what checks that claim rather than restating it.

use core::mem::MaybeUninit;

use harness::gadgets::inverse_power_map::generate_inverse_power_map;
use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{Algebra, Field, PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::full_round::columns::{FullRoundCols, Round, assert_layout, num_cols};
use crate::params::RescuePrimeParams;

/// One row per call.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate full-round Rescue-Prime trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, REGISTERS, ROUNDS>();
    fill_calls::<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>(
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
#[instrument(name = "generate vectorized full-round Rescue-Prime trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, REGISTERS, ROUNDS>() * VECTOR_LEN;
    fill_calls::<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>(
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
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, ALPHA, REGISTERS, HALF_ROUNDS, ROUNDS);
    assert_eq!(params.alpha, ALPHA, "generator alpha must match parameters");

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `FullRoundCols` — `inputs`, and every round's two register blocks and
    // `post` — in layout order, so the buffer is fully initialized.
    unsafe {
        fill_trace::<F, FullRoundCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, REGISTERS, ROUNDS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>(
                    call, input, params,
                );
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>(
                    calls, inputs, params,
                );
            },
        )
    }
}

/// One full round over any algebra, returning the committed state and both
/// register blocks.
#[inline]
fn generate_round<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    state: &mut [A; WIDTH],
    forward_constants: &[F; WIDTH],
    inverse_constants: &[F; WIDTH],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
) -> ([[A; REGISTERS]; WIDTH], [[A; REGISTERS]; WIDTH]) {
    let mut forward_registers: [[A; REGISTERS]; WIDTH] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));
    let mut inverse_registers: [[A; REGISTERS]; WIDTH] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));

    // Forward half. Pinned by `eval_power_map` inside the round's equation.
    let mut forward: [A; WIDTH] = core::array::from_fn(|i| {
        let (value, registers) = generate_power_map::<A, ALPHA, REGISTERS>(state[i].dup());
        forward_registers[i] = registers;
        value
    });
    mds_multiply(&mut forward, &params.m);
    add_round_constants(&mut forward, forward_constants);

    // Inverse half. `air.rs` reconstructs this vector as `M^{-1}(post - c)`
    // rather than computing it; the round's equation is the claim that the two
    // agree.
    let mut inverse: [A; WIDTH] = core::array::from_fn(|i| {
        let (root, registers) =
            generate_inverse_power_map::<A, ALPHA, REGISTERS>(forward[i].dup(), params.alpha_inv);
        inverse_registers[i] = registers;
        root
    });
    mds_multiply(&mut inverse, &params.m);
    for (word, constant) in inverse.iter_mut().zip(inverse_constants) {
        *word += *constant;
    }
    *state = inverse;

    (forward_registers, inverse_registers)
}

/// One call, scalar.
fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    call: &mut FullRoundCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>,
    input: &[F],
    params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }

    for round in 0..ROUNDS {
        let (forward, inverse) = generate_round::<F, F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
            &mut state,
            &params.rcons[2 * round],
            &params.rcons[2 * round + 1],
            params,
        );
        write_round(&mut call.rounds[round], &forward, &inverse, &state, |v| *v);
    }
}

/// `F::Packing::WIDTH` calls at once. Mirrors [`generate_call`] step for step;
/// only the writes differ, because each lane owns its own column struct.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
>(
    calls: &mut [FullRoundCols<MaybeUninit<F>, WIDTH, REGISTERS, ROUNDS>],
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
    for round in 0..ROUNDS {
        let (forward, inverse) =
            generate_round::<F, F::Packing, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>(
                &mut state,
                &params.rcons[2 * round],
                &params.rcons[2 * round + 1],
                params,
            );
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.rounds[round], &forward, &inverse, &state, |v| {
                v.extract(lane)
            });
        }
    }
}

/// Write one round's cells in layout order: forward registers, inverse
/// registers, then the committed state.
#[inline]
fn write_round<F: Field, A, const WIDTH: usize, const REGISTERS: usize>(
    round: &mut Round<MaybeUninit<F>, WIDTH, REGISTERS>,
    forward: &[[A; REGISTERS]; WIDTH],
    inverse: &[[A; REGISTERS]; WIDTH],
    post: &[A; WIDTH],
    project: impl Fn(&A) -> F,
) {
    for (cells, values) in round.forward.iter_mut().zip(forward) {
        for (cell, value) in cells.0.iter_mut().zip(values) {
            cell.write(project(value));
        }
    }
    for (cells, values) in round.inverse.iter_mut().zip(inverse) {
        for (cell, value) in cells.0.iter_mut().zip(values) {
            cell.write(project(value));
        }
    }
    for (cell, value) in round.post.iter_mut().zip(post) {
        cell.write(project(value));
    }
}
