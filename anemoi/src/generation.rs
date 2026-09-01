//! Anemoi witness generation, parallel to `air.rs` but evaluating open Flystel.

use core::mem::MaybeUninit;

use harness::gadgets::inverse_power_map::generate_inverse_power_map;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{Algebra, Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::columns::{AnemoiCols, Round, assert_layout, num_cols};
use crate::native::linear_layer;
use crate::params::AnemoiParams;

/// Generate one call per row.
#[instrument(name = "generate Anemoi trace", skip_all)]
pub fn generate_trace_rows<
    F: Field,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, COLUMNS, REGISTERS, ROUNDS>();
    fill_calls::<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>(
        inputs,
        params,
        ncols,
        extra_capacity_bits,
    )
}

/// Generate `VECTOR_LEN` independent calls per row.
#[instrument(name = "generate vectorized Anemoi trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: Field,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, COLUMNS, REGISTERS, ROUNDS>() * VECTOR_LEN;
    fill_calls::<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>(
        inputs,
        params,
        ncols,
        extra_capacity_bits,
    )
}

/// The allocation and the packing-width dispatch are
/// [`harness::permutation::trace::fill_trace`]'s; what is Anemoi's is the pair of fill
/// functions it drives.
fn fill_calls<
    F: Field,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, COLUMNS, ALPHA, REGISTERS);
    assert_eq!(params.alpha, ALPHA, "generator alpha must match parameters");

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `AnemoiCols` — `inputs`, every round's powers and post state, and
    // `outputs` — in layout order, so the buffer is fully initialized.
    unsafe {
        fill_trace::<F, AnemoiCols<MaybeUninit<F>, WIDTH, COLUMNS, REGISTERS, ROUNDS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, COLUMNS, REGISTERS, ROUNDS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>(call, input, params);
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>(
                    calls, inputs, params,
                );
            },
        )
    }
}

fn generate_call<
    F: Field,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    call: &mut AnemoiCols<MaybeUninit<F>, WIDTH, COLUMNS, REGISTERS, ROUNDS>,
    input: &[F],
    params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }
    for round in 0..ROUNDS {
        let powers = generate_round::<F, F, WIDTH, COLUMNS, REGISTERS, ALPHA>(
            &mut state,
            &params.c[round],
            &params.d[round],
            &params.m_x,
            &params.m_y,
            params.beta,
            params.gamma,
            params.delta,
            params.alpha_inv,
        );
        write_round(&mut call.rounds[round], &powers, &state, |value| *value);
    }
    linear_layer(&mut state, &params.m_x, &params.m_y);
    for (cell, value) in call.outputs.iter_mut().zip(state) {
        cell.write(value);
    }
}

fn generate_call_batch<
    F: Field,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
>(
    calls: &mut [AnemoiCols<MaybeUninit<F>, WIDTH, COLUMNS, REGISTERS, ROUNDS>],
    inputs: &[Vec<F>],
    params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
) {
    debug_assert_eq!(calls.len(), F::Packing::WIDTH);
    for (call, input) in calls.iter_mut().zip(inputs) {
        for (cell, value) in call.inputs.iter_mut().zip(input) {
            cell.write(*value);
        }
    }
    let mut state: [F::Packing; WIDTH] =
        core::array::from_fn(|i| F::Packing::from_fn(|lane| inputs[lane][i]));
    for round in 0..ROUNDS {
        let powers = generate_round::<F, F::Packing, WIDTH, COLUMNS, REGISTERS, ALPHA>(
            &mut state,
            &params.c[round],
            &params.d[round],
            &params.m_x,
            &params.m_y,
            params.beta,
            params.gamma,
            params.delta,
            params.alpha_inv,
        );
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.rounds[round], &powers, &state, |value| {
                value.extract(lane)
            });
        }
    }
    linear_layer(&mut state, &params.m_x, &params.m_y);
    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, value) in call.outputs.iter_mut().zip(&state) {
            cell.write(value.extract(lane));
        }
    }
}

fn write_round<F: Field, A, const WIDTH: usize, const COLUMNS: usize, const REGISTERS: usize>(
    round: &mut Round<MaybeUninit<F>, WIDTH, COLUMNS, REGISTERS>,
    powers: &[[A; REGISTERS]; COLUMNS],
    state: &[A; WIDTH],
    project: impl Fn(&A) -> F,
) {
    for (cells, values) in round.powers.iter_mut().zip(powers) {
        for (cell, value) in cells.0.iter_mut().zip(values) {
            cell.write(project(value));
        }
    }
    for (cell, value) in round.post.iter_mut().zip(state) {
        cell.write(project(value));
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn generate_round<
    F: Field,
    A: Algebra<F>,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ALPHA: u64,
>(
    state: &mut [A; WIDTH],
    c: &[F; COLUMNS],
    d: &[F; COLUMNS],
    m_x: &[[F; COLUMNS]; COLUMNS],
    m_y: &[[F; COLUMNS]; COLUMNS],
    beta: F,
    gamma: F,
    delta: F,
    alpha_inv: u64,
) -> [[A; REGISTERS]; COLUMNS] {
    for i in 0..COLUMNS {
        state[i] += c[i];
        state[COLUMNS + i] += d[i];
    }
    linear_layer(state, m_x, m_y);

    let mut powers: [[A; REGISTERS]; COLUMNS] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));
    for i in 0..COLUMNS {
        let x = state[i].dup();
        let y = state[COLUMNS + i].dup();
        let open = x - (y.square() * beta + gamma);

        // The witnessed cell is `v`; `inverse_power` is `y - v`, pinned by
        // `inverse_power_map::eval_inverse_power_map` against `open`, and the
        // registers with it.
        let (inverse_power, registers) =
            generate_inverse_power_map::<A, ALPHA, REGISTERS>(open.dup(), alpha_inv);
        let v = y - inverse_power;
        let u = open + v.square() * beta + delta;
        powers[i] = registers;
        state[i] = u;
        state[COLUMNS + i] = v;
    }
    powers
}
