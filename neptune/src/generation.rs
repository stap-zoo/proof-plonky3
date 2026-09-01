//! Trace generation, kept in the same leading-MDS / round / post-ARK order as
//! [`crate::air::eval`], and split the same way: one function per round kind,
//! each returning the register values its `air.rs` twin asserts.
//!
//! The round functions are generic over the ring rather than over the builder,
//! which is what lets the same code run over `F` and over `F::Packing`: the
//! scalar path and the SIMD path are the same round function, and only the cell
//! *writing* differs. The allocation around both is
//! [`harness::permutation::trace::fill_trace`]'s.
//!
//! Every register written here is prover-chosen and is worth nothing until
//! `air.rs` pins it (POLICY §9): `lm` by `eval_external_round`'s `assert_eq`,
//! `powers` by the power-map gadget's own assertions.

use core::mem::MaybeUninit;

use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{Algebra, Field, PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use p3_poseidon2::matmul_internal;
use tracing::instrument;

use crate::columns::{ExternalRound, InternalRound, NeptuneCols, num_cols};
use crate::native::{external_sbox, external_sbox_first_square, external_sbox_pair};
use crate::params::NeptuneParams;

/// One external round over any algebra, returning its Lai--Massey registers.
///
/// Mirrors [`crate::air::eval_external_round`] step for step.
#[inline]
fn generate_external_round<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const EXT: usize,
    const INT: usize,
    const LM: usize,
>(
    state: &mut [A; WIDTH],
    round: usize,
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
) -> [A; LM] {
    let mut registers = core::array::from_fn(|_| A::ZERO);
    if LM == 0 {
        external_sbox(state, params.gamma.into());
    } else {
        for (index, pair) in state.chunks_exact_mut(2).enumerate() {
            // Pinned by `eval_external_round`'s `assert_eq` on `cells.lm[j]`.
            let square = external_sbox_first_square(&pair[0], &pair[1]);
            registers[index] = square.dup();
            let (out0, out1) =
                external_sbox_pair(pair[0].dup(), pair[1].dup(), square, params.gamma.into());
            pair[0] = out0;
            pair[1] = out1;
        }
    }
    mds_multiply(state, &params.m_ext);
    add_round_constants(state, &params.rcons[round + 1]);
    registers
}

/// One internal round over any algebra, returning its power-map registers.
///
/// Mirrors [`crate::air::eval_internal_round`] step for step.
#[inline]
fn generate_internal_round<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const PREGS: usize,
>(
    state: &mut [A; WIDTH],
    round: usize,
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
) -> [A; PREGS] {
    // Pinned by the power-map gadget's own assertions inside `eval_power_map`.
    let (output, registers) = generate_power_map::<A, DEGREE, PREGS>(state[0].dup());
    state[0] = output;
    matmul_internal(state, params.m_int_diag_m_1);
    add_round_constants(state, &params.rcons[round + 1]);
    registers
}

/// Write one block of cells from one block of values, projected into the lane.
#[inline]
fn write_cells<F: Field, A>(
    cells: &mut [MaybeUninit<F>],
    values: &[A],
    project: &impl Fn(&A) -> F,
) {
    for (cell, value) in cells.iter_mut().zip(values) {
        cell.write(project(value));
    }
}

#[inline]
fn write_external<F: Field, A, const WIDTH: usize, const LM: usize>(
    cells: &mut ExternalRound<MaybeUninit<F>, WIDTH, LM>,
    registers: &[A; LM],
    state: &[A; WIDTH],
    project: &impl Fn(&A) -> F,
) {
    write_cells(&mut cells.lm, registers, project);
    write_cells(&mut cells.post, state, project);
}

#[inline]
fn write_internal<F: Field, A, const WIDTH: usize, const PREGS: usize>(
    cells: &mut InternalRound<MaybeUninit<F>, WIDTH, PREGS>,
    registers: &[A; PREGS],
    state: &[A; WIDTH],
    project: &impl Fn(&A) -> F,
) {
    write_cells(&mut cells.powers, registers, project);
    write_cells(&mut cells.post, state, project);
}

/// One call, scalar.
///
/// Every cell of [`NeptuneCols`] is written here — `inputs`, then each round's
/// registers and `post` — which is what
/// [`harness::permutation::trace::fill_trace`]'s safety contract asks for.
fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
>(
    call: &mut NeptuneCols<MaybeUninit<F>, WIDTH, HALF_EXT, INT, LM, PREGS>,
    input: &[F],
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
) {
    assert_eq!(EXT, 2 * HALF_EXT);
    let project = |value: &F| *value;
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }

    mds_multiply(&mut state, &params.m_ext);
    let mut round = 0;
    for cells in &mut call.first {
        let registers =
            generate_external_round::<F, F, WIDTH, EXT, INT, LM>(&mut state, round, params);
        write_external(cells, &registers, &state, &project);
        round += 1;
    }
    for cells in &mut call.internal {
        let registers = generate_internal_round::<F, F, WIDTH, EXT, INT, DEGREE, PREGS>(
            &mut state, round, params,
        );
        write_internal(cells, &registers, &state, &project);
        round += 1;
    }
    for cells in &mut call.last {
        let registers =
            generate_external_round::<F, F, WIDTH, EXT, INT, LM>(&mut state, round, params);
        write_external(cells, &registers, &state, &project);
        round += 1;
    }
}

/// `F::Packing::WIDTH` calls at once, running the round function over
/// `F::Packing`. Mirrors [`generate_call`] step for step; only the writes
/// differ, because each lane owns its own column struct.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
>(
    calls: &mut [NeptuneCols<MaybeUninit<F>, WIDTH, HALF_EXT, INT, LM, PREGS>],
    inputs: &[Vec<F>],
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
) {
    assert_eq!(EXT, 2 * HALF_EXT);
    debug_assert_eq!(calls.len(), F::Packing::WIDTH);
    for (call, input) in calls.iter_mut().zip(inputs) {
        for (cell, value) in call.inputs.iter_mut().zip(input) {
            cell.write(*value);
        }
    }

    let mut state: [F::Packing; WIDTH] =
        core::array::from_fn(|i| F::Packing::from_fn(|lane| inputs[lane][i]));
    mds_multiply(&mut state, &params.m_ext);
    let mut round = 0;
    for index in 0..HALF_EXT {
        let registers = generate_external_round::<F, F::Packing, WIDTH, EXT, INT, LM>(
            &mut state, round, params,
        );
        for (lane, call) in calls.iter_mut().enumerate() {
            let project = |value: &F::Packing| value.extract(lane);
            write_external(&mut call.first[index], &registers, &state, &project);
        }
        round += 1;
    }
    for index in 0..INT {
        let registers = generate_internal_round::<F, F::Packing, WIDTH, EXT, INT, DEGREE, PREGS>(
            &mut state, round, params,
        );
        for (lane, call) in calls.iter_mut().enumerate() {
            let project = |value: &F::Packing| value.extract(lane);
            write_internal(&mut call.internal[index], &registers, &state, &project);
        }
        round += 1;
    }
    for index in 0..HALF_EXT {
        let registers = generate_external_round::<F, F::Packing, WIDTH, EXT, INT, LM>(
            &mut state, round, params,
        );
        for (lane, call) in calls.iter_mut().enumerate() {
            let project = |value: &F::Packing| value.extract(lane);
            write_external(&mut call.last[index], &registers, &state, &project);
        }
        round += 1;
    }
}

/// Generate one independent call per row.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate Neptune trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
>(
    inputs: &[Vec<F>],
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, HALF_EXT, INT, LM, PREGS>();
    fill_calls::<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>(
        inputs,
        params,
        ncols,
        extra_capacity_bits,
    )
}

/// Generate `VECTOR_LEN` fully independent calls per row.
///
/// # Panics
///
/// If `inputs.len()` is not `VECTOR_LEN` times a power of two, or an input is
/// not `WIDTH` long.
#[instrument(name = "generate vectorized Neptune trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, HALF_EXT, INT, LM, PREGS>() * VECTOR_LEN;
    fill_calls::<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>(
        inputs,
        params,
        ncols,
        extra_capacity_bits,
    )
}

fn fill_calls<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
>(
    inputs: &[Vec<F>],
    params: &NeptuneParams<F, WIDTH, EXT, INT>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `NeptuneCols` — `inputs`, then every round's registers (`lm`, `powers`)
    // and `post`, across all three round blocks — in layout order, so the
    // buffer is fully initialized.
    unsafe {
        fill_trace::<F, NeptuneCols<MaybeUninit<F>, WIDTH, HALF_EXT, INT, LM, PREGS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, HALF_EXT, INT, LM, PREGS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>(
                    call, input, params,
                );
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>(
                    calls, inputs, params,
                );
            },
        )
    }
}
