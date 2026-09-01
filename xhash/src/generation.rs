//! The witness producer — never evidence.
//!
//! POLICY §6: a free `generate_trace_rows(inputs, constants, extra_capacity_bits)`
//! and its vectorized twin — `MaybeUninit` columns via `align_to_mut`,
//! `par_chunks_mut` over `F::Packing::WIDTH`. It takes inputs and never invents
//! them, and it forwards `extra_capacity_bits` so the prover's LDE happens in
//! place instead of reallocating a trace this file could have sized correctly.
//!
//! POLICY §9: nothing written here is evidence of anything. Every cell it
//! writes is a cell a malicious prover could have written differently, so each
//! one needs a constraint in `air.rs` naming it — and the negative test that
//! corrupts one cell and watches the proof fail is what checks that claim
//! rather than restating it.

use core::mem::MaybeUninit;

use harness::gadgets::inverse_power_map::generate_inverse_power_map;
use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{Algebra, Dup, Field, PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::columns::{Cycle, ExtensionPower, Power, XHashCols, assert_layout, num_cols};
use crate::native::{
    extension_half_power, extension_power_from_cpolys, extension_power_from_half,
    extension_power_from_quadratics, extension_quadratics,
};
use crate::params::XHashParams;

/// One row per independent call.
#[instrument(name = "generate XHash trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    fill_calls::<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>(
        inputs,
        params,
        num_cols::<WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>(),
        extra_capacity_bits,
    )
}

/// `LANES` independent calls per row.
#[instrument(name = "generate vectorized XHash trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
    const LANES: usize,
>(
    inputs: &[Vec<F>],
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), LANES);
    fill_calls::<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>(
        inputs,
        params,
        num_cols::<WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>() * LANES,
        extra_capacity_bits,
    )
}

fn fill_calls<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    inputs: &[Vec<F>],
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, ACTIVE, ALPHA, REGISTERS, P3_BLOCKS, CYCLES);
    assert_eq!(params.alpha, ALPHA);
    assert_eq!(params.skip_middle, ACTIVE != WIDTH);
    // The same refusal `air::XHashAir::from_params` makes, for the same reason:
    // the two files must not disagree about which P3 basis this variant is.
    assert!(
        P3_BLOCKS != 1 || params.structured,
        "{}: the structured P3 basis needs a coordinate table that is x^alpha \
         in the declared quotient (see ERROR.md)",
        params.name
    );

    // SAFETY: both closures write inputs, every register block, both committed
    // states in every cycle, and outputs, in the struct's declaration order.
    unsafe {
        fill_trace::<F, XHashCols<MaybeUninit<F>, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>(
                    call, input, params,
                );
            },
            |calls, inputs| {
                generate_call_batch::<
                    F,
                    WIDTH,
                    ACTIVE,
                    REGISTERS,
                    P3_BLOCKS,
                    CYCLES,
                    CONSTANT_ROWS,
                    ALPHA,
                >(calls, inputs, params);
            },
        )
    }
}

struct GeneratedCycle<
    A,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
> {
    forward_powers: [[A; REGISTERS]; WIDTH],
    backward_powers: [[A; REGISTERS]; ACTIVE],
    backward: [A; WIDTH],
    extension_powers: [[[A; REGISTERS]; P3_BLOCKS]; WIDTH],
    post: [A; WIDTH],
}

/// The P3 layer over one triple, in the basis `air::eval_p3_triple` constrains.
///
/// Kept beside its constraint twin and writing the same registers in the same
/// order — the pairing POLICY §6 asks for.
#[inline]
fn generate_p3_triple<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    input: [A; 3],
    registers: &mut [[[A; REGISTERS]; P3_BLOCKS]],
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
) -> [A; 3] {
    if REGISTERS == 0 {
        return extension_power_from_cpolys(input, &params.cpolys);
    }
    if P3_BLOCKS == 1 {
        let half = extension_half_power::<_, _, ALPHA>(
            input.each_ref().map(Dup::dup),
            &params.reduction,
            &params.reduction_x4,
        );
        for coordinate in 0..3 {
            // Pinned by `air::eval_p3_triple`'s coordinate-wise equality.
            registers[coordinate][0][0] = half[coordinate].dup();
        }
        extension_power_from_half(input, half, &params.reduction, &params.reduction_x4)
    } else {
        let quadratics = extension_quadratics(input.each_ref().map(Dup::dup));
        for index in 0..6 {
            // Pinned by `air::eval_p3_triple`'s six quadratic equalities.
            registers[index / 2][index % 2][0] = quadratics[index].dup();
        }
        extension_power_from_quadratics(input, quadratics, &params.cpolys)
    }
}

/// Generate one F/B/P3 group. This is line-for-line parallel to
/// `air::eval_cycle`: the same constant indices, skip condition and committed
/// intermediates appear in the same order.
#[inline]
fn generate_cycle<
    F: PrimeField64,
    A: Algebra<F>,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    state: &mut [A; WIDTH],
    cycle_index: usize,
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
) -> GeneratedCycle<A, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS> {
    let first = 3 * cycle_index;

    add_round_constants(state, &params.rcons[first]);
    mds_multiply(state, &params.m);
    let mut forward_powers: [[A; REGISTERS]; WIDTH] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));
    for (word, powers) in state.iter_mut().zip(&mut forward_powers) {
        let (output, registers) = generate_power_map::<A, ALPHA, REGISTERS>(word.dup());
        *word = output;
        *powers = registers;
    }

    mds_multiply(state, &params.m);
    add_round_constants(state, &params.rcons[first + 1]);
    let mut backward_powers: [[A; REGISTERS]; ACTIVE] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));
    let mut active = 0;
    let backward: [A; WIDTH] = core::array::from_fn(|i| {
        if params.skip_middle && i % 3 == 1 {
            state[i].dup()
        } else {
            let (root, registers) =
                generate_inverse_power_map::<A, ALPHA, REGISTERS>(state[i].dup(), params.alpha_inv);
            backward_powers[active] = registers;
            active += 1;
            root
        }
    });
    debug_assert_eq!(active, ACTIVE);
    *state = backward.each_ref().map(Dup::dup);

    add_round_constants(state, &params.rcons[first + 2]);
    let mut extension_powers: [[[A; REGISTERS]; P3_BLOCKS]; WIDTH] =
        core::array::from_fn(|_| core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO)));
    let mut post: [A; WIDTH] = core::array::from_fn(|_| A::ZERO);
    for triple in 0..WIDTH / 3 {
        let base = 3 * triple;
        let input = [
            state[base].dup(),
            state[base + 1].dup(),
            state[base + 2].dup(),
        ];
        let output = generate_p3_triple::<F, A, WIDTH, REGISTERS, P3_BLOCKS, CONSTANT_ROWS, ALPHA>(
            input,
            &mut extension_powers[base..base + 3],
            params,
        );
        for coordinate in 0..3 {
            post[base + coordinate] = output[coordinate].dup();
        }
    }
    *state = post.each_ref().map(Dup::dup);

    GeneratedCycle {
        forward_powers,
        backward_powers,
        backward,
        extension_powers,
        post,
    }
}

fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    call: &mut XHashCols<MaybeUninit<F>, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>,
    input: &[F],
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }
    for cycle in 0..CYCLES {
        let values =
            generate_cycle::<F, F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CONSTANT_ROWS, ALPHA>(
                &mut state, cycle, params,
            );
        write_cycle(&mut call.cycles[cycle], &values, |value| *value);
    }
    mds_multiply(&mut state, &params.m);
    add_round_constants(&mut state, &params.rcons[CONSTANT_ROWS - 1]);
    for (cell, value) in call.outputs.iter_mut().zip(state) {
        cell.write(value);
    }
}

fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
>(
    calls: &mut [XHashCols<MaybeUninit<F>, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>],
    inputs: &[Vec<F>],
    params: &XHashParams<F, WIDTH, CONSTANT_ROWS>,
) {
    debug_assert_eq!(calls.len(), F::Packing::WIDTH);
    for (call, input) in calls.iter_mut().zip(inputs) {
        for (cell, value) in call.inputs.iter_mut().zip(input) {
            cell.write(*value);
        }
    }

    let mut state: [F::Packing; WIDTH] =
        core::array::from_fn(|i| F::Packing::from_fn(|lane| inputs[lane][i]));
    for cycle in 0..CYCLES {
        let values = generate_cycle::<
            F,
            F::Packing,
            WIDTH,
            ACTIVE,
            REGISTERS,
            P3_BLOCKS,
            CONSTANT_ROWS,
            ALPHA,
        >(&mut state, cycle, params);
        for (lane, call) in calls.iter_mut().enumerate() {
            write_cycle(&mut call.cycles[cycle], &values, |value| {
                value.extract(lane)
            });
        }
    }
    mds_multiply(&mut state, &params.m);
    add_round_constants(&mut state, &params.rcons[CONSTANT_ROWS - 1]);
    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, value) in call.outputs.iter_mut().zip(&state) {
            cell.write(value.extract(lane));
        }
    }
}

#[inline]
fn write_cycle<
    F: Field,
    A,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
>(
    cycle: &mut Cycle<MaybeUninit<F>, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS>,
    values: &GeneratedCycle<A, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS>,
    project: impl Fn(&A) -> F,
) {
    fn write_powers<F: Field, A, const COUNT: usize, const REGISTERS: usize>(
        cells: &mut [Power<MaybeUninit<F>, REGISTERS>; COUNT],
        values: &[[A; REGISTERS]; COUNT],
        project: &impl Fn(&A) -> F,
    ) {
        for (block, values) in cells.iter_mut().zip(values) {
            for (cell, value) in block.0.iter_mut().zip(values) {
                cell.write(project(value));
            }
        }
    }

    write_powers(&mut cycle.forward_powers, &values.forward_powers, &project);
    write_powers(
        &mut cycle.backward_powers,
        &values.backward_powers,
        &project,
    );
    for (cell, value) in cycle.backward.iter_mut().zip(&values.backward) {
        cell.write(project(value));
    }
    for (block, values) in cycle
        .extension_powers
        .iter_mut()
        .zip(&values.extension_powers)
    {
        let ExtensionPower(cells) = block;
        for (cells, values) in cells.iter_mut().zip(values) {
            for (cell, value) in cells.iter_mut().zip(values) {
                cell.write(project(value));
            }
        }
    }
    for (cell, value) in cycle.post.iter_mut().zip(&values.post) {
        cell.write(project(value));
    }
}
