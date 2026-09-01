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
//!
//! # Kept parallel to `air.rs`
//!
//! Same order, same names, same per-round split: `generate_round` against
//! `eval_round`, `generate_feistel` against `eval_feistel`, `M_IO` at both
//! ends of `generate_call` against `eval`. The one structural difference is
//! that the round arithmetic here is generic over the ring rather than over the
//! builder, which is what lets the same code run over `F` and over
//! `F::Packing` — the scalar path and the SIMD path are the same round function,
//! and only the cell *writing* differs.

use core::mem::MaybeUninit;

use harness::gadgets::power_map::generate_power_map;
use harness::permutation::trace::{
    assert_full_table, assert_full_vectorized_table, fill_trace as fill_calls,
};
use p3_field::{Algebra, Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::full_commit::columns::{PSquareHashCols, Round, assert_layout, assert_rounds, num_cols};
use crate::linear;

/// One row per call.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate pSquareHash trace", skip_all)]
pub fn generate_trace_rows<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
>(
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>();
    fill_trace::<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>(
        inputs,
        constants,
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
#[instrument(name = "generate vectorized pSquareHash trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols =
        num_cols::<WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>() * VECTOR_LEN;
    fill_trace::<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>(
        inputs,
        constants,
        ncols,
        extra_capacity_bits,
    )
}

/// Allocate a trace of `ncols` columns holding `inputs.len()` calls, and fill it.
///
/// The allocation, the alignment assertions and the packing-width dispatch are
/// [`harness::permutation::trace::fill_trace`]'s — including the identity the two public
/// generators rest on, that `VECTOR_LEN` calls side by side in one row is the
/// same byte layout as `VECTOR_LEN` rows of one call. What is pSquareHash's is
/// the pair of fill functions below.
///
/// The packed path is the reason a construction owns its generator at all: one
/// field operation per state element covers every lane, so trace generation costs
/// a `Packing::WIDTH`-th of the scalar path — and generation is timed separately
/// from proving (POLICY §11) precisely because it parallelizes differently.
fn fill_trace<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
>(
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST);
    assert_rounds(SPAN, GROUPS, ROUNDS);

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `PSquareHashCols` — `inputs`, each round's Feistel registers and post
    // state, and `outputs` — in layout order, so the buffer is fully
    // initialized.
    unsafe {
        fill_calls::<
            F,
            PSquareHashCols<
                MaybeUninit<F>,
                WIDTH,
                PAIRS,
                REGISTERS,
                REGISTERS_LAST,
                SPAN,
                POST,
                GROUPS,
            >,
        >(
            inputs,
            WIDTH,
            num_cols::<WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<
                    F,
                    WIDTH,
                    PAIRS,
                    REGISTERS,
                    REGISTERS_LAST,
                    SPAN,
                    POST,
                    GROUPS,
                    ROUNDS,
                >(call, input, constants);
            },
            |calls, inputs| {
                generate_call_batch::<
                    F,
                    WIDTH,
                    PAIRS,
                    REGISTERS,
                    REGISTERS_LAST,
                    SPAN,
                    POST,
                    GROUPS,
                    ROUNDS,
                >(calls, inputs, constants);
            },
        )
    }
}

/// One call, scalar. Mirrors `air::eval`.
fn generate_call<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
>(
    call: &mut PSquareHashCols<
        MaybeUninit<F>,
        WIDTH,
        PAIRS,
        REGISTERS,
        REGISTERS_LAST,
        SPAN,
        POST,
        GROUPS,
    >,
    input: &[F],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, x) in call.inputs.iter_mut().zip(state) {
        cell.write(x);
    }

    linear::m_io(state.as_mut_slice());
    for g in 0..GROUPS {
        let first = g * (SPAN + 1);
        generate_group::<F, F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>(
            &mut state,
            core::slice::from_mut(call),
            g,
            &constants[first..first + SPAN + 1],
            |x, _| *x,
        );
    }
    linear::m_io(state.as_mut_slice());

    for (cell, x) in call.outputs.iter_mut().zip(state) {
        cell.write(x);
    }
}

/// `F::Packing::WIDTH` calls at once, running the round function over
/// `F::Packing`. Mirrors [`generate_call`] step for step; only the writes differ,
/// because each lane owns its own column struct.
fn generate_call_batch<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
>(
    calls: &mut [PSquareHashCols<
        MaybeUninit<F>,
        WIDTH,
        PAIRS,
        REGISTERS,
        REGISTERS_LAST,
        SPAN,
        POST,
        GROUPS,
    >],
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
) {
    debug_assert_eq!(calls.len(), F::Packing::WIDTH);
    debug_assert_eq!(inputs.len(), F::Packing::WIDTH);

    for (call, input) in calls.iter_mut().zip(inputs) {
        for (cell, x) in call.inputs.iter_mut().zip(input) {
            cell.write(*x);
        }
    }

    let mut state: [F::Packing; WIDTH] =
        core::array::from_fn(|i| F::Packing::from_fn(|lane| inputs[lane][i]));

    linear::m_io(state.as_mut_slice());
    for g in 0..GROUPS {
        let first = g * (SPAN + 1);
        generate_group::<F, F::Packing, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>(
            &mut state,
            calls,
            g,
            &constants[first..first + SPAN + 1],
            |packed, lane| packed.extract(lane),
        );
    }
    linear::m_io(state.as_mut_slice());

    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, x) in call.outputs.iter_mut().zip(&state) {
            cell.write(x.extract(lane));
        }
    }
}

/// One group — `SPAN` rounds, the committing round, and the commitment — over
/// every lane at once. Mirrors `air::eval_group`.
///
/// The round arithmetic runs **once** over `A`, which is `F` on the scalar path
/// and `F::Packing` on the SIMD one; `project` is the identity there and a lane
/// extraction here. One writer for both paths is what stops them committing
/// different cells for the same variant, and it is why the packed path cannot
/// drift into computing a different round function than the scalar one.
#[inline]
fn generate_group<
    F: Field,
    A: Algebra<F>,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
>(
    state: &mut [A; WIDTH],
    calls: &mut [PSquareHashCols<
        MaybeUninit<F>,
        WIDTH,
        PAIRS,
        REGISTERS,
        REGISTERS_LAST,
        SPAN,
        POST,
        GROUPS,
    >],
    group: usize,
    constants: &[[[F; 2]; PAIRS]],
    project: impl Fn(&A, usize) -> F,
) {
    debug_assert_eq!(constants.len(), SPAN + 1);

    for (k, constants) in constants.iter().enumerate().take(SPAN) {
        let registers = generate_round::<F, A, WIDTH, PAIRS, REGISTERS>(state, constants);
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.groups[group].rounds[k], &registers, |x| {
                project(x, lane)
            });
        }
    }

    let registers = generate_round::<F, A, WIDTH, PAIRS, REGISTERS_LAST>(state, &constants[SPAN]);
    for (lane, call) in calls.iter_mut().enumerate() {
        let cells = &mut call.groups[group];
        write_round(&mut cells.last, &registers, |x| project(x, lane));
        if POST != 0 {
            // Only the new lower half: the upper half is the previous round's
            // lower half, which this group's own rounds already expressed (see
            // `columns::Group::post`).
            for (cell, x) in cells.post.iter_mut().zip(&state[..WIDTH / 2]) {
                cell.write(project(x, lane));
            }
        }
    }
}

/// Write one round's flattening registers, projecting the ring's values into `F`.
#[inline]
fn write_round<F: Field, A, const PAIRS: usize, const REGISTERS: usize>(
    round: &mut Round<MaybeUninit<F>, PAIRS, REGISTERS>,
    registers: &[[A; REGISTERS]; PAIRS],
    project: impl Fn(&A) -> F,
) {
    for (feistel, regs) in round.feistels.iter_mut().zip(registers) {
        for (cell, r) in feistel.0.iter_mut().zip(regs) {
            cell.write(project(r));
        }
    }
}

/// One round, fused, over any ring. Mirrors `air::eval_round`.
///
/// Returns the flattening cells the variant commits; the caller writes them,
/// together with the new lower half when `POST != 0`.
#[inline]
fn generate_round<
    F: Field,
    A: Algebra<F>,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
>(
    state: &mut [A; WIDTH],
    constants: &[[F; 2]; PAIRS],
) -> [[A; REGISTERS]; PAIRS] {
    let h = WIDTH / 2;

    let (z, extra) = linear::prologue(state.as_slice());

    let mut registers: [[A; REGISTERS]; PAIRS] =
        core::array::from_fn(|_| core::array::from_fn(|_| A::ZERO));
    let mut next: [A; WIDTH] = core::array::from_fn(|_| A::ZERO);

    for p in 0..PAIRS {
        let i = h - 2 - 2 * p;
        let (y, regs) =
            generate_feistel::<F, A, REGISTERS>(state[i].dup(), state[i + 1].dup(), &constants[p]);
        registers[p] = regs;
        for k in 0..2 {
            let mut out = state[h + 2 * p + k].dup() + y[k].dup();
            if p > 0 {
                out += z[k].dup();
            }
            if p == PAIRS - 1 {
                out += extra[k].dup();
            }
            next[2 * p + k] = out;
        }
    }
    linear::write_upper_half(&mut next, state.as_slice());
    *state = next;

    registers
}

/// One Feistel, over any ring. Mirrors `air::eval_feistel`.
///
/// Returns `(y4, y5)` and this variant's flattening cells: nothing at
/// `REGISTERS = 0`, `y1²` at `1`, `y1²` and `y3²` at `2`. Both squarings are
/// computed regardless — they are the round's arithmetic, not its bookkeeping —
/// and `REGISTERS` decides only which of them become columns.
#[inline]
fn generate_feistel<F: Field, A: Algebra<F>, const REGISTERS: usize>(
    x0: A,
    x1: A,
    constants: &[F; 2],
) -> ([A; 2], [A; REGISTERS]) {
    let y1 = x1 + constants[0];
    let (y1_sq, [y1_register]) = generate_power_map::<A, 2, 1>(y1.dup());
    let y2 = x0 + y1_sq.dup();
    let y3 = y1 + y2.dup() + constants[1];
    let (y3_sq, [y3_register]) = generate_power_map::<A, 2, 1>(y3.dup());
    let y4 = y2 + y3_sq.dup();
    let y5 = y3 + y4.dup();

    // POLICY §9: these are the cells `air::eval_feistel`'s `assert_eq`s pin. A
    // generator that wrote anything else here would produce a trace the AIR
    // rejects, which is exactly the point.
    let registers = core::array::from_fn(|k| match k {
        0 => y1_register.dup(),
        1 => y3_register.dup(),
        _ => unreachable!("REGISTERS <= 2, enforced by assert_layout"),
    });
    ([y4, y5], registers)
}
