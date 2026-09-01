//! The witness producer — never evidence.
//!
//! POLICY §6: a free `generate_trace_rows(inputs, constants, extra_capacity_bits)`
//! and its vectorized twin — `MaybeUninit` columns via `align_to_mut`,
//! `par_chunks_mut` over `F::Packing::WIDTH`. It takes inputs and never invents
//! them, and it forwards `extra_capacity_bits` so the prover's LDE happens in
//! place instead of reallocating a trace this file could have sized correctly.
//!
//! POLICY §9: nothing written here is evidence of anything. Every cell it writes
//! is a cell a malicious prover could have written differently, so each one needs
//! a constraint in `air.rs` naming it — and the negative test that corrupts one
//! cell and watches the proof fail is what checks that claim rather than restating
//! it. Here that is one cell per Feistel per round, and the constraint is the
//! `commit_state` call in [`air::eval_round`](crate::half_commit::air).
//!
//! # Kept parallel to `air.rs`
//!
//! Same order, same names, same per-round split: `generate_round` against
//! `eval_round`, `generate_feistel` against `eval_feistel`, `M_IO` at both ends
//! of `generate_call` against `eval`. The one structural difference is that the
//! round arithmetic here is generic over the ring rather than over the builder,
//! which is what lets the same code run over `F` and over `F::Packing` — the
//! scalar path and the SIMD path are the same round function, and only the cell
//! *writing* differs.
//!
//! # Values are values: there is no differencing on this side
//!
//! `air.rs`'s one deliberate departure from the natural expression — writing the
//! round's even output as `out[2p+1] − y3 + (affine)` rather than as `y4` — has no
//! counterpart here, and must not grow one. A generator computes the permutation:
//! it forms `y4` and `y5` and writes the whole new state into `state`, exactly as
//! [`full_commit::generation`](crate::full_commit::generation) does, and the cells
//! it commits are a *strided read* of that state and nothing more. The two forms
//! agree on the variety the committing constraint cuts out (POLICY §9), which is
//! precisely the set of traces this file produces, so the AIR accepts them; a
//! generator that reproduced the differenced form instead would be asserting the
//! algebra rather than being checked by it.
//!
//! What that buys is a generator *simpler* than `full_commit`'s: no register
//! array, no `REGISTERS`, no `POST`, no groups. The whole per-round write is
//! `write_round`'s three lines.

use core::mem::MaybeUninit;

use harness::permutation::trace::{
    assert_full_table, assert_full_vectorized_table, fill_trace as fill_calls,
};
use p3_field::{Algebra, Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::half_commit::columns::{HalfCommitCols, Round, assert_layout, num_cols};
use crate::linear;

/// One row per call.
///
/// # Panics
///
/// If `inputs.len()` is not a power of two, or an input is not `WIDTH` long.
#[instrument(name = "generate half-commitment pSquareHash trace", skip_all)]
pub fn generate_trace_rows<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const ROUNDS: usize,
>(
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, PAIRS, ROUNDS>();
    fill_trace::<F, WIDTH, PAIRS, ROUNDS>(inputs, constants, ncols, extra_capacity_bits)
}

/// `VECTOR_LEN` calls per row.
///
/// # Panics
///
/// If `inputs.len()` is not `VECTOR_LEN` times a power of two, or an input is not
/// `WIDTH` long.
#[instrument(
    name = "generate vectorized half-commitment pSquareHash trace",
    skip_all
)]
pub fn generate_vectorized_trace_rows<
    F: Field,
    const WIDTH: usize,
    const PAIRS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), VECTOR_LEN);
    let ncols = num_cols::<WIDTH, PAIRS, ROUNDS>() * VECTOR_LEN;
    fill_trace::<F, WIDTH, PAIRS, ROUNDS>(inputs, constants, ncols, extra_capacity_bits)
}

/// Allocate a trace of `ncols` columns holding `inputs.len()` calls, and fill it.
///
/// The allocation, the alignment assertions and the packing-width dispatch are
/// [`harness::permutation::trace::fill_trace`]'s — including the identity the two
/// public generators rest on, that `VECTOR_LEN` calls side by side in one row is
/// the same byte layout as `VECTOR_LEN` rows of one call. What is this variant's
/// is the pair of fill functions below.
///
/// The packed path is the reason a construction owns its generator at all: one
/// field operation per state element covers every lane, so trace generation costs
/// a `Packing::WIDTH`-th of the scalar path — and generation is timed separately
/// from proving (POLICY §11) precisely because it parallelizes differently.
fn fill_trace<F: Field, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>(
    inputs: &[Vec<F>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, PAIRS);

    // SAFETY: `generate_call` and `generate_call_batch` write every field of
    // `HalfCommitCols` — `inputs`, each round's `odd`, and `outputs` — in layout
    // order, so the buffer is fully initialized. There is no conditional field:
    // unlike `full_commit`, this layout has no variant parameter that can make a
    // block empty, so "every field" is the same three every time.
    unsafe {
        fill_calls::<F, HalfCommitCols<MaybeUninit<F>, WIDTH, PAIRS, ROUNDS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, PAIRS, ROUNDS>(),
            ncols,
            extra_capacity_bits,
            |call, input| {
                generate_call::<F, WIDTH, PAIRS, ROUNDS>(call, input, constants);
            },
            |calls, inputs| {
                generate_call_batch::<F, WIDTH, PAIRS, ROUNDS>(calls, inputs, constants);
            },
        )
    }
}

/// One call, scalar. Mirrors `air::eval`.
fn generate_call<F: Field, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>(
    call: &mut HalfCommitCols<MaybeUninit<F>, WIDTH, PAIRS, ROUNDS>,
    input: &[F],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
) {
    let mut state: [F; WIDTH] = core::array::from_fn(|i| input[i]);
    for (cell, x) in call.inputs.iter_mut().zip(state) {
        cell.write(x);
    }

    linear::m_io(state.as_mut_slice());
    generate_rounds::<F, F, WIDTH, PAIRS, ROUNDS>(
        &mut state,
        core::slice::from_mut(call),
        constants,
        |x, _| *x,
    );
    linear::m_io(state.as_mut_slice());

    for (cell, x) in call.outputs.iter_mut().zip(state) {
        cell.write(x);
    }
}

/// `F::Packing::WIDTH` calls at once, running the round function over
/// `F::Packing`. Mirrors [`generate_call`] step for step; only the writes differ,
/// because each lane owns its own column struct.
fn generate_call_batch<F: Field, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>(
    calls: &mut [HalfCommitCols<MaybeUninit<F>, WIDTH, PAIRS, ROUNDS>],
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
    generate_rounds::<F, F::Packing, WIDTH, PAIRS, ROUNDS>(
        &mut state,
        calls,
        constants,
        |packed, lane| packed.extract(lane),
    );
    linear::m_io(state.as_mut_slice());

    for (lane, call) in calls.iter_mut().enumerate() {
        for (cell, x) in call.outputs.iter_mut().zip(&state) {
            cell.write(x.extract(lane));
        }
    }
}

/// Every round, in order, over every lane at once. Mirrors `air::eval`'s round
/// loop — which is flat, because this layout has no grouping: every round commits.
///
/// The round arithmetic runs **once** over `A`, which is `F` on the scalar path
/// and `F::Packing` on the SIMD one; `project` is the identity there and a lane
/// extraction here. One writer for both paths is what stops them committing
/// different cells for the same variant, and it is why the packed path cannot
/// drift into computing a different round function than the scalar one.
///
/// The round's cells are read off `state` *after* the round, which is the whole
/// content of this layout: the constraint side reconstructs the even half from
/// them, so the write below is the only place that fixes *which* half of the round
/// output becomes a column. `air.rs` and `columns.rs` both argue why it must be
/// the odd one.
#[inline]
fn generate_rounds<
    F: Field,
    A: Algebra<F>,
    const WIDTH: usize,
    const PAIRS: usize,
    const ROUNDS: usize,
>(
    state: &mut [A; WIDTH],
    calls: &mut [HalfCommitCols<MaybeUninit<F>, WIDTH, PAIRS, ROUNDS>],
    constants: &[[[F; 2]; PAIRS]; ROUNDS],
    project: impl Fn(&A, usize) -> F,
) {
    for (r, constants) in constants.iter().enumerate() {
        generate_round::<F, A, WIDTH, PAIRS>(state, constants);
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.rounds[r], state, |x| project(x, lane));
        }
    }
}

/// Write one round's committed cells: the odd elements of the new lower half,
/// projected into `F`.
///
/// `state[1], state[3], …, state[h−1]` — `PAIRS` of them, since `h = 2·PAIRS`, so
/// the last one written is the last lower-half word. That is the strided read
/// `columns::Round::odd` names as `out[2p+1]`, and the indexing is the *only*
/// thing that decides the committed parity on this side: write `2 * p` here and
/// every known-answer test still passes while the AIR's degree claim silently
/// becomes false. `tests/air.rs` replays the permutation against these cells for
/// exactly that reason.
///
/// POLICY §9: each cell written here is prover-chosen and worth nothing until an
/// `assert_eq` ties it to the state. That assertion is the `commit_state` call in
/// `air::eval_round`, and it carries more weight than any other in this crate —
/// the round's even output is an *expression in these cells*, so a `odd` the
/// prover picked freely would leave both halves of the round output arbitrary.
#[inline]
fn write_round<F: Field, A, const WIDTH: usize, const PAIRS: usize>(
    round: &mut Round<MaybeUninit<F>, PAIRS>,
    state: &[A; WIDTH],
    project: impl Fn(&A) -> F,
) {
    for (p, cell) in round.odd.iter_mut().enumerate() {
        cell.write(project(&state[2 * p + 1]));
    }
}

/// One round, fused, over any ring. Mirrors `air::eval_round`.
///
/// Writes the whole new state — both halves, as values — and returns nothing: the
/// caller reads the cells it commits back off `state`. `full_commit`'s twin
/// returns the flattening registers instead, because there they are cells that do
/// not appear in the state; here every committed cell *is* a state word.
#[inline]
fn generate_round<F: Field, A: Algebra<F>, const WIDTH: usize, const PAIRS: usize>(
    state: &mut [A; WIDTH],
    constants: &[[F; 2]; PAIRS],
) {
    let h = WIDTH / 2;

    let (z, extra) = linear::prologue(state.as_slice());

    let mut next: [A; WIDTH] = core::array::from_fn(|_| A::ZERO);

    for p in 0..PAIRS {
        let i = h - 2 - 2 * p;
        let y = generate_feistel::<F, A>(state[i].dup(), state[i + 1].dup(), &constants[p]);
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
}

/// One Feistel, over any ring. Mirrors `air::eval_feistel`.
///
/// Returns `[y4, y5]`, the output pair, and nothing else: this variant commits no
/// cell inside a Feistel, so there is no register array and no `REGISTERS`
/// parameter to decide which squaring becomes a column.
///
/// `y4` *is* formed here where `air::eval_feistel` never forms it. That is not a
/// divergence between the two sides but the asymmetry the layout is built on: the
/// generator needs the even output as a **value** to continue the permutation, and
/// the AIR needs it as an **expression** whose degree it can afford — so one
/// computes it and the other differences it out. The module docs say why that is
/// sound in one direction only.
#[inline]
fn generate_feistel<F: Field, A: Algebra<F>>(x0: A, x1: A, constants: &[F; 2]) -> [A; 2] {
    let y1 = x1 + constants[0];
    let y2 = x0 + y1.dup().square();
    let y3 = y1 + y2.dup() + constants[1];
    let y4 = y2 + y3.dup().square();
    let y5 = y3 + y4.dup();
    [y4, y5]
}
