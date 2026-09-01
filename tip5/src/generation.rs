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

use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, assert_full_vectorized_table, fill_trace};
use p3_field::{PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::columns::{
    MATCH_FLAGS, PowerWord, QUOTIENT_BITS, Round, SplitWord, Tip5Cols, assert_layout, num_cols,
};
use crate::params::{MONT_R, ROUNDS, SPLIT_WORDS, Tip5Params};

/// One row per call.
#[instrument(name = "generate Tip5 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    inputs: &[Vec<F>],
    params: &Tip5Params<F, WIDTH>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    fill_calls::<F, WIDTH, POWER_WORDS, REGISTERS>(
        inputs,
        params,
        num_cols::<WIDTH, POWER_WORDS, REGISTERS>(),
        extra_capacity_bits,
    )
}

/// `LANES` independent calls per row.
#[instrument(name = "generate vectorized Tip5 trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
    const LANES: usize,
>(
    inputs: &[Vec<F>],
    params: &Tip5Params<F, WIDTH>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_vectorized_table(inputs.len(), LANES);
    fill_calls::<F, WIDTH, POWER_WORDS, REGISTERS>(
        inputs,
        params,
        num_cols::<WIDTH, POWER_WORDS, REGISTERS>() * LANES,
        extra_capacity_bits,
    )
}

fn fill_calls<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    inputs: &[Vec<F>],
    params: &Tip5Params<F, WIDTH>,
    ncols: usize,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_layout(WIDTH, POWER_WORDS, REGISTERS);
    // SAFETY: the scalar and packed generators write inputs, every field of all
    // five rounds, and outputs.
    unsafe {
        fill_trace::<F, Tip5Cols<MaybeUninit<F>, WIDTH, POWER_WORDS, REGISTERS>>(
            inputs,
            WIDTH,
            num_cols::<WIDTH, POWER_WORDS, REGISTERS>(),
            ncols,
            extra_capacity_bits,
            |call, input| generate_call(call, input, params),
            |calls, inputs| generate_call_batch(calls, inputs, params),
        )
    }
}

fn bits<F: PrimeField64, const N: usize>(value: u64) -> [F; N] {
    core::array::from_fn(|i| F::from_bool(((value >> i) & 1) != 0))
}

fn canonical_flags<F: PrimeField64>(value: u64) -> [F; MATCH_FLAGS] {
    let mut result = [F::ZERO; MATCH_FLAGS];
    let mut prev = true;
    let mut pending = None;
    let mut flag = 0;
    for i in (0..64).rev() {
        if ((F::ORDER_U64 >> i) & 1) == 1 {
            let bit = ((value >> i) & 1) != 0;
            if let Some(first) = pending.take() {
                prev = prev && first && bit;
                result[flag] = F::from_bool(prev);
                flag += 1;
            } else {
                pending = Some(bit);
            }
        }
    }
    debug_assert_eq!(flag, MATCH_FLAGS);
    result
}

/// Generate every witness of one split word. Each field is named beside its
/// binding constraint in `air::eval_split_word`.
fn split_word<F: PrimeField64, const WIDTH: usize>(
    value: F,
    params: &Tip5Params<F, WIDTH>,
) -> (SplitWord<F>, F) {
    let mont = (value * F::from_u64(MONT_R)).as_canonical_u64();
    let input_bits = core::array::from_fn(|byte| bits((mont >> (8 * byte)) & 0xff));
    let match_flags = canonical_flags::<F>(mont);
    let mut transformed = 0u64;
    let output_bits = core::array::from_fn(|byte| {
        let digit = ((mont >> (8 * byte)) & 0xff) as usize;
        let output = u64::from(params.lut[digit]);
        transformed |= output << (8 * byte);
        bits(output)
    });
    let quotient_bits = core::array::from_fn(|byte| {
        let digit = (mont >> (8 * byte)) & 0xff;
        let output = u64::from(params.lut[digit as usize]);
        let shifted = digit + 1;
        let quotient = (shifted * shifted * shifted - 1 - output) / 257;
        debug_assert!(quotient < (1 << QUOTIENT_BITS));
        bits(quotient)
    });
    let output = F::from_u64(transformed) * F::from_u64(MONT_R).inverse();
    (
        SplitWord {
            input_bits,
            match_flags,
            output_bits,
            quotient_bits,
        },
        output,
    )
}

/// One round, kept in the same order as `air::eval_round`.
fn generate_round<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    params: &Tip5Params<F, WIDTH>,
    round_index: usize,
) -> Round<F, POWER_WORDS, REGISTERS> {
    let split = core::array::from_fn(|i| {
        let (witness, output) = split_word(state[i], params);
        state[i] = output;
        witness
    });
    let powers = core::array::from_fn(|i| {
        let index = i + SPLIT_WORDS;
        let (output, registers) = generate_power_map::<F, 7, REGISTERS>(state[index]);
        state[index] = output;
        PowerWord { registers, output }
    });
    mds_multiply(state, &params.m);
    add_round_constants(state, &params.rcons[round_index]);
    Round { split, powers }
}

fn write_round<A, F, const POWER_WORDS: usize, const REGISTERS: usize>(
    target: &mut Round<MaybeUninit<F>, POWER_WORDS, REGISTERS>,
    witness: &Round<A, POWER_WORDS, REGISTERS>,
    mut extract: impl FnMut(&A) -> F,
) where
    F: Copy,
{
    for (target, source) in target.split.iter_mut().zip(&witness.split) {
        for (target, source) in target
            .input_bits
            .iter_mut()
            .flatten()
            .zip(source.input_bits.iter().flatten())
        {
            target.write(extract(source));
        }
        for (target, source) in target.match_flags.iter_mut().zip(&source.match_flags) {
            target.write(extract(source));
        }
        for (target, source) in target
            .output_bits
            .iter_mut()
            .flatten()
            .zip(source.output_bits.iter().flatten())
        {
            target.write(extract(source));
        }
        for (target, source) in target
            .quotient_bits
            .iter_mut()
            .flatten()
            .zip(source.quotient_bits.iter().flatten())
        {
            target.write(extract(source));
        }
    }
    for (target, source) in target.powers.iter_mut().zip(&witness.powers) {
        for (target, source) in target.registers.iter_mut().zip(&source.registers) {
            target.write(extract(source));
        }
        target.output.write(extract(&source.output));
    }
}

fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    call: &mut Tip5Cols<MaybeUninit<F>, WIDTH, POWER_WORDS, REGISTERS>,
    input: &[F],
    params: &Tip5Params<F, WIDTH>,
) {
    let mut state = core::array::from_fn(|i| input[i]);
    for (cell, value) in call.inputs.iter_mut().zip(state) {
        cell.write(value);
    }
    for round in 0..ROUNDS {
        let witness = generate_round(&mut state, params, round);
        write_round(&mut call.rounds[round], &witness, |value| *value);
    }
    for (cell, value) in call.outputs.iter_mut().zip(state) {
        cell.write(value);
    }
}

/// `F::Packing::WIDTH` calls at once. Integer byte decomposition is performed
/// lane-wise, then its witnesses are packed; the seventh powers and dense
/// linear layer run over `F::Packing`.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    calls: &mut [Tip5Cols<MaybeUninit<F>, WIDTH, POWER_WORDS, REGISTERS>],
    inputs: &[Vec<F>],
    params: &Tip5Params<F, WIDTH>,
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
    for round_index in 0..ROUNDS {
        let split = core::array::from_fn(|word| {
            let lane_witnesses = (0..F::Packing::WIDTH)
                .map(|lane| split_word(state[word].extract(lane), params))
                .collect::<Vec<_>>();
            state[word] = F::Packing::from_fn(|lane| lane_witnesses[lane].1);
            SplitWord {
                input_bits: core::array::from_fn(|byte| {
                    core::array::from_fn(|bit| {
                        F::Packing::from_fn(|lane| lane_witnesses[lane].0.input_bits[byte][bit])
                    })
                }),
                match_flags: core::array::from_fn(|flag| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.match_flags[flag])
                }),
                output_bits: core::array::from_fn(|byte| {
                    core::array::from_fn(|bit| {
                        F::Packing::from_fn(|lane| lane_witnesses[lane].0.output_bits[byte][bit])
                    })
                }),
                quotient_bits: core::array::from_fn(|byte| {
                    core::array::from_fn(|bit| {
                        F::Packing::from_fn(|lane| lane_witnesses[lane].0.quotient_bits[byte][bit])
                    })
                }),
            }
        });
        let powers = core::array::from_fn(|i| {
            let index = i + SPLIT_WORDS;
            let (output, registers) = generate_power_map::<F::Packing, 7, REGISTERS>(state[index]);
            state[index] = output;
            PowerWord { registers, output }
        });
        mds_multiply(&mut state, &params.m);
        add_round_constants(&mut state, &params.rcons[round_index]);
        let witness = Round { split, powers };
        for (lane, call) in calls.iter_mut().enumerate() {
            write_round(&mut call.rounds[round_index], &witness, |value| {
                value.extract(lane)
            });
        }
    }
    for (lane, call) in calls.iter_mut().enumerate() {
        for (i, cell) in call.outputs.iter_mut().enumerate() {
            cell.write(state[i].extract(lane));
        }
    }
}
