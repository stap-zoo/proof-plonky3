//! Witness generation for the lookup-backed Tip5 AIR.
//!
//! This is a witness producer, never evidence (POLICY §9). Every input byte is
//! pinned by its bus query and by the recomposition against `mont_R * x`, every
//! output byte by the same query, every canonicity cell by
//! `air::eval_split_word`'s `eval_canonicity`, every seventh-power register by
//! `crate::air::eval_power_word`, and the boundary cells by the input columns
//! and the closing `assert_state_eq`.
//!
//! It is kept line-for-line parallel with `air.rs`: same order, same helper
//! names, same per-round split.

use core::mem::MaybeUninit;

use harness::gadgets::canonical_word::{canonicity_witness, decompose, recompose_integer};
use harness::gadgets::power_map::generate_power_map;
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, fill_trace};
use p3_field::{PackedValue, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::columns::{PowerWord, assert_layout};
use crate::logup::columns::{CANONICAL_CELLS, KIND, Round, SplitWord, Tip5LogupCols, num_cols};
use crate::params::{BYTES, MONT_R, ROUNDS, SPLIT_WORDS, Tip5Params};

/// Generate one complete permutation per row.
#[instrument(name = "generate lookup Tip5 trace", skip_all)]
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
    assert_layout(WIDTH, POWER_WORDS, REGISTERS);
    let ncols = num_cols::<WIDTH, POWER_WORDS, REGISTERS>();
    // SAFETY: both closures write inputs, every field of all five rounds, and
    // outputs — i.e. every field declared by `Tip5LogupCols`.
    unsafe {
        fill_trace::<F, Tip5LogupCols<MaybeUninit<F>, WIDTH, POWER_WORDS, REGISTERS>>(
            inputs,
            WIDTH,
            ncols,
            ncols,
            extra_capacity_bits,
            |call, input| generate_call(call, input, params),
            |calls, inputs| generate_call_batch(calls, inputs, params),
        )
    }
}

/// Every witness of one split word, and the word the S-box returns.
fn split_word<F: PrimeField64, const WIDTH: usize>(
    value: F,
    params: &Tip5Params<F, WIDTH>,
) -> (SplitWord<F>, F) {
    let mont = (value * F::from_u64(MONT_R)).as_canonical_u64();
    // Pinned by `air::eval_split_word`'s recomposition against mont_R * x, and
    // range-bound by the byte query.
    let input_bytes = decompose::<BYTES>(mont, KIND);
    // Pinned by the same `(input byte, output byte)` query.
    let output_bytes: [u8; BYTES] =
        core::array::from_fn(|byte| params.lut[input_bytes[byte] as usize]);
    let transformed = recompose_integer(&output_bytes, KIND);
    (
        SplitWord {
            input_bytes: input_bytes.map(F::from_u8),
            output_bytes: output_bytes.map(F::from_u8),
            // Pinned by `air::eval_split_word`'s `eval_canonicity`.
            input_canonicity: canonicity_witness::<F, CANONICAL_CELLS>(&input_bytes, KIND),
        },
        F::from_u64(transformed) * F::from_u64(MONT_R).inverse(),
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
        for (target, source) in target.input_bytes.iter_mut().zip(&source.input_bytes) {
            target.write(extract(source));
        }
        for (target, source) in target.output_bytes.iter_mut().zip(&source.output_bytes) {
            target.write(extract(source));
        }
        for (target, source) in target
            .input_canonicity
            .iter_mut()
            .zip(&source.input_canonicity)
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
    call: &mut Tip5LogupCols<MaybeUninit<F>, WIDTH, POWER_WORDS, REGISTERS>,
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

/// `F::Packing::WIDTH` calls at once. Byte decomposition is performed lane-wise
/// and its witnesses are then packed; the seventh powers and the dense linear
/// layer run over `F::Packing`.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
>(
    calls: &mut [Tip5LogupCols<MaybeUninit<F>, WIDTH, POWER_WORDS, REGISTERS>],
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
                input_bytes: core::array::from_fn(|byte| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.input_bytes[byte])
                }),
                output_bytes: core::array::from_fn(|byte| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.output_bytes[byte])
                }),
                input_canonicity: core::array::from_fn(|cell| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.input_canonicity[cell])
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

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;

    use crate::native::Tip5;
    use crate::params::{MONT_R, tip4_prime};

    use super::split_word;

    /// The byte/canonicity halves are the shared gadget's, tested there. What
    /// is Tip5's is that a split-word witness is `params::lookup` applied
    /// bytewise to `mont_R * x`, agreeing with the native oracle it replaces.
    #[test]
    fn a_split_word_witness_is_the_native_split_sbox() {
        let params = tip4_prime::<Goldilocks>();
        let native = Tip5::new(&params);
        for integer in [0, 1, 255, 256, 0xffff_ffff, Goldilocks::ORDER_U64 - 1] {
            let value = Goldilocks::from_u64(integer);
            let (witness, output) = split_word(value, &params);
            assert_eq!(output, native.split_sbox(value));

            let mont = (value * Goldilocks::from_u64(MONT_R)).as_canonical_u64();
            for byte in 0..8 {
                let input = (mont >> (8 * byte)) & 0xff;
                assert_eq!(witness.input_bytes[byte], Goldilocks::from_u64(input));
                assert_eq!(
                    witness.output_bytes[byte],
                    Goldilocks::from_u8(crate::params::lookup(input as u8))
                );
            }
        }
    }

    /// The generator is a witness producer, but it must still reproduce the
    /// oracle the lookup-free arithmetization is validated against.
    #[test]
    fn a_generated_call_is_the_native_permutation() {
        let params = tip4_prime::<Goldilocks>();
        let native = Tip5::new(&params);
        let inputs = (0..8)
            .map(|row| {
                (0..12)
                    .map(|column| Goldilocks::from_usize(row * 12 + column))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let trace = super::generate_trace_rows::<Goldilocks, 12, 8, 0>(&inputs, &params, 0);
        let ncols = super::num_cols::<12, 8, 0>();
        for (row, input) in inputs.iter().enumerate() {
            let mut expected = <[Goldilocks; 12]>::try_from(input.as_slice()).unwrap();
            native.permute_mut(&mut expected);
            let outputs = &trace.values[(row + 1) * ncols - 12..(row + 1) * ncols];
            assert_eq!(outputs, expected);
        }
    }
}
