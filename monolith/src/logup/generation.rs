//! Witness generation for the lookup-backed Monolith AIR.
//!
//! This is a witness producer, never evidence.  Every chunk is pinned by its
//! bus query and recomposition, every canonicity cell by `air::eval_canonicity`,
//! and every round state by the Bricks/Concrete/constant equality.

use core::mem::MaybeUninit;

use harness::gadgets::canonical_word::{
    WordKind, canonicity_witness, decompose, recompose_integer,
};
use harness::permutation::add_round_constants;
use harness::permutation::trace::{assert_full_table, fill_trace};
use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use tracing::instrument;

use crate::logup::air::MonolithLogupAir;
use crate::logup::columns::{MonolithLogupBar, MonolithLogupCols, MonolithLogupRound, num_cols};
use crate::logup::tables::FixedTableKind;

/// Generate one complete permutation per row.
#[instrument(name = "generate lookup Monolith trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    inputs: &[Vec<F>],
    air: &MonolithLogupAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_full_table(inputs.len());
    let ncols = num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>();
    // SAFETY: both closures write inputs, every Bar field, every post state and
    // the final round, i.e. every field declared by `MonolithLogupCols`.
    unsafe {
        fill_trace::<
            F,
            MonolithLogupCols<
                MaybeUninit<F>,
                WIDTH,
                NUM_FULL_ROUNDS,
                NUM_BARS,
                NUM_CHUNKS,
                CANONICAL_CELLS,
            >,
        >(
            inputs,
            WIDTH,
            ncols,
            ncols,
            extra_capacity_bits,
            |call, input| generate_call(call, input, air),
            |calls, inputs| generate_call_batch(calls, inputs, air),
        )
    }
}

fn table_kind(kind: WordKind, chunk: usize) -> FixedTableKind {
    match kind {
        WordKind::Mersenne31 if chunk == 3 => FixedTableKind::SevenBit,
        WordKind::Goldilocks | WordKind::Mersenne31 => FixedTableKind::Byte,
    }
}

fn bar_witness<F: PrimeField64, const NUM_CHUNKS: usize, const CANONICAL_CELLS: usize>(
    value: F,
    kind: WordKind,
) -> (MonolithLogupBar<F, NUM_CHUNKS, CANONICAL_CELLS>, F) {
    // Pinned by `air::eval_bar`'s recomposition against the incoming word.
    let input_bytes = decompose::<NUM_CHUNKS>(value.as_canonical_u64(), kind);
    // Pinned by the `(input chunk, output chunk)` query on the Bars bus.
    let output_bytes: [u8; NUM_CHUNKS] =
        core::array::from_fn(|chunk| table_kind(kind, chunk).output(input_bytes[chunk]));
    let output_integer = recompose_integer(&output_bytes, kind);
    assert!(
        output_integer < F::ORDER_U64,
        "the upstream Bars map must preserve canonical representatives"
    );

    let witness = MonolithLogupBar {
        input_chunks: input_bytes.map(F::from_u8),
        output_chunks: output_bytes.map(F::from_u8),
        // Both pinned by `air::eval_bar`'s two `eval_canonicity` calls.
        input_canonicity: canonicity_witness(&input_bytes, kind),
        output_canonicity: canonicity_witness(&output_bytes, kind),
    };
    (witness, F::from_u64(output_integer))
}

fn generate_round<
    F: PrimeField64,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    state: &mut [F; WIDTH],
    mds_matrix: &[[F; WIDTH]; WIDTH],
    round_constants: Option<&[F; WIDTH]>,
    kind: WordKind,
) -> MonolithLogupRound<F, WIDTH, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS> {
    let bars = core::array::from_fn(|index| {
        let (witness, output) = bar_witness(state[index], kind);
        state[index] = output;
        witness
    });
    for i in (1..WIDTH).rev() {
        state[i] += state[i - 1].square();
    }
    mds_multiply(state, mds_matrix);
    if let Some(constants) = round_constants {
        add_round_constants(state, constants);
    }
    MonolithLogupRound { bars, post: *state }
}

fn write_bar<A, F, const NUM_CHUNKS: usize, const CANONICAL_CELLS: usize>(
    target: &mut MonolithLogupBar<MaybeUninit<F>, NUM_CHUNKS, CANONICAL_CELLS>,
    source: &MonolithLogupBar<A, NUM_CHUNKS, CANONICAL_CELLS>,
    mut extract: impl FnMut(&A) -> F,
) where
    F: Copy,
{
    for (target, source) in target.input_chunks.iter_mut().zip(&source.input_chunks) {
        target.write(extract(source));
    }
    for (target, source) in target.output_chunks.iter_mut().zip(&source.output_chunks) {
        target.write(extract(source));
    }
    for (target, source) in target
        .input_canonicity
        .iter_mut()
        .zip(&source.input_canonicity)
    {
        target.write(extract(source));
    }
    for (target, source) in target
        .output_canonicity
        .iter_mut()
        .zip(&source.output_canonicity)
    {
        target.write(extract(source));
    }
}

fn write_round<
    A,
    F,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    target: &mut MonolithLogupRound<MaybeUninit<F>, WIDTH, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
    source: &MonolithLogupRound<A, WIDTH, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
    mut extract: impl FnMut(&A) -> F,
) where
    F: Copy,
{
    for (target, source) in target.bars.iter_mut().zip(&source.bars) {
        write_bar(target, source, &mut extract);
    }
    for (target, source) in target.post.iter_mut().zip(&source.post) {
        target.write(extract(source));
    }
}

fn generate_call<
    F: PrimeField64,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    call: &mut MonolithLogupCols<
        MaybeUninit<F>,
        WIDTH,
        NUM_FULL_ROUNDS,
        NUM_BARS,
        NUM_CHUNKS,
        CANONICAL_CELLS,
    >,
    input: &[F],
    air: &MonolithLogupAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
) {
    let mut state = core::array::from_fn(|i| input[i]);
    for (target, value) in call.inputs.iter_mut().zip(state) {
        target.write(value);
    }
    mds_multiply(&mut state, &air.mds_matrix);
    for round in 0..NUM_FULL_ROUNDS {
        let witness = generate_round(
            &mut state,
            &air.mds_matrix,
            Some(&air.round_constants[round]),
            air.kind,
        );
        write_round(&mut call.full_rounds[round], &witness, |value| *value);
    }
    let witness = generate_round(&mut state, &air.mds_matrix, None, air.kind);
    write_round(&mut call.final_round, &witness, |value| *value);
}

/// Fill `F::Packing::WIDTH` calls together: integer chunks are derived per
/// lane, while Bricks and Concrete run over packed field values.
fn generate_call_batch<
    F: PrimeField64,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
>(
    calls: &mut [MonolithLogupCols<
        MaybeUninit<F>,
        WIDTH,
        NUM_FULL_ROUNDS,
        NUM_BARS,
        NUM_CHUNKS,
        CANONICAL_CELLS,
    >],
    inputs: &[Vec<F>],
    air: &MonolithLogupAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>,
) {
    debug_assert_eq!(calls.len(), F::Packing::WIDTH);
    for (call, input) in calls.iter_mut().zip(inputs) {
        for (target, &value) in call.inputs.iter_mut().zip(input) {
            target.write(value);
        }
    }

    let mut state: [F::Packing; WIDTH] =
        core::array::from_fn(|i| F::Packing::from_fn(|lane| inputs[lane][i]));
    mds_multiply(&mut state, &air.mds_matrix);

    for round_index in 0..=NUM_FULL_ROUNDS {
        let bars = core::array::from_fn(|word| {
            let lane_witnesses = (0..F::Packing::WIDTH)
                .map(|lane| {
                    bar_witness::<F, NUM_CHUNKS, CANONICAL_CELLS>(
                        state[word].extract(lane),
                        air.kind,
                    )
                })
                .collect::<Vec<_>>();
            state[word] = F::Packing::from_fn(|lane| lane_witnesses[lane].1);
            MonolithLogupBar {
                input_chunks: core::array::from_fn(|chunk| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.input_chunks[chunk])
                }),
                output_chunks: core::array::from_fn(|chunk| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.output_chunks[chunk])
                }),
                input_canonicity: core::array::from_fn(|cell| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.input_canonicity[cell])
                }),
                output_canonicity: core::array::from_fn(|cell| {
                    F::Packing::from_fn(|lane| lane_witnesses[lane].0.output_canonicity[cell])
                }),
            }
        });
        for i in (1..WIDTH).rev() {
            state[i] += state[i - 1].square();
        }
        mds_multiply(&mut state, &air.mds_matrix);
        if round_index < NUM_FULL_ROUNDS {
            add_round_constants(&mut state, &air.round_constants[round_index]);
        }
        let witness = MonolithLogupRound { bars, post: state };
        for (lane, call) in calls.iter_mut().enumerate() {
            let target = if round_index < NUM_FULL_ROUNDS {
                &mut call.full_rounds[round_index]
            } else {
                &mut call.final_round
            };
            write_round(target, &witness, |value| value.extract(lane));
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::Goldilocks;
    use p3_mersenne_31::Mersenne31;
    use p3_monolith::{MonolithBarsGoldilocks, MonolithBarsM31};

    use harness::gadgets::canonical_word::WordKind;

    use super::bar_witness;

    /// The chunk/canonicity halves are the shared gadget's, tested there. What
    /// is Monolith's is that a Bar witness is the upstream Bars map applied
    /// chunkwise, including Mersenne-31's separate high seven-bit S-box.
    #[test]
    fn a_bar_witness_is_the_upstream_map_applied_chunkwise() {
        for integer in [0, 1, 255, 256, Goldilocks::ORDER_U64 - 1] {
            let value = Goldilocks::from_u64(integer);
            let (witness, output) = bar_witness::<Goldilocks, 8, 2>(value, WordKind::Goldilocks);
            for chunk in 0..8 {
                let input = integer.to_le_bytes()[chunk];
                assert_eq!(witness.input_chunks[chunk], Goldilocks::from_u8(input));
                assert_eq!(
                    witness.output_chunks[chunk],
                    Goldilocks::from_u8(MonolithBarsGoldilocks::<8>::bar(u64::from(input)) as u8)
                );
            }
            assert_eq!(
                output,
                Goldilocks::from_u64(u64::from_le_bytes(core::array::from_fn(|chunk| {
                    MonolithBarsGoldilocks::<8>::bar(u64::from(integer.to_le_bytes()[chunk])) as u8
                })))
            );
        }

        for integer in [0, 1, 255, Mersenne31::ORDER_U64 - 1] {
            let (witness, _) = bar_witness::<Mersenne31, 4, 1>(
                Mersenne31::from_u64(integer),
                WordKind::Mersenne31,
            );
            let bytes = integer.to_le_bytes();
            for (output, &byte) in witness.output_chunks.iter().zip(&bytes[..3]) {
                assert_eq!(*output, Mersenne31::from_u8(MonolithBarsM31::s_box(byte)));
            }
            assert_eq!(
                witness.output_chunks[3],
                Mersenne31::from_u8(MonolithBarsM31::final_s_box(bytes[3] & 0x7f))
            );
        }
    }
}
