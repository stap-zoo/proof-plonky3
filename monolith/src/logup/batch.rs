//! The complete lookup statement: one Monolith call AIR and its fixed tables.
//!
//! The batch prover requires one Rust AIR type for all heterogeneous instances;
//! that type erasure is [`harness::lookup::LookupBatchAir`], shared with every
//! lookup-backed construction. What is Monolith's is only which tables a
//! variant needs, and how a call trace's chunks are counted into their
//! multiplicities.

use core::borrow::Borrow;

use p3_field::PrimeField64;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use harness::gadgets::canonical_word::WordKind;
use harness::lookup::{LookupBatch, LookupBatchAir, LookupGranularity, LookupPermutationAir};
use harness::permutation::Labels;

use super::air::MonolithLogupAir;
use super::columns::MonolithLogupCols;
use super::generation::generate_trace_rows;
use super::tables::FixedTableKind;

/// One AIR in a lookup-backed Monolith batch.
pub type MonolithBatchAir<A> = LookupBatchAir<A, FixedTableKind>;

/// AIRs and aligned witness traces for one complete LogUp statement.
pub type MonolithBatch<A, F> = LookupBatch<A, FixedTableKind, F>;

impl<
    F: PrimeField64 + Sync,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> MonolithLogupAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>
{
    fn fixed_table_kinds(&self) -> Vec<FixedTableKind> {
        match (self.kind, self.granularity()) {
            (WordKind::Goldilocks, LookupGranularity::Byte) => vec![FixedTableKind::Byte],
            (WordKind::Goldilocks, LookupGranularity::AdjacentPair) => {
                vec![FixedTableKind::BytePair]
            }
            (WordKind::Mersenne31, LookupGranularity::Byte) => {
                vec![FixedTableKind::Byte, FixedTableKind::SevenBit]
            }
            (WordKind::Mersenne31, LookupGranularity::AdjacentPair) => {
                vec![FixedTableKind::BytePair, FixedTableKind::ByteSevenPair]
            }
        }
    }

    /// AIR list for setup, without generating any witness.
    #[must_use]
    pub fn batch_airs(&self) -> Vec<MonolithBatchAir<Self>> {
        core::iter::once(MonolithBatchAir::Calls(self.clone()))
            .chain(
                self.fixed_table_kinds()
                    .into_iter()
                    .map(|kind| MonolithBatchAir::Table(kind.air())),
            )
            .collect()
    }

    /// Base log-heights aligned with [`Self::batch_airs`].
    #[must_use]
    pub fn batch_log_heights(&self, call_log_n: usize) -> Vec<usize> {
        core::iter::once(call_log_n)
            .chain(
                self.fixed_table_kinds()
                    .into_iter()
                    .map(|kind| kind.height().ilog2() as usize),
            )
            .collect()
    }

    /// Build the complete call-plus-tables witness from caller-supplied inputs.
    #[must_use]
    pub fn generate_batch(
        &self,
        inputs: &[Vec<F>],
        extra_capacity_bits: usize,
    ) -> MonolithBatch<Self, F> {
        let call_trace = generate_trace_rows(inputs, self, extra_capacity_bits);
        let table_kinds = self.fixed_table_kinds();
        let counts = self.count_table_inputs(&call_trace, &table_kinds);

        let mut airs = Vec::with_capacity(1 + table_kinds.len());
        let mut traces = Vec::with_capacity(1 + table_kinds.len());
        airs.push(MonolithBatchAir::Calls(self.clone()));
        traces.push(call_trace);
        for (kind, multiplicities) in table_kinds.into_iter().zip(counts) {
            let table = kind.air();
            traces.push(table.multiplicity_trace(&multiplicities, extra_capacity_bits));
            airs.push(MonolithBatchAir::Table(table));
        }
        MonolithBatch::new(airs, traces)
    }

    /// Measurement-only seeded form of [`Self::generate_batch`].
    #[must_use]
    pub fn generate_batch_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> MonolithBatch<Self, F>
    where
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let inputs = (0..num_calls)
            .map(|_| rng.sample::<[F; WIDTH], _>(StandardUniform).to_vec())
            .collect::<Vec<_>>();
        self.generate_batch(&inputs, extra_capacity_bits)
    }

    fn count_table_inputs(
        &self,
        trace: &RowMajorMatrix<F>,
        table_kinds: &[FixedTableKind],
    ) -> Vec<Vec<u32>> {
        let mut counts = table_kinds
            .iter()
            .map(|kind| vec![0u32; kind.height()])
            .collect::<Vec<_>>();
        for row in 0..trace.height() {
            let row = trace.row_slice(row).expect("call row exists");
            let cols: &MonolithLogupCols<
                F,
                WIDTH,
                NUM_FULL_ROUNDS,
                NUM_BARS,
                NUM_CHUNKS,
                CANONICAL_CELLS,
            > = (*row).borrow();
            for round in cols
                .full_rounds
                .iter()
                .chain(core::iter::once(&cols.final_round))
            {
                for bar in &round.bars {
                    match self.granularity() {
                        LookupGranularity::Byte => {
                            for (chunk, value) in bar.input_chunks.iter().enumerate() {
                                let table = usize::from(
                                    self.kind == WordKind::Mersenne31 && chunk == NUM_CHUNKS - 1,
                                );
                                let index = value.as_canonical_u64() as usize;
                                counts[table][index] = counts[table][index]
                                    .checked_add(1)
                                    .expect("table multiplicity fits u32");
                            }
                        }
                        LookupGranularity::AdjacentPair => {
                            for chunk in (0..NUM_CHUNKS).step_by(2) {
                                let table = usize::from(
                                    self.kind == WordKind::Mersenne31 && chunk == NUM_CHUNKS - 2,
                                );
                                let lo = bar.input_chunks[chunk].as_canonical_u64() as usize;
                                let hi = bar.input_chunks[chunk + 1].as_canonical_u64() as usize;
                                let index = lo | (hi << 8);
                                counts[table][index] = counts[table][index]
                                    .checked_add(1)
                                    .expect("table multiplicity fits u32");
                            }
                        }
                    }
                }
            }
        }
        counts
    }
}

impl<
    SC: StarkGenericConfig,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const NUM_CHUNKS: usize,
    const CANONICAL_CELLS: usize,
> LookupPermutationAir<Val<SC>, SC>
    for MonolithLogupAir<Val<SC>, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, NUM_CHUNKS, CANONICAL_CELLS>
where
    Val<SC>: PrimeField64 + Sync,
    StandardUniform: Distribution<[Val<SC>; WIDTH]>,
    MonolithBatchAir<Self>: harness::lookup::LookupAir<SC>,
{
    type BatchAir = MonolithBatchAir<Self>;

    fn labels(&self) -> Labels {
        Labels {
            state_width: WIDTH,
            calls_per_row: 1,
            rows_per_call: 1,
            rounds: NUM_FULL_ROUNDS + 1,
            sbox_degree: 0,
            sbox_registers: 0,
            max_constraint_degree: self.packing().max_constraint_degree(),
        }
    }

    fn batch_airs(&self) -> Vec<Self::BatchAir> {
        self.batch_airs()
    }

    fn batch_log_heights(&self, call_log_n: usize) -> Vec<usize> {
        self.batch_log_heights(call_log_n)
    }

    fn generate_batch_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> Vec<RowMajorMatrix<Val<SC>>> {
        self.generate_batch_seeded(num_calls, seed, extra_capacity_bits)
            .traces
    }
}
