//! The complete lookup statement: one Tip5 call AIR and its fixed table.
//!
//! The batch prover requires one Rust AIR type for all heterogeneous instances;
//! that type erasure is [`harness::lookup::LookupBatchAir`], shared with every
//! lookup-backed construction. What is Tip5's is only which table a variant
//! needs — one, at either granularity — and how a call trace's input bytes are
//! counted into its multiplicities.

use core::borrow::Borrow;

use p3_field::PrimeField64;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use harness::lookup::{LookupBatch, LookupBatchAir, LookupGranularity, LookupPermutationAir};
use harness::permutation::Labels;

use super::air::{Tip5LogupAir, max_constraint_degree};
use super::columns::Tip5LogupCols;
use super::generation::generate_trace_rows;
use super::tables::FixedTableKind;
use crate::params::{BYTES, ROUNDS};

/// One AIR in a lookup-backed Tip5 batch.
pub type Tip5BatchAir<A> = LookupBatchAir<A, FixedTableKind>;

/// AIRs and aligned witness traces for one complete LogUp statement.
pub type Tip5Batch<A, F> = LookupBatch<A, FixedTableKind, F>;

impl<F: PrimeField64 + Sync, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>
    Tip5LogupAir<F, WIDTH, POWER_WORDS, REGISTERS>
{
    /// The single fixed table this variant queries.
    ///
    /// Unlike Monolith, whose Mersenne-31 word has a high seven-bit chunk on
    /// its own bus, every Tip5 split byte is a full byte, so one table answers
    /// all of them at either granularity.
    fn fixed_table_kinds(&self) -> Vec<FixedTableKind> {
        match self.granularity() {
            LookupGranularity::Byte => vec![FixedTableKind::Byte],
            LookupGranularity::AdjacentPair => vec![FixedTableKind::BytePair],
        }
    }

    /// AIR list for setup, without generating any witness.
    #[must_use]
    pub fn batch_airs(&self) -> Vec<Tip5BatchAir<Self>> {
        core::iter::once(Tip5BatchAir::Calls(self.clone()))
            .chain(
                self.fixed_table_kinds()
                    .into_iter()
                    .map(|kind| Tip5BatchAir::Table(kind.air())),
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

    /// Build the complete call-plus-table witness from caller-supplied inputs.
    #[must_use]
    pub fn generate_batch(
        &self,
        inputs: &[Vec<F>],
        extra_capacity_bits: usize,
    ) -> Tip5Batch<Self, F> {
        let call_trace = generate_trace_rows::<F, WIDTH, POWER_WORDS, REGISTERS>(
            inputs,
            &self.params,
            extra_capacity_bits,
        );
        let table_kinds = self.fixed_table_kinds();
        let counts = self.count_table_inputs(&call_trace, &table_kinds);

        let mut airs = Vec::with_capacity(1 + table_kinds.len());
        let mut traces = Vec::with_capacity(1 + table_kinds.len());
        airs.push(Tip5BatchAir::Calls(self.clone()));
        traces.push(call_trace);
        for (kind, multiplicities) in table_kinds.into_iter().zip(counts) {
            let table = kind.air();
            traces.push(table.multiplicity_trace(&multiplicities, extra_capacity_bits));
            airs.push(Tip5BatchAir::Table(table));
        }
        Tip5Batch::new(airs, traces)
    }

    /// Measurement-only seeded form of [`Self::generate_batch`].
    #[must_use]
    pub fn generate_batch_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> Tip5Batch<Self, F>
    where
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let inputs = (0..num_calls)
            .map(|_| rng.sample::<[F; WIDTH], _>(StandardUniform).to_vec())
            .collect::<Vec<_>>();
        self.generate_batch(&inputs, extra_capacity_bits)
    }

    /// Count the call trace's input bytes into fixed-table multiplicities.
    ///
    /// Only inputs are counted: a dishonest output byte must leave the
    /// `(input, output)` bus unbalanced rather than steering the table witness
    /// to a different row.
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
            let cols: &Tip5LogupCols<F, WIDTH, POWER_WORDS, REGISTERS> = (*row).borrow();
            for round in &cols.rounds {
                for word in &round.split {
                    match self.granularity() {
                        LookupGranularity::Byte => {
                            for value in &word.input_bytes {
                                let index = value.as_canonical_u64() as usize;
                                counts[0][index] = counts[0][index]
                                    .checked_add(1)
                                    .expect("table multiplicity fits u32");
                            }
                        }
                        LookupGranularity::AdjacentPair => {
                            for byte in (0..BYTES).step_by(2) {
                                let lo = word.input_bytes[byte].as_canonical_u64() as usize;
                                let hi = word.input_bytes[byte + 1].as_canonical_u64() as usize;
                                let index = lo | (hi << 8);
                                counts[0][index] = counts[0][index]
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

impl<SC: StarkGenericConfig, const WIDTH: usize, const POWER_WORDS: usize, const REGISTERS: usize>
    LookupPermutationAir<Val<SC>, SC> for Tip5LogupAir<Val<SC>, WIDTH, POWER_WORDS, REGISTERS>
where
    Val<SC>: PrimeField64 + Sync,
    StandardUniform: Distribution<[Val<SC>; WIDTH]>,
    Tip5BatchAir<Self>: harness::lookup::LookupAir<SC>,
{
    type BatchAir = Tip5BatchAir<Self>;

    fn labels(&self) -> Labels {
        Labels {
            state_width: WIDTH,
            calls_per_row: 1,
            rows_per_call: 1,
            rounds: ROUNDS,
            // The split-and-lookup words have no power map at all; the reported
            // degree is the seventh power the other words take.
            sbox_degree: 7,
            sbox_registers: REGISTERS,
            max_constraint_degree: max_constraint_degree(REGISTERS, self.packing()),
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
