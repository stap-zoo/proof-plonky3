//! Lookup-aware proving beside the construction-blind uni-STARK path.
//!
//! `p3-uni-stark` has no post-challenge commitment in which to place LogUp's
//! auxiliary trace.  This module is therefore deliberately a separate path:
//! it sets up [`p3_batch_stark`] once, proves a heterogeneous-height batch, and
//! verifies it against the same preprocessed commitment and lookup metadata.
//! It still knows no construction names (POLICY §5).
//!
//! The cost helper takes the *post-packing* lookup declarations cached by
//! [`BatchProverVerifier`].  In particular, it never uses
//! `AirLayout::from_air` as the final layout: that helper leaves all
//! permutation fields at zero.  The pinned batch prover opens every auxiliary
//! permutation column at both `zeta` and `g zeta`, so both opening widths below
//! intentionally equal the full permutation width even when an AIR only reads
//! the next accumulator.
//!
//! Three submodules carry the parts of a lookup-backed arithmetization that are
//! the same whichever hash is being proved, so that a construction declares its
//! table values, its bus names and its call AIR and nothing else:
//!
//! * [`variants`] — the granularity and fraction-packing axes it is measured
//!   along;
//! * [`table`] — the fixed-function-table AIR: transparent keys, one witnessed
//!   multiplicity, one `table_entry` interaction;
//! * [`batch`] — the call-plus-tables type erasure `p3-batch-stark` requires.

use p3_air::symbolic::{AirLayout, SymbolicExpressionExt};
use p3_air::{Air, BaseAir, DebugConstraintBuilder};
use p3_batch_stark::proof::BatchProof;
use p3_batch_stark::symbolic::{get_max_constraint_degree, get_symbolic_constraints};
use p3_batch_stark::{
    BatchVerificationError, Challenge, ProverData, StarkGenericConfig, StarkInstance, Val,
    prove_batch, verify_batch,
};
use p3_field::{Algebra, BasedVectorSpace, PrimeField};
use p3_lookup::folder::{ProverConstraintFolderWithLookups, VerifierConstraintFolderWithLookups};
use p3_lookup::{InteractionSymbolicBuilder, LogUpGadget, Lookup, LookupProtocol};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_security::error::ErrorBits;
use p3_security::grinding::GrindingSites;
use p3_security::logup::{self, LogUpAir};
use p3_security::shape::{InstanceShape, StarkAirParams};
use p3_security::stark::proven_security_report;
use std::time::Instant;

use crate::blowup::min_log_blowup;
use crate::measure::Measurement;
use crate::permutation::Labels;

pub mod batch;
pub mod table;
pub mod variants;

pub use batch::{LookupBatch, LookupBatchAir};
pub use table::{FixedTable, FixedTableAir};
pub use variants::{FractionPacking, LookupGranularity};

/// Builder contract exercised by the batch prover, verifier and symbolic path.
///
/// Unlike [`crate::permutation::PermutationAir`], this contract includes the
/// interaction-aware symbolic builder and the LogUp prover/verifier folders.
/// A construction normally implements it through the blanket implementation.
pub trait LookupAir<SC: StarkGenericConfig>:
    Clone
    + BaseAir<Val<SC>>
    + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>
    + Air<InteractionSymbolicBuilder<Val<SC>, Challenge<SC>>>
    + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
    + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>
{
}

/// Construction-owned description of one lookup-backed permutation statement.
pub trait LookupPermutationAir<F: PrimeField, SC: StarkGenericConfig> {
    /// One erased AIR type covering the calls and their fixed tables.
    type BatchAir: LookupAir<SC>;

    /// Labels for the permutation-call AIR. Tables are reported separately by
    /// the lookup-specific measurement columns.
    fn labels(&self) -> Labels;

    /// AIRs in statement order: calls first, followed by fixed tables.
    fn batch_airs(&self) -> Vec<Self::BatchAir>;

    /// Base trace heights, as log2 values, aligned with [`Self::batch_airs`].
    fn batch_log_heights(&self, call_log_n: usize) -> Vec<usize>;

    /// Generate aligned main traces for the complete statement.
    fn generate_batch_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> Vec<RowMajorMatrix<F>>;
}

impl<SC, A> LookupAir<SC> for A
where
    SC: StarkGenericConfig,
    A: Clone
        + BaseAir<Val<SC>>
        + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>
        + Air<InteractionSymbolicBuilder<Val<SC>, Challenge<SC>>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>,
{
}

/// Lookup-aware setup shared by proving and verification.
///
/// This owns the preprocessed commitment and its prover data, plus the packed
/// lookup declarations used by both sides.  Build it outside timed regions;
/// otherwise fixed-table commitment work is incorrectly charged once per
/// proof rather than once per setup.
pub struct BatchProverVerifier<SC: StarkGenericConfig> {
    data: ProverData<SC>,
}

impl<SC> BatchProverVerifier<SC>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, Challenge<SC>>: Algebra<Challenge<SC>>,
{
    /// Commit preprocessed columns and extract/pack each AIR's interactions.
    ///
    /// `log_heights` are base-trace heights. The setup adds the PCS's ZK
    /// extension bit before calling `p3-batch-stark`.
    #[must_use]
    pub fn new<A>(config: &SC, airs: &[A], log_heights: &[usize]) -> Self
    where
        A: LookupAir<SC>,
    {
        assert_eq!(airs.len(), log_heights.len(), "one height per AIR instance");
        let extended_heights = log_heights
            .iter()
            .map(|height| height + config.is_zk())
            .collect::<Vec<_>>();
        Self {
            data: ProverData::from_airs_and_degrees(config, airs, &extended_heights),
        }
    }

    /// The post-packing lookup declarations used by prover and verifier.
    #[must_use]
    pub fn lookups(&self) -> &[p3_lookup::Lookups<Val<SC>>] {
        &self.data.common.lookups
    }

    /// Produce one batch proof. Lookup permutation traces and terminals are
    /// generated by `p3-batch-stark` after the main-trace commitment.
    #[must_use]
    pub fn prove<A>(
        &self,
        config: &SC,
        airs: &[A],
        traces: &[RowMajorMatrix<Val<SC>>],
        public_values: &[Vec<Val<SC>>],
    ) -> BatchProof<SC>
    where
        A: LookupAir<SC>,
        p3_batch_stark::Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        <SC::Pcs as p3_commit::Pcs<Challenge<SC>, SC::Challenger>>::ProverData: Sync,
        <SC::Pcs as p3_commit::Pcs<Challenge<SC>, SC::Challenger>>::Commitment: Sync,
    {
        assert_eq!(airs.len(), traces.len(), "one trace per AIR instance");
        assert_eq!(
            airs.len(),
            public_values.len(),
            "one public-value vector per AIR instance"
        );
        let trace_refs = traces.iter().collect::<Vec<_>>();
        let instances = StarkInstance::new_multiple(airs, &trace_refs, public_values);
        prove_batch(config, &instances, &self.data)
    }

    /// Verify a batch proof against the preprocessed commitment and lookup
    /// declarations fixed during [`Self::new`].
    pub fn verify<A>(
        &self,
        config: &SC,
        airs: &[A],
        proof: &BatchProof<SC>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<(), BatchVerificationError<p3_batch_stark::PcsError<SC>>>
    where
        A: LookupAir<SC>,
    {
        verify_batch(config, airs, proof, public_values, &self.data.common)
    }
}

/// Lookup-aware static shape for one AIR instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupShape {
    /// Witnessed base-field columns.
    pub main_width: usize,
    /// Transparent fixed columns.
    pub preprocessed_width: usize,
    /// Committed extension columns: accumulator plus one per packed lookup.
    pub permutation_width: usize,
    /// Transcript challenges consumed by the packed interactions.
    pub num_permutation_challenges: usize,
    /// Committed terminal values. LogUp uses one for any non-empty AIR.
    pub num_permutation_values: usize,
    /// Auxiliary columns opened at `zeta` by the pinned batch prover.
    pub permutation_local_opening_width: usize,
    /// Auxiliary columns opened at `g zeta` by the pinned batch prover.
    pub permutation_next_opening_width: usize,
    /// Number of packed lookup fraction columns (excluding the accumulator).
    pub num_interactions: usize,
    /// Widest tuple fingerprinted by any interaction.
    pub max_tuple_width: usize,
    /// Base-field constraints, excluding LogUp extension constraints.
    pub num_base_constraints: usize,
    /// LogUp extension-field constraints.
    pub num_extension_constraints: usize,
    /// Maximum degree across base and LogUp constraints.
    pub max_constraint_degree: usize,
}

/// Static and height-dependent shape of one complete lookup batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupBatchShape {
    /// Denominator factors across every row and AIR, before fraction packing.
    pub num_denominator_factors: usize,
    /// Widest fingerprinted payload tuple.
    pub max_tuple_width: usize,
    /// Exact `sum(weight_i * height_i)` enforced by prover and verifier.
    pub multiplicity_bound: u128,
    /// Base-field constraints across all AIR instances.
    pub num_base_constraints: usize,
    /// Extension-field LogUp constraints across all AIR instances.
    pub num_extension_constraints: usize,
    /// Maximum lookup-aware degree in the batch.
    pub max_constraint_degree: usize,
    /// Main plus verifier-bound preprocessed base-field cells.
    pub committed_base_cells: usize,
    /// Prover-chosen main-trace base-field cells.
    pub witness_base_cells: usize,
    /// LogUp auxiliary extension-field cells.
    pub committed_extension_cells: usize,
    /// Sum of auxiliary columns opened at the local point.
    pub permutation_local_opening_width: usize,
    /// Sum of auxiliary columns opened at the next point.
    pub permutation_next_opening_width: usize,
    /// Largest base-trace height in the heterogeneous batch.
    pub max_trace_height: usize,
}

/// Lookup-aware proven-security result before and after the union bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookupSecurity {
    /// Lookup-free STARK/PCS bound for the batch shape.
    pub stark_bits: f64,
    /// `N * (W + 2) / |EF|` LogUp fingerprint/zero-denominator bound.
    pub fingerprint_bits: f64,
    /// Union of the STARK/PCS and LogUp errors, in bits.
    pub composite_bits: f64,
}

impl LookupSecurity {
    /// Conservative whole-bit label used by the CSV and target assertion.
    #[must_use]
    pub fn floor(self) -> usize {
        self.composite_bits.floor() as usize
    }
}

/// Extract lookup-aware cost data using the same declarations as the proof.
#[must_use]
pub fn lookup_shape<SC, A>(air: &A, trace_len: usize, lookups: &[Lookup<Val<SC>>]) -> LookupShape
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, Challenge<SC>>: Algebra<Challenge<SC>>,
    A: LookupAir<SC>,
{
    let gadget = LogUpGadget::new();
    let layout = AirLayout::from_air(air);
    let (base, extension) = get_symbolic_constraints::<Val<SC>, Challenge<SC>, A, LogUpGadget>(
        air, layout, lookups, &gadget,
    );
    let permutation_width = usize::from(!lookups.is_empty()) * (lookups.len() + 1);
    let max_tuple_width = lookups
        .iter()
        .flat_map(|lookup| lookup.elements.iter().map(Vec::len))
        .max()
        .unwrap_or(0);

    LookupShape {
        main_width: air.width(),
        preprocessed_width: air.preprocessed_width(),
        permutation_width,
        num_permutation_challenges: lookups.len() * gadget.num_challenges(),
        num_permutation_values: usize::from(!lookups.is_empty()),
        // p3-batch-stark/prover.rs opens the same full matrix at both points.
        permutation_local_opening_width: permutation_width,
        permutation_next_opening_width: permutation_width,
        num_interactions: lookups.len(),
        max_tuple_width,
        num_base_constraints: base.len(),
        num_extension_constraints: extension.len(),
        max_constraint_degree: get_max_constraint_degree::<Val<SC>, Challenge<SC>, A, LogUpGadget>(
            air, layout, trace_len, lookups, &gadget,
        ),
    }
}

/// Aggregate the exact lookup/cell shape used by a heterogeneous batch proof.
#[must_use]
pub fn lookup_batch_shape<SC, A>(
    airs: &[A],
    trace_heights: &[usize],
    lookups: &[p3_lookup::Lookups<Val<SC>>],
) -> LookupBatchShape
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, Challenge<SC>>: Algebra<Challenge<SC>>,
    A: LookupAir<SC>,
{
    assert_eq!(airs.len(), trace_heights.len(), "one height per AIR");
    assert_eq!(airs.len(), lookups.len(), "one lookup set per AIR");
    p3_lookup::check_multiplicity_height_bound(lookups, trace_heights)
        .expect("LogUp multiplicity height-bound violated");

    let mut batch = LookupBatchShape {
        num_denominator_factors: 0,
        max_tuple_width: 0,
        multiplicity_bound: 0,
        num_base_constraints: 0,
        num_extension_constraints: 0,
        max_constraint_degree: 0,
        committed_base_cells: 0,
        witness_base_cells: 0,
        committed_extension_cells: 0,
        permutation_local_opening_width: 0,
        permutation_next_opening_width: 0,
        max_trace_height: 0,
    };
    for ((air, &height), air_lookups) in airs.iter().zip(trace_heights).zip(lookups) {
        let shape = lookup_shape::<SC, A>(air, height, air_lookups);
        let denominator_slots = air_lookups
            .iter()
            .map(|lookup| {
                if lookup.flags.is_some() {
                    1
                } else {
                    lookup.elements.len()
                }
            })
            .sum::<usize>();
        batch.num_denominator_factors += denominator_slots * height;
        batch.max_tuple_width = batch.max_tuple_width.max(shape.max_tuple_width);
        batch.multiplicity_bound += u128::from(air_lookups.total_count_weight()) * height as u128;
        batch.num_base_constraints += shape.num_base_constraints;
        batch.num_extension_constraints += shape.num_extension_constraints;
        batch.max_constraint_degree = batch.max_constraint_degree.max(shape.max_constraint_degree);
        batch.committed_base_cells += (shape.main_width + shape.preprocessed_width) * height;
        batch.witness_base_cells += shape.main_width * height;
        batch.committed_extension_cells += shape.permutation_width * height;
        batch.permutation_local_opening_width += shape.permutation_local_opening_width;
        batch.permutation_next_opening_width += shape.permutation_next_opening_width;
        batch.max_trace_height = batch.max_trace_height.max(height);
    }
    batch
}

/// Compose the exact LogUp error with the existing proven STARK/PCS error.
///
/// The two error probabilities are union-bounded before taking bits. This is
/// intentionally stricter than merely taking the worse of the two bit labels.
#[must_use]
pub fn lookup_security<SC: StarkGenericConfig>(
    cfg: &crate::config::Configuration<SC>,
    shape: LookupBatchShape,
) -> LookupSecurity {
    assert!(shape.max_trace_height.is_power_of_two());
    let instance = InstanceShape {
        log_trace_length: shape.max_trace_height.ilog2() as usize + cfg.zk.is_zk(),
        modulus_bits: cfg.challenge_bits,
        collision_resistance: crate::config::COLLISION_RESISTANCE_BITS,
        // The existing shared methodology does not yet price the PCS batching
        // RLC separately; keep this consistent with the lookup-free rows.
        num_batched_functions: 1,
    };
    let air = StarkAirParams {
        num_constraints: shape.num_base_constraints + shape.num_extension_constraints,
        max_constraint_degree: shape.max_constraint_degree,
        max_combo: 2,
    };
    let report = proven_security_report(&cfg.fri, &air, &instance, &[], &GrindingSites::NONE);
    let stark_bits = report.security_bits();

    // `fingerprint_error` uses interactions-per-row times trace height. The
    // batch has heterogeneous heights, so pass the already summed exact N as
    // interactions over a one-row synthetic shape.
    let fingerprint_shape = InstanceShape {
        log_trace_length: 0,
        ..instance
    };
    let fingerprint_bits = logup::fingerprint_error(
        &LogUpAir {
            num_interactions: shape.num_denominator_factors,
            max_message_width: shape.max_tuple_width,
        },
        &fingerprint_shape,
    )
    .bits();
    let union = |regime_bits: f64| {
        ErrorBits::sum(&[
            ErrorBits::from_log2(regime_bits),
            ErrorBits::from_log2(fingerprint_bits),
        ])
        .bits()
    };
    let composite_bits = report.ldr.as_ref().map_or_else(
        || union(report.udr.security_bits()),
        |ldr| union(report.udr.security_bits()).max(union(ldr.security_bits())),
    );
    LookupSecurity {
        stark_bits,
        fingerprint_bits,
        composite_bits,
    }
}

/// Measure one complete lookup batch through the same configuration and CSV
/// contract as a lookup-free permutation AIR.
#[must_use]
pub fn measure_lookup<SC, A>(
    cfg: &crate::config::Configuration<SC>,
    air: &A,
    log_n: usize,
) -> Measurement
where
    SC: StarkGenericConfig,
    A: LookupPermutationAir<Val<SC>, SC>,
    Val<SC>: PrimeField,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, Challenge<SC>>: Algebra<Challenge<SC>>,
    p3_batch_stark::Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    <SC::Pcs as p3_commit::Pcs<Challenge<SC>, SC::Challenger>>::ProverData: Sync,
    <SC::Pcs as p3_commit::Pcs<Challenge<SC>, SC::Challenger>>::Commitment: Sync,
{
    // The construction declares its packing point zk-agnostically (POLICY §7),
    // but Plonky3's same-bus packer reuses whatever quotient-chunk headroom
    // ZK's `+1` degree bump opens up, which can fold in more denominators than
    // that point asked for (see `blowup::lookup_packed_degree`). Correct the
    // label here, where zk is in scope, rather than teaching the AIR about zk.
    let mut labels = air.labels();
    labels.max_constraint_degree =
        crate::blowup::lookup_packed_degree(labels.max_constraint_degree, cfg.zk.is_zk() == 1);
    let airs = air.batch_airs();
    let log_heights = air.batch_log_heights(log_n);
    assert_eq!(airs.len(), log_heights.len());
    let heights = log_heights
        .iter()
        .map(|&height| 1usize << height)
        .collect::<Vec<_>>();

    // Includes preprocessing commitment/key generation; setup is outside every
    // timed region, just like construction of the ordinary PCS configuration.
    let setup = BatchProverVerifier::<SC>::new(&cfg.sc, &airs, &log_heights);
    let shape = lookup_batch_shape::<SC, _>(&airs, &heights, setup.lookups());
    assert_eq!(
        shape.max_constraint_degree, labels.max_constraint_degree,
        "declared lookup degree disagrees with symbolic batch degree"
    );
    let security = lookup_security(cfg, shape);
    assert!(
        security.floor() >= cfg.security_target_bits,
        "lookup batch reaches {:.3} bits, below the {} bit target",
        security.composite_bits,
        cfg.security_target_bits
    );

    let num_calls = labels.calls_in_trace(log_n);
    let extra_capacity_bits = cfg.fri.log_blowup + cfg.zk.is_zk();
    let start = Instant::now();
    let traces =
        air.generate_batch_seeded(num_calls, crate::config::INPUT_SEED, extra_capacity_bits);
    let generate_time = start.elapsed();
    assert_eq!(traces.len(), airs.len());
    assert!(
        traces
            .iter()
            .zip(&heights)
            .all(|(trace, &height)| trace.height() == height)
    );

    let public_values = (0..airs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    let start = Instant::now();
    let proof = setup.prove(&cfg.sc, &airs, &traces, &public_values);
    let prove_time = start.elapsed();
    let proof_bytes = postcard::to_allocvec(&proof)
        .expect("batch proof serialization")
        .len();
    let start = Instant::now();
    setup
        .verify(&cfg.sc, &airs, &proof, &public_values)
        .expect("the lookup proof this function produced must verify");
    let verify_time = start.elapsed();

    let trace_width = airs[0].width();
    let preprocessed_width = airs.iter().map(BaseAir::preprocessed_width).sum();
    let permutation_width = setup
        .lookups()
        .iter()
        .map(|lookups| usize::from(!lookups.is_empty()) * (lookups.len() + 1))
        .sum();
    let num_constraints = shape.num_base_constraints + shape.num_extension_constraints;
    Measurement {
        labels,
        log_n,
        num_calls,
        trace_width,
        committed_val_cells: shape.committed_base_cells,
        committed_base_cells: shape.committed_base_cells * cfg.val_dimension,
        witness_base_cells: shape.witness_base_cells * cfg.val_dimension,
        preprocessed_width,
        permutation_width,
        committed_extension_cells: shape.committed_extension_cells,
        lookup_interactions: shape.num_denominator_factors,
        lookup_max_tuple_width: shape.max_tuple_width,
        lookup_multiplicity_bound: shape.multiplicity_bound,
        num_base_constraints: shape.num_base_constraints,
        num_extension_constraints: shape.num_extension_constraints,
        permutation_local_opening_width: shape.permutation_local_opening_width,
        permutation_next_opening_width: shape.permutation_next_opening_width,
        max_constraint_degree: shape.max_constraint_degree,
        num_constraints,
        log_blowup: cfg.fri.log_blowup,
        min_log_blowup: min_log_blowup(shape.max_constraint_degree, cfg.zk.is_zk() == 1),
        num_queries: cfg.fri.num_queries,
        security_bits: security.floor(),
        generate_time,
        prove_time,
        verify_time,
        proof_bytes,
    }
}
