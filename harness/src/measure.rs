//! The unit of work: one AIR, one configuration, one row of the table.

use std::time::{Duration, Instant};

use crate::permutation::{Labels, PermutationAir};
use p3_air::symbolic::{AirLayout, get_max_constraint_degree, get_symbolic_constraints};
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, StarkSecurityParams, Val, prove, verify};

use crate::blowup::min_log_blowup;
use crate::config::Configuration;

/// Everything POLICY §11 reports for one instance × variant.
///
/// Read it in two halves. **Degree, constraint count, width and cells are
/// properties of the arithmetization** and are comparable everywhere, including
/// across fields. **Times, sizes and security also depend on the
/// configuration** and are comparable only within one cell of POLICY §7's
/// matrix.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// What the construction says it is. Printed, never branched on.
    pub labels: Labels,
    /// `log2` of the trace height.
    pub log_n: usize,
    /// Permutation calls this trace proves. Exact — tables are full.
    pub num_calls: usize,

    /// Committed `Val` cells per row.
    pub trace_width: usize,
    /// Committed `Val` cells in the whole trace.
    pub committed_val_cells: usize,
    /// The same in base-field elements: `× Val::DIMENSION`.
    pub committed_base_cells: usize,
    /// Prover-chosen base-field cells, excluding transparent preprocessing.
    pub witness_base_cells: usize,
    /// Transparent preprocessed columns (zero for the uni-STARK path).
    pub preprocessed_width: usize,
    /// LogUp auxiliary extension columns (zero for the uni-STARK path).
    pub permutation_width: usize,
    /// Committed LogUp extension-field cells.
    pub committed_extension_cells: usize,
    /// Denominator factors across the whole heterogeneous lookup batch.
    pub lookup_interactions: usize,
    /// Widest lookup message payload.
    pub lookup_max_tuple_width: usize,
    /// Exact verifier-enforced `sum(weight_i * height_i)`.
    pub lookup_multiplicity_bound: u128,
    /// Base-field constraints, split out for lookup-aware rows.
    pub num_base_constraints: usize,
    /// Extension-field constraints, split out for lookup-aware rows.
    pub num_extension_constraints: usize,
    /// Auxiliary opening width at the local point.
    pub permutation_local_opening_width: usize,
    /// Auxiliary opening width at the next point.
    pub permutation_next_opening_width: usize,

    /// From `get_max_constraint_degree`, cross-checked against
    /// `Labels::max_constraint_degree`.
    pub max_constraint_degree: usize,
    /// From `get_symbolic_constraints(..).len()`.
    pub num_constraints: usize,

    /// The blowup this measurement ran at.
    pub log_blowup: usize,
    /// The smallest blowup this AIR's degree admits — what the degree choice
    /// buys, against the common-blowup reading.
    pub min_log_blowup: usize,
    /// FRI queries at this security target.
    pub num_queries: usize,
    /// `StarkSecurityParams::from_air(..)`, asserted `>= target`.
    pub security_bits: usize,

    /// Trace generation, timed separately from proving: it is the construction's
    /// own witness cost and it parallelizes differently.
    pub generate_time: Duration,
    /// `prove`.
    pub prove_time: Duration,
    /// `verify`.
    pub verify_time: Duration,
    /// Serialized proof size in bytes.
    pub proof_bytes: usize,
}

impl Measurement {
    /// Cost amortized to one permutation call.
    ///
    /// Per-call cost is this figure taken from a sweep over `log_n`, not from a
    /// single point: a trace carries fixed costs that only a sweep separates
    /// from the marginal ones.
    #[must_use]
    pub fn prove_time_per_call(&self) -> Duration {
        self.prove_time
            .checked_div(u32::try_from(self.num_calls).unwrap_or(u32::MAX))
            .unwrap_or_default()
    }
}

/// Measure one AIR in one configuration, at a trace height of `2^log_n`.
///
/// The whole function is generic in the field and blind to the construction:
/// it can see an AIR, its labels and its trace, and nothing else. That is the
/// entire basis for claiming two rows are comparable, so anything that would
/// need to know *which* hash this is belongs in the construction or in the
/// table script, not here.
///
/// Timed regions contain exactly one operation each. The configuration, PCS,
/// MMCS and challenger were built when [`Configuration`] was built, outside all
/// of them.
///
/// # Panics
///
/// * if the AIR's declared `max_constraint_degree` disagrees with the symbolic
///   one — a wrong declaration would otherwise buy a silently cheaper blowup;
/// * if the configuration does not reach its security target;
/// * if `verify` rejects the proof this very function produced.
pub fn measure<SC, A>(cfg: &Configuration<SC>, air: &A, log_n: usize) -> Measurement
where
    SC: StarkGenericConfig,
    A: PermutationAir<Val<SC>, SC>,
{
    let labels = A::LABELS;
    let layout = AirLayout::from_air(air);
    let trace_len = 1usize << log_n;

    // Arithmetization properties: no proof involved, no configuration involved.
    let max_constraint_degree = get_max_constraint_degree(air, layout, trace_len);
    assert_eq!(
        max_constraint_degree, labels.max_constraint_degree,
        "declared max constraint degree disagrees with the symbolic one; \
         the declared value is what derives the blowup"
    );
    let num_constraints = get_symbolic_constraints::<Val<SC>, A>(air, layout).len();
    assert!(
        num_constraints <= crate::config::SECURITY_MAX_CONSTRAINTS,
        "AIR has {num_constraints} constraints, above the shared security envelope of {}",
        crate::config::SECURITY_MAX_CONSTRAINTS
    );

    // Security is held fixed and asserted, not assumed (POLICY §7).
    let security_params = StarkSecurityParams::from_air::<Val<SC>, Val<SC>, A>(
        cfg.fri,
        air,
        layout,
        cfg.challenge_bits,
        crate::config::COLLISION_RESISTANCE_BITS,
        2,
    );
    // Hiding FRI commits one random codeword, doubling the degree bound. The
    // security API's proof-level convention takes that committed degree, not
    // the unblinded witness height.
    let security_trace_len = trace_len << cfg.zk.is_zk();
    let security_bits =
        p3_uni_stark::ProvenSecurity::compute(&security_params, security_trace_len).security_bits();
    assert!(
        security_bits >= cfg.security_target_bits,
        "configuration reaches {security_bits} bits, below the {} bit target",
        cfg.security_target_bits
    );

    let num_calls = labels.calls_in_trace(log_n);
    // The LDE happens in place: the trace is allocated with the prover's extra
    // capacity, so `prove` does not pay for a reallocation the AIR could have
    // avoided.
    let extra_capacity_bits = cfg.fri.log_blowup + cfg.zk.is_zk();

    let start = Instant::now();
    let trace =
        air.generate_trace_seeded(num_calls, crate::config::INPUT_SEED, extra_capacity_bits);
    let generate_time = start.elapsed();

    let trace_width = trace.width();
    let committed_val_cells = trace_width * trace.height();

    let start = Instant::now();
    let proof = prove(&cfg.sc, air, trace, &[]);
    let prove_time = start.elapsed();

    let proof_bytes = postcard::to_allocvec(&proof)
        .expect("proof serialization")
        .len();

    let start = Instant::now();
    verify(&cfg.sc, air, &proof, &[]).expect("the proof this function just produced must verify");
    let verify_time = start.elapsed();

    Measurement {
        labels,
        log_n,
        num_calls,
        trace_width,
        committed_val_cells,
        committed_base_cells: committed_val_cells * cfg.val_dimension,
        witness_base_cells: committed_val_cells * cfg.val_dimension,
        preprocessed_width: 0,
        permutation_width: 0,
        committed_extension_cells: 0,
        lookup_interactions: 0,
        lookup_max_tuple_width: 0,
        lookup_multiplicity_bound: 0,
        num_base_constraints: num_constraints,
        num_extension_constraints: 0,
        permutation_local_opening_width: 0,
        permutation_next_opening_width: 0,
        max_constraint_degree,
        num_constraints,
        log_blowup: cfg.fri.log_blowup,
        min_log_blowup: min_log_blowup(max_constraint_degree, cfg.zk.is_zk() == 1),
        num_queries: cfg.fri.num_queries,
        security_bits,
        generate_time,
        prove_time,
        verify_time,
        proof_bytes,
    }
}
