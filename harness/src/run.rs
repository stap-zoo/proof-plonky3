//! One run: the jobs, the two blowup readings, and the row schema.
//!
//! [`measure`](crate::measure()) is one AIR in one configuration. This module is
//! the layer above it — the sweep, the blowup reading, and the columns a table
//! script reads — and it is still construction-blind: a [`Job`] carries an
//! erased AIR and three `&'static str` labels the runner prints and never
//! branches on, the same rule [`GridPoint`](crate::GridPoint) already lives
//! under. The list of which jobs exist is `bench`'s, because knowing the
//! constructions is `bench`'s job alone (POLICY §5).
//!
//! # The two readings, and which one is primary
//!
//! POLICY §11 reports an instance at both, and they can rank an
//! arithmetization's variants differently. They are not symmetric:
//!
//! * [`Reading::Minimum`] runs every AIR at the smallest blowup its own degree
//!   admits. The degree is then a *parameter of the row* and everything
//!   downstream is priced at the rate it bought: query count, proof size, and
//!   the prover's work over the LDE. This is the primary reading.
//! * [`Reading::Common`] runs everything at [`COMMON_LOG_BLOWUP`], where every
//!   degree up to 9 costs the same rate. It measures the cells a high degree
//!   saves and none of what the degree costs, so on its own it favours
//!   high-degree strategies. It is reported as the secondary reading, and a
//!   design whose degree does not fit it is skipped with a reason rather than
//!   dropped.
//!
//! # Why a job builds its AIR rather than holding one
//!
//! A [`Job`] holds a closure returning the AIR, not the AIR. The same job is
//! measured at several trace heights and, when both readings are asked for, in
//! several configurations; building the AIR per measurement keeps every one of
//! them starting from the same constants, and it happens outside all of
//! [`measure`](crate::measure())'s timed regions.

use std::collections::BTreeMap;
use std::fmt;

use p3_air::symbolic::SymbolicExpressionExt;
use p3_field::{Algebra, BasedVectorSpace, PrimeField};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::blowup::{COMMON_LOG_BLOWUP, fits_common_blowup, measured_log_blowup, min_log_blowup};
use crate::config::{Configuration, FieldId, Zk, try_derive_fri_regime_for_sweep};
use crate::lookup::{LookupPermutationAir, measure_lookup};
use crate::measure::{Measurement, measure};
use crate::permutation::{Labels, PermutationAir};

/// Which blowup a row is measured at (POLICY §11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reading {
    /// The smallest blowup this AIR's declared degree admits, floored at
    /// [`FLOOR_LOG_BLOWUP`](crate::blowup::FLOOR_LOG_BLOWUP). Primary.
    Minimum,
    /// [`COMMON_LOG_BLOWUP`] for every AIR that fits it. Secondary.
    Common,
}

/// Which proving path a job uses.
///
/// This is a proof-system distinction, not construction knowledge: lookup
/// batches and ordinary permutation AIRs have different statement shapes.
/// `bench` uses it to select the lookup-backed side when reducing its full
/// typed plan to the headline table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobKind {
    /// One ordinary permutation AIR proved through `uni-stark`.
    Plain,
    /// A permutation plus its fixed tables proved through `batch-stark`.
    Lookup,
}

impl JobKind {
    /// Stable output label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Lookup => "lookup",
        }
    }
}

impl Reading {
    /// The name this reading carries in the output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Minimum => "minimum",
            Self::Common => "common",
        }
    }

    /// The blowup an AIR of this degree runs at, or `None` when this reading
    /// does not admit the degree at all.
    #[must_use]
    pub const fn log_blowup(self, max_constraint_degree: usize, zk: Zk) -> Option<usize> {
        let is_zk = zk.is_zk() == 1;
        match self {
            Self::Minimum => Some(measured_log_blowup(max_constraint_degree, is_zk)),
            Self::Common if fits_common_blowup(max_constraint_degree, is_zk) => {
                Some(COMMON_LOG_BLOWUP)
            }
            Self::Common => None,
        }
    }
}

impl fmt::Display for Reading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How a cell chooses the trace height every job runs at.
///
/// [`Heights`](Self::Heights) is POLICY §11's sweep and stays the default: a
/// per-call figure is the amortized value over several heights, never a single
/// point, because a trace carries fixed costs only a sweep separates from the
/// marginal ones.
///
/// [`Calls`](Self::Calls) answers a different question — *what does this much
/// work cost* — and exists because the introduction table puts layouts of very
/// different shape on one line: one BLAKE3 compression fills a row, one
/// Poseidon2 row holds eight permutation calls, and one Keccak-f call spans 24
/// rows. Under a shared height those three prove wildly different amounts of
/// work, so the height has to follow from the work rather than the other way
/// round.
///
/// Both are construction-blind. `Calls` reads [`Labels::calls_in_trace`] and
/// nothing else — the same function [`measure`](crate::measure()) already uses
/// to decide how many inputs to generate — so it cannot pick a height for one
/// hash that it would not pick for another of the same layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Workload {
    /// Every job runs at each of these heights.
    Heights(Vec<usize>),
    /// Every job runs at the *smallest* height whose layout proves at least
    /// this many calls, and only at that height.
    ///
    /// "At least", not "exactly": a layout of `rows_per_call > 1` cannot fill a
    /// power-of-two table at an arbitrary call count, so the height is rounded
    /// up and the row reports the call count it actually reached. Rounding up
    /// to the smallest admissible height is what keeps that overshoot — and the
    /// padding it implies — at its minimum.
    Calls(usize),
}

/// The largest height [`Workload::Calls`] will search up to.
///
/// A layout that needs more than `2^40` rows to reach the requested call count
/// is a mistake in the request, not a measurement: the search stops rather than
/// looping.
const MAX_SEARCHED_LOG_N: usize = 40;

impl Workload {
    /// The heights one job runs at, in ascending order.
    ///
    /// # Panics
    ///
    /// If a [`Calls`](Self::Calls) request cannot be reached below
    /// [`MAX_SEARCHED_LOG_N`], or asks for zero calls.
    #[must_use]
    pub fn heights(&self, labels: &Labels) -> Vec<usize> {
        match self {
            Self::Heights(heights) => heights.clone(),
            Self::Calls(calls) => {
                assert!(*calls > 0, "a workload of zero calls measures nothing");
                let log_n = (0..=MAX_SEARCHED_LOG_N)
                    .find(|&log_n| labels.calls_in_trace(log_n) >= *calls)
                    .unwrap_or_else(|| {
                        panic!(
                            "no trace height below 2^{MAX_SEARCHED_LOG_N} proves {calls} calls \
                             at {} calls per row over {} rows per call",
                            labels.calls_per_row, labels.rows_per_call
                        )
                    });
                vec![log_n]
            }
        }
    }

    /// Whether this workload selects no height at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Heights(heights) if heights.is_empty())
    }
}

impl fmt::Display for Workload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Heights(heights) => write!(f, "log_n {heights:?}"),
            Self::Calls(calls) => write!(f, "{calls} call(s) per proof"),
        }
    }
}

/// One measurable arithmetization, named and erased.
///
/// The three labels are what the output identifies a row by; `labels` is the
/// construction's own [`Labels`], read once at construction time so the runner
/// can group jobs by the blowup their degree asks for *before* building any
/// configuration.
pub struct Job<SC> {
    /// Construction directory name, matching a `bench::Construction::name`.
    pub construction: &'static str,
    /// Instance name (POLICY §3) — the string that ties this to a vector file.
    pub instance: &'static str,
    /// Which arithmetization or variant of that instance this is. Unique within
    /// an instance; `bench` asserts it.
    pub variant: &'static str,
    /// Ordinary or lookup-backed proving path.
    pub kind: JobKind,
    /// What the construction declares about the layout.
    pub labels: Labels,
    /// Build the AIR and measure it. Erased, so one list can hold every
    /// construction's types.
    build_and_measure: BuildAndMeasure<SC>,
}

/// A job's erased body: build this construction's AIR, then measure it.
type BuildAndMeasure<SC> = Box<dyn Fn(&Configuration<SC>, usize) -> Measurement>;

impl<SC: StarkGenericConfig> Job<SC> {
    /// Name an AIR type and how to build it.
    ///
    /// The type is named at the call site, which is what makes the work list a
    /// typed reference rather than a string to recognise (POLICY §11): an
    /// arithmetization that stops existing is a compile error here, not a row
    /// that quietly stops appearing.
    pub fn new<A>(
        construction: &'static str,
        instance: &'static str,
        variant: &'static str,
        build: impl Fn() -> A + 'static,
    ) -> Self
    where
        A: PermutationAir<Val<SC>, SC> + 'static,
        SC: 'static,
    {
        Self {
            construction,
            instance,
            variant,
            kind: JobKind::Plain,
            labels: A::LABELS,
            build_and_measure: Box::new(move |cfg, log_n| measure(cfg, &build(), log_n)),
        }
    }

    /// Name a lookup-backed batch AIR and how to build its call instance.
    pub fn new_lookup<A>(
        construction: &'static str,
        instance: &'static str,
        variant: &'static str,
        build: impl Fn() -> A + 'static,
    ) -> Self
    where
        A: LookupPermutationAir<Val<SC>, SC> + 'static,
        Val<SC>: PrimeField,
        SC::Challenge: BasedVectorSpace<Val<SC>>,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
        p3_batch_stark::Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::ProverData: Sync,
        <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::Commitment: Sync,
        SC: 'static,
    {
        let labels = build().labels();
        Self {
            construction,
            instance,
            variant,
            kind: JobKind::Lookup,
            labels,
            build_and_measure: Box::new(move |cfg, log_n| measure_lookup(cfg, &build(), log_n)),
        }
    }

    /// The blowup this job runs at under `reading`, or `None` if the reading
    /// does not admit its declared degree.
    #[must_use]
    pub const fn log_blowup(&self, reading: Reading, zk: Zk) -> Option<usize> {
        reading.log_blowup(self.labels.max_constraint_degree, zk)
    }
}

impl<SC> fmt::Debug for Job<SC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Job")
            .field("construction", &self.construction)
            .field("instance", &self.instance)
            .field("variant", &self.variant)
            .field("labels", &self.labels)
            .finish_non_exhaustive()
    }
}

/// A job list for one field, carrying the field it was written for.
///
/// The field label travels with the list so a cell cannot be run under the
/// wrong name. The list's *type* is already pinned — `Val<SC>` is the field —
/// so this only guards the printed label, which is exactly the half the type
/// system does not see.
pub struct Plan<SC> {
    /// Which field these jobs are written against.
    pub field: FieldId,
    /// The jobs.
    pub jobs: Vec<Job<SC>>,
}

impl<SC> fmt::Debug for Plan<SC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("field", &self.field)
            .field("jobs", &self.jobs)
            .finish()
    }
}

/// One measured row: everything POLICY §11 reports, plus what identifies it.
#[derive(Debug, Clone)]
pub struct Row {
    /// Construction directory name.
    pub construction: &'static str,
    /// Instance name.
    pub instance: &'static str,
    /// Variant or arithmetization name.
    pub variant: &'static str,
    /// Ordinary or lookup-backed proving path.
    pub kind: JobKind,
    /// Which field.
    pub field: FieldId,
    /// Which side of POLICY §7's ZK toggle.
    pub zk: Zk,
    /// Which blowup reading produced this row.
    pub reading: Reading,
    /// The measurement itself.
    pub measurement: Measurement,
}

impl Row {
    /// The output schema, in order.
    ///
    /// One header, one row type, one command (POLICY §11). The table script in
    /// `tools/` reads these names; adding a column is an edit here and nowhere
    /// else, and `header_matches_the_row` asserts the two stay the same length.
    pub const HEADER: &'static str = "construction,instance,variant,kind,field,zk,reading,log_n,\
         num_calls,state_width,calls_per_row,rows_per_call,rounds,sbox_degree,sbox_registers,\
         max_constraint_degree,num_constraints,num_base_constraints,num_extension_constraints,\
         trace_width,preprocessed_width,permutation_width,committed_val_cells,\
         committed_base_cells,witness_base_cells,committed_extension_cells,lookup_interactions,\
         lookup_max_tuple_width,lookup_multiplicity_bound,permutation_local_opening_width,\
         permutation_next_opening_width,log_blowup,min_log_blowup,num_queries,security_bits,\
         generate_ns,prove_ns,verify_ns,proof_bytes";

    /// This row as one CSV line, in [`HEADER`](Self::HEADER)'s order.
    ///
    /// Times are integer nanoseconds rather than a formatted duration: a table
    /// script should choose the unit it prints, and a number that has already
    /// been rounded cannot be amortized over a sweep.
    #[must_use]
    pub fn to_csv(&self) -> String {
        let m = &self.measurement;
        let l = &m.labels;
        [
            self.construction.to_string(),
            self.instance.to_string(),
            self.variant.to_string(),
            self.kind.name().to_string(),
            self.field.name().to_string(),
            self.zk.is_zk().to_string(),
            self.reading.name().to_string(),
            m.log_n.to_string(),
            m.num_calls.to_string(),
            l.state_width.to_string(),
            l.calls_per_row.to_string(),
            l.rows_per_call.to_string(),
            l.rounds.to_string(),
            l.sbox_degree.to_string(),
            l.sbox_registers.to_string(),
            m.max_constraint_degree.to_string(),
            m.num_constraints.to_string(),
            m.num_base_constraints.to_string(),
            m.num_extension_constraints.to_string(),
            m.trace_width.to_string(),
            m.preprocessed_width.to_string(),
            m.permutation_width.to_string(),
            m.committed_val_cells.to_string(),
            m.committed_base_cells.to_string(),
            m.witness_base_cells.to_string(),
            m.committed_extension_cells.to_string(),
            m.lookup_interactions.to_string(),
            m.lookup_max_tuple_width.to_string(),
            m.lookup_multiplicity_bound.to_string(),
            m.permutation_local_opening_width.to_string(),
            m.permutation_next_opening_width.to_string(),
            m.log_blowup.to_string(),
            m.min_log_blowup.to_string(),
            m.num_queries.to_string(),
            m.security_bits.to_string(),
            m.generate_time.as_nanos().to_string(),
            m.prove_time.as_nanos().to_string(),
            m.verify_time.as_nanos().to_string(),
            m.proof_bytes.to_string(),
        ]
        .join(",")
    }
}

/// A row that was not produced, and why.
///
/// Skipped scope is reported, never silent: "this arithmetization cannot be
/// measured alongside the others" is itself a result (POLICY §7).
#[derive(Debug, Clone)]
pub struct Skip {
    /// Construction directory name.
    pub construction: &'static str,
    /// Instance name.
    pub instance: &'static str,
    /// Variant or arithmetization name.
    pub variant: &'static str,
    /// Which field.
    pub field: FieldId,
    /// Which side of the ZK toggle.
    pub zk: Zk,
    /// Which reading asked for it.
    pub reading: Reading,
    /// Why it did not happen.
    pub reason: SkipReason,
}

/// Why a job produced no row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The AIR's degree needs a larger blowup than the common one, so it has no
    /// row in the strictly like-for-like table.
    AboveCommonBlowup {
        /// The AIR's declared maximum constraint degree.
        max_constraint_degree: usize,
        /// The blowup it would need.
        min_log_blowup: usize,
    },
    /// No query count reaches the security target at this blowup in this cell.
    SecurityUnreachable {
        /// The blowup that was asked for.
        log_blowup: usize,
        /// The target it could not reach.
        security_target_bits: usize,
    },
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::AboveCommonBlowup {
                max_constraint_degree,
                min_log_blowup,
            } => write!(
                f,
                "degree {max_constraint_degree} needs log_blowup {min_log_blowup}, \
                 above the common {COMMON_LOG_BLOWUP}"
            ),
            Self::SecurityUnreachable {
                log_blowup,
                security_target_bits,
            } => write!(
                f,
                "no query count reaches {security_target_bits} bits at log_blowup {log_blowup}"
            ),
        }
    }
}

/// Measure one cell of POLICY §7's matrix: one field, one ZK setting, one
/// reading.
///
/// Jobs are grouped by the blowup their declared degree asks for, and one
/// [`Configuration`] is built per group — **outside every timed region**, which
/// is the whole reason a configuration is a value here rather than something
/// the loop constructs inline (POLICY §7).
///
/// `workload` decides the heights: POLICY §11's sweep, or the one height per
/// job that reaches a requested call count (see [`Workload`]).
///
/// `build_configuration` takes `(log_blowup, log_n)` and is this cell's own
/// constructor from [`crate::config`]. The security derivation checks the exact
/// sweep because the required query count is not monotone in trace height.
///
/// # Panics
///
/// If a built configuration does not carry the blowup it was asked for, or the
/// challenge width [`FieldId::challenge_bits`] promised — the reachability
/// question was answered from those two numbers before anything was built.
pub fn run_cell<SC: StarkGenericConfig>(
    plan: &Plan<SC>,
    zk: Zk,
    reading: Reading,
    workload: &Workload,
    build_configuration: impl Fn(usize, &[usize]) -> Configuration<SC>,
    on_row: &mut impl FnMut(&Row),
    on_skip: &mut impl FnMut(&Skip),
) {
    if workload.is_empty() {
        return;
    }
    let field = plan.field;

    // Group before building: a configuration is expensive and shared, and the
    // grouping is what keeps it built once per blowup rather than once per job.
    let mut by_blowup: BTreeMap<usize, Vec<&Job<SC>>> = BTreeMap::new();
    for job in &plan.jobs {
        match job.log_blowup(reading, zk) {
            Some(log_blowup) => by_blowup.entry(log_blowup).or_default().push(job),
            None => on_skip(&Skip {
                construction: job.construction,
                instance: job.instance,
                variant: job.variant,
                field,
                zk,
                reading,
                reason: SkipReason::AboveCommonBlowup {
                    max_constraint_degree: job.labels.max_constraint_degree,
                    min_log_blowup: min_log_blowup(
                        job.labels.max_constraint_degree,
                        zk.is_zk() == 1,
                    ),
                },
            }),
        }
    }

    for (log_blowup, jobs) in by_blowup {
        // The heights this group actually visits. Under `Workload::Heights`
        // that is the requested sweep; under `Workload::Calls` each job's
        // layout picks its own height, so the configuration has to envelope
        // their union — the required query count is not monotone in trace
        // height, which is exactly why the derivation takes the whole set.
        let mut heights: Vec<usize> = jobs
            .iter()
            .flat_map(|job| workload.heights(&job.labels))
            .collect();
        heights.sort_unstable();
        heights.dedup();
        let log_n = heights.as_slice();

        // Ask before building. An unreachable target is a reported skip, never
        // a silently weaker row and never a dead run (POLICY §7).
        if try_derive_fri_regime_for_sweep(field.challenge_bits(), log_blowup, log_n, zk).is_none()
        {
            for job in jobs {
                on_skip(&Skip {
                    construction: job.construction,
                    instance: job.instance,
                    variant: job.variant,
                    field,
                    zk,
                    reading,
                    reason: SkipReason::SecurityUnreachable {
                        log_blowup,
                        security_target_bits: crate::config::SECURITY_TARGET_BITS,
                    },
                });
            }
            continue;
        }

        let cfg = build_configuration(log_blowup, log_n);
        assert_eq!(
            cfg.fri.log_blowup, log_blowup,
            "configuration was built at a blowup other than the one asked for"
        );
        assert_eq!(
            cfg.challenge_bits,
            field.challenge_bits(),
            "configuration's challenge width disagrees with {field:?}'s"
        );
        assert_eq!(cfg.zk, zk, "configuration was built on the wrong ZK side");

        for job in &jobs {
            for log_n in workload.heights(&job.labels) {
                let measurement = (job.build_and_measure)(&cfg, log_n);
                on_row(&Row {
                    construction: job.construction,
                    instance: job.instance,
                    variant: job.variant,
                    kind: job.kind,
                    field,
                    zk,
                    reading,
                    measurement,
                });
            }
        }
    }
}

/// The environment the numbers are pinned with (POLICY §11).
///
/// All four of these move every measured number, so a table that does not carry
/// them is a table of unattributed numbers. Three are read from the workspace
/// itself rather than described: the Plonky3 revision from the workspace
/// manifest, `RUSTFLAGS` from `.cargo/config.toml`, and the profile from
/// `debug_assertions`. Only the thread count is a property of the run.
#[derive(Debug, Clone, Copy)]
pub struct Environment {
    /// The single pinned Plonky3 revision every `p3-*` crate uses.
    pub plonky3_rev: &'static str,
    /// `release` or `debug`. Measurements are release-only.
    pub profile: &'static str,
    /// The workspace's `rustflags`, which decide `F::Packing::WIDTH`.
    pub rustflags: &'static str,
}

/// The workspace manifest, read at compile time.
const WORKSPACE_MANIFEST: &str = include_str!("../../Cargo.toml");
/// The workspace's cargo configuration, read at compile time.
const CARGO_CONFIG: &str = include_str!("../../.cargo/config.toml");

impl Environment {
    /// Read it out of the workspace.
    #[must_use]
    pub fn current() -> Self {
        Self {
            plonky3_rev: pinned_plonky3_rev(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            rustflags: configured_rustflags(),
        }
    }

    /// Whether this build may produce measurements.
    ///
    /// `prove` runs `check_constraints` itself under `debug_assertions`, so a
    /// debug number is a number for a different program (POLICY §10, §11).
    #[must_use]
    pub fn is_measurable(&self) -> bool {
        self.profile == "release"
    }
}

/// The revision every `p3-*` dependency is pinned to.
///
/// One revision for all of them is POLICY §5's rule and the workspace manifest
/// is where it is written; reading it back is what keeps the reported value and
/// the built value the same thing.
///
/// # Panics
///
/// If the manifest has no `rev = "…"` — which would mean the Plonky3
/// dependencies stopped being pinned.
#[must_use]
pub fn pinned_plonky3_rev() -> &'static str {
    let start = WORKSPACE_MANIFEST
        .find("rev = \"")
        .expect("workspace manifest pins no Plonky3 revision")
        + "rev = \"".len();
    let rest = &WORKSPACE_MANIFEST[start..];
    &rest[..rest.find('"').expect("unterminated rev string")]
}

/// The `rustflags` the workspace builds with.
#[must_use]
pub fn configured_rustflags() -> &'static str {
    let Some(start) = CARGO_CONFIG.find("rustflags = [") else {
        return "";
    };
    let rest = &CARGO_CONFIG[start + "rustflags = [".len()..];
    match rest.find(']') {
        Some(end) => rest[..end].trim().trim_matches('"'),
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_matches_the_row() {
        let measurement = Measurement {
            labels: Labels {
                state_width: 8,
                calls_per_row: 8,
                rows_per_call: 1,
                rounds: 7,
                sbox_degree: 7,
                sbox_registers: 1,
                max_constraint_degree: 3,
            },
            log_n: 4,
            num_calls: 128,
            trace_width: 100,
            committed_val_cells: 1600,
            committed_base_cells: 1600,
            witness_base_cells: 1600,
            preprocessed_width: 0,
            permutation_width: 0,
            committed_extension_cells: 0,
            lookup_interactions: 0,
            lookup_max_tuple_width: 0,
            lookup_multiplicity_bound: 0,
            num_base_constraints: 42,
            num_extension_constraints: 0,
            permutation_local_opening_width: 0,
            permutation_next_opening_width: 0,
            max_constraint_degree: 3,
            num_constraints: 42,
            log_blowup: 1,
            min_log_blowup: 1,
            num_queries: 200,
            security_bits: 100,
            generate_time: core::time::Duration::from_nanos(1),
            prove_time: core::time::Duration::from_nanos(2),
            verify_time: core::time::Duration::from_nanos(3),
            proof_bytes: 4,
        };
        let row = Row {
            construction: "griffin",
            instance: "griffin-goldilocks-t8",
            variant: "split",
            kind: JobKind::Plain,
            field: FieldId::Goldilocks,
            zk: Zk::Off,
            reading: Reading::Minimum,
            measurement,
        };
        assert_eq!(
            Row::HEADER.split(',').count(),
            row.to_csv().split(',').count()
        );
        // No column may contain the separator: the schema is read positionally.
        assert!(!row.to_csv().contains(",,"));
    }

    /// A one-call-per-row layout hits a requested call count exactly.
    #[test]
    fn a_call_workload_sizes_a_flat_layout_exactly() {
        let flat = Labels {
            calls_per_row: 1,
            rows_per_call: 1,
            ..LABELS
        };
        assert_eq!(Workload::Calls(512).heights(&flat), vec![9]);
        assert_eq!(flat.calls_in_trace(9), 512);

        // Eight calls to a row needs three fewer rows for the same work.
        let packed = Labels {
            calls_per_row: 8,
            rows_per_call: 1,
            ..LABELS
        };
        assert_eq!(Workload::Calls(512).heights(&packed), vec![6]);
        assert_eq!(packed.calls_in_trace(6), 512);
    }

    /// A multi-row layout cannot hit an arbitrary count, and rounds *up* to the
    /// smallest height that covers it — which is what keeps both the overshoot
    /// and the padding minimal.
    ///
    /// This is Keccak-f's case: 24 rows a call, so `24 * n` is never a power of
    /// two and 512 calls land in a `2^14` trace that holds 682.
    #[test]
    fn a_call_workload_rounds_a_multi_row_layout_up() {
        let keccak = Labels {
            calls_per_row: 1,
            rows_per_call: 24,
            ..LABELS
        };
        assert_eq!(Workload::Calls(512).heights(&keccak), vec![14]);
        assert_eq!(keccak.calls_in_trace(14), 682);
        assert!(keccak.calls_in_trace(13) < 512);
        // One height lower would not cover the request, which is the whole
        // meaning of "smallest height that proves at least this many calls".
        assert_eq!(keccak.calls_in_trace(13), 341);
    }

    /// The sweep is passed through untouched, and an empty one selects nothing.
    #[test]
    fn a_height_workload_is_the_sweep_it_was_given() {
        assert_eq!(
            Workload::Heights(vec![10, 12, 14]).heights(&LABELS),
            vec![10, 12, 14]
        );
        assert!(Workload::Heights(vec![]).is_empty());
        assert!(!Workload::Calls(1).is_empty());
    }

    /// Labels the workload tests vary one field of. Nothing here is a real
    /// construction: the point is that the height rule reads the layout and
    /// nothing else.
    const LABELS: Labels = Labels {
        state_width: 24,
        calls_per_row: 1,
        rows_per_call: 1,
        rounds: 7,
        sbox_degree: 0,
        sbox_registers: 0,
        max_constraint_degree: 3,
    };

    /// The primary reading admits every degree; the secondary one does not, and
    /// what it turns away is reported rather than dropped.
    #[test]
    fn the_readings_admit_different_degrees() {
        assert_eq!(Reading::Minimum.log_blowup(2, Zk::Off), Some(1));
        assert_eq!(Reading::Minimum.log_blowup(9, Zk::Off), Some(3));
        assert_eq!(Reading::Minimum.log_blowup(17, Zk::Off), Some(4));

        assert_eq!(Reading::Common.log_blowup(2, Zk::Off), Some(3));
        assert_eq!(Reading::Common.log_blowup(9, Zk::Off), Some(3));
        assert_eq!(Reading::Common.log_blowup(17, Zk::Off), None);
    }

    /// The pinned environment is read out of the workspace, not described.
    #[test]
    fn the_environment_reads_the_workspace() {
        let rev = pinned_plonky3_rev();
        assert_eq!(rev.len(), 40, "not a git revision: {rev}");
        assert!(rev.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(
            WORKSPACE_MANIFEST
                .matches("rev = \"")
                .count()
                .checked_sub(WORKSPACE_MANIFEST.matches(rev).count())
                == Some(0),
            "the p3-* dependencies are not all on one revision (POLICY §5)"
        );
        assert!(configured_rustflags().contains("target-cpu"));
    }
}
