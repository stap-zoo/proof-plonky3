//! Selection, dispatch, and the one output stream.
//!
//! POLICY §11: **one command must produce everything measured, selected by
//! flags**. This module is that command's body — [`plan`](crate::plan) says what
//! exists, `harness::run` says how a cell is measured, and the loop here is only
//! the cross product of POLICY §7's configuration matrix with the selection.
//!
//! Rows go to stdout as CSV, everything else to stderr. That split is what lets
//! `cargo run --release --bin bench > rows.csv` be the whole interface: the
//! table script in `tools/` reads one schema on one stream, and progress,
//! skipped scope and the pinned environment do not contaminate it.
//!
//! # The dispatch is typed, and the type is the field
//!
//! Each arm below pairs one configuration constructor with one job list, and
//! both are pinned to the same `Val`. There is no string that names an AIR: the
//! only strings are the selection filters, which narrow a list the compiler
//! already built.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use harness::config::{
    GoldilocksConfig, GoldilocksZkConfig, M31CircleConfig, TwoAdic31Config, TwoAdic31ZkConfig,
};
use harness::{
    Configuration, Environment, FieldId, Job, JobKind, Plan, Reading, Row, Skip, Workload, Zk,
    run_cell,
};
use p3_baby_bear::BabyBear;
use p3_koala_bear::KoalaBear;
use p3_uni_stark::StarkGenericConfig;

use crate::plan;

/// Message bytes one permutation call of an introduction-table design absorbs.
///
/// This is a **sponge rate**, so it is reported and never chosen (POLICY §8),
/// and it lives here rather than in `harness` for two reasons: a rate is a mode
/// of operation, which POLICY §8 keeps out of scope for the arithmetization
/// layer, and `harness` may not learn which hashes exist at all (POLICY §5).
/// The height rule below is therefore `bench`'s, and the harness only ever sees
/// the heights it produced.
///
/// The three wrapped traditional designs take theirs from
/// [`traditional::instances::UNITS`], which is beside the AIRs they describe.
/// Poseidon2's is the one entry stated here:
///
/// * `../ref` pins KoalaBear `t = 24` at **rate 16, capacity 8, digest 8** —
///   unanimously, across all six sibling constructions that export params at
///   that grid point (`bench/tests/intro.rs` reads one of those exports back
///   and asserts it, so this cannot drift from the reference silently).
/// * Sixteen elements become **48 bytes** at three bytes per element. That last
///   step is the reference's rate expressed in bytes, and it is the conservative
///   reading: 24 of KoalaBear's ~31 bits is what an injective byte-to-field
///   encoding actually carries. The optimistic reading, 16 x 31 bits = 62
///   bytes, credits the design with capacity no byte-oriented message can fill,
///   and it flatters the arithmetization-oriented side of the very comparison
///   this table exists to make.
const POSEIDON2_RATE_ELEMENTS: usize = 16;
/// Bytes an injective encoding puts in one KoalaBear element.
const BYTES_PER_KOALABEAR_ELEMENT: usize = 3;

/// Bytes one call absorbs, by construction name.
///
/// `None` for a construction with no stated rate, which is an error at the call
/// site rather than a default: a silent zero would make its batch look
/// infinitely cheap per byte.
#[must_use]
pub fn absorbed_bytes(construction: &str) -> Option<usize> {
    if construction == "poseidon2" {
        return Some(POSEIDON2_RATE_ELEMENTS * BYTES_PER_KOALABEAR_ELEMENT);
    }
    traditional::instances::UNITS
        .iter()
        .find(|unit| unit.construction == construction)
        .map(|unit| unit.absorbed_bytes)
}

/// The largest trace height whose batch does not exceed `target_bytes`.
///
/// **Largest without exceeding**, deliberately, rather than nearest: the
/// reachable batch sizes double, so "nearest" is a near-tie for a layout like
/// Keccak-f's (0.708 below against 1.415 above) and would flip on a rounding
/// change. Never exceeding also means no row is handed a bigger batch than the
/// target to amortize its fixed costs over, so where a design cannot land on
/// the target it is charged for the shortfall rather than credited for an
/// overshoot.
///
/// # Panics
///
/// If no height proves even one call within the target — a request smaller than
/// a single call is a mistake in the request, not a measurement.
#[must_use]
pub fn log_n_for_bytes(
    labels: &harness::Labels,
    bytes_per_call: usize,
    target_bytes: usize,
) -> usize {
    assert!(
        bytes_per_call > 0,
        "a call that absorbs nothing has no cost per byte"
    );
    let mut chosen = None;
    for log_n in 0..=MAX_SEARCHED_LOG_N {
        let calls = labels.calls_in_trace(log_n);
        if calls == 0 {
            continue;
        }
        if calls * bytes_per_call > target_bytes {
            break;
        }
        chosen = Some(log_n);
    }
    chosen.unwrap_or_else(|| {
        panic!(
            "no trace height proves one call within {target_bytes} bytes at \
             {bytes_per_call} bytes per call"
        )
    })
}

/// Mirrors `harness::run`'s own search bound.
const MAX_SEARCHED_LOG_N: usize = 40;

/// What the flags selected.
#[derive(Debug, Clone)]
pub struct Selection {
    /// Measure only this construction, if given.
    pub construction: Option<String>,
    /// Measure only instances whose name contains this, if given.
    pub instance: Option<String>,
    /// Measure only this variant, if given.
    pub variant: Option<String>,
    /// Which fields to measure.
    pub fields: Vec<FieldId>,
    /// Which sides of POLICY §7's ZK toggle to measure.
    pub zk: Vec<Zk>,
    /// Which blowup readings to report (POLICY §11).
    pub readings: Vec<Reading>,
    /// Trace heights, as `log2`. Per-call cost is amortized from this sweep,
    /// never read off a single point.
    pub log_n: Vec<usize>,
    /// Reduce the full plan to the paper's one-instance, lowest-degree rows.
    pub headline: bool,
    /// Measure the introduction table's list instead of the construction grid:
    /// the traditional hashes Plonky3 arithmetizes, plus one
    /// arithmetization-oriented reference row (`plan::intro_koalabear`).
    ///
    /// A separate list rather than a filter, because those hashes register no
    /// grid point (POLICY §3). Only the KoalaBear cells carry it; every other
    /// field reports an empty selection.
    pub intro: bool,
    /// Size every trace by the work it proves rather than by its height: each
    /// job runs at the smallest height whose layout proves at least this many
    /// permutation calls.
    ///
    /// `None` keeps POLICY §11's sweep over [`Self::log_n`]. This is what puts
    /// one-call-per-row, eight-calls-per-row and 24-rows-per-call layouts on
    /// one line of the introduction table.
    pub calls: Option<usize>,
    /// Size every trace by the **message bytes** it covers rather than by its
    /// height or its call count: each job runs at the largest height whose
    /// batch does not exceed this many bytes.
    ///
    /// This is what makes the introduction table an equal-*data* comparison
    /// instead of an equal-*calls* one. The two are not the same question: one
    /// BLAKE3 call compresses 64 bytes, one Keccak-f call permutes a state
    /// whose rate is 136, and one Poseidon2 call absorbs 16 field elements.
    ///
    /// Takes precedence over [`Self::calls`]. Only [`absorbed_bytes`] knows the
    /// rates, so a construction without one is an error rather than a default.
    pub bytes: Option<usize>,
}

impl Selection {
    /// Whether a job survives the name filters.
    ///
    /// Field and ZK are not filtered here: they select the *cell*, and a cell
    /// that is not selected is never built.
    #[must_use]
    pub fn admits<SC>(&self, job: &Job<SC>) -> bool {
        self.construction
            .as_ref()
            .is_none_or(|c| job.construction == c)
            && self
                .instance
                .as_ref()
                .is_none_or(|i| job.instance.contains(i))
            && self.variant.as_ref().is_none_or(|v| job.variant == v)
    }

    /// How the heights are chosen when they are the same for every job.
    ///
    /// A byte target is *not* one of these: it gives each layout its own height
    /// (see [`Self::bytes`] and [`partition_by_bytes`]), so it is resolved in
    /// [`cell`] rather than here.
    #[must_use]
    pub fn workload(&self) -> Workload {
        match self.calls {
            Some(calls) => Workload::Calls(calls),
            None => Workload::Heights(self.log_n.clone()),
        }
    }

    /// How this selection describes its workload in the provenance block.
    #[must_use]
    pub fn workload_note(&self) -> String {
        match self.bytes {
            Some(bytes) => format!("{bytes} message byte(s) per proof, at most"),
            None => self.workload().to_string(),
        }
    }

    /// The selected list for one field, already filtered.
    fn filtered<SC>(&self, mut plan: Plan<SC>) -> Plan<SC> {
        plan.jobs.retain(|job| self.admits(job));
        if self.headline {
            let width = headline_width(plan.field);
            plan.jobs.retain(|job| {
                job.labels.state_width == width
                    && headline_rounds(plan.field, job.construction)
                        .is_none_or(|rounds| job.labels.rounds == rounds)
            });

            // The paper reports the lookup-backed AIR wherever one exists.
            // Remove the lookup-free alternative before choosing the best
            // instance/variant, so it is never measured merely to be discarded
            // by the LaTeX renderer.
            let with_lookup: BTreeSet<_> = plan
                .jobs
                .iter()
                .filter(|job| job.kind == JobKind::Lookup)
                .map(|job| job.construction)
                .collect();
            plan.jobs.retain(|job| {
                job.kind == JobKind::Lookup || !with_lookup.contains(job.construction)
            });

            plan.jobs.sort_by(|a, b| {
                a.construction
                    .cmp(b.construction)
                    .then_with(|| headline_key(a).cmp(&headline_key(b)))
            });
            let mut seen = BTreeSet::new();
            plan.jobs.retain(|job| seen.insert(job.construction));
        }
        plan
    }
}

/// The one width the headline table compares at, per field.
///
/// POLICY §3's grid carries two widths per field so that a construction can be
/// costed at both; the paper table compares *one*, and it is the larger of the
/// two — `t = 24` at every 31-bit prime, `t = 12` over Goldilocks. Pinning the
/// width here rather than tie-breaking on it in [`headline_key`] is what makes
/// the choice a stated parameter of the table instead of a by-product of the
/// sort: a construction with no AIR at its field's headline width drops out of
/// the table, and drops out visibly, rather than quietly contributing a row at
/// the other width that no other row is comparable to.
///
/// This is also what selects Tip5's `tip4-prime` over the wider `tip5` and
/// `tip4`: all three are registered, only the first is a `t = 12` point.
#[must_use]
const fn headline_width(field: FieldId) -> usize {
    match field {
        FieldId::Goldilocks => 12,
        FieldId::Mersenne31 | FieldId::BabyBear | FieldId::KoalaBear => 24,
    }
}

/// The round count the table publishes, where a cell offers more than one.
///
/// Almost every cell offers exactly one, and this returns `None` for those: the
/// round count is whatever the construction's single instance carries, and
/// naming it here would be a second place for it to drift from.
///
/// Rescue-Prime over Goldilocks is the exception. Two instances sit at `t = 12`
/// — the reference-pinned `R = 8` and the author-supplied `R = 13` (POLICY §4) —
/// and both are real, measured rows. Which one the *paper* prints is a claim
/// about cryptanalysis, not about arithmetization, so it is stated here rather
/// than left to [`headline_key`]'s name ordering, which would otherwise publish
/// `R = 8` for no better reason than that it sorts first.
///
/// Checked, not merely asserted: a round count named here that no job carries
/// empties the cell, and the headline test pins the whole resulting row set.
#[must_use]
const fn headline_rounds(field: FieldId, construction: &str) -> Option<usize> {
    match (field, construction.as_bytes()) {
        (FieldId::Goldilocks, b"rescue-prime") => Some(13),
        _ => None,
    }
}

/// The deterministic tie-breaks for the requested headline table.
///
/// Alpha and AIR degree are the specified choices. A construction's own named
/// instance wins remaining ties — which is what keeps XHash's aggressive
/// `xhash8` over the full-S-box `xhash12` when both are registered under the one
/// `xhash8` construction name and both sit at `t = 12`. `state_width` is inert
/// now that [`headline_width`] pins it, and stays only so the key is total on a
/// plan filtered by something else.
fn headline_key<SC>(job: &Job<SC>) -> (u64, usize, usize, usize, &'static str, &'static str) {
    let canonical_name = if job.instance == job.construction {
        0
    } else if job
        .instance
        .strip_prefix(job.construction)
        .is_some_and(|suffix| suffix.starts_with('-'))
    {
        1
    } else {
        2
    };
    (
        job.labels.sbox_degree,
        job.labels.max_constraint_degree,
        canonical_name,
        job.labels.state_width,
        job.instance,
        job.variant,
    )
}

/// What a run produced.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    /// Rows written.
    pub rows: usize,
    /// Jobs a reading or a security target turned away, each one reported.
    pub skipped: usize,
    /// Cells POLICY §7's matrix has but this repository cannot fill.
    pub empty_cells: usize,
}

/// Measure everything the selection names, writing CSV rows to `out`.
///
/// # Panics
///
/// Through [`harness::measure`]: a declared degree that disagrees with the
/// symbolic one, a configuration below its security target, or a proof this
/// very function produced failing to verify. None of those are conditions a
/// measurement run should survive.
pub fn run(selection: &Selection, out: &mut impl Write) -> Summary {
    let mut summary = Summary::default();

    writeln!(out, "{}", Row::HEADER).expect("write header");

    let mut on_row = |row: &Row| {
        writeln!(out, "{}", row.to_csv()).expect("write row");
    };
    let mut on_skip = |skip: &Skip| {
        eprintln!(
            "skipped {}/{}/{} [{:?} zk={} {}]: {}",
            skip.construction,
            skip.instance,
            skip.variant,
            skip.field,
            skip.zk.is_zk(),
            skip.reading,
            skip.reason
        );
    };

    for &field in &selection.fields {
        for &zk in &selection.zk {
            for &reading in &selection.readings {
                let before = (summary.rows, summary.skipped);
                // `--intro` replaces the list, it does not filter it: the
                // traditional hashes are registered nowhere else, and the
                // construction grid has no place to put them (POLICY §3).
                if selection.intro && field != FieldId::KoalaBear {
                    eprintln!(
                        "no job selected in {}/zk={}/{reading}: --intro is KoalaBear only",
                        field.name(),
                        zk.is_zk()
                    );
                    continue;
                }
                match (field, zk) {
                    (FieldId::Goldilocks, Zk::Off) => cell(
                        selection,
                        selection.filtered(plan::goldilocks::<GoldilocksConfig>()),
                        zk,
                        reading,
                        harness::config::goldilocks,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    (FieldId::Goldilocks, Zk::On) => cell(
                        selection,
                        selection.filtered(plan::goldilocks::<GoldilocksZkConfig>()),
                        zk,
                        reading,
                        harness::config::goldilocks_zk,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    (FieldId::BabyBear, Zk::Off) => cell(
                        selection,
                        selection.filtered(plan::babybear::<TwoAdic31Config<BabyBear>>()),
                        zk,
                        reading,
                        harness::config::babybear,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    (FieldId::BabyBear, Zk::On) => cell(
                        selection,
                        selection.filtered(plan::babybear::<TwoAdic31ZkConfig<BabyBear>>()),
                        zk,
                        reading,
                        harness::config::babybear_zk,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    (FieldId::KoalaBear, Zk::Off) => cell(
                        selection,
                        selection.filtered(if selection.intro {
                            plan::intro_koalabear::<TwoAdic31Config<KoalaBear>>()
                        } else {
                            plan::koalabear::<TwoAdic31Config<KoalaBear>>()
                        }),
                        zk,
                        reading,
                        harness::config::koalabear,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    (FieldId::KoalaBear, Zk::On) => cell(
                        selection,
                        selection.filtered(if selection.intro {
                            plan::intro_koalabear::<TwoAdic31ZkConfig<KoalaBear>>()
                        } else {
                            plan::koalabear::<TwoAdic31ZkConfig<KoalaBear>>()
                        }),
                        zk,
                        reading,
                        harness::config::koalabear_zk,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    (FieldId::Mersenne31, Zk::Off) => cell(
                        selection,
                        selection.filtered(plan::mersenne31::<M31CircleConfig>()),
                        zk,
                        reading,
                        harness::config::mersenne31,
                        &mut summary,
                        &mut on_row,
                        &mut on_skip,
                    ),
                    // POLICY §7 declares this cell unsupported. Reported once
                    // per selection, never silently absent.
                    (FieldId::Mersenne31, Zk::On) => {
                        summary.empty_cells += 1;
                        eprintln!("empty cell mersenne31/zk: {}", plan::MERSENNE31_ZK);
                    }
                }
                if (summary.rows, summary.skipped) == before
                    && !(field == FieldId::Mersenne31 && zk == Zk::On)
                {
                    eprintln!(
                        "no job selected in {}/zk={}/{reading}",
                        field.name(),
                        zk.is_zk()
                    );
                }
            }
        }
    }

    summary
}

/// One cell, with its own counters.
#[allow(clippy::too_many_arguments)]
fn cell<SC: StarkGenericConfig>(
    selection: &Selection,
    plan: Plan<SC>,
    zk: Zk,
    reading: Reading,
    build_configuration: impl Fn(usize, &[usize]) -> Configuration<SC>,
    summary: &mut Summary,
    on_row: &mut impl FnMut(&Row),
    on_skip: &mut impl FnMut(&Skip),
) {
    if plan.jobs.is_empty() {
        return;
    }
    eprintln!(
        "measuring {} job(s) in {}/zk={}/{reading} at {}",
        plan.jobs.len(),
        plan.field.name(),
        zk.is_zk(),
        selection.workload_note(),
    );

    // A byte target gives each layout its own height, so the cell becomes one
    // `run_cell` per height. Everything else is one call with a shared
    // workload. Either way the harness sees only heights: the rates that
    // produced them stay here (POLICY §5, §8).
    let groups = match selection.bytes {
        Some(target) => partition_by_bytes(plan, target),
        None => vec![(selection.workload(), plan)],
    };

    let mut rows = 0usize;
    let mut skips = 0usize;
    for (workload, plan) in groups {
        run_cell(
            &plan,
            zk,
            reading,
            &workload,
            &build_configuration,
            &mut |row| {
                rows += 1;
                on_row(row);
            },
            &mut |skip| {
                skips += 1;
                on_skip(skip);
            },
        );
    }
    summary.rows += rows;
    summary.skipped += skips;
}

/// Split a plan into one sub-plan per trace height a byte target implies.
///
/// Grouping rather than one `run_cell` per job keeps a configuration shared by
/// every job that lands on the same height, which is what POLICY §7 asks for:
/// built once, outside every timed region.
fn partition_by_bytes<SC>(plan: Plan<SC>, target_bytes: usize) -> Vec<(Workload, Plan<SC>)> {
    let field = plan.field;
    let mut by_height: BTreeMap<usize, Vec<Job<SC>>> = BTreeMap::new();
    for job in plan.jobs {
        let bytes_per_call = absorbed_bytes(job.construction).unwrap_or_else(|| {
            panic!(
                "no stated rate for `{}`; a byte target cannot size its trace \
                 (add it beside `absorbed_bytes`)",
                job.construction
            )
        });
        let log_n = log_n_for_bytes(&job.labels, bytes_per_call, target_bytes);
        let calls = job.labels.calls_in_trace(log_n);
        eprintln!(
            "  {}/{}: 2^{log_n} rows, {calls} call(s), {} of {target_bytes} byte(s)",
            job.construction,
            job.variant,
            calls * bytes_per_call,
        );
        by_height.entry(log_n).or_default().push(job);
    }
    by_height
        .into_iter()
        .map(|(log_n, jobs)| (Workload::Heights(vec![log_n]), Plan { field, jobs }))
        .collect()
}

/// List what would be measured, without measuring it.
///
/// The one thing a debug build may do (POLICY §11 makes measurements
/// release-only), and the fastest way to see what a set of flags selects.
pub fn dry_run(selection: &Selection, out: &mut impl Write) -> usize {
    writeln!(
        out,
        "field,zk,reading,construction,instance,variant,max_constraint_degree,log_blowup,kind"
    )
    .expect("write header");

    let mut listed = 0;
    for &field in &selection.fields {
        for &zk in &selection.zk {
            for &reading in &selection.readings {
                let entries: Vec<(
                    &'static str,
                    &'static str,
                    &'static str,
                    usize,
                    &'static str,
                )> = match (field, zk) {
                    // `--intro` replaces the list, so it is matched before any
                    // field arm: the traditional hashes exist in no other plan,
                    // and every field but KoalaBear selects nothing.
                    (FieldId::KoalaBear, Zk::Off) if selection.intro => describe(
                        &selection.filtered(plan::intro_koalabear::<TwoAdic31Config<KoalaBear>>()),
                    ),
                    (FieldId::KoalaBear, Zk::On) if selection.intro => describe(
                        &selection
                            .filtered(plan::intro_koalabear::<TwoAdic31ZkConfig<KoalaBear>>()),
                    ),
                    _ if selection.intro => Vec::new(),
                    (FieldId::Goldilocks, Zk::Off) => {
                        describe(&selection.filtered(plan::goldilocks::<GoldilocksConfig>()))
                    }
                    (FieldId::Goldilocks, Zk::On) => {
                        describe(&selection.filtered(plan::goldilocks::<GoldilocksZkConfig>()))
                    }
                    (FieldId::BabyBear, Zk::Off) => {
                        describe(&selection.filtered(plan::babybear::<TwoAdic31Config<BabyBear>>()))
                    }
                    (FieldId::BabyBear, Zk::On) => describe(
                        &selection.filtered(plan::babybear::<TwoAdic31ZkConfig<BabyBear>>()),
                    ),
                    (FieldId::KoalaBear, Zk::Off) => describe(
                        &selection.filtered(plan::koalabear::<TwoAdic31Config<KoalaBear>>()),
                    ),
                    (FieldId::KoalaBear, Zk::On) => describe(
                        &selection.filtered(plan::koalabear::<TwoAdic31ZkConfig<KoalaBear>>()),
                    ),
                    (FieldId::Mersenne31, Zk::Off) => {
                        describe(&selection.filtered(plan::mersenne31::<M31CircleConfig>()))
                    }
                    (FieldId::Mersenne31, Zk::On) => Vec::new(),
                };
                for (construction, instance, variant, degree, kind) in entries {
                    let blowup = reading
                        .log_blowup(degree, zk)
                        .map_or_else(|| "-".to_string(), |b| b.to_string());
                    writeln!(
                        out,
                        "{},{},{reading},{construction},{instance},{variant},{degree},{blowup},{kind}",
                        field.name(),
                        zk.is_zk(),
                    )
                    .expect("write row");
                    listed += 1;
                }
            }
        }
    }
    listed
}

/// A plan's jobs as printable tuples, erasing the configuration type.
fn describe<SC>(
    plan: &Plan<SC>,
) -> Vec<(
    &'static str,
    &'static str,
    &'static str,
    usize,
    &'static str,
)> {
    plan.jobs
        .iter()
        .map(|job| {
            (
                job.construction,
                job.instance,
                job.variant,
                job.labels.max_constraint_degree,
                job.kind.name(),
            )
        })
        .collect()
}

/// The construction names a selection will measure, for the provenance block.
fn plan_constructions(selection: &Selection) -> Vec<&'static str> {
    let plan = if selection.intro {
        plan::intro_koalabear::<TwoAdic31ZkConfig<KoalaBear>>()
    } else {
        plan::koalabear::<TwoAdic31ZkConfig<KoalaBear>>()
    };
    let mut names: Vec<_> = plan
        .jobs
        .iter()
        .filter(|job| selection.admits(job))
        .map(|job| job.construction)
        .collect();
    names.dedup();
    names
}

/// The `#`-prefixed provenance block that precedes the rows.
///
/// POLICY §11 pins the environment *with* the numbers: build profile,
/// `RUSTFLAGS`, `RAYON_NUM_THREADS` and the Plonky3 revision all move them. A
/// table script reads these lines; a `#` prefix keeps them out of the schema.
pub fn write_provenance(selection: &Selection, out: &mut impl Write) {
    let environment = Environment::current();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    writeln!(out, "# plonky3-rev: {}", environment.plonky3_rev).expect("write provenance");
    writeln!(out, "# profile: {}", environment.profile).expect("write provenance");
    writeln!(out, "# rustflags: {}", environment.rustflags).expect("write provenance");
    writeln!(out, "# rayon-threads: {threads}").expect("write provenance");
    writeln!(
        out,
        "# security-target-bits: {}",
        harness::SECURITY_TARGET_BITS
    )
    .expect("write provenance");
    writeln!(out, "# input-seed: {:#x}", harness::INPUT_SEED).expect("write provenance");
    writeln!(out, "# workload: {}", selection.workload_note()).expect("write provenance");
    // The rates that sized the traces, reported with the numbers they produced
    // (POLICY §8): the table script converts calls to bytes from these and
    // never carries a rate of its own.
    if selection.bytes.is_some() {
        let rates: Vec<String> = plan_constructions(selection)
            .into_iter()
            .filter_map(|c| absorbed_bytes(c).map(|b| format!("{c}={b}")))
            .collect();
        writeln!(out, "# absorbed-bytes-per-call: {}", rates.join(",")).expect("write provenance");
    }
    writeln!(out, "# headline-selection: {}", selection.headline).expect("write provenance");
    writeln!(out, "# intro-selection: {}", selection.intro).expect("write provenance");
}
