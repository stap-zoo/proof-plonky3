//! The benchmark command: one command produces everything measured.
//!
//! POLICY §11 requires every number in the tables to come out of here, selected
//! by flags, with no second script that measures things slightly differently.
//! The default work list is [`bench::plan`], the typed registry of every AIR
//! this repository can measure; the flags only narrow it.
//!
//! ```text
//! RAYON_NUM_THREADS=16 cargo run --release --bin bench > rows.csv
//! ```
//!
//! CSV rows go to stdout, everything else — progress, skipped scope, the pinned
//! environment's own `#` block goes to stdout above the header — so a redirect
//! is the whole interface.
//!
//! Measurements are **release-only**: `prove` runs `check_constraints` itself
//! under `debug_assertions`, so a debug number is a number for a different
//! program (POLICY §10). A debug build refuses to measure and offers
//! `--dry-run`, which only lists the selection.

use std::io::{BufWriter, Write};

use bench::runner::{Selection, dry_run, run, write_provenance};
use clap::Parser;
use harness::{FieldId, Reading, Zk};

/// Measure the arithmetization cost of every registered permutation.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Measure only this construction, e.g. `griffin`.
    #[arg(long)]
    construction: Option<String>,

    /// Measure only instances whose name contains this string.
    #[arg(long)]
    instance: Option<String>,

    /// Measure only this variant, e.g. `split`, `full-round`, `flattened`.
    #[arg(long)]
    variant: Option<String>,

    /// Fields to measure: goldilocks, mersenne31, babybear, koalabear.
    #[arg(long, value_delimiter = ',')]
    field: Vec<String>,

    /// Which side of POLICY §7's ZK toggle: off, on, both.
    #[arg(long, default_value = "off")]
    zk: String,

    /// Which blowup reading POLICY §11 reports: minimum, common, both.
    ///
    /// `minimum` runs every AIR at the smallest blowup its own declared degree
    /// admits, so the degree is a parameter of the row and everything
    /// downstream is priced at the rate it bought. `common` runs everything at
    /// one blowup, which prices what a high degree saves and not what it costs.
    #[arg(long, default_value = "minimum")]
    reading: String,

    /// Trace heights to sweep, as log2. Per-call cost is the amortized figure
    /// from a sweep, never a single point.
    #[arg(long, value_delimiter = ',', default_values_t = [10usize, 12, 14])]
    log_n: Vec<usize>,

    /// List the selection instead of measuring it. The only mode a debug build
    /// may run.
    #[arg(long)]
    dry_run: bool,

    /// Select one lowest-alpha, lowest-degree instance per construction/field.
    /// Where a lookup-backed AIR exists, select it instead of the ordinary AIR.
    #[arg(long)]
    headline: bool,

    /// Measure the introduction table's list instead of the construction grid:
    /// BLAKE3, SHA-256 and Keccak-f as Plonky3 arithmetizes them, plus the
    /// paper's headline Poseidon2 row. KoalaBear only.
    #[arg(long)]
    intro: bool,

    /// Size every trace by the work it proves: each job runs at the smallest
    /// height whose layout proves at least this many permutation calls, instead
    /// of at the `--log-n` sweep.
    ///
    /// A layout of more than one row per call cannot fill a power-of-two table
    /// at an arbitrary count, so its row reports the count it actually reached.
    #[arg(long)]
    calls: Option<usize>,

    /// Size every trace by the message bytes it covers: each job runs at the
    /// largest height whose batch does not exceed this many bytes.
    ///
    /// Takes precedence over `--calls`. This is the equal-*data* comparison;
    /// `--calls` is the equal-*work* one, and they are different questions
    /// because one call absorbs a different amount per design.
    #[arg(long)]
    bytes: Option<usize>,
}

/// Parse a field name, or list the four.
fn field(name: &str) -> FieldId {
    match name {
        "goldilocks" => FieldId::Goldilocks,
        "mersenne31" => FieldId::Mersenne31,
        "babybear" => FieldId::BabyBear,
        "koalabear" => FieldId::KoalaBear,
        other => panic!("unknown field `{other}`: goldilocks, mersenne31, babybear, koalabear"),
    }
}

fn main() {
    let args = Args::parse();

    let selection = Selection {
        construction: args.construction,
        instance: args.instance,
        variant: args.variant,
        fields: if args.field.is_empty() {
            harness::GRID
                .iter()
                .map(|&(f, _)| f)
                .fold(Vec::new(), |mut fields: Vec<FieldId>, f| {
                    if !fields.contains(&f) {
                        fields.push(f);
                    }
                    fields
                })
        } else {
            args.field.iter().map(|name| field(name)).collect()
        },
        zk: match args.zk.as_str() {
            "off" => vec![Zk::Off],
            "on" => vec![Zk::On],
            "both" => vec![Zk::Off, Zk::On],
            other => panic!("unknown --zk `{other}`: off, on, both"),
        },
        readings: match args.reading.as_str() {
            "minimum" => vec![Reading::Minimum],
            "common" => vec![Reading::Common],
            "both" => vec![Reading::Minimum, Reading::Common],
            other => panic!("unknown --reading `{other}`: minimum, common, both"),
        },
        log_n: args.log_n,
        headline: args.headline,
        intro: args.intro,
        calls: args.calls,
        bytes: args.bytes,
    };
    assert!(
        !selection.log_n.is_empty(),
        "--log-n needs at least one trace height"
    );
    assert!(
        selection.calls.is_none_or(|calls| calls > 0),
        "--calls needs at least one permutation call"
    );
    assert!(
        selection.bytes.is_none_or(|bytes| bytes > 0),
        "--bytes needs at least one message byte"
    );

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // The registry, on stderr: what exists, before what is measured.
    let instances = bench::instances();
    let present = instances.iter().filter(|i| i.instance.is_some()).count();
    eprintln!(
        "registry: {} constructions, {present} instances present, {} absent (POLICY §3)",
        bench::CONSTRUCTIONS.len(),
        instances.len() - present,
    );

    if args.dry_run {
        let listed = dry_run(&selection, &mut out);
        out.flush().expect("flush");
        eprintln!("dry run: {listed} measurement(s) selected");
        return;
    }

    let environment = harness::Environment::current();
    assert!(
        environment.is_measurable(),
        "this is a {} build and measurements are release-only (POLICY §11): \
         run `cargo run --release --bin bench`, or `--dry-run` to list the selection",
        environment.profile
    );

    write_provenance(&selection, &mut out);
    let summary = run(&selection, &mut out);
    out.flush().expect("flush");

    eprintln!(
        "{} row(s), {} skipped, {} empty cell(s)",
        summary.rows, summary.skipped, summary.empty_cells
    );
}
