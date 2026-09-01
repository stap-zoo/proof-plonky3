//! Every construction's instances, the guards that only make sense across
//! constructions, and the work list the command measures.
//!
//! `harness` never learns which hashes exist; this crate is where they are
//! known (POLICY §5). Constructions depend on `harness`, so the aggregation
//! cannot live there, and putting it here keeps the direction one-way:
//! `<construction> → harness`, `bench → <construction>`.
//!
//! It is a library and a binary because the aggregation has two consumers that
//! must not disagree: the command in `main.rs`, and the cross-construction
//! tests. Four things live here and nowhere else:
//!
//! * the concatenated instance list, which the tests and the command filter;
//! * the **coverage guard**. An instance's name is the only thing tying an
//!   implementation to its vectors (POLICY §3), and a mismatch is silent — a
//!   renamed instance quietly stops being validated instead of failing. The
//!   guard is what makes that loud;
//! * the [`plan`], the typed work list of POLICY §11: every AIR this repository
//!   can measure, each named by its type;
//! * the [`runner`], which crosses that list with POLICY §7's configuration
//!   matrix and writes one row schema.
//!
//! `tests/plan.rs` is where the two lists are tied together — every registered
//! instance has a job, and every job names a registered instance.

pub mod coverage;
pub mod plan;
pub mod runner;

pub use coverage::{Absence, CONSTRUCTIONS, Construction, GridPoint, Registration};

/// Every instance registered by every construction.
///
/// It grows by concatenating each construction's own `INSTANCES`; nothing is
/// declared here that the construction does not declare about itself — including
/// the absences, which POLICY §3 makes entries rather than gaps.
///
/// `const` concatenation of slices is not available, so this is a `&[&[_]]`
/// flattened by the guards. The alternative — copying each construction's entries
/// here — would be the duplication the one-way dependency exists to prevent.
pub const REGISTERED: &[&[GridPoint]] = &[
    poseidon2::instances::INSTANCES,
    poseidon1::instances::INSTANCES,
    monolith::instances::INSTANCES,
    anemoi::instances::INSTANCES,
    griffin::instances::INSTANCES,
    tip5::instances::INSTANCES,
    rescue_prime::instances::INSTANCES,
    neptune::instances::INSTANCES,
    psquarehash::instances::INSTANCES,
    gmimc::instances::INSTANCES,
    gmimc2::instances::INSTANCES,
    // One crate, two grid rows: `../ref` has one `XHash` class for all four
    // instances, but the grid compares (construction, field, width) points.
    xhash::instances::GOLDILOCKS_INSTANCES,
    xhash::instances::MERSENNE31_INSTANCES,
];

/// [`REGISTERED`], flattened.
#[must_use]
pub fn instances() -> Vec<GridPoint> {
    REGISTERED
        .iter()
        .flat_map(|xs| xs.iter().copied())
        .collect()
}
