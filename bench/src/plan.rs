//! The work list: every AIR this repository can measure, named by its type.
//!
//! A plan entry is a **typed reference into a construction's own registry**
//! (POLICY §11) — a symbol you can jump to, which cannot resolve to nothing,
//! and which can name one of two arithmetizations of the same hash. Not a
//! string to recognise: a string plan silently measures nothing when an
//! instance is renamed, which is precisely the failure the coverage guard
//! exists to prevent elsewhere. Here that guarantee is the compiler's — every
//! entry below names an AIR type and a parameter constructor, so an
//! arithmetization that stops existing stops this crate from building.
//!
//! # One list per field, both sides of the ZK toggle
//!
//! Each function is generic in the configuration but pins `Val<SC>` to its
//! field, which is what lets one list serve that field's ZK and non-ZK cells
//! (POLICY §7: ZK is a configuration axis, never a construction's property).
//! The pin is the `Domain: PolynomialSpace<Val = ..>` bound; without it the AIR
//! types would not line up with the prover's `Val` and every entry would need
//! writing twice.
//!
//! # The Mersenne-31 ZK cell has no list
//!
//! There is deliberately no `mersenne31_zk`: POLICY §7 declares that
//! configuration unsupported. The runner reports the empty cell; see
//! [`MERSENNE31_ZK`].
//!
//! # What a variant is
//!
//! The cross product POLICY §11 measures is a construction's grid × its
//! register variants × its arithmetizations. The `variant` string names the
//! last two — `plain` and `split` for the S-box register axis, `half-round` and
//! `full-round` for Rescue-Prime's two layouts, and pSquareHash's own names for
//! its three axes: how much of a round is flattened (`flattened`, `one-register`,
//! `state-only`), how many rounds separate two state commitments (`spaced`,
//! `spaced-split`), and how much of one Feistel a commitment covers
//! (`half-commit`). The instance name is the first. Everything else about the
//! shape is in `Labels` and reaches the table there.

// The lists below are registries: one line per entry, appended in the order a
// reader wants to scan them. `vec![..]` would move every AIR type inside one
// expression and lose the per-entry comments that say why a variant exists.
#![allow(clippy::vec_init_then_push)]

use harness::{FieldId, Job, Plan};
use p3_air::symbolic::SymbolicExpressionExt;
use p3_baby_bear::BabyBear;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{Algebra, BasedVectorSpace};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use p3_uni_stark::StarkGenericConfig;

/// Why POLICY §7's ZK Mersenne-31 cell carries no rows.
pub const MERSENNE31_ZK: &str =
    "unsupported: Plonky3 has no hiding Mersenne-31 circle PCS (POLICY §7)";

/// Push one job. The AIR type is named at the call site, which is the whole
/// point of the list; the closure defers building it out of every timed region.
macro_rules! job {
    ($jobs:ident, $construction:literal, $instance:literal, $variant:literal, $air:expr) => {
        $jobs.push(Job::new($construction, $instance, $variant, || $air))
    };
}

/// One job per registered Tip5 lookup variant of one named instance.
///
/// The frontier's `registers` label selects one of two AIR *types* — the
/// seventh-power words are degree 7 unsplit and 3 with one register — so this
/// names both and lets the variant pick. It is deliberately not the naive
/// `2 * 2 * 4` product: [`tip5::logup::VARIANTS`] is the undominated frontier,
/// and pushing anything else here would measure a point that is strictly worse
/// than one already in the list (POLICY §11).
macro_rules! tip5_lookup_jobs {
    ($jobs:ident, $instance:literal, $plain:ty, $split:ty, $params:path) => {
        for variant in tip5::logup::VARIANTS {
            let variant = *variant;
            if variant.registers == 1 {
                $jobs.push(Job::new_lookup(
                    "tip5",
                    $instance,
                    variant.name,
                    move || <$split>::from_params(&$params(), variant.granularity, variant.packing),
                ));
            } else {
                $jobs.push(Job::new_lookup(
                    "tip5",
                    $instance,
                    variant.name,
                    move || <$plain>::from_params(&$params(), variant.granularity, variant.packing),
                ));
            }
        }
    };
}

macro_rules! monolith_lookup_jobs {
    ($jobs:ident, $instance:literal, $constructor:path) => {
        for variant in monolith::logup::VARIANTS {
            let variant = *variant;
            $jobs.push(Job::new_lookup(
                "monolith",
                $instance,
                variant.name,
                move || $constructor(variant.granularity, variant.packing),
            ));
        }
    };
}

/// Goldilocks: the grid's t = 8 and t = 12, plus the Tip5 family's own widths.
// `rustfmt::skip`: this is a registry, and one entry per line is what makes it
// scannable. Formatted, each `job!` becomes seven lines and the list stops
// reading as a table of instance × variant.
#[rustfmt::skip]
#[must_use]
pub fn goldilocks<SC>() -> Plan<SC>
where
    SC: StarkGenericConfig + 'static,
    SC::Challenge: BasedVectorSpace<Goldilocks>,
    SymbolicExpressionExt<Goldilocks, SC::Challenge>: Algebra<SC::Challenge>,
    SC::Pcs: Pcs<
            SC::Challenge,
            SC::Challenger,
            Domain: PolynomialSpace<Val = Goldilocks> + Send + Sync,
        > + Sync,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::ProverData: Sync,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment: Sync,
{
    use anemoi::instances as anemoi_air;
    use griffin::instances as griffin_air;
    use rescue_prime::instances as rp;
    use rescue_prime::instances::full as rp_full;
    use tip5::instances as tip5_air;
    use tip5::logup as tip5_logup;
    use xhash::instances as xhash_air;

    let mut jobs = Vec::new();

    // Poseidon2 and Poseidon1: upstream's own register table, `<0>` and `<1>`.
    job!(jobs, "poseidon2", "poseidon2-goldilocks-t8", "plain",
        poseidon2::instances::GoldilocksT8::<0>::new(poseidon2::params::goldilocks_t8()));
    job!(jobs, "poseidon2", "poseidon2-goldilocks-t8", "split",
        poseidon2::instances::GoldilocksT8::<1>::new(poseidon2::params::goldilocks_t8()));
    job!(jobs, "poseidon2", "poseidon2-goldilocks-t12", "plain",
        poseidon2::instances::GoldilocksT12::<0>::new(poseidon2::params::goldilocks_t12()));
    job!(jobs, "poseidon2", "poseidon2-goldilocks-t12", "split",
        poseidon2::instances::GoldilocksT12::<1>::new(poseidon2::params::goldilocks_t12()));

    job!(jobs, "poseidon1", "poseidon1-goldilocks-t8", "plain",
        poseidon1::instances::GoldilocksT8::<0>::from_raw(&poseidon1::params::goldilocks_t8()));
    job!(jobs, "poseidon1", "poseidon1-goldilocks-t8", "split",
        poseidon1::instances::GoldilocksT8::<1>::from_raw(&poseidon1::params::goldilocks_t8()));
    job!(jobs, "poseidon1", "poseidon1-goldilocks-t12", "plain",
        poseidon1::instances::GoldilocksT12::<0>::from_raw(&poseidon1::params::goldilocks_t12()));
    job!(jobs, "poseidon1", "poseidon1-goldilocks-t12", "split",
        poseidon1::instances::GoldilocksT12::<1>::from_raw(&poseidon1::params::goldilocks_t12()));

    // Monolith: Bars decomposed in-AIR, no lookup, one arithmetization.
    job!(jobs, "monolith", "monolith-goldilocks-t8", "in-air-bars",
        monolith::params::goldilocks_t8());
    job!(jobs, "monolith", "monolith-goldilocks-t12", "in-air-bars",
        monolith::params::goldilocks_t12());
    monolith_lookup_jobs!(jobs, "monolith-goldilocks-t8", monolith::logup::goldilocks_t8_with);
    monolith_lookup_jobs!(jobs, "monolith-goldilocks-t12", monolith::logup::goldilocks_t12_with);

    job!(jobs, "anemoi", "anemoi-goldilocks-t8", "plain",
        anemoi_air::GoldilocksT8::from_params(&anemoi::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "anemoi", "anemoi-goldilocks-t8", "split",
        anemoi_air::GoldilocksT8Split::from_params(&anemoi::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "anemoi", "anemoi-goldilocks-t12", "plain",
        anemoi_air::GoldilocksT12::from_params(&anemoi::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "anemoi", "anemoi-goldilocks-t12", "split",
        anemoi_air::GoldilocksT12Split::from_params(&anemoi::params::goldilocks_t12::<Goldilocks>()));

    job!(jobs, "griffin", "griffin-goldilocks-t8", "plain",
        griffin_air::GoldilocksT8::from_params(&griffin::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "griffin", "griffin-goldilocks-t8", "split",
        griffin_air::GoldilocksT8Split::from_params(&griffin::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "griffin", "griffin-goldilocks-t12", "plain",
        griffin_air::GoldilocksT12::from_params(&griffin::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "griffin", "griffin-goldilocks-t12", "split",
        griffin_air::GoldilocksT12Split::from_params(&griffin::params::goldilocks_t12::<Goldilocks>()));

    // GMiMC and GMiMC2: one committed nonlinear input per round. The register
    // axis is complete here: alpha 7 goes 7 -> 3 and alpha 4 goes 4 -> 2.
    job!(jobs, "gmimc", "gmimc-goldilocks-t12", "plain",
        gmimc::instances::GoldilocksT12::from_params(&gmimc::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "gmimc", "gmimc-goldilocks-t12", "split",
        gmimc::instances::GoldilocksT12Split::from_params(&gmimc::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "gmimc2", "gmimc2-goldilocks-t12", "plain",
        gmimc2::instances::GoldilocksT12::from_params(&gmimc2::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "gmimc2", "gmimc2-goldilocks-t12", "split",
        gmimc2::instances::GoldilocksT12Split::from_params(&gmimc2::params::goldilocks_t12::<Goldilocks>()));

    // Neptune's register axis is two-dimensional, because two layers set its
    // degree: the external Lai-Massey pair map, degree four until its first
    // square is committed, and the internal `x^7`. `flattened` commits both —
    // one square per pair per external round and the three-cell power chain per
    // internal round — and is the only other point worth measuring, since
    // cutting either layer alone leaves the other standing (POLICY §11;
    // `neptune::columns::assert_layout` refuses the mixed variants).
    job!(jobs, "neptune", "neptune-goldilocks-t8", "plain",
        neptune::instances::goldilocks_t8_air());
    job!(jobs, "neptune", "neptune-goldilocks-t8", "flattened",
        neptune::instances::goldilocks_t8_flattened_air());
    job!(jobs, "neptune", "neptune-goldilocks-t12", "plain",
        neptune::instances::goldilocks_t12_air());
    job!(jobs, "neptune", "neptune-goldilocks-t12", "flattened",
        neptune::instances::goldilocks_t12_flattened_air());

    // Rescue-Prime, arithmetized twice over the same instances. Which layout
    // the tables carry is exactly what these rows decide (README, open items).
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t8", "half-round",
        rp::GoldilocksT8::from_params(&rescue_prime::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t8", "half-round-split",
        rp::GoldilocksT8Split::from_params(&rescue_prime::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t8", "full-round",
        rp_full::GoldilocksT8::from_params(&rescue_prime::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t8", "full-round-split",
        rp_full::GoldilocksT8Split::from_params(&rescue_prime::params::goldilocks_t8::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12", "half-round",
        rp::GoldilocksT12::from_params(&rescue_prime::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12", "half-round-split",
        rp::GoldilocksT12Split::from_params(&rescue_prime::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12", "full-round",
        rp_full::GoldilocksT12::from_params(&rescue_prime::params::goldilocks_t12::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12", "full-round-split",
        rp_full::GoldilocksT12Split::from_params(&rescue_prime::params::goldilocks_t12::<Goldilocks>()));
    // The same t = 12 point at the author's R = 13 rather than the reference's
    // R = 8 (POLICY §4). Both are measured; the headline table publishes this
    // one, which `bench::runner::headline_rounds` is what decides.
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12-r13", "half-round",
        rp::GoldilocksT12R13::from_params(&rescue_prime::params::goldilocks_t12_r13::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12-r13", "half-round-split",
        rp::GoldilocksT12R13Split::from_params(&rescue_prime::params::goldilocks_t12_r13::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12-r13", "full-round",
        rp_full::GoldilocksT12R13::from_params(&rescue_prime::params::goldilocks_t12_r13::<Goldilocks>()));
    job!(jobs, "rescue-prime", "rescue-prime-goldilocks-t12-r13", "full-round-split",
        rp_full::GoldilocksT12R13Split::from_params(&rescue_prime::params::goldilocks_t12_r13::<Goldilocks>()));

    // Tip5 family. `tip4` and `tip5` share their permutation parameters and
    // differ only in sponge parameters, which are out of scope (POLICY §8);
    // both names are registered instances, so both are measured.
    job!(jobs, "tip5", "tip4-prime", "plain",
        tip5_air::GoldilocksT12::from_params(&tip5::params::tip4_prime::<Goldilocks>()));
    job!(jobs, "tip5", "tip4-prime", "split",
        tip5_air::GoldilocksT12Split::from_params(&tip5::params::tip4_prime::<Goldilocks>()));
    job!(jobs, "tip5", "tip4", "plain",
        tip5_air::GoldilocksT16::from_params(&tip5::params::tip4::<Goldilocks>()));
    job!(jobs, "tip5", "tip4", "split",
        tip5_air::GoldilocksT16Split::from_params(&tip5::params::tip4::<Goldilocks>()));
    job!(jobs, "tip5", "tip5", "plain",
        tip5_air::GoldilocksT16::from_params(&tip5::params::tip5::<Goldilocks>()));
    job!(jobs, "tip5", "tip5", "split",
        tip5_air::GoldilocksT16Split::from_params(&tip5::params::tip5::<Goldilocks>()));

    // The same three permutations, arithmetized a second time with the
    // split-and-lookup S-box (POLICY §12). The lookup-free rows above are
    // unchanged; what these price is 272 committed cells per split word
    // becoming eighteen and a table query.
    tip5_lookup_jobs!(jobs, "tip4-prime",
        tip5_logup::GoldilocksT12<Goldilocks>, tip5_logup::GoldilocksT12Split<Goldilocks>,
        tip5::params::tip4_prime);
    tip5_lookup_jobs!(jobs, "tip4",
        tip5_logup::GoldilocksT16<Goldilocks>, tip5_logup::GoldilocksT16Split<Goldilocks>,
        tip5::params::tip4);
    tip5_lookup_jobs!(jobs, "tip5",
        tip5_logup::GoldilocksT16<Goldilocks>, tip5_logup::GoldilocksT16Split<Goldilocks>,
        tip5::params::tip5);

    job!(jobs, "xhash8", "xhash8-goldilocks-t12", "plain",
        xhash_air::XHash8::from_params(&xhash::params::xhash8::<Goldilocks>()));
    job!(jobs, "xhash8", "xhash8-goldilocks-t12", "split",
        xhash_air::XHash8Split::from_params(&xhash::params::xhash8::<Goldilocks>()));
    job!(jobs, "xhash8", "xhash12-goldilocks-t12", "plain",
        xhash_air::XHash12::from_params(&xhash::params::xhash12::<Goldilocks>()));
    job!(jobs, "xhash8", "xhash12-goldilocks-t12", "split",
        xhash_air::XHash12Split::from_params(&xhash::params::xhash12::<Goldilocks>()));

    Plan {
        field: FieldId::Goldilocks,
        jobs,
    }
}

/// Mersenne-31, non-ZK: t = 16 and t = 24. The ZK cell has no list at all;
/// see [`MERSENNE31_ZK`].
// `rustfmt::skip`: this is a registry, and one entry per line is what makes it
// scannable. Formatted, each `job!` becomes seven lines and the list stops
// reading as a table of instance × variant.
#[rustfmt::skip]
#[must_use]
pub fn mersenne31<SC>() -> Plan<SC>
where
    SC: StarkGenericConfig + 'static,
    SC::Challenge: BasedVectorSpace<Mersenne31>,
    SymbolicExpressionExt<Mersenne31, SC::Challenge>: Algebra<SC::Challenge>,
    SC::Pcs: Pcs<
            SC::Challenge,
            SC::Challenger,
            Domain: PolynomialSpace<Val = Mersenne31> + Send + Sync,
        > + Sync,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::ProverData: Sync,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment: Sync,
{
    use anemoi::instances as anemoi_air;
    use psquarehash::instances as psq;
    use rescue_prime::instances as rp;
    use rescue_prime::instances::full as rp_full;
    use xhash::instances as xhash_air;

    let mut jobs = Vec::new();

    job!(jobs, "poseidon2", "poseidon2-mersenne-t16", "plain",
        poseidon2::instances::MersenneT16::<0>::new(poseidon2::params::mersenne_t16()));
    job!(jobs, "poseidon2", "poseidon2-mersenne-t16", "split",
        poseidon2::instances::MersenneT16::<1>::new(poseidon2::params::mersenne_t16()));
    job!(jobs, "poseidon2", "poseidon2-mersenne-t24", "plain",
        poseidon2::instances::MersenneT24::<0>::new(poseidon2::params::mersenne_t24()));
    job!(jobs, "poseidon2", "poseidon2-mersenne-t24", "split",
        poseidon2::instances::MersenneT24::<1>::new(poseidon2::params::mersenne_t24()));

    // Poseidon1 has no t = 24 row here: upstream pins no parameters at that
    // point (`Absence::UndefinedUpstream`), and we wrap rather than re-port.
    job!(jobs, "poseidon1", "poseidon1-mersenne-t16", "plain",
        poseidon1::instances::MersenneT16::<0>::from_raw(&poseidon1::params::mersenne_t16()));
    job!(jobs, "poseidon1", "poseidon1-mersenne-t16", "split",
        poseidon1::instances::MersenneT16::<1>::from_raw(&poseidon1::params::mersenne_t16()));

    job!(jobs, "monolith", "monolith-mersenne-t16", "in-air-bars",
        monolith::params::mersenne_t16());
    job!(jobs, "monolith", "monolith-mersenne-t24", "in-air-bars",
        monolith::params::mersenne_t24());
    monolith_lookup_jobs!(jobs, "monolith-mersenne-t16", monolith::logup::mersenne_t16_with);
    monolith_lookup_jobs!(jobs, "monolith-mersenne-t24", monolith::logup::mersenne_t24_with);

    // Anemoi's t = 24 derivation is a reference stub, so t = 16 only.
    job!(jobs, "anemoi", "anemoi-mersenne-t16", "plain",
        anemoi_air::MersenneT16::from_params(&anemoi::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "anemoi", "anemoi-mersenne-t16", "split",
        anemoi_air::MersenneT16Split::from_params(&anemoi::params::mersenne_t16::<Mersenne31>()));

    job!(jobs, "gmimc", "gmimc-mersenne-t24", "plain",
        gmimc::instances::MersenneT24::from_params(&gmimc::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "gmimc", "gmimc-mersenne-t24", "split",
        gmimc::instances::MersenneT24Split::from_params(&gmimc::params::mersenne_t24::<Mersenne31>()));
    // Alpha 2 is already at the degree/cell floor, so GMiMC2 has no register row.
    job!(jobs, "gmimc2", "gmimc2-mersenne-t24", "plain",
        gmimc2::instances::MersenneT24::from_params(&gmimc2::params::mersenne_t24::<Mersenne31>()));

    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t16", "half-round",
        rp::MersenneT16::from_params(&rescue_prime::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t16", "half-round-split",
        rp::MersenneT16Split::from_params(&rescue_prime::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t16", "full-round",
        rp_full::MersenneT16::from_params(&rescue_prime::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t16", "full-round-split",
        rp_full::MersenneT16Split::from_params(&rescue_prime::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t24", "half-round",
        rp::MersenneT24::from_params(&rescue_prime::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t24", "half-round-split",
        rp::MersenneT24Split::from_params(&rescue_prime::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t24", "full-round",
        rp_full::MersenneT24::from_params(&rescue_prime::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "rescue-prime", "rescue-prime-mersenne-t24", "full-round-split",
        rp_full::MersenneT24Split::from_params(&rescue_prime::params::mersenne_t24::<Mersenne31>()));

    // pSquareHash has both of POLICY §11's variant axes, and a third of its own.
    // How much of a round is flattened: `state-only` commits the state and
    // nothing else, `one-register` is the ported baseline, `flattened` commits
    // both squarings and drops the state columns entirely. How many rounds
    // separate two commitments: `spaced-split` and `spaced` commit once every
    // second round, going below `flattened`'s width floor and paying degree 8 and
    // 16 for it. And how much of one Feistel a commitment covers: `half-commit`
    // commits one output of each pair every round and differences the other,
    // reaching `spaced`'s width at degree 4 — a different arithmetization
    // (`psquarehash::half_commit`), not another parameter set, which is why its
    // entry names a different AIR type.
    job!(jobs, "psquarehash", "psquarehash-mersenne-t16", "flattened",
        psq::Flattened16::from_params(&psquarehash::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t16", "state-only",
        psq::StateOnly16::from_params(&psquarehash::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t16", "one-register",
        psq::OneRegister16::from_params(&psquarehash::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t16", "spaced-split",
        psq::SpacedSplit16::from_params(&psquarehash::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t16", "spaced",
        psq::Spaced16::from_params(&psquarehash::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t16", "half-commit",
        psq::HalfCommit16::from_params(&psquarehash::params::mersenne_t16::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t24", "flattened",
        psq::Flattened24::from_params(&psquarehash::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t24", "state-only",
        psq::StateOnly24::from_params(&psquarehash::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t24", "one-register",
        psq::OneRegister24::from_params(&psquarehash::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t24", "spaced-split",
        psq::SpacedSplit24::from_params(&psquarehash::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t24", "spaced",
        psq::Spaced24::from_params(&psquarehash::params::mersenne_t24::<Mersenne31>()));
    job!(jobs, "psquarehash", "psquarehash-mersenne-t24", "half-commit",
        psq::HalfCommit24::from_params(&psquarehash::params::mersenne_t24::<Mersenne31>()));

    job!(jobs, "xhash16", "xhash16-m31-t24", "plain",
        xhash_air::XHash16::from_params(&xhash::params::xhash16::<Mersenne31>()));
    job!(jobs, "xhash16", "xhash16-m31-t24", "split",
        xhash_air::XHash16Split::from_params(&xhash::params::xhash16::<Mersenne31>()));
    job!(jobs, "xhash16", "xhash24-m31-t24", "plain",
        xhash_air::XHash24::from_params(&xhash::params::xhash24::<Mersenne31>()));
    job!(jobs, "xhash16", "xhash24-m31-t24", "split",
        xhash_air::XHash24Split::from_params(&xhash::params::xhash24::<Mersenne31>()));

    Plan {
        field: FieldId::Mersenne31,
        jobs,
    }
}

/// BabyBear: t = 16 and t = 24.
// `rustfmt::skip`: this is a registry, and one entry per line is what makes it
// scannable. Formatted, each `job!` becomes seven lines and the list stops
// reading as a table of instance × variant.
#[rustfmt::skip]
#[must_use]
pub fn babybear<SC>() -> Plan<SC>
where
    SC: StarkGenericConfig + 'static,
    SC::Pcs: Pcs<SC::Challenge, SC::Challenger, Domain: PolynomialSpace<Val = BabyBear>>,
{
    use anemoi::instances as anemoi_air;
    use psquarehash::instances as psq;
    use rescue_prime::instances as rp;
    use rescue_prime::instances::full as rp_full;

    let mut jobs = Vec::new();

    job!(jobs, "poseidon2", "poseidon2-babybear-t16", "plain",
        poseidon2::instances::BabyBearT16::<0>::new(poseidon2::params::babybear_t16()));
    job!(jobs, "poseidon2", "poseidon2-babybear-t16", "split",
        poseidon2::instances::BabyBearT16::<1>::new(poseidon2::params::babybear_t16()));
    job!(jobs, "poseidon2", "poseidon2-babybear-t24", "plain",
        poseidon2::instances::BabyBearT24::<0>::new(poseidon2::params::babybear_t24()));
    job!(jobs, "poseidon2", "poseidon2-babybear-t24", "split",
        poseidon2::instances::BabyBearT24::<1>::new(poseidon2::params::babybear_t24()));

    job!(jobs, "poseidon1", "poseidon1-babybear-t16", "plain",
        poseidon1::instances::BabyBearT16::<0>::from_raw(&poseidon1::params::babybear_t16()));
    job!(jobs, "poseidon1", "poseidon1-babybear-t16", "split",
        poseidon1::instances::BabyBearT16::<1>::from_raw(&poseidon1::params::babybear_t16()));
    job!(jobs, "poseidon1", "poseidon1-babybear-t24", "plain",
        poseidon1::instances::BabyBearT24::<0>::from_raw(&poseidon1::params::babybear_t24()));
    job!(jobs, "poseidon1", "poseidon1-babybear-t24", "split",
        poseidon1::instances::BabyBearT24::<1>::from_raw(&poseidon1::params::babybear_t24()));

    job!(jobs, "anemoi", "anemoi-babybear-t16", "plain",
        anemoi_air::BabyBearT16::from_params(&anemoi::params::babybear_t16::<BabyBear>()));
    job!(jobs, "anemoi", "anemoi-babybear-t16", "split",
        anemoi_air::BabyBearT16Split::from_params(&anemoi::params::babybear_t16::<BabyBear>()));

    job!(jobs, "gmimc", "gmimc-babybear-t24", "plain",
        gmimc::instances::BabyBearT24::from_params(&gmimc::params::babybear_t24::<BabyBear>()));
    job!(jobs, "gmimc", "gmimc-babybear-t24", "split",
        gmimc::instances::BabyBearT24Split::from_params(&gmimc::params::babybear_t24::<BabyBear>()));
    job!(jobs, "gmimc2", "gmimc2-babybear-t24", "plain",
        gmimc2::instances::BabyBearT24::from_params(&gmimc2::params::babybear_t24::<BabyBear>()));

    job!(jobs, "rescue-prime", "rescue-prime-babybear-t16", "half-round",
        rp::BabyBearT16::from_params(&rescue_prime::params::babybear_t16::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t16", "half-round-split",
        rp::BabyBearT16Split::from_params(&rescue_prime::params::babybear_t16::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t16", "full-round",
        rp_full::BabyBearT16::from_params(&rescue_prime::params::babybear_t16::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t16", "full-round-split",
        rp_full::BabyBearT16Split::from_params(&rescue_prime::params::babybear_t16::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t24", "half-round",
        rp::BabyBearT24::from_params(&rescue_prime::params::babybear_t24::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t24", "half-round-split",
        rp::BabyBearT24Split::from_params(&rescue_prime::params::babybear_t24::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t24", "full-round",
        rp_full::BabyBearT24::from_params(&rescue_prime::params::babybear_t24::<BabyBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-babybear-t24", "full-round-split",
        rp_full::BabyBearT24Split::from_params(&rescue_prime::params::babybear_t24::<BabyBear>()));

    job!(jobs, "psquarehash", "psquarehash-babybear-t16", "flattened",
        psq::Flattened16::from_params(&psquarehash::params::babybear_t16::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t16", "state-only",
        psq::StateOnly16::from_params(&psquarehash::params::babybear_t16::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t16", "one-register",
        psq::OneRegister16::from_params(&psquarehash::params::babybear_t16::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t16", "spaced-split",
        psq::SpacedSplit16::from_params(&psquarehash::params::babybear_t16::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t16", "spaced",
        psq::Spaced16::from_params(&psquarehash::params::babybear_t16::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t16", "half-commit",
        psq::HalfCommit16::from_params(&psquarehash::params::babybear_t16::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t24", "flattened",
        psq::Flattened24::from_params(&psquarehash::params::babybear_t24::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t24", "state-only",
        psq::StateOnly24::from_params(&psquarehash::params::babybear_t24::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t24", "one-register",
        psq::OneRegister24::from_params(&psquarehash::params::babybear_t24::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t24", "spaced-split",
        psq::SpacedSplit24::from_params(&psquarehash::params::babybear_t24::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t24", "spaced",
        psq::Spaced24::from_params(&psquarehash::params::babybear_t24::<BabyBear>()));
    job!(jobs, "psquarehash", "psquarehash-babybear-t24", "half-commit",
        psq::HalfCommit24::from_params(&psquarehash::params::babybear_t24::<BabyBear>()));

    Plan {
        field: FieldId::BabyBear,
        jobs,
    }
}

/// KoalaBear: t = 16 and t = 24.
///
/// Most alpha-3 constructions have no register variant because another layer
/// already sets degree three. GMiMC is the exception: its recurrence has no such
/// floor, so a register takes its alpha-3 S-box to degree two and both rows are
/// measured.
// `rustfmt::skip`: this is a registry, and one entry per line is what makes it
// scannable. Formatted, each `job!` becomes seven lines and the list stops
// reading as a table of instance × variant.
#[rustfmt::skip]
#[must_use]
pub fn koalabear<SC>() -> Plan<SC>
where
    SC: StarkGenericConfig + 'static,
    SC::Pcs: Pcs<SC::Challenge, SC::Challenger, Domain: PolynomialSpace<Val = KoalaBear>>,
{
    use anemoi::instances as anemoi_air;
    use psquarehash::instances as psq;
    use rescue_prime::instances as rp;
    use rescue_prime::instances::full as rp_full;

    let mut jobs = Vec::new();

    job!(jobs, "poseidon2", "poseidon2-koalabear-t16", "plain",
        poseidon2::instances::KoalaBearT16::<0>::new(poseidon2::params::koalabear_t16()));
    job!(jobs, "poseidon2", "poseidon2-koalabear-t24", "plain",
        poseidon2::instances::KoalaBearT24::<0>::new(poseidon2::params::koalabear_t24()));

    job!(jobs, "poseidon1", "poseidon1-koalabear-t16", "plain",
        poseidon1::instances::KoalaBearT16::<0>::from_raw(&poseidon1::params::koalabear_t16()));
    job!(jobs, "poseidon1", "poseidon1-koalabear-t24", "plain",
        poseidon1::instances::KoalaBearT24::<0>::from_raw(&poseidon1::params::koalabear_t24()));

    job!(jobs, "anemoi", "anemoi-koalabear-t16", "plain",
        anemoi_air::KoalaBearT16::from_params(&anemoi::params::koalabear_t16::<KoalaBear>()));

    job!(jobs, "gmimc", "gmimc-koalabear-t24", "plain",
        gmimc::instances::KoalaBearT24::from_params(&gmimc::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "gmimc", "gmimc-koalabear-t24", "split",
        gmimc::instances::KoalaBearT24Split::from_params(&gmimc::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "gmimc2", "gmimc2-koalabear-t24", "plain",
        gmimc2::instances::KoalaBearT24::from_params(&gmimc2::params::koalabear_t24::<KoalaBear>()));

    job!(jobs, "rescue-prime", "rescue-prime-koalabear-t16", "half-round",
        rp::KoalaBearT16::from_params(&rescue_prime::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-koalabear-t16", "full-round",
        rp_full::KoalaBearT16::from_params(&rescue_prime::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-koalabear-t24", "half-round",
        rp::KoalaBearT24::from_params(&rescue_prime::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "rescue-prime", "rescue-prime-koalabear-t24", "full-round",
        rp_full::KoalaBearT24::from_params(&rescue_prime::params::koalabear_t24::<KoalaBear>()));

    job!(jobs, "psquarehash", "psquarehash-koalabear-t16", "flattened",
        psq::Flattened16::from_params(&psquarehash::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t16", "state-only",
        psq::StateOnly16::from_params(&psquarehash::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t16", "one-register",
        psq::OneRegister16::from_params(&psquarehash::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t16", "spaced-split",
        psq::SpacedSplit16::from_params(&psquarehash::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t16", "spaced",
        psq::Spaced16::from_params(&psquarehash::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t16", "half-commit",
        psq::HalfCommit16::from_params(&psquarehash::params::koalabear_t16::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t24", "flattened",
        psq::Flattened24::from_params(&psquarehash::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t24", "state-only",
        psq::StateOnly24::from_params(&psquarehash::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t24", "one-register",
        psq::OneRegister24::from_params(&psquarehash::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t24", "spaced-split",
        psq::SpacedSplit24::from_params(&psquarehash::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t24", "spaced",
        psq::Spaced24::from_params(&psquarehash::params::koalabear_t24::<KoalaBear>()));
    job!(jobs, "psquarehash", "psquarehash-koalabear-t24", "half-commit",
        psq::HalfCommit24::from_params(&psquarehash::params::koalabear_t24::<KoalaBear>()));

    Plan {
        field: FieldId::KoalaBear,
        jobs,
    }
}

/// The introduction table's list: the traditional hashes Plonky3 arithmetizes,
/// plus one arithmetization-oriented reference row.
///
/// This is a **second, disjoint list**, not a filter over [`koalabear`]. The
/// reason is POLICY §3: BLAKE3, SHA-256 and Keccak-f are not among POLICY §1's
/// thirteen constructions and sit on no grid point, so they register no
/// [`harness::GridPoint`], appear in neither `bench::CONSTRUCTIONS` nor
/// `bench::REGISTERED`, and the coverage guard never sees them. Putting them in
/// the main list would mean either relaxing that guard or writing eight
/// "not on this grid" absences per hash into the main table.
///
/// What they *do* share with every other row is the measurement: the same
/// [`harness::measure`], the same configuration constructor, the same RNG seed,
/// the same 100-bit target and the same minimum-blowup reading. That is the
/// whole reason this is a plan here rather than a script of its own (POLICY
/// §11).
///
/// The Poseidon2 row is the paper's own headline instance — KoalaBear `t = 24`,
/// no register split — reached through the same type the main table measures,
/// so the two tables cannot disagree about it.
///
/// All four AIRs evaluate to maximum constraint degree 3, so the minimum-blowup
/// reading hands every row of this table the same code rate. The comparison is
/// therefore not confounded by the rate, which is unusual and worth stating.
#[rustfmt::skip]
#[must_use]
pub fn intro_koalabear<SC>() -> Plan<SC>
where
    SC: StarkGenericConfig + 'static,
    SC::Pcs: Pcs<SC::Challenge, SC::Challenger, Domain: PolynomialSpace<Val = KoalaBear>>,
{
    let mut jobs = Vec::new();

    // Traditional, Boolean-domain: one row is one compression (BLAKE3,
    // SHA-256); 24 rows are one permutation (Keccak-f).
    job!(jobs, "blake3", "blake3", "wrapped", traditional::instances::Blake3::new());
    job!(jobs, "sha256", "sha256", "wrapped", traditional::instances::Sha256::new());
    job!(jobs, "keccak-f", "keccak-f", "wrapped", traditional::instances::KeccakF::new());

    // Arithmetization-oriented, for contrast: the paper's headline instance.
    job!(jobs, "poseidon2", "poseidon2-koalabear-t24", "plain",
        poseidon2::instances::KoalaBearT24::<0>::new(poseidon2::params::koalabear_t24()));

    Plan {
        field: FieldId::KoalaBear,
        jobs,
    }
}
