//! How a construction describes its own coverage of POLICY §3's grid.
//!
//! # Why this is in `harness` and not in `bench`
//!
//! `bench::REGISTERED` grows by concatenating each construction's own
//! `INSTANCES`, so a construction has to be able to *write down* a grid point —
//! and a construction cannot depend on `bench`, which depends on it. The types
//! therefore live below both, here, beside the [`FieldId`] and [`GRID`](crate::GRID)
//! they are about, and `bench` re-exports them.
//!
//! This does **not** teach `harness` which hashes exist (POLICY §5).
//! [`GridPoint::construction`] is a `&'static str` the harness prints and never
//! branches on — the same rule [`Labels`](crate::Labels) already lives under —
//! and nothing here is a dependency on a construction crate. The list of the
//! eleven stays in `bench`, which is where knowing them is allowed, along with
//! the guard that no construction ever appears in this crate's manifest.

use crate::FieldId;

/// One instance: a construction at one field and one width.
///
/// The `name` is the reference variable lowercased with `_` replaced by `-`,
/// exactly what the export script emits, and it is the only thing tying this
/// entry to a vector file.
#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    /// Construction name, matching a `bench::Construction::name`.
    pub construction: &'static str,
    /// Instance name (POLICY §3), or `None` when the point is absent.
    pub instance: Option<&'static str>,
    /// Which field.
    pub field: FieldId,
    /// State width `t`.
    pub state_width: usize,
    /// Why this point is absent, when it is.
    pub absence: Option<Absence>,
}

/// Why a grid point does not exist. Absent points are **reported, never filled
/// by hand** (POLICY §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Absence {
    /// The reference deliberately defines no permutation instance at this
    /// width. Tip5's Goldilocks `t=8` derivation is the motivating case: its
    /// permutation helpers can be called there, but constructing the named
    /// instance requires choosing different sponge security parameters, and
    /// POLICY §3 does not turn that choice into a derivation.
    UndefinedByReference,
    /// The reference's derivation for this parameter is a
    /// `NotImplementedError` stub — thirteen of its `params.py` files have one,
    /// Monolith's `_init_rounds` among them, and pSquareHash's too.
    StubbedDerivation,
    /// The design is not defined over this prime upstream. Monolith's Bars
    /// exist for Goldilocks and Mersenne-31 only, so Monolith covers those two
    /// fields.
    UndefinedForField,
    /// A **wrapped** construction whose upstream pins no parameters at this
    /// point, though it does at neighbouring ones. Poseidon1 over Mersenne-31 is
    /// defined at t = 16 and t = 32 upstream and not at t = 24: neither round
    /// constants nor a circulant MDS column exist there. We wrap, we do not
    /// re-port (POLICY §1), and a parameter with no source is absent rather than
    /// invented (POLICY §3) — so the point is reported, not filled by hand.
    ///
    /// Distinct from [`Self::UndefinedForField`], which says the design does not
    /// reach the *prime* at all, and from [`Self::StubbedDerivation`], which is
    /// about `../ref`: here the reference does pin the instance, and it is
    /// upstream that has none.
    UndefinedUpstream,
    /// A construction's ZK Mersenne-31 row; the whole configuration is
    /// unsupported (POLICY §7).
    NoZkMersenne31,
}
