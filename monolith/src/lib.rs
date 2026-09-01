//! Monolith — **wrapped**, and the Type-3 reference point.
//!
//! `p3-monolith` + `p3-monolith-air` (POLICY §1). Upstream decomposes the Bars
//! **in-AIR**: committed bits, chi cells, canonicity flags, no lookup. That is
//! not a limitation of the wrapper but the default this repository measures
//! against (POLICY §12) — `uni-stark` has a single commit round and leaves
//! `AirLayout::permutation_width` at 0, so there is nowhere to put LogUp's
//! auxiliary columns.
//!
//! Three things about this row are unlike the others, and each is reported
//! rather than smoothed over:
//!
//! * **Two fields, not four.** Bars exist for Goldilocks and Mersenne-31 only,
//!   so Monolith covers those two grid columns. BabyBear and KoalaBear are
//!   `Absence::UndefinedForField`.
//! * **No vectorized layout.** `p3-monolith-air` computes one full permutation
//!   per row; a Monolith state is wide enough that packing several calls into a
//!   row is not upstream's design. `Labels::calls_per_row` says so, and the
//!   per-call cost is read accordingly.
//! * **No ZK Mersenne-31 row.** That configuration is unsupported (POLICY §7;
//!   `Absence::NoZkMersenne31`).
//!
//! The reference's Monolith `_init_rounds` is one of the thirteen
//! `NotImplementedError` stubs (POLICY §3), so any grid point that would need
//! it is absent and reported, never filled by hand.
//!
//! # The same hash, twice (POLICY §12)
//!
//! [`logup`] contains the fixed Bars tables and a separate call AIR/generator;
//! the harness has the `prove_batch` measurement path, exact multiplicity
//! bound, and lookup-aware symbolic/security shape that `AirLayout::from_air`
//! alone cannot supply. The wrapped baseline above is unchanged, so the two
//! variants really are arithmetizations of the same hash.
//!
//! `tests/{air,numbers}.rs` contain the wrapper-specific checks; POLICY §10
//! defines the shared validation layers. The release proof round trip is
//! centralized in `bench/tests/prove_verify.rs` under the shared 100-bit
//! configuration.

pub mod instances;
pub mod logup;
pub mod params;
