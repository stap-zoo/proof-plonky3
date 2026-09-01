//! Parameters: structure from `../ref`, values from upstream.
//!
//! POLICY §4. What we wrap is validated against upstream's own implementation,
//! not against a reference KAT, and no constants are injected — a one-to-one
//! match with `../ref` is not required and is not attempted.
//!
//! But **structural parameters come from the reference in both cases**: width,
//! round counts, S-box degree. Cost depends on those and not on the values of
//! the round constants, so taking them from `../ref` is exactly what keeps a
//! wrapped row comparable to a written one.
//!
//! Where upstream's structure cannot be set to the reference's, this row
//! reports its own round counts rather than quietly reporting the reference's.
//!
//! # The structure agrees, at the three points upstream reaches
//!
//! | `../ref` | t | α | `R_ext` | `R_int` | upstream | agrees |
//! |---|---|---|---|---|---|---|
//! | `POSEIDON_GOLDILOCKS_T8`  | 8  | 7 | 8 | 22 | `GOLDILOCKS_POSEIDON_PARTIAL_ROUNDS_8`   | yes |
//! | `POSEIDON_GOLDILOCKS_T12` | 12 | 7 | 8 | 22 | `GOLDILOCKS_POSEIDON_PARTIAL_ROUNDS_12`  | yes |
//! | `POSEIDON_MERSENNE_T16`   | 16 | 5 | 8 | 14 | `MERSENNE31_POSEIDON1_PARTIAL_ROUNDS_16` | yes |
//! | `POSEIDON_MERSENNE_T24`   | 24 | 5 | 8 | 22 | — | **upstream has none** |
//!
//! `tests/structure.rs` checks the first three rather than trusting this table.
//!
//! # Mersenne-31 t = 24 does not exist upstream, and is not invented here
//!
//! `p3-mersenne-31` pins Poseidon1 at t = 16 and t = 32 — `MERSENNE31_POSEIDON1_RC_16`,
//! `MERSENNE31_POSEIDON1_RC_32` — and nothing at t = 24. Neither half of what an
//! instance needs is available there:
//!
//! * no round constants, and
//! * no circulant MDS column. `Poseidon1Constants::mds_circ_col` is a *circulant*
//!   first column, because upstream's AIR dispatches width 24 to
//!   `mds_circulant_karatsuba_24` on the first column of the dense matrix; the
//!   reference's own `POSEIDON_MERSENNE_T24` matrix is Cauchy, not circulant, so
//!   even injecting it — which POLICY §4 rules out for a wrap — would not fit the
//!   shape upstream reads.
//!
//! Borrowing BabyBear's width-24 circulant would be asserting an MDS property
//! nobody has checked over `p = 2^31 − 1`. So the point is
//! [`Absence::UndefinedUpstream`](harness::Absence): reported, never filled by
//! hand (POLICY §3). It is the one gap in Poseidon1's grid, and it is upstream's
//! to close.
//!
//! # BabyBear and KoalaBear: upstream's round counts, and why not `../ref`'s
//!
//! The reference pins no BabyBear or KoalaBear instance and **cannot derive
//! one**: `HadesParams._init_rounds` is one of POLICY §3's `NotImplementedError`
//! stubs, shared by Poseidon, Poseidon2 and Neptune alike.
//!
//! POLICY §3's copy-across rule — Goldilocks t = 8 → t = 16, t = 12 → t = 24 —
//! would give all four `R_int = 22`. This crate takes upstream's instead, for the
//! reason POLICY §1 gives: *if Plonky3 implements a design, that implementation is
//! the artifact.* Upstream's counts are 13 and 21 for BabyBear, 20 and 23 for
//! KoalaBear, each from the round-number analysis for that prime and that α;
//! copying would move BabyBear t = 16 by 9 partial rounds and attach Goldilocks'
//! α = 7 analysis to KoalaBear's α = 3.
//!
//! **This is a decision, not a derivation.** It lives in the four constants below,
//! and every pinned number in `tests/numbers.rs` moves with them (POLICY §3, §11).
//! The same decision is taken, for the same reason, in `poseidon2::params`.

use p3_baby_bear::{
    BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16, BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24,
    BABYBEAR_POSEIDON1_RC_16, BABYBEAR_POSEIDON1_RC_24, BabyBear, MDSBabyBearData,
};
use p3_goldilocks::poseidon1::{
    GOLDILOCKS_POSEIDON_HALF_FULL_ROUNDS, GOLDILOCKS_POSEIDON_PARTIAL_ROUNDS_8,
    GOLDILOCKS_POSEIDON_PARTIAL_ROUNDS_12, GOLDILOCKS_POSEIDON1_RC_8, GOLDILOCKS_POSEIDON1_RC_12,
};
use p3_goldilocks::{Goldilocks, MATRIX_CIRC_MDS_8_COL, MATRIX_CIRC_MDS_12_COL};
use p3_koala_bear::{
    KOALABEAR_POSEIDON_PARTIAL_ROUNDS_16, KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24,
    KOALABEAR_POSEIDON1_RC_16, KOALABEAR_POSEIDON1_RC_24, KoalaBear, MDSKoalaBearData,
};
use p3_mersenne_31::{
    MERSENNE31_POSEIDON1_MDS_CIRC_COL_16, MERSENNE31_POSEIDON1_PARTIAL_ROUNDS_16,
    MERSENNE31_POSEIDON1_RC_16, Mersenne31,
};
use p3_monty_31::MDSUtils;
use p3_poseidon1::Poseidon1Constants;

// ---------------------------------------------------------------------------
// Round structure
// ---------------------------------------------------------------------------

/// `R_ext / 2`: full rounds before the partial ones, and again after.
///
/// The reference sets `R_ext = 8` for every Poseidon1 instance it pins, and
/// upstream sets `HALF_FULL_ROUNDS = 4` for every field. `tests/structure.rs`
/// checks that against each field's own constant rather than trusting this line.
pub const HALF_FULL_ROUNDS: usize = GOLDILOCKS_POSEIDON_HALF_FULL_ROUNDS;

/// `R_int` for Goldilocks t = 8. Reference-pinned (`POSEIDON_GOLDILOCKS_T8`).
pub const GOLDILOCKS_T8_PARTIAL_ROUNDS: usize = GOLDILOCKS_POSEIDON_PARTIAL_ROUNDS_8;
/// `R_int` for Goldilocks t = 12. Reference-pinned (`POSEIDON_GOLDILOCKS_T12`).
pub const GOLDILOCKS_T12_PARTIAL_ROUNDS: usize = GOLDILOCKS_POSEIDON_PARTIAL_ROUNDS_12;
/// `R_int` for Mersenne-31 t = 16. Reference-pinned (`POSEIDON_MERSENNE_T16`).
pub const MERSENNE_T16_PARTIAL_ROUNDS: usize = MERSENNE31_POSEIDON1_PARTIAL_ROUNDS_16;

/// `R_int` for BabyBear t = 16 — **upstream's**, not the reference's. See the
/// module docs: the reference pins no instance and `_init_rounds` raises.
pub const BABYBEAR_T16_PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16;
/// `R_int` for BabyBear t = 24 — upstream's. See [`BABYBEAR_T16_PARTIAL_ROUNDS`].
pub const BABYBEAR_T24_PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24;
/// `R_int` for KoalaBear t = 16 — upstream's. See [`BABYBEAR_T16_PARTIAL_ROUNDS`].
pub const KOALABEAR_T16_PARTIAL_ROUNDS: usize = KOALABEAR_POSEIDON_PARTIAL_ROUNDS_16;
/// `R_int` for KoalaBear t = 24 — upstream's. See [`BABYBEAR_T16_PARTIAL_ROUNDS`].
pub const KOALABEAR_T24_PARTIAL_ROUNDS: usize = KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24;

/// What `../ref` pins, as `(instance, t, α, R_ext, R_int)`.
///
/// Transcribed from `ref/hades/instances.py`, and consumed only by
/// `tests/structure.rs`, which checks each entry against the upstream constant
/// the instance is actually built from. The fourth row is the one upstream cannot
/// match at all; that test records it as such.
pub const REFERENCE_STRUCTURE: &[(&str, usize, u64, usize, usize)] = &[
    ("POSEIDON_GOLDILOCKS_T8", 8, 7, 8, 22),
    ("POSEIDON_GOLDILOCKS_T12", 12, 7, 8, 22),
    ("POSEIDON_MERSENNE_T16", 16, 5, 8, 14),
    ("POSEIDON_MERSENNE_T24", 24, 5, 8, 22),
];

// ---------------------------------------------------------------------------
// The degree table, duplicated on purpose
// ---------------------------------------------------------------------------

/// The AIR's maximum constraint degree for an S-box of degree `α` split over
/// `registers` committed cells.
///
/// This is upstream's `p3_poseidon1_air::sbox_constraint_degree` — `pub(crate)`
/// there — restated so [`harness::permutation::Labels`] can carry it as a `const`.
/// `BaseAir::max_constraint_degree` is a method and the label has to be a
/// compile-time value, so there is no way to borrow upstream's copy.
///
/// **It is a duplicate and duplicates drift**, so `tests/numbers.rs` pins it
/// against both what upstream declares and what the symbolic walk produces.
///
/// One row differs from Poseidon2's otherwise identical table: `(11, 1) → 5`.
/// Poseidon1's S-box implements a one-register `x^11` whose output `(x³)³ · x²`
/// reaches degree 5, where Poseidon2 has only the two-register form. No instance
/// on this grid uses α = 11 — the four fields give α ∈ {3, 5, 7} — so the row is
/// unreachable here and is kept only because leaving it out would make this table
/// silently *disagree* with upstream's rather than merely restate it.
///
/// # Panics
///
/// On a `(degree, registers)` pair upstream's S-box does not implement — which is
/// the point: an instance can only declare a variant that exists.
#[must_use]
pub const fn max_constraint_degree(sbox_degree: u64, sbox_registers: usize) -> usize {
    match (sbox_degree, sbox_registers) {
        (3, 0) => 3,
        (5, 0) => 5,
        (7, 0) => 7,
        (5, 1) | (7, 1) | (11, 2) => 3,
        (11, 1) => 5,
        _ => panic!("p3-poseidon1-air implements no S-box for this (degree, registers)"),
    }
}

// ---------------------------------------------------------------------------
// Constructors, one per grid point upstream reaches
//
// Each returns the *raw* `Poseidon1Constants` — the form before the sparse
// matrix decomposition — because that is the one form both consumers take: the
// AIR through `to_optimized`, and `p3-poseidon1`'s own permutation through
// `Poseidon1::new`. One value, two consumers, so the oracle in `tests/air.rs`
// cannot drift from the AIR under test.
//
// Every field here is upstream's own published constant: the round-constant
// grids and the circulant MDS columns are exactly what `default_*_poseidon1_*()`
// passes.
// ---------------------------------------------------------------------------

/// Upstream's `default_goldilocks_poseidon1_8` parameters.
#[must_use]
pub fn goldilocks_t8() -> Poseidon1Constants<Goldilocks, 8> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: GOLDILOCKS_T8_PARTIAL_ROUNDS,
        mds_circ_col: MATRIX_CIRC_MDS_8_COL,
        round_constants: GOLDILOCKS_POSEIDON1_RC_8.to_vec(),
    }
}

/// Upstream's `default_goldilocks_poseidon1_12` parameters.
#[must_use]
pub fn goldilocks_t12() -> Poseidon1Constants<Goldilocks, 12> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: GOLDILOCKS_T12_PARTIAL_ROUNDS,
        mds_circ_col: MATRIX_CIRC_MDS_12_COL,
        round_constants: GOLDILOCKS_POSEIDON1_RC_12.to_vec(),
    }
}

/// Upstream's `default_mersenne31_poseidon1_16` parameters.
///
/// There is no `mersenne_t24` beside this one, and there cannot be: see the
/// module docs.
#[must_use]
pub fn mersenne_t16() -> Poseidon1Constants<Mersenne31, 16> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: MERSENNE_T16_PARTIAL_ROUNDS,
        mds_circ_col: MERSENNE31_POSEIDON1_MDS_CIRC_COL_16,
        round_constants: MERSENNE31_POSEIDON1_RC_16.to_vec(),
    }
}

/// Upstream's `default_babybear_poseidon1_16` parameters. Round count upstream's,
/// not the reference's — see the module docs.
#[must_use]
pub fn babybear_t16() -> Poseidon1Constants<BabyBear, 16> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: BABYBEAR_T16_PARTIAL_ROUNDS,
        mds_circ_col: MDSBabyBearData::MATRIX_CIRC_MDS_16_COL,
        round_constants: BABYBEAR_POSEIDON1_RC_16.to_vec(),
    }
}

/// Upstream's `default_babybear_poseidon1_24` parameters. See [`babybear_t16`].
#[must_use]
pub fn babybear_t24() -> Poseidon1Constants<BabyBear, 24> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: BABYBEAR_T24_PARTIAL_ROUNDS,
        mds_circ_col: MDSBabyBearData::MATRIX_CIRC_MDS_24_COL,
        round_constants: BABYBEAR_POSEIDON1_RC_24.to_vec(),
    }
}

/// Upstream's `default_koalabear_poseidon1_16` parameters. See [`babybear_t16`].
#[must_use]
pub fn koalabear_t16() -> Poseidon1Constants<KoalaBear, 16> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: KOALABEAR_T16_PARTIAL_ROUNDS,
        mds_circ_col: MDSKoalaBearData::MATRIX_CIRC_MDS_16_COL,
        round_constants: KOALABEAR_POSEIDON1_RC_16.to_vec(),
    }
}

/// Upstream's `default_koalabear_poseidon1_24` parameters. See [`babybear_t16`].
#[must_use]
pub fn koalabear_t24() -> Poseidon1Constants<KoalaBear, 24> {
    Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: KOALABEAR_T24_PARTIAL_ROUNDS,
        mds_circ_col: MDSKoalaBearData::MATRIX_CIRC_MDS_24_COL,
        round_constants: KOALABEAR_POSEIDON1_RC_24.to_vec(),
    }
}
