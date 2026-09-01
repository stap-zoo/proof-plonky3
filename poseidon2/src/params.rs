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
//! # The structure agrees, at every point the reference pins
//!
//! That is the fact this module rests on, and `tests/structure.rs` is where it
//! is checked rather than asserted in prose:
//!
//! | `../ref` | t | α | `R_ext` | `R_int` | upstream | agrees |
//! |---|---|---|---|---|---|---|
//! | `POSEIDON2_GOLDILOCKS_T8`  | 8  | 7 | 8 | 22 | `GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8`  | yes |
//! | `POSEIDON2_GOLDILOCKS_T12` | 12 | 7 | 8 | 22 | `GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_12` | yes |
//! | `POSEIDON2_MERSENNE_T16`   | 16 | 5 | 8 | 14 | `MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_16` | yes |
//! | `POSEIDON2_MERSENNE_T24`   | 24 | 5 | 8 | 22 | `MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_24` | yes |
//!
//! So for four of the eight grid points there is nothing to reconcile: upstream's
//! own parameter set *is* the reference's structure, and this crate takes
//! upstream's constants with it.
//!
//! # BabyBear and KoalaBear: upstream's round counts, and why not `../ref`'s
//!
//! The reference pins no BabyBear or KoalaBear instance and **cannot derive
//! one**: `HadesParams._init_rounds` is one of POLICY §3's `NotImplementedError`
//! stubs ("implement the round-number derivation per subclass … until then R_ext
//! and R_int must be passed explicitly"), and it is shared by Poseidon, Poseidon2
//! and Neptune alike.
//!
//! POLICY §3's copy-across rule — Goldilocks t = 8 → t = 16, t = 12 → t = 24 —
//! would fill those four points with `R_int = 22`. This crate does **not** do
//! that, and the reason is POLICY §1: *if Plonky3 implements a design, that
//! implementation is the artifact.* Upstream pins real, published Poseidon2
//! parameter sets over both primes at both widths, each with a round count from
//! the actual Poseidon2 round-number analysis for that prime and that α:
//!
//! | instance | α | `R_int` upstream | `R_int` if copied from Goldilocks |
//! |---|---|---|---|
//! | BabyBear t = 16  | 7 | 13 | 22 |
//! | BabyBear t = 24  | 7 | 21 | 22 |
//! | KoalaBear t = 16 | 3 | 20 | 22 |
//! | KoalaBear t = 24 | 3 | 23 | 22 |
//!
//! Copying would move BabyBear t = 16 by 9 partial rounds — a ~25% swing in every
//! cost number for that row — and would attach Goldilocks' α = 7 analysis to
//! KoalaBear's α = 3. The copy-across rule exists for instances *we generate from
//! the reference's own derivations*; here nothing is generated, and the artifact
//! being measured has a round count of its own.
//!
//! **This is a decision, not a derivation, and it is the one worth revisiting.**
//! It lives in four constants below; changing them changes nothing else, and every
//! pinned number in `tests/numbers.rs` moves with them (POLICY §3, §11).

use p3_baby_bear::{
    BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_24,
    BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL, BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    BABYBEAR_POSEIDON2_RC_16_INTERNAL, BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
    BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL, BABYBEAR_POSEIDON2_RC_24_INTERNAL, BabyBear,
};
use p3_goldilocks::{
    GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8, GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_12,
    GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
    GOLDILOCKS_POSEIDON2_RC_8_INTERNAL, GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL,
    GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL, GOLDILOCKS_POSEIDON2_RC_12_INTERNAL, Goldilocks,
};
use p3_koala_bear::{
    KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16, KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_24,
    KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    KOALABEAR_POSEIDON2_RC_16_INTERNAL, KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
    KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_24_INTERNAL, KoalaBear,
};
use p3_mersenne_31::{
    MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_16, MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_24,
    MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL, MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    MERSENNE31_POSEIDON2_RC_16_INTERNAL, MERSENNE31_POSEIDON2_RC_24_EXTERNAL_FINAL,
    MERSENNE31_POSEIDON2_RC_24_EXTERNAL_INITIAL, MERSENNE31_POSEIDON2_RC_24_INTERNAL, Mersenne31,
};
use p3_poseidon2_air::RoundConstants;

// ---------------------------------------------------------------------------
// Round structure
// ---------------------------------------------------------------------------

/// `R_ext / 2`: full rounds before the partial ones, and again after.
///
/// The reference sets `R_ext = 8` for every Poseidon2 instance it pins, and
/// upstream sets `HALF_FULL_ROUNDS = 4` for every field. One constant, because
/// there is one value — and `tests/structure.rs` checks that against each field's
/// own upstream constant rather than trusting this line.
pub const HALF_FULL_ROUNDS: usize = 4;

/// `R_int` for Goldilocks t = 8. Reference-pinned (`POSEIDON2_GOLDILOCKS_T8`).
pub const GOLDILOCKS_T8_PARTIAL_ROUNDS: usize = GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8;
/// `R_int` for Goldilocks t = 12. Reference-pinned (`POSEIDON2_GOLDILOCKS_T12`).
pub const GOLDILOCKS_T12_PARTIAL_ROUNDS: usize = GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_12;
/// `R_int` for Mersenne-31 t = 16. Reference-pinned (`POSEIDON2_MERSENNE_T16`).
pub const MERSENNE_T16_PARTIAL_ROUNDS: usize = MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_16;
/// `R_int` for Mersenne-31 t = 24. Reference-pinned (`POSEIDON2_MERSENNE_T24`).
pub const MERSENNE_T24_PARTIAL_ROUNDS: usize = MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_24;

/// `R_int` for BabyBear t = 16 — **upstream's**, not the reference's. See the
/// module docs: the reference pins no instance and `_init_rounds` raises.
pub const BABYBEAR_T16_PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16;
/// `R_int` for BabyBear t = 24 — upstream's. See [`BABYBEAR_T16_PARTIAL_ROUNDS`].
pub const BABYBEAR_T24_PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_24;
/// `R_int` for KoalaBear t = 16 — upstream's. See [`BABYBEAR_T16_PARTIAL_ROUNDS`].
pub const KOALABEAR_T16_PARTIAL_ROUNDS: usize = KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16;
/// `R_int` for KoalaBear t = 24 — upstream's. See [`BABYBEAR_T16_PARTIAL_ROUNDS`].
pub const KOALABEAR_T24_PARTIAL_ROUNDS: usize = KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_24;

/// What `../ref` pins, as `(instance, t, α, R_ext, R_int)`.
///
/// Transcribed from `ref/hades/instances.py`, and consumed only by
/// `tests/structure.rs`, which checks each entry against the upstream constant
/// the instance is actually built from. A table nobody reads is a table that
/// drifts; this one fails the build when it does.
///
/// Four rows, not eight: BabyBear and KoalaBear have no reference instance at
/// all, which is the module docs' subject.
pub const REFERENCE_STRUCTURE: &[(&str, usize, u64, usize, usize)] = &[
    ("POSEIDON2_GOLDILOCKS_T8", 8, 7, 8, 22),
    ("POSEIDON2_GOLDILOCKS_T12", 12, 7, 8, 22),
    ("POSEIDON2_MERSENNE_T16", 16, 5, 8, 14),
    ("POSEIDON2_MERSENNE_T24", 24, 5, 8, 22),
];

// ---------------------------------------------------------------------------
// The degree table, duplicated on purpose
// ---------------------------------------------------------------------------

/// The AIR's maximum constraint degree for an S-box of degree `α` split over
/// `registers` committed cells.
///
/// This is upstream's `p3_poseidon2_air::sbox_constraint_degree` — which is
/// `pub(crate)` there — restated so that [`harness::permutation::Labels`] can carry it as
/// a `const`. `BaseAir::max_constraint_degree` is a method and the label has to
/// be a compile-time value, so there is no way to borrow upstream's copy.
///
/// **It is a duplicate and duplicates drift**, so it is pinned twice over:
/// `tests/numbers.rs` checks it against `BaseAir::max_constraint_degree` (what
/// upstream declares) *and* against `get_max_constraint_degree` (what the
/// symbolic walk over the real constraints produces). The label is what derives
/// the blowup (POLICY §7), so a value that is too low would buy a cheaper
/// configuration than the arithmetization earns.
///
/// The variant table of POLICY §11 is exactly this function's domain.
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
        _ => panic!("p3-poseidon2-air implements no S-box for this (degree, registers)"),
    }
}

// ---------------------------------------------------------------------------
// Constructors, one per grid point
//
// Every one of these is upstream's own published constant set, handed to
// upstream's AIR unchanged: `RoundConstants::new` only reshapes the three arrays
// into the layout the AIR reads them in. Nothing is derived here, nothing is
// injected, and `tests/air.rs` replays each against the matching
// `default_*_poseidon2_*()` permutation (POLICY §4).
// ---------------------------------------------------------------------------

/// Upstream's `default_goldilocks_poseidon2_8` constants, in the AIR's layout.
#[must_use]
pub const fn goldilocks_t8()
-> RoundConstants<Goldilocks, 8, HALF_FULL_ROUNDS, GOLDILOCKS_T8_PARTIAL_ROUNDS> {
    RoundConstants::new(
        GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
        GOLDILOCKS_POSEIDON2_RC_8_INTERNAL,
        GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_goldilocks_poseidon2_12` constants, in the AIR's layout.
#[must_use]
pub const fn goldilocks_t12()
-> RoundConstants<Goldilocks, 12, HALF_FULL_ROUNDS, GOLDILOCKS_T12_PARTIAL_ROUNDS> {
    RoundConstants::new(
        GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL,
        GOLDILOCKS_POSEIDON2_RC_12_INTERNAL,
        GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_mersenne31_poseidon2_16` constants, in the AIR's layout.
#[must_use]
pub const fn mersenne_t16()
-> RoundConstants<Mersenne31, 16, HALF_FULL_ROUNDS, MERSENNE_T16_PARTIAL_ROUNDS> {
    RoundConstants::new(
        MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL,
        MERSENNE31_POSEIDON2_RC_16_INTERNAL,
        MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_mersenne31_poseidon2_24` constants, in the AIR's layout.
#[must_use]
pub const fn mersenne_t24()
-> RoundConstants<Mersenne31, 24, HALF_FULL_ROUNDS, MERSENNE_T24_PARTIAL_ROUNDS> {
    RoundConstants::new(
        MERSENNE31_POSEIDON2_RC_24_EXTERNAL_INITIAL,
        MERSENNE31_POSEIDON2_RC_24_INTERNAL,
        MERSENNE31_POSEIDON2_RC_24_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_babybear_poseidon2_16` constants, in the AIR's layout.
/// Round count upstream's, not the reference's — see the module docs.
#[must_use]
pub const fn babybear_t16()
-> RoundConstants<BabyBear, 16, HALF_FULL_ROUNDS, BABYBEAR_T16_PARTIAL_ROUNDS> {
    RoundConstants::new(
        BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
        BABYBEAR_POSEIDON2_RC_16_INTERNAL,
        BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_babybear_poseidon2_24` constants. See [`babybear_t16`].
#[must_use]
pub const fn babybear_t24()
-> RoundConstants<BabyBear, 24, HALF_FULL_ROUNDS, BABYBEAR_T24_PARTIAL_ROUNDS> {
    RoundConstants::new(
        BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL,
        BABYBEAR_POSEIDON2_RC_24_INTERNAL,
        BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_koalabear_poseidon2_16` constants. See [`babybear_t16`].
#[must_use]
pub const fn koalabear_t16()
-> RoundConstants<KoalaBear, 16, HALF_FULL_ROUNDS, KOALABEAR_T16_PARTIAL_ROUNDS> {
    RoundConstants::new(
        KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
        KOALABEAR_POSEIDON2_RC_16_INTERNAL,
        KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
    )
}

/// Upstream's `default_koalabear_poseidon2_24` constants. See [`babybear_t16`].
#[must_use]
pub const fn koalabear_t24()
-> RoundConstants<KoalaBear, 24, HALF_FULL_ROUNDS, KOALABEAR_T24_PARTIAL_ROUNDS> {
    RoundConstants::new(
        KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL,
        KOALABEAR_POSEIDON2_RC_24_INTERNAL,
        KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
    )
}
