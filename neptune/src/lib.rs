//! Neptune — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has no Neptune implementation. Parameters and vectors are derived
//! from `../ref`'s `hades` module; the native permutation and AIR here reuse
//! Plonky3's dense external MDS and Poseidon2 internal-layer routines.
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref` or it is not validated. The vectors come from a
//! small-prime export script in `../ref`: `export_small_prime_kat.py neptune`.
//! Its output is checked before the AIR tests.
//!
//! The construction-specific validation lives in `tests/{reference,air,numbers}.rs`;
//! POLICY §10 defines the shared layers. The release proof round trip is
//! centralized in `bench/tests/prove_verify.rs` so it uses the same 100-bit
//! configuration as every other construction.

pub mod air;
pub mod columns;
pub mod generation;
pub mod instances;
pub mod native;
pub mod params;
pub mod vectorized;
