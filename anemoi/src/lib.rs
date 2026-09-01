//! Anemoi — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it (POLICY §1). Native reference:
//! [anemoi-rust](https://github.com/anemoi-hash/anemoi-rust).
//!
//! Anemoi's closed Flystel stays construction-specific; its forward power and
//! optional register split use the shared `harness::gadgets::power_map` pair now that
//! Neptune and pSquareHash provide the other callers required by POLICY §6.
//!
//! Native evaluation uses the inverse power, while the closed AIR form uses the
//! small forward power and optional register splitting. Each register is pinned
//! where it is used, with `0`, `1`, and field-edge tests (POLICY §9).
//!
//! Where the reference defines an extra width (t = 10), implement it if free —
//! but the grid is what the tables compare.
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref` or it is not validated. The vectors come from a
//! new small-prime-only export script in `../ref` — `export_kat.py` is
//! hardcoded to BN254 and BLS12-381 and has no entry for this construction over
//! small primes. They are checked into `vectors/` and consumed by the tests.
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
