//! XHash — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it, and **no prior art was found** (POLICY §1): no
//! existing AIR, no third-party native implementation to check against beyond
//! `../ref` itself. That makes the reference the sole oracle here and makes
//! POLICY §2's order load-bearing rather than procedural — a native permutation
//! that is not KAT-green is not a foundation for an AIR.
//!
//! # One crate, four instances, two grid rows
//!
//! `XHash8`, `XHash12`, `XHash16` and `XHash24` are **one design**. `../ref`'s
//! `export_small_prime_kat.py` maps both its `xhash8` and `xhash16` keys onto
//! the same class, `marvellous.hash.XHash`, with the same `XHashParams`; they
//! differ in field, width and `alpha`, all of which were already parameters
//! here. They were two crates and are now one.
//!
//! What did not merge is the comparison grid: `xhash8` and `xhash16` remain two
//! rows, because the grid compares `(construction, field, width)` points and
//! renaming them would rewrite every measured row (see [`instances`]).
//!
//! Both families now use the same structured P3 register basis. `ERROR.md` at
//! the repository root records the defective Mersenne-31 table, the corrected
//! irreducible modulus and the resulting AIR-width reduction;
//! [`air::XHashAir::from_params`] keeps the coordinate-table claim checked.
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref` or it is not validated. The vectors come from
//! `../ref/export_small_prime_kat.py`; giving this construction an entry in that
//! script's construction table is POLICY §2's step 2, which comes before the
//! native permutation and long before any AIR.
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
