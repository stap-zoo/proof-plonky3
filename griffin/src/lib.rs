//! Griffin — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it (POLICY §1). Native references: winterfell
//! `hash/griffin` and [`zkhash`](https://docs.rs/zkhash).
//!
//! Griffin's non-linear layer combines a power map, an inverse power and a
//! quadratic term whose inputs come from earlier state words. The inverse power
//! is witnessed and pinned by `harness::gadgets::inverse_power_map`.
//!
//! The structure does **not** chain: `L_i`'s feedback word is the round's input
//! `x_{i-1}`, not the output `y_{i-1}`, so every word of a round is computable
//! from the round input plus the two S-box outputs. One call fits one row, and
//! the Horst product `x_i * G_i(L_i)` sets a degree floor of three on the whole
//! AIR — which is why alpha 3 is offered no register variant (`columns.rs`,
//! `air.rs`).
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref`, through `export_small_prime_kat.py` — parameters
//! as well as vectors, so that "these constants came from the reference" is a
//! checked claim and not an asserted one.
//!
//! Griffin covers the whole POLICY §3 grid with **no absence**. The reference
//! pins the two Goldilocks instances; the six 31-bit points are constructed at
//! the exporter's call site and derive everything — constants, matrix, *and*
//! round count — from Griffin's own `params.py`. It is the only construction so
//! far whose `_init_rounds` is implemented rather than a `NotImplementedError`
//! stub, which is why its generated instances carry real round counts instead
//! of POLICY §3's provisional copies (see `params.rs`).
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
