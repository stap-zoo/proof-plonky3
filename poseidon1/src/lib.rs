//! Poseidon1 — **wrapped**, and the baseline.
//!
//! `p3-poseidon1` + `p3-poseidon1-air` (POLICY §1). It is here for one reason:
//! it makes the Poseidon1 → Poseidon2 delta directly comparable rather than
//! asserted. The two are wrapped identically and assigned the same configuration
//! cell, so their eventual measurement differs only by design and parameters.
//!
//! No `columns.rs` / `air.rs` / `generation.rs`: wrapped constructions have
//! nothing of ours to drift.
//!
//! # Validation (POLICY §4)
//!
//! Against `p3-poseidon1`'s own permutation. Structural parameters — width,
//! round counts, S-box degree — from `../ref`, since cost depends on those and
//! not on the constants' values.
//!
//! `tests/{air,numbers,structure}.rs` contain the wrapper-specific checks;
//! POLICY §10 defines the shared validation layers. The release proof round trip
//! is centralized in `bench/tests/prove_verify.rs` under the shared 100-bit
//! configuration.

pub mod instances;
pub mod params;
