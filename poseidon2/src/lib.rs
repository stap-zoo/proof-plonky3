//! Poseidon2 — **wrapped**.
//!
//! Plonky3 implements it, so that implementation is the artifact (POLICY §1):
//! `p3-poseidon2` for the permutation, `p3-poseidon2-air` for the vectorized
//! AIR. We wrap it, we do not re-port it, and we make no upstream changes.
//! There is deliberately no `columns.rs` / `air.rs` / `generation.rs` here —
//! nothing of ours to drift from upstream's. The whole of this crate is
//! [`params`] (which instance) and [`instances`] (which variant, and the newtype
//! that makes upstream's AIR a [`harness::permutation::PermutationAir`]).
//!
//! Upstream ships generic linear layers for all four fields of the grid
//! (`GenericPoseidon2LinearLayers{Goldilocks,Mersenne31,BabyBear,KoalaBear}`),
//! so POLICY §3's grid is reachable without extending anything. **Poseidon2 is
//! the one construction here with no absent grid point.**
//!
//! `p3-poseidon2-air/src/` is also the model POLICY §6 points every written
//! construction at: four files, a free `eval` shared between the plain and
//! vectorized layouts, `main_next_row_columns()` overridden to `vec![]`.
//!
//! Where the reference defines extra widths (t = 16, 20), implement them if
//! free — but the grid is what the tables compare.
//!
//! # Validation (POLICY §4)
//!
//! Against `p3-poseidon2`'s own permutation, not against a reference KAT: no
//! constant injection, so a one-to-one match with `../ref` is not required.
//! Structural parameters — width, round counts, S-box degree — still come from
//! `../ref`, because cost depends on those and not on the constants' values,
//! and that is what keeps this row comparable to a written one.
//!
//! **At every point `../ref` pins an instance, upstream's structure already is
//! the reference's** — `(t, α, R_ext, R_int)` agree at Goldilocks t = 8 and 12
//! and Mersenne-31 t = 16 and 24 — so those four rows are reference-structured
//! and upstream-valued at once, with nothing to reconcile. `tests/structure.rs`
//! is where that is checked rather than asserted. The other four points have no
//! reference instance and none can be derived (`HadesParams._init_rounds`
//! raises), so they carry **upstream's** round counts; [`params`] argues that
//! choice and says what POLICY §3's copy-across rule would have produced
//! instead.
//!
//! # Two consequences of wrapping, reported rather than smoothed over
//!
//! * **No ZK Mersenne-31 row.** That configuration is unsupported (POLICY §7).
//! * **The round constants are stored twice.** `VectorizedPoseidon2Air` keeps its
//!   `RoundConstants` private and the only generator that accepts caller-supplied
//!   inputs is the free function, which wants them by reference. The wrapper
//!   holds a clone; it is data, not logic, and `tests/air.rs` would catch the two
//!   copies parting company.
//!
//! # Validation coverage
//!
//! POLICY §10 defines the common layers. `tests/air.rs` checks all eight
//! instances and fourteen variants, including generic-AIR versus specialized
//! upstream linear layers, every vector lane, and an every-cell corruption
//! sweep. `tests/numbers.rs` pins the layout and upstream degree table;
//! `tests/structure.rs` pins reference structure. The release proof round trip
//! is centralized in `bench/tests/prove_verify.rs` under the shared 100-bit
//! configuration.

pub mod instances;
pub mod params;
