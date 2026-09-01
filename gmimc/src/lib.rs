//! GMiMC-erf — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it (POLICY §1). The design is
//! [ESORICS 2019/397](https://eprint.iacr.org/2019/397); the native reference is
//! `../ref`'s `gmimc/`, and gnark-side prior art is `../gnark-hashes/gmimc`.
//!
//! # The design, in one paragraph
//!
//! An unbalanced Feistel over `F^t` with an **expanding round function**: each
//! round powers branch 0 — `y = (x_0 + rc_r)^alpha`, one S-box for the whole
//! round, whatever `t` is — adds `y` into every other branch, and rotates the
//! state left by one. Branch 0 is the only branch ever read nonlinearly, and
//! every other branch merely accumulates. That asymmetry is the whole design and
//! it is what makes the arithmetization a question about a **round count**
//! rather than about a width: `R` is 335 where a Poseidon-shaped design runs
//! tens of rounds, and a round costs one committed cell.
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref`, through `export_small_prime_kat.py` —
//! parameters as well as vectors, so that "these constants came from the
//! reference" is a checked claim and not an asserted one. See
//! [`vectors/README.md`](../../../gmimc/vectors/README.md) for the export's
//! provenance, including the one mode the exporter skips and why that costs no
//! vector.
//!
//! # The grid: four points, four absences
//!
//! One width per field size — Goldilocks `t=12`, every 31-bit prime `t=24` —
//! and all four are generated at the exporter call site, because
//! `gmimc/instances.py` pins the BN254 and BLS12-381 `t=3` pair and nothing
//! else.
//!
//! **Goldilocks `t=8` and the three 31-bit `t=16` points do not exist.**
//! `GMiMCParams._init_rounds` raises `NotImplementedError` and POLICY §3's
//! copy-across rule cannot supply a round count here — there is no Goldilocks
//! `t=8` value to copy from, and copying across widths is unsound for an
//! expanding round function, whose criterion grows steeply with the branch
//! count. All four are `Absence::StubbedDerivation`.
//!
//! The round counts that *do* exist are a **third provenance** beside POLICY
//! §3's two: they are the author's cryptanalysis, and no derivation in `../ref`
//! or here checks them ([`params`]).
//!
//! # Its sibling
//!
//! `gmimc2` is the same family and a distinct design — a different exponent, a
//! round constant that stays in the state, and an input/output matrix — with a
//! distinct oracle. Two crates, not one with a variant axis.
//!
//! # The arithmetization, in one paragraph
//!
//! One committed cell per round and no state at all. Only branch 0 is read
//! nonlinearly, so a branch is a running sum of S-box outputs and the only value
//! worth committing is the one entering each S-box; committing it makes the
//! permutation a single recurrence reaching back exactly `t` rounds, which
//! telescopes into a constraint of four cells and two S-box terms whatever `t`
//! is. Width is `R + 2t` and constraints `R + t`, at degree `alpha` — POLICY
//! §11's floor, attained rather than counted. [`air`] carries the derivation
//! beside the constraints it justifies.
//!
//! # Current state
//!
//! POLICY §2 steps 1–6: the oracle is green against the reference's vectors, the
//! four AIR files are written, and the §10 ladder runs — including the two layers
//! this layout needs on its own, because a trace of no state cannot say what its
//! cells are by inspection: an independent replay of the round loop, and the
//! window algebra the telescoped constraint stands on (`tests/air.rs`).
//!
//! The construction is registered in `bench` at all eight grid points, including
//! the four explicit absences, and all eight admissible AIR variants are in the
//! typed plan. Layer 5's real `prove` + `verify` runs in
//! `bench/tests/prove_verify.rs`. Measurement remains POLICY §2 step 7.

pub mod air;
pub mod columns;
pub mod generation;
pub mod instances;
pub mod native;
pub mod params;
pub mod vectorized;
