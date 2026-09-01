//! GMiMC2 (GMiMC-erf2) — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it and neither does `../ref` (POLICY §1, §4). Its
//! oracle is [`../gnark-hashes/gmimc2_ref.py`](../../../../gnark-hashes/gmimc2_ref.py),
//! and the gnark-side port of the same design is its neighbour there.
//!
//! # The design, in one paragraph
//!
//! GMiMC-erf with three changes, and only three. The S-box exponent is
//! `alpha = 2^k` — optimal degree growth for `k` multiplications, and **not** a
//! permutation of `F_p`, which an expanding-round-function Feistel does not need
//! it to be. The round constant is added **into the state** before the power
//! map, so it travels with the branch instead of being an S-box input; that is
//! what "permits slightly more efficient circuits" in the specification, because
//! the constant merges into the branch addition the round already performs. And
//! an input/output matrix `M_IO` brackets the round loop at both ends. Otherwise
//! it is erf unchanged: one S-box a round on branch 0, added into every other
//! branch, then a cyclic shift.
//!
//! # Validation (POLICY §4), and the second trust boundary
//!
//! Every other export in this repository comes from `../ref`. **This one does
//! not, and it is the only one that does not.** GMiMC2 is not implemented there
//! and is deliberately not to be added, so the author has settled that
//! `gmimc2_ref.py` is the oracle. The exception is written as narrowly as the
//! XHash coordinate one: GMiMC2 alone, because `../ref` does not define it.
//!
//! The driver that produces the vectors — [`tools/export_gmimc2_kat.py`](../../../tools/export_gmimc2_kat.py) —
//! lives here rather than in either sibling, which is what keeps AGENTS.md's
//! "`../ref` is the only other tree you edit" true. It *imports* the oracle and
//! supplies parameters, and that it contains no round function is checked rather
//! than promised: it parses its own syntax tree and fails if it defines anything
//! that shadows the oracle's round function, exponentiates anything, or imports
//! a hash function. [`vectors/README.md`](../../../gmimc2/vectors/README.md)
//! carries the rest, including what `--self-test` proves that no vector can.
//!
//! # The grid: four points, four absences
//!
//! One width per field size — Goldilocks `t=12` at `R=96, α=4`, every 31-bit
//! prime at `t=24` with `R=264, α=2`. The other four points are
//! `Absence::UndefinedByReference`: the specification's round-number tables do
//! not reach those widths, which is an absence rather than a stubbed derivation.
//! `R % t == 0` is the specification's own requirement and is what forced the
//! choice — 264 is not a multiple of 16. See [`params`].
//!
//! # Its sibling
//!
//! `gmimc` is the same family and a distinct design, with a distinct oracle.
//! Two crates, not one with a variant axis: the round functions differ, the
//! exponent means different things, and `../ref` defines one of them and not the
//! other.
//!
//! # The arithmetization, in one paragraph
//!
//! `gmimc`'s, with the two changes the design makes. One committed cell per round
//! and no state at all: only branch 0 is read nonlinearly, so a branch is a
//! running sum of S-box outputs and the one value worth committing is the one
//! entering each S-box — here already carrying its round constant, which is what
//! "permits slightly more efficient circuits" means arithmetically. The
//! permutation is then a recurrence reaching back exactly `t` rounds, telescoping
//! into a constraint of four cells and two S-box terms whatever `t` is, with
//! `M_IO` bracketing the chain at both ends. Width is `R + 2t` and constraints
//! `R + t`, at degree `alpha` — POLICY §11's floor, attained rather than counted.
//! [`air`] carries the derivation beside the constraints it justifies.
//!
//! # Current state
//!
//! POLICY §2 steps 1–6: the oracle is green against the exported vectors, the
//! four AIR files are written, and the §10 ladder runs — including the two layers
//! this layout needs on its own, because a trace of no state cannot say what its
//! cells are by inspection: an independent replay of the round loop, driving the
//! *exported* `M_IO` rather than the crate's own, and the window algebra the
//! telescoped constraint stands on (`tests/air.rs`).
//!
//! The construction is registered in `bench` at all eight grid points, including
//! the four explicit absences, and all five admissible AIR variants are in the
//! typed plan. Layer 5's real `prove` + `verify` runs in
//! `bench/tests/prove_verify.rs`. Measurement remains POLICY §2 step 7.

pub mod air;
pub mod columns;
pub mod generation;
pub mod instances;
pub mod native;
pub mod params;
pub mod vectorized;
