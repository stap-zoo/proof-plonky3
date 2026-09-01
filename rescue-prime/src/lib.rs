//! Rescue-Prime — native wrapped, **AIR written**, arithmetized twice.
//!
//! `p3-rescue` gives the native permutation and no AIR (POLICY §1), so the
//! oracle is upstream's and POLICY §6's four files are ours — twice, in
//! [`half_round`] and [`full_round`], over one set of parameters and one set of
//! vectors.
//!
//! # Which member of the family this is
//!
//! Rescue-Prime, of the reference's `marvellous` module. It covers POLICY §3's
//! grid: `_init_mat` (Vandermonde, transposed), `_init_cons` (SHAKE-256) and
//! `_init_rounds` are implemented for every prime and width, and the reference
//! pins both Goldilocks points itself. And it is what `p3_rescue::Rescue`
//! computes, half-round for half-round —
//!
//! ```text
//! for each of R rounds:
//!     x <- M * x^alpha       + c[2r]        forward half-round
//!     x <- M * x^(1/alpha)   + c[2r+1]      inverse half-round
//! ```
//!
//! — with `2*R*t` constants and no leading or trailing layer. The family's RPO
//! branch reaches the tables through `xhash8` and `xhash16`, which extend it in
//! the same reference module.
//!
//! # Wrapping, with the reference's vectors
//!
//! `Rescue<F, Mds, WIDTH, ALPHA>` is generic in its MDS and takes its constants
//! by value, so the native side is upstream's permutation driven with the
//! reference's own parameters, and one KAT validates both sides (POLICY §4).
//! Two facts make that a wrap rather than a re-port, both checked in
//! `tests/reference.rs`:
//!
//! * **the constants coincide by derivation.** Upstream seeds SHAKE-256 with
//!   `"Rescue-XLIX(p,t,capacity,kappa)"` and reads `ceil(bits/8)+1` bytes
//!   little-endian mod `p`; the reference's `_init_cons` is the same seed string
//!   and the same `sampling="mod"` rule, whose docstring names the Marvellous
//!   script as what it matches.
//! * **the round count coincides.** Upstream's `Rescue::num_rounds` and the
//!   reference's `_init_rounds` both return `8` at every grid point — both sit
//!   on the `max(5, ...)` floor, so `R` does not grow with `t` here.
//!
//! The MDS is the one parameter upstream keeps private: its `MdsMatrix*` types
//! are fast transforms with no entries an AIR could evaluate. `params.rs`
//! derives it from the reference's rule and `native.rs` passes it *into*
//! upstream's permutation.
//!
//! # Two layouts, and why the second exists
//!
//! Rescue alternates `x^alpha` with `x^(1/alpha)`, and the expensive direction
//! is free in an AIR whichever way the trace commits, because a power map read
//! backwards is a power map on the other side of the equation:
//!
//! ```text
//! half_round   commit y = S(x) each half-round:  y_i == x_i^alpha    forward
//!                                                y_i^alpha == x_i    inverse
//!
//! full_round   commit s once per round:
//!              (M^{-1}(s_next - c1))_i ^ alpha == (M * s^alpha + c0)_i
//! ```
//!
//! Both are degree `alpha`, and both put one call in one row at every grid
//! point. The second commits half as many cells for it — POLICY §11's
//! commitment-spacing axis, at the one design where it costs no degree — and
//! `tests/numbers.rs` pins what that is worth: half the cells unsplit, 26% fewer
//! split, fewer constraints, and equal constraint-expression size. Prover time
//! and proof size are the numbers that decide which one the tables carry.
//!
//! # Validation (POLICY §4, §10)
//!
//! Native and both AIRs replay the *same* reference vectors, which keeps the
//! wrapped oracle and the two arithmetizations consistent. POLICY §10 defines
//! the common layers; `tests/air.rs` and `tests/full_round.rs` independently run
//! the trace, constraint, corruption and lane-sensitive checks for each layout,
//! while `tests/numbers.rs` compares their shapes. The release proof test covers
//! both layouts in `bench/tests/prove_verify.rs` under the shared 100-bit
//! configuration.

pub mod full_round;
pub mod half_round;
pub mod instances;
pub mod native;
pub mod params;
