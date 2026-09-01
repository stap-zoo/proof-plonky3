//! The half-commitment layout: one committed cell per Feistel, per round.
//!
//! POLICY §6's four files, for a third variant axis — commit **one output** of
//! each Feistel and recover the other by *differencing*, rather than committing a
//! whole state ([`full_commit`](crate::full_commit::columns)'s `post`) or a whole
//! set of squarings (its `Feistel` registers). `t/4` cells and `t/4` constraints
//! per round, at max constraint degree 4.
//!
//! POLICY §6: two arithmetizations of one permutation are two sets of these four
//! files, one module each, sharing [`crate::params`], [`crate::native`] and the
//! vectors — the precedent is `rescue-prime/src/{half,full}_round/`.
//! [`crate::full_commit`]'s five variants are not edited by this module's
//! existence, so their layouts, degrees and constraint counts are unchanged by
//! construction rather than by inspection.
//!
//! Against `full_commit`'s frontier, per round, with `h = t/2`:
//!
//! | variant | cells / round | max degree |
//! |---|---|---|
//! | `Flattened*` | `h`   | 2 |
//! | `StateOnly*` | `h`   | 4 |
//! | `Spaced*`    | `h/2` | 16 |
//! | here         | `h/2` | **4** |
//!
//! So it dominates `StateOnly*` (half the width at the same degree) and
//! `Spaced*` (the same width at degree 4 instead of 16). Against `Flattened*` it
//! is a tie on POLICY §11's primary reading — `h × 2¹ = h/2 × 2²` — which makes
//! it a measured row rather than an argument: see the crate's benchmark output,
//! not this comment.
//!
//! [`columns`] carries the argument for *which* cell is committed, which is the
//! one choice the whole scheme rests on.

pub mod air;
pub mod columns;
pub mod generation;
pub mod vectorized;
