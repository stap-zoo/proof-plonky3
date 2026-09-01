//! The full-commitment layouts: every commitment covers a Feistel's whole pair.
//!
//! POLICY §6's four files, for the five variants on POLICY §11's two original
//! axes — how much of a round is flattened, and how many rounds separate two
//! state commitments. [`crate::half_commit`] is the second arithmetization, on a
//! third axis, and [`crate`]'s docs compare them.
//!
//! # What "full" names
//!
//! Whatever a round here commits, it commits *both* of each Feistel's two
//! products or *both* of the outputs they feed:
//!
//! * `REGISTERS = 2` (`Flattened*`) commits both squarings, which leaves the
//!   round map affine in committed cells and the state needing no columns at all;
//!   `REGISTERS = 1` commits one squaring and the state.
//! * `POST = t/2` commits the round's whole new lower half — both elements of
//!   every Feistel's output pair — and the upper half is the previous round's
//!   lower half verbatim.
//! * `SPAN` moves *when* that happens, not how much of it does: a committing
//!   round still commits the full state.
//!
//! `half_commit` is the variant that breaks that: one of the two outputs per
//! Feistel, with the other recovered by differencing. The pair of module names
//! is the pair of answers to "how much of a Feistel does one commitment cover",
//! the way `rescue-prime/src/{half,full}_round/` names how much of a round one
//! commitment covers.
//!
//! # Where the shared pieces live
//!
//! [`crate::params`], [`crate::native`], `linear.rs` and the vectors are shared
//! with `half_commit`, and so is [`vectorized::VECTOR_LEN`]: the packing
//! is a property of the harness's taste rather than of an arithmetization, so
//! both modules are measured at the same one and two rows cannot differ by their
//! packing instead of by their layout.
//!
//! Read [`columns`] for the cell counting and [`air`] for the degree accounting;
//! between them they carry the argument for all five variants.

pub mod air;
pub mod columns;
pub mod generation;
pub mod vectorized;
