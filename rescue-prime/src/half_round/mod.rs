//! The half-round layout: one committed state per half-round.
//!
//! POLICY §6's four files, for the arithmetization that commits the non-linear
//! layer's output after **every** half-round. `crate::full_round` is the other
//! one, and [`crate`]'s docs compare them.
//!
//! Every constraint here reads a single committed cell against a single linear
//! layer's worth of the previous ones, which is the simplest form this round
//! function admits and the baseline the coarser layout is measured against.

pub mod air;
pub mod columns;
pub mod generation;
pub mod vectorized;
