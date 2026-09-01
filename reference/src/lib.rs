//! `../ref`, in Rust: its vectors and its derivations.
//!
//! POLICY §4 makes the reference the only ground truth, and this crate is the
//! whole of that boundary. It has two halves, which are the two ways a
//! reference value reaches an implementation:
//!
//! * [`kat`] — reading `../ref`'s exports. Known-answer vectors and the
//!   parameter export, `include_str!`d by the construction that owns them.
//! * [`sampler`] — reproducing `../ref`'s own derivations, for the grid points
//!   where the reference pins no constants and only a rule.
//!
//! # What it never does
//!
//! **Never generate a vector from the implementation under test** — including
//! from an AIR, since a trace is not a vector — and **never hand-write a
//! constant or re-derive outside the reference** (POLICY §3, §4). Both halves
//! are locked byte-exact against `../ref`'s own output in their own tests, so
//! this crate can be the oracle without ever being checked against something
//! downstream of it.
//!
//! Like `harness`, it knows no construction; unlike `harness`, it also knows
//! nothing about arithmetization. A sampler emits no constraints and runs once,
//! at parameter-construction time, out of circuit and out of trace — nothing
//! here ever appears in an AIR.

pub mod kat;
pub mod sampler;

pub use kat::{Instance, Kat, Params, Vectors};
pub use sampler::{Shake128BitmaskSampler, ShakeModSampler, field_order_seed, inverse_exponent};
