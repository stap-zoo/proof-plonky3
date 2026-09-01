//! Everything a construction is written against and measured by.
//!
//! One job, one goal (POLICY §7): **trivial to add a construction, impossible
//! to bench two of them differently.** A construction provides an
//! implementation of [`permutation::PermutationAir`] and nothing else; the
//! harness provides PCS, MMCS hash, DFT, FRI parameters, challenge extension,
//! challenger, security target and RNG seed, builds them **once per
//! configuration outside every timed region**, and does the identical thing to
//! every AIR handed to it.
//!
//! The corollary is the reason this crate exists at all: a methodology change
//! happens here, in one place. If a change has to touch every construction, it
//! is in the wrong layer.
//!
//! # What is in here
//!
//! | module | contents |
//! |---|---|
//! | [`permutation`] | the two contracts a construction implements, plus POLICY §6's column and trace scaffolding |
//! | [`gadgets`] | paired constraint/witness building blocks and their native oracles |
//! | [`lookup`] | the LogUp path POLICY §12 keeps beside `uni-stark`: batch proving, lookup-aware cost and security, the fixed-table AIR and the frontier axes |
//! | [`grid`] | how a construction writes down its coverage of POLICY §3's grid |
//! | [`config`] | POLICY §7's field × zk matrix, built once per configuration |
//! | [`blowup`] | the one function deriving blowup from a declared degree |
//! | [`mod@measure`] | the unit of work: generate, prove, verify, and the pinned cost columns |
//! | [`run`] | the layer above it: the job list's shape, the two blowup readings, the row schema |
//!
//! They share one crate because they share one rule, below.
//!
//! # The dependency rule
//!
//! This crate must never learn which hashes exist (POLICY §5). Constructions
//! depend on `harness`; `bench` depends on the constructions. Adding a
//! construction crate to this manifest inverts that and destroys the basis for
//! calling the numbers comparable — which is why `bench` asserts against this
//! manifest's text that no construction appears in it.

pub mod blowup;
pub mod config;
pub mod gadgets;
pub mod grid;
pub mod lookup;
pub mod measure;
pub mod permutation;
pub mod run;

pub use blowup::{COMMON_LOG_BLOWUP, measured_log_blowup, min_log_blowup};
pub use config::{Configuration, FieldId, GRID, INPUT_SEED, SECURITY_TARGET_BITS, Zk};
pub use grid::{Absence, GridPoint};
pub use measure::{Measurement, measure};
pub use permutation::{Labels, NativePermutation, PermutationAir};
pub use run::{
    Environment, Job, JobKind, Plan, Reading, Row, Skip, SkipReason, Workload, run_cell,
};
