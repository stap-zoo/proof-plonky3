//! The permutation contract, in two halves.
//!
//! * [`NativePermutation`] — the native oracle. It is what the known-answer
//!   vectors run against, and for a construction we write it is green *before
//!   any AIR exists* (POLICY §2, step 3). Everything downstream is checked
//!   against it, so it is never checked against anything downstream.
//! * [`PermutationAir`] — the AIR contract the rest of this crate consumes. A
//!   construction provides exactly what POLICY §7 lists: `Constants` and `new`
//!   (its own inherent API), the trace generators, the `Air<..>` bounds the
//!   prover, verifier, symbolic and debug builders need, and its [`Labels`].
//!
//! Only the bare permutation is arithmetized, and a mode of operation is out of
//! scope (POLICY §8): the vectorized layout proves *independent* calls, and a
//! trace whose values happen to chain proves nothing about the chain.
//!
//! Two more modules carry the parts of POLICY §6's four-file shape that are the
//! same in every construction, so that a construction declares its layout and
//! its round function and nothing else:
//!
//! * [`impl_call_columns`] — `num_cols` and the `Borrow`/`BorrowMut`
//!   reinterpretation of a row as one call's columns, alignment assertions
//!   included.
//! * [`impl_vectorized_air`] — the vectorized row: `BaseAir`, `Air` looping the
//!   free per-call `eval` over the lanes, and [`PermutationAir`].
//! * [`trace::fill_trace`] — the `MaybeUninit` allocation, the packing-width
//!   dispatch, and the forwarded `extra_capacity_bits`.
//! * [`layers`] — adding a constant row, and turning a state expression into
//!   committed cells.

mod air;
mod columns;
mod labels;
pub mod layers;
mod native;
pub mod trace;
mod vectorized;

pub use air::PermutationAir;
pub use labels::Labels;
pub use layers::{add_round_constants, assert_state_eq, commit_state};
pub use native::NativePermutation;

// `#[macro_export]` puts the macro at the crate root; re-exporting it here is
// what lets a construction import it beside the contracts it belongs with.
pub use crate::{impl_call_columns, impl_vectorized_air};
