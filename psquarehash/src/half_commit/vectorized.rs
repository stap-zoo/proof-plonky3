//! `VECTOR_LEN` independent calls per row.
//!
//! POLICY §6: `VectorizedHalfCommitCols { cols: [HalfCommitCols; VECTOR_LEN] }`
//! and an `eval` looping the free `eval` of `air.rs` over the lanes. **No
//! round-function logic here** — duplicated logic is two layouts that can
//! disagree, with only one of them tested. That matters more in this module than
//! anywhere else in the crate: the differenced even half is written in exactly
//! one place ([`air::eval_round`](crate::half_commit::air)), and a second copy of
//! it at this level would be a second degree claim nothing cross-checks.
//!
//! One row is `VECTOR_LEN` independent calls, *declared* by overriding
//! `main_next_row_columns()` to `vec![]`: the prover then skips opening the
//! shifted trace, and generation becomes rayon-over-chunks plus SIMD across the
//! packing width. Everything good here follows from the calls being independent,
//! which is why the bare permutation is the whole scope (POLICY §8).
//!
//! Tables are always full — no padding, no selector. Every lane is constrained,
//! so an all-zero row is invalid: with a zero state and zero cells the round's
//! committing constraint reads `0 == y5(0, 0)`, which the round constants make
//! false.
//!
//! This layout needs its **own** KAT, with the known call at a non-zero lane
//! index and valid calls in the other lanes. It is the only test that catches a
//! lane-indexing bug (POLICY §10).
//!
//! # The packing is shared with `full_commit`, deliberately
//!
//! [`VECTOR_LEN`] is re-exported from
//! [`full_commit::vectorized`](crate::full_commit::vectorized) rather than
//! declared again here. It is a property of the *harness's* taste rather than of
//! either arithmetization, and two rows differing by their packing instead of by
//! their layout is exactly what one shared constant exists to prevent — the same
//! reason `rescue-prime/src/full_round/` takes the half-round layout's.
//!
//! # Where the harness contract lives
//!
//! [`harness::permutation::PermutationAir`] is implemented for the vectorized AIR
//! and not for the scalar one, because `VECTOR_LEN = 1` already *is* the scalar
//! one and two impls would be two things to keep in step. The scalar
//! [`super::air::HalfCommitAir`] stays public because it is what a lane is, and
//! because `check_constraints` on a one-call trace is the smallest failing
//! example when something breaks.

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeCharacteristicRing;

use crate::full_commit::air::FEISTEL_DEGREE;
use crate::half_commit::air::{HalfCommitAir, eval, max_constraint_degree};
use crate::half_commit::columns::HalfCommitCols;
use crate::half_commit::generation::generate_vectorized_trace_rows;
use crate::params::PSquareHashParams;

// The packing the whole crate is measured at, shared with `full_commit` — see
// the module docs. Re-exported so that `instances.rs` names one `VECTOR_LEN`
// whichever arithmetization an alias is built from.
pub use crate::full_commit::vectorized::VECTOR_LEN;

/// `VECTOR_LEN` calls' worth of cells, side by side in one row.
#[repr(C)]
pub struct VectorizedHalfCommitCols<
    T,
    const WIDTH: usize,
    const PAIRS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) cols: [HalfCommitCols<T, WIDTH, PAIRS, ROUNDS>; VECTOR_LEN],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `VECTOR_LEN` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedHalfCommitCols,
    WIDTH: usize,
    PAIRS: usize,
    ROUNDS: usize,
    VECTOR_LEN: usize,
);

/// The measured AIR: one instance, the half-commitment variant, `VECTOR_LEN`
/// calls per row.
#[derive(Debug, Clone)]
pub struct VectorizedHalfCommitAir<
    F,
    const WIDTH: usize,
    const PAIRS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) air: HalfCommitAir<F, WIDTH, PAIRS, ROUNDS>,
}

impl<F, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize, const VECTOR_LEN: usize>
    VectorizedHalfCommitAir<F, WIDTH, PAIRS, ROUNDS, VECTOR_LEN>
{
    /// Build the AIR from an instance's round constants.
    #[must_use]
    pub const fn new(constants: [[[F; 2]; PAIRS]; ROUNDS]) -> Self {
        Self {
            air: HalfCommitAir::new(constants),
        }
    }

    /// The single-call AIR one lane of this row is, for `check_constraints` on a
    /// one-call trace and for the smallest failing example.
    #[must_use]
    pub const fn lane(&self) -> &HalfCommitAir<F, WIDTH, PAIRS, ROUNDS> {
        &self.air
    }
}

impl<F: Copy, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize, const VECTOR_LEN: usize>
    VectorizedHalfCommitAir<F, WIDTH, PAIRS, ROUNDS, VECTOR_LEN>
{
    /// Build this variant of an instance.
    ///
    /// The variant lives in the type (see `crate::instances`' aliases) and the
    /// instance in the parameters, so the two cannot be mixed up: an instance's
    /// `PAIRS` and `ROUNDS` are part of its parameter type and have to agree with
    /// the alias's.
    #[must_use]
    pub fn from_params(params: &PSquareHashParams<F, WIDTH, PAIRS, ROUNDS>) -> Self {
        Self::new(params.rcons)
    }
}

impl_vectorized_air!(
    air: VectorizedHalfCommitAir { lane: air, params: air.constants },
    cols: VectorizedHalfCommitCols<WIDTH, PAIRS, ROUNDS, VECTOR_LEN>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, PAIRS, ROUNDS, VECTOR_LEN>,
    generics: [
        WIDTH: usize,
        PAIRS: usize,
        ROUNDS: usize,
        VECTOR_LEN: usize,
    ],
    // The Feistel is squarings and additions and needs no integer structure,
    // so this AIR carries the loosest bound that works (POLICY §6).
    field: PrimeCharacteristicRing,
    state_width: WIDTH,
    lanes: VECTOR_LEN,
    // Four at every round count, and a fixed point rather than a coincidence —
    // `super::air` derives it and `harness::measure` cross-checks it against the
    // symbolic degree of this very AIR on every measured row (POLICY §7).
    degree: max_constraint_degree(ROUNDS),
    // Not a power map; see `crate::full_commit::air::FEISTEL_DEGREE`.
    //
    // `sbox_registers` is cells committed per S-box, and this layout commits one
    // per Feistel, so 1 is the honest entry where 0 would read as "nothing
    // committed per S-box". The one cell is an **output** of the Feistel —
    // `out[2p+1]`, the odd half of the round's new lower half — and not one of its
    // squarings, which is what `full_commit`'s `REGISTERS` counts; the Feistel's
    // other output is recovered from it by differencing rather than committed.
    // Labels are labels and nothing branches on them (POLICY §7), but this one
    // appears in the cross-construction table, so the difference is worth stating
    // where the number is written.
    labels: {
        rounds: ROUNDS,
        sbox_degree: FEISTEL_DEGREE,
        sbox_registers: 1,
    },
);
