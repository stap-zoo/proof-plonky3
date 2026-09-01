//! `VECTOR_LEN` independent calls per row.
//!
//! POLICY §6: `VectorizedXCols { cols: [XCols; VECTOR_LEN] }` and an `eval`
//! looping the free `eval` of `air.rs` over the lanes. **No round-function
//! logic here** — duplicated logic is two layouts that can disagree, with only
//! one of them tested.
//!
//! One row is `VECTOR_LEN` independent calls, *declared* by overriding
//! `main_next_row_columns()` to `vec![]`: the prover then skips opening the
//! shifted trace, and generation becomes rayon-over-chunks plus SIMD across the
//! packing width. Everything good here follows from the calls being
//! independent, which is why the bare permutation is the whole scope (POLICY
//! §8).
//!
//! Tables are always full — no padding, no selector. Every lane is constrained,
//! so an all-zero row is invalid (`post = L(sbox(0 + rc))` is not 0), and the
//! measured call count is exactly `VECTOR_LEN * 2^k`.
//!
//! This layout needs its **own** KAT, with the known call at a non-zero lane
//! index and valid calls in the other lanes. It is the only test that catches a
//! lane-indexing bug (POLICY §10).
//!
//! # This is also where the harness contract lives
//!
//! [`harness::permutation::PermutationAir`] is implemented for the vectorized AIR and not
//! for the scalar one, because `VECTOR_LEN = 1` already *is* the scalar one and
//! two impls would be two things to keep in step. The scalar
//! [`super::air::PSquareHashAir`] stays public because it is what a lane is, and
//! because `check_constraints` on a one-call trace is the smallest failing
//! example when something breaks.

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeCharacteristicRing;

use crate::full_commit::air::{FEISTEL_DEGREE, PSquareHashAir, eval, max_constraint_degree};
use crate::full_commit::columns::PSquareHashCols;
use crate::full_commit::generation::generate_vectorized_trace_rows;
use crate::params::PSquareHashParams;

/// `VECTOR_LEN` calls' worth of cells, side by side in one row.
#[repr(C)]
pub struct VectorizedPSquareHashCols<
    T,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) cols:
        [PSquareHashCols<T, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS>;
            VECTOR_LEN],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `VECTOR_LEN` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedPSquareHashCols,
    WIDTH: usize,
    PAIRS: usize,
    REGISTERS: usize,
    REGISTERS_LAST: usize,
    SPAN: usize,
    POST: usize,
    GROUPS: usize,
    VECTOR_LEN: usize,
);

/// The measured AIR: one instance, one variant, `VECTOR_LEN` calls per row.
#[derive(Debug, Clone)]
pub struct VectorizedPSquareHashAir<
    F,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) air:
        PSquareHashAir<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>,
}

impl<
    F,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    VectorizedPSquareHashAir<
        F,
        WIDTH,
        PAIRS,
        REGISTERS,
        REGISTERS_LAST,
        SPAN,
        POST,
        GROUPS,
        ROUNDS,
        VECTOR_LEN,
    >
{
    /// Build the AIR from an instance's round constants.
    #[must_use]
    pub const fn new(constants: [[[F; 2]; PAIRS]; ROUNDS]) -> Self {
        Self {
            air: PSquareHashAir::new(constants),
        }
    }

    /// The single-call AIR one lane of this row is, for `check_constraints` on a
    /// one-call trace and for the smallest failing example.
    #[must_use]
    pub const fn lane(
        &self,
    ) -> &PSquareHashAir<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS>
    {
        &self.air
    }
}

impl<
    F: Copy,
    const WIDTH: usize,
    const PAIRS: usize,
    const REGISTERS: usize,
    const REGISTERS_LAST: usize,
    const SPAN: usize,
    const POST: usize,
    const GROUPS: usize,
    const ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    VectorizedPSquareHashAir<
        F,
        WIDTH,
        PAIRS,
        REGISTERS,
        REGISTERS_LAST,
        SPAN,
        POST,
        GROUPS,
        ROUNDS,
        VECTOR_LEN,
    >
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
    air: VectorizedPSquareHashAir { lane: air, params: air.constants },
    cols: VectorizedPSquareHashCols<
        WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, VECTOR_LEN
    >,
    eval: eval,
    generate: generate_vectorized_trace_rows<
        WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS, VECTOR_LEN
    >,
    generics: [
        WIDTH: usize,
        PAIRS: usize,
        REGISTERS: usize,
        REGISTERS_LAST: usize,
        SPAN: usize,
        POST: usize,
        GROUPS: usize,
        ROUNDS: usize,
        VECTOR_LEN: usize,
    ],
    // The Feistel is squarings and additions and needs no integer structure,
    // so this AIR carries the loosest bound that works (POLICY §6).
    field: PrimeCharacteristicRing,
    state_width: WIDTH,
    lanes: VECTOR_LEN,
    degree: max_constraint_degree(REGISTERS, REGISTERS_LAST, SPAN, POST, ROUNDS),
    // Not a power map; see `super::air::FEISTEL_DEGREE`.
    labels: {
        rounds: ROUNDS,
        sbox_degree: FEISTEL_DEGREE,
        sbox_registers: REGISTERS,
    },
);

/// The `VECTOR_LEN` every instance is measured at.
///
/// A row of `VECTOR_LEN` calls is what amortizes the per-row costs the prover
/// pays regardless of width. It is a property of the *harness's* taste rather
/// than of pSquareHash, so it is one number here and the same one for every
/// instance — a per-instance choice would make two rows differ by their packing
/// instead of by their arithmetization.
pub const VECTOR_LEN: usize = 8;
