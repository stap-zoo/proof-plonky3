//! `VECTOR_LEN` independent calls per row.
//!
//! POLICY §6: `VectorizedRescuePrimeCols { cols: [RescuePrimeCols; VECTOR_LEN] }`
//! and an `eval` looping the free `eval` of `air.rs` over the lanes. **No
//! round-function logic here** — duplicated logic is two layouts that can
//! disagree, with only one of them tested.
//!
//! One row is `VECTOR_LEN` independent calls, *declared* by overriding
//! `main_next_row_columns()` to `vec![]`: the prover then skips opening the
//! shifted trace, and generation becomes rayon-over-chunks plus SIMD across the
//! packing width. Everything good here follows from the calls being
//! independent, which is why the bare permutation is the whole scope (POLICY
//! §8).
//!
//! Tables are always full — no padding, no selector. Every lane is constrained,
//! so an all-zero row is invalid: a zero state stays zero through S-box and
//! linear layer alike, and the first half-round's constant row is what moves it
//! (`tests/reference.rs` asserts that rather than arguing it).
//!
//! This layout needs its **own** KAT, with the known call at a non-zero lane
//! index and valid calls in the other lanes. It is the only test that catches a
//! lane-indexing bug (POLICY §10).

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeField64;

use crate::half_round::air::{RescuePrimeAir, eval, max_constraint_degree};
use crate::half_round::columns::RescuePrimeCols;
use crate::half_round::generation::generate_vectorized_trace_rows;
use crate::params::RescuePrimeParams;

/// The common number of independent calls packed into one trace row.
pub const VECTOR_LEN: usize = 8;

/// `LANES` side-by-side independent calls.
#[repr(C)]
pub struct VectorizedRescuePrimeCols<
    T,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [RescuePrimeCols<T, WIDTH, REGISTERS, HALF_ROUNDS>; LANES],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `LANES` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedRescuePrimeCols,
    WIDTH: usize,
    REGISTERS: usize,
    HALF_ROUNDS: usize,
    LANES: usize,
);

/// The measured AIR: several independent calls per trace row.
#[derive(Clone, Debug)]
pub struct VectorizedRescuePrimeAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> {
    pub(crate) air: RescuePrimeAir<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA>,
    pub(crate) params: RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedRescuePrimeAir<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA, LANES>
{
    /// Build a vectorized degree/register variant.
    #[must_use]
    pub fn from_params(params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>) -> Self {
        Self {
            air: RescuePrimeAir::from_params(params),
            params: params.clone(),
        }
    }

    /// The scalar AIR used by one lane.
    #[must_use]
    pub const fn lane(&self) -> &RescuePrimeAir<F, WIDTH, REGISTERS, HALF_ROUNDS, ALPHA> {
        &self.air
    }
}

impl_vectorized_air!(
    air: VectorizedRescuePrimeAir { lane: air, params: params },
    cols: VectorizedRescuePrimeCols<WIDTH, REGISTERS, HALF_ROUNDS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, REGISTERS, HALF_ROUNDS, ALPHA, LANES>,
    generics: [
        WIDTH: usize,
        REGISTERS: usize,
        HALF_ROUNDS: usize,
        ALPHA: u64,
        LANES: usize,
    ],
    field: PrimeField64,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(ALPHA, REGISTERS),
    // The reported round count is `R`, the double round the reference counts —
    // not the half-rounds this layout indexes.
    labels: {
        rounds: HALF_ROUNDS / 2,
        sbox_degree: ALPHA,
        sbox_registers: REGISTERS,
    },
);
