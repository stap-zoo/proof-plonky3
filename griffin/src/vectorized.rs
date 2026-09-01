//! `VECTOR_LEN` independent calls per row.
//!
//! POLICY §6: `VectorizedGriffinCols { cols: [GriffinCols; VECTOR_LEN] }` and an
//! `eval` looping the free `eval` of `air.rs` over the lanes. **No
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
//! so an all-zero row is invalid, and the measured call count is exactly
//! `VECTOR_LEN * 2^k`. Griffin is the construction where "an all-zero row is
//! invalid" is worth checking rather than assuming: its round function *does*
//! fix the all-zero state, and only the round constants move it (see
//! `tests/reference.rs`).
//!
//! This layout needs its **own** KAT, with the known call at a non-zero lane
//! index and valid calls in the other lanes. It is the only test that catches a
//! lane-indexing bug (POLICY §10).

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeField64;

use crate::air::{GriffinAir, eval, max_constraint_degree};
use crate::columns::GriffinCols;
use crate::generation::generate_vectorized_trace_rows;
use crate::params::GriffinParams;

/// The common number of independent calls packed into one trace row.
pub const VECTOR_LEN: usize = 8;

/// `LANES` side-by-side independent calls.
#[repr(C)]
pub struct VectorizedGriffinCols<
    T,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [GriffinCols<T, WIDTH, REGISTERS, ROUNDS>; LANES],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `LANES` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedGriffinCols,
    WIDTH: usize,
    REGISTERS: usize,
    ROUNDS: usize,
    LANES: usize,
);

/// The measured AIR: several independent calls per trace row.
#[derive(Clone, Debug)]
pub struct VectorizedGriffinAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> {
    pub(crate) air: GriffinAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA>,
    pub(crate) params: GriffinParams<F, WIDTH, ROUNDS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedGriffinAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA, LANES>
{
    /// Build a vectorized degree/register variant.
    #[must_use]
    pub fn from_params(params: &GriffinParams<F, WIDTH, ROUNDS>) -> Self {
        Self {
            air: GriffinAir::from_params(params),
            params: params.clone(),
        }
    }

    /// The scalar AIR used by one lane.
    #[must_use]
    pub const fn lane(&self) -> &GriffinAir<F, WIDTH, REGISTERS, ROUNDS, ALPHA> {
        &self.air
    }
}

impl_vectorized_air!(
    air: VectorizedGriffinAir { lane: air, params: params },
    cols: VectorizedGriffinCols<WIDTH, REGISTERS, ROUNDS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, REGISTERS, ROUNDS, ALPHA, LANES>,
    generics: [WIDTH: usize, REGISTERS: usize, ROUNDS: usize, ALPHA: u64, LANES: usize],
    field: PrimeField64,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(ALPHA, REGISTERS),
    labels: { rounds: ROUNDS, sbox_degree: ALPHA, sbox_registers: REGISTERS },
);
