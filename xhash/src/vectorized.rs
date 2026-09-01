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

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeField64;

use crate::air::{XHashAir, eval, max_constraint_degree};
use crate::columns::XHashCols;
use crate::generation::generate_vectorized_trace_rows;
use crate::params::XHashParams;

/// Independent calls packed side by side in one trace row.
pub const VECTOR_LEN: usize = 8;

/// `LANES` side-by-side XHash calls.
#[repr(C)]
pub struct VectorizedXHashCols<
    T,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const LANES: usize,
> {
    pub(crate) cols: [XHashCols<T, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES>; LANES],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `LANES` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedXHashCols,
    WIDTH: usize,
    ACTIVE: usize,
    REGISTERS: usize,
    P3_BLOCKS: usize,
    CYCLES: usize,
    LANES: usize,
);

/// The measured AIR: several independent calls per trace row.
#[derive(Clone, Debug)]
pub struct VectorizedXHashAir<
    F,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
    const LANES: usize,
> {
    pub(crate) air: XHashAir<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA>,
    pub(crate) params: XHashParams<F, WIDTH, CONSTANT_ROWS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const ACTIVE: usize,
    const REGISTERS: usize,
    const P3_BLOCKS: usize,
    const CYCLES: usize,
    const CONSTANT_ROWS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedXHashAir<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA, LANES>
{
    /// Build a vectorized degree/register variant.
    #[must_use]
    pub fn from_params(params: &XHashParams<F, WIDTH, CONSTANT_ROWS>) -> Self {
        Self {
            air: XHashAir::from_params(params),
            params: params.clone(),
        }
    }

    /// The scalar AIR used by one lane.
    #[must_use]
    pub const fn lane(
        &self,
    ) -> &XHashAir<F, WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA> {
        &self.air
    }
}

impl_vectorized_air!(
    air: VectorizedXHashAir { lane: air, params: params },
    cols: VectorizedXHashCols<WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<
        WIDTH, ACTIVE, REGISTERS, P3_BLOCKS, CYCLES, CONSTANT_ROWS, ALPHA, LANES
    >,
    generics: [
        WIDTH: usize,
        ACTIVE: usize,
        REGISTERS: usize,
        P3_BLOCKS: usize,
        CYCLES: usize,
        CONSTANT_ROWS: usize,
        ALPHA: u64,
        LANES: usize,
    ],
    field: PrimeField64,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(ALPHA, REGISTERS),
    // One reported round is one F/B pair, so a cycle is two.
    labels: { rounds: CYCLES * 2, sbox_degree: ALPHA, sbox_registers: REGISTERS },
);
