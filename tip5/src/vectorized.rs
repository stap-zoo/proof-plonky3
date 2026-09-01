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
//! so an all-zero row is invalid: its boundary and round cells cannot all
//! satisfy `post = L(sbox(0)) + rc`. The measured call count is exactly
//! `VECTOR_LEN * 2^k`.
//!
//! This layout needs its **own** KAT, with the known call at a non-zero lane
//! index and valid calls in the other lanes. It is the only test that catches a
//! lane-indexing bug (POLICY §10).

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeField64;

use crate::air::{Tip5Air, eval, max_constraint_degree};
use crate::columns::Tip5Cols;
use crate::generation::generate_vectorized_trace_rows;
use crate::params::{ROUNDS, Tip5Params};

/// Two calls per row. The split witnesses make one call more than five
/// thousand columns; two preserves the required non-zero-lane coverage without
/// multiplying the commitment width by eight.
pub const VECTOR_LEN: usize = 2;

/// `LANES` independent calls side by side.
#[repr(C)]
pub struct VectorizedTip5Cols<
    T,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [Tip5Cols<T, WIDTH, POWER_WORDS, REGISTERS>; LANES],
}

impl_call_columns!(
    VectorizedTip5Cols,
    WIDTH: usize,
    POWER_WORDS: usize,
    REGISTERS: usize,
    LANES: usize,
);

/// The measured vectorized AIR.
#[derive(Clone, Debug)]
pub struct VectorizedTip5Air<
    F,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
    const LANES: usize,
> {
    air: Tip5Air<F, WIDTH, POWER_WORDS, REGISTERS>,
    params: Tip5Params<F, WIDTH>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const POWER_WORDS: usize,
    const REGISTERS: usize,
    const LANES: usize,
> VectorizedTip5Air<F, WIDTH, POWER_WORDS, REGISTERS, LANES>
{
    /// Construct one vectorized register variant.
    #[must_use]
    pub fn from_params(params: &Tip5Params<F, WIDTH>) -> Self {
        Self {
            air: Tip5Air::from_params(params),
            params: params.clone(),
        }
    }

    /// The scalar AIR used by one lane.
    #[must_use]
    pub const fn lane(&self) -> &Tip5Air<F, WIDTH, POWER_WORDS, REGISTERS> {
        &self.air
    }
}

impl_vectorized_air!(
    air: VectorizedTip5Air { lane: air, params: params },
    cols: VectorizedTip5Cols<WIDTH, POWER_WORDS, REGISTERS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, POWER_WORDS, REGISTERS, LANES>,
    generics: [WIDTH: usize, POWER_WORDS: usize, REGISTERS: usize, LANES: usize],
    field: PrimeField64,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(REGISTERS),
    // The split-and-lookup words have no power map at all; the reported degree
    // is the seventh power the other words take.
    labels: { rounds: ROUNDS, sbox_degree: 7, sbox_registers: REGISTERS },
);
