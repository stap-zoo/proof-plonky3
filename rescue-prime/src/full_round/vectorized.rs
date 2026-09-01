//! `VECTOR_LEN` independent calls per row, full-round layout.
//!
//! POLICY §6: the columns struct is `LANES` copies of one call's, and `eval`
//! loops `air.rs`'s free function over the lanes. **No round-function logic
//! here.**
//!
//! `VECTOR_LEN` is the half-round layout's, deliberately: the two
//! arithmetizations are compared per call, and a different packing factor would
//! rescale one of them for no reason.

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeField64;

use crate::full_round::air::{FullRoundAir, eval, max_constraint_degree};
use crate::full_round::columns::FullRoundCols;
use crate::full_round::generation::generate_vectorized_trace_rows;
use crate::params::RescuePrimeParams;

/// `LANES` side-by-side independent calls.
#[repr(C)]
pub struct VectorizedFullRoundCols<
    T,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [FullRoundCols<T, WIDTH, REGISTERS, ROUNDS>; LANES],
}

impl_call_columns!(
    VectorizedFullRoundCols,
    WIDTH: usize,
    REGISTERS: usize,
    ROUNDS: usize,
    LANES: usize,
);

/// The measured AIR: several independent calls per trace row.
#[derive(Clone, Debug)]
pub struct VectorizedFullRoundAir<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> {
    pub(crate) air: FullRoundAir<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA>,
    pub(crate) params: RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const HALF_ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedFullRoundAir<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA, LANES>
{
    /// Build a vectorized degree/register variant.
    #[must_use]
    pub fn from_params(params: &RescuePrimeParams<F, WIDTH, HALF_ROUNDS>) -> Self {
        Self {
            air: FullRoundAir::from_params(params),
            params: params.clone(),
        }
    }

    /// The scalar AIR used by one lane.
    #[must_use]
    pub const fn lane(&self) -> &FullRoundAir<F, WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA> {
        &self.air
    }
}

impl_vectorized_air!(
    air: VectorizedFullRoundAir { lane: air, params: params },
    cols: VectorizedFullRoundCols<WIDTH, REGISTERS, ROUNDS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<
        WIDTH, REGISTERS, ROUNDS, HALF_ROUNDS, ALPHA, LANES
    >,
    generics: [
        WIDTH: usize,
        REGISTERS: usize,
        ROUNDS: usize,
        HALF_ROUNDS: usize,
        ALPHA: u64,
        LANES: usize,
    ],
    field: PrimeField64,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(ALPHA, REGISTERS),
    labels: { rounds: ROUNDS, sbox_degree: ALPHA, sbox_registers: REGISTERS },
);
