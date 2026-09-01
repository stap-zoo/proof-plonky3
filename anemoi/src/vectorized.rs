//! Vectorized Anemoi AIR: independent calls sharing one trace row.

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeCharacteristicRing;

use crate::air::{AnemoiAir, eval, max_constraint_degree};
use crate::columns::AnemoiCols;
use crate::generation::generate_vectorized_trace_rows;
use crate::params::AnemoiParams;

/// Calls packed into each measured row.
pub const VECTOR_LEN: usize = 8;

/// Side-by-side independent calls.
#[repr(C)]
pub struct VectorizedAnemoiCols<
    T,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [AnemoiCols<T, WIDTH, COLUMNS, REGISTERS, ROUNDS>; LANES],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `LANES` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedAnemoiCols,
    WIDTH: usize,
    COLUMNS: usize,
    REGISTERS: usize,
    ROUNDS: usize,
    LANES: usize,
);

/// Measured Anemoi AIR plus the native-only inverse exponent needed by its
/// witness generator.
#[derive(Clone, Debug)]
pub struct VectorizedAnemoiAir<
    F,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> {
    pub(crate) air: AnemoiAir<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA>,
    params: AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
}

impl<
    F: Copy,
    const WIDTH: usize,
    const COLUMNS: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedAnemoiAir<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA, LANES>
{
    /// Build a vectorized degree/register variant.
    #[must_use]
    pub fn from_params(params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>) -> Self {
        Self {
            air: AnemoiAir::from_params(params),
            params: params.clone(),
        }
    }

    /// The scalar AIR used by one lane.
    #[must_use]
    pub const fn lane(&self) -> &AnemoiAir<F, WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA> {
        &self.air
    }
}

impl_vectorized_air!(
    air: VectorizedAnemoiAir { lane: air, params: params },
    cols: VectorizedAnemoiCols<WIDTH, COLUMNS, REGISTERS, ROUNDS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, COLUMNS, REGISTERS, ROUNDS, ALPHA, LANES>,
    generics: [
        WIDTH: usize,
        COLUMNS: usize,
        REGISTERS: usize,
        ROUNDS: usize,
        ALPHA: u64,
        LANES: usize,
    ],
    // The closed Flystel needs no integer structure: only the *oracle* takes the
    // inverse power, so this AIR stays generic in the loosest bound (POLICY §6).
    field: PrimeCharacteristicRing,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(ALPHA, REGISTERS),
    labels: { rounds: ROUNDS, sbox_degree: ALPHA, sbox_registers: REGISTERS },
);
