//! Vectorized, independent Neptune calls.  This module owns no round-function
//! logic: it loops the scalar AIR and delegates witness construction to
//! [`crate::generation`].

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeField64;

use crate::air::{NeptuneAir, eval, max_constraint_degree};
use crate::columns::NeptuneCols;
use crate::generation::generate_vectorized_trace_rows;
use crate::params::NeptuneParams;

/// The common number of independent calls packed into one trace row.
pub const VECTOR_LEN: usize = 8;

/// `LANES` side-by-side independent calls.
#[repr(C)]
pub struct VectorizedNeptuneCols<
    T,
    const WIDTH: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const LM: usize,
    const PREGS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [NeptuneCols<T, WIDTH, HALF_EXT, INT, LM, PREGS>; LANES],
}

// The same reinterpretation as one call's columns, one level up: a row is
// `LANES` calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedNeptuneCols,
    WIDTH: usize,
    HALF_EXT: usize,
    INT: usize,
    LM: usize,
    PREGS: usize,
    LANES: usize,
);

/// The measured AIR: several independent calls per trace row.
#[derive(Clone, Debug)]
pub struct VectorizedNeptuneAir<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
    const LANES: usize,
> {
    pub(crate) air: NeptuneAir<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS>,
}

impl<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const HALF_EXT: usize,
    const INT: usize,
    const DEGREE: u64,
    const LM: usize,
    const PREGS: usize,
    const LANES: usize,
> VectorizedNeptuneAir<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS, LANES>
{
    /// Build a vectorized AIR from reference-derived parameters.
    #[must_use]
    pub const fn new(params: NeptuneParams<F, WIDTH, EXT, INT>) -> Self {
        Self {
            air: NeptuneAir::new(params),
        }
    }
}

impl_vectorized_air!(
    air: VectorizedNeptuneAir { lane: air, params: air.params },
    cols: VectorizedNeptuneCols<WIDTH, HALF_EXT, INT, LM, PREGS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS, LANES>,
    generics: [
        WIDTH: usize,
        EXT: usize,
        HALF_EXT: usize,
        INT: usize,
        DEGREE: u64,
        LM: usize,
        PREGS: usize,
        LANES: usize,
    ],
    field: PrimeField64,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(DEGREE, LM, PREGS),
    // `sbox_degree` is Neptune's internal exponent, which is what alpha means
    // everywhere else in the table. It used to carry the Lai--Massey floor
    // instead, which agreed with alpha only because 7 > 4; once a register
    // variant takes the pair map to degree two that reading would report
    // "alpha 2" for a construction whose S-box is still `x^7`. The floor is
    // not lost — it lives in `max_constraint_degree`, which is the label the
    // harness actually consumes.
    labels: {
        rounds: EXT + INT,
        sbox_degree: DEGREE,
        sbox_registers: PREGS,
    },
);
