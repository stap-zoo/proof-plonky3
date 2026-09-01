//! `VECTOR_LEN` independent calls per row.
//!
//! POLICY §6: `VectorizedGMiMC2Cols { cols: [GMiMC2Cols; VECTOR_LEN] }` and an
//! `eval` looping the free `eval` of `air.rs` over the lanes. **No round-function
//! logic here** — duplicated logic is two layouts that can disagree, with only one
//! of them tested.
//!
//! One row is `VECTOR_LEN` independent calls, *declared* by overriding
//! `main_next_row_columns()` to `vec![]`: the prover then skips opening the
//! shifted trace, and generation becomes rayon-over-chunks plus SIMD across the
//! packing width. Everything good here follows from the calls being independent,
//! which is why the bare permutation is the whole scope (POLICY §8).
//!
//! Tables are always full — no padding, no selector. Every lane is constrained, so
//! an all-zero row is invalid: an all-zero chain makes the base constraint read
//! `0 == rc_0`, which the oracle's non-zero constants refuse.
//!
//! This layout needs its **own** KAT, with the known call at a non-zero lane index
//! and valid calls in the other lanes. It is the only test that catches a
//! lane-indexing bug (POLICY §10).
//!
//! # The packing is the same 8 as everywhere else
//!
//! `VECTOR_LEN` is a property of the harness's taste rather than of an
//! arithmetization, and two rows differing by their packing instead of by their
//! layout is what one shared value prevents — here it also keeps this row directly
//! readable against `gmimc`'s, which is the comparison both crates exist for.

use harness::permutation::{impl_call_columns, impl_vectorized_air};
use p3_field::PrimeCharacteristicRing;

use crate::air::{GMiMC2Air, eval, max_constraint_degree};
use crate::columns::GMiMC2Cols;
use crate::generation::generate_vectorized_trace_rows;
use crate::params::GMiMC2Params;

/// The common number of independent calls packed into one trace row.
pub const VECTOR_LEN: usize = 8;

/// `LANES` side-by-side independent calls.
#[repr(C)]
pub struct VectorizedGMiMC2Cols<
    T,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const LANES: usize,
> {
    pub(crate) cols: [GMiMC2Cols<T, WIDTH, REGISTERS, ROUNDS>; LANES],
}

// The same reinterpretation as one call's columns, one level up: a row is `LANES`
// calls side by side, and `num_cols` here is that row's width.
impl_call_columns!(
    VectorizedGMiMC2Cols,
    WIDTH: usize,
    REGISTERS: usize,
    ROUNDS: usize,
    LANES: usize,
);

/// The measured AIR: one instance, one variant, several independent calls per
/// trace row.
#[derive(Clone, Debug)]
pub struct VectorizedGMiMC2Air<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> {
    pub(crate) air: GMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA>,
}

impl<
    F,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedGMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA, LANES>
{
    /// Build the AIR from one instance's round constants.
    #[must_use]
    pub const fn new(constants: [F; ROUNDS]) -> Self {
        Self {
            air: GMiMC2Air::new(constants),
        }
    }

    /// The single-call AIR one lane of this row is, for `check_constraints` on a
    /// one-call trace and for the smallest failing example.
    #[must_use]
    pub const fn lane(&self) -> &GMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA> {
        &self.air
    }
}

impl<
    F: Copy,
    const WIDTH: usize,
    const REGISTERS: usize,
    const ROUNDS: usize,
    const ALPHA: u64,
    const LANES: usize,
> VectorizedGMiMC2Air<F, WIDTH, REGISTERS, ROUNDS, ALPHA, LANES>
{
    /// Build a vectorized degree/register variant of an instance.
    ///
    /// The variant lives in the type and the instance in the parameters, so the
    /// two cannot be mixed up: an instance's `WIDTH` and `ROUNDS` are part of its
    /// parameter type and have to agree with the alias's.
    #[must_use]
    pub fn from_params(params: &GMiMC2Params<F, WIDTH, ROUNDS>) -> Self {
        Self {
            air: GMiMC2Air::from_params(params),
        }
    }
}

impl_vectorized_air!(
    air: VectorizedGMiMC2Air { lane: air, params: air.constants },
    cols: VectorizedGMiMC2Cols<WIDTH, REGISTERS, ROUNDS, LANES>,
    eval: eval,
    generate: generate_vectorized_trace_rows<WIDTH, REGISTERS, ROUNDS, ALPHA, LANES>,
    generics: [
        WIDTH: usize,
        REGISTERS: usize,
        ROUNDS: usize,
        ALPHA: u64,
        LANES: usize,
    ],
    // A power map, additions, a rotation and two circulant brackets: no integer
    // structure anywhere, so this AIR carries the loosest bound that works
    // (POLICY §6). `params.rs` needs `PrimeField64` for the oracle's own
    // derivation; the constraints do not.
    field: PrimeCharacteristicRing,
    state_width: WIDTH,
    lanes: LANES,
    degree: max_constraint_degree(ALPHA, REGISTERS),
    labels: { rounds: ROUNDS, sbox_degree: ALPHA, sbox_registers: REGISTERS },
);
