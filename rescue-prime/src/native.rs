//! The native permutation: **upstream's**, and the oracle everything else is
//! checked against.
//!
//! POLICY §1 — if Plonky3 implements a design, that implementation is the
//! artifact. It does here: `p3_rescue::Rescue` is Rescue-Prime, half-round for
//! half-round. So this file wires it up and computes nothing itself; the round
//! function below is a doc comment, not code, and that is the point.
//!
//! ```text
//! for each of R rounds:
//!     x <- M * x^alpha       + c[2r]        forward half-round
//!     x <- M * x^(1/alpha)   + c[2r+1]      inverse half-round
//! ```
//!
//! No leading or trailing layer, `2R` constant rows, constants added *after*
//! the linear layer.
//!
//! KAT-green **before any AIR exists** (POLICY §2, step 3) and byte-exact
//! against `../ref` (POLICY §4). Never generate a vector from this file or from
//! anything downstream of it — including from an AIR, since a trace is not a
//! vector.
//!
//! # The one piece upstream leaves open
//!
//! `Rescue<F, Mds, WIDTH, ALPHA>` is generic in its MDS, which is what lets the
//! reference's Vandermonde in: upstream's own `MdsMatrix*` types are fast
//! transforms with no entries an AIR could evaluate. [`DenseMds`] is that
//! parameter, applied with `p3_mds::util::mds_multiply`, and it is what this
//! file adds to a type alias.

use harness::permutation::NativePermutation;
use p3_field::{Algebra, PermutationMonomial, PrimeField64};
use p3_mds::MdsPermutation;
use p3_mds::util::mds_multiply;
use p3_rescue::Rescue;
use p3_symmetric::Permutation;

use crate::params::RescuePrimeParams;

/// The reference's dense MDS, as the `MdsPermutation` upstream's permutation
/// asks for.
///
/// Borrows the matrix from the instance's parameters: it is `t x t` field
/// elements, up to 576 of them, and every call would otherwise copy them.
#[derive(Clone, Copy, Debug)]
pub struct DenseMds<'a, F, const WIDTH: usize> {
    matrix: &'a [[F; WIDTH]; WIDTH],
}

impl<'a, F, const WIDTH: usize> DenseMds<'a, F, WIDTH> {
    /// Wrap a reference-derived matrix.
    #[must_use]
    pub const fn new(matrix: &'a [[F; WIDTH]; WIDTH]) -> Self {
        Self { matrix }
    }
}

impl<F: PrimeField64, A: Algebra<F>, const WIDTH: usize> Permutation<[A; WIDTH]>
    for DenseMds<'_, F, WIDTH>
{
    fn permute_mut(&self, state: &mut [A; WIDTH]) {
        mds_multiply(state, self.matrix);
    }
}

impl<F: PrimeField64, A: Algebra<F>, const WIDTH: usize> MdsPermutation<A, WIDTH>
    for DenseMds<'_, F, WIDTH>
{
}

/// Native Rescue-Prime: upstream's permutation, this instance's parameters.
#[derive(Clone, Debug)]
pub struct RescuePrime<'a, F, const WIDTH: usize, const HALF_ROUNDS: usize, const ALPHA: u64>
where
    F: PrimeField64 + PermutationMonomial<ALPHA>,
{
    /// Reference-derived parameters.
    pub params: &'a RescuePrimeParams<F, WIDTH, HALF_ROUNDS>,
    /// Upstream's permutation, holding the same parameters in its own shape.
    inner: Rescue<F, DenseMds<'a, F, WIDTH>, WIDTH, ALPHA>,
}

impl<'a, F, const WIDTH: usize, const HALF_ROUNDS: usize, const ALPHA: u64>
    RescuePrime<'a, F, WIDTH, HALF_ROUNDS, ALPHA>
where
    F: PrimeField64 + PermutationMonomial<ALPHA>,
{
    /// Hand reference-derived parameters to upstream's permutation.
    ///
    /// The constants are flattened in the order upstream indexes them —
    /// `rcons[2r]` is the forward half-round's row and `rcons[2r + 1]` the
    /// inverse one — which is the same row-major order the reference's `grid`
    /// produced them in. `Rescue::new` asserts the count is `2 * t * R`.
    ///
    /// # Panics
    ///
    /// If `ALPHA` disagrees with the parameters' own exponent.
    #[must_use]
    pub fn new(params: &'a RescuePrimeParams<F, WIDTH, HALF_ROUNDS>) -> Self {
        assert_eq!(
            params.alpha, ALPHA,
            "oracle alpha must match its parameters"
        );
        let constants = params.rcons.iter().flatten().copied().collect();
        let inner = Rescue::new(HALF_ROUNDS / 2, constants, DenseMds::new(&params.m));
        Self { params, inner }
    }

    /// Rounds this instance runs — half the half-rounds.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        HALF_ROUNDS / 2
    }
}

impl<F, const WIDTH: usize, const HALF_ROUNDS: usize, const ALPHA: u64> NativePermutation<F>
    for RescuePrime<'_, F, WIDTH, HALF_ROUNDS, ALPHA>
where
    F: PrimeField64 + PermutationMonomial<ALPHA>,
{
    fn name(&self) -> &'static str {
        self.params.name
    }

    fn width(&self) -> usize {
        WIDTH
    }

    fn permute(&self, state: &mut [F]) {
        assert_eq!(state.len(), WIDTH, "one call is one WIDTH-element state");
        let mut work: [F; WIDTH] = core::array::from_fn(|i| state[i]);
        Permutation::permute_mut(&self.inner, &mut work);
        state.copy_from_slice(&work);
    }
}

impl<F, const WIDTH: usize, const HALF_ROUNDS: usize, const ALPHA: u64> Permutation<[F; WIDTH]>
    for RescuePrime<'_, F, WIDTH, HALF_ROUNDS, ALPHA>
where
    F: PrimeField64 + PermutationMonomial<ALPHA>,
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        Permutation::permute_mut(&self.inner, state);
    }
}
