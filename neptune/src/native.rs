//! The native permutation: the oracle everything else is checked against.
//!
//! KAT-green **before any AIR exists** (POLICY §2, step 3) and byte-exact
//! against `../ref` (POLICY §4). Never generate a vector from this file or from
//! anything downstream of it — including from an AIR, since a trace is not a
//! vector.
//!
//! It is separate from `params.rs`, mirroring the reference's split between
//! `hash.py` and `params.py` and keeping the native oracle distinct from its
//! parameter derivation (POLICY §5).

use harness::gadgets::power_map::power_map;
use harness::permutation::add_round_constants;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_mds::util::mds_multiply;
use p3_poseidon2::matmul_internal;
use p3_symmetric::Permutation;

use crate::params::NeptuneParams;

/// The quadratic pair-wise external S-box, Equation 22 of Neptune.
///
/// Generic in the ring rather than in the field, which is what lets the oracle,
/// `generation.rs` and `air.rs` run the *same* map: the oracle over `F`, the
/// generator over `F` and `F::Packing`, the AIR over `AB::Expr`. It was three
/// copies before, two of them byte-identical, which is exactly the drift POLICY
/// §6 warns about — and nothing is lost by sharing, because the oracle is
/// byte-exact against `../ref` (POLICY §4), so a bug in this function fails the
/// known-answer test before it reaches anything downstream.
///
/// Degree four in its input: `(u - v)²` is a square of a value that is itself a
/// square. That is the floor for an AIR that commits nothing inside the pair
/// map — which is what [`crate::air`]'s unsplit variant does, and why its
/// degree is four however cheap the internal exponent is.
///
/// It is *not* a floor on the construction. Committing the first square breaks
/// the composition in half and takes the pair map to degree two, because
/// `u - v = a - 2b + gamma` is affine in `(p0, p1, s)`; that is the split the
/// [`external_sbox_first_square`] / [`external_sbox_pair`] pair exists to
/// serve, and the reason this function is written through them rather than
/// beside them (POLICY §6: one round function, three callers).
#[inline]
pub fn external_sbox<E: PrimeCharacteristicRing>(state: &mut [E], gamma: E) {
    for pair in state.chunks_exact_mut(2) {
        let square = external_sbox_first_square(&pair[0], &pair[1]);
        let (out0, out1) = external_sbox_pair(pair[0].dup(), pair[1].dup(), square, gamma.dup());
        pair[0] = out0;
        pair[1] = out1;
    }
}

/// The pair map's first Lai--Massey square, `(p0 - p1)²`.
///
/// Split out because it is the one value a degree-reduced AIR commits: the
/// register variant witnesses this, pins it with `register == (p0 - p1)²`, and
/// hands the register back to [`external_sbox_pair`] in place of the expression.
#[inline]
pub fn external_sbox_first_square<E: PrimeCharacteristicRing>(p0: &E, p1: &E) -> E {
    (p0.dup() - p1.dup()).square()
}

/// The rest of Equation 22, given that first square.
///
/// Degree two in `(p0, p1, first_square, gamma)` jointly — every term is affine
/// in them except the single `(u - v)²`. So the caller decides the AIR's degree
/// by deciding what `first_square` is: an expression (degree two in the state,
/// making this degree four) or a committed cell (degree one, making this two).
#[inline]
pub fn external_sbox_pair<E: PrimeCharacteristicRing>(
    p0: E,
    p1: E,
    first_square: E,
    gamma: E,
) -> (E, E) {
    let a = p0 + first_square.dup();
    let b = p1 + first_square;
    let u = a.dup().double() + b.dup() + gamma.dup();
    let v = a + b.dup().double() + b;
    let s = (u.dup() - v.dup()).square();
    (u + s.dup() - gamma, v + s)
}

/// Native Neptune permutation. It uses Plonky3's generic dense MDS multiply
/// for Neptune's external layer and Poseidon2's optimized `J + diag` routine
/// for its internal layer. Only Neptune's split matrix and Lai--Massey map are
/// local.
#[derive(Clone, Debug)]
pub struct Neptune<
    F: PrimeField64,
    const WIDTH: usize,
    const EXT: usize,
    const INT: usize,
    const DEGREE: u64,
> {
    /// Reference-derived parameters.
    pub params: NeptuneParams<F, WIDTH, EXT, INT>,
}

impl<F: PrimeField64, const WIDTH: usize, const EXT: usize, const INT: usize, const DEGREE: u64>
    Neptune<F, WIDTH, EXT, INT, DEGREE>
{
    /// Construct from reference-derived parameters.
    #[must_use]
    pub const fn new(params: NeptuneParams<F, WIDTH, EXT, INT>) -> Self {
        Self { params }
    }

    /// Apply the permutation for the field's selected S-box exponent.
    pub fn permute_with_degree(&self, state: &mut [F; WIDTH]) {
        mds_multiply(state, &self.params.m_ext);
        let half_ext = EXT / 2;
        for round in 0..(EXT + INT) {
            if round >= half_ext && round < half_ext + INT {
                state[0] = power_map::<F, DEGREE>(state[0]);
                matmul_internal(state, self.params.m_int_diag_m_1);
            } else {
                external_sbox(state, self.params.gamma);
                mds_multiply(state, &self.params.m_ext);
            }
            add_round_constants(state, &self.params.rcons[round + 1]);
        }
    }
}

impl<F: PrimeField64, const WIDTH: usize, const EXT: usize, const INT: usize, const DEGREE: u64>
    Permutation<[F; WIDTH]> for Neptune<F, WIDTH, EXT, INT, DEGREE>
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        self.permute_with_degree(state);
    }
}
