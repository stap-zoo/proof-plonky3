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
//!
//! # The round, and the one asymmetry in it
//!
//! ```text
//! y_0 = x_0^(1/alpha)                        the inverse power
//! y_1 = x_1^alpha                            the forward power
//! y_i = x_i * G_i(L_i(y_0, y_1, x_{i-1}))    the Horst words, i >= 2
//! ```
//!
//! with `L_i(y_0, y_1, z) = (i-1)*y_0 + y_1 + z`, and `z = 0` at `i = 2` where
//! there is no feedback word. Then the linear layer, then the round constants.
//!
//! **`L_i`'s third argument is the round's *input* word `x_{i-1}`, not the
//! output `y_{i-1}`.** That is what makes the Horst layer parallel rather than
//! sequential — every `y_i` depends on the round input plus the two S-box
//! outputs and on nothing else — and the AIR's whole degree accounting rests on
//! it. Reading it as `y_{i-1}` still gives a permutation, one that would pass
//! every structural test and no vector.

use harness::gadgets::inverse_power_map::inverse_power_map;
use harness::gadgets::power_map::power_map;
use harness::permutation::{NativePermutation, add_round_constants};
use p3_field::{Algebra, PrimeField64};
use p3_mds::util::mds_multiply;
use p3_symmetric::Permutation;

use crate::params::GriffinParams;

/// Native Griffin permutation, borrowing an instance's parameters.
#[derive(Clone, Debug)]
pub struct Griffin<'a, F, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64> {
    /// Reference-derived parameters.
    pub params: &'a GriffinParams<F, WIDTH, ROUNDS>,
}

/// The non-linear layer, over any algebra over `F`.
///
/// Generic in the ring rather than in the field so that `generation.rs` runs the
/// identical code over `F::Packing`. The one thing it cannot do generically is
/// the inverse power, which is why it takes both S-box outputs as arguments
/// rather than computing `y_0` itself — and that split is also exactly the AIR's
/// split, where `y_0` is a witnessed cell.
#[inline]
pub fn horst_layer<F: PrimeField64, A: Algebra<F>, const WIDTH: usize, const ROUNDS: usize>(
    params: &GriffinParams<F, WIDTH, ROUNDS>,
    state: &[A; WIDTH],
    y_0: &A,
    y_1: &A,
) -> [A; WIDTH] {
    core::array::from_fn(|i| match i {
        0 => y_0.dup(),
        1 => y_1.dup(),
        _ => {
            // `x_{i-1}` is the round *input* word, and there is none at i = 2.
            let z = if i == 2 { A::ZERO } else { state[i - 1].dup() };
            let l = y_0.dup() * F::from_usize(i - 1) + y_1.dup() + z;
            state[i].dup() * params.quadratic(i, l)
        }
    })
}

impl<'a, F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    Griffin<'a, F, WIDTH, ROUNDS, ALPHA>
{
    /// Borrow reference-derived parameters as the oracle.
    ///
    /// # Panics
    ///
    /// If `ALPHA` disagrees with the parameters' own exponent.
    #[must_use]
    pub fn new(params: &'a GriffinParams<F, WIDTH, ROUNDS>) -> Self {
        assert_eq!(
            params.alpha, ALPHA,
            "oracle alpha must match its parameters"
        );
        Self { params }
    }

    /// Rounds this instance runs.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        ROUNDS
    }
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    NativePermutation<F> for Griffin<'_, F, WIDTH, ROUNDS, ALPHA>
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
        Permutation::permute_mut(self, &mut work);
        state.copy_from_slice(&work);
    }
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    Permutation<[F; WIDTH]> for Griffin<'_, F, WIDTH, ROUNDS, ALPHA>
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        let params = self.params;

        // `_pre_rounds`: one linear layer before the first round.
        mds_multiply(state, &params.m);

        for round in 0..ROUNDS {
            let y_0 = inverse_power_map(state[0], params.alpha_inv);
            let y_1 = power_map::<F, ALPHA>(state[1]);
            *state = horst_layer(params, state, &y_0, &y_1);

            mds_multiply(state, &params.m);
            // `rcons[ROUNDS - 1]` is the zero row: the final round adds none.
            add_round_constants(state, &params.rcons[round]);
        }
    }
}
