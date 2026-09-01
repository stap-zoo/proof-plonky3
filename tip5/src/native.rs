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
use harness::permutation::{NativePermutation, add_round_constants};
use p3_field::PrimeField64;
use p3_mds::util::mds_multiply;
use p3_symmetric::Permutation;

use crate::params::{BYTES, MONT_R, ROUNDS, SPLIT_WORDS, Tip5Params};

/// Tip5-family native permutation, borrowing exact exported parameters.
#[derive(Clone, Debug)]
pub struct Tip5<'a, F, const WIDTH: usize> {
    /// Reference parameters.
    pub params: &'a Tip5Params<F, WIDTH>,
}

impl<'a, F: PrimeField64, const WIDTH: usize> Tip5<'a, F, WIDTH> {
    /// Construct the native oracle.
    #[must_use]
    pub const fn new(params: &'a Tip5Params<F, WIDTH>) -> Self {
        Self { params }
    }

    /// Apply the split-and-lookup S-box using the canonical integer encoding.
    #[must_use]
    pub fn split_sbox(&self, value: F) -> F {
        let mont = (value * F::from_u64(MONT_R)).as_canonical_u64();
        let mut transformed = 0u64;
        for byte in 0..BYTES {
            let digit = ((mont >> (8 * byte)) & 0xff) as usize;
            transformed |= u64::from(self.params.lut[digit]) << (8 * byte);
        }
        F::from_u64(transformed) * F::from_u64(MONT_R).inverse()
    }
}

impl<F: PrimeField64, const WIDTH: usize> NativePermutation<F> for Tip5<'_, F, WIDTH> {
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

impl<F: PrimeField64, const WIDTH: usize> Permutation<[F; WIDTH]> for Tip5<'_, F, WIDTH> {
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        for round in 0..ROUNDS {
            for (i, word) in state.iter_mut().enumerate() {
                *word = if i < SPLIT_WORDS {
                    self.split_sbox(*word)
                } else {
                    power_map::<F, 7>(*word)
                };
            }
            mds_multiply(state, &self.params.m);
            add_round_constants(state, &self.params.rcons[round]);
        }
    }
}
