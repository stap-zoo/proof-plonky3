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
//! # The round
//!
//! ```text
//! y      = (x_0 + rc_r)^alpha         one power map, on branch 0 alone
//! x_i   += y                          for every i >= 1
//! x      = shift(x)                   out[i] = x[i + 1], out[t-1] = x[0]
//! ```
//!
//! Branch 0 keeps its own value — the constant is an *input to the S-box*, not
//! something added into the state — and then the shift moves it to the end. That
//! is the one thing GMiMC2 changes, and it is why these are two crates: there
//! `x_0 += rc_r` first, so the constant travels with the branch.
//!
//! `_pre_rounds` and `_post_rounds` are both the identity in `hash.py`, so the
//! permutation is the round loop and nothing else. GMiMC2 brackets the same loop
//! with `M_IO`.
//!
//! # Why this file loops over branches
//!
//! `t - 1` additions a round, written out, over a `Vec` indexed at run time —
//! the shape of `hash.py`, not the shape of `air.rs`. That is deliberate and it
//! is the only reason this file earns the word *oracle*.
//!
//! The arithmetization does something quite different: it never materializes a
//! state at all, because only branch 0 is ever read nonlinearly. It commits the
//! value entering each S-box and reaches back exactly `t` rounds for it, which
//! is an algebraic claim about *this* loop. Written the same way here, that
//! claim would be checked against itself; written this way, the known-answer
//! vectors are what say the recurrence is right.
//!
//! Nothing here is on a measured path: the oracle runs once per vector.

use harness::gadgets::power_map::power_map;
use harness::permutation::NativePermutation;
use p3_field::PrimeField64;

use crate::params::{GMiMCParams, shift_source};

/// Native GMiMC-erf permutation, borrowing an instance's parameters.
///
/// The shape is carried in const parameters rather than as run-time values so
/// that `ALPHA` can select the [`power_map`] arm — the same gadget, and the same
/// arm, the AIR will evaluate.
#[derive(Clone, Debug)]
pub struct GMiMC<'a, F, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64> {
    /// Reference-derived parameters.
    pub params: &'a GMiMCParams<F, WIDTH, ROUNDS>,
}

impl<'a, F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    GMiMC<'a, F, WIDTH, ROUNDS, ALPHA>
{
    /// Borrow reference-derived parameters as the oracle.
    ///
    /// # Panics
    ///
    /// If `ALPHA` disagrees with the parameters' own exponent — the one way the
    /// const parameter and the checked value could drift apart.
    #[must_use]
    pub fn new(params: &'a GMiMCParams<F, WIDTH, ROUNDS>) -> Self {
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

    /// `hash.py`'s `nonlinear_layer`: power branch 0 with its round constant and
    /// add the result into every other branch.
    ///
    /// Branch 0 is returned unchanged. `rcons[r]` is an argument to the power
    /// map and never lands in the state.
    fn nonlinear_layer(&self, state: &mut [F; WIDTH], round: usize) {
        let y = power_map::<F, ALPHA>(state[0] + self.params.rcons[round]);
        for word in state.iter_mut().skip(1) {
            *word += y;
        }
    }

    /// `hash.py`'s `linear_layer`, which is `_init_mat` read as a re-indexing:
    /// `matvecmul(M, state)` with `M` the cyclic-shift permutation matrix.
    ///
    /// See [`shift_source`] for the direction and for why no matrix is stored.
    fn linear_layer(&self, state: &mut [F; WIDTH]) {
        let shifted = core::array::from_fn(|i| state[shift_source(i, WIDTH)]);
        *state = shifted;
    }
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    NativePermutation<F> for GMiMC<'_, F, WIDTH, ROUNDS, ALPHA>
{
    fn name(&self) -> &'static str {
        self.params.name
    }

    fn width(&self) -> usize {
        WIDTH
    }

    fn permute(&self, state: &mut [F]) {
        assert_eq!(
            state.len(),
            WIDTH,
            "{}: one call is one WIDTH-element state",
            self.params.name
        );
        let mut work: [F; WIDTH] = core::array::from_fn(|i| state[i]);
        for round in 0..ROUNDS {
            self.nonlinear_layer(&mut work, round);
            self.linear_layer(&mut work);
        }
        state.copy_from_slice(&work);
    }
}
