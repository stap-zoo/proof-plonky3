//! The native permutation: the oracle everything else is checked against.
//!
//! KAT-green **before any AIR exists** (POLICY §2, step 3). Byte-exact against
//! `../gnark-hashes/gmimc2_ref.py`, which is this construction's oracle and the
//! repository's second trust boundary (POLICY §4, `params.rs`). Never generate a
//! vector from this file or from anything downstream of it — including from an
//! AIR, since a trace is not a vector.
//!
//! # The round, and the three things that are not GMiMC's
//!
//! ```text
//! x      = M_IO . x                   once, before the rounds
//! ---- R times ----
//! x_0   += rc_r                       the constant lands IN the state
//! y      = x_0^alpha                  alpha = 2^k, not a permutation
//! x_i   += y                          for every i >= 1
//! x      = shift(x)                   out[i] = x[i + 1], out[t-1] = x[0]
//! -----------------
//! x      = M_IO . x                   once, after
//! ```
//!
//! Everything between the brackets is GMiMC-erf except two lines, and both are
//! deliberate:
//!
//! 1. **The constant is added into branch 0 and stays there.** GMiMC computes
//!    `y = (x_0 + rc_r)^alpha` and leaves `x_0` alone; here `x_0` carries `rc_r`
//!    onward for the whole of its trip around the state. The specification fixes
//!    this reading twice over — the round figure taps the wire feeding the S-box
//!    *below* the constant's node, and its efficient circuit never removes a
//!    constant a branch picked up — and the oracle's `permutation_efficient`,
//!    which reproduces this round function and no other, is checked against
//!    `permutation` at all four points by the export's self-test.
//! 2. **`alpha = 2^k`**, so the S-box is two-to-one. See `params.rs` for why an
//!    erf does not care.
//!
//! and the third is `M_IO` bracketing the loop, where GMiMC's `_pre_rounds` and
//! `_post_rounds` are both the identity.
//!
//! # Why this file loops over branches
//!
//! `t - 1` additions a round, written out — the shape of the oracle's
//! `permutation`, which is the *definition*. The oracle also carries
//! `permutation_efficient`, a sliding-window form that keeps one live branch and
//! an accumulator; it is much closer to what the arithmetization does, and it is
//! deliberately **not** what this file reproduces. Written that way, the
//! recurrence the AIR rests on would be checked against itself. Written this
//! way, the known-answer vectors are what say it is right.
//!
//! Nothing here is on a measured path: the oracle runs once per vector.

use harness::gadgets::power_map::power_map;
use harness::permutation::NativePermutation;
use p3_field::PrimeField64;

use crate::params::{GMiMC2Params, m_io, shift_source};

/// Native GMiMC2 permutation, borrowing an instance's parameters.
///
/// The shape is carried in const parameters rather than as run-time values so
/// that `ALPHA` can select the [`power_map`] arm — the same gadget, and the same
/// arm, the AIR will evaluate. Degree 4 is the arm GMiMC2 added to that gadget:
/// no other construction here has a non-bijective S-box.
#[derive(Clone, Debug)]
pub struct GMiMC2<'a, F, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64> {
    /// The instance's parameters.
    pub params: &'a GMiMC2Params<F, WIDTH, ROUNDS>,
}

impl<'a, F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    GMiMC2<'a, F, WIDTH, ROUNDS, ALPHA>
{
    /// Borrow an instance's parameters as the oracle.
    ///
    /// # Panics
    ///
    /// If `ALPHA` disagrees with the parameters' own exponent — the one way the
    /// const parameter and the seeded value could drift apart. It matters more
    /// here than elsewhere: `alpha` is in the constant seed, so a mismatch means
    /// the constants belong to a different instance than the S-box does.
    #[must_use]
    pub fn new(params: &'a GMiMC2Params<F, WIDTH, ROUNDS>) -> Self {
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

    /// The oracle's `nonlinear_layer`: the constant into branch 0, then its
    /// power into every other branch.
    ///
    /// The order of the first two lines is the design. Swapping them gives
    /// `../ref`'s GMiMC round, a permutation that passes every structural test
    /// and no vector.
    fn nonlinear_layer(&self, state: &mut [F; WIDTH], round: usize) {
        state[0] += self.params.rcons[round];
        let y = power_map::<F, ALPHA>(state[0]);
        for word in state.iter_mut().skip(1) {
            *word += y;
        }
    }

    /// The oracle's `linear_layer`: `matvecmul(M, state)` with `M` the
    /// cyclic-shift permutation matrix. See [`shift_source`].
    fn linear_layer(&self, state: &mut [F; WIDTH]) {
        let shifted = core::array::from_fn(|i| state[shift_source(i, WIDTH)]);
        *state = shifted;
    }
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize, const ALPHA: u64>
    NativePermutation<F> for GMiMC2<'_, F, WIDTH, ROUNDS, ALPHA>
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
        let input: [F; WIDTH] = core::array::from_fn(|i| state[i]);
        let mut work = m_io(&input);
        for round in 0..ROUNDS {
            self.nonlinear_layer(&mut work, round);
            self.linear_layer(&mut work);
        }
        state.copy_from_slice(&m_io(&work));
    }
}
