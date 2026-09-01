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
//! # Why this file is written the way it is
//!
//! Layer by layer, over `Vec`, indexed at run time — the shape of `hash.py`,
//! not the shape of `air.rs`. That is deliberate and it is the only reason this
//! file earns the word *oracle*.
//!
//! `air.rs` and `generation.rs` both evaluate a **fused** round: they never
//! materialize the intermediate state between the nonlinear layer and the matrix,
//! because the matrix is a permutation-plus-a-few-additions and folding it into
//! the Feistel's write is what makes a round cost `t/2` cells instead of `t`.
//! That fusion is an algebraic claim about `_init_mat`. If the oracle were
//! written the same way, the claim would be untested — the fused round would be
//! checked against itself. Written this way, `M` is applied as a matrix and the
//! KAT is what says the fusion is right.
//!
//! Nothing here is on a measured path: the oracle runs once per known-answer
//! vector.

use harness::permutation::NativePermutation;
use p3_field::Field;

use crate::params::{PSquareHashParams, assert_shape};

/// The pSquareHash permutation, computed natively.
///
/// Owns the shape as run-time values rather than const parameters: a KAT runner
/// walks a heterogeneous list of `dyn NativePermutation`, and the oracle has no
/// layout to keep aligned with anything.
#[derive(Debug, Clone)]
pub struct PSquareHash<F> {
    name: &'static str,
    /// State size `t`.
    width: usize,
    /// `R` rows of `t / 2` round constants, as the reference holds them.
    rcons: Vec<Vec<F>>,
}

impl<F: Field> PSquareHash<F> {
    /// Build the oracle for an instance.
    ///
    /// # Panics
    ///
    /// If `WIDTH != 4 * PAIRS` (see [`assert_shape`]).
    #[must_use]
    pub fn new<const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>(
        params: &PSquareHashParams<F, WIDTH, PAIRS, ROUNDS>,
    ) -> Self {
        assert_shape(WIDTH, PAIRS);
        Self {
            name: params.name,
            width: WIDTH,
            rcons: params
                .rcons
                .iter()
                .map(|row| row.iter().flatten().copied().collect())
                .collect(),
        }
    }

    /// Rounds `R`.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.rcons.len()
    }

    /// `hash.py`'s `feistel`: two field elements and two round constants in, two
    /// out, at a cost of two squarings.
    ///
    /// ```text
    /// y1 = x1 + c0
    /// y2 = x0 + y1^2
    /// y3 = y1 + y2 + c1
    /// y4 = y2 + y3^2
    /// y5 = y3 + y4
    /// ```
    ///
    /// The two squarings are the *entire* nonlinearity of the design, and `y4`,
    /// `y5` are what the round adds into the upper half. Note that `y5` is
    /// affine in `y3` and `y4` — a fact `air.rs` spends nothing to exploit and
    /// which is why flattening a Feistel costs two cells, not four.
    #[must_use]
    fn feistel(x0: F, x1: F, c0: F, c1: F) -> [F; 2] {
        let y1 = x1 + c0;
        let y2 = x0 + y1.square();
        let y3 = y1 + y2 + c1;
        let y4 = y2 + y3.square();
        let y5 = y3 + y4;
        [y4, y5]
    }

    /// `hash.py`'s `nonlinear_layer`: a Feistel step, lower half into upper.
    ///
    /// The lower half is read and left alone; each of its `t / 4` pairs is fed
    /// through [`Self::feistel`] and *added into* one upper-half pair, in reverse
    /// order — input pair `i` writes output pair `t - i - 2`. Round constants
    /// mirror the same reversal, `rcons[r][t/2 - (i+2) .. t/2 - i]`.
    fn nonlinear_layer(&self, state: &[F], round: usize) -> Vec<F> {
        let (t, h) = (self.width, self.width / 2);
        let rc = &self.rcons[round];
        let mut out = state.to_vec();
        for i in (0..h).step_by(2) {
            let y = Self::feistel(state[i], state[i + 1], rc[h - i - 2], rc[h - i - 1]);
            out[t - i - 2] += y[0];
            out[t - i - 1] += y[1];
        }
        out
    }

    /// `hash.py`'s `linear_layer`, which is `params.py`'s `_init_mat` read as an
    /// addition program.
    ///
    /// `M` swaps the halves, then adds two things back into the new lower half:
    ///
    /// * `z = (2·x[h-2] + x[h-1], x[h-2] + x[h-1])` into every lower-half pair
    ///   from index 2 up — *not* into pair 0;
    /// * the sum of the interior lower-half pairs into pair `h - 2`.
    ///
    /// The half swap is the Feistel's swap; everything else is what stops the
    /// two halves evolving independently.
    fn linear_layer(&self, x: &[F]) -> Vec<F> {
        let (t, h) = (self.width, self.width / 2);
        let mut out = vec![F::ZERO; t];

        // mat[i][h+i] += 1, mat[h+i][i] += 1 — the swap.
        for i in 0..h {
            out[i] += x[h + i];
            out[h + i] += x[i];
        }
        // mat[h-2][i] += 1, mat[h-1][i+1] += 1 for the interior pairs.
        for i in (2..h.saturating_sub(2)).step_by(2) {
            out[h - 2] += x[i];
            out[h - 1] += x[i + 1];
        }
        // mat[i][h-2] += 2, mat[i][h-1] += 1, mat[i+1][h-2] += 1, mat[i+1][h-1] += 1.
        for i in (2..h).step_by(2) {
            out[i] += x[h - 2].double() + x[h - 1];
            out[i + 1] += x[h - 2] + x[h - 1];
        }
        out
    }

    /// `hash.py`'s `_pre_rounds` and `_post_rounds`, which are both `M_IO`
    /// (`_init_mat_IO`): `out[i] = x[i] + x[h+i]`, `out[h+i] = 2·x[i] + x[h+i]`.
    ///
    /// The same matrix at both ends, and the reason it is there is that a bare
    /// Feistel round leaves the lower half untouched — without `M_IO` the first
    /// round's output would expose `t / 2` input elements verbatim.
    fn m_io(&self, x: &[F]) -> Vec<F> {
        let h = self.width / 2;
        let mut out = vec![F::ZERO; self.width];
        for i in 0..h {
            out[i] = x[i] + x[h + i];
            out[h + i] = x[i].double() + x[h + i];
        }
        out
    }
}

impl<F: Field> NativePermutation<F> for PSquareHash<F> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    fn permute(&self, state: &mut [F]) {
        assert_eq!(state.len(), self.width, "{}: wrong state size", self.name);

        let mut s = self.m_io(state);
        for round in 0..self.rounds() {
            s = self.nonlinear_layer(&s, round);
            s = self.linear_layer(&s);
        }
        state.copy_from_slice(&self.m_io(&s));
    }
}
