//! Native Anemoi permutation, kept layer-for-layer parallel with `../ref`.
//!
//! This oracle deliberately evaluates the open Flystel with the inverse power;
//! the AIR uses the closed verification form. Their only shared code is the
//! parameter object, so the KAT catches a wrong closed-form arithmetization.

use harness::permutation::NativePermutation;
use p3_field::{Algebra, Field, PrimeCharacteristicRing};
use p3_mds::util::mds_multiply;

use crate::params::{AnemoiParams, assert_shape};

/// Anemoi's linear layer: `M_x` on the first half, `M_y` on the second, then the
/// pseudo-Hadamard transform that couples them.
///
/// The two matrix products are ordinary dense multiplies and go through
/// upstream's shared path; the PHT — `y += x` then `x += y`, in that order — is
/// the part that is Anemoi's own. Reversing those two lines is still an
/// invertible map and still passes every structural test, so the order is
/// load-bearing and the known-answer vectors are what check it.
///
/// Generic in the algebra, so `air.rs` over `AB::Expr` and `generation.rs` over
/// `F` and `F::Packing` run one implementation rather than two identical ones.
/// The oracle below keeps its own run-time-shaped copy: it walks `Vec`s because a
/// KAT runner holds a heterogeneous list of `dyn NativePermutation`, so it is a
/// different program rather than a duplicate of this one.
#[inline]
pub fn linear_layer<F, A, const WIDTH: usize, const COLUMNS: usize>(
    state: &mut [A; WIDTH],
    m_x: &[[F; COLUMNS]; COLUMNS],
    m_y: &[[F; COLUMNS]; COLUMNS],
) where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    let mut x: [A; COLUMNS] = core::array::from_fn(|i| state[i].dup());
    let mut y: [A; COLUMNS] = core::array::from_fn(|i| state[COLUMNS + i].dup());
    mds_multiply(&mut x, m_x);
    mds_multiply(&mut y, m_y);
    for i in 0..COLUMNS {
        y[i] += x[i].dup();
        x[i] += y[i].dup();
        state[i] = x[i].dup();
        state[COLUMNS + i] = y[i].dup();
    }
}

/// Native oracle for one reference instance.
#[derive(Clone, Debug)]
pub struct Anemoi<F> {
    name: &'static str,
    width: usize,
    columns: usize,
    alpha_inv: u64,
    beta: F,
    gamma: F,
    delta: F,
    m_x: Vec<Vec<F>>,
    m_y: Vec<Vec<F>>,
    c: Vec<Vec<F>>,
    d: Vec<Vec<F>>,
}

impl<F: Field> Anemoi<F> {
    /// Build the oracle from a fully derived instance.
    #[must_use]
    pub fn new<const WIDTH: usize, const COLUMNS: usize, const ROUNDS: usize>(
        params: &AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>,
    ) -> Self {
        assert_shape(WIDTH, COLUMNS);
        Self {
            name: params.name,
            width: WIDTH,
            columns: COLUMNS,
            alpha_inv: params.alpha_inv,
            beta: params.beta,
            gamma: params.gamma,
            delta: params.delta,
            m_x: params.m_x.iter().map(|row| row.to_vec()).collect(),
            m_y: params.m_y.iter().map(|row| row.to_vec()).collect(),
            c: params.c.iter().map(|row| row.to_vec()).collect(),
            d: params.d.iter().map(|row| row.to_vec()).collect(),
        }
    }

    /// Number of rounds.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.c.len()
    }

    fn matmul(matrix: &[Vec<F>], input: &[F]) -> Vec<F> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(input)
                    .fold(F::ZERO, |acc, (coefficient, value)| {
                        acc + *coefficient * *value
                    })
            })
            .collect()
    }

    /// The oracle's own run-time-shaped linear layer; see the free
    /// [`linear_layer`] above for why this is a second program rather than a
    /// duplicate.
    fn linear_layer(&self, state: &mut [F]) {
        let (x, y) = state.split_at(self.columns);
        let mut x = Self::matmul(&self.m_x, x);
        let mut y = Self::matmul(&self.m_y, y);
        for i in 0..self.columns {
            y[i] += x[i];
            x[i] += y[i];
        }
        state[..self.columns].copy_from_slice(&x);
        state[self.columns..].copy_from_slice(&y);
    }

    fn nonlinear_layer(&self, state: &mut [F]) {
        for i in 0..self.columns {
            let x = state[i];
            let y = state[self.columns + i];
            let mut u = x - (self.beta * y.square() + self.gamma);
            let v = y - u.exp_u64(self.alpha_inv);
            u += self.beta * v.square() + self.delta;
            state[i] = u;
            state[self.columns + i] = v;
        }
    }
}

impl<F: Field> NativePermutation<F> for Anemoi<F> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn width(&self) -> usize {
        self.width
    }

    fn permute(&self, state: &mut [F]) {
        assert_eq!(state.len(), self.width, "{}: wrong state size", self.name);
        for round in 0..self.rounds() {
            for i in 0..self.columns {
                state[i] += self.c[round][i];
                state[self.columns + i] += self.d[round][i];
            }
            self.linear_layer(state);
            self.nonlinear_layer(state);
        }
        self.linear_layer(state);
    }
}
