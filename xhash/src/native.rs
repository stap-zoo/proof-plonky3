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
//! # One schedule, four instances
//!
//! `../ref`'s `XHash` runs the same F/B/P3 schedule for all four exact
//! instances, and so does [`XHash::permute_state`]:
//!
//! ```text
//! step % 3 == 0   F:  add constants, mix, x^alpha on every word
//! step % 3 == 1   B:  mix, add constants, x^(1/alpha) on every active word
//! step % 3 == 2   P3: add constants, then x^alpha triple by triple, in the
//!                     degree-three quotient, via the exported cpolys
//! ```
//!
//! for `STEPS = 3R/2` steps, then a final mix and constant row. The B step is
//! active on every word for `XHash12`/`XHash24` and every word outside
//! `1 mod 3` for the aggressive `XHash8`/`XHash16`; F is always full.
//!
//! **The oracle evaluates the P3 layer the way the reference does** — as three
//! coordinate polynomials, [`extension_power_from_cpolys`] against
//! `_sbox_P3`'s `eval_aos` — and never through the quotient arithmetic the split
//! AIR uses. That is what leaves `params::cpolys_are_the_power_map` something
//! to prove rather than something to assume.

use harness::permutation::{NativePermutation, add_round_constants};
use p3_field::{Algebra, PrimeCharacteristicRing, PrimeField64};
use p3_mds::util::mds_multiply;
use p3_symmetric::Permutation;

use crate::params::{CoordinateTerm, STEPS, XHashParams, extension_mul};

/// Evaluate the exported coordinate polynomials at a point.
///
/// The reference's own P3 operation, and the only one the oracle uses.
#[inline]
pub fn extension_power_from_cpolys<F, A>(
    point: [A; 3],
    cpolys: &[Vec<CoordinateTerm<F>>; 3],
) -> [A; 3]
where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    core::array::from_fn(|coordinate| {
        cpolys[coordinate].iter().fold(A::ZERO, |sum, term| {
            let monomial = point
                .each_ref()
                .into_iter()
                .zip(term.exponents)
                .fold(A::ONE, |product, (value, exponent)| {
                    product * small_power(value.dup(), exponent)
                });
            sum + monomial * term.coefficient.dup()
        })
    })
}

#[inline]
fn small_power<R: PrimeCharacteristicRing>(value: R, exponent: u8) -> R {
    match exponent {
        0 => R::ONE,
        1 => value,
        2 => value.square(),
        3 => value.cube(),
        4 => value.square().square(),
        5 => value.square().square() * value,
        6 => value.cube().square(),
        7 => value.cube().square() * value,
        _ => panic!("XHash coordinate exponent exceeds alpha"),
    }
}

/// `x^{(alpha - 1) / 2}` in the quotient — the value the split AIR commits.
///
/// One committed triple, not `alpha`: `x^alpha = (x^{(alpha-1)/2})^2 * x` and
/// the quotient's multiplication is bilinear, so the whole P3 layer becomes
/// degree three in committed cells. `alpha = 7` commits the cube and `alpha = 5`
/// the square; there is no third case in the grid.
#[inline]
pub fn extension_half_power<F, A, const ALPHA: u64>(
    point: [A; 3],
    reduction: &[F; 3],
    x4: &[F; 3],
) -> [A; 3]
where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    let square = extension_mul(
        point.each_ref().map(|value| value.dup()),
        point.each_ref().map(|value| value.dup()),
        reduction,
        x4,
    );
    match ALPHA {
        5 => square,
        7 => extension_mul(square, point, reduction, x4),
        _ => panic!("XHash supports the fifth and seventh extension powers"),
    }
}

/// `x^alpha` from a supplied `x^{(alpha - 1) / 2}`.
///
/// Degree three in committed cells when `half` is committed — but only sound
/// once `half` is pinned to `point`. [`crate::air`] owns that assertion.
#[inline]
pub fn extension_power_from_half<F, A>(
    point: [A; 3],
    half: [A; 3],
    reduction: &[F; 3],
    x4: &[F; 3],
) -> [A; 3]
where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    let square = extension_mul(
        half.each_ref().map(|value| value.dup()),
        half,
        reduction,
        x4,
    );
    extension_mul(square, point, reduction, x4)
}

/// The six quadratic monomials `(x0², x0x1, x0x2, x1², x1x2, x2²)`.
///
/// The modulus-free P3 register basis: every degree-five monomial factors as
/// two of these times one input coordinate, whatever the coefficients are. It
/// costs six cells per triple where [`extension_half_power`] costs three, and it
/// is what an instance whose exported table is *not* a power map has to use.
#[inline]
pub fn extension_quadratics<R: PrimeCharacteristicRing>(point: [R; 3]) -> [R; 6] {
    let [x0, x1, x2] = point;
    [
        x0.dup().square(),
        x0.dup() * x1.dup(),
        x0 * x2.dup(),
        x1.dup().square(),
        x1 * x2.dup(),
        x2.square(),
    ]
}

const QUADRATIC_EXPONENTS: [[u8; 3]; 6] = [
    [2, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2],
];

fn degree_five_factorization(exponents: [u8; 3]) -> (usize, usize, usize) {
    for (first, first_exponents) in QUADRATIC_EXPONENTS.iter().enumerate() {
        for (second, second_exponents) in QUADRATIC_EXPONENTS.iter().enumerate().skip(first) {
            for variable in 0..3 {
                let mut sum = [0; 3];
                for coordinate in 0..3 {
                    sum[coordinate] = first_exponents[coordinate]
                        + second_exponents[coordinate]
                        + u8::from(coordinate == variable);
                }
                if sum == exponents {
                    return (first, second, variable);
                }
            }
        }
    }
    panic!("XHash term is not homogeneous of degree five");
}

/// Evaluate the coordinate polynomials from committed quadratic monomials.
///
/// Degree three, as [`extension_power_from_half`] is, at twice the cells. The
/// caller must pin all six registers to the input first; [`crate::air`] does.
#[inline]
pub fn extension_power_from_quadratics<F, A>(
    point: [A; 3],
    quadratics: [A; 6],
    cpolys: &[Vec<CoordinateTerm<F>>; 3],
) -> [A; 3]
where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    core::array::from_fn(|coordinate| {
        cpolys[coordinate].iter().fold(A::ZERO, |sum, term| {
            let (first, second, variable) = degree_five_factorization(term.exponents);
            sum + quadratics[first].dup()
                * quadratics[second].dup()
                * point[variable].dup()
                * term.coefficient.dup()
        })
    })
}

/// Native XHash, parameterized by one exact reference instance.
#[derive(Clone, Debug)]
pub struct XHash<F, const WIDTH: usize, const CONSTANT_ROWS: usize> {
    /// Reference-derived parameters.
    pub params: XHashParams<F, WIDTH, CONSTANT_ROWS>,
}

impl<F: PrimeField64, const WIDTH: usize, const CONSTANT_ROWS: usize>
    XHash<F, WIDTH, CONSTANT_ROWS>
{
    /// Build the native oracle.
    #[must_use]
    pub fn new(params: &XHashParams<F, WIDTH, CONSTANT_ROWS>) -> Self {
        assert!(
            matches!(params.alpha, 5 | 7),
            "XHash uses the fifth or seventh power"
        );
        assert!(
            CONSTANT_ROWS > STEPS,
            "XHash needs nine scheduled rows and one final row"
        );
        assert!(WIDTH.is_multiple_of(3), "the P3 layer consumes triples");
        Self {
            params: params.clone(),
        }
    }

    /// Apply the reference's F/B/P3 schedule over any `F`-algebra.
    pub fn permute_state<A: Algebra<F>>(&self, state: &mut [A; WIDTH]) {
        for step in 0..STEPS {
            match step % 3 {
                0 => {
                    add_round_constants(state, &self.params.rcons[step]);
                    mds_multiply(state, &self.params.m);
                    for word in state.iter_mut() {
                        *word = word.dup().exp_u64(self.params.alpha);
                    }
                }
                1 => {
                    mds_multiply(state, &self.params.m);
                    for (i, (word, constant)) in
                        state.iter_mut().zip(self.params.rcons[step]).enumerate()
                    {
                        *word += constant;
                        if !(self.params.skip_middle && i % 3 == 1) {
                            *word = word.dup().exp_u64(self.params.alpha_inv);
                        }
                    }
                }
                2 => {
                    add_round_constants(state, &self.params.rcons[step]);
                    for chunk in state.chunks_exact_mut(3) {
                        let output = extension_power_from_cpolys(
                            [chunk[0].dup(), chunk[1].dup(), chunk[2].dup()],
                            &self.params.cpolys,
                        );
                        chunk.clone_from_slice(&output);
                    }
                }
                _ => unreachable!(),
            }
        }
        mds_multiply(state, &self.params.m);
        add_round_constants(state, &self.params.rcons[CONSTANT_ROWS - 1]);
    }
}

impl<F: PrimeField64, const WIDTH: usize, const CONSTANT_ROWS: usize> NativePermutation<F>
    for XHash<F, WIDTH, CONSTANT_ROWS>
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
        self.permute_state(&mut work);
        state.copy_from_slice(&work);
    }
}

impl<F: PrimeField64, const WIDTH: usize, const CONSTANT_ROWS: usize> Permutation<[F; WIDTH]>
    for XHash<F, WIDTH, CONSTANT_ROWS>
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        self.permute_state(state);
    }
}
