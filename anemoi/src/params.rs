//! Reference-derived Anemoi parameters.
//!
//! The two Goldilocks instances are the exact `../ref` instances. The three
//! 31-bit `t=16` instances are produced by the reference's own derivations with
//! `R=11`, copied from Goldilocks `t=8` as POLICY §3 requires. The reference
//! cannot derive `l=12`, so the three 31-bit `t=24` points are absent.

use p3_field::PrimeField64;
use reference::sampler::inverse_exponent;

const PI_0: &str = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679";
const PI_1: &str = "8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196";

/// Goldilocks `t=8` and every generated `t=16` instance use eleven rounds.
pub const ROUNDS_T8_T16: usize = 11;
/// Goldilocks `t=12` uses ten rounds.
pub const ROUNDS_T12: usize = 10;

/// A fully specified Anemoi instance.
#[derive(Clone, Debug)]
pub struct AnemoiParams<F, const WIDTH: usize, const COLUMNS: usize, const ROUNDS: usize> {
    /// Reference/export name.
    pub name: &'static str,
    /// Forward Flystel exponent.
    pub alpha: u64,
    /// Inverse exponent modulo `p - 1`, used only by the native/open Flystel.
    pub alpha_inv: u64,
    /// `Q(z) = beta * z^2`; the reference sets `gamma = 0`.
    pub beta: F,
    /// The reference's `gamma`, currently zero for every prime-field instance.
    pub gamma: F,
    /// Translation in the second quadratic, `g^-1` in the reference.
    pub delta: F,
    /// Linear map on the x lane.
    pub m_x: [[F; COLUMNS]; COLUMNS],
    /// Linear map on the rotated y lane (`M_x * P_rho`).
    pub m_y: [[F; COLUMNS]; COLUMNS],
    /// x-lane round constants.
    pub c: [[F; COLUMNS]; ROUNDS],
    /// y-lane round constants.
    pub d: [[F; COLUMNS]; ROUNDS],
}

/// The layout invariant shared by parameters, columns, AIR, and generation.
pub const fn assert_shape(width: usize, columns: usize) {
    assert!(width == 2 * columns, "Anemoi WIDTH must equal 2 * COLUMNS");
    assert!(columns > 0, "Anemoi needs at least one Flystel column");
}

fn decimal_mod<F: PrimeField64>(digits: &str) -> F {
    let modulus = u128::from(F::ORDER_U64);
    let value = digits.bytes().fold(0u128, |acc, digit| {
        (acc * 10 + u128::from(digit - b'0')) % modulus
    });
    F::from_u64(value as u64)
}

impl<F: PrimeField64, const WIDTH: usize, const COLUMNS: usize, const ROUNDS: usize>
    AnemoiParams<F, WIDTH, COLUMNS, ROUNDS>
{
    fn derive(
        name: &'static str,
        alpha: u64,
        generator: u64,
        first_circulant_row: Option<&[u64]>,
    ) -> Self {
        assert_shape(WIDTH, COLUMNS);
        assert!(matches!(alpha, 3 | 5 | 7), "supported small-prime alpha");
        let beta = F::from_u64(generator);
        let gamma = F::ZERO;
        let delta = beta.inverse();
        let alpha_inv = inverse_exponent(alpha, F::ORDER_U64 - 1);

        let m_x = if COLUMNS == 4 {
            // `dl_m46_83_matrix(g)`. The Goldilocks instance accepts the first
            // generator power tested by the reference's MDS search.
            let a = beta;
            let a2 = a.square();
            let one = F::ONE;
            let rows = [
                [one, one + a, a, a],
                [a2, a2 + a, one + a, one + a.double()],
                [a2, a2, one, one + a],
                [one + a, one + a.double(), a, one + a],
            ];
            core::array::from_fn(|i| core::array::from_fn(|j| rows[i][j]))
        } else {
            let row = first_circulant_row.expect("large Anemoi matrices are circulant");
            assert_eq!(row.len(), COLUMNS);
            core::array::from_fn(|i| {
                core::array::from_fn(|j| F::from_u64(row[(j + COLUMNS - i) % COLUMNS]))
            })
        };
        // `_init_My_from_Mx`: rotate each Mx row right once.
        let m_y =
            core::array::from_fn(|i| core::array::from_fn(|j| m_x[i][(j + COLUMNS - 1) % COLUMNS]));

        let pi_0 = decimal_mod::<F>(PI_0);
        let pi_1 = decimal_mod::<F>(PI_1);
        let c = core::array::from_fn(|round| {
            let pi_0_r = pi_0.exp_u64(round as u64);
            core::array::from_fn(|i| {
                let pi_1_i = pi_1.exp_u64(i as u64);
                beta * pi_0_r.square() + (pi_0_r + pi_1_i).exp_u64(alpha)
            })
        });
        let d = core::array::from_fn(|round| {
            let pi_0_r = pi_0.exp_u64(round as u64);
            core::array::from_fn(|i| {
                let pi_1_i = pi_1.exp_u64(i as u64);
                beta * pi_1_i.square() + (pi_0_r + pi_1_i).exp_u64(alpha) + delta
            })
        });

        Self {
            name,
            alpha,
            alpha_inv,
            beta,
            gamma,
            delta,
            m_x,
            m_y,
            c,
            d,
        }
    }
}

/// Exact reference instance `ANEMOI_GOLDILOCKS_T8`.
#[must_use]
pub fn goldilocks_t8<F: PrimeField64>() -> AnemoiParams<F, 8, 4, ROUNDS_T8_T16> {
    AnemoiParams::derive("anemoi-goldilocks-t8", 7, 7, None)
}

/// Exact extra-width reference instance `ANEMOI_GOLDILOCKS_T10`.
#[must_use]
pub fn goldilocks_t10<F: PrimeField64>() -> AnemoiParams<F, 10, 5, ROUNDS_T8_T16> {
    AnemoiParams::derive("anemoi-goldilocks-t10", 7, 7, Some(&[1, 1, 3, 4, 5]))
}

/// Exact reference instance `ANEMOI_GOLDILOCKS_T12`.
#[must_use]
pub fn goldilocks_t12<F: PrimeField64>() -> AnemoiParams<F, 12, 6, ROUNDS_T12> {
    AnemoiParams::derive("anemoi-goldilocks-t12", 7, 7, Some(&[1, 1, 3, 4, 5, 6]))
}

/// Generated `ANEMOI_MERSENNE_T16`; round count provisional (POLICY §3).
#[must_use]
pub fn mersenne_t16<F: PrimeField64>() -> AnemoiParams<F, 16, 8, ROUNDS_T8_T16> {
    AnemoiParams::derive("anemoi-mersenne-t16", 5, 7, Some(&[1, 2, 3, 5, 7, 8, 8, 9]))
}

/// Generated `ANEMOI_BABYBEAR_T16`; round count provisional (POLICY §3).
#[must_use]
pub fn babybear_t16<F: PrimeField64>() -> AnemoiParams<F, 16, 8, ROUNDS_T8_T16> {
    AnemoiParams::derive(
        "anemoi-babybear-t16",
        7,
        31,
        Some(&[1, 2, 3, 5, 7, 8, 8, 9]),
    )
}

/// Generated `ANEMOI_KOALABEAR_T16`; round count provisional (POLICY §3).
#[must_use]
pub fn koalabear_t16<F: PrimeField64>() -> AnemoiParams<F, 16, 8, ROUNDS_T8_T16> {
    AnemoiParams::derive(
        "anemoi-koalabear-t16",
        3,
        3,
        Some(&[1, 2, 3, 5, 7, 8, 8, 9]),
    )
}
