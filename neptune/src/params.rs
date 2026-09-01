//! Parameters, from `../ref` and only from `../ref`.
//!
//! POLICY §3. Goldilocks and Mersenne-31 match the reference **exactly** —
//! rounds, constants, matrix, byte-exact against its KATs.
//!
//! This file has no field-specific constructors: [`NeptuneParams::derive`] is
//! generic over `F`, and `neptune/src/instances.rs` is where a field is chosen
//! for a registered instance. That file registers Goldilocks only. The six
//! 31-bit points (Mersenne-31, BabyBear, KoalaBear at t = 16, 24) are absent
//! (`Absence::UndefinedForField`), not generated: their round count would have
//! been the construction's Goldilocks value copied straight across (t = 8 to
//! t = 16, t = 12 to t = 24) with no independent derivation behind it at
//! all — no reference criterion to run, unlike Griffin, whose 31-bit points at
//! least mechanically execute the paper's own bound. A copied number with
//! nothing computed for the new prime is a stand-in a reader could mistake for
//! an analyzed instance, and this project no longer registers it as one; see
//! `instances.rs` for the grid entries and [the round-number table]
//! (../../round_numbers_overview.md), whose 31-bit Neptune row is blank.
//!
//! Never hand-write a constant; never re-derive outside the reference.
//! Extending `../ref` is an edit to shared property: minimal, and each one
//! reported.

use p3_field::PrimeField64;
use reference::sampler::{Shake128BitmaskSampler, field_order_seed};

/// Fully derived Neptune constants.  The construction derives exactly the
/// `Neptune || p.to_le_bytes(8)` SHAKE128 stream the reference uses: round
/// constants, then the split external matrix, then the internal diagonal, then
/// the Lai--Massey gamma.
#[derive(Clone, Debug)]
pub struct NeptuneParams<F: PrimeField64, const WIDTH: usize, const EXT: usize, const INT: usize> {
    /// `R + 1` rows: a leading zero row and the final output-whitening row.
    pub rcons: Vec<[F; WIDTH]>,
    /// Neptune's even/odd split external matrix.
    pub m_ext: [[F; WIDTH]; WIDTH],
    /// The `mu - 1` diagonal of `J + diag(mu - 1)`.
    ///
    /// This is precisely the representation consumed by Plonky3's
    /// `p3_poseidon2::matmul_internal`, so internal rounds stay linear in the
    /// state width instead of materializing a dense matrix product.
    pub m_int_diag_m_1: [F; WIDTH],
    /// The Lai--Massey external S-box's fixed non-zero translation.
    pub gamma: F,
}

impl<F: PrimeField64, const WIDTH: usize, const EXT: usize, const INT: usize>
    NeptuneParams<F, WIDTH, EXT, INT>
{
    /// Derive one reference instance. `WIDTH` is even; Neptune's reference
    /// specifies the two fixed t=8 circulants and samples all other widths.
    #[must_use]
    pub fn derive() -> Self {
        assert!(WIDTH.is_multiple_of(2), "Neptune width must be even");
        // `NeptuneParams._init_sampler`: `b"Neptune" || p`, field bytes rounded
        // up to a u64 word.
        let mut sampler = Shake128BitmaskSampler::<F>::new(&field_order_seed::<F>(b"Neptune"));
        let rounds = EXT + INT;

        let mut rcons = Vec::with_capacity(rounds + 1);
        rcons.push([F::ZERO; WIDTH]);
        rcons.extend((0..rounds).map(|_| core::array::from_fn(|_| sampler.next_element())));

        let half = WIDTH / 2;
        let fixed_p = [3u64, 2, 1, 1];
        let fixed_pp = [1u64, 1, 2, 3];
        let mp: Vec<Vec<F>> = if WIDTH == 8 {
            (0..half)
                .map(|row| {
                    (0..half)
                        .map(|col| F::from_u64(fixed_p[(col + half - row) % half]))
                        .collect()
                })
                .collect()
        } else {
            // Unlike t=8, the reference samples the *whole* half-size matrix,
            // not a circulant row.  Draw order is therefore half² M' cells,
            // followed by half² M'' cells.
            (0..half)
                .map(|_| (0..half).map(|_| sampler.next_element()).collect())
                .collect()
        };
        let mpp: Vec<Vec<F>> = if WIDTH == 8 {
            (0..half)
                .map(|row| {
                    (0..half)
                        .map(|col| F::from_u64(fixed_pp[(col + half - row) % half]))
                        .collect()
                })
                .collect()
        } else {
            (0..half)
                .map(|_| (0..half).map(|_| sampler.next_element()).collect())
                .collect()
        };
        let m_ext = core::array::from_fn(|row| {
            core::array::from_fn(|col| {
                if row % 2 != col % 2 {
                    F::ZERO
                } else {
                    let r = row / 2;
                    let c = col / 2;
                    if row % 2 == 0 { mp[r][c] } else { mpp[r][c] }
                }
            })
        });

        let m_int_diag_m_1 = core::array::from_fn(|_| sampler.next_nonzero() - F::ONE);
        let gamma = sampler.next_nonzero();

        Self {
            rcons,
            m_ext,
            m_int_diag_m_1,
            gamma,
        }
    }
}
