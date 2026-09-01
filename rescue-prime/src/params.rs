//! Parameters, from `../ref` and only from `../ref`.
//!
//! POLICY §3. Both Goldilocks widths are instances the reference itself pins;
//! the six 31-bit points are constructed at the `export_small_prime_kat.py` call
//! site, which runs `RescuePrimeParams`' own `_init_rounds` / `_init_cons` /
//! `_init_mat` for the new prime. What lands here is a Rust reimplementation of
//! the *same* derivation, locked against the reference's export in
//! `tests/reference.rs`. Never hand-write a constant; never re-derive outside
//! the reference.
//!
//! # Why re-derive at all, when the native side is upstream's
//!
//! Because the AIR needs every parameter as a *value* it can evaluate, and
//! `p3-rescue` hands over none of them: `Rescue`'s fields are private, and its
//! `MdsMatrix*` types are fast transforms with no entries to read. So the
//! parameters are derived here and passed *into* upstream's permutation
//! (`native.rs`), which is what lets one KAT validate both sides (POLICY §4).
//!
//! Two of the three derivations coincide with upstream's own, which is what
//! makes this a wrap rather than a re-port — checked in `tests/reference.rs`,
//! not assumed:
//!
//! * **round constants.** Upstream's `get_round_constants_rescue_prime` seeds
//!   SHAKE-256 with `"Rescue-XLIX(p,t,c,kappa)"` and reads `ceil(bits/8)+1`
//!   bytes little-endian mod `p`. That is exactly the reference's
//!   `_init_cons` with `XOFFieldElementSampler(sampling="mod")`, whose docstring
//!   names the Marvellous script as what it matches.
//! * **round count.** Upstream's `Rescue::num_rounds` and the reference's
//!   `_init_rounds` agree at every grid point.
//!
//! # The reference derives 8 everywhere, and that is a finding
//!
//! `_init_rounds` is `ceil(1.5 * max(5, l0, l1))`, and both attack bounds sit
//! *under* the mandated minimum of five rounds at every point of POLICY §3's
//! grid — for all four fields, both widths, and alpha 3, 5 and 7. So `R = 8` is
//! not a value copied across from Goldilocks under POLICY §3's provisional rule:
//! it is what the reference's own criterion returns at each point, and the round
//! count simply does not grow with `t` for this design. Every cost number here
//! is therefore comparable across the grid in a way it is not for a construction
//! whose rounds track the width.
//!
//! # …and one instance does not take that number
//!
//! [`goldilocks_t12_r13`] is Goldilocks `t = 12` at `R = 13`, the author's own
//! more recent cryptanalysis, and it is the one structural parameter in this
//! crate `../ref` does not also derive (POLICY §4). It is an *addition*, not a
//! correction: [`goldilocks_t12`] stays at the reference's `R = 8`, exact and
//! measured, and the two are separate instances with separate names and separate
//! vectors. The comparability argument above therefore still holds among the
//! eight derived points; a table that mixes `R = 13` in is comparing one
//! construction's newer analysis against others' published rounds, which is a
//! statement about round counts and not about arithmetization.
//!
//! Its constants are still the reference's, because `_init_cons` seeds SHAKE-256
//! with `Rescue-XLIX(p,t,c,kappa)` — no round count in the string — and draws
//! `2R` rows off that one stream. The `R = 13` table is the `R = 8` table plus
//! ten more rows, which `tests/reference.rs` checks byte for byte rather than
//! taking on trust.
//!
//! # alpha is a property of the field
//!
//! The S-box is a bijection only when `gcd(alpha, p - 1) == 1`, so the exponent
//! is not a free choice per instance: 7 over Goldilocks and BabyBear, 5 over
//! Mersenne-31 (where `7 | p - 1`), 3 over KoalaBear. Those are the values
//! `../ref`'s `utils/field.py` pins per prime — and the same values
//! `p3-rescue`'s `PermutationMonomial<ALPHA>` bound admits per field, so an
//! instance built at the wrong exponent does not compile.
//!
//! # Capacity is reported, not chosen
//!
//! POLICY §8: nothing here compares modes, so rate, capacity and digest are the
//! reference's. Capacity appears in this file for one reason only — it is in the
//! seed string, so a different capacity is a different constant set. The values
//! are the reference's own derivation: 4 at Goldilocks, 8 at every 31-bit point.

use p3_field::{Field, PrimeField64};
use reference::sampler::{ShakeModSampler, inverse_exponent};

/// Rounds `_init_rounds` derives, at every point of the grid.
///
/// A "round" is a *double* round: two half-rounds, one forward S-box and one
/// inverse, each with its own linear layer and constant row.
pub const ROUNDS: usize = 8;

/// Half-rounds per call — `2 * ROUNDS`, and the number of round-constant rows.
///
/// This is the layout parameter rather than `ROUNDS`, because every half-round
/// is one S-box layer, one linear layer and one constant row, and the trace
/// commits in units of half-rounds. Stable Rust cannot compute `2 * ROUNDS` in a
/// const-generic position, so instances name this constant directly.
pub const HALF_ROUNDS: usize = 16;

/// The author-supplied Goldilocks round count: `R = 13`, so 26 half-rounds.
///
/// Every other instance here carries the reference's own `_init_rounds` value.
/// This one does not, and it is the only structural parameter in this crate that
/// `../ref` does not also derive — the same standing POLICY §4 gives the GMiMC
/// and GMiMC2 small-prime round counts, and it is recorded there.
///
/// It does not replace [`HALF_ROUNDS`] at Goldilocks `t = 12`: the
/// reference-pinned `RESCUE_PRIME_GOLDILOCKS_T12` stays, exact and measured,
/// beside it (POLICY §3 — an instance the reference pins matches it exactly, and
/// `marvellous/instances.py` is not ours to edit). Two instances at one grid
/// point, differing only in round count, is what an author-supplied count *is*.
pub const HALF_ROUNDS_GOLDILOCKS_R13: usize = 26;

/// The security level in the round-constant seed string, `kappa`.
///
/// The reference's default, and what every instance on this grid uses. It is not
/// the harness's security target (POLICY §7) and has nothing to do with it: it
/// is a byte in a seed.
pub const SECURITY_LEVEL: usize = 128;

/// Capacity of the reference's Goldilocks instances.
pub const CAPACITY_GOLDILOCKS: usize = 4;
/// Capacity the reference derives at every 31-bit grid point.
pub const CAPACITY_31: usize = 8;

/// A fully specified Rescue-Prime instance.
#[derive(Clone, Debug)]
pub struct RescuePrimeParams<F, const WIDTH: usize, const HALF_ROUNDS: usize> {
    /// Reference/export name — the only thing tying this to its vectors.
    pub name: &'static str,
    /// S-box exponent, with `gcd(alpha, p - 1) == 1`.
    pub alpha: u64,
    /// `alpha^{-1} mod (p - 1)`. Used by the native oracle and by the witness
    /// generator; no AIR ever exponentiates by it.
    pub alpha_inv: u64,
    /// Sponge capacity. Carried because it seeds [`Self::rcons`], reported
    /// because POLICY §8 compares no mode.
    pub capacity: usize,
    /// The dense Vandermonde-derived MDS matrix.
    pub m: [[F; WIDTH]; WIDTH],
    /// `m^{-1}`, which the full-round AIR layout evaluates and the half-round
    /// one does not. Derived here rather than in `air.rs` so that the one test
    /// asserting `m * m_inv == I` covers both layouts.
    pub m_inv: [[F; WIDTH]; WIDTH],
    /// One row per half-round, added *after* that half-round's linear layer.
    pub rcons: [[F; WIDTH]; HALF_ROUNDS],
}

/// `left^{-1} * right`, by Gauss-Jordan on the augmented matrix.
///
/// The two callers are the two things this file needs a linear solve for: the
/// echelon form the reference's Vandermonde construction takes, and the matrix
/// inverse the full-round layout evaluates.
///
/// # Panics
///
/// If `left` is singular.
fn left_solve<F: Field, const WIDTH: usize>(
    mut left: [[F; WIDTH]; WIDTH],
    mut right: [[F; WIDTH]; WIDTH],
) -> [[F; WIDTH]; WIDTH] {
    for col in 0..WIDTH {
        let pivot = (col..WIDTH)
            .find(|&row| left[row][col] != F::ZERO)
            .expect("the matrix to solve against is invertible");
        left.swap(col, pivot);
        right.swap(col, pivot);

        let scale = left[col][col].inverse();
        for j in 0..WIDTH {
            left[col][j] *= scale;
            right[col][j] *= scale;
        }

        for row in 0..WIDTH {
            if row == col {
                continue;
            }
            let factor = left[row][col];
            if factor == F::ZERO {
                continue;
            }
            for j in 0..WIDTH {
                let (l, r) = (left[col][j], right[col][j]);
                left[row][j] -= factor * l;
                right[row][j] -= factor * r;
            }
        }
    }
    right
}

/// `../ref`'s `vandermonde_mds_matrix(p, t, g, transpose=True)`.
///
/// The reference builds the `t x 2t` Vandermonde `V[i][j] = g^(i*j)`, takes the
/// right half of its echelon form, and transposes it. Because the left half is a
/// Vandermonde on the distinct nodes `g^0, ..., g^(t-1)`, it is invertible, the
/// echelon form is `[I | V_left^{-1} * V_right]`, and the row operations Sage
/// performs cannot change the answer.
///
/// The result is **dense with unstructured entries** — the reference says so.
/// That costs prover work in generation and expression size in the AIR, at no
/// constraint and no degree: the entries touch neither. It is a property of the
/// instance, and the same family's structured alternative is RPO's circulant,
/// which XHash's row carries.
///
/// `F::GENERATOR` is the reference's `g`. Both are the field's multiplicative
/// generator and they agree on all four primes (7, 7, 31, 3), which is asserted
/// against the export's `g` rather than trusted.
fn vandermonde_mds<F: PrimeField64, const WIDTH: usize>() -> [[F; WIDTH]; WIDTH] {
    let g = F::GENERATOR;
    let left = core::array::from_fn(|i| core::array::from_fn(|j| g.exp_u64((i * j) as u64)));
    let right =
        core::array::from_fn(|i| core::array::from_fn(|j| g.exp_u64((i * (j + WIDTH)) as u64)));

    let echelon = left_solve::<F, WIDTH>(left, right);
    core::array::from_fn(|i| core::array::from_fn(|j| echelon[j][i]))
}

impl<F: PrimeField64, const WIDTH: usize, const HALF_ROUNDS: usize>
    RescuePrimeParams<F, WIDTH, HALF_ROUNDS>
{
    /// Reproduce `RescuePrimeParams._init_cons` and `_init_mat` for this field.
    ///
    /// The constant stream has no rejection, so the derivation is pinned by the
    /// seed string and the draw count alone — but the seed string carries four
    /// values, and getting any of them wrong yields a full, plausible-looking
    /// constant set. `tests/reference.rs` checks every constant against the
    /// export.
    ///
    /// # Panics
    ///
    /// If `HALF_ROUNDS` is odd, or if `gcd(alpha, p - 1) != 1` — the condition
    /// making the S-box a bijection, without which the AIR's `root^alpha` pin
    /// would not pin a unique root.
    #[must_use]
    pub fn derive(name: &'static str, alpha: u64, capacity: usize) -> Self {
        assert!(
            HALF_ROUNDS.is_multiple_of(2),
            "a Rescue-Prime round is two half-rounds"
        );
        assert!(capacity < WIDTH, "capacity must leave a positive rate");
        let alpha_inv = inverse_exponent(alpha, F::ORDER_U64 - 1);

        let m = vandermonde_mds::<F, WIDTH>();
        let identity = core::array::from_fn(|i| core::array::from_fn(|j| F::from_bool(i == j)));
        let m_inv = left_solve::<F, WIDTH>(m, identity);

        // The reference's `_init_cons`: `LABEL(p,t,c,kappa)`, one SHAKE-256
        // stream, `2R` rows of `t` drawn row-major. Upstream derives the same
        // bytes from the same string.
        let seed = format!(
            "Rescue-XLIX({},{},{},{})",
            F::ORDER_U64,
            WIDTH,
            capacity,
            SECURITY_LEVEL
        );
        let rows = ShakeModSampler::<F>::new(seed.as_bytes()).grid(HALF_ROUNDS, WIDTH);
        let rcons = core::array::from_fn(|r| core::array::from_fn(|i| rows[r][i]));

        Self {
            name,
            alpha,
            alpha_inv,
            capacity,
            m,
            m_inv,
            rcons,
        }
    }

    /// Rounds — half the half-rounds.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        HALF_ROUNDS / 2
    }
}

/// Goldilocks `t = 8`, pinned by the reference as `RESCUE_PRIME_GOLDILOCKS_T8`.
#[must_use]
pub fn goldilocks_t8<F: PrimeField64>() -> RescuePrimeParams<F, 8, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-goldilocks-t8", 7, CAPACITY_GOLDILOCKS)
}

/// Goldilocks `t = 12`, pinned by the reference as `RESCUE_PRIME_GOLDILOCKS_T12`.
#[must_use]
pub fn goldilocks_t12<F: PrimeField64>() -> RescuePrimeParams<F, 12, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-goldilocks-t12", 7, CAPACITY_GOLDILOCKS)
}

/// Goldilocks `t = 12` at the author's `R = 13`, generated at the export call
/// site as `RESCUE_PRIME_GOLDILOCKS_T12_R13`.
///
/// The constants are not a second derivation. `_init_cons` seeds SHAKE-256 with
/// `Rescue-XLIX(p,t,c,kappa)` — a string with no round count in it — and reads
/// `2R` rows off that one stream, so this table *extends* [`goldilocks_t12`]'s:
/// the first sixteen rows are the same bytes, and ten more follow. The
/// Vandermonde MDS depends on `g` and `t` alone and does not move at all.
/// `tests/reference.rs` checks both halves of that claim against the export.
#[must_use]
pub fn goldilocks_t12_r13<F: PrimeField64>() -> RescuePrimeParams<F, 12, HALF_ROUNDS_GOLDILOCKS_R13>
{
    RescuePrimeParams::derive("rescue-prime-goldilocks-t12-r13", 7, CAPACITY_GOLDILOCKS)
}

/// Mersenne-31 `t = 16`, generated at the export call site.
#[must_use]
pub fn mersenne_t16<F: PrimeField64>() -> RescuePrimeParams<F, 16, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-mersenne-t16", 5, CAPACITY_31)
}

/// Mersenne-31 `t = 24`, generated at the export call site.
#[must_use]
pub fn mersenne_t24<F: PrimeField64>() -> RescuePrimeParams<F, 24, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-mersenne-t24", 5, CAPACITY_31)
}

/// BabyBear `t = 16`, generated at the export call site.
#[must_use]
pub fn babybear_t16<F: PrimeField64>() -> RescuePrimeParams<F, 16, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-babybear-t16", 7, CAPACITY_31)
}

/// BabyBear `t = 24`, generated at the export call site.
#[must_use]
pub fn babybear_t24<F: PrimeField64>() -> RescuePrimeParams<F, 24, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-babybear-t24", 7, CAPACITY_31)
}

/// KoalaBear `t = 16`, generated at the export call site.
#[must_use]
pub fn koalabear_t16<F: PrimeField64>() -> RescuePrimeParams<F, 16, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-koalabear-t16", 3, CAPACITY_31)
}

/// KoalaBear `t = 24`, generated at the export call site.
#[must_use]
pub fn koalabear_t24<F: PrimeField64>() -> RescuePrimeParams<F, 24, HALF_ROUNDS> {
    RescuePrimeParams::derive("rescue-prime-koalabear-t24", 3, CAPACITY_31)
}
