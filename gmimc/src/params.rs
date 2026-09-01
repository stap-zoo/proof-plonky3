//! Parameters, from `../ref` and only from `../ref`.
//!
//! POLICY §3. **Every** grid point here is generated: `gmimc/instances.py`
//! defines the BN254 and BLS12-381 `t=3` pair and nothing over a small prime, so
//! all four points are constructed at the `export_small_prime_kat.py` call site
//! (`_gmimc_generated`), which runs `GMiMCParams`' own `_init_cons` and
//! `_init_mat` for the new prime. What lands here is a Rust reimplementation of
//! the *same* derivation, locked against the reference's export in
//! `tests/reference.rs`. Never hand-write a constant; never re-derive outside
//! the reference.
//!
//! # The round counts are a third provenance
//!
//! POLICY §3 names two sources, reference-exact and reference-derived. `R` is
//! neither. `GMiMCParams._init_rounds` raises `NotImplementedError` — the
//! paper's Section 5 criterion is unimplemented — so **93 at Goldilocks `t=12`
//! and 335 at every 31-bit `t=24` come from the author's own cryptanalysis**,
//! and nothing in `../ref` or in this repository checks them. They are reported,
//! never derived; README's standing "all round numbers are subject to
//! confirmation" item covers them, and every cost number in this row moves if
//! one is corrected.
//!
//! Everything else at these four points *is* reference-derived, which is why the
//! export is worth checking against: `_init_cons` samples the `R` constants
//! through SHAKE256 over the seed `GMiMC(p,t,R)aff`, and `_init_mat` builds the
//! shift.
//!
//! # Four absences, and why the provisional rule cannot rescue them
//!
//! Goldilocks `t=8` and the three 31-bit `t=16` points have no round count.
//! POLICY §3's provisional rule — copy the Goldilocks value, `t=8 → t=16`,
//! `t=12 → t=24` — needs a Goldilocks `t=8` value and there is none; and copying
//! across widths is unsound for an expanding round function in a way it is not
//! elsewhere, because the round criterion grows steeply with the branch count
//! (gnark's own registry records 228 rounds at `t=3` and 231 at `t=4` for one
//! alpha). So those four points are absent and reported
//! ([`Absence::StubbedDerivation`](harness::Absence)), never invented.
//!
//! # alpha is a property of the field
//!
//! The S-box `x -> x^alpha` must be a bijection, which `_input_sanitization`
//! enforces as `gcd(alpha, p - 1) == 1`, so the exponent is not a free choice
//! per instance: 7 over Goldilocks and BabyBear, 5 over Mersenne-31 (where
//! `7 | p - 1`), 3 over KoalaBear. Those are the values
//! [`ref/utils/field.py`](../../../ref/utils/field.py) pins per prime, which the
//! export call site passes rather than leaving to `_init_alpha` — a search that
//! returns the same value. [`GMiMCParams::derive`] re-runs the gcd check through
//! [`reference::sampler::inverse_exponent`] so a wrong exponent is a panic here
//! and not a permutation that is merely not the reference's.
//!
//! GMiMC2 is where that stops holding: its exponent is `2^k`, deliberately not a
//! permutation, which an expanding-round-function Feistel does not need it to be.
//! The two crates are separate for reasons like this one.

use p3_field::PrimeField64;
use reference::sampler::{ShakeModSampler, inverse_exponent};

/// Rounds at Goldilocks `t=12`. The author's, from cryptanalysis: no derivation
/// in `../ref` produces it and none can check it.
pub const ROUNDS_GOLDILOCKS_T12: usize = 93;

/// Rounds at every 31-bit prime, `t=24`. The author's, from cryptanalysis, and
/// the same at all three primes — the criterion is driven by the branch count
/// and the field size, both of which those three share.
pub const ROUNDS_31_T24: usize = 335;

/// A fully specified GMiMC-erf instance.
///
/// `ROUNDS` is `R`, and there is one constant per round rather than a row per
/// round: GMiMC is the only construction here whose `rcons` is flat.
/// `_init_cons` samples an `R × 1` grid and immediately flattens it
/// (`[con[0] for con in rcons]`), because the round function adds a single
/// constant to a single branch.
///
/// The linear layer is **not** a field of this struct. It is the cyclic shift
/// [`shift_source`] — a re-indexing, not a matrix — so there is nothing to store
/// and nothing for `p3_mds::util::mds_multiply` to do. The export carries `M`
/// anyway, and `tests/reference.rs` checks the shift's *direction* against it;
/// a reversed shift is exactly the bug that a known-answer test catches only
/// after the code has already been written the wrong way round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GMiMCParams<F, const WIDTH: usize, const ROUNDS: usize> {
    /// Reference/export name — the only thing tying this to its vectors
    /// (POLICY §3), and a mismatch here is silent.
    pub name: &'static str,
    /// S-box exponent, with `gcd(alpha, p - 1) == 1`.
    pub alpha: u64,
    /// `R` affine round constants, one per round, added to branch 0 before the
    /// power map.
    pub rcons: [F; ROUNDS],
}

/// The branch the cyclic-shift linear layer moves into position `i`.
///
/// `_init_mat` builds `mat[i][i + 1] = 1` for `i < t - 1` and `mat[t-1][0] = 1`,
/// so `matvecmul(M, state)` is `out[i] = state[i + 1]` with wraparound: the
/// state rotates **left** by one, and branch 0 — the one the S-box reads — is
/// the branch that entered at position 1.
///
/// A permutation matrix costs no cell and no multiplication, so the native
/// permutation re-indexes instead of multiplying. In the arithmetization the
/// shift disappears entirely: a branch leaving position 0 takes `t` rounds to
/// come back, which is what turns the whole layer into the index `r` of the
/// round recurrence rather than a layer at all (`air.rs`).
#[must_use]
pub const fn shift_source(i: usize, width: usize) -> usize {
    if i + 1 == width { 0 } else { i + 1 }
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize> GMiMCParams<F, WIDTH, ROUNDS> {
    /// Reproduce `GMiMCParams._init_cons` for this field.
    ///
    /// One SHAKE256 stream seeded with `GMiMC(p,t,R)aff`, drawn under the
    /// reference's `sampling="mod"` — one byte more than the field's serialized
    /// width, little-endian, reduced. The seed string is the reference's byte
    /// for byte, including the decimal `p` and the `aff` suffix, and it is
    /// invisible in the output: a wrong prefix or a wrong draw width yields a
    /// stream that looks exactly as random and is a different instance. That is
    /// what `tests/reference.rs` locks against the export.
    ///
    /// # Panics
    ///
    /// If `WIDTH < 2` (the reference's own bound: an erf needs a branch to
    /// expand into), or if `alpha` is not invertible modulo `p - 1`.
    #[must_use]
    pub fn derive(name: &'static str, alpha: u64) -> Self {
        assert!(WIDTH >= 2, "an erf Feistel needs at least two branches");
        // `_input_sanitization`: "power map does not define a permutation".
        // The value is not kept — GMiMC never inverts its S-box — but the check
        // is the reference's hard error, so it is run rather than assumed.
        let _ = inverse_exponent(alpha, F::ORDER_U64 - 1);

        let seed = format!("GMiMC({},{WIDTH},{ROUNDS})aff", F::ORDER_U64);
        let mut sampler = ShakeModSampler::<F>::new(seed.as_bytes());
        // `.grid(R, 1)` flattened: one constant per round, in stream order.
        let rcons = core::array::from_fn(|_| sampler.next_element());

        Self { name, alpha, rcons }
    }
}

/// Generated `GMIMC_GOLDILOCKS_T12`; `R = 93` is the author's, not derived.
#[must_use]
pub fn goldilocks_t12<F: PrimeField64>() -> GMiMCParams<F, 12, ROUNDS_GOLDILOCKS_T12> {
    GMiMCParams::derive("gmimc-goldilocks-t12", 7)
}

/// Generated `GMIMC_MERSENNE_T24`; `R = 335` is the author's, not derived.
#[must_use]
pub fn mersenne_t24<F: PrimeField64>() -> GMiMCParams<F, 24, ROUNDS_31_T24> {
    GMiMCParams::derive("gmimc-mersenne-t24", 5)
}

/// Generated `GMIMC_BABYBEAR_T24`; `R = 335` is the author's, not derived.
#[must_use]
pub fn babybear_t24<F: PrimeField64>() -> GMiMCParams<F, 24, ROUNDS_31_T24> {
    GMiMCParams::derive("gmimc-babybear-t24", 7)
}

/// Generated `GMIMC_KOALABEAR_T24`; `R = 335` is the author's, not derived.
#[must_use]
pub fn koalabear_t24<F: PrimeField64>() -> GMiMCParams<F, 24, ROUNDS_31_T24> {
    GMiMCParams::derive("gmimc-koalabear-t24", 3)
}

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;

    use super::*;

    /// The rotation is a left rotation and it wraps exactly once. Getting this
    /// backwards still gives a permutation, and one whose only witness is a
    /// known-answer vector — so it is worth stating on its own, where a reader
    /// can see the direction rather than infer it.
    #[test]
    fn the_shift_rotates_left_and_wraps() {
        assert_eq!(shift_source(0, 12), 1);
        assert_eq!(shift_source(10, 12), 11);
        assert_eq!(shift_source(11, 12), 0);
        // Applied `t` times it is the identity, which is what makes a branch's
        // round trip exactly `t` rounds — the fact the arithmetization is built on.
        let mut i = 0;
        for _ in 0..24 {
            i = shift_source(i, 24);
        }
        assert_eq!(i, 0);
    }

    /// A wrong exponent is the reference's hard error, not a different instance.
    /// `7 | p - 1` over Mersenne-31, which is why that field takes alpha 5.
    #[test]
    #[should_panic(expected = "alpha must be invertible")]
    fn a_non_bijective_exponent_is_refused() {
        let _ = GMiMCParams::<p3_mersenne_31::Mersenne31, 24, ROUNDS_31_T24>::derive("toy", 7);
    }

    /// The stream is a function of the seed, so a wrong `R` in the seed string
    /// is a wrong constant set from the first element on — including the case a
    /// reader would least expect to matter, two instances over the same prime.
    #[test]
    fn the_constant_stream_depends_on_the_whole_seed() {
        let real = goldilocks_t12::<Goldilocks>();
        let wrong_rounds = GMiMCParams::<Goldilocks, 12, 94>::derive("toy", 7);
        assert_ne!(real.rcons[0], wrong_rounds.rcons[0]);
        let wrong_width = GMiMCParams::<Goldilocks, 8, ROUNDS_GOLDILOCKS_T12>::derive("toy", 7);
        assert_ne!(real.rcons[0], wrong_width.rcons[0]);
    }
}
