//! Native parameter derivation, reproducing `../ref`'s own samplers.
//!
//! # The rule this module exists to enforce
//!
//! **Never hand-write a constant, never re-derive outside the reference**
//! (POLICY §3). Instances the reference defines are matched exactly — rounds,
//! constants, matrix, byte-exact against its KATs. For other instances,
//! `export_small_prime_kat.py` constructs the params object and lets its class
//! run `_init_cons` / `_init_mat`; what lands here is a Rust reimplementation
//! of the *same* sampler, locked against the reference output in a test.
//!
//! Where a derivation is a `NotImplementedError` stub — thirteen of the
//! reference's `params.py` files have them, Monolith's `_init_rounds` among
//! them — that grid point is **absent and reported**, never filled by a
//! plausible-looking sampler here.
//!
//! # Contents
//!
//! SHAKE, BLAKE3, SHA-256 and Grain LFSR are the families the reference uses.
//! Each lands here when the repository needs to reproduce or audit it, locked
//! against `../ref` before any generated parameter set is built on it.
//!
//! * [`ShakeModSampler`] — `XOFFieldElementSampler(xof="shake_256",
//!   sampling="mod")`. First user: pSquareHash's `_init_cons`.
//! * [`Shake128BitmaskSampler`] — `XOFFieldElementSampler(xof="shake_128",
//!   sampling="bitmask")`. Users: Neptune, Griffin.
//! * [`Blake3ModSampler`] — `XOFFieldElementSampler(xof="blake3",
//!   sampling="mod", n_bytes=…)`, as used by Tip5's `_init_cons` in the
//!   reference. Exact Tip4/Tip5 instances currently load their exported values.
//! * [`Sha256NaiveSampler`] — `XOFFieldElementSampler(xof="sha256",
//!   sampling="naive", n_bytes=…)`, as used by Tip5's MDS column derivation.
//!   Exact Tip4/Tip5 instances currently load the exported matrix. SHA-256 is
//!   not an XOF at all; the reference wraps its 32-byte digest in the same
//!   `digest(n)` interface and raises past it, which is why this one is bounded
//!   where the other three are not.
//! * Grain LFSR — not yet needed (Poseidon1/2 are wrapped, so their constants
//!   come from upstream).
//!
//! The two samplers used by Tip5's reference derivation both take their draw width as an argument rather than
//! reading it off the field. That is the reference's `n_bytes` override, and it
//! is a property of the *derivation*, not of the prime: Tip5 draws `t` bytes per
//! round constant and two per MDS entry, where the field's own width is nine and
//! eight. A sampler that computed the width from `p` would produce a perfectly
//! plausible and completely different constant set.
//!
//! Two derivations that are not samplers live here for the same reason — they
//! reproduce a reference computation, out of circuit: [`field_order_seed`], the
//! seed string those samplers are started from, and [`inverse_exponent`], the
//! S-box exponent's inverse.

use core::marker::PhantomData;

use p3_field::PrimeField64;
use sha2::{Digest, Sha256};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Shake128, Shake256};

/// Widest draw any reference sampler here asks for, in bytes.
///
/// Nine is the `sampling="mod"` width of a 64-bit prime; sixteen is Tip5's
/// `n_bytes = t`. A `u128` holds either with room to spare, and `from_int` does
/// the reduction — which is why every draw below reads into a 16-byte buffer.
const MAX_DRAW_BYTES: usize = 16;

/// The reference's `name || p` sampler seed: a construction's byte string
/// followed by the field characteristic, little-endian, padded to whole 64-bit
/// limbs.
///
/// Both users write it the same way —
/// `b"Neptune" + self.p.to_bytes(((self.p.bit_length() + 63) // 64) * 8, "little")`
/// in `hades/params.py::_init_sampler` and `b"Griffin" + ...` in
/// `griffin/params.py::_init_cons`. Every field in POLICY §3's grid has
/// `p < 2^64`, so that padding is exactly one limb and
/// [`PrimeField64::ORDER_U64`] serialized little-endian is the whole tail.
///
/// The trap is that this seed is invisible in the output: a wrong prefix, a
/// big-endian order, or a differently padded limb all produce a stream that
/// still looks like uniformly random field elements. Nothing catches that
/// except comparing derived constants against the reference's export, which is
/// what each construction's `parameters_match_reference_export` test does.
#[must_use]
pub fn field_order_seed<F: PrimeField64>(prefix: &[u8]) -> Vec<u8> {
    let mut seed = prefix.to_vec();
    seed.extend_from_slice(&F::ORDER_U64.to_le_bytes());
    seed
}

/// `alpha^{-1} mod (p - 1)`: the exponent that inverts the S-box power map.
///
/// The extended Euclidean algorithm over `i128`, which is wide enough that the
/// intermediate `old_s - q * s` cannot overflow for a `u64` modulus.
///
/// **It panics unless `gcd(alpha, p - 1) == 1`**, and that panic is load-bearing
/// rather than defensive: the power map `x -> x^alpha` is a bijection exactly
/// when the gcd is one, and only then is "the alpha-th root" a single value. A
/// design that witnesses a root and pins it with `root^alpha == target`
/// (`gadgets::inverse_power_map`) proves nothing about *which* root when several
/// exist, so an instance whose alpha shares a factor with `p - 1` is not a
/// weaker instance — it is a different, unsound arithmetization. That is why
/// Mersenne-31 takes `alpha = 5` where Goldilocks takes 7: `7 | p - 1` there.
///
/// # Panics
///
/// If `alpha` is not invertible modulo `modulus`.
#[must_use]
pub fn inverse_exponent(alpha: u64, modulus: u64) -> u64 {
    let (mut old_r, mut r) = (i128::from(alpha), i128::from(modulus));
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    assert!(old_r == 1, "alpha must be invertible modulo p - 1");
    old_s.rem_euclid(i128::from(modulus)) as u64
}

/// The reference's `XOFFieldElementSampler(xof="shake_256", sampling="mod")`,
/// little-endian.
///
/// One draw is `ceil(bits(p) / 8) + 1` bytes of the SHAKE256 stream, read
/// little-endian and reduced mod `p`. The extra byte is what makes the
/// reduction's bias negligible instead of catastrophic, and it is *the*
/// distinguishing feature of `sampling="mod"`: the other three strategies read
/// exactly `ceil(bits / 8)` bytes and reject out-of-range draws rather than
/// reducing them. Reading one byte too few or too many silently produces a
/// different, plausible-looking constant stream — which is why the sampler is
/// locked against the reference's own output in a test before anything is built
/// on it.
///
/// Reading sequentially off the reader reproduces the reference's growing
/// `digest(n)` buffer exactly: SHAKE's longer output extends its shorter one, so
/// every previously read position stays valid.
///
/// The stream position advances by a fixed amount per draw here — there is no
/// rejection — so the *sequence* of calls is all that a derivation has to pin
/// down. Samplers with rejection do not have that property.
pub struct ShakeModSampler<F> {
    reader: sha3::Shake256Reader,
    _phantom: PhantomData<F>,
}

/// The reference's `XOFFieldElementSampler(xof="shake_128", sampling="bitmask")`.
///
/// Neptune draws `ceil(log2(p) / 8)` little-endian bytes, masks unused high
/// bits, and rejects candidates outside the field.  The fixed byte width means
/// rejection changes the stream position, so this belongs in the shared sampler
/// rather than being recreated at a construction call site.
pub struct Shake128BitmaskSampler<F> {
    reader: sha3::Shake128Reader,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField64> Shake128BitmaskSampler<F> {
    /// Bytes in a masked candidate.
    pub const DRAW_BYTES: usize = {
        let bits = (u64::BITS - F::ORDER_U64.leading_zeros()) as usize;
        bits.div_ceil(8)
    };

    /// Mask applied to the most significant little-endian byte.
    pub const TOP_MASK: u8 = {
        let bits = (u64::BITS - F::ORDER_U64.leading_zeros()) as usize;
        let rem = bits % 8;
        if rem == 0 { u8::MAX } else { (1u8 << rem) - 1 }
    };

    /// Start a SHAKE128 stream at `seed`.
    #[must_use]
    pub fn new(seed: &[u8]) -> Self {
        let mut hasher = Shake128::default();
        hasher.update(seed);
        Self {
            reader: hasher.finalize_xof(),
            _phantom: PhantomData,
        }
    }

    /// One rejection-sampled, canonically represented field element.
    pub fn next_element(&mut self) -> F {
        loop {
            let mut bytes = [0u8; 8];
            self.reader.read(&mut bytes[..Self::DRAW_BYTES]);
            bytes[Self::DRAW_BYTES - 1] &= Self::TOP_MASK;
            let candidate = u64::from_le_bytes(bytes);
            if candidate < F::ORDER_U64 {
                return F::from_u64(candidate);
            }
        }
    }

    /// The next non-zero field element, consuming rejected zero draws exactly
    /// as the reference's `next_nonzero` does.
    pub fn next_nonzero(&mut self) -> F {
        loop {
            let value = self.next_element();
            if value != F::ZERO {
                return value;
            }
        }
    }
}

impl<F: PrimeField64> ShakeModSampler<F> {
    /// Bytes per draw: the field's serialized width, plus one.
    pub const DRAW_BYTES: usize = {
        let bits = (u64::BITS - F::ORDER_U64.leading_zeros()) as usize;
        bits.div_ceil(8) + 1
    };

    /// Seed the stream. The seed is the construction's own byte string; this
    /// sampler does not invent one.
    #[must_use]
    pub fn new(seed: &[u8]) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(seed);
        Self {
            reader: hasher.finalize_xof(),
            _phantom: PhantomData,
        }
    }

    /// The next field element off the stream.
    pub fn next_element(&mut self) -> F {
        // 9 bytes is the widest draw a `p < 2^64` field asks for, so a u128
        // holds the raw value with room to spare and `from_int` does the
        // reduction.
        let mut bytes = [0u8; 16];
        self.reader.read(&mut bytes[..Self::DRAW_BYTES]);
        F::from_int(u128::from_le_bytes(bytes))
    }

    /// `rows × cols` elements, row-major — the reference's `grid`.
    ///
    /// Row-major matters: the grid is one flat stream cut into rows, so a
    /// column-major fill would produce the same multiset in the wrong places.
    #[must_use]
    pub fn grid(&mut self, rows: usize, cols: usize) -> Vec<Vec<F>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_element()).collect())
            .collect()
    }
}

/// The reference's `XOFFieldElementSampler(xof="blake3", sampling="mod",
/// n_bytes=…)`, little-endian.
///
/// One draw is `n_bytes` of the BLAKE3 extendable output, read little-endian and
/// reduced mod `p`. There is no rejection, so the stream position advances by a
/// fixed amount and the *sequence* of calls is all a derivation has to pin down.
///
/// **`n_bytes` is an argument, not the field's width.** Tip5 reads `t` bytes per
/// round constant — twelve or sixteen where a 64-bit prime's own `sampling="mod"`
/// width would be nine — and reading the field's width instead yields constants
/// that look exactly as random and are not the reference's. Tip5 also seeds a
/// *fresh* sampler per constant (`"Tip5" || byte(i + r*t)`) and takes a single
/// element from each, so this type is usually constructed and dropped in one
/// expression rather than iterated.
pub struct Blake3ModSampler<F> {
    reader: blake3::OutputReader,
    draw_bytes: usize,
    _phantom: PhantomData<F>,
}

/// The reference's `XOFFieldElementSampler(xof="sha256", sampling="naive",
/// n_bytes=…)`, little-endian.
///
/// SHA-256 is a fixed 32-byte digest, which the reference exposes through the
/// XOF `digest(n)` interface and **bounds**: drawing past 32 bytes raises there
/// and panics here. That bound is the load-bearing part — a sampler that
/// silently rehashed to extend the stream would agree with the reference on
/// every draw the reference actually takes and diverge on the first one it
/// refuses.
///
/// `sampling="naive"` reads `n_bytes` little-endian bytes with no bit-trimming
/// and rejects a draw at or above `p`, where `sampling="bitmask"` would first
/// mask the top byte down to the prime's exact bit length. With Tip5's
/// `n_bytes = 2` a draw is at most `2^16 - 1` and nothing is ever rejected —
/// which is exactly why the entries are 16-bit by design and why the column is
/// the same over every prime above `2^16`.
pub struct Sha256NaiveSampler<F> {
    digest: [u8; 32],
    position: usize,
    draw_bytes: usize,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField64> Blake3ModSampler<F> {
    /// Seed a BLAKE3 stream, drawing `draw_bytes` per element.
    ///
    /// # Panics
    ///
    /// If `draw_bytes` is zero or exceeds 16.
    #[must_use]
    pub fn new(seed: &[u8], draw_bytes: usize) -> Self {
        assert!(
            (1..=MAX_DRAW_BYTES).contains(&draw_bytes),
            "a draw is 1..={MAX_DRAW_BYTES} bytes"
        );
        let mut hasher = blake3::Hasher::new();
        hasher.update(seed);
        Self {
            reader: hasher.finalize_xof(),
            draw_bytes,
            _phantom: PhantomData,
        }
    }

    /// The next field element off the stream.
    pub fn next_element(&mut self) -> F {
        let mut bytes = [0u8; MAX_DRAW_BYTES];
        self.reader.fill(&mut bytes[..self.draw_bytes]);
        F::from_int(u128::from_le_bytes(bytes))
    }
}

impl<F: PrimeField64> Sha256NaiveSampler<F> {
    /// Hash `seed` and start reading `draw_bytes` per element off the digest.
    ///
    /// # Panics
    ///
    /// If `draw_bytes` is zero or exceeds 16.
    #[must_use]
    pub fn new(seed: &[u8], draw_bytes: usize) -> Self {
        assert!(
            (1..=MAX_DRAW_BYTES).contains(&draw_bytes),
            "a draw is 1..={MAX_DRAW_BYTES} bytes"
        );
        Self {
            digest: Sha256::digest(seed).into(),
            position: 0,
            draw_bytes,
            _phantom: PhantomData,
        }
    }

    /// The next rejection-sampled field element off the digest.
    ///
    /// # Panics
    ///
    /// If the draw runs past the digest's 32 bytes, which is where the
    /// reference's bounded byte source raises.
    pub fn next_element(&mut self) -> F {
        loop {
            let end = self.position + self.draw_bytes;
            assert!(
                end <= self.digest.len(),
                "SHA-256 is a bounded source: {end} bytes drawn of 32"
            );
            let mut bytes = [0u8; MAX_DRAW_BYTES];
            bytes[..self.draw_bytes].copy_from_slice(&self.digest[self.position..end]);
            self.position = end;
            let candidate = u128::from_le_bytes(bytes);
            if candidate < u128::from(F::ORDER_U64) {
                return F::from_int(candidate);
            }
        }
    }

    /// `rows × cols` elements, row-major — the reference's `grid`.
    #[must_use]
    pub fn grid(&mut self, rows: usize, cols: usize) -> Vec<Vec<F>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_element()).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_mersenne_31::Mersenne31;

    use super::*;

    /// The draw width is the one thing a reader cannot check by inspection, and
    /// getting it wrong still yields a stream that looks random.
    #[test]
    fn draw_width_is_the_field_width_plus_one_byte() {
        assert_eq!(ShakeModSampler::<Mersenne31>::DRAW_BYTES, 5);
        assert_eq!(ShakeModSampler::<BabyBear>::DRAW_BYTES, 5);
        assert_eq!(ShakeModSampler::<KoalaBear>::DRAW_BYTES, 5);
        assert_eq!(ShakeModSampler::<Goldilocks>::DRAW_BYTES, 9);
    }

    /// A fresh sampler on the same seed is the same stream; that is the whole
    /// point of deriving constants instead of storing them.
    #[test]
    fn the_stream_is_a_function_of_the_seed() {
        let a: Vec<BabyBear> = (0..8)
            .map(|_| ShakeModSampler::<BabyBear>::new(b"seed").next_element())
            .collect();
        assert!(a.iter().all(|x| *x == a[0]));

        let mut s = ShakeModSampler::<BabyBear>::new(b"seed");
        let first: Vec<_> = (0..8).map(|_| s.next_element()).collect();
        let mut s = ShakeModSampler::<BabyBear>::new(b"seed");
        assert_eq!(s.grid(2, 4), vec![first[..4].to_vec(), first[4..].to_vec()]);

        let mut other = ShakeModSampler::<BabyBear>::new(b"seed ");
        assert_ne!(other.next_element(), first[0]);
    }

    #[test]
    fn bitmask_draw_width_matches_the_reference_strategy() {
        assert_eq!(Shake128BitmaskSampler::<Mersenne31>::DRAW_BYTES, 4);
        assert_eq!(Shake128BitmaskSampler::<Mersenne31>::TOP_MASK, 0x7f);
        assert_eq!(Shake128BitmaskSampler::<Goldilocks>::DRAW_BYTES, 8);
        assert_eq!(Shake128BitmaskSampler::<Goldilocks>::TOP_MASK, 0xff);
    }

    /// The padding rule is `((bit_length + 63) // 64) * 8`, which is one limb
    /// for every prime in the grid — including the 31-bit ones, where a
    /// byte-tight encoding would be four bytes and would silently produce a
    /// different stream.
    #[test]
    fn the_seed_is_the_prefix_and_one_little_endian_limb() {
        assert_eq!(
            field_order_seed::<Mersenne31>(b"Griffin"),
            [b"Griffin".as_slice(), &[0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0]].concat()
        );
        assert_eq!(
            field_order_seed::<Goldilocks>(b"Neptune"),
            [b"Neptune".as_slice(), &Goldilocks::ORDER_U64.to_le_bytes()].concat()
        );
        assert_eq!(field_order_seed::<BabyBear>(b"").len(), 8);
    }

    /// Round-tripping is the check that needs no constant: `x^alpha` then
    /// `x^alpha_inv` is the identity on every element exactly when the exponent
    /// is right, and it exercises `0` and `1`, the two elements a wrong
    /// exponent still maps correctly.
    fn round_trips<F: PrimeField64>(alpha: u64) {
        let alpha_inv = inverse_exponent(alpha, F::ORDER_U64 - 1);
        for raw in [0, 1, 2, 7, F::ORDER_U64 - 1, F::ORDER_U64 / 2] {
            let x = F::from_u64(raw);
            assert_eq!(x.exp_u64(alpha).exp_u64(alpha_inv), x);
            assert_eq!(x.exp_u64(alpha_inv).exp_u64(alpha), x);
        }
    }

    #[test]
    fn the_inverse_exponent_inverts_the_power_map() {
        round_trips::<Goldilocks>(7);
        round_trips::<Mersenne31>(5);
        round_trips::<BabyBear>(7);
        round_trips::<KoalaBear>(3);
        // KoalaBear is the one grid field where `p - 1 = 127 * 2^24` is coprime
        // to all three exponents, so all three are admissible there.
        round_trips::<KoalaBear>(5);
        round_trips::<KoalaBear>(7);
    }

    /// The four values `../ref`'s `utils/field.py` pins for its own fields.
    #[test]
    fn the_inverse_exponent_matches_the_reference() {
        assert_eq!(
            inverse_exponent(7, Goldilocks::ORDER_U64 - 1),
            10_540_996_611_094_048_183
        );
        assert_eq!(
            inverse_exponent(5, Mersenne31::ORDER_U64 - 1),
            1_717_986_917
        );
        assert_eq!(inverse_exponent(7, BabyBear::ORDER_U64 - 1), 1_725_656_503);
        assert_eq!(inverse_exponent(3, KoalaBear::ORDER_U64 - 1), 1_420_470_955);
    }

    /// The panic is the guard on unique roots (see [`inverse_exponent`]), so it
    /// is tested rather than assumed: `7 | p - 1` over Mersenne-31.
    #[test]
    #[should_panic(expected = "alpha must be invertible")]
    fn a_non_invertible_exponent_panics() {
        let _ = inverse_exponent(7, Mersenne31::ORDER_U64 - 1);
    }
}
