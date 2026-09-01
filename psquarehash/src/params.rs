//! Parameters, from `../ref` and only from `../ref`.
//!
//! POLICY §3. Goldilocks and Mersenne-31 match the reference **exactly** —
//! rounds, constants, matrix, byte-exact against its KATs. BabyBear and
//! KoalaBear are, under current policy, constructed at the
//! `export_small_prime_kat.py` call site,
//! which runs this construction's `_init_cons` / `_init_mat` for the new prime.
//! Never hand-write a constant; never re-derive outside the reference.
//!
//! Round counts for generated instances are **provisional**: the construction's
//! Goldilocks value copied across (t = 8 to t = 16, t = 12 to t = 24) pending
//! real analysis. Every cost number scales with the round count, so it is a
//! reported column (POLICY §11) and the pinned numbers move when it is
//! corrected. No security claim attaches to a generated instance until then.
//!
//! Extending `../ref` is an edit to shared property: minimal, and each one
//! reported.
//!
//! # What the reference gives, and what it does not
//!
//! Two edits to `../ref` were needed and are the whole of them:
//!
//! 1. `psquarehash/instances.py` gained `PSQUAREHASH_{BABYBEAR,KOALABEAR}_T{16,24}`
//!    — `rcons` omitted, so `_init_cons` derives them, exactly as POLICY §3
//!    prescribes for a grid point the reference does not pin.
//! 2. `export_small_prime_kat.py` is new: the small-prime sibling of
//!    `export_kat.py` (POLICY §4), which is hardcoded to BN254 / BLS12-381 and
//!    has no entry for this construction. It emits the vectors and, with
//!    `--params`, the instance parameters.
//!
//! **There is no Goldilocks instance, and there cannot be one yet.**
//! `pSquareHashParams._init_rounds` raises `NotImplementedError` ("round number
//! derivation for pSquare-hash", pending <https://eprint.iacr.org/2026/1129>),
//! and the reference pins no Goldilocks parameter set, so `R` for t = 8 and
//! t = 12 exists nowhere. POLICY §3 makes those two grid points absent and
//! reported ([`Absence::StubbedDerivation`](harness::Absence)), never filled by
//! hand — and the copy-across rule cannot rescue them either, because it copies
//! *from* Goldilocks.
//!
//! # Two sources for the round constants, because the reference has two
//!
//! Mersenne-31's constants are **embedded** below. They have no derivation in
//! the reference: `_init_cons`'s SHAKE256 fallback does *not* reproduce them
//! (`tests/test_psquarehash.py::test_cons_generated_matches_instance` is
//! `@pytest.mark.skip`ped saying exactly that, pending the paper's own
//! derivation), so the pinned table in `instances.py` is their only definition
//! and copying it byte-for-byte is what "match exactly" means here.
//!
//! BabyBear's and KoalaBear's are **derived**, through
//! [`reference::sampler::ShakeModSampler`] reproducing `_init_cons`. Nothing is
//! hand-written either way, and `tests/reference.rs` locks both against
//! `vectors/params.json` — the reference's own output — before any permutation
//! runs.

use p3_field::PrimeField64;
use reference::sampler::ShakeModSampler;

/// One fully expanded instance: the shape, and the round constants in the shape
/// the round function reads them.
///
/// `WIDTH` is the grid coordinate `t`. `PAIRS` is `t / 4` — the number of
/// Feistels one round applies, which is also the number of round-constant pairs
/// per round and the number of output pairs the round writes. Carrying it as its
/// own const parameter rather than computing `WIDTH / 4` is forced: array lengths
/// in a `#[repr(C)]` column struct cannot be arithmetic on other const
/// parameters on stable Rust. [`assert_shape`] is what keeps the two honest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PSquareHashParams<F, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize> {
    /// The instance name (POLICY §3): the reference variable lowercased with `_`
    /// replaced by `-`. The only thing tying this instance to its vectors.
    pub name: &'static str,
    /// `ROUNDS` rows of `PAIRS` constant pairs.
    ///
    /// The reference holds one row of `t / 2` constants per round and the
    /// Feistel writing output pair `p` reads `rcons[r][2p .. 2p + 2]`
    /// (`nonlinear_layer` asks for `rcons[r][t/2 - (i+2) .. t/2 - i]` on input
    /// pair `i`, and `i = t/2 - 2 - 2p`). Pairing them here is that indexing
    /// done once, in the type, instead of at every use.
    pub rcons: [[[F; 2]; PAIRS]; ROUNDS],
}

/// The shape invariant every type in this crate is parameterized on.
///
/// `WIDTH = 4 * PAIRS` is not a convenience: the round function's structure
/// depends on it. `t % 4 == 0` is the reference's own hard check
/// (`_input_sanitization`), because one round consumes the lower half two
/// elements at a time and the linear layer singles out the *last* lower-half
/// pair, which only exists as a pair when `t / 2` is even.
///
/// Called from every `new`, so a mismatched instantiation fails at
/// monomorphization rather than producing a trace with a silently wrong layout.
pub const fn assert_shape(width: usize, pairs: usize) {
    assert!(width == 4 * pairs, "WIDTH must be 4 * PAIRS");
    assert!(
        pairs >= 2,
        "t < 8 leaves the linear layer's z-pair undefined"
    );
}

impl<F: PrimeField64, const WIDTH: usize, const PAIRS: usize, const ROUNDS: usize>
    PSquareHashParams<F, WIDTH, PAIRS, ROUNDS>
{
    /// Reshape a flat, row-major `ROUNDS × (WIDTH / 2)` table of canonical
    /// values — the layout `../ref` emits — into constant pairs.
    ///
    /// `from_canonical_checked` rather than `from_int`: every value the reference
    /// emits is already reduced, so a rejection here means the table and the
    /// field do not belong together (a Mersenne-31 constant handed to BabyBear,
    /// say, whose `p` is smaller). `from_int` would quietly reduce it instead.
    ///
    /// # Panics
    ///
    /// If `flat` is not `ROUNDS * WIDTH / 2` long, or holds a value `>= p`.
    fn from_flat(name: &'static str, flat: &[u32]) -> Self {
        assert_shape(WIDTH, PAIRS);
        assert_eq!(
            flat.len(),
            ROUNDS * 2 * PAIRS,
            "{name}: expected ROUNDS rows of t/2 constants"
        );
        let mut rcons = [[[F::ZERO; 2]; PAIRS]; ROUNDS];
        for (r, row) in rcons.iter_mut().enumerate() {
            for (p, pair) in row.iter_mut().enumerate() {
                for (k, c) in pair.iter_mut().enumerate() {
                    let raw = flat[r * 2 * PAIRS + 2 * p + k];
                    *c = F::from_canonical_checked(raw)
                        .unwrap_or_else(|| panic!("{name}: {raw} is not canonical in this field"));
                }
            }
        }
        Self { name, rcons }
    }

    /// The reference's `_init_cons`: `ROUNDS × (WIDTH / 2)` elements off a
    /// SHAKE256 stream seeded with `pSquare-hash(p,t,R)aff`.
    ///
    /// The seed string is the reference's, byte for byte, including the `aff`
    /// suffix and the decimal `p`. Anything else here is a different, equally
    /// plausible-looking instance — hence the lock against `vectors/params.json`
    /// in `tests/reference.rs`.
    fn derived(name: &'static str) -> Self {
        assert_shape(WIDTH, PAIRS);
        let seed = format!("pSquare-hash({},{WIDTH},{ROUNDS})aff", F::ORDER_U64);
        let mut sampler = ShakeModSampler::<F>::new(seed.as_bytes());
        let mut rcons = [[[F::ZERO; 2]; PAIRS]; ROUNDS];
        for row in &mut rcons {
            for pair in row.iter_mut() {
                *pair = [sampler.next_element(), sampler.next_element()];
            }
        }
        Self { name, rcons }
    }
}

// ---------------------------------------------------------------------------
// The reference's Mersenne-31 tables
//
// Embedded, not derived: see the module docs. `../ref`'s `instances.py` is the
// only definition these have, and `tests/reference.rs` checks them against the
// export rather than trusting the transcription.
// ---------------------------------------------------------------------------

/// `PSQUAREHASH_MERSENNE_T16`'s round constants, exactly as `../ref` pins
/// them: R = 52 rows of t/2 = 8, flattened row-major.
const MERSENNE_T16_RCONS: [u32; 52 * 8] = [
    0x2168c234, 0x27a7186e, 0x00dc1cd1, 0x43ea45ba, 0x0a67cc74, 0x1e9a8194, 0x3b139b22, 0x46f82383,
    0x42d18469, 0x4f4e30da, 0x01b839a3, 0x07d48b74, 0x14cf98e8, 0x3d350329, 0x76273644, 0x0df04707,
    0x05a308d3, 0x1e9c61b5, 0x03707347, 0x0fa916e7, 0x299f31d0, 0x7a6a0651, 0x6c4e6c88, 0x1be08e0f,
    0x0b4611a6, 0x3d38c36b, 0x06e0e68e, 0x1f522dce, 0x533e63a1, 0x74d40ca2, 0x589cd910, 0x37c11c20,
    0x168c234c, 0x7a7186d5, 0x0dc1cd1c, 0x3ea45b9c, 0x267cc742, 0x69a81945, 0x3139b220, 0x6f823842,
    0x2d184699, 0x74e30daa, 0x1b839a38, 0x7d48b737, 0x4cf98e85, 0x5350328b, 0x62736440, 0x5f047086,
    0x5a308d32, 0x69c61b55, 0x37073471, 0x7a916e6e, 0x19f31d0a, 0x26a06518, 0x44e6c880, 0x3e08e10d,
    0x34611a64, 0x538c36ab, 0x6e0e68e2, 0x7522dcdd, 0x33e63a14, 0x4d40ca30, 0x09cd9101, 0x7c11c21b,
    0x68c234c9, 0x27186d57, 0x5c1cd1c4, 0x6a45b9bb, 0x67cc7429, 0x1a819460, 0x139b2202, 0x78238438,
    0x51846992, 0x4e30daaf, 0x3839a389, 0x548b7377, 0x4f98e852, 0x350328c1, 0x27364404, 0x70470871,
    0x2308d324, 0x1c61b561, 0x70734713, 0x2916e6ef, 0x1f31d0a4, 0x6a065184, 0x4e6c8808, 0x608e10e3,
    0x4611a648, 0x38c36ac3, 0x60e68e26, 0x522dcddf, 0x3e63a148, 0x540ca30a, 0x1cd91010, 0x411c21c8,
    0x0c234c90, 0x7186d586, 0x41cd1c4c, 0x245b9bc1, 0x7cc74290, 0x28194615, 0x39b22020, 0x02384391,
    0x18469921, 0x630dab0c, 0x039a3898, 0x48b73783, 0x798e8520, 0x50328c2b, 0x73644041, 0x04708721,
    0x308d3243, 0x461b5617, 0x07347131, 0x116e6f06, 0x731d0a40, 0x20651858, 0x66c88082, 0x08e10e42,
    0x611a6487, 0x0c36ac2d, 0x0e68e263, 0x22dcde0b, 0x663a1481, 0x40ca30b1, 0x4d910105, 0x11c21c84,
    0x4234c90f, 0x186d5859, 0x1cd1c4c6, 0x45b9bc16, 0x4c742902, 0x01946165, 0x1b22020b, 0x23843908,
    0x0469921f, 0x30dab0b2, 0x39a3898c, 0x0b73782c, 0x18e85204, 0x0328c2cb, 0x36440417, 0x47087210,
    0x08d3243f, 0x61b56164, 0x73471319, 0x16e6f056, 0x31d0a409, 0x06518596, 0x6c88082e, 0x0e10e420,
    0x11a6487e, 0x436ac2c8, 0x668e2633, 0x2dcde0ac, 0x63a14812, 0x0ca30b2c, 0x5910105d, 0x1c21c83f,
    0x234c90fd, 0x06d5858f, 0x4d1c4c66, 0x5b9bc15a, 0x47429024, 0x19461658, 0x322020bb, 0x3843907e,
    0x469921fb, 0x0dab0b1d, 0x1a3898cc, 0x373782b7, 0x0e852049, 0x328c2cb0, 0x64404177, 0x708720fb,
    0x0d3243f6, 0x1b56163a, 0x34713198, 0x6e6f056e, 0x1d0a4093, 0x6518595f, 0x488082ef, 0x610e41f6,
    0x1a6487ed, 0x36ac2c74, 0x68e26331, 0x5cde0adc, 0x3a148127, 0x4a30b2be, 0x110105df, 0x421c83ee,
    0x34c90fda, 0x6d5858e7, 0x51c4c662, 0x39bc15b9, 0x7429024e, 0x1461657d, 0x22020bbe, 0x043907dd,
    0x69921fb5, 0x5ab0b1ce, 0x23898cc5, 0x73782b73, 0x6852049c, 0x28c2cafb, 0x4404177d, 0x08720fb9,
    0x53243f6a, 0x3561639d, 0x4713198a, 0x66f056e8, 0x50a40938, 0x518595f8, 0x08082efa, 0x10e41f72,
    0x26487ed5, 0x6ac2c73b, 0x0e263314, 0x4de0add2, 0x21481270, 0x230b2bf3, 0x10105df5, 0x21c83ee4,
    0x4c90fdaa, 0x55858e78, 0x1c4c6628, 0x1bc15ba5, 0x429024e0, 0x461657e6, 0x2020bbea, 0x43907dc8,
    0x1921fb54, 0x2b0b1cf2, 0x3898cc51, 0x3782b749, 0x052049c1, 0x0c2cafcd, 0x404177d4, 0x0720fb90,
    0x3243f6a8, 0x561639e4, 0x713198a2, 0x6f056e91, 0x0a409382, 0x18595f9b, 0x0082efa9, 0x0e41f71f,
    0x6487ed51, 0x2c2c73c8, 0x62633145, 0x5e0add22, 0x14812704, 0x30b2bf36, 0x0105df53, 0x1c83ee3e,
    0x490fdaa2, 0x5858e791, 0x44c6628b, 0x3c15ba45, 0x29024e08, 0x61657e6b, 0x020bbea6, 0x3907dc7c,
    0x121fb544, 0x30b1cf25, 0x098cc517, 0x782b748b, 0x52049c11, 0x42cafcd6, 0x04177d4c, 0x720fb8f8,
    0x243f6a88, 0x61639e4a, 0x13198a2e, 0x7056e918, 0x24093822, 0x0595f9ae, 0x082efa98, 0x641f71f0,
    0x487ed511, 0x42c73c94, 0x2633145c, 0x60add231, 0x48127044, 0x0b2bf35d, 0x105df531, 0x483ee3df,
    0x10fdaa22, 0x058e792a, 0x4c6628b8, 0x415ba463, 0x1024e088, 0x1657e6ba, 0x20bbea63, 0x107dc7bd,
    0x21fb5444, 0x0b1cf255, 0x18cc5170, 0x02b748c8, 0x2049c111, 0x2cafcd74, 0x4177d4c7, 0x20fb8f79,
    0x43f6a888, 0x1639e4aa, 0x3198a2e0, 0x056e9191, 0x40938222, 0x595f9ae7, 0x02efa98e, 0x41f71ef2,
    0x07ed5110, 0x2c73c954, 0x633145c0, 0x0add2322, 0x01270445, 0x32bf35cf, 0x05df531d, 0x03ee3de4,
    0x0fdaa221, 0x58e792a8, 0x46628b80, 0x15ba4644, 0x024e088a, 0x657e6b9f, 0x0bbea63b, 0x07dc7bc7,
    0x1fb54442, 0x31cf2550, 0x0cc51701, 0x2b748c88, 0x049c1114, 0x4afcd73e, 0x177d4c76, 0x0fb8f78e,
    0x3f6a8885, 0x639e4a9e, 0x198a2e03, 0x56e91910, 0x09382229, 0x15f9ae7b, 0x2efa98ec, 0x1f71ef1c,
    0x7ed5110b, 0x473c953c, 0x33145c06, 0x2dd23220, 0x12704453, 0x2bf35cf5, 0x5df531d8, 0x3ee3de37,
    0x7daa2216, 0x0e792a79, 0x6628b80d, 0x5ba4643e, 0x24e088a6, 0x57e6b9ea, 0x3bea63b1, 0x7dc7bc6e,
    0x7b54442d, 0x1cf254f3, 0x4c51701b, 0x3748c87c, 0x49c1114c, 0x2fcd73d4, 0x77d4c762, 0x7b8f78de,
    0x76a8885a, 0x39e4a9e8, 0x18a2e037, 0x6e9190f9, 0x13822299, 0x5f9ae7a7, 0x6fa98ec4, 0x771ef1bd,
    0x6d5110b4, 0x73c953d2, 0x3145c06e, 0x5d2321f4, 0x27044533, 0x3f35cf4e, 0x5f531d89, 0x6e3de37b,
    0x5aa22168, 0x6792a7a6, 0x628b80dc, 0x3a4643e9, 0x4e088a67, 0x7e6b9e9a, 0x3ea63b13, 0x5c7bc6f7,
    0x354442d1, 0x4f254f4d, 0x451701b8, 0x748c87d3, 0x1c1114cf, 0x7cd73d34, 0x7d4c7627, 0x38f78def,
    0x6a8885a3, 0x1e4a9e9b, 0x0a2e0370, 0x69190fa9, 0x3822299f, 0x79ae7a69, 0x7a98ec4e, 0x71ef1bdf,
    0x55110b46, 0x3c953d37, 0x145c06e0, 0x52321f53, 0x7044533e, 0x735cf4d3, 0x7531d89c, 0x63de37c0,
];

/// `PSQUAREHASH_MERSENNE_T24`'s round constants, exactly as `../ref` pins
/// them: R = 52 rows of t/2 = 12, flattened row-major.
const MERSENNE_T24_RCONS: [u32; 52 * 12] = [
    0x2168c234, 0x27a7186e, 0x00dc1cd1, 0x43ea45ba, 0x0a67cc74, 0x1e9a8194, 0x3b139b22, 0x46f82383,
    0x0e3404dd, 0x4316039c, 0x4d3a431b, 0x225ad698, 0x42d18469, 0x4f4e30da, 0x01b839a3, 0x07d48b74,
    0x14cf98e8, 0x3d350329, 0x76273644, 0x0df04707, 0x1c6809ba, 0x062c0739, 0x1a748637, 0x44b5ad30,
    0x05a308d3, 0x1e9c61b5, 0x03707347, 0x0fa916e7, 0x299f31d0, 0x7a6a0651, 0x6c4e6c88, 0x1be08e0f,
    0x38d01375, 0x0c580e71, 0x34e90c6f, 0x096b5a60, 0x0b4611a6, 0x3d38c36b, 0x06e0e68e, 0x1f522dce,
    0x533e63a1, 0x74d40ca2, 0x589cd910, 0x37c11c20, 0x71a026ea, 0x18b01ce1, 0x69d218df, 0x12d6b4bf,
    0x168c234c, 0x7a7186d5, 0x0dc1cd1c, 0x3ea45b9c, 0x267cc742, 0x69a81945, 0x3139b220, 0x6f823842,
    0x63404dd5, 0x316039c2, 0x53a431be, 0x25ad697e, 0x2d184699, 0x74e30daa, 0x1b839a38, 0x7d48b737,
    0x4cf98e85, 0x5350328b, 0x62736440, 0x5f047086, 0x46809baa, 0x62c07386, 0x2748637d, 0x4b5ad2fc,
    0x5a308d32, 0x69c61b55, 0x37073471, 0x7a916e6e, 0x19f31d0a, 0x26a06518, 0x44e6c880, 0x3e08e10d,
    0x0d013754, 0x4580e70f, 0x4e90c6fb, 0x16b5a5f8, 0x34611a64, 0x538c36ab, 0x6e0e68e2, 0x7522dcdd,
    0x33e63a14, 0x4d40ca30, 0x09cd9101, 0x7c11c21b, 0x1a026ea8, 0x0b01ce1f, 0x1d218df7, 0x2d6b4bef,
    0x68c234c9, 0x27186d57, 0x5c1cd1c4, 0x6a45b9bb, 0x67cc7429, 0x1a819460, 0x139b2202, 0x78238438,
    0x3404dd51, 0x16039c3d, 0x3a431bef, 0x5ad697dd, 0x51846992, 0x4e30daaf, 0x3839a389, 0x548b7377,
    0x4f98e852, 0x350328c1, 0x27364404, 0x70470871, 0x6809baa2, 0x2c073879, 0x748637df, 0x35ad2fba,
    0x2308d324, 0x1c61b561, 0x70734713, 0x2916e6ef, 0x1f31d0a4, 0x6a065184, 0x4e6c8808, 0x608e10e3,
    0x50137545, 0x580e70f2, 0x690c6fbe, 0x6b5a5f75, 0x4611a648, 0x38c36ac3, 0x60e68e26, 0x522dcddf,
    0x3e63a148, 0x540ca30a, 0x1cd91010, 0x411c21c8, 0x2026ea8a, 0x301ce1e7, 0x5218df7c, 0x56b4beec,
    0x0c234c90, 0x7186d586, 0x41cd1c4c, 0x245b9bc1, 0x7cc74290, 0x28194615, 0x39b22020, 0x02384391,
    0x404dd514, 0x6039c3ce, 0x2431bef9, 0x2d697dda, 0x18469921, 0x630dab0c, 0x039a3898, 0x48b73783,
    0x798e8520, 0x50328c2b, 0x73644041, 0x04708721, 0x009baa29, 0x4073879d, 0x48637df2, 0x5ad2fbb4,
    0x308d3243, 0x461b5617, 0x07347131, 0x116e6f06, 0x731d0a40, 0x20651858, 0x66c88082, 0x08e10e42,
    0x01375452, 0x00e70f3b, 0x10c6fbe5, 0x35a5f769, 0x611a6487, 0x0c36ac2d, 0x0e68e263, 0x22dcde0b,
    0x663a1481, 0x40ca30b1, 0x4d910105, 0x11c21c84, 0x026ea8a5, 0x01ce1e75, 0x218df7ca, 0x6b4beed2,
    0x4234c90f, 0x186d5859, 0x1cd1c4c6, 0x45b9bc16, 0x4c742902, 0x01946165, 0x1b22020b, 0x23843908,
    0x04dd514a, 0x039c3cea, 0x431bef95, 0x5697dda4, 0x0469921f, 0x30dab0b2, 0x39a3898c, 0x0b73782c,
    0x18e85204, 0x0328c2cb, 0x36440417, 0x47087210, 0x09baa294, 0x073879d4, 0x0637df2a, 0x2d2fbb4a,
    0x08d3243f, 0x61b56164, 0x73471319, 0x16e6f056, 0x31d0a409, 0x06518596, 0x6c88082e, 0x0e10e420,
    0x13754528, 0x0e70f3a8, 0x0c6fbe54, 0x5a5f7695, 0x11a6487e, 0x436ac2c8, 0x668e2633, 0x2dcde0ac,
    0x63a14812, 0x0ca30b2c, 0x5910105d, 0x1c21c83f, 0x26ea8a50, 0x1ce1e750, 0x18df7ca8, 0x34beed2a,
    0x234c90fd, 0x06d5858f, 0x4d1c4c66, 0x5b9bc15a, 0x47429024, 0x19461658, 0x322020bb, 0x3843907e,
    0x4dd514a0, 0x39c3ce9f, 0x31bef951, 0x697dda52, 0x469921fb, 0x0dab0b1d, 0x1a3898cc, 0x373782b7,
    0x0e852049, 0x328c2cb0, 0x64404177, 0x708720fb, 0x1baa2941, 0x73879d3e, 0x637df2a3, 0x52fbb4a4,
    0x0d3243f6, 0x1b56163a, 0x34713198, 0x6e6f056e, 0x1d0a4093, 0x6518595f, 0x488082ef, 0x610e41f6,
    0x37545282, 0x670f3a7e, 0x46fbe546, 0x25f7694a, 0x1a6487ed, 0x36ac2c74, 0x68e26331, 0x5cde0adc,
    0x3a148127, 0x4a30b2be, 0x110105df, 0x421c83ee, 0x6ea8a504, 0x4e1e74fd, 0x0df7ca8c, 0x4beed295,
    0x34c90fda, 0x6d5858e7, 0x51c4c662, 0x39bc15b9, 0x7429024e, 0x1461657d, 0x22020bbe, 0x043907dd,
    0x5d514a08, 0x1c3ce9fc, 0x1bef9519, 0x17dda52a, 0x69921fb5, 0x5ab0b1ce, 0x23898cc5, 0x73782b73,
    0x6852049c, 0x28c2cafb, 0x4404177d, 0x08720fb9, 0x3aa29410, 0x3879d3f9, 0x37df2a33, 0x2fbb4a53,
    0x53243f6a, 0x3561639d, 0x4713198a, 0x66f056e8, 0x50a40938, 0x518595f8, 0x08082efa, 0x10e41f72,
    0x75452821, 0x70f3a7f1, 0x6fbe5466, 0x5f7694a5, 0x26487ed5, 0x6ac2c73b, 0x0e263314, 0x4de0add2,
    0x21481270, 0x230b2bf3, 0x10105df5, 0x21c83ee4, 0x6a8a5043, 0x61e74fe2, 0x5f7ca8cd, 0x3eed294a,
    0x4c90fdaa, 0x55858e78, 0x1c4c6628, 0x1bc15ba5, 0x429024e0, 0x461657e6, 0x2020bbea, 0x43907dc8,
    0x5514a087, 0x43ce9fc5, 0x3ef9519b, 0x7dda5295, 0x1921fb54, 0x2b0b1cf2, 0x3898cc51, 0x3782b749,
    0x052049c1, 0x0c2cafcd, 0x404177d4, 0x0720fb90, 0x2a29410f, 0x079d3f8c, 0x7df2a336, 0x7bb4a52c,
    0x3243f6a8, 0x561639e4, 0x713198a2, 0x6f056e91, 0x0a409382, 0x18595f9b, 0x0082efa9, 0x0e41f71f,
    0x5452821e, 0x0f3a7f19, 0x7be5466c, 0x77694a59, 0x6487ed51, 0x2c2c73c8, 0x62633145, 0x5e0add22,
    0x14812704, 0x30b2bf36, 0x0105df53, 0x1c83ee3e, 0x28a5043c, 0x1e74fe32, 0x77ca8cd9, 0x6ed294b3,
    0x490fdaa2, 0x5858e791, 0x44c6628b, 0x3c15ba45, 0x29024e08, 0x61657e6b, 0x020bbea6, 0x3907dc7c,
    0x514a0879, 0x3ce9fc63, 0x6f9519b3, 0x5da52967, 0x121fb544, 0x30b1cf25, 0x098cc517, 0x782b748b,
    0x52049c11, 0x42cafcd6, 0x04177d4c, 0x720fb8f8, 0x229410f3, 0x79d3f8c6, 0x5f2a3367, 0x3b4a52cf,
    0x243f6a88, 0x61639e4a, 0x13198a2e, 0x7056e918, 0x24093822, 0x0595f9ae, 0x082efa98, 0x641f71f0,
    0x452821e6, 0x73a7f18e, 0x3e5466cf, 0x7694a59f, 0x487ed511, 0x42c73c94, 0x2633145c, 0x60add231,
    0x48127044, 0x0b2bf35d, 0x105df531, 0x483ee3df, 0x0a5043cc, 0x674fe31e, 0x7ca8cd9e, 0x6d294b40,
    0x10fdaa22, 0x058e792a, 0x4c6628b8, 0x415ba463, 0x1024e088, 0x1657e6ba, 0x20bbea63, 0x107dc7bd,
    0x14a08798, 0x4e9fc63d, 0x79519b3c, 0x5a529681, 0x21fb5444, 0x0b1cf255, 0x18cc5170, 0x02b748c8,
    0x2049c111, 0x2cafcd74, 0x4177d4c7, 0x20fb8f79, 0x29410f31, 0x1d3f8c79, 0x72a33679, 0x34a52d03,
    0x43f6a888, 0x1639e4aa, 0x3198a2e0, 0x056e9191, 0x40938222, 0x595f9ae7, 0x02efa98e, 0x41f71ef2,
    0x52821e63, 0x3a7f18f0, 0x65466cf3, 0x694a5a07, 0x07ed5110, 0x2c73c954, 0x633145c0, 0x0add2322,
    0x01270445, 0x32bf35cf, 0x05df531d, 0x03ee3de4, 0x25043cc7, 0x74fe31e0, 0x4a8cd9e6, 0x5294b410,
    0x0fdaa221, 0x58e792a8, 0x46628b80, 0x15ba4644, 0x024e088a, 0x657e6b9f, 0x0bbea63b, 0x07dc7bc7,
    0x4a08798e, 0x69fc63c2, 0x1519b3cd, 0x25296822, 0x1fb54442, 0x31cf2550, 0x0cc51701, 0x2b748c88,
    0x049c1114, 0x4afcd73e, 0x177d4c76, 0x0fb8f78e, 0x1410f31c, 0x53f8c786, 0x2a33679a, 0x4a52d045,
    0x3f6a8885, 0x639e4a9e, 0x198a2e03, 0x56e91910, 0x09382229, 0x15f9ae7b, 0x2efa98ec, 0x1f71ef1c,
    0x2821e638, 0x27f18f0d, 0x5466cf34, 0x14a5a08a, 0x7ed5110b, 0x473c953c, 0x33145c06, 0x2dd23220,
    0x12704453, 0x2bf35cf5, 0x5df531d8, 0x3ee3de37, 0x5043cc71, 0x4fe31e18, 0x28cd9e69, 0x294b4113,
    0x7daa2216, 0x0e792a79, 0x6628b80d, 0x5ba4643e, 0x24e088a6, 0x57e6b9ea, 0x3bea63b1, 0x7dc7bc6e,
    0x208798e3, 0x1fc63c31, 0x519b3cd3, 0x52968225, 0x7b54442d, 0x1cf254f3, 0x4c51701b, 0x3748c87c,
    0x49c1114c, 0x2fcd73d4, 0x77d4c762, 0x7b8f78de, 0x410f31c6, 0x3f8c7862, 0x233679a7, 0x252d044b,
    0x76a8885a, 0x39e4a9e8, 0x18a2e037, 0x6e9190f9, 0x13822299, 0x5f9ae7a7, 0x6fa98ec4, 0x771ef1bd,
    0x021e638d, 0x7f18f0c4, 0x466cf34e, 0x4a5a0896, 0x6d5110b4, 0x73c953d2, 0x3145c06e, 0x5d2321f4,
    0x27044533, 0x3f35cf4e, 0x5f531d89, 0x6e3de37b, 0x043cc71a, 0x7e31e18a, 0x0cd9e69d, 0x14b4112d,
    0x5aa22168, 0x6792a7a6, 0x628b80dc, 0x3a4643e9, 0x4e088a67, 0x7e6b9e9a, 0x3ea63b13, 0x5c7bc6f7,
    0x08798e34, 0x7c63c315, 0x19b3cd3a, 0x2968225b, 0x354442d1, 0x4f254f4d, 0x451701b8, 0x748c87d3,
    0x1c1114cf, 0x7cd73d34, 0x7d4c7627, 0x38f78def, 0x10f31c68, 0x78c7862b, 0x33679a74, 0x52d044b5,
    0x6a8885a3, 0x1e4a9e9b, 0x0a2e0370, 0x69190fa9, 0x3822299f, 0x79ae7a69, 0x7a98ec4e, 0x71ef1bdf,
    0x21e638d0, 0x718f0c57, 0x66cf34e9, 0x25a0896a, 0x55110b46, 0x3c953d37, 0x145c06e0, 0x52321f53,
    0x7044533e, 0x735cf4d3, 0x7531d89c, 0x63de37c0, 0x43cc71a0, 0x631e18af, 0x4d9e69d2, 0x4b4112d5,
];

// ---------------------------------------------------------------------------
// Constructors, one per grid point the reference reaches
// ---------------------------------------------------------------------------

/// `t` values on the grid for a 31-bit prime (POLICY §3), and the `PAIRS` and
/// `ROUNDS` that go with them.
///
/// `ROUNDS = 52` is the reference's own value at both widths. For Mersenne-31 it
/// is the pinned instance's; for BabyBear and KoalaBear it is that value copied
/// across at equal `t` and is **provisional** (POLICY §3), because
/// `_init_rounds` derives nothing.
pub const ROUNDS_T16: usize = 52;
/// See [`ROUNDS_T16`].
pub const ROUNDS_T24: usize = 52;

/// The reference's `PSQUAREHASH_MERSENNE_T16`, matched exactly.
#[must_use]
pub fn mersenne_t16<F: PrimeField64>() -> PSquareHashParams<F, 16, 4, ROUNDS_T16> {
    PSquareHashParams::from_flat("psquarehash-mersenne-t16", &MERSENNE_T16_RCONS)
}

/// The reference's `PSQUAREHASH_MERSENNE_T24`, matched exactly.
#[must_use]
pub fn mersenne_t24<F: PrimeField64>() -> PSquareHashParams<F, 24, 6, ROUNDS_T24> {
    PSquareHashParams::from_flat("psquarehash-mersenne-t24", &MERSENNE_T24_RCONS)
}

/// `PSQUAREHASH_BABYBEAR_T16`, generated from the reference's `_init_cons`.
/// Round count provisional (POLICY §3).
#[must_use]
pub fn babybear_t16<F: PrimeField64>() -> PSquareHashParams<F, 16, 4, ROUNDS_T16> {
    PSquareHashParams::derived("psquarehash-babybear-t16")
}

/// `PSQUAREHASH_BABYBEAR_T24`, generated from the reference's `_init_cons`.
/// Round count provisional (POLICY §3).
#[must_use]
pub fn babybear_t24<F: PrimeField64>() -> PSquareHashParams<F, 24, 6, ROUNDS_T24> {
    PSquareHashParams::derived("psquarehash-babybear-t24")
}

/// `PSQUAREHASH_KOALABEAR_T16`, generated from the reference's `_init_cons`.
/// Round count provisional (POLICY §3).
#[must_use]
pub fn koalabear_t16<F: PrimeField64>() -> PSquareHashParams<F, 16, 4, ROUNDS_T16> {
    PSquareHashParams::derived("psquarehash-koalabear-t16")
}

/// `PSQUAREHASH_KOALABEAR_T24`, generated from the reference's `_init_cons`.
/// Round count provisional (POLICY §3).
#[must_use]
pub fn koalabear_t24<F: PrimeField64>() -> PSquareHashParams<F, 24, 6, ROUNDS_T24> {
    PSquareHashParams::derived("psquarehash-koalabear-t24")
}
