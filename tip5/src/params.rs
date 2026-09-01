//! Parameters, from `../ref` and only from `../ref`.
//!
//! POLICY §3. Tip4, Tip4′ and Tip5 are exact Goldilocks reference instances:
//! rounds, constants and matrix are read from the reference export and replayed
//! byte-for-byte against its KATs. No 31-bit instance exists: the reference
//! rejects fields whose bit length is not 64 because the split S-box is an
//! eight-byte decomposition. Goldilocks `t=8` is also absent because naming it
//! would require choosing different sponge security parameters rather than
//! running an existing reference derivation.
//!
//! Extending `../ref` is an edit to shared property: minimal, and each one
//! reported.

use p3_field::PrimeField64;
use reference::kat::Params;

/// Tip5's fixed round count.
pub const ROUNDS: usize = 5;
/// Number of state words using the split-and-lookup S-box.
pub const SPLIT_WORDS: usize = 4;
/// Number of bytes in one split word.
pub const BYTES: usize = 8;
/// Goldilocks' 64-bit Montgomery factor, `2^64 mod p`.
pub const MONT_R: u64 = 0xffff_ffff;

const EXPORTED: Params = Params::new(include_str!("../vectors/params.json"));

/// A fully specified Tip5-family permutation instance.
#[derive(Clone, Debug)]
pub struct Tip5Params<F, const WIDTH: usize> {
    /// Reference/export name.
    pub name: &'static str,
    /// Power-map exponent used by words `SPLIT_WORDS..WIDTH`.
    pub alpha: u64,
    /// Inverse exponent, used only by tests and the reference lock.
    pub alpha_inv: u64,
    /// The byte permutation of the split-and-lookup S-box.
    pub lut: [u8; 256],
    /// Dense circulant MDS matrix, exactly as exported by the reference.
    pub m: [[F; WIDTH]; WIDTH],
    /// One round-constant row per round.
    pub rcons: [[F; WIDTH]; ROUNDS],
}

/// The lookup table's defining permutation over the integers `0..=255`.
///
/// The reference defines `L(x) = (x + 1)^3 - 1 (mod 257)`. The omitted value
/// 256 is a fixed point, so restricting the map to bytes remains a permutation.
#[must_use]
pub const fn lookup(byte: u8) -> u8 {
    let x = byte as u32 + 1;
    ((x * x * x - 1) % 257) as u8
}

fn array_2d<F: Copy, const ROWS: usize, const COLS: usize>(
    rows: Vec<Vec<F>>,
    key: &str,
) -> [[F; COLS]; ROWS] {
    assert_eq!(rows.len(), ROWS, "{key} row count");
    rows.into_iter()
        .map(|row| <[F; COLS]>::try_from(row).unwrap_or_else(|_| panic!("{key} column count")))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("row count was checked"))
}

impl<F: PrimeField64, const WIDTH: usize> Tip5Params<F, WIDTH> {
    fn from_export(name: &'static str) -> Self {
        assert_eq!(
            F::ORDER_U64,
            0xffff_ffff_0000_0001,
            "Tip5 is defined over Goldilocks"
        );
        assert!(
            matches!(WIDTH, 12 | 16),
            "exported Tip5 widths are 12 and 16"
        );
        let entry = EXPORTED.instance(name);
        assert_eq!(entry.raw("p"), F::ORDER_U64);
        assert_eq!(entry.int("t"), WIDTH as u64);
        assert_eq!(entry.int("R"), ROUNDS as u64);
        assert_eq!(entry.int("u"), SPLIT_WORDS as u64);
        assert_eq!(entry.raw("mont_R"), MONT_R);
        let alpha = entry.int("alpha");
        assert_eq!(alpha, 7);

        Self {
            name,
            alpha,
            alpha_inv: entry.int("alpha_inv"),
            lut: core::array::from_fn(|i| lookup(i as u8)),
            m: array_2d(entry.grid("M"), "M"),
            rcons: array_2d(entry.grid("rcons"), "rcons"),
        }
    }
}

/// Exact reference instance `TIP4` (the same width-16 permutation as Tip5).
#[must_use]
pub fn tip4<F: PrimeField64>() -> Tip5Params<F, 16> {
    Tip5Params::from_export("tip4")
}

/// Exact reference instance `TIP4_PRIME`, the policy-grid Goldilocks `t=12` point.
#[must_use]
pub fn tip4_prime<F: PrimeField64>() -> Tip5Params<F, 12> {
    Tip5Params::from_export("tip4-prime")
}

/// Exact reference instance `TIP5`, an extra Goldilocks `t=16` instance.
#[must_use]
pub fn tip5<F: PrimeField64>() -> Tip5Params<F, 16> {
    Tip5Params::from_export("tip5")
}
