//! Parameters, from `../ref` and only from `../ref`.
//!
//! One parameter type for all four exact instances, because `../ref` has one
//! class for them: `export_small_prime_kat.py` maps both its `xhash8` and
//! `xhash16` keys onto `marvellous.hash.XHash` with one `XHashParams`. What
//! differs between the two families is the field, the width and `alpha` — every
//! one of which was already a parameter.
//!
//! | instance | field | `t` | `alpha` | B skips `1 mod 3` |
//! |---|---|---|---|---|
//! | `xhash8-goldilocks-t12` | Goldilocks | 12 | 7 | yes |
//! | `xhash12-goldilocks-t12` | Goldilocks | 12 | 7 | no |
//! | `xhash16-m31-t24` | Mersenne-31 | 24 | 5 | yes |
//! | `xhash24-m31-t24` | Mersenne-31 | 24 | 5 | no |
//!
//! All other grid points are absent. `_init_mat` is a `NotImplementedError` at
//! `t = 8` and `t = 16`; although it has a `t = 24` branch, `_init_sbox_P3`
//! refuses to construct any new field instance without supplied coordinate
//! polynomials or a modulus. POLICY §3 forbids filling either hole here.
//!
//! # The P3 layer, and the one thing the export does not say
//!
//! The reference evaluates the P3 S-box from three exported **coordinate
//! polynomials** (`cpolys`). The legacy Goldilocks export has `fmod = None`, so
//! its `X^3 - X - 1` modulus remains a checked reconstruction from the table.
//! The repaired Mersenne-31 export states `fmod = X^3 + 5` and derives its table
//! through `XHashParams::_init_sbox_P3`; this crate reads that modulus rather
//! than duplicating it. In both cases [`XHashParams::structured`] compares every
//! monomial before the three-cell P3 register basis is admitted.

use p3_field::{Algebra, PrimeCharacteristicRing, PrimeField64};
use reference::kat::Params;

/// Double rounds as reported by the reference, for every instance.
pub const ROUNDS: usize = 6;
/// The permutation executes `3R/2` F/B/P3 steps.
pub const STEPS: usize = 9;
/// Complete F/B/P3 groups.
pub const CYCLES: usize = 3;
/// Rows supplied by all four exact instances.
///
/// The schedule consumes rows `0..STEPS` and the final row at index 20. The
/// eleven rows between them are legacy surplus accepted by the reference's
/// `len(rcons) >= n_rcons` check; they remain in the exported parameters so the
/// final Python `rcons[-1]` lookup stays byte-exact.
pub const CONSTANT_ROWS: usize = 21;

/// The exact state width of `XHASH8_`/`XHASH12_`, over Goldilocks.
pub const WIDTH_GOLDILOCKS: usize = 12;
/// The exact state width of `XHASH16_`/`XHASH24_`, over Mersenne-31.
pub const WIDTH_MERSENNE31: usize = 24;

const EXPORTED_GOLDILOCKS: Params = Params::new(include_str!("../vectors/xhash8/params.json"));
const EXPORTED_MERSENNE31: Params = Params::new(include_str!("../vectors/xhash16/params.json"));

/// One monomial of an exported extension-coordinate polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinateTerm<F> {
    /// Field coefficient.
    pub coefficient: F,
    /// Exponents of the three input coordinates.
    pub exponents: [u8; 3],
}

/// A fully specified XHash instance.
#[derive(Clone, Debug)]
pub struct XHashParams<F, const W: usize, const N: usize> {
    /// Reference/export name.
    pub name: &'static str,
    /// Base- and extension-field power-map exponent.
    pub alpha: u64,
    /// `alpha^{-1} mod (p-1)`, used to witness the B step.
    pub alpha_inv: u64,
    /// Whether indices `1 mod 3` skip the inverse S-box in the B step.
    pub skip_middle: bool,
    /// The sponge capacity, reported because it is part of the constant seed.
    pub capacity: usize,
    /// The reference's circulant MDS matrix.
    pub m: [[F; W]; W],
    /// The three exact exported coordinate polynomials of the P3 S-box.
    pub cpolys: [Vec<CoordinateTerm<F>>; 3],
    /// `X^3 = reduction[0] + reduction[1]*X + reduction[2]*X^2`, the quotient
    /// the instance's `cpolys` are supposed to have been derived in.
    ///
    /// Read from `fmod` where the export supplies it; reconstructed and checked
    /// against the coordinate table for the legacy Goldilocks instances.
    pub reduction: [F; 3],
    /// `X^4` reduced, precomputed from [`Self::reduction`].
    pub reduction_x4: [F; 3],
    /// Whether `cpolys` really is `x^alpha` in `F_p[X]/(f)` for the declared
    /// `f` — checked over every monomial by [`cpolys_are_the_power_map`].
    ///
    /// `true` for all current instances. [`crate::air`] refuses the three-cell
    /// P3 layout if an export ever regresses to `false`.
    pub structured: bool,
    /// Exact exported table; rows 0..8 and the final row are consumed.
    pub rcons: [[F; W]; N],
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

/// Multiply two coordinate triples in `F[X] / (X^3 - r2 X^2 - r1 X - r0)`.
///
/// Generic in the ring so that one implementation serves the AIR over
/// `AB::Expr`, trace generation over `F` and `F::Packing`, and the parameter
/// check below over `F`.
///
/// Bilinear, which is the whole reason the split AIR variant is degree three
/// rather than degree `alpha`: a committed `x^{(alpha-1)/2}` enters this twice
/// and the input once.
#[inline]
pub fn extension_mul<F, A>(left: [A; 3], right: [A; 3], reduction: &[F; 3], x4: &[F; 3]) -> [A; 3]
where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    let [a0, a1, a2] = left;
    let [b0, b1, b2] = right;
    let d0 = a0.dup() * b0.dup();
    let d1 = a0.dup() * b1.dup() + a1.dup() * b0.dup();
    let d2 = a0 * b2.dup() + a1.dup() * b1.dup() + a2.dup() * b0;
    let d3 = a1 * b2.dup() + a2.dup() * b1;
    let d4 = a2 * b2;
    core::array::from_fn(|i| {
        d3.dup() * reduction[i].dup()
            + d4.dup() * x4[i].dup()
            + match i {
                0 => d0.dup(),
                1 => d1.dup(),
                _ => d2.dup(),
            }
    })
}

/// `X^4` reduced, given `X^3 = r0 + r1 X + r2 X^2`.
fn reduce_x4<F: PrimeCharacteristicRing>(r: &[F; 3]) -> [F; 3] {
    [
        r[2].dup() * r[0].dup(),
        r[0].dup() + r[2].dup() * r[1].dup(),
        r[1].dup() + r[2].dup() * r[2].dup(),
    ]
}

/// Does `cpolys` compute `x^alpha` in `F_p[X]/(f)` for the declared `f`?
///
/// Evaluated symbolically over the monomial basis rather than sampled: the
/// answer decides whether the AIR may use a three-cell P3 layout, and a
/// sampling check that missed one coefficient would make that layout compute a
/// different permutation than the oracle. Degree-`alpha` homogeneous forms in
/// three variables have at most 36 monomials, so the full comparison is cheap.
#[must_use]
pub fn cpolys_are_the_power_map<F: PrimeField64>(
    cpolys: &[Vec<CoordinateTerm<F>>; 3],
    reduction: &[F; 3],
    x4: &[F; 3],
    alpha: u64,
) -> bool {
    // `derived[coordinate][(a, b, c)]` from repeated squaring in the quotient,
    // with each basis vector carried as its own polynomial in three formal
    // variables. Small enough to hold densely.
    let mut power: [Vec<(F, [u8; 3])>; 3] = [
        vec![(F::ONE, [1, 0, 0])],
        vec![(F::ONE, [0, 1, 0])],
        vec![(F::ONE, [0, 0, 1])],
    ];
    for _ in 1..alpha {
        power = multiply_symbolic(&power, reduction, x4);
    }

    (0..3).all(|coordinate| {
        let mut theirs: Vec<(F, [u8; 3])> = cpolys[coordinate]
            .iter()
            .filter(|term| term.coefficient != F::ZERO)
            .map(|term| (term.coefficient, term.exponents))
            .collect();
        let mut ours: Vec<(F, [u8; 3])> = power[coordinate]
            .iter()
            .filter(|(coefficient, _)| *coefficient != F::ZERO)
            .copied()
            .collect();
        let key = |(_, e): &(F, [u8; 3])| *e;
        theirs.sort_by_key(key);
        ours.sort_by_key(key);
        theirs == ours
    })
}

/// One symbolic multiplication by the formal element `(x0, x1, x2)`.
fn multiply_symbolic<F: PrimeField64>(
    value: &[Vec<(F, [u8; 3])>; 3],
    reduction: &[F; 3],
    x4: &[F; 3],
) -> [Vec<(F, [u8; 3])>; 3] {
    let mut out: [Vec<(F, [u8; 3])>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, terms) in value.iter().enumerate() {
        for (coefficient, exponents) in terms {
            for j in 0..3 {
                // `X^i * X^j`, reduced.
                let contribution: [F; 3] = match i + j {
                    0 => [F::ONE, F::ZERO, F::ZERO],
                    1 => [F::ZERO, F::ONE, F::ZERO],
                    2 => [F::ZERO, F::ZERO, F::ONE],
                    3 => *reduction,
                    _ => *x4,
                };
                let mut raised = *exponents;
                raised[j] += 1;
                for (coordinate, scale) in contribution.into_iter().enumerate() {
                    if scale != F::ZERO {
                        push_term(&mut out[coordinate], *coefficient * scale, raised);
                    }
                }
            }
        }
    }
    out
}

fn push_term<F: PrimeField64>(terms: &mut Vec<(F, [u8; 3])>, coefficient: F, exponents: [u8; 3]) {
    if let Some(slot) = terms.iter_mut().find(|(_, e)| *e == exponents) {
        slot.0 += coefficient;
    } else {
        terms.push((coefficient, exponents));
    }
}

impl<F: PrimeField64, const W: usize, const N: usize> XHashParams<F, W, N> {
    fn from_export(
        exported: &Params,
        name: &'static str,
        skip_middle: bool,
        modulus: u64,
        alpha: u64,
        fallback_reduction: Option<[i64; 3]>,
    ) -> Self {
        assert_eq!(F::ORDER_U64, modulus, "{name} is defined over one field");
        assert_eq!(N, CONSTANT_ROWS, "the exact instances supply 21 rows");

        let entry = exported.instance(name);
        assert_eq!(entry.raw("p"), F::ORDER_U64);
        assert_eq!(entry.int("t"), W as u64);
        assert_eq!(entry.int("R"), ROUNDS as u64);
        assert_eq!(entry.int("alpha"), alpha);

        let cpolys = entry.coordinate_polynomials("cpolys").map(|polynomial| {
            polynomial
                .into_iter()
                .map(|(coefficient, exponents)| {
                    assert_eq!(
                        exponents.iter().map(|&e| u64::from(e)).sum::<u64>(),
                        alpha,
                        "{name}: the P3 layer is homogeneous of degree alpha"
                    );
                    CoordinateTerm {
                        coefficient,
                        exponents,
                    }
                })
                .collect()
        });

        let reduction = if entry.has("fmod") {
            let fmod = entry.ints("fmod");
            assert_eq!(fmod.len(), 4, "{name}: cubic fmod has four coefficients");
            assert_eq!(fmod[3], 1, "{name}: fmod is monic");
            core::array::from_fn(|i| {
                -F::from_canonical_checked(fmod[i])
                    .unwrap_or_else(|| panic!("{name}: fmod[{i}] is not canonical"))
            })
        } else {
            fallback_reduction
                .unwrap_or_else(|| panic!("{name}: the export must supply fmod"))
                .map(|coefficient| {
                    if coefficient < 0 {
                        -F::from_u64(coefficient.unsigned_abs())
                    } else {
                        F::from_u64(coefficient.unsigned_abs())
                    }
                })
        };
        let reduction_x4 = reduce_x4(&reduction);
        let structured = cpolys_are_the_power_map(&cpolys, &reduction, &reduction_x4, alpha);

        Self {
            name,
            alpha,
            alpha_inv: entry.int("alpha_inv"),
            skip_middle,
            capacity: entry.int("capacity") as usize,
            m: array_2d(entry.grid("M"), "M"),
            cpolys,
            reduction,
            reduction_x4,
            structured,
            rcons: array_2d(entry.grid("rcons"), "rcons"),
        }
    }
}

/// Goldilocks `t = 12`: the reference's `XHash*` reduce with `X^3 = X + 1`.
const GOLDILOCKS_REDUCTION: [i64; 3] = [1, 1, 0];
const GOLDILOCKS: u64 = 0xffff_ffff_0000_0001;
const MERSENNE31: u64 = 0x7fff_ffff;

/// Exact aggressive reference instance `XHASH8_GOLDILOCKS_T12`.
#[must_use]
pub fn xhash8<F: PrimeField64>() -> XHashParams<F, WIDTH_GOLDILOCKS, CONSTANT_ROWS> {
    XHashParams::from_export(
        &EXPORTED_GOLDILOCKS,
        "xhash8-goldilocks-t12",
        true,
        GOLDILOCKS,
        7,
        Some(GOLDILOCKS_REDUCTION),
    )
}

/// Exact full-S-box reference instance `XHASH12_GOLDILOCKS_T12`.
#[must_use]
pub fn xhash12<F: PrimeField64>() -> XHashParams<F, WIDTH_GOLDILOCKS, CONSTANT_ROWS> {
    XHashParams::from_export(
        &EXPORTED_GOLDILOCKS,
        "xhash12-goldilocks-t12",
        false,
        GOLDILOCKS,
        7,
        Some(GOLDILOCKS_REDUCTION),
    )
}

/// Exact aggressive reference instance `XHASH16_M31_T24`.
#[must_use]
pub fn xhash16<F: PrimeField64>() -> XHashParams<F, WIDTH_MERSENNE31, CONSTANT_ROWS> {
    XHashParams::from_export(
        &EXPORTED_MERSENNE31,
        "xhash16-m31-t24",
        true,
        MERSENNE31,
        5,
        None,
    )
}

/// Exact full-S-box reference instance `XHASH24_M31_T24`.
#[must_use]
pub fn xhash24<F: PrimeField64>() -> XHashParams<F, WIDTH_MERSENNE31, CONSTANT_ROWS> {
    XHashParams::from_export(
        &EXPORTED_MERSENNE31,
        "xhash24-m31-t24",
        false,
        MERSENNE31,
        5,
        None,
    )
}
