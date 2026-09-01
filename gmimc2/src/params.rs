//! Parameters, from the GMiMC2 oracle — the one construction here whose oracle
//! is not `../ref`.
//!
//! POLICY §4 makes `../ref` the only source of known-answer vectors. GMiMC2 is
//! the single exception, written as narrowly as the XHash coordinate one:
//! `../ref` does not implement it and is deliberately not to be extended
//! (`../gnark-hashes/gmimc2_ref.py` says so in its own header), so that file is
//! the oracle, driven from this repository by
//! [`tools/export_gmimc2_kat.py`](../../../tools/export_gmimc2_kat.py). See
//! [`vectors/README.md`](../../../gmimc2/vectors/README.md) for the boundary and
//! what the driver is allowed to be.
//!
//! What lands here is a Rust reimplementation of the oracle's own derivation,
//! locked against its export in `tests/reference.rs`. Never hand-write a
//! constant.
//!
//! # Three numbers, three provenances
//!
//! | number | source |
//! |---|---|
//! | `p`, and the capacity that selects `M_IO` | `../ref`, through the GMiMC export |
//! | `M`, `M_IO`, `rcons` | the oracle's own derivations |
//! | `R`, `alpha` | the author's cryptanalysis — nothing derives or checks them |
//!
//! The middle row is what [`GMiMC2Params::derive`] and [`m_io`] reproduce; the
//! bottom row is reported, and README's standing "all round numbers are subject
//! to confirmation" item covers it.
//!
//! # alpha is `2^k`, and is *not* a permutation of `F_p`
//!
//! `gcd(2^k, p - 1) > 1` for every odd `p`, so `x -> x^alpha` is two-to-one and
//! "the alpha-th root" is not a value. That is sound here **and only here**: in
//! an expanding-round-function Feistel the branch function is never inverted — a
//! round is undone by subtracting `F(x_0)` from the branches it was added to —
//! so `F` may be any function at all. So `alpha` is a parameter of the instance
//! rather than the field's pinned exponent, and the specification's round-number
//! table is indexed by it. This is why [`GMiMC2Params::derive`] does **not** call
//! `reference::sampler::inverse_exponent`, where its sibling `gmimc` does: the
//! panic that guards a unique root is exactly what this design opts out of.
//!
//! # Four points, and four absences
//!
//! One width per field size — Goldilocks `t=12` at `R=96, α=4`, every 31-bit
//! prime at `t=24` with `R=264, α=2`. The other four grid points are
//! [`Absence::UndefinedByReference`](harness::Absence), *not* stubbed: the
//! specification's round-number tables are indexed by `log2(q)` and `t`, the two
//! subtables in scope reach `t ∈ {8,12}` at `log2(q) ≈ 64` and `t ∈ {16,24}` at
//! `log2(q) ≈ 32`, and the author's numbers cover one width from each. A width
//! the table does not reach is an absence, not a missing derivation.
//!
//! `R % t == 0` is the specification's own requirement — after `R` rounds the
//! cyclic shift is back to the identity, which its efficient circuit needs — and
//! it is what forced those widths: `264` is not a multiple of `16`, so a `t=16`
//! instance could not have reused the `t=24` round count even if the table had
//! reached it. [`GMiMC2Params::derive`] asserts it.

use p3_field::{Dup, PrimeCharacteristicRing, PrimeField64};
use reference::sampler::ShakeModSampler;

/// Rounds at Goldilocks `t=12`, with `alpha = 4`. The author's, from
/// cryptanalysis: nothing derives it and nothing checks it.
pub const ROUNDS_GOLDILOCKS_T12: usize = 96;

/// Rounds at every 31-bit prime, `t=24`, with `alpha = 2`. The author's, from
/// cryptanalysis. One round number covers all three exponents of the
/// specification's 32-bit subtable, which is why the constant seed carries
/// `alpha`: `(p, t, R)` alone would not identify an instance.
pub const ROUNDS_31_T24: usize = 264;

/// A fully specified GMiMC2 instance.
///
/// `ROUNDS` is `R`, and `rcons` is flat — one constant per round, as in GMiMC,
/// because a round touches one branch. Where the two differ is *what the round
/// does with it*: here `x_0 += rc_r` **before** the power map, so the constant
/// stays on the branch it entered and travels with it, where GMiMC's is an S-box
/// input only. That difference is the whole reason for two crates rather than
/// one with a variant axis, and it is visible in `native.rs`.
///
/// Neither matrix is a field of this struct. `M` is the cyclic shift
/// ([`shift_source`]) and `M_IO` the specification's circulant ([`m_io`]); both
/// are re-indexings and constant folds, so there is nothing to store and nothing
/// for `p3_mds::util::mds_multiply` to do. The export carries both dense forms,
/// and `tests/reference.rs` checks each entry against the rule the code applies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GMiMC2Params<F, const WIDTH: usize, const ROUNDS: usize> {
    /// Export name — the only thing tying this to its vectors (POLICY §3), and
    /// a mismatch here is silent.
    pub name: &'static str,
    /// S-box exponent, a power of two, deliberately not a permutation of `F_p`.
    pub alpha: u64,
    /// `R` round constants, one per round, added *into* branch 0 before the
    /// power map.
    pub rcons: [F; ROUNDS],
}

/// The branch the cyclic-shift linear layer moves into position `i`.
///
/// `shift_matrix(t)` is `m[i][i + 1] = 1` for `i < t - 1` and `m[t-1][0] = 1`,
/// so the state rotates **left** by one and branch 0 — the one the S-box reads —
/// is the branch that entered at position 1. Identical to `../ref`'s GMiMC, and
/// deliberately restated here rather than shared: each crate checks its own
/// direction against its own export, which is what makes a reversed shift fail
/// with the instance's name on it.
#[must_use]
pub const fn shift_source(i: usize, width: usize) -> usize {
    if i + 1 == width { 0 } else { i + 1 }
}

/// `t` must admit the specification's `c = t/3` circulant.
///
/// Both offsets `t/3` and `t/2` have to be whole and distinct from each other
/// and from `0`, which is `t % 6 == 0` — true at both widths on this grid
/// (12 and 24) and false at, for instance, `t = 16`. Called from every
/// construction of an instance, so a mismatched width fails at
/// monomorphization rather than silently building a different matrix.
pub const fn assert_shape(width: usize) {
    assert!(
        width.is_multiple_of(6),
        "M_IO's t/3 and t/2 offsets need t divisible by 6"
    );
}

/// The specification's input/output matrix for capacity `t/3`, applied as an
/// addition program.
///
/// `M_IO = circ(1, …, 2 at t/3, …, 2 at t/2, …)` under
/// `right_circulant(first_row)`, whose row `i` is `first_row` rotated right by
/// `i` — so `M_IO[i][j] = first_row[(j - i) mod t]` and
///
/// ```text
/// out[i] = x[i] + 2*x[(i + t/3) mod t] + 2*x[(i + t/2) mod t]
/// ```
///
/// Three nonzero entries a row, two of them the constant `2`: this costs two
/// doublings and two additions per output and no cell at all, which is why it is
/// a function here rather than a stored `t × t` matrix.
///
/// **`M_IO` is not an independent choice.** The specification fixes it per
/// capacity, and the capacity is `../ref`'s own derived value — `SpongeLE`
/// derives `c = 4` at `t=12` and `c = 8` at `t=24`, exactly `t/3` at both. So
/// POLICY §8's rule for a permutation parameter that reads a sponge parameter
/// (Rescue-Prime's constant seed is the precedent) applies unchanged: the
/// reference's derived value is what we take, and it is reported beside the
/// parameters.
///
/// Generic in the algebra rather than the field so that one definition serves
/// the native oracle over `F`, the AIR over `AB::Expr` and the packed generator
/// over `F::Packing`.
///
/// # Panics
///
/// If `WIDTH` is not divisible by 6 (see [`assert_shape`]).
#[must_use]
pub fn m_io<A: PrimeCharacteristicRing + Dup, const WIDTH: usize>(
    state: &[A; WIDTH],
) -> [A; WIDTH] {
    assert_shape(WIDTH);
    core::array::from_fn(|i| {
        let third = state[(i + WIDTH / 3) % WIDTH].dup();
        let half = state[(i + WIDTH / 2) % WIDTH].dup();
        state[i].dup() + (third + half).double()
    })
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize> GMiMC2Params<F, WIDTH, ROUNDS> {
    /// Reproduce `GMiMC2Params.derive_round_constants` for this field.
    ///
    /// One SHAKE256 stream seeded with `GMiMC2(p,t,R,alpha)aff`, drawn under the
    /// reference's `sampling="mod"` — one byte more than the field's serialized
    /// width, little-endian, reduced. The oracle borrows both the sampler and
    /// the seed shape from `../ref`'s GMiMC (`GMiMC(p,t,R)aff`) and appends
    /// `alpha`, because the specification's 32-bit table shares one round number
    /// across three exponents.
    ///
    /// **That seed is gnark's choice, not the specification's**, which leaves the
    /// constant derivation open. It is followed here for cross-repository
    /// consistency, and a later spec convention is a one-line change — this line.
    ///
    /// # Panics
    ///
    /// If `WIDTH` is not divisible by 6, if `alpha` is not a power of two at
    /// least 2, or if `ROUNDS` is not a multiple of `WIDTH`.
    #[must_use]
    pub fn derive(name: &'static str, alpha: u64) -> Self {
        assert_shape(WIDTH);
        assert!(
            alpha >= 2 && alpha.is_power_of_two(),
            "GMiMC2 requires alpha = 2^k with k >= 1"
        );
        // The specification's own requirement, which `GMiMC2Params` enforces:
        // after R rounds the cyclic shift is the identity again.
        assert!(
            ROUNDS.is_multiple_of(WIDTH),
            "the specification requires R to be a multiple of t"
        );
        // Note what is *not* here: no `inverse_exponent`, and so no
        // `gcd(alpha, p - 1) == 1` check. An erf never inverts its branch
        // function, which is the whole reason alpha may be a power of two.

        let seed = format!("GMiMC2({},{WIDTH},{ROUNDS},{alpha})aff", F::ORDER_U64);
        let mut sampler = ShakeModSampler::<F>::new(seed.as_bytes());
        let rcons = core::array::from_fn(|_| sampler.next_element());

        Self { name, alpha, rcons }
    }
}

/// Goldilocks `t=12`; `R = 96`, `alpha = 4`, both the author's.
#[must_use]
pub fn goldilocks_t12<F: PrimeField64>() -> GMiMC2Params<F, 12, ROUNDS_GOLDILOCKS_T12> {
    GMiMC2Params::derive("gmimc2-goldilocks-t12", 4)
}

/// Mersenne-31 `t=24`; `R = 264`, `alpha = 2`, both the author's.
#[must_use]
pub fn mersenne_t24<F: PrimeField64>() -> GMiMC2Params<F, 24, ROUNDS_31_T24> {
    GMiMC2Params::derive("gmimc2-mersenne-t24", 2)
}

/// BabyBear `t=24`; `R = 264`, `alpha = 2`, both the author's.
#[must_use]
pub fn babybear_t24<F: PrimeField64>() -> GMiMC2Params<F, 24, ROUNDS_31_T24> {
    GMiMC2Params::derive("gmimc2-babybear-t24", 2)
}

/// KoalaBear `t=24`; `R = 264`, `alpha = 2`, both the author's.
#[must_use]
pub fn koalabear_t24<F: PrimeField64>() -> GMiMC2Params<F, 24, ROUNDS_31_T24> {
    GMiMC2Params::derive("gmimc2-koalabear-t24", 2)
}

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;

    use super::*;

    /// The rotation is a left rotation and it wraps exactly once — the same
    /// shift `../ref`'s GMiMC uses, restated because this crate's export is the
    /// oracle's and not the reference's.
    #[test]
    fn the_shift_rotates_left_and_wraps() {
        assert_eq!(shift_source(0, 12), 1);
        assert_eq!(shift_source(11, 12), 0);
        let mut i = 0;
        for _ in 0..12 {
            i = shift_source(i, 12);
        }
        assert_eq!(i, 0);
    }

    /// `M_IO` is applied at both ends of the permutation, so its being a
    /// bijection is a precondition of the word *permutation* — `π = M_IO ∘
    /// rounds ∘ M_IO` would collapse inputs before a round ran otherwise. The
    /// export's self-test proves it by an exact determinant; this is the cheap
    /// standing check that the addition program is at least injective on the
    /// basis, which a mis-indexed offset would break.
    #[test]
    fn the_io_matrix_moves_every_branch() {
        let mut basis = [Goldilocks::ZERO; 12];
        basis[0] = Goldilocks::ONE;
        let image = m_io(&basis);
        // Column 0 of M_IO: a 1 at row 0, and a 2 wherever the rotation puts it.
        assert_eq!(image[0], Goldilocks::ONE);
        assert_eq!(image[12 - 12 / 3], Goldilocks::TWO);
        assert_eq!(image[12 - 12 / 2], Goldilocks::TWO);
        assert_eq!(
            image.iter().filter(|x| **x != Goldilocks::ZERO).count(),
            3,
            "three nonzero entries a column"
        );
    }

    /// The exponent is in the seed, which is the reason it is there: the
    /// specification's 32-bit table gives one round number for three exponents,
    /// so without it two distinct instances would share a constant set.
    #[test]
    fn the_exponent_is_part_of_the_constant_seed() {
        let two = GMiMC2Params::<Goldilocks, 12, ROUNDS_GOLDILOCKS_T12>::derive("toy", 2);
        let four = goldilocks_t12::<Goldilocks>();
        assert_eq!(four.alpha, 4);
        assert_ne!(two.rcons[0], four.rcons[0]);
    }

    /// The specification's `R % t == 0`, which is what forced one width per
    /// field size: 264 is not a multiple of 16.
    #[test]
    #[should_panic(expected = "R to be a multiple of t")]
    fn a_round_count_that_is_not_a_multiple_of_t_is_refused() {
        let _ = GMiMC2Params::<Goldilocks, 12, 95>::derive("toy", 4);
    }

    /// An odd exponent is not this design: the round-number table is indexed by
    /// `2^k`, and a non-power-of-two would be a different instance whose rounds
    /// nothing here justifies.
    #[test]
    #[should_panic(expected = "alpha = 2^k")]
    fn an_exponent_that_is_not_a_power_of_two_is_refused() {
        let _ = GMiMC2Params::<Goldilocks, 12, ROUNDS_GOLDILOCKS_T12>::derive("toy", 5);
    }
}
