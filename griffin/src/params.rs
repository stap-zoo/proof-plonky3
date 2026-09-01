//! Parameters, from `../ref` and only from `../ref`.
//!
//! POLICY §3. Goldilocks matches the reference **exactly** — rounds, constants,
//! matrix, byte-exact against its KATs. Never hand-write a constant; never
//! re-derive outside the reference.
//!
//! # Why there is no 31-bit constructor here
//!
//! POLICY §3's usual rule for a generated instance is to copy the
//! construction's Goldilocks round count across, because thirteen of the
//! reference's `params.py` files raise `NotImplementedError` from
//! `_init_rounds` at other primes. Griffin's is not one of the thirteen: it
//! implements the Gröbner and differential bounds of the paper's Section 5.2
//! with its 20% margin, so calling it at Mersenne-31, BabyBear or KoalaBear
//! returns a number rather than an error — 15 rounds at alpha 5 or 7, 14 at
//! alpha 3 — instead of raising the stub that would make the point absent by
//! `Absence::StubbedDerivation`.
//!
//! That a criterion *runs* at a prime is not the same as the paper's analysis
//! *covering* it. Section 5.2 is written for the primes and widths the paper
//! studies; [the round-number table](../../round_numbers_overview.md) records
//! that Griffin's own analysis reaches Goldilocks and the curve fields and no
//! further, leaving every 31-bit cell blank. Executing `_init_rounds` outside
//! that footprint produces a round count with no margin behind it and no
//! independent cryptanalysis of the exponent it was run at — a table entry a
//! reader could mistake for the paper's own, which is worse than an absent
//! one. So this project does not implement Griffin at Mersenne-31, BabyBear or
//! KoalaBear; `instances.rs` reports those six grid points absent
//! (`Absence::UndefinedForField`) rather than generating them from a
//! mechanical run this file used to carry.
//!
//! One thing that generation exposed while it still ran is worth keeping on
//! record: over the reference's own Goldilocks parameters, `_init_rounds`
//! returns **9**, not the 8 that `GRIFFIN_GOLDILOCKS_T8` and `_T12` carry. We
//! keep the reference's pinned instances at 8, because an instance the
//! reference defines keeps its own rounds (POLICY §3) — but the discrepancy is
//! the reference's to resolve, and until it does, the Goldilocks rows carry one
//! round fewer than the reference's own criterion asks for.
//!
//! # alpha is a property of the field
//!
//! The S-box is a bijection only when `gcd(alpha, p - 1) == 1`, so the exponent
//! is not a free choice per instance: 7 over Goldilocks and BabyBear, 5 over
//! Mersenne-31 (where `7 | p - 1`), 3 over KoalaBear. Those are the values
//! `../ref`'s `utils/field.py` pins per prime, and the export call site passes
//! exactly them.

use p3_field::{Algebra, PrimeField64};
use reference::sampler::{Shake128BitmaskSampler, field_order_seed, inverse_exponent};

/// Rounds of the two reference-pinned Goldilocks instances.
pub const ROUNDS_GOLDILOCKS: usize = 8;

/// A fully specified Griffin instance.
///
/// `ROUNDS` counts the round-constant rows, which is `R`: the reference samples
/// `R - 1` of them and appends a zero row, because the final round adds nothing.
/// Carrying the zero row rather than special-casing the last round is what lets
/// `air.rs` and `generation.rs` index `rcons[round]` uniformly.
#[derive(Clone, Debug)]
pub struct GriffinParams<F, const WIDTH: usize, const ROUNDS: usize> {
    /// Reference/export name — the only thing tying this to its vectors.
    pub name: &'static str,
    /// S-box exponent, with `gcd(alpha, p - 1) == 1`.
    pub alpha: u64,
    /// `alpha^{-1} mod (p - 1)`. Used by native evaluation and by the witness
    /// generator; no AIR ever exponentiates by it.
    pub alpha_inv: u64,
    /// The `(a_i, b_i)` pairs of the quadratics `G_i(l) = l^2 + a_i*l + b_i`,
    /// one per Horst word, so `WIDTH - 2` of them.
    ///
    /// A `Vec` rather than an array because stable Rust cannot write
    /// `[[F; 2]; WIDTH - 2]`; `rcons` stays an array because `ROUNDS` is a
    /// layout parameter and so is a const generic already.
    pub coeffs_g: Vec<[F; 2]>,
    /// The block-circulant mixing matrix.
    pub m: [[F; WIDTH]; WIDTH],
    /// `R` rows, the last of them zero.
    pub rcons: [[F; WIDTH]; ROUNDS],
}

/// `../ref`'s `dl_m44_84_matrix(alpha = 2)`: the Duval–Leurent `M^{8,4}_{4,4}`,
/// MDS for every prime above `2^31`.
const M4: [[u64; 4]; 4] = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]];

/// `../ref`'s `m4_to_block_circulant_matrix(t)`: `M = circ(2,1,...,1) (x) M4`.
///
/// Deliberately *not* MDS as a whole — the reference says so — but it gives full
/// diffusion in one round at `O(t)` cost, which is why Griffin and Poseidon2's
/// external layer both use this shape. This builds the dense `t x t` form, which
/// is what the export's `M` is compared against and what `mds_multiply` applies.
/// Evaluating it in `O(t)` instead is a separate arithmetization choice: it
/// changes no constraint and no cell, only prover work, and it would need a test
/// against this dense form before being used.
fn block_circulant<F: PrimeField64, const WIDTH: usize>() -> [[F; WIDTH]; WIDTH] {
    assert!(
        WIDTH.is_multiple_of(4),
        "Griffin's generic matrix needs t a multiple of 4"
    );
    core::array::from_fn(|row| {
        core::array::from_fn(|col| {
            let value = M4[row % 4][col % 4];
            // The diagonal blocks carry the circulant's leading 2.
            F::from_u64(if row / 4 == col / 4 { 2 * value } else { value })
        })
    })
}

/// `legendre_symbol(value, p) == -1`: `value` is a quadratic non-residue.
///
/// Zero returns `false` here as it does in the reference, where the symbol is
/// `0`. That case is not a formality: a zero discriminant gives `G_i` a double
/// root, and a root of `G_i` is exactly the input that collapses the Horst
/// product `x_i * G_i(l)` to zero regardless of `x_i`.
fn is_non_residue<F: PrimeField64>(value: F) -> bool {
    value.exp_u64((F::ORDER_U64 - 1) / 2) == -F::ONE
}

impl<F: PrimeField64, const WIDTH: usize, const ROUNDS: usize> GriffinParams<F, WIDTH, ROUNDS> {
    /// Reproduce `GriffinParams._init_cons` and `_init_mat` for this field.
    ///
    /// The whole derivation hangs off one SHAKE128 stream, so the *order* of the
    /// draws is as load-bearing as their count: round constants first, then the
    /// `(a, b)` rejection loop, then any resampled `b_i`. A draw taken out of
    /// order still yields plausible-looking constants, which is why
    /// `tests/reference.rs` checks every one of them against the export.
    #[must_use]
    pub fn derive(name: &'static str, alpha: u64) -> Self {
        assert!(WIDTH >= 4, "Griffin needs at least two Horst words");
        assert!(
            matches!(alpha, 3 | 5 | 7),
            "the power-map gadget covers alpha 3, 5 and 7"
        );
        // Panics unless `gcd(alpha, p - 1) == 1`, which is the condition making
        // the S-box a bijection and its witnessed root unique.
        let alpha_inv = inverse_exponent(alpha, F::ORDER_U64 - 1);
        let mut sampler = Shake128BitmaskSampler::<F>::new(&field_order_seed::<F>(b"Griffin"));

        // `R - 1` sampled rows, then the zero row the final round adds.
        let mut rcons = [[F::ZERO; WIDTH]; ROUNDS];
        for row in rcons.iter_mut().take(ROUNDS - 1) {
            for cell in row.iter_mut() {
                *cell = sampler.next_element();
            }
        }

        // `a` and `b` distinct and non-zero, redrawn until `a^2 - 4b` is a
        // non-residue — which is what makes `G_i` root-free.
        let (a_0, b_0) = loop {
            let a = sampler.next_nonzero();
            let mut b = sampler.next_nonzero();
            while a == b {
                b = sampler.next_nonzero();
            }
            if is_non_residue(a.square() - b * F::from_u64(4)) {
                break (a, b);
            }
        };

        // `coeffs_g[k]` belongs to state word `k + 2`; the reference's loop
        // variable is that word index minus one, which is where the `scale` runs
        // from 2 rather than from 3.
        let mut coeffs_g = Vec::with_capacity(WIDTH - 2);
        coeffs_g.push([a_0, b_0]);
        for i in 2..WIDTH - 1 {
            let scale = F::from_usize(i);
            let a_i = a_0 * scale;
            let mut b_i = b_0 * scale.square();
            while a_i == b_i {
                b_i = sampler.next_nonzero();
            }
            coeffs_g.push([a_i, b_i]);
        }
        debug_assert_eq!(coeffs_g.len(), WIDTH - 2);

        Self {
            name,
            alpha,
            alpha_inv,
            coeffs_g,
            m: block_circulant::<F, WIDTH>(),
            rcons,
        }
    }

    /// `G_i(l) = l^2 + a_i*l + b_i` for state word `i`, over any algebra.
    ///
    /// `i` indexes the state, so `i >= 2` and `coeffs_g[i - 2]` is its pair.
    /// That offset is the reference's; getting it wrong by one still yields a
    /// permutation, just not this one, which is why `native.rs` and `air.rs`
    /// both come through here instead of indexing themselves.
    #[inline]
    #[must_use]
    pub fn quadratic<A: Algebra<F>>(&self, i: usize, l: A) -> A {
        let [a, b] = self.coeffs_g[i - 2];
        l.dup().square() + l * a + b
    }
}

/// Exact reference instance `GRIFFIN_GOLDILOCKS_T8`.
#[must_use]
pub fn goldilocks_t8<F: PrimeField64>() -> GriffinParams<F, 8, ROUNDS_GOLDILOCKS> {
    GriffinParams::derive("griffin-goldilocks-t8", 7)
}

/// Exact reference instance `GRIFFIN_GOLDILOCKS_T12`.
#[must_use]
pub fn goldilocks_t12<F: PrimeField64>() -> GriffinParams<F, 12, ROUNDS_GOLDILOCKS> {
    GriffinParams::derive("griffin-goldilocks-t12", 7)
}
