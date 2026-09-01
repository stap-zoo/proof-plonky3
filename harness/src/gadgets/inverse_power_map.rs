//! The witnessed alpha-th root, and the constraint that pins it.
//!
//! `x -> x^(1/alpha)` is the expensive direction of a power map: computing it
//! natively costs a full exponentiation by `alpha^{-1} mod (p-1)`, and writing
//! that exponent chain into an AIR would cost some sixty constraints per call.
//! Every design that uses one therefore does the same thing instead — **commit
//! the root and constrain it forwards**:
//!
//! ```text
//! witness  root                       (one prover-chosen cell)
//! assert   root^alpha == target       (degree alpha, or 3 with one register)
//! ```
//!
//! The AIR never divides, never exponentiates by anything large, and pays the
//! forward power map's degree exactly once. `generate_inverse_power_map` owns
//! the witness, `eval_inverse_power_map` owns the assertion, and
//! [`inverse_power_map`] is the cell-free native oracle both are tested
//! against.
//!
//! # Cost
//!
//! One committed cell for the root, plus `REGISTERS` cells for the forward
//! power map it is pinned with, plus one constraint. Degree is
//! [`crate::gadgets::power_map`]'s: `alpha` unsplit, three with one register for alpha 5
//! or 7.
//!
//! # The two traps
//!
//! **A witnessed root is free money until `root^alpha == target` is asserted**
//! (POLICY §9). It is one `assert_eq`, it is easy to forget when the root
//! happens to be used somewhere that "looks constrained", and no known-answer
//! test can see the omission: an honest generator writes the right value, so
//! every KAT still passes. The negative test in this module's tests, and each
//! caller's own, are what check the claim.
//!
//! **The assertion pins a root, not *the* root, unless `gcd(alpha, p-1) == 1`.**
//! When the gcd is `g > 1` the map `x -> x^alpha` is `g`-to-one, so `g`
//! different cells satisfy the same constraint and a prover picks whichever it
//! likes. `reference::sampler::inverse_exponent` panics in exactly that case, which is why
//! an instance's alpha is a property of the field: Mersenne-31 takes 5 where
//! Goldilocks takes 7, because `7 | p - 1` there.
//!
//! # Returning the power, not the target
//!
//! [`eval_inverse_power_map`] returns `root^alpha` — the expression it just
//! asserted equal to `target`. A caller that continues from that value keeps
//! the power map's degree in whatever it builds next; a caller that continues
//! from `target` instead is equally sound, because of the assertion, and often
//! cheaper. Griffin wants neither: it continues from the root itself.

use p3_air::AirBuilder;
use p3_field::{Dup, PrimeCharacteristicRing};

use crate::gadgets::power_map::{eval_power_map, generate_power_map};

/// Evaluate the native inverse-power oracle: the `alpha`-th root of `value`.
///
/// `inverse_exponent` is `alpha^{-1} mod (p-1)`, which
/// `reference::sampler::inverse_exponent` derives. This is the direction the reference
/// evaluates and the direction no AIR takes.
#[inline]
pub fn inverse_power_map<E: PrimeCharacteristicRing>(value: E, inverse_exponent: u64) -> E {
    value.exp_u64(inverse_exponent)
}

/// Pin a witnessed `alpha`-th root, and return the forward power it was pinned
/// with.
///
/// `root` is the prover-chosen cell (or an expression built from cells, as in
/// Anemoi's `y - v`); `target` is the value it must be the root of. The
/// registers are the forward power map's, pinned by [`eval_power_map`].
#[inline]
pub fn eval_inverse_power_map<AB: AirBuilder, const DEGREE: u64, const REGISTERS: usize>(
    root: AB::Expr,
    target: AB::Expr,
    registers: &[AB::Var; REGISTERS],
    builder: &mut AB,
) -> AB::Expr {
    let power = eval_power_map::<AB, DEGREE, REGISTERS>(root, registers, builder);
    // This is the constraint that pins the witnessed root; without it the cell
    // is arbitrary and every known-answer test still passes.
    builder.assert_eq(power.dup(), target);
    power
}

/// Compute a witnessed `alpha`-th root and the registers
/// [`eval_inverse_power_map`] consumes.
///
/// Returns `(root, registers)`. The root is pinned by that function's
/// `root^alpha == target` assertion, and the registers by the [`eval_power_map`]
/// call inside it.
#[inline]
pub fn generate_inverse_power_map<
    A: PrimeCharacteristicRing,
    const DEGREE: u64,
    const REGISTERS: usize,
>(
    value: A,
    inverse_exponent: u64,
) -> (A, [A; REGISTERS]) {
    let root = inverse_power_map(value, inverse_exponent);
    let (_, registers) = generate_power_map::<A, DEGREE, REGISTERS>(root.dup());
    (root, registers)
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess, check_constraints};
    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::distr::{Distribution, StandardUniform};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use reference::sampler::inverse_exponent;

    use super::{eval_inverse_power_map, generate_inverse_power_map, inverse_power_map};
    use crate::gadgets::power_map::power_map;

    /// Test values: the two elements every power map fixes, the largest
    /// canonical element, and a random sample. `0` matters most — it is the one
    /// input whose root is itself for every alpha, so a gadget that returned a
    /// constant zero would pass a careless test.
    fn values<F: Field + PrimeField64>() -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x726f_6f74_6d61_7000);
        let mut values = vec![F::ZERO, F::ONE, F::from_u64(F::ORDER_U64 - 1)];
        values.extend((0..32).map(|_| rng.sample(StandardUniform)));
        values
    }

    fn check_witness<F, const DEGREE: u64, const REGISTERS: usize>()
    where
        F: Field + PrimeField64,
        StandardUniform: Distribution<F>,
    {
        let alpha_inv = inverse_exponent(DEGREE, F::ORDER_U64 - 1);
        for value in values::<F>() {
            let (root, _) = generate_inverse_power_map::<F, DEGREE, REGISTERS>(value, alpha_inv);
            assert_eq!(root, inverse_power_map(value, alpha_inv), "oracle");
            // The property the AIR's single constraint actually checks.
            assert_eq!(power_map::<F, DEGREE>(root), value, "root^alpha == target");
        }
    }

    #[test]
    fn the_witness_is_the_root_the_constraint_will_check() {
        check_witness::<Goldilocks, 7, 0>();
        check_witness::<Goldilocks, 7, 1>();
        check_witness::<Mersenne31, 5, 0>();
        check_witness::<Mersenne31, 5, 1>();
        check_witness::<BabyBear, 7, 0>();
        check_witness::<BabyBear, 7, 1>();
        check_witness::<KoalaBear, 3, 0>();
        check_witness::<KoalaBear, 5, 1>();
        check_witness::<KoalaBear, 7, 1>();
    }

    /// `x -> x^alpha` is a bijection for every grid alpha, so distinct inputs
    /// have distinct roots. A gadget that quietly collapsed to a constant, or
    /// that used the wrong exponent, would fail here even though `0` and `1`
    /// round-trip correctly.
    #[test]
    fn distinct_targets_have_distinct_roots() {
        let alpha_inv = inverse_exponent(5, Mersenne31::ORDER_U64 - 1);
        let mut roots: Vec<Mersenne31> = (0u64..64)
            .map(|x| {
                generate_inverse_power_map::<Mersenne31, 5, 0>(Mersenne31::from_u64(x), alpha_inv).0
            })
            .collect();
        roots.sort_by_key(|x| x.as_canonical_u64());
        roots.dedup();
        assert_eq!(roots.len(), 64);
    }

    /// `root | register | target`: the smallest AIR that uses the gadget the
    /// way a construction does — the root and register are committed cells and
    /// the target is a cell the caller has constrained elsewhere.
    struct RootAir;

    impl<F> BaseAir<F> for RootAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB: AirBuilder> Air<AB> for RootAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            eval_inverse_power_map::<AB, 5, 1>(
                local[0].into(),
                local[2].into(),
                &[local[1]],
                builder,
            );
        }
    }

    fn trace(targets: &[Mersenne31]) -> RowMajorMatrix<Mersenne31> {
        let alpha_inv = inverse_exponent(5, Mersenne31::ORDER_U64 - 1);
        let mut rows = Vec::new();
        for &target in targets {
            let (root, [register]) =
                generate_inverse_power_map::<Mersenne31, 5, 1>(target, alpha_inv);
            rows.extend([root, register, target]);
        }
        RowMajorMatrix::new(rows, 3)
    }

    #[test]
    fn an_honest_trace_satisfies_the_constraint() {
        let targets = [Mersenne31::ZERO, Mersenne31::ONE, Mersenne31::from_u32(2)];
        check_constraints(&RootAir, &trace(&targets), &[]);
    }

    /// POLICY §9's per-gadget negative test. Every cell of the gadget is
    /// corrupted in turn, including the row whose target is zero — the case
    /// where a `x^2 * y == V * x`-shaped constraint would have let the witness
    /// float free.
    #[test]
    fn corrupting_any_cell_is_rejected() {
        let targets = [Mersenne31::ZERO, Mersenne31::from_u32(2)];
        let honest = trace(&targets);

        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for cell in 0..honest.values.len() {
            let mut corrupted = honest.clone();
            corrupted.values[cell] += Mersenne31::ONE;
            assert!(
                std::panic::catch_unwind(|| check_constraints(&RootAir, &corrupted, &[])).is_err(),
                "cell {cell} is unconstrained"
            );
        }
        std::panic::set_hook(hook);
    }
}
