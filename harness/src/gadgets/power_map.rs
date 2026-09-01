//! Power maps and the optional registers that flatten them.
//!
//! A direct `x^alpha` constraint costs degree `alpha`.  For alpha 5 or 7, one
//! committed intermediate lowers the largest constraint degree to three:
//! witness `x^2` or `x^3`, constrain that witness, then compute `witness^2*x`.
//! The degree-2 form with one register is also exposed because pSquareHash
//! commits individual squarings while flattening its Feistel map.
//!
//! The register buys one degree at alpha 3 and 4 rather than two: witness `x^2`,
//! then `register*x` or `register^2`.  Whether that is worth a cell per S-box is
//! a measured question and not an argued one — GMiMC is where it is asked, being
//! the first construction here whose whole cost is a round count.
//!
//! Three registers take alpha 7 all the way to degree two, which is the floor
//! POLICY §11 states as one cell per `log2(D)` chained multiplications: at
//! `D = 2` that is one cell per multiplication, and `x^7` is three of them once
//! the output itself rides an existing expression.  The chain is the same two
//! values the one-register arms already witness, plus their product's square:
//!
//! ```text
//! r0 = x^2        pinned by  r0 == x*x
//! r1 = x^3        pinned by  r1 == r0*x
//! r2 = x^6        pinned by  r2 == r1*r1
//! x^7 = r2*x                                 <- degree two, returned unpinned
//! ```
//!
//! Three is also the minimum: with two cells the reachable exponents are
//! `{2, 4}` or `{2, 3}`, and no two of those — with or without `x` — sum to
//! seven inside a single degree-two product.
//!
//! Degree 4 is the one exponent here that is **not** a permutation of `F_p`:
//! `gcd(4, p - 1) > 1` for every odd `p`, so `x -> x^4` is two-to-one and "the
//! fourth root of y" is not a value.  That is sound in an expanding-round-function
//! Feistel and nowhere else — the branch function is never inverted there, a round
//! being undone by subtracting `F(x_0)` from the branches it was added to — which
//! is why GMiMC2 takes `alpha = 2^k`.  Nothing in this module inverts a power map
//! (that is `super::inverse_power_map`, which is where the bijection matters), so
//! the arm is safe to offer; the caller is what has to be an erf.
//!
//! The trap is that a register is a prover-chosen cell.  Using it in the final
//! expression without first asserting `register = x^2` (or `x^3`) makes the
//! power map arbitrary.  [`eval_power_map`] owns those assertions and
//! [`generate_power_map`] owns the matching witness values; [`power_map`] is
//! the cell-free native oracle used to test both.

use p3_air::AirBuilder;
use p3_field::{Dup, PrimeCharacteristicRing};

/// Evaluate the native power-map oracle.
#[inline]
pub fn power_map<E: PrimeCharacteristicRing, const DEGREE: u64>(value: E) -> E {
    match DEGREE {
        2 => value.square(),
        3 => value.cube(),
        4 => value.square().square(),
        5 => value.square().square() * value,
        7 => value.cube().square() * value,
        _ => panic!("power-map gadget supports degrees 2, 3, 4, 5, or 7"),
    }
}

/// Evaluate a power map in an AIR and pin every optional register.
///
/// Supported `(DEGREE, REGISTERS)` pairs are `(2,0)`, `(2,1)`, `(3,0)`, `(3,1)`,
/// `(4,0)`, `(4,1)`, `(5,0)`, `(5,1)`, `(7,0)`, `(7,1)`, and `(7,3)`.
///
/// `(2,0)` is here so that an AIR parameterized by `(ALPHA, REGISTERS)` has one
/// call site: it is [`power_map`] with the builder unused, and committing a
/// square's output — `(2,1)` — only makes sense when a caller needs that value to
/// be affine in the rest of a larger expression.
#[inline]
pub fn eval_power_map<AB: AirBuilder, const DEGREE: u64, const REGISTERS: usize>(
    value: AB::Expr,
    registers: &[AB::Var; REGISTERS],
    builder: &mut AB,
) -> AB::Expr {
    match (DEGREE, REGISTERS) {
        (2, 1) => {
            let register = registers[0];
            builder.assert_eq(register, value.square());
            register.into()
        }
        (2, 0) => value.square(),
        (3, 0) => value.cube(),
        (4, 0) => value.square().square(),
        (5, 0) => value.square().square() * value,
        (7, 0) => value.cube().square() * value,
        (3, 1) => {
            let register = registers[0];
            // This is the constraint that pins the witnessed x^2.  The register
            // buys one degree rather than the two it buys at 5 and 7 — 3 -> 2 —
            // which is still the difference between an AIR that fits the common
            // blowup's cheapest rate and one that does not.
            builder.assert_eq(register, value.square());
            let register: AB::Expr = register.into();
            register * value
        }
        (4, 1) => {
            let register = registers[0];
            // This is the constraint that pins the witnessed x^2.  Unlike the
            // odd exponents, the register halves the degree rather than
            // thirding it: 4 -> 2, which is the floor a square can reach.
            builder.assert_eq(register, value.square());
            let register: AB::Expr = register.into();
            register.square()
        }
        (5, 1) => {
            let register = registers[0];
            // This is the constraint that pins the witnessed x^2.
            builder.assert_eq(register, value.square());
            let register: AB::Expr = register.into();
            register.square() * value
        }
        (7, 1) => {
            let register = registers[0];
            // This is the constraint that pins the witnessed x^3.
            builder.assert_eq(register, value.cube());
            let register: AB::Expr = register.into();
            register.square() * value
        }
        (7, 3) => {
            let (square, cube, sixth) = (registers[0], registers[1], registers[2]);
            // Three assertions, one per chained multiplication, each degree
            // two: the whole point of the variant is that no expression here
            // ever multiplies two non-committed values.
            builder.assert_eq(square, value.square());
            let square: AB::Expr = square.into();
            // Pins the witnessed x^3 against the witnessed x^2, not against a
            // recomputed one — `square` is the committed cell, so this stays
            // degree two rather than becoming the degree-three `value.cube()`.
            builder.assert_eq(cube, square * value.dup());
            let cube: AB::Expr = cube.into();
            // Pins the witnessed x^6.
            builder.assert_eq(sixth, cube.square());
            let sixth: AB::Expr = sixth.into();
            // Returned unpinned: the caller's own constraint absorbs this last
            // multiplication, which is what keeps the cell count at three.
            sixth * value
        }
        _ => panic!("unsupported power-map degree/register variant"),
    }
}

/// Compute a power map and the register values consumed by
/// [`eval_power_map`].
#[inline]
pub fn generate_power_map<A: PrimeCharacteristicRing, const DEGREE: u64, const REGISTERS: usize>(
    value: A,
) -> (A, [A; REGISTERS]) {
    let mut registers = core::array::from_fn(|_| A::ZERO);
    let output = match (DEGREE, REGISTERS) {
        (2, 1) => {
            // Pinned by eval_power_map's `register = value.square()` assertion.
            let register = value.square();
            registers[0] = register.dup();
            register
        }
        (2, 0) => value.square(),
        (3, 0) => value.cube(),
        (4, 0) => value.square().square(),
        (5, 0) => value.square().square() * value,
        (7, 0) => value.cube().square() * value,
        (3, 1) => {
            // Pinned by eval_power_map's `register = value.square()` assertion.
            let register = value.square();
            registers[0] = register.dup();
            register * value
        }
        (4, 1) => {
            // Pinned by eval_power_map's `register = value.square()` assertion.
            let register = value.square();
            registers[0] = register.dup();
            register.square()
        }
        (5, 1) => {
            // Pinned by eval_power_map's `register = value.square()` assertion.
            let register = value.square();
            registers[0] = register.dup();
            register.square() * value
        }
        (7, 1) => {
            // Pinned by eval_power_map's `register = value.cube()` assertion.
            let register = value.cube();
            registers[0] = register.dup();
            register.square() * value
        }
        (7, 3) => {
            // Pinned by eval_power_map's three assertions, in this order:
            // `r0 = x*x`, `r1 = r0*x`, `r2 = r1*r1`.
            let square = value.square();
            let cube = square.dup() * value.dup();
            let sixth = cube.square();
            registers[0] = square;
            registers[1] = cube;
            registers[2] = sixth.dup();
            sixth * value
        }
        _ => panic!("unsupported power-map degree/register variant"),
    };
    (output, registers)
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

    use super::{eval_power_map, generate_power_map, power_map};

    fn check_values<F: Field, const DEGREE: u64, const REGISTERS: usize>(values: &[F]) {
        for &value in values {
            let (output, _) = generate_power_map::<_, DEGREE, REGISTERS>(value);
            assert_eq!(output, power_map::<_, DEGREE>(value));
        }
    }

    fn check_field<F>()
    where
        F: Field + PrimeField64,
        StandardUniform: Distribution<F>,
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x706f_7765_726d_6170);
        let mut values = vec![F::ZERO, F::ONE, F::from_u64(F::ORDER_U64 - 1)];
        values.extend((0..32).map(|_| rng.sample(StandardUniform)));
        check_values::<F, 2, 0>(&values);
        check_values::<F, 2, 1>(&values);
        check_values::<F, 3, 0>(&values);
        check_values::<F, 3, 1>(&values);
        check_values::<F, 4, 0>(&values);
        check_values::<F, 4, 1>(&values);
        check_values::<F, 5, 0>(&values);
        check_values::<F, 5, 1>(&values);
        check_values::<F, 7, 0>(&values);
        check_values::<F, 7, 1>(&values);
        check_values::<F, 7, 3>(&values);
    }

    /// The module's claim about degree 4, run rather than asserted: `x^4` maps
    /// `x` and `-x` to the same value in every grid field, so it is not a
    /// permutation and no caller may treat the register as an invertible S-box.
    #[test]
    fn the_degree_four_map_is_two_to_one() {
        fn check<F: Field + PrimeField64>() {
            for raw in [1, 2, 7, F::ORDER_U64 / 2] {
                let x = F::from_u64(raw);
                assert_eq!(power_map::<F, 4>(x), power_map::<F, 4>(-x));
                assert_ne!(x, -x, "p is odd, so x != -x away from zero");
            }
        }
        check::<Goldilocks>();
        check::<Mersenne31>();
        check::<BabyBear>();
        check::<KoalaBear>();
    }

    #[test]
    fn witness_and_native_oracle_agree_over_every_benchmark_field() {
        check_field::<Goldilocks>();
        check_field::<Mersenne31>();
        check_field::<BabyBear>();
        check_field::<KoalaBear>();
    }

    struct SplitFifthPowerAir;

    impl<F> BaseAir<F> for SplitFifthPowerAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB: AirBuilder> Air<AB> for SplitFifthPowerAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let output = eval_power_map::<AB, 5, 1>(local[0].into(), &[local[1]], builder);
            builder.assert_eq(output, local[2]);
        }
    }

    #[test]
    fn corrupting_a_witnessed_register_is_rejected() {
        let values = [Mersenne31::from_u32(2), Mersenne31::from_u32(3)];
        let mut rows = Vec::new();
        for value in values {
            let (output, [register]) = generate_power_map::<_, 5, 1>(value);
            rows.extend([value, register, output]);
        }
        let trace = RowMajorMatrix::new(rows, 3);
        check_constraints(&SplitFifthPowerAir, &trace, &[]);

        let mut corrupted = trace;
        corrupted.values[1] += Mersenne31::ONE;
        assert!(
            std::panic::catch_unwind(|| {
                check_constraints(&SplitFifthPowerAir, &corrupted, &[]);
            })
            .is_err()
        );
    }

    /// `x | x^2 | x^3 | x^6 | x^7`, the three-register alpha-7 chain with its
    /// output pinned by a constraint of the caller's, which is what the arm
    /// assumes when it returns the last multiplication unpinned.
    struct FlattenedSeventhPowerAir;

    impl<F> BaseAir<F> for FlattenedSeventhPowerAir {
        fn width(&self) -> usize {
            5
        }
    }

    impl<AB: AirBuilder> Air<AB> for FlattenedSeventhPowerAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let registers = [local[1], local[2], local[3]];
            let output = eval_power_map::<AB, 7, 3>(local[0].into(), &registers, builder);
            builder.assert_eq(output, local[4]);
        }
    }

    fn flattened_seventh_power_trace(values: &[Mersenne31]) -> RowMajorMatrix<Mersenne31> {
        let mut rows = Vec::new();
        for &value in values {
            let (output, [square, cube, sixth]) = generate_power_map::<_, 7, 3>(value);
            rows.extend([value, square, cube, sixth, output]);
        }
        RowMajorMatrix::new(rows, 5)
    }

    /// The claim the whole arm exists for: three cells and nothing above
    /// degree two, including the caller's own constraint on the output.
    #[test]
    fn the_three_register_seventh_power_is_degree_two() {
        use p3_air::symbolic::{AirLayout, get_max_constraint_degree};

        let air = FlattenedSeventhPowerAir;
        let layout = AirLayout::from_air::<Mersenne31>(&air);
        assert_eq!(
            get_max_constraint_degree::<Mersenne31, _>(&air, layout, 1 << 4),
            2
        );
    }

    /// Every one of the three registers is load-bearing: the whole point is
    /// that a prover cannot pick them, and an unpinned one would be free money
    /// (POLICY §9).
    #[test]
    fn corrupting_any_of_the_three_registers_is_rejected() {
        let values = [
            Mersenne31::ZERO,
            Mersenne31::ONE,
            Mersenne31::from_u32(2),
            Mersenne31::from_u32(3),
        ];
        let trace = flattened_seventh_power_trace(&values);
        check_constraints(&FlattenedSeventhPowerAir, &trace, &[]);

        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for cell in 0..5 {
            let mut corrupted = trace.clone();
            corrupted.values[cell] += Mersenne31::ONE;
            assert!(
                std::panic::catch_unwind(|| {
                    check_constraints(&FlattenedSeventhPowerAir, &corrupted, &[]);
                })
                .is_err(),
                "cell {cell} is dead"
            );
        }
        std::panic::set_hook(hook);
    }
}
