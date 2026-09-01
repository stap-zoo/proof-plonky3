//! This construction's instances, and the assertions that keep them honest.
//!
//! One entry per grid point of POLICY §3 — Goldilocks at t = 8, 12;
//! Mersenne-31, BabyBear and KoalaBear at t = 16, 24 — with the compile-time
//! assertions POLICY §2 step 6 asks for. Nothing is added to `harness`.
//!
//! An instance's name is the reference variable lowercased with `_` replaced by
//! `-`, which is what the export script emits. That string is the only thing
//! tying this implementation to its vectors, and a mismatch is **silent** —
//! hence the coverage guard in `bench`.
//!
//! Rescue-Prime covers the grid with no absence: the reference pins both
//! Goldilocks widths, and its `_init_rounds`, `_init_cons` and `_init_mat` run
//! for every prime and width, so every 31-bit point is a real derivation.
//!
//! # Variants
//!
//! Every instance is measured at every variant it admits (POLICY §11). Here that
//! is the register split: `(alpha, 0)` at degree `alpha` and `(alpha, 1)` at
//! degree three for alpha 5 and 7. **KoalaBear's alpha 3 admits one variant,
//! because here the S-box is the degree**: the linear layer is affine and the
//! constants are additive, so alpha 3 is a degree-three AIR outright.
//!
//! The split is expensive at this construction and easy to price: every word
//! runs an S-box in every half-round, so a register per S-box doubles the
//! committed cells to divide the degree by `alpha/3`. It is the same trade at
//! every grid point, which makes this a clean read on it.
//!
//! POLICY §11's other axis — how many half-rounds sit between state commitments
//! — is `crate::full_round`, whose instances are these instances.

use harness::{FieldId, GridPoint};

use crate::full_round::vectorized::VectorizedFullRoundAir;
use crate::half_round::vectorized::{VECTOR_LEN, VectorizedRescuePrimeAir};
use crate::params::{HALF_ROUNDS, HALF_ROUNDS_GOLDILOCKS_R13, ROUNDS};

/// Goldilocks `t=8`, native degree seven.
pub type GoldilocksT8<F> = VectorizedRescuePrimeAir<F, 8, 0, HALF_ROUNDS, 7, VECTOR_LEN>;
/// Goldilocks `t=8`, one register per S-box and degree three.
pub type GoldilocksT8Split<F> = VectorizedRescuePrimeAir<F, 8, 1, HALF_ROUNDS, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, native degree seven.
pub type GoldilocksT12<F> = VectorizedRescuePrimeAir<F, 12, 0, HALF_ROUNDS, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, split to degree three.
pub type GoldilocksT12Split<F> = VectorizedRescuePrimeAir<F, 12, 1, HALF_ROUNDS, 7, VECTOR_LEN>;

/// Goldilocks `t=12` at the author's `R = 13`, native degree seven.
pub type GoldilocksT12R13<F> =
    VectorizedRescuePrimeAir<F, 12, 0, HALF_ROUNDS_GOLDILOCKS_R13, 7, VECTOR_LEN>;
/// Goldilocks `t=12` at the author's `R = 13`, split to degree three.
pub type GoldilocksT12R13Split<F> =
    VectorizedRescuePrimeAir<F, 12, 1, HALF_ROUNDS_GOLDILOCKS_R13, 7, VECTOR_LEN>;

/// Mersenne-31 `t=16`, native degree five.
pub type MersenneT16<F> = VectorizedRescuePrimeAir<F, 16, 0, HALF_ROUNDS, 5, VECTOR_LEN>;
/// Mersenne-31 `t=16`, split to degree three.
pub type MersenneT16Split<F> = VectorizedRescuePrimeAir<F, 16, 1, HALF_ROUNDS, 5, VECTOR_LEN>;
/// Mersenne-31 `t=24`, native degree five.
pub type MersenneT24<F> = VectorizedRescuePrimeAir<F, 24, 0, HALF_ROUNDS, 5, VECTOR_LEN>;
/// Mersenne-31 `t=24`, split to degree three.
pub type MersenneT24Split<F> = VectorizedRescuePrimeAir<F, 24, 1, HALF_ROUNDS, 5, VECTOR_LEN>;

/// BabyBear `t=16`, native degree seven.
pub type BabyBearT16<F> = VectorizedRescuePrimeAir<F, 16, 0, HALF_ROUNDS, 7, VECTOR_LEN>;
/// BabyBear `t=16`, split to degree three.
pub type BabyBearT16Split<F> = VectorizedRescuePrimeAir<F, 16, 1, HALF_ROUNDS, 7, VECTOR_LEN>;
/// BabyBear `t=24`, native degree seven.
pub type BabyBearT24<F> = VectorizedRescuePrimeAir<F, 24, 0, HALF_ROUNDS, 7, VECTOR_LEN>;
/// BabyBear `t=24`, split to degree three.
pub type BabyBearT24Split<F> = VectorizedRescuePrimeAir<F, 24, 1, HALF_ROUNDS, 7, VECTOR_LEN>;

/// KoalaBear `t=16`; alpha three is already degree three.
pub type KoalaBearT16<F> = VectorizedRescuePrimeAir<F, 16, 0, HALF_ROUNDS, 3, VECTOR_LEN>;
/// KoalaBear `t=24`; alpha three is already degree three.
pub type KoalaBearT24<F> = VectorizedRescuePrimeAir<F, 24, 0, HALF_ROUNDS, 3, VECTOR_LEN>;

/// The same eight grid points, arithmetized the other way (`full_round`).
///
/// One committed state per round instead of two, at the same degree — the
/// second arithmetization POLICY §11's commitment-spacing axis asks for, and
/// what the bench plan's typed reference names when it wants this one.
pub mod full {
    use super::{ROUNDS, VECTOR_LEN, VectorizedFullRoundAir};
    use crate::params::{HALF_ROUNDS, HALF_ROUNDS_GOLDILOCKS_R13};

    /// `R = 13`: the author-supplied Goldilocks round count, in whole rounds.
    const ROUNDS_R13: usize = HALF_ROUNDS_GOLDILOCKS_R13 / 2;

    /// Goldilocks `t=8`, native degree seven.
    pub type GoldilocksT8<F> = VectorizedFullRoundAir<F, 8, 0, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;
    /// Goldilocks `t=8`, one register per S-box and degree three.
    pub type GoldilocksT8Split<F> =
        VectorizedFullRoundAir<F, 8, 1, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;
    /// Goldilocks `t=12`, native degree seven.
    pub type GoldilocksT12<F> =
        VectorizedFullRoundAir<F, 12, 0, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;
    /// Goldilocks `t=12`, split to degree three.
    pub type GoldilocksT12Split<F> =
        VectorizedFullRoundAir<F, 12, 1, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;

    /// Goldilocks `t=12` at the author's `R = 13`, native degree seven.
    pub type GoldilocksT12R13<F> =
        VectorizedFullRoundAir<F, 12, 0, ROUNDS_R13, HALF_ROUNDS_GOLDILOCKS_R13, 7, VECTOR_LEN>;
    /// Goldilocks `t=12` at the author's `R = 13`, split to degree three.
    pub type GoldilocksT12R13Split<F> =
        VectorizedFullRoundAir<F, 12, 1, ROUNDS_R13, HALF_ROUNDS_GOLDILOCKS_R13, 7, VECTOR_LEN>;

    /// Mersenne-31 `t=16`, native degree five.
    pub type MersenneT16<F> = VectorizedFullRoundAir<F, 16, 0, ROUNDS, HALF_ROUNDS, 5, VECTOR_LEN>;
    /// Mersenne-31 `t=16`, split to degree three.
    pub type MersenneT16Split<F> =
        VectorizedFullRoundAir<F, 16, 1, ROUNDS, HALF_ROUNDS, 5, VECTOR_LEN>;
    /// Mersenne-31 `t=24`, native degree five.
    pub type MersenneT24<F> = VectorizedFullRoundAir<F, 24, 0, ROUNDS, HALF_ROUNDS, 5, VECTOR_LEN>;
    /// Mersenne-31 `t=24`, split to degree three.
    pub type MersenneT24Split<F> =
        VectorizedFullRoundAir<F, 24, 1, ROUNDS, HALF_ROUNDS, 5, VECTOR_LEN>;

    /// BabyBear `t=16`, native degree seven.
    pub type BabyBearT16<F> = VectorizedFullRoundAir<F, 16, 0, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;
    /// BabyBear `t=16`, split to degree three.
    pub type BabyBearT16Split<F> =
        VectorizedFullRoundAir<F, 16, 1, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;
    /// BabyBear `t=24`, native degree seven.
    pub type BabyBearT24<F> = VectorizedFullRoundAir<F, 24, 0, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;
    /// BabyBear `t=24`, split to degree three.
    pub type BabyBearT24Split<F> =
        VectorizedFullRoundAir<F, 24, 1, ROUNDS, HALF_ROUNDS, 7, VECTOR_LEN>;

    /// KoalaBear `t=16`; alpha three is already degree three.
    pub type KoalaBearT16<F> = VectorizedFullRoundAir<F, 16, 0, ROUNDS, HALF_ROUNDS, 3, VECTOR_LEN>;
    /// KoalaBear `t=24`; alpha three is already degree three.
    pub type KoalaBearT24<F> = VectorizedFullRoundAir<F, 24, 0, ROUNDS, HALF_ROUNDS, 3, VECTOR_LEN>;
}

/// Every comparison-grid point. Rescue-Prime has no absences.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    // Not a ninth grid point: the same Goldilocks t = 12 point, at the round
    // count the author's cryptanalysis supplies rather than the one the
    // reference derives. Registered so the coverage guard sees it and so
    // nothing measures it under the pinned instance's name.
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-goldilocks-t12-r13"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-mersenne-t16"),
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-mersenne-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-babybear-t16"),
        field: FieldId::BabyBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-babybear-t24"),
        field: FieldId::BabyBear,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-koalabear-t16"),
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "rescue-prime",
        instance: Some("rescue-prime-koalabear-t24"),
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: None,
    },
];

const _: () = {
    // Eight grid points, plus the one author-supplied round count that shares
    // Goldilocks t = 12 with the reference-pinned instance (POLICY §4). Nine
    // entries over eight points is the whole shape of that exception, so it is
    // asserted rather than left for a reader to count.
    assert!(INSTANCES.len() == 9);
    let mut i = 0;
    while i < INSTANCES.len() {
        // No absence anywhere: every grid point has an implementation and a name.
        assert!(INSTANCES[i].instance.is_some());
        assert!(INSTANCES[i].absence.is_none());
        i += 1;
    }
};

const _: () = {
    // The S-box is the whole degree here: alpha 3 is already at three, and one
    // register brings alpha 5 and 7 down to it.
    assert!(crate::half_round::air::max_constraint_degree(3, 0) == 3);
    assert!(crate::half_round::air::max_constraint_degree(5, 0) == 5);
    assert!(crate::half_round::air::max_constraint_degree(7, 0) == 7);
    assert!(crate::half_round::air::max_constraint_degree(5, 1) == 3);
    assert!(crate::half_round::air::max_constraint_degree(7, 1) == 3);
};

const _: () = {
    // A round is two half-rounds, and the constant rows are indexed per
    // half-round: if these ever disagree, `rcons[half_round]` is out of bounds
    // or silently short.
    assert!(HALF_ROUNDS == 2 * crate::params::ROUNDS);
};

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::params;

    /// The grid's names are the parameter constructors' names, in grid order.
    /// This is the join POLICY §3 calls silent when it breaks.
    #[test]
    fn grid_names_match_the_parameter_constructors() {
        let names: Vec<_> = INSTANCES
            .iter()
            .filter_map(|point| point.instance)
            .collect();
        assert_eq!(
            names,
            vec![
                params::goldilocks_t8::<Goldilocks>().name,
                params::goldilocks_t12::<Goldilocks>().name,
                params::goldilocks_t12_r13::<Goldilocks>().name,
                params::mersenne_t16::<Mersenne31>().name,
                params::mersenne_t24::<Mersenne31>().name,
                params::babybear_t16::<BabyBear>().name,
                params::babybear_t24::<BabyBear>().name,
                params::koalabear_t16::<KoalaBear>().name,
                params::koalabear_t24::<KoalaBear>().name,
            ]
        );
    }

    /// Each name encodes the width its type carries, which is what makes a
    /// mismatched pairing visible rather than merely wrong.
    ///
    /// A name may carry one suffix past the width — `-r13`, the author-supplied
    /// round count (POLICY §4) — because two instances share Goldilocks t = 12
    /// and a name is what tells their vectors apart. The suffix is matched
    /// exactly rather than by "anything after the width", so a typo in it is
    /// still a failure here.
    #[test]
    fn names_encode_their_width() {
        for point in INSTANCES {
            let name = point.instance.expect("Rescue-Prime has no absences");
            // The reference's own spelling, which is not always `FieldId`'s:
            // its variable is `RESCUE_PRIME_MERSENNE_T16`, so the name says
            // `mersenne` where `FieldId::name()` says `mersenne31`.
            let field = match point.field {
                FieldId::Goldilocks => "goldilocks",
                FieldId::Mersenne31 => "mersenne",
                FieldId::BabyBear => "babybear",
                FieldId::KoalaBear => "koalabear",
            };
            let stem = format!("rescue-prime-{field}-t{}", point.state_width);
            assert!(
                name == stem || name == format!("{stem}-r13"),
                "{name} does not encode t = {}",
                point.state_width
            );
        }
    }

    /// The author-supplied round count is a second instance at an existing grid
    /// point, not a ninth point.
    ///
    /// Worth its own assertion because every other guard in this file counts
    /// entries or checks names, and both stay green if the R = 13 entry ever
    /// drifts to a width or field it does not belong to.
    #[test]
    fn the_author_round_count_shares_an_existing_grid_point() {
        let r13 = INSTANCES
            .iter()
            .find(|point| point.instance == Some("rescue-prime-goldilocks-t12-r13"))
            .expect("the R = 13 instance is registered");
        let pinned = INSTANCES
            .iter()
            .find(|point| point.instance == Some("rescue-prime-goldilocks-t12"))
            .expect("the reference-pinned instance is registered");
        assert_eq!(r13.field, pinned.field);
        assert_eq!(r13.state_width, pinned.state_width);
        assert_eq!(params::goldilocks_t12_r13::<Goldilocks>().rounds(), 13);
        assert_eq!(params::goldilocks_t12::<Goldilocks>().rounds(), 8);
    }
}
