//! Anemoi variants and POLICY §3 grid registration.

use harness::{Absence, FieldId, GridPoint};

use crate::params::{ROUNDS_T8_T16, ROUNDS_T12};
use crate::vectorized::{VECTOR_LEN, VectorizedAnemoiAir};

/// Goldilocks `t=8`, native degree seven.
pub type GoldilocksT8<F> = VectorizedAnemoiAir<F, 8, 4, 0, ROUNDS_T8_T16, 7, VECTOR_LEN>;
/// Goldilocks `t=8`, one register per Flystel and degree three.
pub type GoldilocksT8Split<F> = VectorizedAnemoiAir<F, 8, 4, 1, ROUNDS_T8_T16, 7, VECTOR_LEN>;
/// Extra reference width `t=10`, native degree seven.
pub type GoldilocksT10<F> = VectorizedAnemoiAir<F, 10, 5, 0, ROUNDS_T8_T16, 7, VECTOR_LEN>;
/// Extra reference width `t=10`, split to degree three.
pub type GoldilocksT10Split<F> = VectorizedAnemoiAir<F, 10, 5, 1, ROUNDS_T8_T16, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, native degree seven.
pub type GoldilocksT12<F> = VectorizedAnemoiAir<F, 12, 6, 0, ROUNDS_T12, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, split to degree three.
pub type GoldilocksT12Split<F> = VectorizedAnemoiAir<F, 12, 6, 1, ROUNDS_T12, 7, VECTOR_LEN>;

/// Mersenne-31 `t=16`, native degree five.
pub type MersenneT16<F> = VectorizedAnemoiAir<F, 16, 8, 0, ROUNDS_T8_T16, 5, VECTOR_LEN>;
/// Mersenne-31 `t=16`, split to degree three.
pub type MersenneT16Split<F> = VectorizedAnemoiAir<F, 16, 8, 1, ROUNDS_T8_T16, 5, VECTOR_LEN>;
/// BabyBear `t=16`, native degree seven.
pub type BabyBearT16<F> = VectorizedAnemoiAir<F, 16, 8, 0, ROUNDS_T8_T16, 7, VECTOR_LEN>;
/// BabyBear `t=16`, split to degree three.
pub type BabyBearT16Split<F> = VectorizedAnemoiAir<F, 16, 8, 1, ROUNDS_T8_T16, 7, VECTOR_LEN>;
/// KoalaBear `t=16`; alpha three needs no register variant.
pub type KoalaBearT16<F> = VectorizedAnemoiAir<F, 16, 8, 0, ROUNDS_T8_T16, 3, VECTOR_LEN>;

/// Every comparison-grid point, including explicit derivation absences.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "anemoi",
        instance: Some("anemoi-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "anemoi",
        instance: Some("anemoi-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "anemoi",
        instance: Some("anemoi-mersenne-t16"),
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "anemoi",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "anemoi",
        instance: Some("anemoi-babybear-t16"),
        field: FieldId::BabyBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "anemoi",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "anemoi",
        instance: Some("anemoi-koalabear-t16"),
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "anemoi",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: Some(Absence::StubbedDerivation),
    },
];

const _: () = {
    assert!(INSTANCES.len() == 8);
    let mut present = 0;
    let mut absent = 0;
    let mut i = 0;
    while i < INSTANCES.len() {
        match INSTANCES[i].absence {
            None => {
                assert!(INSTANCES[i].instance.is_some());
                present += 1;
            }
            Some(Absence::StubbedDerivation) => {
                assert!(INSTANCES[i].instance.is_none());
                assert!(INSTANCES[i].state_width == 24);
                absent += 1;
            }
            _ => panic!("Anemoi has no other absence kind"),
        }
        i += 1;
    }
    assert!(present == 5);
    assert!(absent == 3);
};

const _: () = {
    assert!(crate::air::max_constraint_degree(7, 0) == 7);
    assert!(crate::air::max_constraint_degree(7, 1) == 3);
    assert!(crate::air::max_constraint_degree(5, 0) == 5);
    assert!(crate::air::max_constraint_degree(5, 1) == 3);
    assert!(crate::air::max_constraint_degree(3, 0) == 3);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_names_match_parameter_constructors() {
        use p3_baby_bear::BabyBear;
        use p3_goldilocks::Goldilocks;
        use p3_koala_bear::KoalaBear;
        use p3_mersenne_31::Mersenne31;

        let names: Vec<_> = INSTANCES
            .iter()
            .filter_map(|point| point.instance)
            .collect();
        assert_eq!(
            names,
            vec![
                crate::params::goldilocks_t8::<Goldilocks>().name,
                crate::params::goldilocks_t12::<Goldilocks>().name,
                crate::params::mersenne_t16::<Mersenne31>().name,
                crate::params::babybear_t16::<BabyBear>().name,
                crate::params::koalabear_t16::<KoalaBear>().name,
            ]
        );
    }
}
