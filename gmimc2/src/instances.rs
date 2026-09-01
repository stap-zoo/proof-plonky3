//! The variants of this construction, as types.
//!
//! One alias per instance × variant, which is what POLICY §11 asks a work list
//! to be: a symbol a reader can jump to and the compiler cannot resolve to
//! nothing. **Five** of them for four grid points, not eight, and the shortfall
//! is a finding rather than an omission.
//!
//! An instance's name is the exporter's, `gmimc2-<field>-t<t>`, and it is the
//! only thing tying an implementation to its vectors; a mismatch is **silent**
//! (POLICY §3).
//!
//! # Four grid points, four absences
//!
//! Goldilocks `t=12` at `R=96, α=4`, every 31-bit prime at `t=24` with
//! `R=264, α=2`. The other four points are undefined rather than stubbed: the
//! specification's round-number tables do not reach those widths, and `R % t == 0`
//! is what forced the choice ([`params`](crate::params)).
//!
//! # `alpha = 2` admits exactly one variant
//!
//! A register would commit `y = head²` at degree 2 and leave the recurrence
//! affine — the same max degree for twice the cells — so the three 31-bit points
//! have no register axis at all, and
//! [`columns::assert_layout`](crate::columns::assert_layout) refuses one. Only
//! Goldilocks' `alpha = 4` has a split to measure. Griffin's KoalaBear note is
//! the precedent for reporting that as a result (POLICY §11).

use harness::{Absence, FieldId, GridPoint};

use crate::columns::num_cols;
use crate::params::{ROUNDS_31_T24, ROUNDS_GOLDILOCKS_T12};
use crate::vectorized::{VECTOR_LEN, VectorizedGMiMC2Air};

/// Goldilocks `t=12`, native degree four.
pub type GoldilocksT12<F> = VectorizedGMiMC2Air<F, 12, 0, ROUNDS_GOLDILOCKS_T12, 4, VECTOR_LEN>;
/// Goldilocks `t=12`, one register per S-box and degree two.
pub type GoldilocksT12Split<F> =
    VectorizedGMiMC2Air<F, 12, 1, ROUNDS_GOLDILOCKS_T12, 4, VECTOR_LEN>;

/// Mersenne-31 `t=24`, degree two; the S-box is a squaring and there is no
/// second variant.
pub type MersenneT24<F> = VectorizedGMiMC2Air<F, 24, 0, ROUNDS_31_T24, 2, VECTOR_LEN>;
/// BabyBear `t=24`, degree two.
pub type BabyBearT24<F> = VectorizedGMiMC2Air<F, 24, 0, ROUNDS_31_T24, 2, VECTOR_LEN>;
/// KoalaBear `t=24`, degree two.
pub type KoalaBearT24<F> = VectorizedGMiMC2Air<F, 24, 0, ROUNDS_31_T24, 2, VECTOR_LEN>;

/// Every grid point of POLICY §3, present or absent.
///
/// The specification's round-number subtables cover only `t=12` at the
/// Goldilocks size and `t=24` at the 31-bit size. The other widths are therefore
/// undefined by the reference, not failed parameter derivations.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "gmimc2",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "gmimc2",
        instance: Some("gmimc2-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "gmimc2",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "gmimc2",
        instance: Some("gmimc2-mersenne-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "gmimc2",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "gmimc2",
        instance: Some("gmimc2-babybear-t24"),
        field: FieldId::BabyBear,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "gmimc2",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "gmimc2",
        instance: Some("gmimc2-koalabear-t24"),
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: None,
    },
];

const _: () = {
    assert!(INSTANCES.len() == 8);

    let mut absent = 0;
    let mut i = 0;
    while i < INSTANCES.len() {
        let point = &INSTANCES[i];
        match point.absence {
            Some(Absence::UndefinedByReference) => {
                assert!(point.instance.is_none());
                assert!(point.state_width == 8 || point.state_width == 16);
                absent += 1;
            }
            None => {
                assert!(point.instance.is_some());
                assert!(point.state_width == 12 || point.state_width == 24);
            }
            _ => panic!("GMiMC2 has no other kind of absence"),
        }
        i += 1;
    }
    assert!(absent == 4);
};

const _: () = {
    // The finding, as an assertion: at alpha 2 the register would not move the
    // degree, and at alpha 4 it halves it.
    assert!(crate::air::max_constraint_degree(2, 0) == 2);
    assert!(crate::air::max_constraint_degree(4, 0) == 4);
    assert!(crate::air::max_constraint_degree(4, 1) == 2);

    // The alpha-2 rows deliberately have no doubled-width register variant.
    // Goldilocks carries both admissible rows; every 31-bit field carries one.
    assert!(num_cols::<12, 0, ROUNDS_GOLDILOCKS_T12>() == 120);
    assert!(num_cols::<12, 1, ROUNDS_GOLDILOCKS_T12>() == 216);
    assert!(num_cols::<24, 0, ROUNDS_31_T24>() == 312);
};

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::params;

    /// The grid names are the parameter constructors' names, in grid order.
    /// This is the vector join POLICY §3 calls silent when it breaks.
    #[test]
    fn grid_names_match_the_parameter_constructors() {
        let names: Vec<_> = INSTANCES
            .iter()
            .filter_map(|point| point.instance)
            .collect();
        assert_eq!(
            names,
            vec![
                params::goldilocks_t12::<Goldilocks>().name,
                params::mersenne_t24::<Mersenne31>().name,
                params::babybear_t24::<BabyBear>().name,
                params::koalabear_t24::<KoalaBear>().name,
            ]
        );
    }
}
