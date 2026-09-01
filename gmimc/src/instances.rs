//! The variants of this construction, as types.
//!
//! One alias per instance × variant, which is what POLICY §11 asks a work list
//! to be: a symbol a reader can jump to and the compiler cannot resolve to
//! nothing. Eight of them — four grid points, each at `(alpha, 0)` and
//! `(alpha, 1)`.
//!
//! An instance's name is the reference variable lowercased with `_` replaced by
//! `-`, which is what the export emits; that string is the only thing tying an
//! implementation to its vectors, and a mismatch is **silent** (POLICY §3).
//!
//! # Four grid points, four absences
//!
//! Goldilocks `t=12` and every 31-bit prime at `t=24`, and nothing else:
//! `GMiMCParams._init_rounds` is a stub and POLICY §3's copy-across rule cannot
//! supply a round count at `t=8` or `t=16` (see [`params`](crate::params)). The
//! four absent points are reported, never invented.
//!
//! # Variants
//!
//! The register split alone. Every alpha on this grid admits it, KoalaBear's 3
//! included — nothing in these constraints sets a degree floor above the S-box,
//! so a register takes 3 to 2 where Griffin's Horst layer leaves it at 3. The
//! two axes POLICY §11 names beyond the power map are argued away in
//! [`air`](crate::air), which is where the derivation that makes them moot
//! lives: this layout is already one cell per round.

use harness::{Absence, FieldId, GridPoint};

use crate::columns::num_cols;
use crate::params::{ROUNDS_31_T24, ROUNDS_GOLDILOCKS_T12};
use crate::vectorized::{VECTOR_LEN, VectorizedGMiMCAir};

/// Goldilocks `t=12`, native degree seven.
pub type GoldilocksT12<F> = VectorizedGMiMCAir<F, 12, 0, ROUNDS_GOLDILOCKS_T12, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, one register per S-box and degree three.
pub type GoldilocksT12Split<F> = VectorizedGMiMCAir<F, 12, 1, ROUNDS_GOLDILOCKS_T12, 7, VECTOR_LEN>;

/// Mersenne-31 `t=24`, native degree five.
pub type MersenneT24<F> = VectorizedGMiMCAir<F, 24, 0, ROUNDS_31_T24, 5, VECTOR_LEN>;
/// Mersenne-31 `t=24`, split to degree three.
pub type MersenneT24Split<F> = VectorizedGMiMCAir<F, 24, 1, ROUNDS_31_T24, 5, VECTOR_LEN>;

/// BabyBear `t=24`, native degree seven.
pub type BabyBearT24<F> = VectorizedGMiMCAir<F, 24, 0, ROUNDS_31_T24, 7, VECTOR_LEN>;
/// BabyBear `t=24`, split to degree three.
pub type BabyBearT24Split<F> = VectorizedGMiMCAir<F, 24, 1, ROUNDS_31_T24, 7, VECTOR_LEN>;

/// KoalaBear `t=24`, native degree three.
pub type KoalaBearT24<F> = VectorizedGMiMCAir<F, 24, 0, ROUNDS_31_T24, 3, VECTOR_LEN>;
/// KoalaBear `t=24`, split to degree two — the one alpha where the register buys
/// the last bit of degree rather than three of them.
pub type KoalaBearT24Split<F> = VectorizedGMiMCAir<F, 24, 1, ROUNDS_31_T24, 3, VECTOR_LEN>;

/// Every grid point of POLICY §3, present or absent.
///
/// The four missing widths are entries rather than gaps: the reference's round
/// derivation is stubbed and the author's round counts cover only `t=12` over
/// Goldilocks and `t=24` over the 31-bit fields.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "gmimc",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "gmimc",
        instance: Some("gmimc-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "gmimc",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "gmimc",
        instance: Some("gmimc-mersenne-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "gmimc",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "gmimc",
        instance: Some("gmimc-babybear-t24"),
        field: FieldId::BabyBear,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "gmimc",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "gmimc",
        instance: Some("gmimc-koalabear-t24"),
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
            Some(Absence::StubbedDerivation) => {
                assert!(point.instance.is_none());
                assert!(point.state_width == 8 || point.state_width == 16);
                absent += 1;
            }
            None => {
                assert!(point.instance.is_some());
                assert!(point.state_width == 12 || point.state_width == 24);
            }
            _ => panic!("GMiMC has no other kind of absence"),
        }
        i += 1;
    }
    assert!(absent == 4);
};

const _: () = {
    // The register buys three degrees at alpha 7, two at 5, one at 3 — and at
    // alpha 3 it reaches 2, which no other written construction here does,
    // because nothing in these constraints sets a floor above the S-box.
    assert!(crate::air::max_constraint_degree(7, 0) == 7);
    assert!(crate::air::max_constraint_degree(5, 0) == 5);
    assert!(crate::air::max_constraint_degree(3, 0) == 3);
    assert!(crate::air::max_constraint_degree(7, 1) == 3);
    assert!(crate::air::max_constraint_degree(5, 1) == 3);
    assert!(crate::air::max_constraint_degree(3, 1) == 2);

    // One committed head per round, plus one register per round in the split
    // variant and the two boundary states. These are the cell pins from the
    // plan, checked at compile time as well as symbolically in `tests/numbers.rs`.
    assert!(num_cols::<12, 0, ROUNDS_GOLDILOCKS_T12>() == 117);
    assert!(num_cols::<12, 1, ROUNDS_GOLDILOCKS_T12>() == 210);
    assert!(num_cols::<24, 0, ROUNDS_31_T24>() == 383);
    assert!(num_cols::<24, 1, ROUNDS_31_T24>() == 718);
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
