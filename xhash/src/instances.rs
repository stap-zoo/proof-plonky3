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
//! A grid point the reference cannot derive is absent and reported
//! (`bench::Absence`), never filled by hand.
//!
//! # Two grid rows, one crate
//!
//! `../ref` has one `XHash` class for all four instances and this crate follows
//! it, but the **comparison grid keeps two rows**: `xhash8` at Goldilocks
//! `t = 12` and `xhash16` at Mersenne-31 `t = 24`. The grid compares
//! `(construction, field, width)` points (POLICY §3), and collapsing the two
//! names would silently rewrite every measured row's construction column. So
//! this file exports two `GridPoint` tables, unchanged from when they lived in
//! two crates, and the merge stays what it is: a code-level fact, not a
//! reporting change.

use harness::{Absence, FieldId, GridPoint};

use crate::params::{CONSTANT_ROWS, CYCLES, ROUNDS, WIDTH_GOLDILOCKS, WIDTH_MERSENNE31};
use crate::vectorized::{VECTOR_LEN, VectorizedXHashAir};

/// Aggressive XHash8, direct degree-seven constraints.
pub type XHash8<F> =
    VectorizedXHashAir<F, WIDTH_GOLDILOCKS, 8, 0, 1, CYCLES, CONSTANT_ROWS, 7, VECTOR_LEN>;
/// Aggressive XHash8, one committed extension cube per P3 triple.
pub type XHash8Split<F> =
    VectorizedXHashAir<F, WIDTH_GOLDILOCKS, 8, 1, 1, CYCLES, CONSTANT_ROWS, 7, VECTOR_LEN>;
/// Full-S-box XHash12, direct degree-seven constraints.
pub type XHash12<F> =
    VectorizedXHashAir<F, WIDTH_GOLDILOCKS, 12, 0, 1, CYCLES, CONSTANT_ROWS, 7, VECTOR_LEN>;
/// Full-S-box XHash12, split to degree three.
pub type XHash12Split<F> =
    VectorizedXHashAir<F, WIDTH_GOLDILOCKS, 12, 1, 1, CYCLES, CONSTANT_ROWS, 7, VECTOR_LEN>;

/// Aggressive XHash16, direct degree-five constraints.
pub type XHash16<F> =
    VectorizedXHashAir<F, WIDTH_MERSENNE31, 16, 0, 1, CYCLES, CONSTANT_ROWS, 5, VECTOR_LEN>;
/// Aggressive XHash16, split to degree three with one extension square per triple.
pub type XHash16Split<F> =
    VectorizedXHashAir<F, WIDTH_MERSENNE31, 16, 1, 1, CYCLES, CONSTANT_ROWS, 5, VECTOR_LEN>;
/// Full-S-box XHash24, direct degree-five constraints.
pub type XHash24<F> =
    VectorizedXHashAir<F, WIDTH_MERSENNE31, 24, 0, 1, CYCLES, CONSTANT_ROWS, 5, VECTOR_LEN>;
/// Full-S-box XHash24, split to degree three with one extension square per triple.
pub type XHash24Split<F> =
    VectorizedXHashAir<F, WIDTH_MERSENNE31, 24, 1, 1, CYCLES, CONSTANT_ROWS, 5, VECTOR_LEN>;

/// POLICY §3's grid for the `XHASH8_`/`XHASH12_` row: both exact variants at the
/// one defined point, everything else absent and reported.
pub const GOLDILOCKS_INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash8",
        instance: Some("xhash8-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "xhash8",
        instance: Some("xhash12-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash8",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: Some(Absence::UndefinedByReference),
    },
];

/// POLICY §3's grid for the `XHASH16_`/`XHASH24_` row.
pub const MERSENNE31_INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash16",
        instance: Some("xhash16-m31-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "xhash16",
        instance: Some("xhash24-m31-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "xhash16",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: Some(Absence::UndefinedByReference),
    },
];

const _: () = {
    assert!(GOLDILOCKS_INSTANCES.len() == 9);
    assert!(MERSENNE31_INSTANCES.len() == 9);
    assert!(ROUNDS == 2 * CYCLES);
    assert!(crate::air::max_constraint_degree(7, 0) == 7);
    assert!(crate::air::max_constraint_degree(7, 1) == 3);
    assert!(crate::air::max_constraint_degree(5, 0) == 5);
    assert!(crate::air::max_constraint_degree(5, 1) == 3);
};

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::params;

    #[test]
    fn exact_grid_names_match_the_parameter_constructors() {
        assert_eq!(
            GOLDILOCKS_INSTANCES[1].instance,
            Some(params::xhash8::<Goldilocks>().name)
        );
        assert_eq!(
            GOLDILOCKS_INSTANCES[2].instance,
            Some(params::xhash12::<Goldilocks>().name)
        );
        assert_eq!(
            MERSENNE31_INSTANCES[3].instance,
            Some(params::xhash16::<Mersenne31>().name)
        );
        assert_eq!(
            MERSENNE31_INSTANCES[4].instance,
            Some(params::xhash24::<Mersenne31>().name)
        );
    }
}
