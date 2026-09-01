//! This construction's instances, and the assertions that keep them honest.
//!
//! One entry per grid point of POLICY §3 — Goldilocks at t = 8, 12;
//! Mersenne-31, BabyBear and KoalaBear at t = 16, 24 — plus the reference's two
//! extra Goldilocks `t=16` names, with the compile-time assertions POLICY §2
//! step 6 asks for. Nothing construction-specific is added to `harness`.
//!
//! An instance's name is the reference variable lowercased with `_` replaced by
//! `-`, which is what the export script emits. That string is the only thing
//! tying this implementation to its vectors, and a mismatch is **silent** —
//! hence the coverage guard in `bench`.
//!
//! A grid point the reference cannot derive is absent and reported
//! (`bench::Absence`), never filled by hand.

use harness::{Absence, FieldId, GridPoint};

use crate::vectorized::{VECTOR_LEN, VectorizedTip5Air};

/// Goldilocks `t=12` Tip4′, native degree seven.
pub type GoldilocksT12<F> = VectorizedTip5Air<F, 12, 8, 0, VECTOR_LEN>;
/// Goldilocks `t=12` Tip4′, one register and degree three.
pub type GoldilocksT12Split<F> = VectorizedTip5Air<F, 12, 8, 1, VECTOR_LEN>;
/// Goldilocks `t=16` Tip4/Tip5, native degree seven.
pub type GoldilocksT16<F> = VectorizedTip5Air<F, 16, 12, 0, VECTOR_LEN>;
/// Goldilocks `t=16` Tip4/Tip5, one register and degree three.
pub type GoldilocksT16Split<F> = VectorizedTip5Air<F, 16, 12, 1, VECTOR_LEN>;

/// The comparison grid plus the two exact extra-width-16 named instances.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: Some(Absence::UndefinedByReference),
    },
    GridPoint {
        construction: "tip5",
        instance: Some("tip4-prime"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "tip5",
        instance: Some("tip4"),
        field: FieldId::Goldilocks,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "tip5",
        instance: Some("tip5"),
        field: FieldId::Goldilocks,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "tip5",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
];

const _: () = {
    assert!(INSTANCES.len() == 10);
    assert!(crate::air::max_constraint_degree(0) == 7);
    assert!(crate::air::max_constraint_degree(1) == 3);
};
