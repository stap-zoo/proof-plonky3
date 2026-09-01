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
//! # Two grid points, six absences
//!
//! Only the two Goldilocks points are registered. The six 31-bit points would
//! have carried the construction's Goldilocks round count copied straight
//! across with no derivation run for the new prime at all (see `params.rs`),
//! which is a stand-in rather than an analyzed instance — so they are reported
//! absent (`Absence::UndefinedForField`) instead, matching [the round-number
//! table](../../round_numbers_overview.md)'s blank 31-bit Neptune row.

use harness::{Absence, FieldId, GridPoint};
use p3_goldilocks::Goldilocks;

use crate::native::Neptune;
use crate::params::NeptuneParams;
use crate::vectorized::{VECTOR_LEN, VectorizedNeptuneAir};

/// Goldilocks, t = 8, alpha = 7.
pub type GoldilocksT8 = Neptune<Goldilocks, 8, 6, 38, 7>;
/// Goldilocks, t = 12, alpha = 7.
pub type GoldilocksT12 = Neptune<Goldilocks, 12, 6, 42, 7>;

// The measured AIR types, one per degree/register variant an instance admits
// (POLICY §11). The const parameters are
// `<F, WIDTH, EXT, HALF_EXT, INT, DEGREE, LM, PREGS, LANES>`, and
// `crate::columns::assert_layout` refuses every triple that is not below.
//
// | suffix | LM | PREGS | degree | cells per call, t = 12 |
// |---|---|---|---|---|
// | `Air` | 0 | 0 | 7 | 588 |
// | `Split` | `t/2` | 1 | 3 | 666 |
// | `Flattened` | `t/2` | 3 | 2 | 750 |

/// Measured AIR type for Goldilocks t = 8, unsplit: degree 7.
pub type GoldilocksT8Air = VectorizedNeptuneAir<Goldilocks, 8, 6, 3, 38, 7, 0, 0, VECTOR_LEN>;
/// Goldilocks t = 8 with the pair map and `x^7` split: degree 3.
pub type GoldilocksT8Split = VectorizedNeptuneAir<Goldilocks, 8, 6, 3, 38, 7, 4, 1, VECTOR_LEN>;
/// Goldilocks t = 8 with every multiplication committed: degree 2.
pub type GoldilocksT8Flattened = VectorizedNeptuneAir<Goldilocks, 8, 6, 3, 38, 7, 4, 3, VECTOR_LEN>;
/// Measured AIR type for Goldilocks t = 12, unsplit: degree 7.
pub type GoldilocksT12Air = VectorizedNeptuneAir<Goldilocks, 12, 6, 3, 42, 7, 0, 0, VECTOR_LEN>;
/// Goldilocks t = 12 with the pair map and `x^7` split: degree 3.
pub type GoldilocksT12Split = VectorizedNeptuneAir<Goldilocks, 12, 6, 3, 42, 7, 6, 1, VECTOR_LEN>;
/// Goldilocks t = 12 with every multiplication committed: degree 2.
pub type GoldilocksT12Flattened =
    VectorizedNeptuneAir<Goldilocks, 12, 6, 3, 42, 7, 6, 3, VECTOR_LEN>;

macro_rules! constructor {
    ($name:ident, $ty:ty, $field:ty, $width:expr, $ext:expr, $int:expr) => {
        #[doc = concat!("Construct [`", stringify!($ty), "`] from the reference derivation.")]
        #[must_use]
        pub fn $name() -> $ty {
            Neptune::new(NeptuneParams::<$field, $width, $ext, $int>::derive())
        }
    };
}

constructor!(goldilocks_t8, GoldilocksT8, Goldilocks, 8, 6, 38);
constructor!(goldilocks_t12, GoldilocksT12, Goldilocks, 12, 6, 42);

macro_rules! air_constructor {
    ($name:ident, $ty:ty) => {
        #[doc = concat!("Construct [`", stringify!($ty), "`] from the reference derivation.")]
        #[must_use]
        pub fn $name() -> $ty {
            <$ty>::new(NeptuneParams::derive())
        }
    };
}

// One per registered grid point: the benchmark plan names an AIR constructor
// per instance, and an instance whose AIR cannot be named is an instance that
// silently never gets measured.
air_constructor!(goldilocks_t8_air, GoldilocksT8Air);
air_constructor!(goldilocks_t8_split_air, GoldilocksT8Split);
air_constructor!(goldilocks_t8_flattened_air, GoldilocksT8Flattened);
air_constructor!(goldilocks_t12_air, GoldilocksT12Air);
air_constructor!(goldilocks_t12_split_air, GoldilocksT12Split);
air_constructor!(goldilocks_t12_flattened_air, GoldilocksT12Flattened);

/// The complete registered benchmark grid, including the six 31-bit absences.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "neptune",
        instance: Some("neptune-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "neptune",
        instance: Some("neptune-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "neptune",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "neptune",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "neptune",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "neptune",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "neptune",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "neptune",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
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
            Some(Absence::UndefinedForField) => {
                assert!(INSTANCES[i].instance.is_none());
                absent += 1;
            }
            _ => panic!("Neptune has no other absence kind"),
        }
        i += 1;
    }
    assert!(present == 2);
    assert!(absent == 6);
};

/// The variant table of POLICY §11, as a compile-time assertion.
///
/// The two register columns are not independent: cutting the power map alone
/// leaves the Lai--Massey floor of four standing, which is the whole reason
/// `assert_layout` refuses `LM = 0` with `PREGS > 0`.
const _: () = {
    assert!(crate::air::max_constraint_degree(3, 0, 0) == 4);
    assert!(crate::air::max_constraint_degree(5, 0, 0) == 5);
    assert!(crate::air::max_constraint_degree(7, 0, 0) == 7);
    assert!(crate::air::max_constraint_degree(7, 6, 1) == 3);
    assert!(crate::air::max_constraint_degree(7, 6, 3) == 2);
    // Splitting only the pair map: the internal `x^7` is what is left standing.
    assert!(crate::air::max_constraint_degree(7, 6, 0) == 7);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_encode_their_width() {
        for point in INSTANCES.iter().filter(|point| point.absence.is_none()) {
            let name = point.instance.expect("a present point has a name");
            assert!(
                name.ends_with(&format!("-t{}", point.state_width)),
                "{name} is not at t = {}",
                point.state_width
            );
        }
    }
}
