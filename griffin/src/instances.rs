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
//! # The grid has six absences, and they are not stubs
//!
//! `../ref`'s own `_init_rounds` is implemented for Griffin at every prime this
//! grid asks for, so nothing here fails the way `Absence::StubbedDerivation`
//! describes. The six 31-bit points are absent anyway: the paper's Section 5.2
//! analysis backing that criterion is written for the primes it studies, not
//! for Mersenne-31, BabyBear or KoalaBear, and [the round-number table]
//! (../../round_numbers_overview.md) leaves those cells blank. Running the
//! criterion at an unstudied prime produces a number, not an analyzed
//! instance — see `params.rs` for the full account — so this project reports
//! those points absent (`Absence::UndefinedForField`) rather than measuring a
//! round count nothing vouches for.
//!
//! # Variants
//!
//! Every instance is measured at every variant it admits (POLICY §11), and here
//! that is the register split alone: `(7, 0)` at degree seven and `(7, 1)` at
//! degree three, both Goldilocks widths' only alpha.
//!
//! The two axes POLICY §11 names beyond the power map — how much of a round is
//! flattened, and how many rounds sit between state commitments — are not
//! explored here. Griffin's round is already committed at its cheapest
//! non-linear boundary (see `columns.rs`); skipping a commitment would multiply
//! the degree by three per round skipped, which is the experiment to run when
//! there are numbers to compare it against.

use harness::{Absence, FieldId, GridPoint};

use crate::params::ROUNDS_GOLDILOCKS;
use crate::vectorized::{VECTOR_LEN, VectorizedGriffinAir};

/// Goldilocks `t=8`, native degree seven.
pub type GoldilocksT8<F> = VectorizedGriffinAir<F, 8, 0, ROUNDS_GOLDILOCKS, 7, VECTOR_LEN>;
/// Goldilocks `t=8`, one register per S-box and degree three.
pub type GoldilocksT8Split<F> = VectorizedGriffinAir<F, 8, 1, ROUNDS_GOLDILOCKS, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, native degree seven.
pub type GoldilocksT12<F> = VectorizedGriffinAir<F, 12, 0, ROUNDS_GOLDILOCKS, 7, VECTOR_LEN>;
/// Goldilocks `t=12`, split to degree three.
pub type GoldilocksT12Split<F> = VectorizedGriffinAir<F, 12, 1, ROUNDS_GOLDILOCKS, 7, VECTOR_LEN>;

/// Every comparison-grid point, including the six 31-bit absences.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "griffin",
        instance: Some("griffin-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "griffin",
        instance: Some("griffin-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "griffin",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "griffin",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "griffin",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "griffin",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "griffin",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "griffin",
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
            _ => panic!("Griffin has no other absence kind"),
        }
        i += 1;
    }
    assert!(present == 2);
    assert!(absent == 6);
};

const _: () = {
    // The Horst layer's floor is three, so alpha 3 gains nothing from a register
    // and alpha 5 and 7 come all the way down to three with one.
    assert!(crate::air::max_constraint_degree(3, 0) == 3);
    assert!(crate::air::max_constraint_degree(5, 0) == 5);
    assert!(crate::air::max_constraint_degree(7, 0) == 7);
    assert!(crate::air::max_constraint_degree(5, 1) == 3);
    assert!(crate::air::max_constraint_degree(7, 1) == 3);
};

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;

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
            ]
        );
    }

    /// Each present point's name encodes the width its type carries, which is
    /// what makes a mismatched pairing visible rather than merely wrong.
    #[test]
    fn names_encode_their_width() {
        for point in INSTANCES.iter().filter(|point| point.absence.is_none()) {
            let name = point.instance.expect("a present point has a name");
            assert!(
                name.ends_with(&format!("-t{}", point.state_width)),
                "{name} does not encode t = {}",
                point.state_width
            );
            // The reference's own spelling, which is not always `FieldId`'s:
            // its variable is `GRIFFIN_MERSENNE_T16`, so the name says
            // `mersenne` where `FieldId::name()` says `mersenne31`.
            let field = match point.field {
                FieldId::Goldilocks => "goldilocks",
                FieldId::Mersenne31 => "mersenne",
                FieldId::BabyBear => "babybear",
                FieldId::KoalaBear => "koalabear",
            };
            assert!(name == format!("griffin-{field}-t{}", point.state_width));
        }
    }
}
