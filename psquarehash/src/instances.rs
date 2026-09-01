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
//! # Six points, two absences
//!
//! Goldilocks at t = 8 and t = 12 do not exist and cannot be generated.
//! `pSquareHashParams._init_rounds` raises `NotImplementedError`, and the
//! reference pins no Goldilocks instance, so `R` has no value there — not a
//! provisional one either, because POLICY §3's copy-across rule copies *from*
//! Goldilocks and there is nothing to copy. Both are
//! [`Absence::StubbedDerivation`].
//!
//! The other six exist: Mersenne-31 matched exactly, BabyBear and KoalaBear
//! generated from `_init_cons` with `R = 52` carried over from Mersenne-31 at the
//! same `t` — **provisional**, and a reported column, because every cost number
//! scales with it.
//!
//! # Six variants per instance
//!
//! POLICY §11 measures every variant an instance admits, on all three of its
//! axes: how much of a round is flattened, how many rounds separate two state
//! commitments, and how much of one Feistel a commitment covers. The aliases
//! below are those variants — the first five
//! [`full_commit`](crate::full_commit)'s, the last
//! [`half_commit`](crate::half_commit)'s, in one shared `instances.rs` for both
//! arithmetizations as in `rescue-prime`. See
//! [`full_commit::columns`](crate::full_commit::columns) and
//! [`half_commit::columns`](crate::half_commit::columns) for the cell counts,
//! and the two `air` modules for the degree accounting.
//!
//! | alias | `REGISTERS` / `_LAST` | `SPAN` | max degree | cells / round |
//! |---|---|---|---|---|
//! | `Flattened*`   | 2 / 2 | 0 | 2  | `t/2`  |
//! | `StateOnly*`   | 0 / 0 | 0 | 4  | `t/2`  |
//! | `OneRegister*` | 1 / 1 | 0 | 2  | `3t/4` |
//! | `SpacedSplit*` | 0 / 1 | 1 | 8  | `3t/8` |
//! | `Spaced*`      | 0 / 0 | 1 | 16 | `t/4`  |
//! | `HalfCommit*`  | — | — | 4  | `t/4`  |
//!
//! `HalfCommit*` has none of those parameters, which is the point of it sitting
//! on its own axis: it commits the same one cell per Feistel in every round, so
//! there is no flattening choice left to make and nothing to space. Read its row
//! against `Spaced*` — identical width, degree 4 against 16 — and against
//! `Flattened*`, which is twice as wide at degree 2.
//!
//! Instantiate one against an instance's parameters:
//!
//! ```
//! use psquarehash::instances::Flattened16;
//! use psquarehash::params;
//! use p3_mersenne_31::Mersenne31;
//!
//! let air = Flattened16::from_params(&params::mersenne_t16::<Mersenne31>());
//! ```

use harness::{Absence, FieldId, GridPoint};

use crate::full_commit::vectorized::{VECTOR_LEN, VectorizedPSquareHashAir};
use crate::half_commit::vectorized::VectorizedHalfCommitAir;
use crate::params::{ROUNDS_T16, ROUNDS_T24};

// ---------------------------------------------------------------------------
// The variants, as types
//
// Two axes (POLICY §11): how much of a round is flattened — `REGISTERS`,
// `REGISTERS_LAST` — and how many rounds separate two state commitments —
// `SPAN`, with `GROUPS = ROUNDS / (SPAN + 1)`. `POST` follows from both and is
// still spelled out, because stable Rust cannot compute one array length from
// another and an alias is the only place a caller cannot pair them wrongly.
//
// The parameter order is
// `<F, WIDTH, PAIRS, REGISTERS, REGISTERS_LAST, SPAN, POST, GROUPS, ROUNDS, VECTOR_LEN>`.
// ---------------------------------------------------------------------------

/// t = 16, both squarings committed: no state columns, max degree 2. The variant
/// to reach for.
pub type Flattened16<F> =
    VectorizedPSquareHashAir<F, 16, 4, 2, 2, 0, 0, 52, ROUNDS_T16, VECTOR_LEN>;
/// t = 16, state columns only: max degree 4, same width as [`Flattened16`].
pub type StateOnly16<F> =
    VectorizedPSquareHashAir<F, 16, 4, 0, 0, 0, 8, 52, ROUNDS_T16, VECTOR_LEN>;
/// t = 16, one register plus state columns: max degree 2 at 1.5× the per-round
/// cells of [`Flattened16`]. This is the ported baseline.
pub type OneRegister16<F> =
    VectorizedPSquareHashAir<F, 16, 4, 1, 1, 0, 8, 52, ROUNDS_T16, VECTOR_LEN>;

/// t = 16, one state commitment every **second** round and no registers:
/// `h/2` cells per round, the narrowest layout here, at max degree 16.
///
/// The bottom row of the frontier in
/// [`full_commit::air`](crate::full_commit::air): two rounds of unflattened
/// Feistels between commitments multiply the degree by `4²`. Whether the width
/// is worth the code rate that degree buys is what the measured rows answer —
/// `log_blowup = 4` against [`Flattened16`]'s 1.
pub type Spaced16<F> = VectorizedPSquareHashAir<F, 16, 4, 0, 0, 1, 8, 26, ROUNDS_T16, VECTOR_LEN>;
/// t = 16, one state commitment every second round, with a register on the
/// committing round: `3h/4` cells per round at max degree 8.
///
/// The middle row of the frontier, and the minimal-width layout that still fits
/// POLICY §7's common blowup. The register is spent on the committing round
/// because that is where the commitment absorbs it; a round earlier it would
/// halve a degree the next round multiplies straight back.
pub type SpacedSplit16<F> =
    VectorizedPSquareHashAir<F, 16, 4, 0, 1, 1, 8, 26, ROUNDS_T16, VECTOR_LEN>;

/// t = 24, both squarings committed. See [`Flattened16`].
pub type Flattened24<F> =
    VectorizedPSquareHashAir<F, 24, 6, 2, 2, 0, 0, 52, ROUNDS_T24, VECTOR_LEN>;
/// t = 24, state columns only. See [`StateOnly16`].
pub type StateOnly24<F> =
    VectorizedPSquareHashAir<F, 24, 6, 0, 0, 0, 12, 52, ROUNDS_T24, VECTOR_LEN>;
/// t = 24, one register plus state columns. See [`OneRegister16`].
pub type OneRegister24<F> =
    VectorizedPSquareHashAir<F, 24, 6, 1, 1, 0, 12, 52, ROUNDS_T24, VECTOR_LEN>;
/// t = 24, one commitment every second round. See [`Spaced16`].
pub type Spaced24<F> = VectorizedPSquareHashAir<F, 24, 6, 0, 0, 1, 12, 26, ROUNDS_T24, VECTOR_LEN>;
/// t = 24, one commitment every second round, register on the committing round.
/// See [`SpacedSplit16`].
pub type SpacedSplit24<F> =
    VectorizedPSquareHashAir<F, 24, 6, 0, 1, 1, 12, 26, ROUNDS_T24, VECTOR_LEN>;

// ---------------------------------------------------------------------------
// The third axis, as types
//
// `crate::half_commit`: one committed cell per Feistel, the other output of the
// pair recovered by differencing. The parameter list is
// `<F, WIDTH, PAIRS, ROUNDS, VECTOR_LEN>` and that is all of it — no `REGISTERS`,
// no `SPAN`, no `POST`, no `GROUPS`, because every round commits the same half of
// the same pair and there is no second choice to pair wrongly. `VECTOR_LEN` is
// the same constant the five above use, re-exported by
// `half_commit::vectorized` so that it cannot become two (POLICY §11: a row
// differing by its packing rather than its layout is not a comparison).
// ---------------------------------------------------------------------------

/// t = 16, one output per Feistel committed and the other differenced: `t/4`
/// cells per round at max degree 4.
///
/// The same 240 cells as [`Spaced16`] at degree 4 instead of 16, and half of
/// [`StateOnly16`]'s width at the same degree — it dominates both. Against
/// [`Flattened16`] it is a tie on POLICY §11's primary reading (`t/2 × 2¹`
/// against `t/4 × 2²`), which is what makes it a measured row rather than an
/// argument. See [`half_commit::columns`](crate::half_commit::columns) for why the
/// committed cell has to be the odd one.
pub type HalfCommit16<F> = VectorizedHalfCommitAir<F, 16, 4, ROUNDS_T16, VECTOR_LEN>;
/// t = 24, one output per Feistel committed and the other differenced. See
/// [`HalfCommit16`].
pub type HalfCommit24<F> = VectorizedHalfCommitAir<F, 24, 6, ROUNDS_T24, VECTOR_LEN>;

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------

/// Every grid point of POLICY §3, present or absent.
///
/// `bench::uncovered` is the guard that stops one of these going missing
/// silently; this list is what it checks against, and an absence is an entry
/// here rather than a gap.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "psquarehash",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "psquarehash",
        instance: None,
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: Some(Absence::StubbedDerivation),
    },
    GridPoint {
        construction: "psquarehash",
        instance: Some("psquarehash-mersenne-t16"),
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "psquarehash",
        instance: Some("psquarehash-mersenne-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "psquarehash",
        instance: Some("psquarehash-babybear-t16"),
        field: FieldId::BabyBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "psquarehash",
        instance: Some("psquarehash-babybear-t24"),
        field: FieldId::BabyBear,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "psquarehash",
        instance: Some("psquarehash-koalabear-t16"),
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "psquarehash",
        instance: Some("psquarehash-koalabear-t24"),
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: None,
    },
];

// ---------------------------------------------------------------------------
// Compile-time assertions (POLICY §2, step 6)
//
// Each of these is a mistake that no known-answer vector can see: a variant
// paired with the wrong `POST`, a declared degree that buys a cheaper blowup than
// the arithmetization earns, a grid point quietly dropped.
// ---------------------------------------------------------------------------

const _: () = {
    // The grid is eight points, whatever their contents.
    assert!(INSTANCES.len() == 8);

    // Exactly the two Goldilocks points are absent, and both for the same reason.
    let mut absent = 0;
    let mut i = 0;
    while i < INSTANCES.len() {
        let point = &INSTANCES[i];
        match point.absence {
            Some(Absence::StubbedDerivation) => {
                assert!(point.instance.is_none());
                assert!(matches!(point.field, FieldId::Goldilocks));
                absent += 1;
            }
            None => assert!(point.instance.is_some()),
            _ => panic!("pSquareHash has no other kind of absence"),
        }
        i += 1;
    }
    assert!(absent == 2);
};

/// The variant table of the module docs, asserted rather than described.
///
/// The declared degree is what derives the blowup (POLICY §7), so a wrong entry
/// here is a silently cheaper configuration that every KAT still passes.
const _: () = {
    use crate::full_commit::air::max_constraint_degree as degree;

    // Commit every round: the three original variants, unchanged.
    assert!(degree(2, 2, 0, 0, ROUNDS_T16) == 2);
    assert!(degree(1, 1, 0, 8, ROUNDS_T16) == 2);
    assert!(degree(0, 0, 0, 8, ROUNDS_T16) == 4);

    // Commit every second round: the two spaced variants.
    assert!(degree(0, 1, 1, 8, ROUNDS_T16) == 8);
    assert!(degree(0, 0, 1, 8, ROUNDS_T16) == 16);

    // Commit half a pair every round: the third axis. Same width as the last
    // line above and a quarter of its degree, which is the whole finding — and a
    // fixed point in the round count rather than a value that happens to be 4 at
    // 52, which is what `half_commit::air`'s own tests are for.
    assert!(crate::half_commit::air::max_constraint_degree(ROUNDS_T16) == 4);
};

/// Cells per round, per variant, at both widths — the claim the reduction rests
/// on. `num_cols` is `inputs + ROUNDS · per_round + outputs`, so the per-round
/// figure is recoverable and worth pinning here as well as in POLICY §11's
/// number test, because this is where the *comparison* between variants lives.
///
/// Read the six together: `h`, `h`, `3h/2`, `3h/4`, `h/2`, `h/2` at degrees 2, 4,
/// 2, 8, 16, 4. Every step below `h` is bought with degree, and a degree is bought
/// with a code rate — which is why the ordering these numbers suggest is not the
/// ordering the measured rows produce. The last two are the pair to read against
/// each other: the same `h/2`, one at degree 16 and one at degree 4, so the
/// spacing axis is strictly dominated at this width.
const _: () = {
    use crate::full_commit::columns::num_cols;
    use crate::half_commit::columns::num_cols as half;

    const fn per_round(total: usize, width: usize, rounds: usize) -> usize {
        (total - 2 * width) / rounds
    }

    // t = 16, h = 8: flattening and state-only both cost h; one register 3h/2.
    assert!(per_round(num_cols::<16, 4, 2, 2, 0, 0, 52>(), 16, ROUNDS_T16) == 8);
    assert!(per_round(num_cols::<16, 4, 0, 0, 0, 8, 52>(), 16, ROUNDS_T16) == 8);
    assert!(per_round(num_cols::<16, 4, 1, 1, 0, 8, 52>(), 16, ROUNDS_T16) == 12);
    // Spaced: 3h/4 with the register, h/2 without.
    assert!(per_round(num_cols::<16, 4, 0, 1, 1, 8, 26>(), 16, ROUNDS_T16) == 6);
    assert!(per_round(num_cols::<16, 4, 0, 0, 1, 8, 26>(), 16, ROUNDS_T16) == 4);
    // Half-commitment: the same h/2 as the line above, every round, at degree 4.
    assert!(per_round(half::<16, 4, 52>(), 16, ROUNDS_T16) == 4);

    // t = 24, h = 12: 12, 12, 18, 9, 6, 6.
    assert!(per_round(num_cols::<24, 6, 2, 2, 0, 0, 52>(), 24, ROUNDS_T24) == 12);
    assert!(per_round(num_cols::<24, 6, 0, 0, 0, 12, 52>(), 24, ROUNDS_T24) == 12);
    assert!(per_round(num_cols::<24, 6, 1, 1, 0, 12, 52>(), 24, ROUNDS_T24) == 18);
    assert!(per_round(num_cols::<24, 6, 0, 1, 1, 12, 26>(), 24, ROUNDS_T24) == 9);
    assert!(per_round(num_cols::<24, 6, 0, 0, 1, 12, 26>(), 24, ROUNDS_T24) == 6);
    assert!(per_round(half::<24, 6, 52>(), 24, ROUNDS_T24) == 6);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every present instance's name has to match the parameters it is built
    /// from, or the grid entry and the vectors part company silently (POLICY §3).
    #[test]
    fn grid_names_match_the_parameter_constructors() {
        use p3_baby_bear::BabyBear;
        use p3_koala_bear::KoalaBear;
        use p3_mersenne_31::Mersenne31;

        use crate::params;

        let names: Vec<&str> = INSTANCES.iter().filter_map(|i| i.instance).collect();
        assert_eq!(
            names,
            vec![
                params::mersenne_t16::<Mersenne31>().name,
                params::mersenne_t24::<Mersenne31>().name,
                params::babybear_t16::<BabyBear>().name,
                params::babybear_t24::<BabyBear>().name,
                params::koalabear_t16::<KoalaBear>().name,
                params::koalabear_t24::<KoalaBear>().name,
            ]
        );
    }

    /// The name encodes the width, and nothing else checks that it encodes the
    /// *right* one.
    #[test]
    fn names_encode_their_width() {
        for point in INSTANCES {
            if let Some(name) = point.instance {
                assert!(
                    name.ends_with(&format!("-t{}", point.state_width)),
                    "{name} is not at t = {}",
                    point.state_width
                );
            }
        }
    }
}
