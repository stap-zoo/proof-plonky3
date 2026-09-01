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
//! # Seven points, one absence
//!
//! **Mersenne-31 t = 24 does not exist upstream** — no round constants, no
//! circulant MDS column — and a wrapped construction does not invent either
//! (POLICY §1, §3). It is [`Absence::UndefinedUpstream`], and [`crate::params`]
//! sets out why borrowing a matrix from elsewhere would not do.
//!
//! The name deviates from POLICY §3's rule by one character, and deliberately:
//! the reference's variables are `POSEIDON_*`, which would give
//! `poseidon-goldilocks-t8`, but the construction is registered as `poseidon1`
//! and `bench::misnamed` requires an instance name to start with its
//! construction's. Since a wrapped construction has no reference vectors for the
//! name to key into (POLICY §4), the string is a table label, and
//! `poseidon1-goldilocks-t8` is the label that cannot be mistaken for Poseidon2's
//! row in the same table.
//!
//! # This is also where the harness contract lives
//!
//! [`WrappedPoseidon1Air`] is the whole of it: a newtype around upstream's
//! `VectorizedPoseidon1Air` that forwards `BaseAir` and `Air` and adds the two
//! trace generators and the [`Labels`] POLICY §7 asks a construction for. It
//! exists **only** because the orphan rule does, and holds no round-function
//! logic, no column layout and no constraint — which is why POLICY §5 gives a
//! wrapped construction no `columns.rs` / `air.rs` / `generation.rs`.
//!
//! # Variants (POLICY §11)
//!
//! | α | registers | AIR degree | which instances |
//! |---|---|---|---|
//! | 7 | 0 | 7 | Goldilocks t = 8, 12; BabyBear t = 16, 24 |
//! | 7 | 1 | 3 | the same |
//! | 5 | 0 | 5 | Mersenne-31 t = 16 |
//! | 5 | 1 | 3 | the same |
//! | 3 | 0 | 3 | KoalaBear t = 16, 24 |
//!
//! KoalaBear admits one variant, and that is itself a finding (POLICY §11).
//!
//! ```
//! use poseidon1::instances::GoldilocksT8;
//! use poseidon1::params;
//!
//! let raw = params::goldilocks_t8();
//! let plain = GoldilocksT8::<0>::from_raw(&raw); // degree 7
//! let split = GoldilocksT8::<1>::from_raw(&raw); // degree 3, one register
//! ```

use core::fmt;

use harness::permutation::{Labels, PermutationAir};
use harness::{Absence, FieldId, GridPoint};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BABYBEAR_S_BOX_DEGREE, BabyBear};
use p3_field::{PrimeCharacteristicRing, PrimeField};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{MERSENNE31_POSEIDON1_S_BOX_DEGREE, Mersenne31};
use p3_poseidon1::Poseidon1Constants;
use p3_poseidon1_air::{FullRoundConstants, PartialRoundConstants, VectorizedPoseidon1Air};
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::params::{
    BABYBEAR_T16_PARTIAL_ROUNDS, BABYBEAR_T24_PARTIAL_ROUNDS, GOLDILOCKS_T8_PARTIAL_ROUNDS,
    GOLDILOCKS_T12_PARTIAL_ROUNDS, HALF_FULL_ROUNDS, KOALABEAR_T16_PARTIAL_ROUNDS,
    KOALABEAR_T24_PARTIAL_ROUNDS, MERSENNE_T16_PARTIAL_ROUNDS, max_constraint_degree,
};

/// The `VECTOR_LEN` every instance is measured at.
///
/// One row is this many independent calls. It is a property of the *harness's*
/// taste rather than of Poseidon1, so it is one number here and the same one for
/// every instance — and it must equal every other construction's, or two rows
/// differ by their packing instead of by their arithmetization. The
/// cross-construction guard in `bench` is what makes that loud.
pub const VECTOR_LEN: usize = 8;

/// The S-box degree KoalaBear admits, from `p3-koala-bear`.
pub const KOALABEAR_SBOX_DEGREE: u64 = p3_koala_bear::KOALABEAR_S_BOX_DEGREE;
/// The S-box degree Goldilocks admits, from `p3-goldilocks`.
pub const GOLDILOCKS_SBOX_DEGREE: u64 = p3_goldilocks::poseidon1::GOLDILOCKS_S_BOX_DEGREE;

// ---------------------------------------------------------------------------
// The harness adapter
// ---------------------------------------------------------------------------

/// Upstream's vectorized Poseidon1 AIR, as [`PermutationAir`].
///
/// A newtype and nothing more: `width`, `main_next_row_columns`,
/// `max_constraint_degree` and `eval` are forwarded verbatim, so the
/// arithmetization measured here is exactly `p3-poseidon1-air`'s — dense
/// Karatsuba MDS in the full rounds, the paper's Appendix B sparse decomposition
/// in the partial ones, one committed cell per partial round instead of a full
/// post-state. That last point is where Poseidon1's width comes from and it is
/// worth naming, because it is *not* the naive `t`-wide-per-round layout the
/// Poseidon1 → Poseidon2 delta is often quoted against.
pub struct WrappedPoseidon1Air<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    air: VectorizedPoseidon1Air<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >,
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    WrappedPoseidon1Air<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    /// Build this variant from constants already put through the sparse
    /// decomposition.
    #[must_use]
    pub const fn new(
        full: FullRoundConstants<F, WIDTH>,
        partial: PartialRoundConstants<F, WIDTH>,
    ) -> Self {
        Self {
            air: VectorizedPoseidon1Air::new(full, partial),
        }
    }

    /// Build this variant from an instance's raw parameters.
    ///
    /// `to_optimized` is upstream's: it factors the dense MDS into the Appendix B
    /// sparse form and folds each partial round's constant vector down to a
    /// scalar. Doing it here rather than in [`crate::params`] keeps *one* value —
    /// the raw `Poseidon1Constants` — as the single source both this AIR and
    /// `p3-poseidon1`'s own permutation are built from, so the oracle in
    /// `tests/air.rs` cannot drift from the AIR under test.
    #[must_use]
    pub fn from_raw(raw: &Poseidon1Constants<F, WIDTH>) -> Self
    where
        F: PrimeField,
    {
        let (full, partial) = raw.to_optimized();
        Self::new(full, partial)
    }

    /// The wrapped AIR, for anything that wants upstream's type directly.
    #[must_use]
    pub const fn upstream(
        &self,
    ) -> &VectorizedPoseidon1Air<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        &self.air
    }
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> fmt::Debug
    for WrappedPoseidon1Air<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WrappedPoseidon1Air")
            .field("state_width", &WIDTH)
            .field("sbox_degree", &SBOX_DEGREE)
            .field("sbox_registers", &SBOX_REGISTERS)
            .field("rounds", &(2 * HALF_FULL_ROUNDS + PARTIAL_ROUNDS))
            .field("calls_per_row", &VECTOR_LEN)
            .finish()
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> BaseAir<F>
    for WrappedPoseidon1Air<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    fn width(&self) -> usize {
        self.air.width()
    }

    /// Upstream's declaration, forwarded: one row is `VECTOR_LEN` *independent*
    /// calls, nothing reads the next row (POLICY §6).
    fn main_next_row_columns(&self) -> Vec<usize> {
        self.air.main_next_row_columns()
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        self.air.max_constraint_degree()
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> Air<AB>
    for WrappedPoseidon1Air<
        AB::F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);
    }
}

/// One call's input state, as upstream's generators take it.
///
/// # Panics
///
/// If a call's input is not `WIDTH` long — the harness contract says each element
/// of `inputs` is one call's state, and a short one would otherwise be padded
/// with something nobody chose.
fn states<F: Copy, const WIDTH: usize>(inputs: &[Vec<F>]) -> Vec<[F; WIDTH]> {
    inputs
        .iter()
        .map(|state| {
            <[F; WIDTH]>::try_from(state.as_slice())
                .unwrap_or_else(|_| panic!("a call's input state must be {WIDTH} long"))
        })
        .collect()
}

impl<
    SC: StarkGenericConfig,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> PermutationAir<Val<SC>, SC>
    for WrappedPoseidon1Air<
        Val<SC>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
where
    // Upstream's generator is bound to `PrimeField`; POLICY §7 separately
    // declares the ZK Mersenne-31 configuration unsupported.
    Val<SC>: PrimeField,
    StandardUniform: Distribution<Val<SC>>,
{
    /// Labels only — the harness prints these and never branches on them, except
    /// for `max_constraint_degree`, which derives the blowup and is cross-checked
    /// against the symbolic value (POLICY §7).
    const LABELS: Labels = Labels {
        state_width: WIDTH,
        calls_per_row: VECTOR_LEN,
        rows_per_call: 1,
        // `R_ext + R_int`, the reference's round count for this instance.
        rounds: 2 * HALF_FULL_ROUNDS + PARTIAL_ROUNDS,
        sbox_degree: SBOX_DEGREE,
        sbox_registers: SBOX_REGISTERS,
        max_constraint_degree: max_constraint_degree(SBOX_DEGREE, SBOX_REGISTERS),
    };

    fn generate_trace(
        &self,
        inputs: &[Vec<Val<SC>>],
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<Val<SC>> {
        self.air.generate_vectorized_trace_rows_from_inputs(
            states::<_, WIDTH>(inputs),
            extra_capacity_bits,
        )
    }

    fn generate_trace_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<Val<SC>> {
        // Measurement only, never validation (POLICY §7). The seed is the
        // harness's and is the same for every construction, so two rows differ by
        // their AIR and not by their inputs — which is also why this draws its own
        // inputs rather than calling upstream's random generator, whose RNG is a
        // `SmallRng` seeded with 1 that no other construction uses.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let inputs: Vec<[Val<SC>; WIDTH]> = (0..num_calls)
            .map(|_| core::array::from_fn(|_| rng.sample(StandardUniform)))
            .collect();
        self.air
            .generate_vectorized_trace_rows_from_inputs(inputs, extra_capacity_bits)
    }
}

// ---------------------------------------------------------------------------
// The variants, as types
// ---------------------------------------------------------------------------

/// Goldilocks t = 8, α = 7. Variants: `<0>` at degree 7, `<1>` at degree 3.
pub type GoldilocksT8<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    Goldilocks,
    8,
    GOLDILOCKS_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    GOLDILOCKS_T8_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// Goldilocks t = 12, α = 7. See [`GoldilocksT8`].
pub type GoldilocksT12<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    Goldilocks,
    12,
    GOLDILOCKS_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    GOLDILOCKS_T12_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// Mersenne-31 t = 16, α = 5. Variants: `<0>` at degree 5, `<1>` at degree 3.
///
/// There is no `MersenneT24`: upstream pins no instance at that width, and this
/// crate does not invent one (see [`crate::params`]).
pub type MersenneT16<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    Mersenne31,
    16,
    MERSENNE31_POSEIDON1_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    MERSENNE_T16_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// BabyBear t = 16, α = 7. Round count upstream's (see [`crate::params`]).
pub type BabyBearT16<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    BabyBear,
    16,
    BABYBEAR_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    BABYBEAR_T16_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// BabyBear t = 24, α = 7. See [`BabyBearT16`].
pub type BabyBearT24<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    BabyBear,
    24,
    BABYBEAR_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    BABYBEAR_T24_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// KoalaBear t = 16, α = 3 — **one variant only**, `<0>`, already at degree 3.
pub type KoalaBearT16<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    KoalaBear,
    16,
    KOALABEAR_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    KOALABEAR_T16_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// KoalaBear t = 24, α = 3. See [`KoalaBearT16`].
pub type KoalaBearT24<const SBOX_REGISTERS: usize> = WrappedPoseidon1Air<
    KoalaBear,
    24,
    KOALABEAR_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    KOALABEAR_T24_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------

/// Every grid point of POLICY §3, present or absent.
///
/// `bench::uncovered` is the guard that stops one of these going missing
/// silently; this list is what it checks against, and the one absence is an entry
/// here rather than a gap.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-mersenne-t16"),
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "poseidon1",
        instance: None,
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: Some(Absence::UndefinedUpstream),
    },
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-babybear-t16"),
        field: FieldId::BabyBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-babybear-t24"),
        field: FieldId::BabyBear,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-koalabear-t16"),
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "poseidon1",
        instance: Some("poseidon1-koalabear-t24"),
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: None,
    },
];

// ---------------------------------------------------------------------------
// Compile-time assertions (POLICY §2, step 6)
// ---------------------------------------------------------------------------

const _: () = {
    // The grid is eight points, and exactly one of them — Mersenne-31 t = 24 — is
    // absent, for exactly one reason.
    assert!(INSTANCES.len() == 8);
    let mut absent = 0;
    let mut i = 0;
    while i < INSTANCES.len() {
        let point = &INSTANCES[i];
        match point.absence {
            Some(Absence::UndefinedUpstream) => {
                assert!(point.instance.is_none());
                assert!(matches!(point.field, FieldId::Mersenne31));
                assert!(point.state_width == 24);
                absent += 1;
            }
            None => assert!(point.instance.is_some()),
            _ => panic!("Poseidon1 has no other kind of absence"),
        }
        i += 1;
    }
    assert!(absent == 1);
};

/// The variant table of the module docs, asserted rather than described.
///
/// `max_constraint_degree` panics on a pair upstream does not implement, so
/// naming a variant here is also a check that it exists at all.
const _: () = {
    assert!(max_constraint_degree(GOLDILOCKS_SBOX_DEGREE, 0) == 7);
    assert!(max_constraint_degree(GOLDILOCKS_SBOX_DEGREE, 1) == 3);
    assert!(BABYBEAR_S_BOX_DEGREE == GOLDILOCKS_SBOX_DEGREE);

    assert!(max_constraint_degree(MERSENNE31_POSEIDON1_S_BOX_DEGREE, 0) == 5);
    assert!(max_constraint_degree(MERSENNE31_POSEIDON1_S_BOX_DEGREE, 1) == 3);

    assert!(max_constraint_degree(KOALABEAR_SBOX_DEGREE, 0) == 3);
};

/// `R_ext = 8` at every point, which is what makes `rounds` in the labels
/// `8 + R_int` and comparable across the grid — and across to Poseidon2, whose
/// grid carries the same `R_ext`.
const _: () = assert!(HALF_FULL_ROUNDS * 2 == 8);

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The name encodes the field too, under the reference's spelling of it
    /// (`MERSENNE`, not `MERSENNE31`).
    #[test]
    fn names_encode_their_field() {
        for point in INSTANCES {
            if let Some(name) = point.instance {
                let field = match point.field {
                    FieldId::Goldilocks => "goldilocks",
                    FieldId::Mersenne31 => "mersenne",
                    FieldId::BabyBear => "babybear",
                    FieldId::KoalaBear => "koalabear",
                };
                assert!(name.contains(field), "{name} is not over {field}");
            }
        }
    }
}
