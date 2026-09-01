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
//! # Eight points, no absences
//!
//! Poseidon2 is the one construction here that reaches every grid point.
//! Upstream ships a `GenericPoseidon2LinearLayers` impl at exactly the widths the
//! grid asks for — Goldilocks 8 and 12, Mersenne-31 16 and 24, and (through
//! `GenericPoseidon2LinearLayersMonty31`) BabyBear and KoalaBear 16 and 24 — with
//! a published constant set at each. Four of the eight are the reference's own
//! instances; the other four are upstream's, with upstream's round counts, for
//! the reason in [`crate::params`].
//!
//! # This is also where the harness contract lives
//!
//! [`WrappedPoseidon2Air`] is the whole of it: a newtype around upstream's
//! `VectorizedPoseidon2Air` that forwards `BaseAir` and `Air` and adds the two
//! trace generators and the [`Labels`] POLICY §7 asks a construction for. It
//! exists **only** because the orphan rule does — `harness::permutation::PermutationAir`
//! and `p3_poseidon2_air::VectorizedPoseidon2Air` are both foreign to this crate
//! — and it deliberately holds no round-function logic, no column layout and no
//! constraint: there is nothing here that *can* drift from upstream, which is why
//! POLICY §5 gives a wrapped construction no `columns.rs` / `air.rs` /
//! `generation.rs`.
//!
//! # Variants (POLICY §11)
//!
//! Every instance is measured at every degree/register variant it admits, and
//! `SBOX_REGISTERS` is a parameter of each alias below, so a variant is a type:
//!
//! | α | registers | AIR degree | which instances |
//! |---|---|---|---|
//! | 7 | 0 | 7 | Goldilocks t = 8, 12; BabyBear t = 16, 24 |
//! | 7 | 1 | 3 | the same |
//! | 5 | 0 | 5 | Mersenne-31 t = 16, 24 |
//! | 5 | 1 | 3 | the same |
//! | 3 | 0 | 3 | KoalaBear t = 16, 24 |
//!
//! **KoalaBear admits one variant, and that is itself a finding** (POLICY §11):
//! α = 3 is already degree 3, so there is no register split to buy anything with,
//! and the α = 3 row is the one where Poseidon2's degree column stops being a
//! choice.
//!
//! ```
//! use poseidon2::instances::GoldilocksT8;
//! use poseidon2::params;
//!
//! // The same instance at both of its variants: degree 7, and degree 3 for one
//! // committed cell per S-box.
//! let plain = GoldilocksT8::<0>::new(params::goldilocks_t8());
//! let split = GoldilocksT8::<1>::new(params::goldilocks_t8());
//! ```

use core::fmt;

use harness::permutation::{Labels, PermutationAir};
use harness::{FieldId, GridPoint};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_field::{PrimeCharacteristicRing, PrimeField};
use p3_goldilocks::{GenericPoseidon2LinearLayersGoldilocks, Goldilocks};
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{GenericPoseidon2LinearLayersMersenne31, MERSENNE31_S_BOX_DEGREE, Mersenne31};
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air, generate_vectorized_trace_rows};
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::params::{
    BABYBEAR_T16_PARTIAL_ROUNDS, BABYBEAR_T24_PARTIAL_ROUNDS, GOLDILOCKS_T8_PARTIAL_ROUNDS,
    GOLDILOCKS_T12_PARTIAL_ROUNDS, HALF_FULL_ROUNDS, KOALABEAR_T16_PARTIAL_ROUNDS,
    KOALABEAR_T24_PARTIAL_ROUNDS, MERSENNE_T16_PARTIAL_ROUNDS, MERSENNE_T24_PARTIAL_ROUNDS,
    max_constraint_degree,
};

/// The `VECTOR_LEN` every instance is measured at.
///
/// A row of `VECTOR_LEN` independent calls is what amortizes the per-row costs
/// the prover pays regardless of width. It is a property of the *harness's*
/// taste rather than of Poseidon2, so it is one number here and the same one for
/// every instance — and it must equal every other construction's, or two rows
/// differ by their packing instead of by their arithmetization. The
/// cross-construction guard in `bench` is what makes that loud; there is no
/// `harness::VECTOR_LEN` to share, because the harness owns the configuration and
/// the packing belongs to the AIR.
pub const VECTOR_LEN: usize = 8;

/// The S-box degree KoalaBear admits — `p3-koala-bear`'s own, restated because
/// its constant lives behind `p3_koala_bear::KOALABEAR_S_BOX_DEGREE` and is used
/// here in const-generic position.
pub const KOALABEAR_SBOX_DEGREE: u64 = p3_koala_bear::KOALABEAR_S_BOX_DEGREE;
/// The S-box degree Goldilocks admits, from `p3-goldilocks`.
pub const GOLDILOCKS_SBOX_DEGREE: u64 = p3_goldilocks::poseidon1::GOLDILOCKS_S_BOX_DEGREE;

// ---------------------------------------------------------------------------
// The harness adapter
// ---------------------------------------------------------------------------

/// Upstream's vectorized Poseidon2 AIR, as [`PermutationAir`].
///
/// A newtype and nothing more: `width`, `main_next_row_columns`,
/// `max_constraint_degree` and `eval` are forwarded verbatim, so the
/// arithmetization measured here is exactly `p3-poseidon2-air`'s. What this type
/// adds is the two trace generators of POLICY §7 — one over caller-supplied
/// inputs, one over the harness's seed — and the [`Labels`] the cost table
/// prints.
///
/// One row is `VECTOR_LEN` independent calls, which upstream declares the same
/// way POLICY §6 asks for: `main_next_row_columns()` returns `vec![]`, so the
/// prover skips opening the shifted trace. Tables are full, so the measured call
/// count is exactly `VECTOR_LEN × 2^k`.
pub struct WrappedPoseidon2Air<
    F: PrimeCharacteristicRing,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    air: VectorizedPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >,
    /// A second copy of the constants the AIR was built from.
    ///
    /// Not a design choice: `VectorizedPoseidon2Air` keeps its `RoundConstants`
    /// `pub(crate)` and offers no accessor, while the free
    /// `generate_vectorized_trace_rows` — the only generator that takes
    /// caller-supplied inputs, which the harness contract requires — asks for
    /// them by reference. Cloning the table is cheaper than the alternatives,
    /// and it is *data*, not logic: the two copies are the same value, and
    /// `tests/air.rs` replays the generated trace against upstream's own
    /// permutation, which would catch them parting company.
    constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
}

impl<
    F: PrimeCharacteristicRing,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    WrappedPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    /// Build this variant of an instance from upstream's round constants.
    ///
    /// The variant lives in the type (the aliases below) and the instance in the
    /// constants, so the two cannot be mixed up: an instance's `HALF_FULL_ROUNDS`
    /// and `PARTIAL_ROUNDS` are part of its `RoundConstants` type and have to
    /// agree with the alias's.
    #[must_use]
    pub fn new(constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>) -> Self {
        Self {
            air: VectorizedPoseidon2Air::new(constants.clone()),
            constants,
        }
    }

    /// The wrapped AIR, for anything that wants upstream's type directly —
    /// `check_constraints` on a hand-built trace, say.
    #[must_use]
    pub const fn upstream(
        &self,
    ) -> &VectorizedPoseidon2Air<
        F,
        LinearLayers,
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
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> fmt::Debug
    for WrappedPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WrappedPoseidon2Air")
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
    LinearLayers: Sync,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> BaseAir<F>
    for WrappedPoseidon2Air<
        F,
        LinearLayers,
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
    /// calls, nothing reads the next row, and the prover skips opening the
    /// shifted trace (POLICY §6).
    fn main_next_row_columns(&self) -> Vec<usize> {
        self.air.main_next_row_columns()
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        self.air.max_constraint_degree()
    }
}

impl<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> Air<AB>
    for WrappedPoseidon2Air<
        AB::F,
        LinearLayers,
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
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> PermutationAir<Val<SC>, SC>
    for WrappedPoseidon2Air<
        Val<SC>,
        LinearLayers,
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
        // A call fits one row: every round commits its post-state, so nothing
        // needs a transition constraint.
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
        generate_vectorized_trace_rows::<
            _,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >(
            states::<_, WIDTH>(inputs),
            &self.constants,
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
        // inputs rather than calling upstream's `generate_vectorized_trace_rows`,
        // whose RNG is a `SmallRng` seeded with 1 that no other construction uses.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let inputs: Vec<[Val<SC>; WIDTH]> = (0..num_calls)
            .map(|_| core::array::from_fn(|_| rng.sample(StandardUniform)))
            .collect();
        generate_vectorized_trace_rows::<
            _,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >(inputs, &self.constants, extra_capacity_bits)
    }
}

// ---------------------------------------------------------------------------
// The variants, as types
//
// `SBOX_REGISTERS` is the alias's one free parameter, so `GoldilocksT8<0>` and
// `GoldilocksT8<1>` are POLICY §11's two variants of one instance and cannot be
// confused with each other or with another instance's.
// ---------------------------------------------------------------------------

/// Goldilocks t = 8, α = 7. Variants: `<0>` at degree 7, `<1>` at degree 3.
pub type GoldilocksT8<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    Goldilocks,
    GenericPoseidon2LinearLayersGoldilocks,
    8,
    GOLDILOCKS_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    GOLDILOCKS_T8_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// Goldilocks t = 12, α = 7. See [`GoldilocksT8`].
pub type GoldilocksT12<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    Goldilocks,
    GenericPoseidon2LinearLayersGoldilocks,
    12,
    GOLDILOCKS_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    GOLDILOCKS_T12_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// Mersenne-31 t = 16, α = 5. Variants: `<0>` at degree 5, `<1>` at degree 3.
pub type MersenneT16<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    Mersenne31,
    GenericPoseidon2LinearLayersMersenne31,
    16,
    MERSENNE31_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    MERSENNE_T16_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// Mersenne-31 t = 24, α = 5. See [`MersenneT16`].
pub type MersenneT24<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    Mersenne31,
    GenericPoseidon2LinearLayersMersenne31,
    24,
    MERSENNE31_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    MERSENNE_T24_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// BabyBear t = 16, α = 7. Round count upstream's (see [`crate::params`]).
pub type BabyBearT16<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    16,
    BABYBEAR_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    BABYBEAR_T16_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// BabyBear t = 24, α = 7. See [`BabyBearT16`].
pub type BabyBearT24<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    24,
    BABYBEAR_S_BOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    BABYBEAR_T24_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// KoalaBear t = 16, α = 3 — **one variant only**, `<0>`, already at degree 3.
pub type KoalaBearT16<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
    16,
    KOALABEAR_SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    KOALABEAR_T16_PARTIAL_ROUNDS,
    VECTOR_LEN,
>;

/// KoalaBear t = 24, α = 3. See [`KoalaBearT16`].
pub type KoalaBearT24<const SBOX_REGISTERS: usize> = WrappedPoseidon2Air<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
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
/// silently; this list is what it checks against. Poseidon2 is the one
/// construction with nothing absent.
///
/// The four Goldilocks and Mersenne-31 names are the reference's variables
/// lowercased with `_` → `-`; the four 31-bit ones follow the same convention for
/// instances the reference does not have.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-mersenne-t16"),
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-mersenne-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-babybear-t16"),
        field: FieldId::BabyBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-babybear-t24"),
        field: FieldId::BabyBear,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-koalabear-t16"),
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "poseidon2",
        instance: Some("poseidon2-koalabear-t24"),
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: None,
    },
];

// ---------------------------------------------------------------------------
// Compile-time assertions (POLICY §2, step 6)
//
// Each of these is a mistake no test of the round function can see: a variant
// that upstream's S-box does not implement, a declared degree that buys a
// cheaper blowup than the arithmetization earns, a grid point quietly dropped.
// ---------------------------------------------------------------------------

const _: () = {
    // The grid is eight points, and Poseidon2 reaches all of them.
    assert!(INSTANCES.len() == 8);
    let mut i = 0;
    while i < INSTANCES.len() {
        assert!(INSTANCES[i].instance.is_some());
        assert!(INSTANCES[i].absence.is_none());
        i += 1;
    }
};

/// The variant table of the module docs, asserted rather than described.
///
/// `max_constraint_degree` panics on a pair upstream does not implement, so
/// naming a variant here is also a check that it exists at all.
const _: () = {
    // α = 7: Goldilocks and BabyBear.
    assert!(max_constraint_degree(GOLDILOCKS_SBOX_DEGREE, 0) == 7);
    assert!(max_constraint_degree(GOLDILOCKS_SBOX_DEGREE, 1) == 3);
    assert!(BABYBEAR_S_BOX_DEGREE == GOLDILOCKS_SBOX_DEGREE);

    // α = 5: Mersenne-31.
    assert!(max_constraint_degree(MERSENNE31_S_BOX_DEGREE, 0) == 5);
    assert!(max_constraint_degree(MERSENNE31_S_BOX_DEGREE, 1) == 3);

    // α = 3: KoalaBear, one variant.
    assert!(max_constraint_degree(KOALABEAR_SBOX_DEGREE, 0) == 3);
};

/// `R_ext = 8` at every point, which is what makes `rounds` in the labels
/// `8 + R_int` and comparable across the grid.
const _: () = assert!(HALF_FULL_ROUNDS * 2 == 8);

#[cfg(test)]
mod tests {
    use super::*;

    /// The name encodes the width, and nothing else checks that it encodes the
    /// *right* one.
    #[test]
    fn names_encode_their_width() {
        for point in INSTANCES {
            let name = point.instance.expect("Poseidon2 has no absent point");
            assert!(
                name.ends_with(&format!("-t{}", point.state_width)),
                "{name} is not at t = {}",
                point.state_width
            );
        }
    }

    /// The name encodes the field too, under the reference's spelling of it
    /// (`MERSENNE`, not `MERSENNE31`).
    #[test]
    fn names_encode_their_field() {
        for point in INSTANCES {
            let name = point.instance.expect("Poseidon2 has no absent point");
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
