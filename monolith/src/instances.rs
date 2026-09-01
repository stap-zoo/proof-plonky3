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

use core::fmt;

use harness::permutation::{Labels, PermutationAir};
use harness::{Absence, FieldId, GridPoint};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use p3_monolith::{Monolith, MonolithBars, MonolithBarsGoldilocks, MonolithBarsM31};
use p3_monolith_air::{
    GOLDILOCKS_8_LIMB_BITS, MERSENNE31_LIMB_BITS, MonolithAir, generate_trace_rows,
};
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::params::{NUM_FULL_ROUNDS, TOTAL_ROUNDS};

/// The algebraic degree of the Bars decomposition AIR.  This is not a power-map
/// S-box degree (`sbox_degree` is therefore 0 in [`Labels`]); it is the degree
/// of the actual constraints that determines the common blowup.
pub const MAX_CONSTRAINT_DEGREE: usize = 3;

/// Upstream's Monolith AIR, adapted to the harness contract.
///
/// The newtype owns only upstream data: its AIR plus the Bars implementation
/// which upstream's caller-input trace generator takes separately.  Constraint
/// evaluation is forwarded verbatim, so this crate has no alternate Monolith
/// arithmetization to drift from `p3-monolith-air`.
pub struct WrappedMonolithAir<
    F: PrimeCharacteristicRing,
    B,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
    const NUM_CHI_CELLS: usize,
> {
    air: MonolithAir<
        F,
        WIDTH,
        NUM_FULL_ROUNDS,
        NUM_BARS,
        FIELD_BITS,
        NUM_MATCH_FLAGS,
        NUM_CHI_CELLS,
    >,
    bars: B,
}

impl<
    F: PrimeCharacteristicRing,
    B,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
    const NUM_CHI_CELLS: usize,
> WrappedMonolithAir<F, B, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS, NUM_CHI_CELLS>
{
    /// The wrapped upstream AIR.
    #[must_use]
    pub const fn upstream(
        &self,
    ) -> &MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS, NUM_CHI_CELLS>
    {
        &self.air
    }
}

impl<
    F: PrimeCharacteristicRing,
    B: Sync,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
    const NUM_CHI_CELLS: usize,
> fmt::Debug
    for WrappedMonolithAir<F, B, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS, NUM_CHI_CELLS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WrappedMonolithAir")
            .field("state_width", &WIDTH)
            .field("rounds", &TOTAL_ROUNDS)
            .field("calls_per_row", &1)
            .finish()
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    B: Sync,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
    const NUM_CHI_CELLS: usize,
> BaseAir<F>
    for WrappedMonolithAir<F, B, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS, NUM_CHI_CELLS>
{
    fn width(&self) -> usize {
        self.air.width()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        self.air.main_next_row_columns()
    }
}

impl<
    AB: AirBuilder,
    B: Sync,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
    const NUM_CHI_CELLS: usize,
> Air<AB>
    for WrappedMonolithAir<AB::F, B, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS, NUM_CHI_CELLS>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);
    }
}

/// Monolith-64, Goldilocks t = 8.
pub type GoldilocksT8 = WrappedMonolithAir<Goldilocks, MonolithBarsGoldilocks<8>, 8, 4, 64, 16, 64>;
/// Monolith-64, Goldilocks t = 12.
pub type GoldilocksT12 =
    WrappedMonolithAir<Goldilocks, MonolithBarsGoldilocks<8>, 12, 4, 64, 16, 64>;
/// Monolith-31, Mersenne-31 t = 16.
pub type MersenneT16 = WrappedMonolithAir<Mersenne31, MonolithBarsM31, 16, 8, 31, 15, 24>;
/// Monolith-31, Mersenne-31 t = 24.
pub type MersenneT24 = WrappedMonolithAir<Mersenne31, MonolithBarsM31, 24, 8, 31, 15, 24>;

macro_rules! field_constructor {
    ($name:ident, $field:ty, $bars:ty, $width:expr, $num_bars:expr, $bits:expr, $flags:expr, $chi:expr, $limbs:expr) => {
        impl $name {
            /// Construct this field and width from `p3-monolith`'s native permutation.
            #[must_use]
            pub fn from_native<Mds>(
                native: Monolith<$field, $bars, Mds, $width, NUM_FULL_ROUNDS>,
            ) -> Self
            where
                Mds: p3_mds::MdsPermutation<$field, $width>,
            {
                let mds_matrix = MonolithAir::<
                    $field,
                    $width,
                    NUM_FULL_ROUNDS,
                    $num_bars,
                    $bits,
                    $flags,
                    $chi,
                >::extract_mds_matrix(&native.mds);
                let air = MonolithAir::new(native.round_constants, mds_matrix, $limbs);
                Self {
                    air,
                    bars: native.bars,
                }
            }
        }
    };
}

field_constructor!(
    GoldilocksT8,
    Goldilocks,
    MonolithBarsGoldilocks<8>,
    8,
    4,
    64,
    16,
    64,
    GOLDILOCKS_8_LIMB_BITS
);
field_constructor!(
    GoldilocksT12,
    Goldilocks,
    MonolithBarsGoldilocks<8>,
    12,
    4,
    64,
    16,
    64,
    GOLDILOCKS_8_LIMB_BITS
);
field_constructor!(
    MersenneT16,
    Mersenne31,
    MonolithBarsM31,
    16,
    8,
    31,
    15,
    24,
    MERSENNE31_LIMB_BITS
);
field_constructor!(
    MersenneT24,
    Mersenne31,
    MonolithBarsM31,
    24,
    8,
    31,
    15,
    24,
    MERSENNE31_LIMB_BITS
);

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
    B,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
    const NUM_CHI_CELLS: usize,
> PermutationAir<Val<SC>, SC>
    for WrappedMonolithAir<Val<SC>, B, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS, NUM_CHI_CELLS>
where
    Val<SC>: PrimeField64,
    B: MonolithBars<Val<SC>, WIDTH> + Sync,
    StandardUniform: Distribution<[Val<SC>; WIDTH]>,
{
    const LABELS: Labels = Labels {
        state_width: WIDTH,
        calls_per_row: 1,
        rows_per_call: 1,
        rounds: TOTAL_ROUNDS,
        sbox_degree: 0,
        sbox_registers: 0,
        max_constraint_degree: MAX_CONSTRAINT_DEGREE,
    };

    fn generate_trace(
        &self,
        inputs: &[Vec<Val<SC>>],
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<Val<SC>> {
        generate_trace_rows(
            states::<_, WIDTH>(inputs),
            &self.air,
            &self.bars,
            extra_capacity_bits,
        )
    }

    fn generate_trace_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<Val<SC>> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let inputs = (0..num_calls)
            .map(|_| rng.sample(StandardUniform))
            .collect();
        generate_trace_rows(inputs, &self.air, &self.bars, extra_capacity_bits)
    }
}

/// Every grid point, present or explicitly absent.
pub const INSTANCES: &[GridPoint] = &[
    GridPoint {
        construction: "monolith",
        instance: Some("monolith-goldilocks-t8"),
        field: FieldId::Goldilocks,
        state_width: 8,
        absence: None,
    },
    GridPoint {
        construction: "monolith",
        instance: Some("monolith-goldilocks-t12"),
        field: FieldId::Goldilocks,
        state_width: 12,
        absence: None,
    },
    GridPoint {
        construction: "monolith",
        instance: Some("monolith-mersenne-t16"),
        field: FieldId::Mersenne31,
        state_width: 16,
        absence: None,
    },
    GridPoint {
        construction: "monolith",
        instance: Some("monolith-mersenne-t24"),
        field: FieldId::Mersenne31,
        state_width: 24,
        absence: None,
    },
    GridPoint {
        construction: "monolith",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "monolith",
        instance: None,
        field: FieldId::BabyBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "monolith",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 16,
        absence: Some(Absence::UndefinedForField),
    },
    GridPoint {
        construction: "monolith",
        instance: None,
        field: FieldId::KoalaBear,
        state_width: 24,
        absence: Some(Absence::UndefinedForField),
    },
];

const _: () = {
    assert!(INSTANCES.len() == 8);
    assert!(MAX_CONSTRAINT_DEGREE == 3);
};
