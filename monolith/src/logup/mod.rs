//! The lookup-backed Monolith arithmetization.
//!
//! This module is separate from [`crate::instances`], which remains the exact
//! wrapper around `p3-monolith-air`. Each call stays in one main row; only
//! LogUp's auxiliary accumulator has a next-row constraint.

use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use p3_monolith::{
    MonolithBarsGoldilocks, MonolithBarsM31, MonolithGoldilocks8, MonolithMdsMatrixGoldilocks,
    MonolithMdsMatrixMersenne31, MonolithMersenne31,
};

use crate::params::NUM_FULL_ROUNDS;

use self::air::MonolithLogupAir;
use harness::gadgets::canonical_word::WordKind;

// The two frontier axes are the harness's, not Monolith's; they are re-exported
// here because [`VARIANTS`] is written in them.
pub use harness::lookup::{FractionPacking, LookupGranularity};

pub mod air;
pub mod batch;
pub mod columns;
pub mod generation;
pub mod tables;

/// One point in the lookup granularity × fraction-packing frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogupVariant {
    /// Stable benchmark label.
    pub name: &'static str,
    /// Width-two byte messages or width-four adjacent-pair messages.
    pub granularity: LookupGranularity,
    /// Denominators folded into each auxiliary fraction column.
    pub packing: FractionPacking,
}

/// Every step-4 Monolith lookup variant, in increasing packing degree within
/// each table granularity.
pub const VARIANTS: &[LogupVariant] = &[
    LogupVariant {
        name: "logup-byte-frac1",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::One,
    },
    LogupVariant {
        name: "logup-byte-frac2",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Two,
    },
    LogupVariant {
        name: "logup-byte-frac4",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Four,
    },
    LogupVariant {
        name: "logup-byte-frac8",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Eight,
    },
    LogupVariant {
        name: "logup-pair-frac1",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::One,
    },
    LogupVariant {
        name: "logup-pair-frac2",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Two,
    },
    LogupVariant {
        name: "logup-pair-frac4",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Four,
    },
    LogupVariant {
        name: "logup-pair-frac8",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Eight,
    },
];

/// Lookup-backed Monolith-64, Goldilocks t = 8.
pub type GoldilocksT8 = MonolithLogupAir<Goldilocks, 8, NUM_FULL_ROUNDS, 4, 8, 2>;
/// Lookup-backed Monolith-64, Goldilocks t = 12.
pub type GoldilocksT12 = MonolithLogupAir<Goldilocks, 12, NUM_FULL_ROUNDS, 4, 8, 2>;
/// Lookup-backed Monolith-31, Mersenne-31 t = 16.
pub type MersenneT16 = MonolithLogupAir<Mersenne31, 16, NUM_FULL_ROUNDS, 8, 4, 1>;
/// Lookup-backed Monolith-31, Mersenne-31 t = 24.
pub type MersenneT24 = MonolithLogupAir<Mersenne31, 24, NUM_FULL_ROUNDS, 8, 4, 1>;

/// Parameters for the Goldilocks t = 8 lookup AIR.
#[must_use]
pub fn goldilocks_t8() -> GoldilocksT8 {
    goldilocks_t8_with(LookupGranularity::Byte, FractionPacking::One)
}

/// Parameters for a selected Goldilocks t = 8 lookup variant.
#[must_use]
pub fn goldilocks_t8_with(
    granularity: LookupGranularity,
    packing: FractionPacking,
) -> GoldilocksT8 {
    let native: MonolithGoldilocks8<_, 8, NUM_FULL_ROUNDS> =
        MonolithGoldilocks8::new(MonolithBarsGoldilocks::<8>, MonolithMdsMatrixGoldilocks);
    MonolithLogupAir::from_native(native, WordKind::Goldilocks, granularity, packing)
}

/// Parameters for the Goldilocks t = 12 lookup AIR.
#[must_use]
pub fn goldilocks_t12() -> GoldilocksT12 {
    goldilocks_t12_with(LookupGranularity::Byte, FractionPacking::One)
}

/// Parameters for a selected Goldilocks t = 12 lookup variant.
#[must_use]
pub fn goldilocks_t12_with(
    granularity: LookupGranularity,
    packing: FractionPacking,
) -> GoldilocksT12 {
    let native: MonolithGoldilocks8<_, 12, NUM_FULL_ROUNDS> =
        MonolithGoldilocks8::new(MonolithBarsGoldilocks::<8>, MonolithMdsMatrixGoldilocks);
    MonolithLogupAir::from_native(native, WordKind::Goldilocks, granularity, packing)
}

/// Parameters for the Mersenne-31 t = 16 lookup AIR.
#[must_use]
pub fn mersenne_t16() -> MersenneT16 {
    mersenne_t16_with(LookupGranularity::Byte, FractionPacking::One)
}

/// Parameters for a selected Mersenne-31 t = 16 lookup variant.
#[must_use]
pub fn mersenne_t16_with(granularity: LookupGranularity, packing: FractionPacking) -> MersenneT16 {
    let native: MonolithMersenne31<_, 16, NUM_FULL_ROUNDS> = MonolithMersenne31::new(
        MonolithBarsM31,
        MonolithMdsMatrixMersenne31::<16, NUM_FULL_ROUNDS>::new(),
    );
    MonolithLogupAir::from_native(native, WordKind::Mersenne31, granularity, packing)
}

/// Parameters for the Mersenne-31 t = 24 lookup AIR.
#[must_use]
pub fn mersenne_t24() -> MersenneT24 {
    mersenne_t24_with(LookupGranularity::Byte, FractionPacking::One)
}

/// Parameters for a selected Mersenne-31 t = 24 lookup variant.
#[must_use]
pub fn mersenne_t24_with(granularity: LookupGranularity, packing: FractionPacking) -> MersenneT24 {
    let native: MonolithMersenne31<_, 24, NUM_FULL_ROUNDS> = MonolithMersenne31::new(
        MonolithBarsM31,
        MonolithMdsMatrixMersenne31::<24, NUM_FULL_ROUNDS>::new(),
    );
    MonolithLogupAir::from_native(native, WordKind::Mersenne31, granularity, packing)
}
