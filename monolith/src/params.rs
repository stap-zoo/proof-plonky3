//! Parameters: structure from `../ref`, values from upstream.
//!
//! POLICY §4. What we wrap is validated against upstream's own implementation,
//! not against a reference KAT, and no constants are injected — a one-to-one
//! match with `../ref` is not required and is not attempted.
//!
//! But **structural parameters come from the reference in both cases**: width,
//! round counts, S-box degree. Cost depends on those and not on the values of
//! the round constants, so taking them from `../ref` is exactly what keeps a
//! wrapped row comparable to a written one.
//!
//! Where upstream's structure cannot be set to the reference's, this row
//! reports its own round counts rather than quietly reporting the reference's.

//! Monolith has no independent parameter table to transcribe: `p3-monolith`
//! deterministically derives its round constants from the paper domain
//! separator, field and width.  These constructors deliberately use that
//! upstream derivation and pass its constants and MDS matrix unchanged to
//! `p3-monolith-air`.

use p3_monolith::{
    MonolithBarsGoldilocks, MonolithBarsM31, MonolithGoldilocks8, MonolithMdsMatrixGoldilocks,
    MonolithMdsMatrixMersenne31, MonolithMersenne31,
};

use crate::instances::{GoldilocksT8, GoldilocksT12, MersenneT16, MersenneT24};

/// Full rounds which carry a round-constant addition.  The final sixth round
/// omits the addition, as the upstream permutation does.
pub const NUM_FULL_ROUNDS: usize = 5;
/// Total Bars/Bricks/Concrete rounds, including the final round.
pub const TOTAL_ROUNDS: usize = NUM_FULL_ROUNDS + 1;

/// Parameters for Monolith-64, Goldilocks t = 8.
#[must_use]
pub fn goldilocks_t8() -> GoldilocksT8 {
    let bars = MonolithBarsGoldilocks::<8>;
    let mds = MonolithMdsMatrixGoldilocks;
    let native: MonolithGoldilocks8<_, 8, NUM_FULL_ROUNDS> = MonolithGoldilocks8::new(bars, mds);
    GoldilocksT8::from_native(native)
}

/// Parameters for Monolith-64, Goldilocks t = 12.
#[must_use]
pub fn goldilocks_t12() -> GoldilocksT12 {
    let bars = MonolithBarsGoldilocks::<8>;
    let mds = MonolithMdsMatrixGoldilocks;
    let native: MonolithGoldilocks8<_, 12, NUM_FULL_ROUNDS> = MonolithGoldilocks8::new(bars, mds);
    GoldilocksT12::from_native(native)
}

/// Parameters for Monolith-31, Mersenne-31 t = 16.
#[must_use]
pub fn mersenne_t16() -> MersenneT16 {
    let bars = MonolithBarsM31;
    let mds = MonolithMdsMatrixMersenne31::<16, NUM_FULL_ROUNDS>::new();
    let native: MonolithMersenne31<_, 16, NUM_FULL_ROUNDS> = MonolithMersenne31::new(bars, mds);
    MersenneT16::from_native(native)
}

/// Parameters for Monolith-31, Mersenne-31 t = 24.
#[must_use]
pub fn mersenne_t24() -> MersenneT24 {
    let bars = MonolithBarsM31;
    let mds = MonolithMdsMatrixMersenne31::<24, NUM_FULL_ROUNDS>::new();
    let native: MonolithMersenne31<_, 24, NUM_FULL_ROUNDS> = MonolithMersenne31::new(bars, mds);
    MersenneT24::from_native(native)
}
