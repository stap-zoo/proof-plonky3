//! The three wrapped AIRs, adapted to the harness contract.
//!
//! Each newtype owns upstream's AIR and forwards `BaseAir` and `Air` verbatim,
//! so constraint evaluation here *is* upstream's (POLICY §1). The only code
//! this crate adds is the two trace entry points the harness asks for, and both
//! of them end in upstream's own `generate_trace_rows`.
//!
//! # Why a state is measured in 16-bit limbs
//!
//! [`harness::Labels::state_width`] counts base-field elements, and every
//! arithmetization-oriented row here runs over a 31-bit prime. A BLAKE3 or
//! SHA-256 input word is 32 bits and a Keccak lane is 64, so neither fits in
//! one `KoalaBear` element. The [`PermutationAir::generate_trace`] contract
//! therefore takes a state already split into **little-endian 16-bit limbs**:
//! 48 elements for the two 32-bit designs, 100 for Keccak-f. That is a property
//! of this adapter's input encoding, not of upstream's arithmetization —
//! upstream takes `u32`/`u64` words and this module converts.
//!
//! The measurement path never goes through that conversion: it draws whole
//! words from the shared seeded RNG and hands them straight to upstream, the
//! same shape `monolith` uses.

use core::fmt;

use harness::permutation::{Labels, PermutationAir};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Bits per field element of an encoded input state.
///
/// Sixteen, so that one limb is canonical in every prime this repository
/// measures over — including the 31-bit ones, where a 32-bit word is not.
pub const LIMB_BITS: usize = 16;

/// Words in one `p3-blake3-air` / `p3-sha256-air` input: a 16-word block
/// followed by an 8-word chaining state.
pub const WORDS_32: usize = 24;
/// Lanes in one Keccak-f\[1600\] state.
pub const LANES_64: usize = 25;

/// Encoded state width of the two 32-bit designs, in base-field elements.
pub const WIDTH_32: usize = WORDS_32 * 2;
/// Encoded state width of Keccak-f\[1600\], in base-field elements.
pub const WIDTH_64: usize = LANES_64 * 4;

/// Rows one Keccak-f permutation occupies: upstream spends one row per round.
pub const KECCAK_ROWS_PER_CALL: usize = 24;

/// The maximum constraint degree all three AIRs declare or evaluate to.
///
/// Pinned here and cross-checked against `get_max_constraint_degree` by
/// `harness::measure` on every measured row, so a wrong value is a panic rather
/// than a silently cheaper blowup.
pub const MAX_CONSTRAINT_DEGREE: usize = 3;

/// Read a limb back as an integer, rejecting anything a limb cannot hold.
fn limb<F: PrimeField64>(value: F) -> u64 {
    let raw = value.as_canonical_u64();
    assert!(
        raw < (1 << LIMB_BITS),
        "input element {raw} is not a {LIMB_BITS}-bit limb"
    );
    raw
}

/// Decode one call's input state from little-endian 16-bit limbs.
///
/// `LIMBS` is how many limbs one word takes: 2 for a `u32`, 4 for a `u64`.
fn decode<F: PrimeField64, const WORDS: usize, const LIMBS: usize>(state: &[F]) -> [u64; WORDS] {
    assert_eq!(
        state.len(),
        WORDS * LIMBS,
        "a call's input state must be {} limbs long",
        WORDS * LIMBS
    );
    core::array::from_fn(|word| {
        (0..LIMBS).fold(0u64, |acc, i| {
            acc | (limb(state[word * LIMBS + i]) << (i * LIMB_BITS))
        })
    })
}

/// The inverse of [`decode`], for the round-trip test and for callers building
/// a specific input.
#[must_use]
pub fn encode_words<F: PrimeField64, const LIMBS: usize>(words: &[u64]) -> Vec<F> {
    words
        .iter()
        .flat_map(|&word| {
            (0..LIMBS).map(move |i| F::from_u64((word >> (i * LIMB_BITS)) & ((1 << LIMB_BITS) - 1)))
        })
        .collect()
}

/// Generate one wrapper's `BaseAir`/`Air`/`Debug` forwarding and its two trace
/// entry points.
///
/// The three wrappers differ only in the upstream types they name and in how a
/// decoded state is handed to upstream's generator, so the shape is written
/// once. Anything that differed *per constraint* would defeat the purpose of
/// wrapping and is deliberately not expressible here.
macro_rules! wrapper {
    (
        $(#[$meta:meta])*
        $name:ident,
        upstream = $upstream:ty,
        upstream_new = $upstream_new:expr,
        generate = $generate:path,
        words = $words:expr,
        limbs = $limbs:expr,
        word_ty = $word_ty:ty,
        width = $width:expr,
        calls_per_row = $calls_per_row:expr,
        rows_per_call = $rows_per_call:expr,
        rounds = $rounds:expr,
    ) => {
        $(#[$meta])*
        pub struct $name {
            air: $upstream,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            /// Build the wrapper around upstream's AIR.
            ///
            /// Upstream's AIR carries no parameters — the round constants are
            /// compile-time tables inside the crate — so there is nothing to
            /// inject and nothing that could differ from upstream's own.
            #[must_use]
            pub const fn new() -> Self {
                Self { air: $upstream_new }
            }

            /// The wrapped upstream AIR.
            #[must_use]
            pub const fn upstream(&self) -> &$upstream {
                &self.air
            }

            /// Upstream's generator, over states decoded from limbs.
            fn trace_from_states<F: PrimeField64>(
                inputs: &[Vec<F>],
                extra_capacity_bits: usize,
            ) -> RowMajorMatrix<F> {
                let states = inputs
                    .iter()
                    .map(|state| {
                        decode::<F, $words, $limbs>(state)
                            .map(|word| <$word_ty>::try_from(word).expect("word fits its type"))
                    })
                    .collect();
                $generate(states, extra_capacity_bits)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("state_width", &$width)
                    .field("rounds", &$rounds)
                    .field("calls_per_row", &$calls_per_row)
                    .field("rows_per_call", &$rows_per_call)
                    .finish()
            }
        }

        impl<F> BaseAir<F> for $name {
            fn width(&self) -> usize {
                BaseAir::<F>::width(&self.air)
            }

            fn main_next_row_columns(&self) -> Vec<usize> {
                BaseAir::<F>::main_next_row_columns(&self.air)
            }

            fn max_constraint_degree(&self) -> Option<usize> {
                BaseAir::<F>::max_constraint_degree(&self.air)
            }
        }

        impl<AB: AirBuilder> Air<AB> for $name {
            #[inline]
            fn eval(&self, builder: &mut AB) {
                self.air.eval(builder);
            }
        }

        impl<SC: StarkGenericConfig> PermutationAir<Val<SC>, SC> for $name
        where
            Val<SC>: PrimeField64,
        {
            const LABELS: Labels = Labels {
                state_width: $width,
                calls_per_row: $calls_per_row,
                rows_per_call: $rows_per_call,
                rounds: $rounds,
                // Not a power map: these designs have no S-box in the
                // arithmetization-oriented sense. `0` is what `monolith`
                // already reports for its Bars, and the table prints it as a
                // dash rather than as an algebraic degree it is not.
                sbox_degree: 0,
                sbox_registers: 0,
                max_constraint_degree: MAX_CONSTRAINT_DEGREE,
            };

            fn generate_trace(
                &self,
                inputs: &[Vec<Val<SC>>],
                extra_capacity_bits: usize,
            ) -> RowMajorMatrix<Val<SC>> {
                Self::trace_from_states(inputs, extra_capacity_bits)
            }

            fn generate_trace_seeded(
                &self,
                num_calls: usize,
                seed: u64,
                extra_capacity_bits: usize,
            ) -> RowMajorMatrix<Val<SC>> {
                // The harness owns the seed and gives every construction the
                // same one, so two rows differ by the AIR and not by their
                // inputs (POLICY §7). Upstream's own convenience generator owns
                // a fixed seed of its own and is therefore not used.
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let inputs = (0..num_calls)
                    .map(|_| rng.random::<[$word_ty; $words]>())
                    .collect();
                $generate(inputs, extra_capacity_bits)
            }
        }
    };
}

wrapper! {
    /// BLAKE3 — one row is one compression of a 64-byte block (7 rounds).
    ///
    /// `p3-blake3-air` declares `main_next_row_columns() == vec![]`: the rows
    /// are independent, so the prover opens no shifted trace. That is the same
    /// property POLICY §6 asks a written vectorized AIR for, reached here
    /// without vectorization because one compression already fills a row.
    Blake3,
    upstream = p3_blake3_air::Blake3Air,
    upstream_new = p3_blake3_air::Blake3Air {},
    generate = p3_blake3_air::generate_trace_rows,
    words = WORDS_32,
    limbs = 2,
    word_ty = u32,
    width = WIDTH_32,
    calls_per_row = 1,
    rows_per_call = 1,
    rounds = 7,
}

wrapper! {
    /// SHA-256 — one row is one compression of a 64-byte block (64 rounds).
    ///
    /// Upstream's row carries the whole 64-round message schedule and all 64
    /// compression rounds unrolled, which is why it is by far the widest row
    /// here. Rows are independent, as for BLAKE3.
    Sha256,
    upstream = p3_sha256_air::Sha256Air,
    upstream_new = p3_sha256_air::Sha256Air,
    generate = p3_sha256_air::generate_trace_rows,
    words = WORDS_32,
    limbs = 2,
    word_ty = u32,
    width = WIDTH_32,
    calls_per_row = 1,
    rows_per_call = 1,
    rounds = 64,
}

wrapper! {
    /// Keccak-f\[1600\] — 24 rows is one permutation, one row per round.
    ///
    /// The only multi-row layout in this table. It carries real transition
    /// constraints and declares next-row columns, so the prover opens the
    /// shifted trace as well; the round-flag rotation is what ties the 24 rows
    /// of a call together. It is also the only row that pads: `24 · n` is never
    /// a power of two (POLICY §6's full-table rule is stated for
    /// arithmetizations this repository writes, and this one is upstream's).
    KeccakF,
    upstream = p3_keccak_air::KeccakAir,
    upstream_new = p3_keccak_air::KeccakAir {},
    generate = p3_keccak_air::generate_trace_rows,
    words = LANES_64,
    limbs = 4,
    word_ty = u64,
    width = WIDTH_64,
    calls_per_row = 1,
    rows_per_call = KECCAK_ROWS_PER_CALL,
    rounds = 24,
}

/// What one call of each wrapped design is, for the table's caption.
///
/// The four rows of the introduction table do not measure the same unit, and
/// the caption has to say so or the comparison overstates the
/// arithmetization-oriented side. This is the machine-readable half of that
/// sentence; the rate is reported, never chosen (POLICY §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unit {
    /// The construction name a measured row carries.
    pub construction: &'static str,
    /// The upstream crate the AIR comes from.
    pub upstream: &'static str,
    /// What one call is: a compression or a permutation.
    pub call: &'static str,
    /// Message bytes one call absorbs — the block size for a compression
    /// function, the sponge rate for a permutation.
    pub absorbed_bytes: usize,
}

/// One entry per wrapped design.
pub const UNITS: &[Unit] = &[
    Unit {
        construction: "blake3",
        upstream: "p3-blake3-air",
        call: "compression",
        absorbed_bytes: 64,
    },
    Unit {
        construction: "sha256",
        upstream: "p3-sha256-air",
        call: "compression",
        absorbed_bytes: 64,
    },
    Unit {
        construction: "keccak-f",
        upstream: "p3-keccak-air",
        call: "permutation",
        // Keccak-256's rate: 1088 bits, matching `p3-keccak-air`'s own
        // `RATE_BITS`.
        absorbed_bytes: 136,
    },
];

const _: () = {
    assert!(UNITS.len() == 3);
    assert!(WIDTH_32 == 48);
    assert!(WIDTH_64 == 100);
};
