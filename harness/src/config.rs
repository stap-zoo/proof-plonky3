//! The configuration matrix: field × ZK, never per construction.
//!
//! POLICY §7's table, written as types. Everything a proof needs beyond the AIR
//! is fixed here — same MMCS hash, same challenger, same security target, same
//! blowup and query rules in every cell — so that a difference between two rows
//! is a difference between two arithmetizations.
//!
//! | field | non-ZK | ZK |
//! |---|---|---|
//! | Goldilocks | [`GoldilocksConfig`] | [`GoldilocksZkConfig`] |
//! | BabyBear, KoalaBear | [`TwoAdic31Config`] | [`TwoAdic31ZkConfig`] |
//! | Mersenne-31 | [`M31CircleConfig`] | unsupported (POLICY §7) |
//!
//! **ZK is exactly a choice of PCS.** `Pcs::ZK` is a const, true for
//! `HidingFriPcs` alone (`CirclePcs`, `TwoAdicFriPcs`, `StirPcs` are false), and
//! `StarkGenericConfig::is_zk()` just reads it.
//!
//! POLICY §7 is the single source of truth for the unsupported ZK Mersenne-31
//! cell. The complex-field types below exercise the component stack only; the
//! benchmark does not register them as a supported configuration.

use p3_baby_bear::BabyBear;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
    SerializingChallenger32, SerializingChallenger64,
};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::{BinomialExtensionField, Complex};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_koala_bear::KoalaBear;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::{Mersenne31, QM31};
use p3_security::error::ErrorBits;
use p3_security::fri::FriRegime;
use p3_security::grinding::GrindingSites;
use p3_security::logup::{self, LogUpAir};
use p3_security::shape::{InstanceShape, StarkAirParams};
use p3_security::stark::proven_security_report;
use p3_symmetric::{CompressionFunctionFromHasher, MerkleCap, SerializingHasher};
use p3_uni_stark::{ProvenSecurity, StarkConfig, StarkSecurityParams};
use p3_util::log2_ceil_u64;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Fixed across the whole matrix
// ---------------------------------------------------------------------------

/// The security target every row is held to.
///
/// Security is held fixed and blowup is not (POLICY §7): query count and
/// grinding bits are derived from this one number, and
/// `StarkSecurityParams::from_air(..)` is asserted against it per row rather
/// than assumed.
///
/// This is a proven-security target: every measurement evaluates Plonky3's
/// round-by-round UDR/LDR bounds and rejects a configuration below it. Moving
/// it moves every proof time and size in the tables.
pub const SECURITY_TARGET_BITS: usize = 100;

/// Largest symbolic constraint system covered by the shared configurations.
///
/// The current maximum is 12,792 constraints (Tip5's split t=16 layout), and
/// its pinned-number test is the assertion behind this headroom. Keeping the
/// envelope construction-independent lets one configuration be built once and
/// reused across the whole field × ZK cell. [`crate::measure::measure`] checks
/// the actual AIR again, so a future AIR which exceeds the envelope fails loud
/// rather than receiving an overstated security label.
pub const SECURITY_MAX_CONSTRAINTS: usize = 1 << 14;

/// Worst per-call LogUp denominator count among the registered Monolith
/// variants. Fraction-column packing does not remove denominator factors.
pub const SECURITY_MAX_LOOKUP_INTERACTIONS: usize = 192;
/// Widest lookup tuple at the interaction-count worst case.
pub const SECURITY_MAX_LOOKUP_TUPLE_WIDTH: usize = 2;
/// Extra denominator factors from Mersenne-31's two byte tables.
pub const SECURITY_MAX_LOOKUP_TABLE_FACTORS: usize = (1 << 8) + (1 << 7);
/// Per-call denominator count of the adjacent-pair variant.
pub const SECURITY_MAX_PAIR_LOOKUP_INTERACTIONS: usize = 96;
/// Paired messages keep four coordinates separate.
pub const SECURITY_MAX_PAIR_LOOKUP_TUPLE_WIDTH: usize = 4;
/// Once-per-proof paired-table factors in the largest supported field batch.
pub const SECURITY_MAX_PAIR_LOOKUP_TABLE_FACTORS: usize = (1 << 16) + (1 << 15);
/// Largest once-per-proof fixed-table height among registered variants.
pub const SECURITY_MAX_LOOKUP_TABLE_LOG_HEIGHT: usize = 16;

/// Binary Merkle trees use a single root in every configuration.
pub const MERKLE_CAP_HEIGHT: usize = 0;

/// Hiding FRI adds one random codeword, matching uni-STARK's ZK degree bound.
pub const NUM_RANDOM_CODEWORDS: usize = 1;

/// Collision resistance assumed of the MMCS hash, in bits.
///
/// Keccak-256 truncated to 32 bytes; the same in every cell, because the MMCS
/// hash is the harness's, not a construction's.
pub const COLLISION_RESISTANCE_BITS: usize = 128;

/// The one seed behind every measured input.
///
/// Random inputs are for measurement only, never for validation (POLICY §7).
/// One seed everywhere means two rows differ by their AIR and not by their
/// inputs.
pub const INPUT_SEED: u64 = 0x7a6b_6861_7368_736f;

/// A grid point: one field at one width.
///
/// The grid is POLICY §3's, and it is the same for every construction so that a
/// cross-construction row is a like-for-like row.
pub const GRID: &[(FieldId, usize)] = &[
    (FieldId::Goldilocks, 8),
    (FieldId::Goldilocks, 12),
    (FieldId::Mersenne31, 16),
    (FieldId::Mersenne31, 24),
    (FieldId::BabyBear, 16),
    (FieldId::BabyBear, 24),
    (FieldId::KoalaBear, 16),
    (FieldId::KoalaBear, 24),
];

/// The four fields of the grid.
///
/// A label for the tables and the selector for a configuration cell — never a
/// branch inside a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldId {
    /// `p = 2^64 - 2^32 + 1`.
    Goldilocks,
    /// `p = 2^31 - 1`. Not two-adic; see the module docs.
    Mersenne31,
    /// `p = 15 · 2^27 + 1`.
    BabyBear,
    /// `p = 127 · 2^24 + 1`.
    KoalaBear,
}

impl FieldId {
    /// The name used in an instance string and in the tables.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Goldilocks => "goldilocks",
            Self::Mersenne31 => "mersenne31",
            Self::BabyBear => "babybear",
            Self::KoalaBear => "koalabear",
        }
    }

    /// Widths this field carries in the grid (POLICY §3).
    #[must_use]
    pub const fn widths(self) -> &'static [usize] {
        match self {
            Self::Goldilocks => &[8, 12],
            Self::Mersenne31 | Self::BabyBear | Self::KoalaBear => &[16, 24],
        }
    }

    /// Bit-length of the challenge extension this field's cells operate in.
    ///
    /// A property of POLICY §7's matrix — degree 2 over Goldilocks, degree 4
    /// over the 31-bit primes, `QM31` over Mersenne-31 — and the one input the
    /// security derivation needs *before* a configuration exists. The runner
    /// asks whether a blowup can reach the target at all before building
    /// anything, so this value and each constructor's own `challenge_bits`
    /// must agree; `constructors_match_their_matrix_cells` asserts they do.
    #[must_use]
    pub const fn challenge_bits(self) -> usize {
        match self {
            Self::Goldilocks => 128,
            Self::Mersenne31 | Self::BabyBear | Self::KoalaBear => 124,
        }
    }
}

/// The ZK toggle. A configuration axis, never a construction's property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Zk {
    /// `TwoAdicFriPcs` / `CirclePcs`: `Pcs::ZK == false`.
    Off,
    /// `HidingFriPcs`: `Pcs::ZK == true`.
    On,
}

impl Zk {
    /// `1` when on, matching `StarkGenericConfig::is_zk()`.
    #[must_use]
    pub const fn is_zk(self) -> usize {
        match self {
            Self::Off => 0,
            Self::On => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// The MMCS and challenger, identical in every cell
// ---------------------------------------------------------------------------

/// Byte hash behind the Merkle tree and the challenger.
///
/// Keccak rather than a Poseidon2 sponge on purpose: an arithmetization-cost
/// comparison whose own Merkle hash is one of the constructions under test
/// would put a thumb on the scale in whichever direction that construction is
/// fast.
pub type ByteHash = Keccak256Hash;
/// Row hash: field elements serialized to bytes, then Keccak.
pub type FieldHash = SerializingHasher<ByteHash>;
/// Node compression, from the same byte hash.
pub type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
/// The trace MMCS.
pub type ValMmcs<F> = MerkleTreeMmcs<F, u8, FieldHash, Compress, 2, 32>;
/// The trace MMCS under ZK, which must also blind the openings.
pub type ValHidingMmcs<F> = MerkleTreeHidingMmcs<F, u8, FieldHash, Compress, StdRng, 2, 32, 32>;
/// The FRI-folding MMCS, over the challenge extension.
pub type ChallengeMmcs<F, EF> = ExtensionMmcs<F, EF, ValMmcs<F>>;
/// The FRI-folding MMCS under ZK.
pub type ChallengeHidingMmcs<F, EF> = ExtensionMmcs<F, EF, ValHidingMmcs<F>>;
/// Challenger for a 32-bit `Val`.
pub type Challenger32<F> = SerializingChallenger32<F, HashChallenger<u8, ByteHash, 32>>;
/// Challenger for a 64-bit `Val`.
pub type Challenger64<F> = SerializingChallenger64<F, HashChallenger<u8, ByteHash, 32>>;

/// Byte-backed challenger for the experimental complex-M31 component stack.
///
/// `SerializingChallenger32` cannot serve this cell because its observed field
/// must itself implement `PrimeField32`, while the trace field is
/// `Complex<Mersenne31>`. This wrapper serializes both M31 coordinates and
/// samples every algebra element coefficient-wise from the same Keccak byte
/// stream. Thus `Val = F_{p^2}` and `Challenge = F_{p^4}` share one transcript
/// without pretending the extension is a prime field.
#[derive(Clone, Debug)]
pub struct M31ComplexChallenger {
    inner: HashChallenger<u8, ByteHash, 32>,
}

impl M31ComplexChallenger {
    /// Start an empty Keccak transcript.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: HashChallenger::new(Vec::new(), ByteHash {}),
        }
    }

    fn sample_m31(&mut self) -> Mersenne31 {
        let log_size = log2_ceil_u64(Mersenne31::ORDER_U32.into());
        let mask = ((1_u64 << log_size) - 1) as u32;
        loop {
            let value = u32::from_le_bytes(self.inner.sample_array()) & mask;
            if value < Mersenne31::ORDER_U32 {
                return Mersenne31::new(value);
            }
        }
    }
}

impl Default for M31ComplexChallenger {
    fn default() -> Self {
        Self::new()
    }
}

impl CanObserve<M31Complex> for M31ComplexChallenger {
    fn observe(&mut self, value: M31Complex) {
        for coordinate in value.to_array() {
            self.inner
                .observe_slice(&coordinate.to_unique_u32().to_le_bytes());
        }
    }
}

impl<const N: usize> CanObserve<MerkleCap<M31Complex, [u8; N]>> for M31ComplexChallenger {
    fn observe(&mut self, cap: MerkleCap<M31Complex, [u8; N]>) {
        for digest in cap.into_roots() {
            self.inner.observe_slice(&digest);
        }
    }
}

impl<EF> CanSample<EF> for M31ComplexChallenger
where
    EF: BasedVectorSpace<Mersenne31>,
{
    fn sample(&mut self) -> EF {
        EF::from_basis_coefficients_fn(|_| self.sample_m31())
    }
}

impl CanSampleBits<usize> for M31ComplexChallenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < usize::BITS as usize, "bit sample must fit usize");
        let bytes = self.inner.sample_array::<{ size_of::<usize>() }>();
        usize::from_le_bytes(bytes) & ((1_usize << bits) - 1)
    }
}

impl GrindingChallenger for M31ComplexChallenger {
    type Witness = M31Complex;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        assert!(bits < 31, "M31 grinding supports fewer than 31 bits");
        if bits == 0 {
            return M31Complex::ZERO;
        }
        (0..Mersenne31::ORDER_U32)
            .map(|value| M31Complex::new_real(Mersenne31::new(value)))
            .find(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find a proof-of-work witness")
    }
}

impl FieldChallenger<M31Complex> for M31ComplexChallenger {}

// ---------------------------------------------------------------------------
// Supported cells and the experimental complex-M31 stack
// ---------------------------------------------------------------------------

/// Challenge extension over Goldilocks: degree 2 (POLICY §7).
pub type GoldilocksChallenge = BinomialExtensionField<Goldilocks, 2>;
/// Challenge extension over a two-adic 31-bit prime: degree 4.
pub type Challenge31<F> = BinomialExtensionField<F, 4>;
/// Two-adic quadratic extension used by the experimental component stack.
pub type M31Complex = Complex<Mersenne31>;

/// Goldilocks, non-ZK.
pub type GoldilocksConfig = StarkConfig<
    TwoAdicFriPcs<
        Goldilocks,
        Radix2DitParallel<Goldilocks>,
        ValMmcs<Goldilocks>,
        ChallengeMmcs<Goldilocks, GoldilocksChallenge>,
    >,
    GoldilocksChallenge,
    Challenger64<Goldilocks>,
>;

/// Goldilocks, ZK. Same challenge extension; the PCS is the only difference.
pub type GoldilocksZkConfig = StarkConfig<
    HidingFriPcs<
        Goldilocks,
        Radix2DitParallel<Goldilocks>,
        ValHidingMmcs<Goldilocks>,
        ChallengeHidingMmcs<Goldilocks, GoldilocksChallenge>,
        StdRng,
    >,
    GoldilocksChallenge,
    Challenger64<Goldilocks>,
>;

/// BabyBear or KoalaBear, non-ZK.
pub type TwoAdic31Config<F> = StarkConfig<
    TwoAdicFriPcs<F, Radix2DitParallel<F>, ValMmcs<F>, ChallengeMmcs<F, Challenge31<F>>>,
    Challenge31<F>,
    Challenger32<F>,
>;

/// BabyBear or KoalaBear, ZK.
pub type TwoAdic31ZkConfig<F> = StarkConfig<
    HidingFriPcs<
        F,
        Radix2DitParallel<F>,
        ValHidingMmcs<F>,
        ChallengeHidingMmcs<F, Challenge31<F>>,
        StdRng,
    >,
    Challenge31<F>,
    Challenger32<F>,
>;

/// Mersenne-31, non-ZK: the circle PCS, challenge `QM31`.
pub type M31CircleConfig = StarkConfig<
    CirclePcs<Mersenne31, ValMmcs<Mersenne31>, ChallengeMmcs<Mersenne31, QM31>>,
    QM31,
    Challenger32<Mersenne31>,
>;

/// Experimental hiding stack over `Complex<Mersenne31>`.
///
/// This proves that the Plonky3 primitives compose, but it is not a supported
/// Mersenne-31 benchmark configuration; see POLICY §7.
pub type M31ComplexZkConfig = StarkConfig<
    HidingFriPcs<
        M31Complex,
        p3_mersenne_31::Mersenne31ComplexRadix2Dit,
        ValHidingMmcs<M31Complex>,
        ChallengeHidingMmcs<M31Complex, QM31>,
        StdRng,
    >,
    QM31,
    M31ComplexChallenger,
>;

// ---------------------------------------------------------------------------
// What a built configuration carries
// ---------------------------------------------------------------------------

/// A configuration, built once, reused for every measurement in its cell.
///
/// Building this per measurement is how setup cost leaks into a hash's number,
/// which is why it is a value with a lifetime longer than any timed region
/// rather than something a runner constructs inline.
///
/// The three fields beside `sc` are what the security assertion of POLICY §7
/// needs and what `StarkGenericConfig` does not expose: the FRI shape lives
/// inside the PCS, and `Val`'s dimension over its prime subfield is a property
/// of the configuration.
#[derive(Debug, Clone)]
pub struct Configuration<SC> {
    /// The Plonky3 configuration itself.
    pub sc: SC,
    /// The FRI shape, mirrored out of the PCS for the security computation.
    pub fri: FriRegime,
    /// Bit-length of the field FRI operates in — the challenge extension.
    pub challenge_bits: usize,
    /// Base-field elements per `Val` cell.
    pub val_dimension: usize,
    /// Bits of security every row in this cell is asserted to reach.
    pub security_target_bits: usize,
    /// Whether this cell's PCS is hiding.
    pub zk: Zk,
}

// ---------------------------------------------------------------------------
// Proven-security derivation
// ---------------------------------------------------------------------------

/// Derive one binary-FRI regime for one trace height.
///
/// This is the single-height convenience wrapper around
/// [`derive_fri_regime_for_sweep`]. Measurement sweeps must use the latter:
/// the required query count is not monotone in trace height.
///
/// Query grinding starts at zero and stays there unless queries alone cannot
/// reach the target. Commit-phase grinding is increased only when the folding
/// challenge bound is the blocker; it cannot be repaired by adding queries.
/// This ordering avoids silently replacing linear proof-size cost with an
/// exponential prover-time cost.
#[must_use]
pub fn derive_fri_regime(
    challenge_bits: usize,
    log_blowup: usize,
    log_n: usize,
    zk: Zk,
) -> FriRegime {
    derive_fri_regime_for_sweep(challenge_bits, log_blowup, &[log_n], zk)
}

/// Derive one binary-FRI regime that reaches the target at every height in a
/// measurement sweep.
///
/// Query grinding starts at zero and stays there unless queries alone cannot
/// reach the target. Commit-phase grinding is increased only when the folding
/// challenge bound is the blocker; it cannot be repaired by adding queries.
/// This ordering avoids silently replacing linear proof-size cost with an
/// exponential prover-time cost.
///
/// # Panics
///
/// If the sweep is empty or no regime reaches the target at every requested
/// height.
#[must_use]
pub fn derive_fri_regime_for_sweep(
    challenge_bits: usize,
    log_blowup: usize,
    log_n: &[usize],
    zk: Zk,
) -> FriRegime {
    assert!(
        !log_n.is_empty(),
        "cannot derive a regime for an empty sweep"
    );
    try_derive_fri_regime_for_sweep(challenge_bits, log_blowup, log_n, zk).unwrap_or_else(|| {
        panic!(
            "100-bit proven security is unattainable for challenge_bits={challenge_bits}, \
             log_blowup={log_blowup}, log_n={log_n:?}, zk={zk:?}"
        )
    })
}

/// The same derivation, answering instead of aborting.
///
/// A low blowup is where this matters: the primary reading of POLICY §11 runs
/// each AIR at the smallest blowup its degree admits, and a low enough rate can
/// put the target out of reach of any query count. That is a fact about the
/// arithmetization's degree and belongs in the report — POLICY §7 forbids a
/// silently weaker row, not a row that says it cannot exist. The runner asks
/// this before building a configuration, so the answer is a skip with a reason
/// rather than a dead run.
#[must_use]
pub fn try_derive_fri_regime(
    challenge_bits: usize,
    log_blowup: usize,
    log_n: usize,
    zk: Zk,
) -> Option<FriRegime> {
    try_derive_fri_regime_for_sweep(challenge_bits, log_blowup, &[log_n], zk)
}

/// The sweep-aware derivation, answering instead of aborting.
///
/// Every candidate regime is checked against every requested trace height.
/// Checking the exact sweep matters: Plonky3's round-by-round bounds are not
/// monotone in trace height, so neither endpoint is a sound proxy for all of
/// the heights between them.
#[must_use]
pub fn try_derive_fri_regime_for_sweep(
    challenge_bits: usize,
    log_blowup: usize,
    log_n: &[usize],
    zk: Zk,
) -> Option<FriRegime> {
    if log_n.is_empty() {
        return None;
    }
    let trace_lens: Vec<_> = log_n
        .iter()
        .map(|&log_n| {
            let proof_log_n = log_n
                .checked_add(zk.is_zk())
                .expect("trace height overflows usize");
            assert!(
                proof_log_n < usize::BITS as usize,
                "trace height overflows usize"
            );
            1usize << proof_log_n
        })
        .collect();

    // This is the largest degree the selected quotient blowup admits. Actual
    // AIR degree and constraint count are re-evaluated by `measure`.
    let max_degree = (1 << log_blowup) + usize::from(zk == Zk::Off);

    for commit_pow_bits in 0..31 {
        // First determine whether this grinding level removes every non-query
        // blocker. A generous finite query cap keeps a broken derivation from
        // looping forever.
        let max_queries = 4096;
        let ceiling = FriRegime {
            log_blowup,
            num_queries: max_queries,
            log_final_poly_len: 0,
            max_log_arity: 1,
            commit_pow_bits,
            query_pow_bits: 0,
        };
        let ceiling_params = StarkSecurityParams::new(
            ceiling,
            challenge_bits,
            COLLISION_RESISTANCE_BITS,
            SECURITY_MAX_CONSTRAINTS,
            max_degree,
            2,
        );
        if !trace_lens.iter().zip(log_n).all(|(&trace_len, &log_n)| {
            ProvenSecurity::compute(&ceiling_params, trace_len).security_bits()
                >= SECURITY_TARGET_BITS
                && lookup_envelope_security_bits(ceiling, challenge_bits, max_degree, log_n, zk)
                    >= SECURITY_TARGET_BITS
        }) {
            continue;
        }

        let num_queries = (1..=max_queries)
            .find(|&num_queries| {
                let regime = FriRegime {
                    num_queries,
                    ..ceiling
                };
                let params = StarkSecurityParams::new(
                    regime,
                    challenge_bits,
                    COLLISION_RESISTANCE_BITS,
                    SECURITY_MAX_CONSTRAINTS,
                    max_degree,
                    2,
                );
                trace_lens.iter().zip(log_n).all(|(&trace_len, &log_n)| {
                    ProvenSecurity::compute(&params, trace_len).security_bits()
                        >= SECURITY_TARGET_BITS
                        && lookup_envelope_security_bits(
                            regime,
                            challenge_bits,
                            max_degree,
                            log_n,
                            zk,
                        ) >= SECURITY_TARGET_BITS
                })
            })
            .expect("security ceiling was reachable");

        return Some(FriRegime {
            num_queries,
            ..ceiling
        });
    }

    None
}

fn lookup_envelope_security_bits(
    fri: FriRegime,
    challenge_bits: usize,
    max_degree: usize,
    log_n: usize,
    zk: Zk,
) -> usize {
    let shape = InstanceShape {
        log_trace_length: log_n.max(SECURITY_MAX_LOOKUP_TABLE_LOG_HEIGHT) + zk.is_zk(),
        modulus_bits: challenge_bits,
        collision_resistance: COLLISION_RESISTANCE_BITS,
        num_batched_functions: 1,
    };
    let air = StarkAirParams {
        num_constraints: SECURITY_MAX_CONSTRAINTS,
        max_constraint_degree: max_degree,
        max_combo: 2,
    };
    let report = proven_security_report(&fri, &air, &shape, &[], &GrindingSites::NONE);
    let byte_denominators =
        SECURITY_MAX_LOOKUP_INTERACTIONS * (1usize << log_n) + SECURITY_MAX_LOOKUP_TABLE_FACTORS;
    let pair_denominators = SECURITY_MAX_PAIR_LOOKUP_INTERACTIONS * (1usize << log_n)
        + SECURITY_MAX_PAIR_LOOKUP_TABLE_FACTORS;
    let fingerprint_shape = InstanceShape {
        log_trace_length: 0,
        ..shape
    };
    let byte_fingerprint = logup::fingerprint_error(
        &LogUpAir {
            num_interactions: byte_denominators,
            max_message_width: SECURITY_MAX_LOOKUP_TUPLE_WIDTH,
        },
        &fingerprint_shape,
    );
    let pair_fingerprint = logup::fingerprint_error(
        &LogUpAir {
            num_interactions: pair_denominators,
            max_message_width: SECURITY_MAX_PAIR_LOOKUP_TUPLE_WIDTH,
        },
        &fingerprint_shape,
    );
    // The shared configuration must admit either real variant. They are
    // alternatives, so select the larger error rather than unioning them.
    let fingerprint = ErrorBits::min(&[byte_fingerprint, pair_fingerprint]);
    let union = |bits: f64| ErrorBits::sum(&[ErrorBits::from_log2(bits), fingerprint]).floor();
    report.ldr.as_ref().map_or_else(
        || union(report.udr.security_bits()),
        |ldr| union(report.udr.security_bits()).max(union(ldr.security_bits())),
    )
}

fn fri_parameters<M>(
    mmcs: M,
    challenge_bits: usize,
    log_blowup: usize,
    log_n: &[usize],
    zk: Zk,
) -> (FriParameters<M>, FriRegime) {
    let regime = derive_fri_regime_for_sweep(challenge_bits, log_blowup, log_n, zk);
    let params = FriParameters {
        log_blowup: regime.log_blowup,
        log_final_poly_len: regime.log_final_poly_len,
        max_log_arity: regime.max_log_arity,
        num_queries: regime.num_queries,
        commit_proof_of_work_bits: regime.commit_pow_bits,
        query_proof_of_work_bits: regime.query_pow_bits,
        mmcs,
    };
    (params, regime)
}

fn mmcs<F>() -> ValMmcs<F> {
    MerkleTreeMmcs::new(
        SerializingHasher::new(ByteHash {}),
        CompressionFunctionFromHasher::new(ByteHash {}),
        MERKLE_CAP_HEIGHT,
    )
}

fn hiding_mmcs<F>() -> ValHidingMmcs<F> {
    let mut system_rng = rand::rng();
    MerkleTreeHidingMmcs::new(
        SerializingHasher::new(ByteHash {}),
        CompressionFunctionFromHasher::new(ByteHash {}),
        MERKLE_CAP_HEIGHT,
        StdRng::from_rng(&mut system_rng),
    )
}

fn challenger32<F: PrimeField32>() -> Challenger32<F> {
    SerializingChallenger32::from_hasher(Vec::new(), ByteHash {})
}

fn challenger64() -> Challenger64<Goldilocks> {
    SerializingChallenger64::from_hasher(Vec::new(), ByteHash {})
}

/// Build the Goldilocks non-ZK cell.
#[must_use]
pub fn goldilocks(log_blowup: usize, log_n: &[usize]) -> Configuration<GoldilocksConfig> {
    let val_mmcs = mmcs::<Goldilocks>();
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let (fri, regime) = fri_parameters(challenge_mmcs, 128, log_blowup, log_n, Zk::Off);
    let pcs = TwoAdicFriPcs::new(Radix2DitParallel::default(), val_mmcs, fri);
    Configuration {
        sc: StarkConfig::new(pcs, challenger64()),
        fri: regime,
        challenge_bits: 128,
        val_dimension: 1,
        security_target_bits: SECURITY_TARGET_BITS,
        zk: Zk::Off,
    }
}

/// Build the Goldilocks ZK cell.
#[must_use]
pub fn goldilocks_zk(log_blowup: usize, log_n: &[usize]) -> Configuration<GoldilocksZkConfig> {
    let val_mmcs = hiding_mmcs::<Goldilocks>();
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let (fri, regime) = fri_parameters(challenge_mmcs, 128, log_blowup, log_n, Zk::On);
    let mut system_rng = rand::rng();
    let pcs = HidingFriPcs::new(
        Radix2DitParallel::default(),
        val_mmcs,
        fri,
        NUM_RANDOM_CODEWORDS,
        StdRng::from_rng(&mut system_rng),
    );
    Configuration {
        sc: StarkConfig::new(pcs, challenger64()),
        fri: regime,
        challenge_bits: 128,
        val_dimension: 1,
        security_target_bits: SECURITY_TARGET_BITS,
        zk: Zk::On,
    }
}

fn two_adic_31<F>(log_blowup: usize, log_n: &[usize]) -> Configuration<TwoAdic31Config<F>>
where
    F: PrimeField32 + p3_field::TwoAdicField,
    BinomialExtensionField<F, 4>: p3_field::TwoAdicField,
{
    let val_mmcs = mmcs::<F>();
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let (fri, regime) = fri_parameters(challenge_mmcs, 124, log_blowup, log_n, Zk::Off);
    let pcs = TwoAdicFriPcs::new(Radix2DitParallel::default(), val_mmcs, fri);
    Configuration {
        sc: StarkConfig::new(pcs, challenger32()),
        fri: regime,
        challenge_bits: 124,
        val_dimension: 1,
        security_target_bits: SECURITY_TARGET_BITS,
        zk: Zk::Off,
    }
}

fn two_adic_31_zk<F>(log_blowup: usize, log_n: &[usize]) -> Configuration<TwoAdic31ZkConfig<F>>
where
    F: PrimeField32 + p3_field::TwoAdicField,
    BinomialExtensionField<F, 4>: p3_field::TwoAdicField,
{
    let val_mmcs = hiding_mmcs::<F>();
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let (fri, regime) = fri_parameters(challenge_mmcs, 124, log_blowup, log_n, Zk::On);
    let mut system_rng = rand::rng();
    let pcs = HidingFriPcs::new(
        Radix2DitParallel::default(),
        val_mmcs,
        fri,
        NUM_RANDOM_CODEWORDS,
        StdRng::from_rng(&mut system_rng),
    );
    Configuration {
        sc: StarkConfig::new(pcs, challenger32()),
        fri: regime,
        challenge_bits: 124,
        val_dimension: 1,
        security_target_bits: SECURITY_TARGET_BITS,
        zk: Zk::On,
    }
}

/// Build the BabyBear non-ZK cell.
#[must_use]
pub fn babybear(log_blowup: usize, log_n: &[usize]) -> Configuration<TwoAdic31Config<BabyBear>> {
    two_adic_31(log_blowup, log_n)
}

/// Build the BabyBear ZK cell.
#[must_use]
pub fn babybear_zk(
    log_blowup: usize,
    log_n: &[usize],
) -> Configuration<TwoAdic31ZkConfig<BabyBear>> {
    two_adic_31_zk(log_blowup, log_n)
}

/// Build the KoalaBear non-ZK cell.
#[must_use]
pub fn koalabear(log_blowup: usize, log_n: &[usize]) -> Configuration<TwoAdic31Config<KoalaBear>> {
    two_adic_31(log_blowup, log_n)
}

/// Build the KoalaBear ZK cell.
#[must_use]
pub fn koalabear_zk(
    log_blowup: usize,
    log_n: &[usize],
) -> Configuration<TwoAdic31ZkConfig<KoalaBear>> {
    two_adic_31_zk(log_blowup, log_n)
}

/// Build the Mersenne-31 circle-PCS cell.
#[must_use]
pub fn mersenne31(log_blowup: usize, log_n: &[usize]) -> Configuration<M31CircleConfig> {
    let val_mmcs = mmcs::<Mersenne31>();
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let (fri, regime) = fri_parameters(challenge_mmcs, 124, log_blowup, log_n, Zk::Off);
    let pcs = CirclePcs::new(val_mmcs, fri);
    Configuration {
        sc: StarkConfig::new(pcs, challenger32()),
        fri: regime,
        challenge_bits: 124,
        val_dimension: 1,
        security_target_bits: SECURITY_TARGET_BITS,
        zk: Zk::Off,
    }
}

/// Build the experimental hiding stack over `Complex<Mersenne31>`.
///
/// This constructor is exercised as a component round trip but is deliberately
/// absent from the benchmark plan (POLICY §7).
#[must_use]
pub fn mersenne31_zk(log_blowup: usize, log_n: &[usize]) -> Configuration<M31ComplexZkConfig> {
    let val_mmcs = hiding_mmcs::<M31Complex>();
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let (fri, regime) = fri_parameters(challenge_mmcs, 124, log_blowup, log_n, Zk::On);
    let mut system_rng = rand::rng();
    let pcs = HidingFriPcs::new(
        p3_mersenne_31::Mersenne31ComplexRadix2Dit,
        val_mmcs,
        fri,
        NUM_RANDOM_CODEWORDS,
        StdRng::from_rng(&mut system_rng),
    );
    Configuration {
        sc: StarkConfig::new(pcs, M31ComplexChallenger::new()),
        fri: regime,
        challenge_bits: 124,
        val_dimension: 2,
        security_target_bits: SECURITY_TARGET_BITS,
        zk: Zk::On,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure;
    use crate::permutation::{Labels, PermutationAir};
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_field::Field;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_uni_stark::StarkGenericConfig;

    #[derive(Clone, Copy, Debug)]
    struct OneAir;

    impl<F> BaseAir<F> for OneAir {
        fn width(&self) -> usize {
            1
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            Vec::new()
        }

        fn max_constraint_degree(&self) -> Option<usize> {
            Some(1)
        }
    }

    impl<AB: AirBuilder> Air<AB> for OneAir {
        fn eval(&self, builder: &mut AB) {
            builder.assert_one(builder.main().current(0).expect("one column"));
        }
    }

    impl<F, SC> PermutationAir<F, SC> for OneAir
    where
        F: Field,
        SC: StarkGenericConfig,
    {
        const LABELS: Labels = Labels {
            state_width: 1,
            calls_per_row: 1,
            rows_per_call: 1,
            rounds: 0,
            sbox_degree: 0,
            sbox_registers: 0,
            max_constraint_degree: 1,
        };

        fn generate_trace(
            &self,
            inputs: &[Vec<F>],
            extra_capacity_bits: usize,
        ) -> RowMajorMatrix<F> {
            let mut values = Vec::with_capacity(inputs.len() << extra_capacity_bits);
            values.resize(inputs.len(), F::ONE);
            RowMajorMatrix::new(values, 1)
        }

        fn generate_trace_seeded(
            &self,
            num_calls: usize,
            _seed: u64,
            extra_capacity_bits: usize,
        ) -> RowMajorMatrix<F> {
            let inputs = vec![Vec::new(); num_calls];
            <Self as PermutationAir<F, SC>>::generate_trace(self, &inputs, extra_capacity_bits)
        }
    }

    #[test]
    fn derivation_reaches_the_target_for_every_cell() {
        let log_n = [10, 12, 14];
        for challenge_bits in [124, 128] {
            for zk in [Zk::Off, Zk::On] {
                for log_blowup in 1..=4 {
                    let regime =
                        derive_fri_regime_for_sweep(challenge_bits, log_blowup, &log_n, zk);
                    let max_degree = (1 << log_blowup) + usize::from(zk == Zk::Off);
                    let params = StarkSecurityParams::new(
                        regime,
                        challenge_bits,
                        COLLISION_RESISTANCE_BITS,
                        SECURITY_MAX_CONSTRAINTS,
                        max_degree,
                        2,
                    );
                    for &log_n in &log_n {
                        let trace_len = 1 << (log_n + zk.is_zk());
                        assert!(
                            ProvenSecurity::compute(&params, trace_len).security_bits()
                                >= SECURITY_TARGET_BITS,
                            "challenge={challenge_bits} blowup={log_blowup} \
                             log_n={log_n} zk={zk:?}"
                        );
                    }
                }
            }
        }
    }

    /// A sweep's most demanding height need not be its largest. At the minimum
    /// blowup, the default sweep needs one more query at `2^10` than at `2^14`.
    /// Pin both the query count and the height that makes 241 insufficient so
    /// the runner cannot regress to choosing either endpoint by assumption.
    #[test]
    fn sweep_derivation_uses_the_most_demanding_height() {
        let log_n = [10, 12, 14];
        let regime = derive_fri_regime_for_sweep(128, 1, &log_n, Zk::Off);
        assert_eq!(regime.num_queries, 242);

        let max_degree = (1 << regime.log_blowup) + 1;
        let params = StarkSecurityParams::new(
            regime,
            128,
            COLLISION_RESISTANCE_BITS,
            SECURITY_MAX_CONSTRAINTS,
            max_degree,
            2,
        );
        for &log_n in &log_n {
            assert!(
                ProvenSecurity::compute(&params, 1 << log_n).security_bits()
                    >= SECURITY_TARGET_BITS,
                "sweep regime misses log_n={log_n}"
            );
        }

        let weaker = FriRegime {
            num_queries: regime.num_queries - 1,
            ..regime
        };
        let weaker_params = StarkSecurityParams::new(
            weaker,
            128,
            COLLISION_RESISTANCE_BITS,
            SECURITY_MAX_CONSTRAINTS,
            max_degree,
            2,
        );
        assert!(
            ProvenSecurity::compute(&weaker_params, 1 << 10).security_bits() < SECURITY_TARGET_BITS
        );
    }

    #[test]
    fn constructors_match_their_modes() {
        assert_eq!(goldilocks(3, &[4]).sc.is_zk(), 0);
        assert_eq!(goldilocks_zk(3, &[4]).sc.is_zk(), 1);
        assert_eq!(babybear(3, &[4]).sc.is_zk(), 0);
        assert_eq!(babybear_zk(3, &[4]).sc.is_zk(), 1);
        assert_eq!(koalabear(3, &[4]).sc.is_zk(), 0);
        assert_eq!(koalabear_zk(3, &[4]).sc.is_zk(), 1);
        assert_eq!(mersenne31(3, &[4]).sc.is_zk(), 0);
        let m31_zk = mersenne31_zk(3, &[4]);
        assert_eq!(m31_zk.sc.is_zk(), 1);
        assert_eq!(m31_zk.val_dimension, 2);

        // The runner decides whether a blowup is reachable *before* it builds
        // anything, from `FieldId::challenge_bits` alone. A constructor that
        // disagreed with it would have its rows skipped, or attempted, on the
        // wrong number.
        assert_eq!(
            goldilocks(3, &[4]).challenge_bits,
            FieldId::Goldilocks.challenge_bits()
        );
        assert_eq!(
            goldilocks_zk(3, &[4]).challenge_bits,
            FieldId::Goldilocks.challenge_bits()
        );
        assert_eq!(
            babybear(3, &[4]).challenge_bits,
            FieldId::BabyBear.challenge_bits()
        );
        assert_eq!(
            babybear_zk(3, &[4]).challenge_bits,
            FieldId::BabyBear.challenge_bits()
        );
        assert_eq!(
            koalabear(3, &[4]).challenge_bits,
            FieldId::KoalaBear.challenge_bits()
        );
        assert_eq!(
            koalabear_zk(3, &[4]).challenge_bits,
            FieldId::KoalaBear.challenge_bits()
        );
        assert_eq!(
            mersenne31(3, &[4]).challenge_bits,
            FieldId::Mersenne31.challenge_bits()
        );
        assert_eq!(m31_zk.challenge_bits, FieldId::Mersenne31.challenge_bits());
    }

    /// Every blowup the primary reading can ask for must be *reachable*, or the
    /// runner reports a skip instead of a row. This records which ones are:
    /// with the floor at `log_blowup = 1`, the whole degree slice is.
    #[test]
    fn the_measured_blowups_reach_the_target() {
        let log_n = [10, 12, 14];
        for field in [
            FieldId::Goldilocks,
            FieldId::Mersenne31,
            FieldId::BabyBear,
            FieldId::KoalaBear,
        ] {
            for zk in [Zk::Off, Zk::On] {
                for degree in 2..=9usize {
                    let log_blowup = crate::blowup::measured_log_blowup(degree, zk == Zk::On);
                    assert!(
                        try_derive_fri_regime_for_sweep(
                            field.challenge_bits(),
                            log_blowup,
                            &log_n,
                            zk,
                        )
                        .is_some(),
                        "{field:?} zk={zk:?} degree={degree} log_blowup={log_blowup}"
                    );
                }
            }
        }
    }

    /// One real quotient/PCS/transcript round trip per supported cell, plus the
    /// experimental complex-M31 component stack. Construction tests exercise
    /// their own AIRs; this keeps a broken DFT, MMCS, challenger or hiding
    /// configuration local to the harness instead of surfacing eleven crates
    /// later.
    #[test]
    fn supported_cells_and_m31_components_prove_and_verify() {
        let air = OneAir;
        let log_n = 3;
        let log_blowup = 3;

        measure(&goldilocks(log_blowup, &[log_n]), &air, log_n);
        measure(&goldilocks_zk(log_blowup, &[log_n]), &air, log_n);
        measure(&babybear(log_blowup, &[log_n]), &air, log_n);
        measure(&babybear_zk(log_blowup, &[log_n]), &air, log_n);
        measure(&koalabear(log_blowup, &[log_n]), &air, log_n);
        measure(&koalabear_zk(log_blowup, &[log_n]), &air, log_n);
        measure(&mersenne31(log_blowup, &[log_n]), &air, log_n);
        measure(&mersenne31_zk(log_blowup, &[log_n]), &air, log_n);
    }
}
