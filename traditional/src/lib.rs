//! The traditional hash functions, wrapped — the introduction table's baseline.
//!
//! BLAKE3, SHA-256 and Keccak-f[1600] are **not** POLICY §1 constructions and
//! do not sit on POLICY §3's grid. They are here for one purpose: the
//! introduction's claim that a Boolean-domain hash is expensive to prove needs
//! a measured number beside it, and a number measured by a second script would
//! not be comparable with the paper's main table (POLICY §11).
//!
//! So this crate is a construction-shaped crate that is deliberately outside
//! the construction machinery:
//!
//! * it implements [`harness::permutation::PermutationAir`], so its rows come
//!   out of the same `measure` call, the same configuration, the same seed and
//!   the same security target as every arithmetization-oriented row;
//! * it registers **no** [`harness::GridPoint`] and appears in neither
//!   `bench::CONSTRUCTIONS` nor `bench::REGISTERED`, so POLICY §3's coverage
//!   guard is untouched — there is no grid point here to be uncovered, and
//!   pretending otherwise would put eight "not on this grid" absences per hash
//!   into the main table.
//!
//! Everything here is **wrapped** (POLICY §1): upstream's AIR *and* upstream's
//! trace generator, forwarded verbatim. This crate contains no round function,
//! no constraint and no constant, so there is no second implementation to drift
//! from `p3-blake3-air`, `p3-sha256-air` and `p3-keccak-air`. Validation is
//! upstream's own — `p3-sha256-air` checks its generator against the `sha2`
//! crate, `p3-blake3-air` and `p3-keccak-air` against their reference
//! implementations — plus the harness's own `check_constraints`, degree pin and
//! release proof round trip (POLICY §4, §10).
//!
//! # What a "call" is here, and why the caption has to say it
//!
//! The four rows of the introduction table do not measure the same unit, and
//! the difference is not small:
//!
//! | AIR | one call is | rows per call | bytes absorbed |
//! |---|---|---|---|
//! | `p3-blake3-air` | one BLAKE3 compression, 7 rounds | 1 | 64 |
//! | `p3-sha256-air` | one SHA-256 compression, 64 rounds | 1 | 64 |
//! | `p3-keccak-air` | one Keccak-f\[1600\] permutation, 24 rounds | 24 | 136 (rate) |
//! | `p3-poseidon2-air` | one Poseidon2 permutation | 1/8 | 16 field elements |
//!
//! The table reports cost per call because that is what each AIR proves. The
//! byte counts above are what a reader needs to convert it into cost per
//! message, and they belong in the caption rather than in a column, because a
//! rate is a sponge parameter and POLICY §8 keeps modes of operation out of
//! scope.
//!
//! # Keccak-f is the one row that pads
//!
//! POLICY §6's "tables are always full" is a rule about arithmetizations this
//! repository writes. `p3-keccak-air` spends 24 rows on one permutation with
//! real transition constraints, so a full power-of-two table is arithmetically
//! impossible: `24 · n` is never a power of two. [`Labels::calls_in_trace`]
//! rounds down, which puts the padding at its minimum — at `2^17` rows, 5461
//! calls occupy 131 064 of 131 072 rows and 8 rows pad. That is reported, not
//! hidden, and it is why Keccak-f's call count in the table differs from the
//! other three.
//!
//! [`Labels::calls_in_trace`]: harness::Labels::calls_in_trace

pub mod instances;
