//! Shared in-AIR building blocks.
//!
//! What POLICY §6 expects to end up here, once each has two users: power-map
//! S-boxes with register splitting, witnessed inverse power, Flystel /
//! Lai-Massey, limb decomposition and canonicity, linear layers.
//!
//! Present: [`power_map`] (Anemoi, Neptune, pSquareHash, Griffin, GMiMC,
//! GMiMC2 — whose `alpha = 2^k` is the one non-bijective exponent it serves),
//! [`inverse_power_map`] (Anemoi's closed Flystel, Griffin's first S-box word)
//! and [`canonical_word`] (Monolith's lookup-backed Bars; Tip5's
//! split-and-lookup word is its second user). Dense linear layers are upstream's
//! `p3_mds::util::mds_multiply` rather than a gadget here — a shared building
//! block does not have to be ours.
//!
//! # What a gadget is, here
//!
//! Not an algebra library. **A pair**, plus an oracle:
//!
//! * the **constraint side** — what `air.rs` evaluates over a row's cells;
//! * the **witness side** — what `generation.rs` writes into those cells;
//! * the **native oracle** it is tested against.
//!
//! The pairing is the whole point (POLICY §6). The two sides will drift; they
//! are kept line-for-line parallel — same order, same helper names, same
//! per-round split — and a gadget lands here as a pair or not at all.
//!
//! # What every gadget carries
//!
//! * A doc comment with its argument: **why this arithmetization**, what it
//!   costs in cells and in degree, and the trap in using it wrong.
//! * For each witnessed value, the name of the constraint that pins it, in a
//!   comment *where it is witnessed* (POLICY §9). Every cell is prover-chosen:
//!   a committed `x³` is free money until an `assert_eq` ties it to `x`, and so
//!   is every limb, every inverse-power output, every canonicity flag.
//! * Tests against its native oracle **before** it is wired into any AIR, on
//!   random inputs plus `0`, `1` and the field's edge values — including the
//!   non-canonical representations the arithmetic tolerates but a decomposition
//!   must not.
//! * Its own negative test: corrupt one witnessed cell, assert the constraints
//!   reject it.
//!
//! # Field genericity
//!
//! The same code monomorphizes over Goldilocks and all three 31-bit primes.
//! Carry the loosest bound that works and reach for `PrimeField32` /
//! `PrimeField64` only where the design genuinely needs integer structure — a
//! decomposition does, a power map does not.

pub mod canonical_word;
pub mod inverse_power_map;
pub mod power_map;
