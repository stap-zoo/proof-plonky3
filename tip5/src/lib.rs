//! Tip5 (and Tip4, Tip4') — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it (POLICY §1).
//!
//! # Read this first (POLICY §13)
//!
//! [Triton VM's Hash Table](https://triton-vm.org/spec/hash-table.html) is a
//! complete Tip5 arithmetization: 8 rows per permutation, split-and-lookup via
//! a Cascade table (16 to 8 bit) and a Lookup table, joined by lookup
//! arguments. Native implementations: `twenty-first` `tip5.rs`, winterfell
//! `hash/tip/*`.
//!
//! # But not that arithmetization first (POLICY §12)
//!
//! Triton's joins are lookup arguments, and `uni-stark` cannot host them: it
//! has a single commit round and leaves `AirLayout::permutation_width` at 0, so
//! LogUp's auxiliary columns have nowhere to live. The default here is
//! therefore **in-AIR decomposition, the way `p3-monolith-air` does it** —
//! committed bits, canonicity flags, no lookup.
//!
//! Tip5 measured both ways is a genuine result, and [`logup`] is that second
//! arithmetization: the harness's `prove_batch` path and lookup-aware
//! `AirLayout` are in place, so the call AIR trades each split word's 272
//! committed cells for eighteen plus one `(input byte, output byte)` query per
//! byte. The baseline above is untouched, which is what makes the pair two
//! readings of one hash rather than a replacement.
//!
//! # Consequences of the split-and-lookup S-box
//!
//! * The lookup-free baseline fits one (very wide) row, so every call remains
//!   independent and the AIR honestly declares no next-row openings. Its width
//!   is pinned in `tests/numbers.rs`; it is the principal cost being measured.
//! * The design is undefined over every 31-bit field: its split S-box is eight
//!   bytes of a canonical 64-bit field element, and the reference rejects any
//!   prime whose bit length is not 64.
//! * Every limb and every canonicity flag is prover-chosen. Name the constraint
//!   that pins each, where it is witnessed, and test the non-canonical
//!   representations the field arithmetic tolerates but a decomposition must
//!   not (POLICY §9).
//!
//! Where the reference defines an extra width (t = 16), implement it if free —
//! but the grid is what the tables compare.
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref` or it is not validated. The vectors come from
//! `../ref/export_small_prime_kat.py`; giving this construction an entry in that
//! script's construction table is POLICY §2's step 2, which comes before the
//! native permutation and long before any AIR.
//!
//! The construction-specific validation lives in `tests/{reference,air,numbers}.rs`;
//! POLICY §10 defines the shared layers. The release proof round trip is
//! centralized in `bench/tests/prove_verify.rs` so it uses the same 100-bit
//! configuration as every other construction.

pub mod air;
pub mod columns;
pub mod generation;
pub mod instances;
pub mod logup;
pub mod native;
pub mod params;
pub mod vectorized;
