//! pSquareHash — **written**: native permutation and AIR, both ours.
//!
//! Plonky3 has nothing for it, and POLICY §1 records no prior art. There is,
//! however, an **earlier Plonky3 arithmetization** of this design outside this
//! repository — `hash_pfm/hash_pfm_zk/Rust/p3-psquarehash{,-air}` — and what is
//! here is a port of it, not a fresh design. What the port changed, and why, is
//! the last section of this page.
//!
//! # The design, in one paragraph
//!
//! A Feistel network over `F^t` with `t ≡ 0 (mod 4)`. Each round feeds the lower
//! half through `t/4` independent two-element Feistels — two squarings each, no
//! power map, no inverse — adds the results into the upper half, and swaps the
//! halves; the swap plus a handful of additions *is* the linear layer `M`. An
//! affine `M_IO` brackets the round loop at both ends. Nothing decomposes, nothing
//! inverts, nothing needs a lookup: the whole design is squarings and additions,
//! which is what makes its arithmetization a question about **cells** rather than
//! about degree.
//!
//! # Validation (POLICY §4)
//!
//! Byte-exact against `../ref` or it is not validated. The vectors come from
//! `ref/export_small_prime_kat.py`, a new small-prime-only script written for
//! this construction (`export_kat.py` is hardcoded to BN254 and BLS12-381 and has
//! no entry for it). Two edits to `../ref`, and they are the whole of them:
//!
//! * `psquarehash/instances.py` gained the four BabyBear / KoalaBear instances,
//!   `rcons` omitted so `_init_cons` derives them (POLICY §3);
//! * `export_small_prime_kat.py` is new, and emits both `vectors/vectors.json`
//!   and — with `--params` — `vectors/params.json`.
//!
//! # The grid: six points, two absences
//!
//! Mersenne-31 at t = 16 and t = 24 are the reference's, matched exactly. BabyBear
//! and KoalaBear at both widths are generated, with `R = 52` carried across from
//! Mersenne-31 at equal `t` and therefore **provisional** (POLICY §3).
//!
//! **Goldilocks at t = 8 and t = 12 do not exist.**
//! `pSquareHashParams._init_rounds` raises `NotImplementedError` — the round-number
//! criterion of the pSquare paper is not implemented in the reference — and the
//! reference pins no Goldilocks instance, so `R` has no value there and POLICY §3's
//! copy-across rule cannot supply one, because it copies *from* Goldilocks. Both
//! points are `Absence::StubbedDerivation` in [`instances::INSTANCES`]. This is the
//! one grid gap and it is the reference's to close.
//!
//! # Arithmetized twice, over three axes
//!
//! POLICY §6: two arithmetizations of one permutation are two sets of the four
//! files, one module each, sharing `params.rs`, `native.rs` and the vectors.
//! [`full_commit`] is the five variants on POLICY §11's two original axes — how
//! much of a round is flattened, and how many rounds separate two state
//! commitments — and every commitment there covers a Feistel's whole pair.
//! [`half_commit`] is the third axis, and covers half of one. Every
//! [`instances`] alias, in one table:
//!
//! | [`instances`] alias | flattening | commitment | committed per round | max degree |
//! |---|---|---|---|---|
//! | `Flattened*`    | both squarings | none needed | `t/2`  | 2 |
//! | `StateOnly*`    | none           | every round | `t/2`  | 4 |
//! | `OneRegister*`  | `y1²`          | every round | `3t/4` | 2 |
//! | `SpacedSplit*`  | `y1²`, on the committing round | every second round | `3t/8` | 8 |
//! | `Spaced*`       | none           | every second round | `t/4`  | 16 |
//! | `HalfCommit*`   | the even output, by differencing | every round, half of each pair | `t/4` | 4 |
//!
//! `Flattened` is the widest of the low-degree layouts and the cheapest to prove,
//! because a degree buys a code rate: the two spaced layouts are narrower and pay
//! `log_blowup` 3 and 4 for it against `Flattened`'s 1. That is the comparison
//! this design exists to make — see [`full_commit::air`] for the degree
//! accounting and [`full_commit::columns`] for the cell counting, and the
//! measured rows for which side wins. `HalfCommit` is the first row to go *below*
//! `t/2` without paying a degree the primary reading punishes: at degree 4 it
//! still fits POLICY §7's common blowup and its own minimum is `log_blowup` 2,
//! so `width × 2^log_blowup` ties `Flattened` per round instead of losing to it
//! by the factor of 4 the spaced layouts concede.
//!
//! [`half_commit`] is that third axis: **how much of one Feistel** a commitment
//! covers. It commits one output per Feistel and recovers the other by
//! differencing — `t/4` cells per round at max degree 4, where `full_commit`
//! reaches `t/4` only at degree 16. Its own docs carry that argument, including
//! why the committed cell must be the odd one.
//!
//! # Validation coverage
//!
//! POLICY §10 defines the common layers. `tests/reference.rs` pins all six
//! instances and their constants. `tests/air.rs` is construction-specific in
//! two important ways: it compares the fused trace round with the reference
//! while the native path applies `M` separately, and it pairs an every-cell
//! corruption sweep with the constraint-count pin in `tests/numbers.rs`. The
//! pair matters: removing the `y3²` assertion still fails the sweep indirectly
//! through `outputs`, but changes the pinned constraint count. The release proof
//! round trip is centralized in `bench/tests/prove_verify.rs` under the shared
//! 100-bit configuration.
//!
//! # What the port changed
//!
//! Three corrections, one reduction, and the modern API. In order of how much
//! they matter:
//!
//! 1. **`M_IO` is now proved.** The earlier AIR commented out the input matrix and
//!    never applied the output one, on both the constraint and the generation
//!    side. It was self-consistent, so its tests passed — but the statement it
//!    proved was the round loop *without* the brackets, which is a different
//!    permutation from the reference's. The `outputs` columns are new for the same
//!    reason: the trailing `M_IO` is a constraint now, not something a test
//!    recomputes.
//! 2. **The S-box array was twice the size it needed.** `sboxes: [SBox<T, R>; WIDTH]`
//!    with only `WIDTH/2` entries ever touched — a workaround for `[T; WIDTH/2]`
//!    not being expressible, at a cost of `t/4` dead cells per round. Here the
//!    length is `PAIRS = t/4` with the invariant checked in
//!    [`full_commit::columns::assert_layout`].
//! 3. **An unconstrained `export` column** was written `ONE` by the generator and
//!    never read by the AIR. Exactly POLICY §9's failure mode, if a harmless
//!    instance of it. Gone.
//! 4. **The reduction: two registers per Feistel instead of one plus the state.**
//!    A Feistel performs two multiplications and its second output is *affine* in
//!    them, so committing both leaves the entire round map affine in committed
//!    cells — the state stops needing columns at all. Same max degree 2, `t/2`
//!    cells per round instead of `3t/4`. See [`full_commit::air`]'s degree
//!    accounting.
//!
//! Net, at t = 16, per call: **849 → 448 cells**, of which 656 is the corrected
//! baseline (`OneRegister16`, same design, same degree, `M_IO` and `outputs`
//! included) and 448 the reduction. The 849 is not a like-for-like number, since
//! it proves less.
//!
//! Also: one row is now one *reference* round rather than two. The earlier AIR
//! paired rounds so that its `[T; WIDTH]` post-state array cost `t/2` per round,
//! which is the right count reached the obscure way; a single round writing `t/2`
//! `post` cells is the same count said directly, and `ROUNDS` is now `R`.

pub mod full_commit;
pub mod half_commit;
pub mod instances;
// `M`, over any ring: shared by both arithmetizations and by both sides of each.
mod linear;
pub mod native;
pub mod params;
