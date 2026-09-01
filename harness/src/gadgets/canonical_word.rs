//! Little-endian chunk decomposition of a field word, and the witness that
//! pins the integer representative those chunks encode.
//!
//! A lookup-backed S-box splits a state word into small chunks, queries each
//! chunk on a fixed table, and recomposes the result.  The table query is what
//! range-binds a chunk; this gadget owns the two things that are *not* the
//! query: the weighted sum tying the chunks to the field word, and the small
//! field-specific argument ruling out the non-canonical integer encodings that
//! sum tolerates.
//!
//! **Why this arithmetization.**  Recomposition is equality in the AIR field,
//! so `sum(chunk_i * 2^s_i) == word` accepts every 64-bit integer congruent to
//! `word`, not only its canonical representative.  Since the chunks are the
//! lookup keys, an accepted second encoding would query a different table row
//! and return a different "S-box output".  Rather than a general comparison
//! against `p`, each field gets the cheapest argument its modulus admits, using
//! the fact that every chunk is already range-bound:
//!
//! * **Mersenne-31** (`p = 2^31 - 1`, chunks `8,8,8,7`): the all-maximal tuple
//!   is the *only* non-canonical one.  One witnessed inverse of the
//!   non-negative gap `sum(max_i - chunk_i)` rejects exactly it.
//! * **Goldilocks** (`p = 2^64 - 2^32 + 1`, eight bytes): a tuple is
//!   non-canonical exactly when all four high bytes are `0xff` and the low word
//!   is non-zero — `p - 1 = 0xffff_ffff_0000_0000` is the one accepted tuple
//!   with maximal high bytes.  Two cells, a boolean `high_is_max` and an
//!   inverse of the high gap, decide that branch and force the low word to zero
//!   on it.
//!
//! **What it costs.**  Mersenne-31: one cell and one degree-2 constraint.
//! Goldilocks: two cells and five constraints, all degree 2. Recomposition adds
//! no cell and no degree; it is affine in the chunks.
//!
//! **The trap.**  Both arguments assume every chunk is *already* bound to
//! `0..=chunk_max`.  That bound comes from the caller's lookup, never from
//! here: with unconstrained chunks the gap is an arbitrary field element and
//! its inverse always exists.  Wiring [`eval_canonicity`] without also
//! querying every chunk on a fixed table proves nothing.
//!
//! First user: `monolith::logup`, whose Bars input and output words both need
//! it.  Second user: `tip5::logup`'s split-and-lookup input word.

use p3_air::AirBuilder;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Which prime's integer structure a chunk decomposition follows.
///
/// The chunk widths are the ones the lookup tables are built for, not a free
/// choice: they are what makes a chunk one table key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordKind {
    /// `p = 2^64 - 2^32 + 1`: eight byte chunks, two canonicity cells.
    Goldilocks,
    /// `p = 2^31 - 1`: three byte chunks and a seven-bit chunk, one cell.
    Mersenne31,
}

impl WordKind {
    /// The prime this decomposition is canonical against.
    #[must_use]
    pub const fn modulus(self) -> u64 {
        match self {
            Self::Goldilocks => 0xffff_ffff_0000_0001,
            Self::Mersenne31 => 0x7fff_ffff,
        }
    }

    /// Chunks in one complete word.
    #[must_use]
    pub const fn num_chunks(self) -> usize {
        match self {
            Self::Goldilocks => 8,
            Self::Mersenne31 => 4,
        }
    }

    /// Canonicity cells [`eval_canonicity`] reads for one word.
    #[must_use]
    pub const fn canonical_cells(self) -> usize {
        match self {
            Self::Goldilocks => 2,
            Self::Mersenne31 => 1,
        }
    }

    /// Bit width of one chunk.
    #[must_use]
    pub const fn chunk_bits(self, index: usize) -> usize {
        assert!(index < self.num_chunks(), "chunk index outside the word");
        match self {
            Self::Goldilocks => 8,
            Self::Mersenne31 if index == self.num_chunks() - 1 => 7,
            Self::Mersenne31 => 8,
        }
    }

    /// Largest value the lookup permits in one chunk.
    #[must_use]
    pub const fn chunk_max(self, index: usize) -> u8 {
        ((1u16 << self.chunk_bits(index)) - 1) as u8
    }

    /// Weight of one chunk in the recomposed word, as a bit position.
    #[must_use]
    pub const fn chunk_shift(self, index: usize) -> usize {
        assert!(index < self.num_chunks(), "chunk index outside the word");
        let mut shift = 0;
        let mut chunk = 0;
        while chunk < index {
            shift += self.chunk_bits(chunk);
            chunk += 1;
        }
        shift
    }
}

/// The weighted sum of a chunk prefix, as an affine expression.
///
/// `chunks` may be shorter than a whole word — [`eval_canonicity`] recomposes
/// Goldilocks' low half that way — and chunk `i` always carries the weight it
/// has in the full word.
pub fn recompose<AB: AirBuilder>(chunks: &[AB::Var], kind: WordKind) -> AB::Expr {
    assert!(chunks.len() <= kind.num_chunks(), "too many chunks");
    let mut result = AB::Expr::ZERO;
    for (index, &chunk) in chunks.iter().enumerate() {
        result += AB::Expr::from(chunk) * AB::F::from_u64(1u64 << kind.chunk_shift(index));
    }
    result
}

/// The same weighted sum over integers: the witness side's oracle.
#[must_use]
pub fn recompose_integer(chunks: &[u8], kind: WordKind) -> u64 {
    assert!(chunks.len() <= kind.num_chunks(), "too many chunks");
    chunks
        .iter()
        .enumerate()
        .fold(0u64, |word, (index, &chunk)| {
            assert!(chunk <= kind.chunk_max(index), "chunk exceeds its range");
            word | (u64::from(chunk) << kind.chunk_shift(index))
        })
}

/// Enforce that already range-bound chunks encode a canonical representative.
///
/// This says nothing about *which* word they encode: pair it with a
/// `recompose(chunks) == word` assertion, and with the lookup that range-binds
/// every chunk.
pub fn eval_canonicity<AB: AirBuilder>(
    chunks: &[AB::Var],
    cells: &[AB::Var],
    kind: WordKind,
    builder: &mut AB,
) where
    AB::F: PrimeField64,
{
    assert_eq!(chunks.len(), kind.num_chunks(), "one canonicity per word");
    assert_eq!(cells.len(), kind.canonical_cells(), "wrong witness width");
    debug_assert_eq!(AB::F::ORDER_U64, kind.modulus());
    match kind {
        WordKind::Mersenne31 => {
            // p is the sole non-canonical tuple.  Since every chunk is
            // range-bound, this non-negative gap is zero exactly at p.
            let gap = chunks
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |sum, (index, &chunk)| {
                    sum + AB::F::from_u8(kind.chunk_max(index)) - chunk
                });
            // The inverse is prover-chosen and uniquely pinned for every
            // accepted tuple; the all-maximal tuple has no satisfying inverse.
            builder.assert_one(gap * cells[0]);
        }
        WordKind::Goldilocks => {
            // p = FFFFFFFF_00000001.  A 64-bit tuple is non-canonical exactly
            // when all four high bytes are FF and the low word is non-zero.
            // Range-bounded bytes make this small sum zero iff all are FF.
            let high_gap = chunks[4..].iter().fold(AB::Expr::ZERO, |sum, &chunk| {
                sum + AB::F::from_u8(0xff) - chunk
            });
            let low_word = recompose::<AB>(&chunks[..4], kind);
            let is_high_max: AB::Expr = cells[0].into();
            let inverse: AB::Expr = cells[1].into();
            builder.assert_bool(cells[0]);
            builder.assert_eq(
                high_gap.clone() * inverse.clone(),
                AB::Expr::ONE - is_high_max.clone(),
            );
            builder.assert_zero(high_gap * is_high_max.clone());
            // Pin the inverse to zero on its otherwise-degenerate branch.
            builder.assert_zero(inverse * is_high_max.clone());
            builder.assert_zero(is_high_max * low_word);
        }
    }
}

/// Split a canonical integer into the chunks [`recompose`] reads.
#[must_use]
pub fn decompose<const NUM_CHUNKS: usize>(integer: u64, kind: WordKind) -> [u8; NUM_CHUNKS] {
    assert_eq!(NUM_CHUNKS, kind.num_chunks(), "wrong chunk count");
    let chunks: [u8; NUM_CHUNKS] = core::array::from_fn(|index| {
        ((integer >> kind.chunk_shift(index)) & u64::from(kind.chunk_max(index))) as u8
    });
    assert_eq!(
        recompose_integer(&chunks, kind),
        integer,
        "the word does not fit its chunk decomposition"
    );
    chunks
}

/// The canonicity cells [`eval_canonicity`] pins, for one chunk tuple.
///
/// Panics on a non-canonical tuple rather than emitting an unsatisfiable
/// witness: the constraints reject it either way, and the generator is where
/// the reason is still legible (POLICY §9 — this is a witness producer, never
/// evidence).
#[must_use]
pub fn canonicity_witness<F: PrimeField64, const CANONICAL_CELLS: usize>(
    chunks: &[u8],
    kind: WordKind,
) -> [F; CANONICAL_CELLS] {
    assert_eq!(chunks.len(), kind.num_chunks(), "one witness per word");
    assert_eq!(CANONICAL_CELLS, kind.canonical_cells(), "wrong cell count");
    let mut cells = [F::ZERO; CANONICAL_CELLS];
    match kind {
        WordKind::Mersenne31 => {
            // Pinned by eval_canonicity's `gap * cells[0] == 1`.
            let gap = chunks
                .iter()
                .enumerate()
                .map(|(index, &chunk)| u64::from(kind.chunk_max(index) - chunk))
                .sum::<u64>();
            assert_ne!(gap, 0, "Mersenne-31 chunk tuple must be canonical");
            cells[0] = F::from_u64(gap).inverse();
        }
        WordKind::Goldilocks => {
            let high_gap = chunks[4..]
                .iter()
                .map(|&chunk| u64::from(0xff - chunk))
                .sum::<u64>();
            if high_gap == 0 {
                let low = recompose_integer(&chunks[..4], kind);
                assert_eq!(low, 0, "Goldilocks chunk tuple must be canonical");
                // Pinned by `high_gap * is_high_max == 0` and the boolean.
                cells[0] = F::ONE;
            } else {
                // Pinned by `high_gap * inverse == 1 - is_high_max`.
                cells[1] = F::from_u64(high_gap).inverse();
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess, check_constraints};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::distr::{Distribution, StandardUniform};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::{
        WordKind, canonicity_witness, decompose, eval_canonicity, recompose, recompose_integer,
    };

    /// One word, its chunks and its canonicity cells, in that order.
    struct WordAir(WordKind);

    impl<F: Field> BaseAir<F> for WordAir {
        fn width(&self) -> usize {
            1 + self.0.num_chunks() + self.0.canonical_cells()
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            vec![]
        }
    }

    impl<AB: AirBuilder> Air<AB> for WordAir
    where
        AB::F: PrimeField64,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();
            let chunks = &row[1..=self.0.num_chunks()];
            let cells = &row[1 + self.0.num_chunks()..];
            builder.assert_eq(recompose::<AB>(chunks, self.0), row[0]);
            eval_canonicity(chunks, cells, self.0, builder);
        }
    }

    /// Rows built the way a generator would, from canonical field values.
    fn trace<F: PrimeField64, const NUM_CHUNKS: usize, const CANONICAL_CELLS: usize>(
        values: &[F],
        kind: WordKind,
    ) -> RowMajorMatrix<F> {
        let mut rows = Vec::new();
        for &value in values {
            let chunks = decompose::<NUM_CHUNKS>(value.as_canonical_u64(), kind);
            assert_eq!(recompose_integer(&chunks, kind), value.as_canonical_u64());
            rows.push(value);
            rows.extend(chunks.map(F::from_u8));
            rows.extend(canonicity_witness::<F, CANONICAL_CELLS>(&chunks, kind));
        }
        RowMajorMatrix::new(rows, 1 + NUM_CHUNKS + CANONICAL_CELLS)
    }

    fn edge_values<F: PrimeField64>() -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6361_6e6f_6e69_6361);
        let mut values = vec![
            F::ZERO,
            F::ONE,
            F::from_u64(F::ORDER_U64 - 1),
            F::from_u64(F::ORDER_U64 - 2),
        ];
        values.extend((0..32).map(|_| rng.sample(StandardUniform)));
        values
    }

    #[test]
    fn every_canonical_word_decomposes_and_is_accepted() {
        let goldilocks = edge_values::<Goldilocks>();
        check_constraints(
            &WordAir(WordKind::Goldilocks),
            &trace::<Goldilocks, 8, 2>(&goldilocks, WordKind::Goldilocks),
            &[],
        );

        let mersenne = edge_values::<Mersenne31>();
        check_constraints(
            &WordAir(WordKind::Mersenne31),
            &trace::<Mersenne31, 4, 1>(&mersenne, WordKind::Mersenne31),
            &[],
        );
    }

    #[test]
    fn the_canonicity_branches_are_the_ones_the_moduli_have() {
        // Goldilocks: only p - 1 = FFFFFFFF_00000000 has maximal high bytes.
        assert_eq!(
            canonicity_witness::<Goldilocks, 2>(&[0; 8], WordKind::Goldilocks),
            [Goldilocks::ZERO, Goldilocks::from_u32(4 * 255).inverse()]
        );
        assert_eq!(
            canonicity_witness::<Goldilocks, 2>(
                &decompose::<8>(Goldilocks::ORDER_U64 - 1, WordKind::Goldilocks),
                WordKind::Goldilocks
            ),
            [Goldilocks::ONE, Goldilocks::ZERO]
        );
        // Mersenne-31: every canonical tuple has a non-zero gap.
        for integer in [0, 1, Mersenne31::ORDER_U64 - 1] {
            let chunks = decompose::<4>(integer, WordKind::Mersenne31);
            assert!(
                !canonicity_witness::<Mersenne31, 1>(&chunks, WordKind::Mersenne31)[0].is_zero()
            );
        }
    }

    #[test]
    fn no_second_encoding_has_a_witness() {
        // p and p + 1 are the field elements 0 and 1 written non-canonically.
        for integer in [Goldilocks::ORDER_U64, Goldilocks::ORDER_U64 + 1] {
            let chunks = integer.to_le_bytes();
            assert_eq!(recompose_integer(&chunks, WordKind::Goldilocks), integer);
            assert!(
                std::panic::catch_unwind(|| canonicity_witness::<Goldilocks, 2>(
                    &chunks,
                    WordKind::Goldilocks
                ))
                .is_err()
            );
        }
        assert!(
            std::panic::catch_unwind(|| canonicity_witness::<Mersenne31, 1>(
                &[255, 255, 255, 127],
                WordKind::Mersenne31
            ))
            .is_err()
        );
    }

    /// The constraints, not merely the generator, are what reject a second
    /// encoding — and a forged canonicity cell does not rescue it.
    #[test]
    fn constraints_reject_forged_cells_and_second_encodings() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut forged = trace::<Goldilocks, 8, 2>(&[Goldilocks::ONE], WordKind::Goldilocks);
        // The integer p + 1, with the only cells that could satisfy the
        // maximal-high-byte branch.
        let chunks = (Goldilocks::ORDER_U64 + 1).to_le_bytes();
        for (cell, chunk) in forged.values[1..9].iter_mut().zip(chunks) {
            *cell = Goldilocks::from_u8(chunk);
        }
        forged.values[9] = Goldilocks::ONE;
        forged.values[10] = Goldilocks::ZERO;
        assert!(
            std::panic::catch_unwind(|| check_constraints(
                &WordAir(WordKind::Goldilocks),
                &forged,
                &[]
            ))
            .is_err()
        );

        // Mersenne-31's all-maximal tuple is the field element 0; no inverse
        // of its zero gap exists, so every cell value fails.
        let mut forged = trace::<Mersenne31, 4, 1>(&[Mersenne31::ZERO], WordKind::Mersenne31);
        for (cell, chunk) in forged.values[1..5].iter_mut().zip([255, 255, 255, 127]) {
            *cell = Mersenne31::from_u8(chunk);
        }
        for cell in [Mersenne31::ZERO, Mersenne31::ONE, Mersenne31::from_u32(7)] {
            forged.values[5] = cell;
            let corrupted = forged.clone();
            assert!(
                std::panic::catch_unwind(move || check_constraints(
                    &WordAir(WordKind::Mersenne31),
                    &corrupted,
                    &[]
                ))
                .is_err()
            );
        }

        // A correct tuple with one corrupted canonicity cell is also rejected.
        let mut corrupted = trace::<Goldilocks, 8, 2>(&[Goldilocks::ONE], WordKind::Goldilocks);
        corrupted.values[10] += Goldilocks::ONE;
        assert!(
            std::panic::catch_unwind(|| check_constraints(
                &WordAir(WordKind::Goldilocks),
                &corrupted,
                &[]
            ))
            .is_err()
        );

        std::panic::set_hook(hook);
    }
}
