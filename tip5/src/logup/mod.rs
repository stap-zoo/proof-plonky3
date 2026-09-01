//! The lookup-backed Tip5 arithmetization — the same hash, twice (POLICY §12).
//!
//! This module is separate from [`crate::air`] and [`crate::vectorized`], which
//! remain the in-AIR-decomposition baseline unchanged. Both prove the same
//! permutation, against the same `../ref` vectors, from the same
//! [`crate::params`]; they differ only in how the split-and-lookup S-box is
//! arithmetized, which is exactly what makes the pair a result rather than two
//! unrelated rows.
//!
//! # What the lookup buys, and what it does not
//!
//! Per split word the baseline commits 272 cells — 64 input bits, sixteen
//! canonical-prefix flags, 64 output bits and eight sixteen-bit division
//! quotients — to prove `L(x) = (x + 1)^3 - 1 (mod 257)` bytewise. Here one
//! `(input byte, output byte)` query per byte does the range proof and the
//! substitution together, leaving eighteen cells. What survives is the part the
//! lookup cannot do: `recompose(bytes) == mont_R * x` is equality in the AIR
//! field, so the second Goldilocks encodings in `[p, 2^64)` still need the
//! two-cell argument of [`harness::gadgets::canonical_word`]. The output side
//! needs no counterpart, because its bytes are a function of the queried input
//! bytes and the reference's definition reduces the recomposed word into the
//! field itself.
//!
//! # One call per row, one table per proof
//!
//! Each call stays in one main row, as in `monolith::logup`; only LogUp's
//! auxiliary accumulator has a next-row constraint, and `main_next_row_columns`
//! is still `vec![]`. At byte granularity a call sends `5 * 4 * 8 = 160`
//! queries against a 256-row table; paired, 80 against a 65,536-row one. The
//! raw `table rows + queries` crossover is `(2^16 - 2^8) / (160 - 80) = 816`
//! calls — an analytic figure, not a prover measurement. There is deliberately
//! no vectorized lookup layout: calls-per-row is worth pricing only if the
//! first measurements make it a question.
//!
//! # The registered frontier is not the full product
//!
//! Tip5 carries a degree axis Monolith does not: the seventh-power words are
//! degree 7 unsplit and 3 with one register, and a fraction column folding `n`
//! same-bus denominators is degree `n + 1`. The AIR's degree is the larger of
//! the two, so most of the `2 * 2 * 4` product is dominated. [`VARIANTS`] keeps
//! the four points per granularity that are not: the split AIR at `frac2`,
//! `frac4` and `frac8` (degrees 3, 5 and 9), and the unsplit AIR at `frac8`,
//! whose base degree 7 already occupies the degree-9 quotient bucket and may as
//! well fold eight denominators into each auxiliary column.

pub mod air;
pub mod batch;
pub mod columns;
pub mod generation;
pub mod tables;

// The two frontier axes are the harness's, not Tip5's; they are re-exported
// here because [`VARIANTS`] is written in them.
pub use harness::lookup::{FractionPacking, LookupGranularity};

use self::air::{Tip5LogupAir, max_constraint_degree};

/// One point in the register × granularity × fraction-packing frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogupVariant {
    /// Stable benchmark label.
    pub name: &'static str,
    /// Width-two byte messages or width-four adjacent-pair messages.
    pub granularity: LookupGranularity,
    /// Denominators folded into each auxiliary fraction column.
    pub packing: FractionPacking,
    /// Seventh-power registers — the `REGISTERS` const of the AIR type. A
    /// label here: the benchmark registry names the type it selects.
    pub registers: usize,
}

impl LogupVariant {
    /// The maximum constraint degree this point lands on.
    #[must_use]
    pub const fn max_constraint_degree(&self) -> usize {
        max_constraint_degree(self.registers, self.packing)
    }
}

/// Every registered Tip5 lookup variant, in increasing degree within each
/// table granularity.
pub const VARIANTS: &[LogupVariant] = &[
    LogupVariant {
        name: "logup-byte-split-frac2",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Two,
        registers: 1,
    },
    LogupVariant {
        name: "logup-byte-split-frac4",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Four,
        registers: 1,
    },
    LogupVariant {
        name: "logup-byte-split-frac8",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Eight,
        registers: 1,
    },
    LogupVariant {
        name: "logup-byte-plain-frac8",
        granularity: LookupGranularity::Byte,
        packing: FractionPacking::Eight,
        registers: 0,
    },
    LogupVariant {
        name: "logup-pair-split-frac2",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Two,
        registers: 1,
    },
    LogupVariant {
        name: "logup-pair-split-frac4",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Four,
        registers: 1,
    },
    LogupVariant {
        name: "logup-pair-split-frac8",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Eight,
        registers: 1,
    },
    LogupVariant {
        name: "logup-pair-plain-frac8",
        granularity: LookupGranularity::AdjacentPair,
        packing: FractionPacking::Eight,
        registers: 0,
    },
];

/// Lookup-backed Tip4′, Goldilocks t = 12, native degree seven.
pub type GoldilocksT12<F> = Tip5LogupAir<F, 12, 8, 0>;
/// Lookup-backed Tip4′, Goldilocks t = 12, one register and degree three.
pub type GoldilocksT12Split<F> = Tip5LogupAir<F, 12, 8, 1>;
/// Lookup-backed Tip4/Tip5, Goldilocks t = 16, native degree seven.
pub type GoldilocksT16<F> = Tip5LogupAir<F, 16, 12, 0>;
/// Lookup-backed Tip4/Tip5, Goldilocks t = 16, one register and degree three.
pub type GoldilocksT16Split<F> = Tip5LogupAir<F, 16, 12, 1>;

#[cfg(test)]
mod tests {
    use super::{FractionPacking, LookupGranularity, VARIANTS};

    /// Eight points, not sixteen: the excluded ones are dominated rather than
    /// unmeasured, and each survivor's degree is the reason it survived.
    #[test]
    fn the_registered_frontier_is_the_undominated_one() {
        assert_eq!(VARIANTS.len(), 8);
        let degrees = VARIANTS
            .iter()
            .map(|variant| (variant.name, variant.max_constraint_degree()))
            .collect::<Vec<_>>();
        assert_eq!(
            degrees,
            vec![
                ("logup-byte-split-frac2", 3),
                ("logup-byte-split-frac4", 5),
                ("logup-byte-split-frac8", 9),
                ("logup-byte-plain-frac8", 9),
                ("logup-pair-split-frac2", 3),
                ("logup-pair-split-frac4", 5),
                ("logup-pair-split-frac8", 9),
                ("logup-pair-plain-frac8", 9),
            ]
        );

        // Each granularity carries the same four points.
        for granularity in [LookupGranularity::Byte, LookupGranularity::AdjacentPair] {
            assert_eq!(
                VARIANTS
                    .iter()
                    .filter(|variant| variant.granularity == granularity)
                    .count(),
                4
            );
        }

        // `frac1` is absent because the split AIR's own degree 3 already pays
        // for `frac2`, which halves the auxiliary columns at no extra degree.
        assert!(
            VARIANTS
                .iter()
                .all(|variant| variant.packing != FractionPacking::One)
        );
        // The unsplit AIR appears only at maximal packing: below it, degree 7
        // is paid for and nothing is bought.
        assert!(
            VARIANTS
                .iter()
                .all(|variant| variant.registers == 1 || variant.packing == FractionPacking::Eight)
        );
    }
}
