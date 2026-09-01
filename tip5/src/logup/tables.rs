//! Fixed `(input byte, output byte)` tables for Tip5's split-and-lookup S-box.
//!
//! What is Tip5's here is only what the AIR cannot know: the values in a row,
//! the bus each table provides, and how a call trace's bytes are counted into
//! multiplicities. Everything else — the transparent key columns, the sole
//! witnessed multiplicity column and its `table_entry` interaction — is
//! [`harness::lookup::FixedTableAir`], shared with `monolith::logup`.
//!
//! The tabulated function is [`crate::params::lookup`] and nothing else: the
//! reference's `L(x) = (x + 1)^3 - 1 (mod 257)` restricted to bytes. The
//! baseline AIR proves that relation with a sixteen-bit quotient per byte; here
//! it is 256 preprocessed rows the verifier is bound to, evaluated once.
//!
//! At byte granularity that is the 256-row table. At adjacent-pair granularity
//! it is one 65,536-row table whose four coordinates stay separate, so each is
//! independently range-bound. The buses are Tip5's own: Monolith's byte table
//! tabulates a different function on a bus of the same width, and sharing a bus
//! name would silently balance two different maps against each other.

use harness::lookup::{FixedTable, FixedTableAir, LookupGranularity};
use p3_field::Field;

use crate::params::lookup;

/// Bus carrying one `(input byte, output byte)` pair of the split S-box.
pub const BYTE_BUS: &str = "tip5-split-byte";
/// Bus carrying two adjacent input bytes and their two images.
pub const BYTE_PAIR_BUS: &str = "tip5-split-byte-pair";

/// One fixed split-and-lookup table as a batch AIR instance.
pub type Tip5TableAir = FixedTableAir<FixedTableKind>;

/// Which fixed map this AIR provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedTableKind {
    /// 256 rows, `x -> L(x)`.
    Byte,
    /// 65,536 rows, `(x_0, x_1) -> (L(x_0), L(x_1))`.
    BytePair,
}

impl FixedTableKind {
    /// Number of rows in the full fixed table.
    #[must_use]
    pub const fn height(self) -> usize {
        match self {
            Self::Byte => 1 << 8,
            Self::BytePair => 1 << 16,
        }
    }

    /// The global bus this table provides.
    #[must_use]
    pub const fn bus(self) -> &'static str {
        match self {
            Self::Byte => BYTE_BUS,
            Self::BytePair => BYTE_PAIR_BUS,
        }
    }

    /// Message granularity this table answers.
    #[must_use]
    pub const fn granularity(self) -> LookupGranularity {
        match self {
            Self::Byte => LookupGranularity::Byte,
            Self::BytePair => LookupGranularity::AdjacentPair,
        }
    }

    /// Number of independently range-bound coordinates in a table message.
    #[must_use]
    pub const fn tuple_width(self) -> usize {
        self.granularity().tuple_width()
    }

    /// This table as a batch AIR instance.
    #[must_use]
    pub const fn air(self) -> Tip5TableAir {
        FixedTableAir::new(self)
    }

    /// Apply the reference's byte permutation to one coordinate.
    #[must_use]
    pub const fn output(self, input: u8) -> u8 {
        match self {
            Self::Byte => lookup(input),
            Self::BytePair => panic!("the paired table has two outputs; use pair_output"),
        }
    }

    /// Apply the reference's byte permutation to two separate coordinates.
    #[must_use]
    pub const fn pair_output(self, input: [u8; 2]) -> [u8; 2] {
        match self {
            Self::BytePair => [lookup(input[0]), lookup(input[1])],
            Self::Byte => panic!("the byte table has one output; use output"),
        }
    }

    const fn pair_index(self, input: [u8; 2]) -> usize {
        match self {
            Self::BytePair => input[0] as usize | ((input[1] as usize) << 8),
            Self::Byte => panic!("the byte table does not have pair indices"),
        }
    }

    /// Count query inputs into table-row multiplicities.
    ///
    /// Outputs are intentionally not accepted here: a dishonest output must
    /// leave the `(input, output)` bus unbalanced rather than steering the
    /// table witness to a different row.
    #[must_use]
    pub fn count_inputs(self, inputs: impl IntoIterator<Item = u8>) -> Vec<u32> {
        assert_eq!(
            self.tuple_width(),
            2,
            "use count_pair_inputs for the paired table"
        );
        let mut counts = vec![0u32; self.height()];
        for input in inputs {
            let index = usize::from(input);
            counts[index] = counts[index]
                .checked_add(1)
                .expect("table multiplicity does not fit u32");
        }
        counts
    }

    /// Count paired query inputs into paired-table multiplicities.
    #[must_use]
    pub fn count_pair_inputs(self, inputs: impl IntoIterator<Item = [u8; 2]>) -> Vec<u32> {
        assert_eq!(
            self.tuple_width(),
            4,
            "use count_inputs for the single-byte table"
        );
        let mut counts = vec![0u32; self.height()];
        for input in inputs {
            let index = self.pair_index(input);
            counts[index] = counts[index]
                .checked_add(1)
                .expect("table multiplicity does not fit u32");
        }
        counts
    }
}

impl FixedTable for FixedTableKind {
    fn bus(&self) -> &'static str {
        Self::bus(*self)
    }

    fn height(&self) -> usize {
        Self::height(*self)
    }

    fn tuple_width(&self) -> usize {
        Self::tuple_width(*self)
    }

    fn append_row<F: Field>(&self, index: usize, row: &mut Vec<F>) {
        match self {
            Self::Byte => {
                row.push(F::from_usize(index));
                row.push(F::from_u8(self.output(index as u8)));
            }
            Self::BytePair => {
                let input = [index as u8, (index >> 8) as u8];
                let output = self.pair_output(input);
                // Four separate coordinates are deliberate: packing either
                // pair into one field element would not independently
                // range-bind them and would admit fingerprint collisions.
                row.extend(input.into_iter().map(F::from_u8));
                row.extend(output.into_iter().map(F::from_u8));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FixedTableKind, LookupGranularity};

    /// The tables tabulate `params::lookup` and nothing else, and the paired
    /// table's row index is the little-endian pair its call side sends.
    #[test]
    fn every_row_is_the_reference_byte_permutation() {
        for byte in 0..=u8::MAX {
            assert_eq!(
                FixedTableKind::Byte.output(byte),
                crate::params::lookup(byte)
            );
        }
        for index in 0..FixedTableKind::BytePair.height() {
            let input = [index as u8, (index >> 8) as u8];
            assert_eq!(
                FixedTableKind::BytePair.pair_output(input),
                [
                    crate::params::lookup(input[0]),
                    crate::params::lookup(input[1])
                ]
            );
            assert_eq!(FixedTableKind::BytePair.pair_index(input), index);
        }
    }

    #[test]
    fn the_two_granularities_have_the_shapes_they_declare() {
        assert_eq!(FixedTableKind::Byte.height(), 256);
        assert_eq!(FixedTableKind::Byte.tuple_width(), 2);
        assert_eq!(FixedTableKind::Byte.granularity(), LookupGranularity::Byte);
        assert_eq!(FixedTableKind::BytePair.height(), 1 << 16);
        assert_eq!(FixedTableKind::BytePair.tuple_width(), 4);
        assert_eq!(
            FixedTableKind::BytePair.granularity(),
            LookupGranularity::AdjacentPair
        );
    }
}
