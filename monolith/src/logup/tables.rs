//! Fixed `(input chunk, output chunk)` tables for Monolith Bars.
//!
//! What is Monolith's here is only what the AIR cannot know: the values in a
//! row, the bus each table provides, and how a call trace's chunks are counted
//! into multiplicities. Everything else — the transparent key columns, the sole
//! witnessed multiplicity column and its `table_entry` interaction — is
//! [`harness::lookup::FixedTableAir`], shared with every other lookup-backed
//! construction.
//!
//! At byte granularity Goldilocks uses the 256-row byte table. Mersenne-31 uses
//! that table for its three 8-bit chunks and a separate 128-row table for its
//! high 7-bit chunk. At adjacent-pair granularity the corresponding costs are
//! one 65,536-row byte-pair table, or that table plus a 32,768-row
//! byte/seven-bit table. Separate buses keep every table a full power-of-two
//! trace rather than padding a union.

use harness::lookup::{FixedTable, FixedTableAir, LookupGranularity};
use p3_field::Field;
use p3_monolith::{MonolithBarsGoldilocks, MonolithBarsM31};

/// Bus carrying every 8-bit Bars input/output pair.
pub const BYTE_BUS: &str = "monolith-bars-byte";
/// Bus carrying Mersenne-31's high 7-bit Bars input/output pair.
pub const SEVEN_BIT_BUS: &str = "monolith-bars-seven-bit";
/// Bus carrying two adjacent 8-bit Bars input/output coordinates.
pub const BYTE_PAIR_BUS: &str = "monolith-bars-byte-pair";
/// Bus carrying an 8-bit chunk followed by Mersenne-31's high 7-bit chunk.
pub const BYTE_SEVEN_PAIR_BUS: &str = "monolith-bars-byte-seven-pair";

/// One fixed Bars table as a batch AIR instance.
pub type MonolithTableAir = FixedTableAir<FixedTableKind>;

/// Which fixed Bars map this AIR provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedTableKind {
    /// 256 rows, `x -> S_8(x)`.
    Byte,
    /// 128 rows, `x -> S_7(x)`.
    SevenBit,
    /// 65,536 rows, `(x_0, x_1) -> (S_8(x_0), S_8(x_1))`.
    BytePair,
    /// 32,768 rows, `(x_0, x_1) -> (S_8(x_0), S_7(x_1))`.
    ByteSevenPair,
}

impl FixedTableKind {
    /// Number of rows in the full fixed table.
    #[must_use]
    pub const fn height(self) -> usize {
        match self {
            Self::Byte => 1 << 8,
            Self::SevenBit => 1 << 7,
            Self::BytePair => 1 << 16,
            Self::ByteSevenPair => 1 << 15,
        }
    }

    /// The global bus this table provides.
    #[must_use]
    pub const fn bus(self) -> &'static str {
        match self {
            Self::Byte => BYTE_BUS,
            Self::SevenBit => SEVEN_BIT_BUS,
            Self::BytePair => BYTE_PAIR_BUS,
            Self::ByteSevenPair => BYTE_SEVEN_PAIR_BUS,
        }
    }

    /// Message granularity this table answers.
    #[must_use]
    pub const fn granularity(self) -> LookupGranularity {
        match self {
            Self::Byte | Self::SevenBit => LookupGranularity::Byte,
            Self::BytePair | Self::ByteSevenPair => LookupGranularity::AdjacentPair,
        }
    }

    /// Number of independently range-bound coordinates in a table message.
    #[must_use]
    pub const fn tuple_width(self) -> usize {
        self.granularity().tuple_width()
    }

    /// This table as a batch AIR instance.
    #[must_use]
    pub const fn air(self) -> MonolithTableAir {
        FixedTableAir::new(self)
    }

    /// Apply the corresponding upstream `p3-monolith` chunk map.
    #[must_use]
    pub const fn output(self, input: u8) -> u8 {
        match self {
            // A one-byte word makes the upstream SWAR function evaluate exactly
            // one byte lane; no local reimplementation of chi lives here.
            Self::Byte => MonolithBarsGoldilocks::<8>::bar(input as u64) as u8,
            Self::SevenBit => MonolithBarsM31::final_s_box(input),
            Self::BytePair | Self::ByteSevenPair => {
                panic!("paired tables have two outputs; use pair_output")
            }
        }
    }

    /// Apply a paired table to two independently represented coordinates.
    #[must_use]
    pub const fn pair_output(self, input: [u8; 2]) -> [u8; 2] {
        match self {
            Self::BytePair => [
                MonolithBarsGoldilocks::<8>::bar(input[0] as u64) as u8,
                MonolithBarsGoldilocks::<8>::bar(input[1] as u64) as u8,
            ],
            Self::ByteSevenPair => [
                MonolithBarsM31::s_box(input[0]),
                MonolithBarsM31::final_s_box(input[1]),
            ],
            Self::Byte | Self::SevenBit => {
                panic!("single-chunk tables have one output; use output")
            }
        }
    }

    const fn pair_index(self, input: [u8; 2]) -> usize {
        match self {
            Self::BytePair => input[0] as usize | ((input[1] as usize) << 8),
            Self::ByteSevenPair => {
                assert!(input[1] < 128, "high chunk is not seven-bit");
                input[0] as usize | ((input[1] as usize) << 8)
            }
            Self::Byte | Self::SevenBit => {
                panic!("single-chunk tables do not have pair indices")
            }
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
            "use count_pair_inputs for a paired table"
        );
        let mut counts = vec![0u32; self.height()];
        for input in inputs {
            let index = usize::from(input);
            assert!(index < counts.len(), "chunk is outside this fixed table");
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
            "use count_inputs for a single-chunk table"
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
            Self::Byte | Self::SevenBit => {
                row.push(F::from_usize(index));
                row.push(F::from_u8(self.output(index as u8)));
            }
            Self::BytePair | Self::ByteSevenPair => {
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
