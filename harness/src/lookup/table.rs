//! The fixed-function-table AIR: transparent keys, one witnessed multiplicity.
//!
//! Every lookup-backed arithmetization here needs the same table AIR: the
//! `(input…, output…)` coordinates are preprocessed constants the verifier is
//! bound to, the only committed column is how many call-side queries used that
//! row, and one global LogUp `table_entry` interaction pins that count to the
//! call AIRs sharing the bus. None of that knows what function is tabulated.
//!
//! What the *construction* owns is exactly what does: the values in a row, the
//! bus name, the table's height, and counting its own queries into
//! multiplicities. A construction provides them through [`FixedTable`]; this
//! module provides everything else, once.
//!
//! Tables are full and unpadded: heights are powers of two by construction
//! (`2^8`, `2^7`, `2^16`, `2^15`), so no selector column and no padding row
//! exists, exactly as POLICY §6 requires of a call trace.

use core::fmt::Debug;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::Field;
use p3_lookup::{InteractionBuilder, LookupBus};
use p3_matrix::dense::RowMajorMatrix;

/// Construction-owned description of one fixed function table.
///
/// The implementor is a small value type — an enum of the tables a
/// construction defines — carried by [`FixedTableAir`] and cloned with it.
pub trait FixedTable: Clone + Debug {
    /// The global bus this table provides.
    fn bus(&self) -> &'static str;

    /// Rows in the full table. Must be a power of two: there is no padding.
    fn height(&self) -> usize;

    /// Coordinates in one message: the key coordinates then the outputs.
    fn tuple_width(&self) -> usize;

    /// Append row `index`: exactly [`Self::tuple_width`] values, keys first.
    ///
    /// This is the only place a construction's tabulated function is
    /// evaluated, and it is evaluated into preprocessed — verifier-bound —
    /// columns, never into a witness.
    fn append_row<F: Field>(&self, index: usize, row: &mut Vec<F>);
}

/// One fixed table as a batch AIR instance.
#[derive(Debug, Clone, Copy)]
pub struct FixedTableAir<T> {
    table: T,
}

impl<T: FixedTable> FixedTableAir<T> {
    /// Wrap a construction's table description.
    #[must_use]
    pub const fn new(table: T) -> Self {
        Self { table }
    }

    /// The description this AIR was built from.
    #[must_use]
    pub const fn table(&self) -> &T {
        &self.table
    }

    /// Materialize the transparent `(input…, output…)` columns.
    #[must_use]
    pub fn fixed_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let height = self.table.height();
        let tuple_width = self.table.tuple_width();
        assert!(height.is_power_of_two(), "a fixed table is never padded");
        let mut values = Vec::with_capacity(height * tuple_width);
        for index in 0..height {
            self.table.append_row(index, &mut values);
            assert_eq!(
                values.len(),
                (index + 1) * tuple_width,
                "fixed table row {index} is not its declared tuple width"
            );
        }
        RowMajorMatrix::new(values, tuple_width)
    }

    /// Build the sole witnessed column from one count per fixed entry.
    ///
    /// Every value is prover-chosen. Its LogUp `table_entry` interaction in
    /// [`Air::eval`] is what pins it to the total number of matching call-side
    /// queries.
    #[must_use]
    pub fn multiplicity_trace<F: Field>(
        &self,
        multiplicities: &[u32],
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        assert_eq!(
            multiplicities.len(),
            self.table.height(),
            "one multiplicity is required for every fixed table row"
        );
        let mut values = Vec::with_capacity(self.table.height() << extra_capacity_bits);
        values.extend(multiplicities.iter().copied().map(F::from_u32));
        RowMajorMatrix::new(values, 1)
    }
}

impl<F: Field, T: FixedTable + Sync> BaseAir<F> for FixedTableAir<T> {
    fn width(&self) -> usize {
        // Only the multiplicity is committed as a main-trace witness.
        1
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.fixed_trace())
    }

    fn preprocessed_width(&self) -> usize {
        self.table.tuple_width()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB, T> Air<AB> for FixedTableAir<T>
where
    AB: AirBuilder + InteractionBuilder,
    AB::F: Field,
    T: FixedTable + Sync,
{
    fn eval(&self, builder: &mut AB) {
        let fixed = builder.preprocessed();
        let main = builder.main();
        let multiplicity = main.current(0).expect("multiplicity column");

        // This is the only relation reading the prover-chosen multiplicity.
        // Every input/output coordinate is a verifier-bound preprocessed
        // constant, including all four coordinates of a paired row.
        let tuple = (0..self.table.tuple_width())
            .map(|column| fixed.current(column).expect("fixed table column").into())
            .collect::<Vec<_>>();
        LookupBus::new(self.table.bus()).table_entry(builder, tuple, multiplicity.into());
    }
}

#[cfg(test)]
mod tests {
    use p3_air::BaseAir;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;

    use super::{FixedTable, FixedTableAir};

    /// A stand-in for a construction's own table: eight rows of `x -> 3x`.
    #[derive(Debug, Clone, Copy)]
    struct TripleTable {
        /// Rows claimed, so a wrong `append_row` width can be provoked.
        tuple_width: usize,
    }

    impl FixedTable for TripleTable {
        fn bus(&self) -> &'static str {
            "triple"
        }

        fn height(&self) -> usize {
            8
        }

        fn tuple_width(&self) -> usize {
            self.tuple_width
        }

        fn append_row<F: p3_field::Field>(&self, index: usize, row: &mut Vec<F>) {
            row.push(F::from_usize(index));
            row.push(F::from_usize(3 * index));
        }
    }

    #[test]
    fn the_keys_are_preprocessed_and_only_the_multiplicity_is_witnessed() {
        let air = FixedTableAir::new(TripleTable { tuple_width: 2 });
        assert_eq!(BaseAir::<Goldilocks>::width(&air), 1);
        assert_eq!(BaseAir::<Goldilocks>::preprocessed_width(&air), 2);
        assert_eq!(
            BaseAir::<Goldilocks>::main_next_row_columns(&air),
            Vec::<usize>::new()
        );
        assert_eq!(
            BaseAir::<Goldilocks>::preprocessed_next_row_columns(&air),
            Vec::<usize>::new()
        );

        let fixed = air.fixed_trace::<Goldilocks>();
        assert_eq!((fixed.width(), fixed.height()), (2, 8));
        for index in 0..8 {
            let row = fixed.row_slice(index).unwrap();
            assert_eq!(row[0], Goldilocks::from_usize(index));
            assert_eq!(row[1], Goldilocks::from_usize(3 * index));
        }

        // The LDE capacity is reserved, not filled: the table is exactly its
        // declared height, with no padding row and no selector.
        let multiplicities = air.multiplicity_trace::<Goldilocks>(&[1, 0, 0, 0, 0, 0, 0, 2], 3);
        assert_eq!((multiplicities.width(), multiplicities.height()), (1, 8));
        assert!(multiplicities.values.capacity() >= 8 << 3);
    }

    #[test]
    fn a_row_that_is_not_its_declared_width_is_rejected() {
        let air = FixedTableAir::new(TripleTable { tuple_width: 3 });
        assert!(std::panic::catch_unwind(move || air.fixed_trace::<Goldilocks>()).is_err());
    }

    #[test]
    fn one_multiplicity_per_row_is_required() {
        let air = FixedTableAir::new(TripleTable { tuple_width: 2 });
        assert!(
            std::panic::catch_unwind(move || air.multiplicity_trace::<Goldilocks>(&[0; 7], 0))
                .is_err()
        );
    }
}
