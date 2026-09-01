//! Type erasure over the two AIRs a lookup statement is made of.
//!
//! `p3-batch-stark` proves heterogeneous heights under one commitment, but all
//! its instances must be one Rust type. A lookup-backed permutation statement
//! is always the same two kinds of instance — the calls, and the once-per-proof
//! fixed tables they query — so that erasure is written once here.
//!
//! This enum is *only* type erasure: every layout and constraint method
//! delegates. No round logic and no table value lives here, and a construction
//! keeps owning both sides it hands in.

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use super::table::{FixedTable, FixedTableAir};

/// One AIR instance in a lookup-backed batch.
#[derive(Debug, Clone)]
pub enum LookupBatchAir<A, T> {
    /// The permutation calls being proved.
    Calls(A),
    /// One once-per-proof fixed function table.
    Table(FixedTableAir<T>),
}

impl<A, T, F> BaseAir<F> for LookupBatchAir<A, T>
where
    A: BaseAir<F>,
    T: FixedTable + Sync,
    F: Field,
{
    fn width(&self) -> usize {
        match self {
            Self::Calls(air) => air.width(),
            Self::Table(air) => BaseAir::<F>::width(air),
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Calls(air) => air.main_next_row_columns(),
            Self::Table(air) => BaseAir::<F>::main_next_row_columns(air),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::Calls(air) => air.preprocessed_trace(),
            Self::Table(air) => BaseAir::<F>::preprocessed_trace(air),
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Calls(air) => air.preprocessed_width(),
            Self::Table(air) => BaseAir::<F>::preprocessed_width(air),
        }
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Calls(air) => air.preprocessed_next_row_columns(),
            Self::Table(air) => BaseAir::<F>::preprocessed_next_row_columns(air),
        }
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        match self {
            Self::Calls(air) => air.max_constraint_degree(),
            Self::Table(air) => BaseAir::<F>::max_constraint_degree(air),
        }
    }
}

impl<AB, A, T> Air<AB> for LookupBatchAir<A, T>
where
    AB: AirBuilder + p3_lookup::InteractionBuilder,
    AB::F: Field,
    A: Air<AB>,
    T: FixedTable + Sync,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Calls(air) => air.eval(builder),
            Self::Table(air) => air.eval(builder),
        }
    }
}

/// AIRs and aligned witness traces for one complete lookup statement.
#[derive(Debug, Clone)]
pub struct LookupBatch<A, T, F> {
    /// Call AIR first, followed by its one or more fixed tables.
    pub airs: Vec<LookupBatchAir<A, T>>,
    /// Main traces aligned with [`Self::airs`].
    pub traces: Vec<RowMajorMatrix<F>>,
}

impl<A, T, F: Clone + Send + Sync> LookupBatch<A, T, F> {
    /// Assemble a statement, checking the two halves line up.
    #[must_use]
    pub fn new(airs: Vec<LookupBatchAir<A, T>>, traces: Vec<RowMajorMatrix<F>>) -> Self {
        assert_eq!(airs.len(), traces.len(), "one trace per AIR instance");
        Self { airs, traces }
    }

    /// No permutation or fixed-table AIR exposes public values.
    #[must_use]
    pub fn public_values(&self) -> Vec<Vec<F>> {
        (0..self.airs.len()).map(|_| Vec::new()).collect()
    }

    /// Base trace heights aligned with [`Self::airs`].
    #[must_use]
    pub fn log_heights(&self) -> Vec<usize> {
        self.traces
            .iter()
            .map(|trace| {
                let height = trace.height();
                assert!(height.is_power_of_two(), "a batch trace is never padded");
                height.ilog2() as usize
            })
            .collect()
    }
}
