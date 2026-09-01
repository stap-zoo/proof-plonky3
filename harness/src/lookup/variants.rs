//! The two axes a lookup-backed arithmetization is measured along.
//!
//! Both are construction-blind design choices, not properties of a hash: how
//! wide one table message is, and how many same-bus denominators are folded
//! into one auxiliary column. A construction picks a point on each axis and
//! registers the product it actually wants measured — never the naive full
//! cross product (POLICY §11).

/// Granularity of one lookup message.
///
/// Pairing adjacent chunks halves the call-side query count and squares the
/// fixed table's height. Which side wins is a measurement, and the raw
/// `table rows + queries` crossover is only an analytic estimate: table
/// commitments, auxiliary columns, quotient degree and openings are not
/// interchangeable units in the actual proof system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupGranularity {
    /// One `(input chunk, output chunk)` pair per lookup.
    Byte,
    /// Two adjacent input chunks and their two outputs per lookup.
    AdjacentPair,
}

impl LookupGranularity {
    /// Input chunks carried by one message.
    #[must_use]
    pub const fn chunks_per_query(self) -> usize {
        match self {
            Self::Byte => 1,
            Self::AdjacentPair => 2,
        }
    }

    /// Independently range-bound coordinates in one message: inputs and their
    /// outputs, always kept separate. Packing coordinates into one field
    /// element would admit collisions and stop the lookup range-binding either.
    #[must_use]
    pub const fn tuple_width(self) -> usize {
        2 * self.chunks_per_query()
    }
}

/// Maximum number of same-bus fractions folded into one auxiliary column.
///
/// A column carrying `n` denominators has a degree-`n + 1` fraction-pin
/// constraint. These four points therefore span every degree admitted by the
/// common `log_blowup = 3` reading (POLICY §11), and the trade they price is
/// extension columns against quotient degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractionPacking {
    /// One denominator per auxiliary fraction column, degree 2.
    One,
    /// Two denominators per column, degree 3.
    Two,
    /// Four denominators per column, degree 5.
    Four,
    /// Eight denominators per column, degree 9.
    Eight,
}

impl FractionPacking {
    /// Denominators carried by each full fraction column.
    #[must_use]
    pub const fn denominators_per_column(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Degree bought by this packing point.
    #[must_use]
    pub const fn max_constraint_degree(self) -> usize {
        self.denominators_per_column() + 1
    }
}
