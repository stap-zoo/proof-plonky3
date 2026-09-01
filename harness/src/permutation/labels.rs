//! Labels: what a construction tells the harness about itself.

/// The shape facts a construction declares about one arithmetization.
///
/// **These are labels only.** The harness prints them into the cost table
/// (POLICY §11) and never branches on them — the moment a measurement path
/// reads one of these to decide *what to do*, two constructions are being
/// benched differently and the numbers stop being comparable (POLICY §7).
///
/// The single exception is [`Labels::max_constraint_degree`], which the harness
/// does consume — to derive the blowup, in one function, for every construction
/// alike (POLICY §7). It is cross-checked against the symbolic value the AIR
/// actually produces, so a wrong declaration is a test failure rather than a
/// silently cheaper configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Labels {
    /// State width `t` of the permutation (POLICY §3's grid coordinate).
    pub state_width: usize,

    /// Independent permutation calls packed into one row — `VECTOR_LEN`.
    ///
    /// One row is this many independent calls, declared by overriding
    /// `main_next_row_columns()` to `vec![]` (POLICY §6).
    pub calls_per_row: usize,

    /// Rows one call occupies. `1` where the chosen layout keeps a complete
    /// call in one row — including the current wide Monolith and lookup-free
    /// Tip5 baselines. A multi-row layout must carry explicit transition
    /// constraints and declare its next-row columns honestly (POLICY §6).
    pub rows_per_call: usize,

    /// Round count. Provisional for a generated instance (POLICY §3): every
    /// cost number scales with it, so it is a reported column and the pinned
    /// numbers move when it is corrected.
    pub rounds: usize,

    /// S-box degree `alpha`.
    pub sbox_degree: u64,

    /// Registers the S-box is split over. Trades degree against committed
    /// cells; every instance is measured at every variant it admits (POLICY
    /// §11).
    pub sbox_registers: usize,

    /// The AIR's maximum constraint degree, as the construction declares it.
    ///
    /// Cross-checked against `get_max_constraint_degree` in the pinned-number
    /// test (POLICY §11).
    pub max_constraint_degree: usize,
}

impl Labels {
    /// Calls proved by a trace of `2^log_n` rows.
    ///
    /// Tables are always full — no padding, no selector (POLICY §6) — so this
    /// is exact, not an upper bound.
    #[must_use]
    pub const fn calls_in_trace(&self, log_n: usize) -> usize {
        (self.calls_per_row << log_n) / self.rows_per_call
    }
}
