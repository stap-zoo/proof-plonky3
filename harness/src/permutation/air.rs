//! The AIR contract the harness consumes.

use p3_air::symbolic::SymbolicAirBuilder;
use p3_air::{Air, BaseAir, DebugConstraintBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{QuotientAir, StarkGenericConfig, VerifierConstraintFolder};

use super::Labels;

/// One arithmetization of one instance, as the harness sees it.
///
/// The bound list is not decoration: it is exactly the four builders a real
/// proof drives an AIR through — the debug builder `prove` itself runs under
/// `debug_assertions`, the symbolic builder that yields the degree and the
/// constraint count, the quotient builder, and the verifier's folder. Upstream
/// `p3_examples::airs::ExampleHashAir` is the same list, and it is repeated
/// here rather than reused because that crate is an example, not an interface.
///
/// A construction implements this and nothing else. Everything a measurement
/// needs beyond it — PCS, MMCS hash, DFT, FRI parameters, challenge extension,
/// challenger, security target, RNG seed — belongs to the harness, is built
/// once per configuration outside every timed region, and is identical for
/// every construction (POLICY §7). That, and nothing softer, is the basis for
/// claiming the rows are comparable.
pub trait PermutationAir<F: Field, SC: StarkGenericConfig>:
    BaseAir<F>
    + for<'a> Air<DebugConstraintBuilder<'a, F>>
    + Air<SymbolicAirBuilder<F>>
    + QuotientAir<SC>
    + for<'a> Air<VerifierConstraintFolder<'a, SC>>
{
    /// What this arithmetization is, for the table. Labels only — see
    /// [`Labels`].
    const LABELS: Labels;

    /// Build the trace proving exactly these permutation calls.
    ///
    /// This is the KAT path (POLICY §10, layer 2): the output cells of a trace
    /// generated from KAT inputs are compared against the reference's own
    /// outputs, which validates `generation.rs` against the reference rather
    /// than merely against the native oracle.
    ///
    /// Each element of `inputs` is one call's input state, of length
    /// `LABELS.state_width`. Tables are full (POLICY §6), so `inputs.len()`
    /// must equal `LABELS.calls_in_trace(log_n)` for the `log_n` the caller
    /// wants — an implementation does not pad and does not invent inputs.
    ///
    /// `extra_capacity_bits` is forwarded to the allocation so the prover's LDE
    /// happens in place.
    fn generate_trace(&self, inputs: &[Vec<F>], extra_capacity_bits: usize) -> RowMajorMatrix<F>;

    /// The same trace, over `num_calls` inputs drawn from a seeded RNG.
    ///
    /// This is the measurement path and only the measurement path: random
    /// inputs are never used for validation (POLICY §7). The seed comes from
    /// the harness and is the same everywhere, so two rows differ by the AIR
    /// and not by their inputs.
    fn generate_trace_seeded(
        &self,
        num_calls: usize,
        seed: u64,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>;
}
