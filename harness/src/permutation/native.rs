//! The native oracle.

/// A bare permutation, computed natively.
///
/// This is the oracle, not a convenience: it is what `../ref`'s known-answer
/// vectors are replayed against for a construction we write, and what
/// upstream's own implementation is replayed against for one we wrap (POLICY
/// §4). A vector is *never* generated from the implementation under test —
/// including from an AIR, since a trace is not a vector.
///
/// Object-safe on purpose: the KAT runner walks a heterogeneous list of these,
/// keyed by [`NativePermutation::name`].
pub trait NativePermutation<F> {
    /// The instance name, which is the *only* thing tying an implementation to
    /// its vectors (POLICY §3): the reference variable lowercased with `_`
    /// replaced by `-`, exactly what the export script emits. A mismatch here
    /// is silent, which is why `bench` carries a coverage guard.
    fn name(&self) -> &'static str;

    /// State width `t`.
    fn width(&self) -> usize;

    /// Permute `state` in place.
    ///
    /// # Panics
    ///
    /// If `state.len() != self.width()`.
    fn permute(&self, state: &mut [F]);

    /// Convenience wrapper for a KAT: permute a copy and return it.
    ///
    /// # Panics
    ///
    /// If `input.len() != self.width()`.
    fn permute_vec(&self, input: &[F]) -> Vec<F>
    where
        F: Clone,
    {
        let mut state = input.to_vec();
        self.permute(&mut state);
        state
    }
}
