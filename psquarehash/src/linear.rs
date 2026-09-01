//! The fused round's linear parts, over any ring.
//!
//! Not a fifth of POLICY §6's four files — it is the piece §6 lets two
//! arithmetizations *share*, next to [`crate::params`] and [`crate::native`]. Two
//! arithmetizations times two sides each is four copies of `M`, and `M` is the
//! same matrix in all four.
//!
//! What earns a home here is exactly what [`harness::permutation::layers`] earns
//! one for, and for the reason that module states: these are the loops that sit
//! *between* the gadgets, they have no witness side and no prover-chosen value,
//! and a hand-written index is one keystroke from a different hand-written index
//! that every known-answer test still passes. `M` is
//!
//! ```text
//! out[2p]     = x[h+2p]   + y4(pair h-2-2p) + [p > 0] z.0 + [p = t/4-1] extra.0
//! out[2p+1]   = x[h+2p+1] + y5(pair h-2-2p) + [p > 0] z.1 + [p = t/4-1] extra.1
//! out[h + i]  = x[i]
//! ```
//!
//! and everything in it except the two Feistel outputs is here: [`prologue`] is
//! `z` and `extra`, [`write_upper_half`] is the swap, [`m_io`] is the affine
//! bracket. The `y` terms are not, because *which* of them a variant writes and
//! which it recovers by differencing is the whole content of a variant — see
//! [`full_commit::air::eval`](crate::full_commit::air::eval) against
//! [`half_commit::air::eval`](crate::half_commit::air::eval).
//!
//! **One ring parameter serves all four callers.** `M` is additions and doublings,
//! so the same function runs over `AB::Expr` in the two AIRs, over `F` in the
//! scalar generators and over `F::Packing` in the packed ones. That is why sharing
//! it does not weaken POLICY §6's rule that `air.rs` and `generation.rs` stay
//! line-for-line parallel: the rule exists so a *constraint* cannot read what a
//! generator did not write, and a single shared linear layer is the strongest
//! possible version of that, not an exception to it.
//!
//! **The independent oracle is untouched.** `native.rs` still applies the
//! nonlinear layer and then `M` as two steps over a materialized intermediate
//! state, with its own `m_io`, exactly as `hash.py` does — so the fusion is still
//! checked against something that does not share a line with it, and the KAT is
//! still what settles it (POLICY §4). Nothing in this module may be reused there.

use p3_field::PrimeCharacteristicRing;

/// `M_IO`, in place: `out[i] = x[i] + x[h+i]`, `out[h+i] = 2·x[i] + x[h+i]`.
///
/// Affine, so it costs no cells and no degree — which is why the trace commits
/// the permutation's input rather than `M_IO`'s output.
///
/// Brackets the round loop at both ends. The earlier port of this design left
/// both brackets out and was self-consistent about it, which is why every one of
/// its tests passed while it proved a different permutation than the reference's
/// (`crate::lib`, "What the port changed").
#[inline]
pub(crate) fn m_io<A: PrimeCharacteristicRing>(state: &mut [A]) {
    let h = state.len() / 2;
    let (lo, hi) = state.split_at_mut(h);
    for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
        let (a, b) = (l.dup(), h.dup());
        *l = a.dup() + b.dup();
        *h = a.double() + b;
    }
}

/// `M`'s two contributions to the new lower half that do not come from a Feistel:
/// `(z, extra)`.
///
/// ```text
/// z     = (2·x[h-2] + x[h-1],  x[h-2] + x[h-1])
/// extra = (Σ x[2q],  Σ x[2q+1])          for q = 1 .. t/4 - 2
/// ```
///
/// `z` is added into every output pair but the first, `extra` into the last one
/// only. Both read the *old* lower half, which a round does not modify, so a
/// caller is free to write its outputs in any order — and both are affine, so
/// neither costs a cell or a degree in any variant.
///
/// `z.0 − z.1 = x[h−2]` exactly, which is the identity
/// [`crate::half_commit::air`] differences its even half with. It is a property of
/// this function, so a change here is a change there.
///
/// Takes a slice rather than `&[A; WIDTH]`: `t = 4·PAIRS` is a layout invariant
/// (`assert_layout`), so the length determines both `h` and the number of pairs
/// and there is nothing for a second const parameter to disagree with.
#[inline]
pub(crate) fn prologue<A: PrimeCharacteristicRing>(state: &[A]) -> ([A; 2], [A; 2]) {
    let h = state.len() / 2;
    let pairs = state.len() / 4;
    debug_assert_eq!(state.len(), 4 * pairs, "t must be 4 * PAIRS");
    debug_assert!(pairs >= 2, "t < 8 leaves the z-pair undefined");

    let z = [
        state[h - 2].dup().double() + state[h - 1].dup(),
        state[h - 2].dup() + state[h - 1].dup(),
    ];
    let mut extra = [A::ZERO, A::ZERO];
    // `saturating_sub` rather than `pairs - 1`: the `debug_assert` above and
    // `assert_layout` both rule `pairs < 2` out, and neither runs in a release
    // build, so this keeps a hypothetical bad width to an empty sum instead of an
    // underflow.
    for q in 1..pairs.saturating_sub(1) {
        extra[0] += state[2 * q].dup();
        extra[1] += state[2 * q + 1].dup();
    }
    (z, extra)
}

/// The half swap: the new upper half is the old lower half, verbatim.
///
/// This is `M`'s first job and the reason committing "the state after round `r`"
/// costs `t/2` cells rather than `t` — the other half is already a column one
/// round back (`full_commit::columns`).
///
/// A caller does this **after** writing its lower half, which is what lets
/// [`prologue`] and the Feistels read the old values out of `state` while `next`
/// is being filled.
#[inline]
pub(crate) fn write_upper_half<A: PrimeCharacteristicRing>(next: &mut [A], state: &[A]) {
    debug_assert_eq!(next.len(), state.len(), "one new word per old word");
    let h = state.len() / 2;
    for i in 0..h {
        next[h + i] = state[i].dup();
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_mersenne_31::Mersenne31;

    use super::*;

    type F = Mersenne31;

    fn state(t: usize) -> Vec<F> {
        (1..=t as u32).map(F::from_u32).collect()
    }

    /// `M_IO` against the reference's `_init_mat_IO`, written out by hand at
    /// `t = 8`, and against the native oracle's own copy at both grid widths.
    ///
    /// The second half is the one that matters: `native.rs` deliberately keeps its
    /// own `m_io` so that the fused path is checked against something independent
    /// (POLICY §4), and this is where the two are compared directly rather than
    /// through a whole permutation.
    #[test]
    fn m_io_is_the_reference_matrix() {
        let mut s = state(8);
        m_io(&mut s);
        // h = 4: out[i] = x[i] + x[4+i], out[4+i] = 2·x[i] + x[4+i].
        let expect = [1 + 5, 2 + 6, 3 + 7, 4 + 8, 2 + 5, 4 + 6, 6 + 7, 8 + 8];
        assert_eq!(s, expect.map(F::from_u32).to_vec());
    }

    /// `z.0 − z.1 = x[h−2]`, the identity `half_commit`'s differenced even half
    /// rests on.
    ///
    /// Asserted here because it is a property of [`prologue`] rather than of that
    /// variant: this is the test that fails if `z` is ever redefined, which is the
    /// only way the differencing could silently stop computing the round function.
    #[test]
    fn z_differences_to_the_last_lower_half_even_word() {
        for t in [8, 16, 24] {
            let s = state(t);
            let (z, _) = prologue(&s);
            assert_eq!(z[0] - z[1], s[t / 2 - 2], "t = {t}");
        }
    }

    /// `extra` sums the interior lower-half pairs and nothing else, and is zero
    /// when there are none.
    #[test]
    fn extra_covers_the_interior_pairs_only() {
        // t = 8: PAIRS = 2, so `1..1` is empty — the last pair is the only
        // interior candidate and it is the one `z` already covers.
        let (_, extra) = prologue(&state(8));
        assert_eq!(extra, [F::ZERO, F::ZERO]);

        // t = 16: PAIRS = 4, so q = 1, 2 — words 2,4 and 3,5.
        let (_, extra) = prologue(&state(16));
        assert_eq!(extra, [F::from_u32(3 + 5), F::from_u32(4 + 6)]);

        // t = 24: PAIRS = 6, so q = 1..4 — words 2,4,6,8 and 3,5,7,9.
        let (_, extra) = prologue(&state(24));
        assert_eq!(
            extra,
            [F::from_u32(3 + 5 + 7 + 9), F::from_u32(4 + 6 + 8 + 10)]
        );
    }

    /// The swap writes the upper half from the lower one and touches nothing else.
    #[test]
    fn the_swap_copies_the_lower_half_up() {
        let s = state(8);
        let mut next = vec![F::ZERO; 8];
        write_upper_half(&mut next, &s);
        assert_eq!(next[..4], vec![F::ZERO; 4]);
        assert_eq!(next[4..], s[..4]);
    }
}
