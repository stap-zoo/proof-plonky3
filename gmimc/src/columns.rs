//! One call's cells, declared once, as a type.
//!
//! POLICY §6: a `#[repr(C)]` struct for **one call's** cells, `num_cols()` via
//! `size_of`, `Borrow`/`BorrowMut` for `[T]`. Nothing anywhere else computes an
//! offset by hand, and the `prefix.is_empty()` / `suffix.is_empty()` assertions
//! that stand between a layout edit and silent misalignment come with
//! `harness::permutation::impl_call_columns!`.
//!
//! First of the four files (POLICY §2, step 4) — `air.rs` and `generation.rs`
//! both read the layout off this type, which is what stops them drifting apart
//! silently.
//!
//! # The layout never materialises a state
//!
//! Every other written construction here commits a *state* per round, `t` cells
//! wide. GMiMC commits **one cell** per round, and the width does not grow with
//! `t` at all.
//!
//! The reason is the expanding round function. Only branch 0 is ever read
//! nonlinearly; every other branch merely accumulates the same `y` that branch 0
//! produced, and the linear layer is a rotation, so no branch is ever mixed with
//! another. A branch is therefore a running sum of S-box outputs, and the only
//! moment its value is *needed* is the round in which it reaches position 0 and
//! is powered. That value is what this layout commits, and nothing else.
//!
//! Write `b_r` for the value of branch 0 entering round `r`, and
//! `y_r = (b_r + rc_r)^alpha` for the S-box output that round. Then, because a
//! branch leaving position 0 takes exactly `t` rounds to come back and is added
//! to in `t - 1` of them,
//!
//! ```text
//! b_r = b_{r-t} + sum_{j=r-t+1}^{r-1} y_j
//! ```
//!
//! and that is the whole permutation. [`air`](crate::air) derives it, telescopes
//! it into a constraint of bounded size, and argues why committing `b_r` — rather
//! than `y_r`, the other obvious choice — is what keeps it bounded.
//!
//! # One chain, three named blocks
//!
//! The recurrence holds at the boundaries too, given two conventions that cost
//! nothing:
//!
//! * `y_j = 0` for `j < 0` and for `j >= R` — outside the round loop no S-box
//!   runs;
//! * `b_{r-t} = input[r]` for `r` in `0..t` — the branch that enters round `r` at
//!   position 0 is the one that started at position `r`.
//!
//! With those, `b_r = b_{r-t} + sum(...)` is a single rule over `r` in
//! `-t .. R+t`, and `out[i] = b_{R+i}`: after the last round the state is fixed,
//! so `t` further steps of the same recurrence just rotate it past position 0
//! once each, reading off the output in order.
//!
//! So the cells are **one chain** `b_{-t} .. b_{R+t-1}`, named in three blocks
//! only because the two ends are what a known-answer vector compares against:
//!
//! ```text
//! chain index  0            WIDTH              WIDTH+ROUNDS      WIDTH*2+ROUNDS
//!              | inputs     | rounds           | outputs         |
//! round index  -t           0                  R                 R+t
//! ```
//!
//! [`GMiMCCols::chain`] is that index, and it is the only place the three blocks
//! are joined. `air.rs` writes one rule over it and no boundary special cases.
//!
//! # Cost
//!
//! `2*WIDTH + ROUNDS*(1 + REGISTERS)` cells per call — 117 at Goldilocks
//! `t=12, R=93` and 383 at every 31-bit `t=24, R=335`, doubling to 210 and 718
//! with a register.
//!
//! **The width is essentially the round count**, which is what makes 335 rounds
//! affordable and is this construction's whole cost story: pSquareHash's
//! narrowest `t=24` layout is 360 cells for 52 rounds, GMiMC's is 383 for 335.
//! It is also POLICY §11's floor at degree `alpha` — one cell per round, one
//! round per `alpha`-th power — and `air.rs` is where that is derived rather than
//! asserted.

use harness::permutation::impl_call_columns;

/// One independent permutation call.
///
/// The three blocks are one chain (see the module docs); [`Self::chain`] is what
/// reads it as one. `REGISTERS` is the power-map gadget's register split and the
/// only variant axis this construction has — flattening is already at the floor
/// and spacing has nothing to buy, both argued in [`air`](crate::air).
#[repr(C)]
pub struct GMiMCCols<T, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize> {
    /// The permutation input: `b_{-t} .. b_{-1}`, the first `WIDTH` links of the
    /// chain.
    ///
    /// `inputs[i]` is the branch that starts at position `i`, which is the branch
    /// that reaches the S-box at round `i`.
    pub inputs: [T; WIDTH],

    /// The rounds, in order: `b_0 .. b_{R-1}`.
    pub rounds: [Round<T, REGISTERS>; ROUNDS],

    /// The permutation output: `b_R .. b_{R+t-1}`, the last `WIDTH` links.
    ///
    /// Committed rather than left as an expression, for the same reason as
    /// everywhere else — POLICY §10's second layer reads these cells off a
    /// KAT-input trace — and here it is free besides: they are links of the same
    /// chain, pinned by the same rule, and `air.rs` needs no output boundary
    /// constraint of its own.
    pub outputs: [T; WIDTH],
}

/// One round's cells: the branch value entering the S-box, and the register
/// flattening it.
///
/// Both are prover-chosen and worth nothing until a constraint ties them down
/// (POLICY §9): `head` by the chain rule in
/// [`air::eval`](crate::air::eval), `registers` by the power-map gadget's own
/// `register == value^2` assertion.
#[repr(C)]
pub struct Round<T, const REGISTERS: usize> {
    /// `b_r`: branch 0's value entering round `r`, **before** the round constant.
    ///
    /// The constant is an argument to the S-box and never lands in the state
    /// (`native.rs`), so `y_r = (head + rc_r)^alpha`. GMiMC2 is where that
    /// differs, and it differs in this cell's meaning.
    pub head: T,

    /// The power map's register, or nothing. See
    /// `harness::gadgets::power_map`.
    pub registers: [T; REGISTERS],
}

impl<T: Copy, const WIDTH: usize, const REGISTERS: usize, const ROUNDS: usize>
    GMiMCCols<T, WIDTH, REGISTERS, ROUNDS>
{
    /// Link `i` of the chain: `b_{i-WIDTH}`.
    ///
    /// The offset is what removes the boundaries as a special case: the three
    /// blocks are contiguous in round order, so `chain(i)` is `b_r` with
    /// `r = i - WIDTH`, and `i` runs over `0 .. chain_len()`.
    ///
    /// # Panics
    ///
    /// If `i >= chain_len::<WIDTH, ROUNDS>()`.
    #[must_use]
    #[inline]
    pub fn chain(&self, i: usize) -> T {
        if i < WIDTH {
            self.inputs[i]
        } else if i < WIDTH + ROUNDS {
            self.rounds[i - WIDTH].head
        } else {
            self.outputs[i - WIDTH - ROUNDS]
        }
    }
}

/// Links in one call's chain: `R + 2t`.
#[must_use]
pub const fn chain_len<const WIDTH: usize, const ROUNDS: usize>() -> usize {
    ROUNDS + 2 * WIDTH
}

/// Layout and variant invariant.
///
/// `WIDTH >= 2` is the reference's own bound — an expanding round function needs
/// a branch to expand into. The register variants are the power-map gadget's, and
/// unlike Griffin's, alpha 3 gets one: nothing in this AIR sets a degree floor
/// above the S-box, so a register takes 3 to 2. Whether that is worth a cell per
/// round is the measured question POLICY §11 asks, not one this file answers.
///
/// Called from every `new`, so an illegal parameter set fails at monomorphization
/// rather than building a trace whose cells mean something other than what
/// `air.rs` reads.
pub const fn assert_layout(width: usize, alpha: u64, registers: usize) {
    assert!(width >= 2, "an erf Feistel needs at least two branches");
    match (alpha, registers) {
        (3 | 5 | 7, 0 | 1) => {}
        _ => panic!("GMiMC supports alpha 3, 5 or 7 with zero or one register"),
    }
}

impl_call_columns!(GMiMCCols, WIDTH: usize, REGISTERS: usize, ROUNDS: usize);

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    /// The width formula, spelled out, against `size_of`. This is the number
    /// every other cost column is downstream of, and POLICY §11 pins it — but a
    /// pin only catches a *change*, not a layout that was wrong from the start.
    #[test]
    fn the_cell_count_is_the_chain_plus_its_registers() {
        // Goldilocks t = 12, R = 93.
        assert_eq!(num_cols::<12, 0, 93>(), 2 * 12 + 93);
        assert_eq!(num_cols::<12, 0, 93>(), 117);
        assert_eq!(num_cols::<12, 1, 93>(), 210);

        // Every 31-bit prime, t = 24, R = 335.
        assert_eq!(num_cols::<24, 0, 335>(), 2 * 24 + 335);
        assert_eq!(num_cols::<24, 0, 335>(), 383);
        assert_eq!(num_cols::<24, 1, 335>(), 718);

        // The width is the round count plus a constant: doubling `t` at a fixed
        // `R` adds `2t` cells and nothing else. That is the claim the module docs
        // make and the reason this construction can afford 335 rounds.
        assert_eq!(
            num_cols::<24, 0, 335>() - num_cols::<12, 0, 335>(),
            2 * (24 - 12)
        );
    }

    /// An empty register array must cost nothing, or the unsplit variant pays for
    /// cells it does not have and `align_to` stops landing on row boundaries.
    #[test]
    fn empty_registers_cost_nothing() {
        assert_eq!(core::mem::size_of::<Round<u8, 0>>(), 1);
        assert_eq!(core::mem::size_of::<Round<u8, 1>>(), 2);
    }

    /// The chain index against the three blocks it hides, at both seams.
    ///
    /// `air.rs` writes one rule over `chain`, so an off-by-one here is an
    /// off-by-one in every constraint at once — and one that still yields a
    /// consistent-looking trace, because `generation.rs` would be writing the
    /// same cells. The KAT is what would catch it; this is what says where.
    #[test]
    fn the_chain_runs_inputs_then_rounds_then_outputs() {
        let cells: Vec<u32> = (0..num_cols::<3, 0, 4>() as u32).collect();
        let cols: &GMiMCCols<u32, 3, 0, 4> = cells.as_slice().borrow();

        assert_eq!(chain_len::<3, 4>(), 10);
        let chain: Vec<u32> = (0..chain_len::<3, 4>()).map(|i| cols.chain(i)).collect();
        assert_eq!(chain, (0..10).collect::<Vec<_>>());

        // The seams: last input, first round cell; last round cell, first output.
        assert_eq!(cols.chain(2), cols.inputs[2]);
        assert_eq!(cols.chain(3), cols.rounds[0].head);
        assert_eq!(cols.chain(6), cols.rounds[3].head);
        assert_eq!(cols.chain(7), cols.outputs[0]);
    }

    /// With a register the chain is no longer contiguous in memory, which is
    /// exactly why `chain` exists rather than a slice offset.
    #[test]
    fn a_register_interleaves_the_chain_without_moving_it() {
        let cells: Vec<u32> = (0..num_cols::<2, 1, 3>() as u32).collect();
        let cols: &GMiMCCols<u32, 2, 1, 3> = cells.as_slice().borrow();

        // inputs 0,1 | (head 2, register 3) (4, 5) (6, 7) | outputs 8,9
        assert_eq!(cols.chain(2), 2);
        assert_eq!(cols.chain(3), 4);
        assert_eq!(cols.chain(4), 6);
        assert_eq!(cols.chain(5), 8);
        assert_eq!(cols.rounds[0].registers, [3]);
    }

    #[test]
    fn the_layout_invariant_holds_at_the_grid_points() {
        // Goldilocks t=12 alpha 7, Mersenne-31 alpha 5, BabyBear 7, KoalaBear 3.
        assert_layout(12, 7, 0);
        assert_layout(12, 7, 1);
        assert_layout(24, 5, 1);
        assert_layout(24, 3, 1);
    }

    #[test]
    #[should_panic(expected = "at least two branches")]
    fn a_state_too_small_to_expand_into_is_rejected() {
        assert_layout(1, 7, 0);
    }

    /// alpha 4 is GMiMC2's, not this crate's: a non-bijective S-box is refused
    /// here because `params.rs` refuses it, the reference's own hard error.
    #[test]
    #[should_panic(expected = "alpha 3, 5 or 7")]
    fn an_exponent_the_reference_would_refuse_is_rejected() {
        assert_layout(12, 4, 0);
    }
}
