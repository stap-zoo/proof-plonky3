//! The three state-level steps every construction's round writes out by hand.
//!
//! These are not gadgets in [`crate::gadgets`]' sense — there is no witness side
//! and no oracle, because nothing here is prover-chosen. They are the loops that
//! sit *between* the gadgets: adding a constant row, and turning a state
//! expression into committed cells.
//!
//! They earn a home here for the reason POLICY §6 gives for keeping `air.rs` and
//! `generation.rs` line-for-line parallel. A hand-written
//! `zip(rcons[round])` is one index away from a hand-written
//! `zip(rcons[round + 1])`, the two files are the only places that index differs,
//! and every known-answer test still passes when a *constraint* reads the row a
//! generator did not. Naming the step makes the round index the only thing a
//! reader has to check.

use p3_air::AirBuilder;
use p3_field::{Algebra, Dup, PrimeCharacteristicRing};

/// Add one row of round constants to the state.
///
/// Generic in the algebra rather than the field, so the same call serves
/// `air.rs` over `AB::Expr`, the scalar generator over `F`, and the packed
/// generator over `F::Packing`.
#[inline]
pub fn add_round_constants<F, A, const WIDTH: usize>(state: &mut [A; WIDTH], constants: &[F; WIDTH])
where
    F: PrimeCharacteristicRing,
    A: Algebra<F>,
{
    for (word, constant) in state.iter_mut().zip(constants) {
        *word += constant.dup();
    }
}

/// Assert that a state expression equals a row of committed cells.
///
/// The boundary constraint POLICY §9 asks for at the end of a call: without it
/// the output cells are exposed to the tests but pinned by nothing, and every
/// known-answer test still passes.
#[inline]
pub fn assert_state_eq<AB: AirBuilder>(builder: &mut AB, state: &[AB::Expr], cells: &[AB::Var]) {
    debug_assert_eq!(state.len(), cells.len(), "one cell per state word");
    for (value, &cell) in state.iter().zip(cells) {
        builder.assert_eq(value.dup(), cell);
    }
}

/// Assert a state expression equals committed cells, then continue from the
/// cells rather than the expression.
///
/// This is what a mid-call state commitment *is*: the assertion pins the cells,
/// and continuing from them is what resets the expression's degree to one. Doing
/// only the first half leaves the commitment paying width and buying nothing;
/// doing only the second half leaves every committed cell free (POLICY §9).
///
/// The slices may be shorter than the state — pSquareHash commits only the lower
/// half — so this takes slices rather than arrays.
#[inline]
pub fn commit_state<AB: AirBuilder>(builder: &mut AB, state: &mut [AB::Expr], cells: &[AB::Var]) {
    debug_assert_eq!(state.len(), cells.len(), "one cell per committed word");
    for (word, &cell) in state.iter_mut().zip(cells) {
        builder.assert_eq(word.dup(), cell);
        *word = cell.into();
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess, check_constraints};
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{add_round_constants, assert_state_eq, commit_state};

    const CONSTANTS: [Goldilocks; 2] = [Goldilocks::new(7), Goldilocks::new(9)];

    /// `in | mid | out`: add constants, commit, add them again, assert. Small
    /// enough that the trace below can be read off by hand.
    struct LayersAir;

    impl BaseAir<Goldilocks> for LayersAir {
        fn width(&self) -> usize {
            6
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            vec![]
        }
    }

    impl<AB: AirBuilder<F = Goldilocks>> Air<AB> for LayersAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();
            let mut state: [AB::Expr; 2] = core::array::from_fn(|i| row[i].into());
            add_round_constants(&mut state, &CONSTANTS);
            let mid: [AB::Var; 2] = core::array::from_fn(|i| row[2 + i]);
            commit_state(builder, &mut state, &mid);
            add_round_constants(&mut state, &CONSTANTS);
            let out: [AB::Var; 2] = core::array::from_fn(|i| row[4 + i]);
            assert_state_eq(builder, &state, &out);
        }
    }

    fn trace(inputs: [u64; 2]) -> RowMajorMatrix<Goldilocks> {
        let mut row = Vec::new();
        let input = inputs.map(Goldilocks::from_u64);
        let mid = core::array::from_fn::<_, 2, _>(|i| input[i] + CONSTANTS[i]);
        let out = core::array::from_fn::<_, 2, _>(|i| mid[i] + CONSTANTS[i]);
        row.extend(input);
        row.extend(mid);
        row.extend(out);
        RowMajorMatrix::new(row, 6)
    }

    #[test]
    fn an_honest_trace_satisfies_the_constraints() {
        check_constraints(&LayersAir, &trace([0, 1]), &[]);
    }

    /// Every cell is prover-chosen, including the intermediate one `commit_state`
    /// continues from. Corrupt each in turn and expect a rejection — the check
    /// that `commit_state` asserts as well as substitutes.
    #[test]
    fn corrupting_any_cell_is_rejected() {
        let honest = trace([3, 4]);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for cell in 0..honest.values.len() {
            let mut corrupted = honest.clone();
            corrupted.values[cell] += Goldilocks::ONE;
            assert!(
                std::panic::catch_unwind(|| check_constraints(&LayersAir, &corrupted, &[]))
                    .is_err(),
                "cell {cell} is unconstrained"
            );
        }
        std::panic::set_hook(hook);
    }
}
