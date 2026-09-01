//! The full-round layout: one committed state per **round**.
//!
//! POLICY §6's four files again, for the second arithmetization of the same
//! permutation — POLICY §11's "how many rounds between state commitments" axis,
//! at the one design where that axis is not a degree-for-width trade.
//!
//! # Why the degree does not move
//!
//! Skipping a state commitment normally multiplies the degree: two S-boxes in a
//! row become `alpha^2`. Not here, because Rescue-Prime's two S-boxes point in
//! *opposite* directions, so each one can be written as a forward power map on
//! its own side of a single equation. With `a` the committed state entering the
//! round and `s` the committed state leaving it,
//!
//! ```text
//!   forward half:   u = M * a^alpha + c_forward
//!   inverse half:   s = M * w       + c_inverse,   w = u^(1/alpha)
//!
//!   so, per word:   (M^{-1} * (s - c_inverse))_i ^ alpha  ==  (M * a^alpha + c_forward)_i
//!                    \_______ degree alpha in s _______/       \____ degree alpha in a ___/
//! ```
//!
//! Both sides are degree `alpha` and they are never multiplied together. The
//! round costs `t` cells instead of `2t`, at the same degree the half-round
//! layout pays.
//!
//! # What the density costs
//!
//! Both sides of that equation are dense — `M^{-1}` is dense however structured
//! `M` is, and the right-hand side is a `t`-term combination of power maps —
//! where the half-round layout compares a power map of *one cell* against a
//! `t`-term affine form. That is the obvious place for this layout to give its
//! saving back, in the expression the prover folds per row.
//!
//! It does not: `tests/numbers.rs` counts the constraint DAG of both layouts and
//! this one is 2% *smaller*. Both build `2R * t^2` matrix terms per call, and
//! this layout packs them into half as many, twice-as-dense constraints. Prover
//! time is the number that settles the choice, which is why both layouts are
//! registered.
//!
//! # The output boundary
//!
//! The last round's committed state **is** the permutation output: there is no
//! trailing layer to apply to it. So this layout has no separate `outputs`
//! block, and adding one would be `t` cells and `t` constraints of pure copy.
//! The last `t` cells of a call are still the output, which is what every
//! consumer reads.

pub mod air;
pub mod columns;
pub mod generation;
pub mod vectorized;
