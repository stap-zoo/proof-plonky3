//! The vectorized row every written construction repeats.
//!
//! POLICY §6 fixes this layer completely: `VectorizedXCols { cols: [XCols;
//! LANES] }`, an `eval` that loops `air.rs`'s **free** per-call `eval` over the
//! lanes and contains no round-function logic of its own, a `BaseAir` whose
//! width is one call's times `LANES` and whose `main_next_row_columns()` is
//! `vec![]`, and the [`PermutationAir`](super::PermutationAir) contract with its
//! labels and two trace generators.
//!
//! Written out per construction that is some hundred and ten lines in which the
//! only construction-specific tokens are type names and const-parameter lists —
//! and three of those lines are load-bearing in a way review does not catch:
//!
//! * `main_next_row_columns() -> vec![]` is the *declaration* that a row is
//!   independent calls. Omit it and the prover silently opens the shifted trace,
//!   every test still passes, and the measured numbers are for a different
//!   layout.
//! * `width() == one call * LANES` and `calls_per_row: LANES` are what make a
//!   per-call cost a per-call cost. A mismatch misreports every row that AIR
//!   appears in rather than failing.
//! * looping the free `eval` is what stops the per-call and per-row column
//!   layouts diverging, because it is the same function `air.rs` is tested
//!   through (POLICY §6). A hand-inlined round here is a second layout, and only
//!   the first one is tested.
//!
//! [`impl_vectorized_air`] writes all three, so they are structural rather than
//! reviewed. What stays in the construction is the lane struct, the AIR struct
//! and its constructor — the parts that genuinely differ.

/// Implement `BaseAir`, `Air` and
/// [`PermutationAir`](super::PermutationAir) for a vectorized AIR.
///
/// The construction still declares the two structs; this writes the impls over
/// them. Every argument is tokens from the call site, so a const parameter named
/// in `degree:` or `labels:` resolves to the one `generics:` introduced.
///
/// ```ignore
/// impl_vectorized_air!(
///     air: VectorizedGriffinAir { lane: air, params: params },
///     cols: VectorizedGriffinCols<WIDTH, REGISTERS, ROUNDS, LANES>,
///     eval: eval,
///     generate: generate_vectorized_trace_rows<WIDTH, REGISTERS, ROUNDS, ALPHA, LANES>,
///     generics: [WIDTH: usize, REGISTERS: usize, ROUNDS: usize, ALPHA: u64, LANES: usize],
///     field: p3_field::PrimeField64,
///     state_width: WIDTH,
///     lanes: LANES,
///     degree: max_constraint_degree(ALPHA, REGISTERS),
///     labels: { rounds: ROUNDS, sbox_degree: ALPHA, sbox_registers: REGISTERS },
/// );
/// ```
///
/// * `air:` names the vectorized AIR struct, the field holding one lane's AIR,
///   and the field path holding whatever the generator takes (`params`,
///   `air.params`, `air.constants` — a dotted path is accepted).
/// * `generics:` is the vectorized AIR's full const-parameter list, in
///   declaration order. `cols:` and `generate:` each name the subset they take,
///   in *their* order, which is why the lists are given separately rather than
///   derived from it.
/// * `field:` is the bound each impl puts on the value type. It is the
///   construction's own — `PrimeField64` for most, `PrimeCharacteristicRing`
///   where the AIR never needs integer structure — so this macro never widens or
///   narrows what a construction is generic in.
/// * `degree:` is evaluated in three places that must agree: `BaseAir`'s
///   declaration, the label the harness reports, and the blowup derived from it.
///   One expression here is what keeps them equal.
#[macro_export]
macro_rules! impl_vectorized_air {
    (
        air: $air:ident { lane: $lane:ident, params: $($params:ident).+ $(,)? },
        cols: $cols:ident<$($cols_param:ident),* $(,)?>,
        eval: $eval:ident,
        generate: $generate:ident<$($gen_param:ident),* $(,)?>,
        generics: [$($param:ident : $ty:ty),* $(,)?],
        field: $field:path,
        state_width: $state_width:ident,
        lanes: $lanes:ident,
        degree: $degree:expr,
        labels: {
            rounds: $rounds:expr,
            sbox_degree: $sbox_degree:expr,
            sbox_registers: $sbox_registers:expr $(,)?
        } $(,)?
    ) => {
        impl<F: $field + ::core::marker::Sync $(, const $param: $ty)*>
            ::p3_air::BaseAir<F> for $air<F $(, $param)*>
        {
            fn width(&self) -> usize {
                ::p3_air::BaseAir::<F>::width(&self.$lane) * $lanes
            }

            /// The declaration that makes the layout pay off: one row is
            /// `LANES` *independent* calls, nothing reads the next row, and the
            /// prover skips opening the shifted trace (POLICY §6).
            fn main_next_row_columns(&self) -> ::std::vec::Vec<usize> {
                ::std::vec![]
            }

            fn max_constraint_degree(&self) -> ::core::option::Option<usize> {
                ::core::option::Option::Some($degree)
            }
        }

        impl<AB: ::p3_air::AirBuilder $(, const $param: $ty)*> ::p3_air::Air<AB>
            for $air<AB::F $(, $param)*>
        where
            AB::F: $field,
        {
            fn eval(&self, builder: &mut AB) {
                use ::core::borrow::Borrow;
                use ::p3_air::WindowAccess;

                let main = builder.main();
                let cols: &$cols<_ $(, $cols_param)*> = main.current_slice().borrow();

                // Every lane is constrained, which is what makes a full table
                // valid without a selector and an all-zero row invalid. The
                // per-call `eval` is `air.rs`'s own: no round-function logic
                // lives at this level (POLICY §6).
                for call in &cols.cols {
                    $eval(&self.$lane, builder, call);
                }
            }
        }

        impl<SC: ::p3_uni_stark::StarkGenericConfig $(, const $param: $ty)*>
            $crate::permutation::PermutationAir<::p3_uni_stark::Val<SC>, SC>
            for $air<::p3_uni_stark::Val<SC> $(, $param)*>
        where
            ::p3_uni_stark::Val<SC>: $field,
            ::rand::distr::StandardUniform:
                ::rand::distr::Distribution<::p3_uni_stark::Val<SC>>,
        {
            /// Labels only — the harness prints these and never branches on
            /// them, except for `max_constraint_degree`, which derives the
            /// blowup and is cross-checked against the symbolic value
            /// (POLICY §7, §11).
            const LABELS: $crate::permutation::Labels = $crate::permutation::Labels {
                state_width: $state_width,
                calls_per_row: $lanes,
                rows_per_call: 1,
                rounds: $rounds,
                sbox_degree: $sbox_degree,
                sbox_registers: $sbox_registers,
                max_constraint_degree: $degree,
            };

            fn generate_trace(
                &self,
                inputs: &[::std::vec::Vec<::p3_uni_stark::Val<SC>>],
                extra_capacity_bits: usize,
            ) -> ::p3_matrix::dense::RowMajorMatrix<::p3_uni_stark::Val<SC>> {
                $generate::<_ $(, $gen_param)*>(
                    inputs,
                    &self.$($params).+,
                    extra_capacity_bits,
                )
            }

            fn generate_trace_seeded(
                &self,
                num_calls: usize,
                seed: u64,
                extra_capacity_bits: usize,
            ) -> ::p3_matrix::dense::RowMajorMatrix<::p3_uni_stark::Val<SC>> {
                use ::rand::{RngExt, SeedableRng};

                // Measurement only, never validation (POLICY §7). The seed is
                // the harness's and the same for every construction, so two rows
                // differ by their AIR and not by their inputs.
                let mut rng = ::rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
                let inputs: ::std::vec::Vec<::std::vec::Vec<::p3_uni_stark::Val<SC>>> = (0
                    ..num_calls)
                    .map(|_| {
                        (0..$state_width)
                            .map(|_| rng.sample(::rand::distr::StandardUniform))
                            .collect()
                    })
                    .collect();
                $generate::<_ $(, $gen_param)*>(
                    &inputs,
                    &self.$($params).+,
                    extra_capacity_bits,
                )
            }
        }
    };
}
