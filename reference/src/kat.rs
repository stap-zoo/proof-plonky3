//! Reading `../ref`'s two exports, for the tests of POLICY §10's first layers.
//!
//! What arrives here is always `ref/export_small_prime_kat.py`'s output,
//! verbatim, `include_str!`d by the construction that owns it:
//!
//! ```ignore
//! const VECTORS: Vectors = Vectors::new(include_str!("../vectors/vectors.json"));
//! ```
//!
//! # Why this is shared and not four copies of twenty lines
//!
//! Every written construction needs the same four readers — hex to field
//! element, array to vector, the instance entry of the parameter export, and the
//! permutation vectors of one instance. Four copies drift, and the way they
//! drift is quiet: a loader that silently matches *no* vectors turns a
//! known-answer test into a test that asserts nothing at all and still passes.
//!
//! That failure mode is what shapes the API. **An instance's name is the only
//! thing tying an implementation to its vectors** (POLICY §3), so
//! [`Vectors::permutation`] panics when a name matches nothing, and
//! [`Params::instance`] panics when a name is absent from the parameter export.
//! `bench`'s coverage guard makes the same mismatch loud across constructions;
//! this makes it loud inside one.
//!
//! # Only the permutation vectors
//!
//! The exports carry `compression` and `sponge` vectors too, because they are
//! `../ref`'s own output; POLICY §8 puts modes out of scope, so the readers here
//! filter on `mode == "permutation"`. Filtering rather than rejecting is what
//! lets a construction whose export gains a mode still parse.

use p3_field::PrimeField64;
use serde_json::Value;

/// One known-answer vector: an input state and the state the reference's own
/// permutation produces from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kat<F> {
    /// The permutation input, `t` elements.
    pub input: Vec<F>,
    /// The reference's output, `t` elements.
    pub output: Vec<F>,
}

/// `../ref`'s `vectors.json`, parsed lazily on first use.
pub struct Vectors(&'static str);

/// `../ref`'s `params.json`, parsed lazily on first use.
pub struct Params(&'static str);

/// One instance's entry in the parameter export.
pub struct Instance {
    name: String,
    entry: Value,
}

/// Hex as the export writes it: `"0x1"`, not zero-padded, always a field
/// element of a `p < 2^64` prime.
fn hex_u64(value: &Value) -> u64 {
    let text = value.as_str().expect("a hex string");
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("a hex integer")
}

/// A hex string as a canonical element of `F`.
///
/// `from_canonical_checked` rather than a reduction: the export writes the
/// canonical representative, so a value outside the field is a wrong export or a
/// wrong field, not something to fold quietly into range.
fn elem<F: PrimeField64>(value: &Value) -> F {
    F::from_canonical_checked(hex_u64(value)).expect("a canonical element of this field")
}

fn elems<F: PrimeField64>(value: &Value) -> Vec<F> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(elem)
        .collect()
}

impl Vectors {
    /// Wrap the `include_str!`'d export.
    #[must_use]
    pub const fn new(json: &'static str) -> Self {
        Self(json)
    }

    /// Every `permutation` vector of one instance, in export order.
    ///
    /// The exporter emits two for each instance — the counting input every
    /// construction shares, and the all-zero state, which is where a design
    /// whose round function fixes zero would go unnoticed. Fewer than two means
    /// the instance name or the export is wrong, so this panics rather than
    /// returning a short list a caller might loop over zero times.
    ///
    /// # Panics
    ///
    /// If the export is not valid JSON, or fewer than two vectors carry this
    /// instance name.
    #[must_use]
    pub fn permutation<F: PrimeField64>(&self, instance: &str) -> Vec<Kat<F>> {
        let json: Value = serde_json::from_str(self.0).expect("the export is valid JSON");
        let vectors: Vec<Kat<F>> = json["vectors"]
            .as_array()
            .expect("a vectors array")
            .iter()
            .filter(|vector| vector["instance"] == instance && vector["mode"] == "permutation")
            .map(|vector| Kat {
                input: elems(&vector["input"]),
                output: elems(&vector["output"]),
            })
            .collect();
        assert!(
            vectors.len() >= 2,
            "{instance}: {} permutation vectors in the export — the instance name is the only \
             thing tying an implementation to its vectors (POLICY §3)",
            vectors.len()
        );
        vectors
    }

    /// The instance names the export carries permutation vectors for, sorted.
    ///
    /// A construction's own test uses this to check that it implements every
    /// instance the reference exports, which is the direction
    /// [`Vectors::permutation`] cannot see: that function catches an
    /// implementation whose vectors are missing, this catches a vector whose
    /// implementation is missing.
    ///
    /// # Panics
    ///
    /// If the export is not valid JSON.
    #[must_use]
    pub fn instance_names(&self) -> Vec<String> {
        let json: Value = serde_json::from_str(self.0).expect("the export is valid JSON");
        let mut names: Vec<String> = json["vectors"]
            .as_array()
            .expect("a vectors array")
            .iter()
            .filter(|vector| vector["mode"] == "permutation")
            .filter_map(|vector| vector["instance"].as_str().map(str::to_owned))
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

impl Params {
    /// Wrap the `include_str!`'d export.
    #[must_use]
    pub const fn new(json: &'static str) -> Self {
        Self(json)
    }

    /// One instance's parameter entry.
    ///
    /// # Panics
    ///
    /// If the export is not valid JSON, or carries no instance of this name.
    #[must_use]
    pub fn instance(&self, name: &str) -> Instance {
        let json: Value = serde_json::from_str(self.0).expect("the export is valid JSON");
        let entry = json["instances"]
            .as_array()
            .expect("an instances array")
            .iter()
            .find(|entry| entry["name"] == name)
            .unwrap_or_else(|| panic!("{name}: absent from the parameter export"))
            .clone();
        Instance {
            name: name.to_owned(),
            entry,
        }
    }
}

impl Instance {
    /// A plain integer field of the entry — `t`, `R`, `alpha`, `alpha_inv`.
    ///
    /// # Panics
    ///
    /// If the key is absent or not an integer.
    #[must_use]
    pub fn int(&self, key: &str) -> u64 {
        self.entry[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: {key} is not an integer", self.name))
    }

    /// A hex-encoded field element — `p`, `gamma`, `beta`.
    ///
    /// Read as a `u64` rather than an element of `F`, because `p` itself is one
    /// of these keys and is never canonical in its own field.
    ///
    /// # Panics
    ///
    /// If the key is absent or not a hex string.
    #[must_use]
    pub fn raw(&self, key: &str) -> u64 {
        hex_u64(&self.entry[key])
    }

    /// A hex-encoded field element, as an element.
    ///
    /// # Panics
    ///
    /// If the key is absent, not a hex string, or not canonical in `F`.
    #[must_use]
    pub fn elem<F: PrimeField64>(&self, key: &str) -> F {
        elem(&self.entry[key])
    }

    /// A flat array of field elements — a diagonal, a constant row.
    ///
    /// # Panics
    ///
    /// If the key is absent or is not an array of canonical elements.
    #[must_use]
    pub fn elems<F: PrimeField64>(&self, key: &str) -> Vec<F> {
        elems(&self.entry[key])
    }

    /// A flat array of plain JSON integers — e.g. a byte lookup table or a
    /// mixed-radix base list.
    ///
    /// # Panics
    ///
    /// If the key is absent or is not an array of unsigned integers.
    #[must_use]
    pub fn ints(&self, key: &str) -> Vec<u64> {
        self.entry[key]
            .as_array()
            .unwrap_or_else(|| panic!("{}: {key} is not an integer array", self.name))
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .unwrap_or_else(|| panic!("{}: {key} contains a non-integer", self.name))
            })
            .collect()
    }

    /// A rectangular grid of field elements — a matrix, a round-constant table.
    ///
    /// # Panics
    ///
    /// If the key is absent or is not an array of arrays of canonical elements.
    #[must_use]
    pub fn grid<F: PrimeField64>(&self, key: &str) -> Vec<Vec<F>> {
        self.entry[key]
            .as_array()
            .unwrap_or_else(|| panic!("{}: {key} is not a grid", self.name))
            .iter()
            .map(|row| elems(row))
            .collect()
    }

    /// Three coordinate polynomials in array-of-terms form.
    ///
    /// XHash's export represents each term as `[coefficient, [e0, e1, e2]]`.
    /// Keeping that decoding here makes both XHash families reject malformed
    /// coefficients or exponent tuples in exactly the same way.
    #[must_use]
    pub fn coordinate_polynomials<F: PrimeField64>(&self, key: &str) -> [Vec<(F, [u8; 3])>; 3] {
        let polynomials = self.entry[key]
            .as_array()
            .unwrap_or_else(|| panic!("{}: {key} is not a polynomial array", self.name));
        assert_eq!(
            polynomials.len(),
            3,
            "{}: {key} coordinate count",
            self.name
        );
        core::array::from_fn(|coordinate| {
            polynomials[coordinate]
                .as_array()
                .unwrap_or_else(|| panic!("{}: {key}[{coordinate}] is not a term array", self.name))
                .iter()
                .map(|term| {
                    let term = term
                        .as_array()
                        .unwrap_or_else(|| panic!("{}: {key} term is not an array", self.name));
                    assert_eq!(term.len(), 2, "{}: {key} term length", self.name);
                    let exponents = term[1].as_array().unwrap_or_else(|| {
                        panic!("{}: {key} exponents are not an array", self.name)
                    });
                    assert_eq!(exponents.len(), 3, "{}: {key} exponent count", self.name);
                    (
                        elem(&term[0]),
                        core::array::from_fn(|i| {
                            u8::try_from(exponents[i].as_u64().unwrap_or_else(|| {
                                panic!("{}: {key} exponent is not an integer", self.name)
                            }))
                            .unwrap_or_else(|_| {
                                panic!("{}: {key} exponent does not fit u8", self.name)
                            })
                        }),
                    )
                })
                .collect()
        })
    }

    /// Whether the entry carries this key at all.
    ///
    /// The export's per-construction keys differ, and an instance generated at
    /// the exporter call site may carry one an exact instance does not.
    #[must_use]
    pub fn has(&self, key: &str) -> bool {
        !self.entry[key].is_null()
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_mersenne_31::Mersenne31;

    use super::{Params, Vectors};

    const VECTORS: Vectors = Vectors::new(
        r#"{"vectors": [
            {"instance": "toy-mersenne-t2", "mode": "permutation",
             "input": ["0x1", "0x2"], "output": ["0x3", "0x4"]},
            {"instance": "toy-mersenne-t2", "mode": "permutation",
             "input": ["0x0", "0x0"], "output": ["0x5", "0x6"]},
            {"instance": "toy-mersenne-t2", "mode": "sponge",
             "input": ["0x1"], "output": ["0x7"]},
            {"instance": "toy-mersenne-t4", "mode": "permutation",
             "input": ["0x1"], "output": ["0x1"]}
        ]}"#,
    );

    const PARAMS: Params = Params::new(
        r#"{"instances": [
            {"name": "toy-mersenne-t2", "p": "0x7fffffff", "t": 2, "R": 3,
             "gamma": "0x9", "diag": ["0x1", "0x2"],
             "M": [["0x1", "0x2"], ["0x3", "0x4"]]}
        ]}"#,
    );

    #[test]
    fn permutation_vectors_are_filtered_by_instance_and_mode() {
        let kats = VECTORS.permutation::<Mersenne31>("toy-mersenne-t2");
        assert_eq!(kats.len(), 2, "the sponge vector is not a permutation KAT");
        assert_eq!(kats[0].input, vec![Mersenne31::ONE, Mersenne31::TWO]);
        assert_eq!(
            kats[1].output,
            vec![Mersenne31::from_u32(5), Mersenne31::from_u32(6)]
        );
        assert_eq!(
            VECTORS.instance_names(),
            vec!["toy-mersenne-t2", "toy-mersenne-t4"]
        );
    }

    /// The failure these readers exist to make loud: a name that matches nothing
    /// would otherwise turn a known-answer test into a loop over zero vectors.
    #[test]
    #[should_panic(expected = "0 permutation vectors")]
    fn an_unmatched_instance_name_panics() {
        let _ = VECTORS.permutation::<Mersenne31>("toy-mersenne-t8");
    }

    /// One instance in this export has a single vector, which is a truncated or
    /// hand-edited export rather than something to test against.
    #[test]
    #[should_panic(expected = "1 permutation vectors")]
    fn a_single_vector_is_not_enough() {
        let _ = VECTORS.permutation::<Mersenne31>("toy-mersenne-t4");
    }

    #[test]
    fn parameter_entries_are_read_by_key_and_shape() {
        let instance = PARAMS.instance("toy-mersenne-t2");
        assert_eq!(instance.int("t"), 2);
        assert_eq!(instance.int("R"), 3);
        assert_eq!(instance.raw("p"), Mersenne31::ORDER_U64);
        assert_eq!(
            instance.elem::<Mersenne31>("gamma"),
            Mersenne31::from_u32(9)
        );
        assert_eq!(
            instance.elems::<Mersenne31>("diag"),
            vec![Mersenne31::ONE, Mersenne31::TWO]
        );
        assert_eq!(instance.grid::<Mersenne31>("M").len(), 2);
        assert!(instance.has("gamma"));
        assert!(!instance.has("delta"));
    }

    #[test]
    #[should_panic(expected = "absent from the parameter export")]
    fn an_absent_instance_panics() {
        let _ = PARAMS.instance("toy-mersenne-t8");
    }
}
