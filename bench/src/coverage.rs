//! The cross-construction guards.

use harness::FieldId;
// `GridPoint` and `Absence` live in `harness` so a construction can write one
// down: the arrow runs `<construction> -> harness`, so they cannot live here.
// See `harness::grid`. They are re-exported through `bench` because this is
// where the cross-construction guards that consume them live.
pub use harness::{Absence, GridPoint};

/// How a construction reaches this repository (POLICY §1, §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registration {
    /// Plonky3 implements it, so that implementation is the artifact. We wrap
    /// it, we do not re-port it, and we make no upstream changes. Validated
    /// against upstream's own native implementation; no reference KAT and no
    /// constant injection, so a one-to-one match with `../ref` is not required.
    Wrapped {
        /// The upstream AIR crate this construction is measured through.
        upstream: &'static str,
    },
    /// Upstream has no AIR. We write one, and it is byte-exact against `../ref`
    /// or it is not validated.
    Written,
    /// Upstream has the native permutation but no AIR: the native side is the
    /// oracle, the AIR is ours.
    ///
    /// The AIR needs every parameter as a value it can evaluate, so upstream's
    /// permutation is driven with the reference's own parameters and both sides
    /// replay the reference's vectors (POLICY §4). A construction registered
    /// this way therefore owes the *written* test ladder, not the wrapped one.
    NativeWrappedAirWritten {
        /// The upstream crate supplying the native permutation.
        upstream: &'static str,
    },
}

/// One construction, as POLICY §1's table lists it.
#[derive(Debug, Clone, Copy)]
pub struct Construction {
    /// The construction's name — the `construction` column of every measured
    /// row and of every [`GridPoint`], and therefore what the comparison grid
    /// identifies a row by (POLICY §3).
    pub name: &'static str,
    /// The crate that provides it, when that is not [`Self::name`].
    ///
    /// One crate may provide more than one construction row: `../ref` defines
    /// XHash8/12 and XHash16/24 through a single class, so they are one crate
    /// here, while the grid keeps them as two rows because it compares
    /// `(construction, field, width)` points.
    pub provided_by: Option<&'static str>,
    /// How it is built.
    pub registration: Registration,
    /// Whether its instances are registered yet. The coverage guard applies to
    /// a construction only once this is true — before that there is nothing to
    /// be inconsistent with.
    pub registered: bool,
}

impl Construction {
    /// The crate `bench` must depend on to reach this construction.
    #[must_use]
    pub const fn crate_name(&self) -> &'static str {
        match self.provided_by {
            Some(krate) => krate,
            None => self.name,
        }
    }
}

/// POLICY §1's thirteen constructions, permutation only.
///
/// Out of scope and deliberately absent: S-GMiMC, Arion, Skyscraper, Polocolo —
/// the reference defines them over curve fields only, and POLICY §3 does not
/// invent instances.
pub const CONSTRUCTIONS: &[Construction] = &[
    Construction {
        name: "poseidon2",
        provided_by: None,
        registration: Registration::Wrapped {
            upstream: "p3-poseidon2-air",
        },
        registered: true,
    },
    Construction {
        name: "poseidon1",
        provided_by: None,
        registration: Registration::Wrapped {
            upstream: "p3-poseidon1-air",
        },
        registered: true,
    },
    Construction {
        name: "monolith",
        provided_by: None,
        registration: Registration::Wrapped {
            upstream: "p3-monolith-air",
        },
        registered: true,
    },
    Construction {
        name: "rescue-prime",
        provided_by: None,
        registration: Registration::NativeWrappedAirWritten {
            upstream: "p3-rescue",
        },
        registered: true,
    },
    Construction {
        name: "neptune",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "anemoi",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "griffin",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "tip5",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
    // Two grid rows, one crate: `../ref` maps both its `xhash8` and `xhash16`
    // export keys onto the same `marvellous.hash.XHash` class.
    Construction {
        name: "xhash8",
        provided_by: Some("xhash"),
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "xhash16",
        provided_by: Some("xhash"),
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "psquarehash",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "gmimc",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
    Construction {
        name: "gmimc2",
        provided_by: None,
        registration: Registration::Written,
        registered: true,
    },
];

/// Grid points a registered construction has neither implemented nor declared
/// absent.
///
/// This is the guard, not a report: a point that is genuinely out of reach gets
/// an [`Absence`] and stays visible in the tables. Silence is the one outcome
/// that is not allowed, because it is indistinguishable from a typo in an
/// instance name.
#[must_use]
pub fn uncovered(instances: &[GridPoint]) -> Vec<(&'static str, FieldId, usize)> {
    let mut missing = Vec::new();
    for construction in CONSTRUCTIONS.iter().filter(|c| c.registered) {
        for &(field, width) in harness::GRID {
            let covered = instances.iter().any(|i| {
                i.construction == construction.name && i.field == field && i.state_width == width
            });
            if !covered {
                missing.push((construction.name, field, width));
            }
        }
    }
    missing
}

/// Instances whose construction is not in [`CONSTRUCTIONS`], or whose name does
/// not start with their construction's name.
///
/// The second half is the actual anti-typo check: POLICY §3 fixes the name as
/// the reference variable lowercased, so `griffin-goldilocks-t8` belongs to
/// `griffin` and a `griffn-…` typo has no home.
#[must_use]
pub fn misnamed(instances: &[GridPoint]) -> Vec<&'static str> {
    instances
        .iter()
        .filter_map(|i| {
            let name = i.instance?;
            let known = CONSTRUCTIONS.iter().any(|c| c.name == i.construction);
            // Tip4/Tip4', XHash12 and XHash24 are named variants in their reference
            // families; POLICY §3 requires preserving those exact names.
            let family_name = (i.construction == "tip5" && matches!(name, "tip4" | "tip4-prime"))
                || (i.construction == "xhash8" && name.starts_with("xhash12-"))
                || (i.construction == "xhash16" && name.starts_with("xhash24-"));
            (!known || (!name.starts_with(i.construction) && !family_name)).then_some(name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `harness`'s manifest, read at compile time. This crate is allowed to
    /// know the construction names; `harness` is not, which is why the check
    /// lives here and not there.
    const HARNESS_MANIFEST: &str = include_str!("../../harness/Cargo.toml");
    /// This crate's own manifest.
    const BENCH_MANIFEST: &str = include_str!("../Cargo.toml");

    /// Whether `manifest` declares a dependency whose key is exactly `name`.
    ///
    /// Key-exact on purpose: `p3-monolith` is not `monolith`, and a substring
    /// search would confuse the upstream crate a wrapper legitimately depends
    /// on with the wrapper itself.
    fn declares(manifest: &str, name: &str) -> bool {
        manifest.lines().any(|line| {
            line.strip_prefix(name)
                .is_some_and(|rest| rest.starts_with('.') || rest.starts_with(" ="))
        })
    }

    #[test]
    fn the_thirteen_of_policy_section_1() {
        assert_eq!(CONSTRUCTIONS.len(), 13);
    }

    /// The subtle half of the two dependency guards below: three of the thirteen
    /// construction names are prefixes of the upstream crate they wrap, so a
    /// substring search would read `p3-monolith` as `monolith` and flag a
    /// legitimate dependency.
    #[test]
    fn key_matching_is_exact() {
        assert!(declares("p3-monolith.workspace = true", "p3-monolith"));
        assert!(!declares("p3-monolith.workspace = true", "monolith"));
        assert!(declares(
            "monolith = { path = \"../monolith\" }",
            "monolith"
        ));
        assert!(!declares("# monolith.workspace = true", "monolith"));
    }

    /// POLICY §5's hard rule: **`harness` never learns which hashes exist.**
    ///
    /// Constructions depend on `harness`, so a dependency the other way is
    /// also a cycle — but the reason to forbid it is not the cycle. A harness
    /// that can name a construction can special-case one, and every number in
    /// the tables stops being comparable the moment it does.
    #[test]
    fn harness_depends_on_no_construction() {
        for construction in CONSTRUCTIONS {
            assert!(
                !declares(HARNESS_MANIFEST, construction.name),
                "harness/Cargo.toml depends on `{}`; POLICY §5 forbids it",
                construction.name
            );
        }
    }

    /// The other half: a construction crate that nothing aggregates is a crate
    /// whose instances are never covered, never validated and never measured —
    /// and it fails silently, because an empty registry is a valid registry.
    #[test]
    fn bench_depends_on_every_construction() {
        for construction in CONSTRUCTIONS {
            assert!(
                declares(BENCH_MANIFEST, construction.crate_name()),
                "bench/Cargo.toml does not depend on `{}`, which provides `{}`",
                construction.crate_name(),
                construction.name
            );
        }
    }

    #[test]
    fn construction_names_are_unique() {
        let mut names: Vec<_> = CONSTRUCTIONS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before);
    }

    /// The grid is eight points: Goldilocks at t = 8, 12 and each 31-bit prime
    /// at t = 16, 24 (POLICY §3).
    #[test]
    fn the_grid_is_the_same_for_every_construction() {
        assert_eq!(harness::GRID.len(), 8);
        for &(field, width) in harness::GRID {
            assert!(field.widths().contains(&width));
        }
    }

    /// The guard, biting: pSquareHash is registered, so all eight of its grid
    /// points must be declared — the two absent Goldilocks ones included.
    #[test]
    fn registered_constructions_cover_the_grid() {
        assert_eq!(uncovered(&crate::instances()), Vec::new());
    }

    #[test]
    fn every_instance_name_belongs_to_its_construction() {
        assert_eq!(misnamed(&crate::instances()), Vec::<&str>::new());
    }

    /// Absences are reported, not silent (POLICY §3): a point without an instance
    /// name carries a reason, and one with a name carries none.
    #[test]
    fn every_absence_has_a_reason() {
        for point in crate::instances() {
            assert_eq!(
                point.instance.is_none(),
                point.absence.is_some(),
                "{}/{:?}/t{}: an absence needs a reason and a present point needs a name",
                point.construction,
                point.field,
                point.state_width
            );
        }
    }

    /// Every registered construction's instance names are distinct, since the name
    /// is what a vector file is looked up by.
    #[test]
    fn instance_names_are_unique() {
        let mut names: Vec<_> = crate::instances()
            .iter()
            .filter_map(|i| i.instance)
            .collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before);
    }
}
