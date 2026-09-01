//! Blowup, derived from the AIR's declared degree, in one function.
//!
//! This is the only place a construction's degree is allowed to change the
//! configuration (POLICY §7), and it is a pure function of the degree and the
//! ZK toggle — never of which construction is being measured.

use p3_util::log2_ceil_usize;

/// The common blowup a row can also be measured at.
///
/// `log_blowup = 3` covers every constraint degree up to 9, hence the whole
/// slice of designs here, so it is the setting in which two rows are strictly
/// like-for-like *in the code rate*. It is the secondary reading: it hands the
/// same rate to a degree-2 and a degree-9 arithmetization, so it prices the
/// narrowness a high degree buys and none of what it costs (POLICY §7, §11).
pub const COMMON_LOG_BLOWUP: usize = 3;

/// The smallest blowup any measurement runs at.
///
/// [`min_log_blowup`] returns `0` for a degree-2 AIR, and a rate-1 code is not
/// a code: FRI has nothing to test and proven security cannot reach any target
/// at all. The floor is a property of the proof system rather than of the
/// arithmetization, so it is applied here and reported as the blowup the row
/// ran at — a degree-2 design is not charged for a degree it does not have,
/// but it is not given a free proof either.
pub const FLOOR_LOG_BLOWUP: usize = 1;

/// The smallest `log_blowup` at which Plonky3 can commit a quotient for an AIR
/// of this maximum constraint degree.
///
/// Mirrors `p3_uni_stark::get_log_num_quotient_chunks`: the number of quotient
/// chunks is `2^ceil(log2(max(d + is_zk, 2) - 1))`, and the PCS blowup must be
/// at least that. Equivalently, the familiar `d <= 2^log_blowup + 1`, one
/// tighter under ZK because the prover commits a trace of twice the length and
/// the quotient bound loses the `+ 1`.
///
/// Deriving this rather than hardcoding a table is what keeps the "minimum
/// blowup" column honest when a new design lands with an unusual degree.
#[must_use]
pub const fn min_log_blowup(max_constraint_degree: usize, zk: bool) -> usize {
    let is_zk = if zk { 1 } else { 0 };
    let degree = {
        let d = max_constraint_degree + is_zk;
        if d < 2 { 2 } else { d }
    };
    log2_ceil_usize(degree - 1)
}

/// The fraction-pin degree Plonky3's same-bus lookup packer actually produces
/// for a lookup AIR whose own `FractionPacking` choice declares `declared_degree`.
///
/// `ProverData::from_airs_and_degrees` (pinned Plonky3 rev, `batch-stark/src/
/// common.rs`) folds same-bus interactions up to `2^log_chunks + 1 - is_zk`,
/// where `log_chunks` is exactly [`min_log_blowup`]'s quotient-chunk bucket for
/// `declared_degree`. Under ZK that bucket boundary can move by a whole
/// doubling, handing the packer more headroom than the construction's chosen
/// packing point asked for, so the realized degree is higher than declared
/// exactly at the boundary degrees (POLICY §11's lookup frontier: 3, 5, 9).
/// This mirrors [`min_log_blowup`] back into a degree so a lookup AIR's
/// declared label stays honest under either side of the ZK toggle.
#[must_use]
pub const fn lookup_packed_degree(declared_degree: usize, zk: bool) -> usize {
    let is_zk = if zk { 1 } else { 0 };
    (1usize << min_log_blowup(declared_degree, zk)) + 1 - is_zk
}

/// The blowup a measurement at each AIR's own minimum actually runs at.
///
/// [`min_log_blowup`] clamped to [`FLOOR_LOG_BLOWUP`]. This is the primary
/// reading of POLICY §11: the degree an arithmetization declares buys it a
/// code rate, and every downstream number — query count, proof size, prover
/// work over the LDE — is charged at that rate. A design that reaches degree 9
/// to save cells pays `2^3` for them here, which is the comparison the common
/// blowup cannot make.
#[must_use]
pub const fn measured_log_blowup(max_constraint_degree: usize, zk: bool) -> usize {
    let minimum = min_log_blowup(max_constraint_degree, zk);
    if minimum < FLOOR_LOG_BLOWUP {
        FLOOR_LOG_BLOWUP
    } else {
        minimum
    }
}

/// Whether `COMMON_LOG_BLOWUP` is usable for this degree.
///
/// A design whose degree does not fit it is not silently dropped from the
/// common-blowup table: it is reported, because "this construction cannot be
/// measured alongside the others at one blowup" is itself a result.
#[must_use]
pub const fn fits_common_blowup(max_constraint_degree: usize, zk: bool) -> bool {
    min_log_blowup(max_constraint_degree, zk) <= COMMON_LOG_BLOWUP
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The degrees this slice of designs actually produces, per POLICY §11's
    /// variant table: `(alpha, registers) -> AIR degree` is `(3,0)->3`,
    /// `(5,0)->5`, `(7,0)->7`, `(5,1)->3`, `(7,1)->3`, `(11,2)->3`.
    #[test]
    fn covers_the_degree_slice() {
        assert_eq!(min_log_blowup(3, false), 1);
        assert_eq!(min_log_blowup(5, false), 2);
        assert_eq!(min_log_blowup(7, false), 3);
        assert_eq!(min_log_blowup(9, false), 3);
        // The first degree that does not fit the common blowup.
        assert_eq!(min_log_blowup(10, false), 4);

        for degree in 2..=9 {
            assert!(fits_common_blowup(degree, false), "degree {degree}");
        }
        assert!(!fits_common_blowup(10, false));
    }

    /// ZK costs exactly one doubling at the boundary degrees and nothing
    /// elsewhere.
    #[test]
    fn zk_is_one_tighter() {
        assert_eq!(min_log_blowup(9, true), 4);
        assert_eq!(min_log_blowup(8, true), 3);
        assert_eq!(min_log_blowup(5, true), 3);
        assert_eq!(min_log_blowup(4, true), 2);
    }

    /// Without ZK the packer never has spare bucket headroom, so the realized
    /// degree is exactly what `FractionPacking` declared.
    #[test]
    fn lookup_packed_degree_matches_declared_without_zk() {
        for degree in [2, 3, 5, 9] {
            assert_eq!(lookup_packed_degree(degree, false), degree);
        }
    }

    /// Under ZK the quotient-chunk bucket can jump by a whole doubling at the
    /// lookup frontier's boundary degrees (POLICY §11: 3, 5, 9), so the packer
    /// folds more same-bus denominators than the declared packing point asked
    /// for. `frac1`'s degree 2 sits inside its bucket already and is
    /// untouched — this is exactly what harness/src/lookup/mod.rs's assertion
    /// caught when the ZK headline sweep panicked.
    #[test]
    fn lookup_packed_degree_grows_at_the_boundary_under_zk() {
        assert_eq!(lookup_packed_degree(2, true), 2);
        assert_eq!(lookup_packed_degree(3, true), 4);
        assert_eq!(lookup_packed_degree(5, true), 8);
        assert_eq!(lookup_packed_degree(9, true), 16);
    }

    /// A trivial AIR must not ask for a rate-1 code.
    #[test]
    fn degenerate_degrees_are_clamped() {
        assert_eq!(min_log_blowup(0, false), 0);
        assert_eq!(min_log_blowup(1, false), 0);
        assert_eq!(min_log_blowup(2, false), 0);
    }

    /// The floor bites exactly where `min_log_blowup` would ask for rate 1, and
    /// nowhere else: a measured row never claims a blowup its AIR did not buy.
    #[test]
    fn the_floor_applies_only_below_it() {
        for degree in 0..=2 {
            assert_eq!(measured_log_blowup(degree, false), FLOOR_LOG_BLOWUP);
        }
        for degree in 3..=32 {
            assert_eq!(
                measured_log_blowup(degree, false),
                min_log_blowup(degree, false),
                "degree {degree}"
            );
            assert_eq!(
                measured_log_blowup(degree, true),
                min_log_blowup(degree, true),
                "degree {degree}, zk"
            );
        }
    }

    /// The primary reading separates the degree slice this repository produces;
    /// the common blowup does not. That is the whole reason it is primary.
    #[test]
    fn the_primary_reading_separates_the_degree_slice() {
        let degrees = [2usize, 3, 4, 5, 7, 9];
        let measured: Vec<_> = degrees
            .iter()
            .map(|&d| measured_log_blowup(d, false))
            .collect();
        assert_eq!(measured, vec![1, 1, 2, 2, 3, 3]);
        for &degree in &degrees {
            assert_eq!(COMMON_LOG_BLOWUP, 3, "degree {degree} pays the same here");
        }
    }
}
