//! The bins of one dimension of a tally, other than space.

use std::ops::Range;

/// One dimension of a tally's values (energy, time, ...): its bins and,
/// when MCNP printed one, the Total bin after them. A Total bin is one more
/// index along the dimension, the last; it is never dropped, since MCNP
/// computes its error from the histories themselves, which summing the
/// bins can't reproduce.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    pub bins: Bins,
    /// Whether the last index is MCNP's Total bin.
    pub total: bool,
    /// Whether each bin holds everything up to its upper bound (MCNP's `C`
    /// option), rather than only what falls within it.
    pub cumulative: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Bins {
    /// Bin boundaries: `n + 1` of them for `n` bins.
    Bounds(Vec<f64>),
    /// Upper bounds only, one per bin, as MCTAL files give them; the first
    /// bin starts wherever the quantity does (0 MeV, a cosine of -1, ...).
    Upper(Vec<f64>),
    /// Only how many bins there are.
    Count(usize),
}

impl Axis {
    /// The bins between `bounds`.
    pub fn bounded(bounds: Vec<f64>, total: bool) -> Axis {
        Axis {
            bins: Bins::Bounds(bounds),
            total,
            cumulative: false,
        }
    }

    /// The bins between `bounds`, with a Total bin when there is more than
    /// one (as MCNP prints them).
    pub fn mcnp(bounds: Vec<f64>) -> Axis {
        let total = bounds.len() > 2;
        Axis::bounded(bounds, total)
    }

    /// Number of bins, the Total one excluded.
    pub fn n_bins(&self) -> usize {
        match &self.bins {
            Bins::Bounds(b) => b.len().saturating_sub(1),
            Bins::Upper(b) => b.len(),
            Bins::Count(n) => *n,
        }
    }

    /// Number of indices, the Total one included.
    pub fn len(&self) -> usize {
        self.n_bins() + usize::from(self.total)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn total_index(&self) -> Option<usize> {
        self.total.then(|| self.n_bins())
    }

    /// The bin boundaries, when the file gives all of them.
    pub fn bounds(&self) -> Option<&[f64]> {
        match &self.bins {
            Bins::Bounds(b) => Some(b),
            _ => None,
        }
    }

    /// The upper bound of each bin, when the file gives them.
    pub fn upper(&self) -> Option<&[f64]> {
        match &self.bins {
            Bins::Bounds(b) => b.get(1..),
            Bins::Upper(b) => Some(b),
            Bins::Count(_) => None,
        }
    }

    /// The axis of one bin spanning this one: what summing over it, or
    /// taking its Total, leaves.
    pub fn collapsed(&self) -> Axis {
        let bins = match &self.bins {
            Bins::Bounds(b) if b.len() >= 2 => Bins::Bounds(vec![b[0], b[b.len() - 1]]),
            Bins::Upper(b) if !b.is_empty() => Bins::Upper(vec![b[b.len() - 1]]),
            _ => Bins::Count(1),
        };
        Axis {
            bins,
            total: false,
            cumulative: false,
        }
    }

    /// Bins `range` (which must be within the real ones), with no Total.
    pub fn select(&self, range: Range<usize>) -> Axis {
        let bins = match &self.bins {
            Bins::Bounds(b) => Bins::Bounds(b[range.start..=range.end].to_vec()),
            Bins::Upper(b) => Bins::Upper(b[range.clone()].to_vec()),
            Bins::Count(_) => Bins::Count(range.len()),
        };
        Axis {
            bins,
            total: false,
            cumulative: self.cumulative,
        }
    }

    /// Whether both have the same bins, bounds equal to a relative `tol`.
    pub fn matches(&self, other: &Axis, tol: f64) -> bool {
        self.total == other.total
            && self.cumulative == other.cumulative
            && match (&self.bins, &other.bins) {
                (Bins::Bounds(a), Bins::Bounds(b)) | (Bins::Upper(a), Bins::Upper(b)) => {
                    close(a, b, tol)
                }
                (a, b) => a == b,
            }
    }
}

/// Whether `a` and `b` are equal to a relative `tol`.
pub(crate) fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol * x.abs().max(y.abs()).max(1.0))
}
