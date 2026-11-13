//! Selecting part of a tally's axes, and what can go wrong combining
//! tallies.

use std::ops::Range;

use thiserror::Error;

use super::Axis;

#[derive(Debug, Error, PartialEq)]
pub enum OpsError {
    #[error("tallies have different shapes: {0:?} and {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),
    #[error("tallies have different {0} bins")]
    BinningMismatch(&'static str),
    #[error("{axis}: bins {start}..{end} are not within 0..{len}")]
    OutOfBounds {
        axis: &'static str,
        start: usize,
        end: usize,
        len: usize,
    },
    #[error("{axis}: the selection is empty")]
    Empty { axis: &'static str },
    #[error("{axis}: the tally has no Total bin")]
    NoTotal { axis: &'static str },
}

/// What to keep of an axis.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Select {
    /// All of it, its Total bin included.
    #[default]
    All,
    /// These bins (no Total).
    Bins(Range<usize>),
    /// Its Total bin only: the axis becomes one bin spanning it.
    Total,
}

pub(crate) fn check(axis: &'static str, r: &Range<usize>, len: usize) -> Result<(), OpsError> {
    if r.start >= r.end {
        return Err(OpsError::Empty { axis });
    }
    if r.end > len {
        return Err(OpsError::OutOfBounds {
            axis,
            start: r.start,
            end: r.end,
            len,
        });
    }
    Ok(())
}

/// The indices `select` keeps of `axis`, and the axis left.
pub(crate) fn select_axis(
    name: &'static str,
    axis: &Axis,
    select: &Select,
) -> Result<(Range<usize>, Axis), OpsError> {
    match select {
        Select::All => Ok((0..axis.len(), axis.clone())),
        Select::Bins(r) => {
            check(name, r, axis.n_bins())?;
            Ok((r.clone(), axis.select(r.clone())))
        }
        Select::Total => {
            let t = axis.total_index().ok_or(OpsError::NoTotal { axis: name })?;
            Ok((t..t + 1, axis.collapsed()))
        }
    }
}
