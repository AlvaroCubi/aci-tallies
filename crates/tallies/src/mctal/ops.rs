//! Arithmetic, selection and integration of MCTAL [`Tally`]s, along any of
//! their eight axes. Errors propagate as for independent quantities (see
//! [`crate::common::values`]); summing an axis returns MCNP's Total bin
//! when it printed one.

use ndarray::{ArrayD, Axis as Dim, Slice};

use super::model::{Tally, DIMS, F};
use crate::common::select::select_axis;
pub use crate::common::select::{OpsError, Select};
use crate::common::values;

/// Relative tolerance for "the same bins".
const BOUNDS_TOL: f64 = 1e-9;

fn check_compatible(a: &Tally, b: &Tally) -> Result<(), OpsError> {
    if a.result.shape() != b.result.shape() {
        return Err(OpsError::ShapeMismatch(
            a.result.shape().to_vec(),
            b.result.shape().to_vec(),
        ));
    }
    for (k, name) in DIMS.iter().enumerate() {
        if !a.axes[k].matches(&b.axes[k], BOUNDS_TOL) {
            return Err(OpsError::BinningMismatch(name));
        }
    }
    Ok(())
}

/// `a`'s metadata with new values; no warnings, no fluctuation chart.
fn with_values(a: &Tally, result: ArrayD<f64>, rel_error: ArrayD<f64>) -> Tally {
    Tally {
        id: a.id,
        particles: a.particles.clone(),
        detector: a.detector,
        modifier: a.modifier,
        comment: a.comment.clone(),
        axes: a.axes.clone(),
        numbers: a.numbers.clone(),
        bins: a.bins.clone(),
        mesh: a.mesh.clone(),
        perturbation: a.perturbation,
        result,
        rel_error,
        tfc: None,
        warnings: Vec::new(),
    }
}

/// `a + b` per value. They must have the same bins.
pub fn add(a: &Tally, b: &Tally) -> Result<Tally, OpsError> {
    linear(a, b, 1.0)
}

/// `a - b` per value. They must have the same bins.
pub fn sub(a: &Tally, b: &Tally) -> Result<Tally, OpsError> {
    linear(a, b, -1.0)
}

fn linear(a: &Tally, b: &Tally, sign: f64) -> Result<Tally, OpsError> {
    check_compatible(a, b)?;
    let (r, e) = values::sum(
        (a.result.view(), a.rel_error.view()),
        (b.result.view(), b.rel_error.view()),
        sign,
    );
    Ok(with_values(a, r, e))
}

/// `a * b` per value. They must have the same bins.
pub fn mul(a: &Tally, b: &Tally) -> Result<Tally, OpsError> {
    product(a, b, false)
}

/// `a / b` per value (`0 / 0` is 0). They must have the same bins.
pub fn div(a: &Tally, b: &Tally) -> Result<Tally, OpsError> {
    product(a, b, true)
}

fn product(a: &Tally, b: &Tally, divide: bool) -> Result<Tally, OpsError> {
    check_compatible(a, b)?;
    let (r, e) = values::product(
        (a.result.view(), a.rel_error.view()),
        (b.result.view(), b.rel_error.view()),
        divide,
    );
    Ok(with_values(a, r, e))
}

/// Every value times `factor`, an exact constant: relative errors don't
/// change.
pub fn scale(a: &Tally, factor: f64) -> Tally {
    with_values(a, &a.result * factor, a.rel_error.clone())
}

/// Part of a tally: along each of [`DIMS`], what `select` keeps. Bins and
/// `numbers` are cut to match.
pub fn isel(a: &Tally, select: &[Select; 8]) -> Result<Tally, OpsError> {
    let mut ranges = Vec::with_capacity(8);
    let mut axes = a.axes.clone();
    for k in 0..8 {
        let (range, axis) = select_axis(DIMS[k], &a.axes[k], &select[k])?;
        ranges.push(range);
        axes[k] = axis;
    }
    let cut = |x: &ArrayD<f64>| {
        x.slice_each_axis(|ax| Slice::from(ranges[ax.axis.index()].clone()))
            .to_owned()
    };
    let mut out = with_values(a, cut(&a.result), cut(&a.rel_error));
    out.axes = axes;
    if !a.numbers.is_empty() && select[F] != Select::All {
        out.numbers = a
            .numbers
            .get(ranges[F].clone())
            .map(<[_]>::to_vec)
            .unwrap_or_default();
    }
    Ok(out)
}

/// Sum over axis `dim` (an index into [`DIMS`]), leaving one bin: MCNP's
/// Total bin when it printed one, the last bin of a cumulative axis, else
/// the bins added up, errors in quadrature.
pub fn integrate(a: &Tally, dim: usize) -> Tally {
    let axis = &a.axes[dim];
    let take = |x: &ArrayD<f64>, i: usize| x.slice_axis(Dim(dim), Slice::from(i..i + 1)).to_owned();
    let (r, e) = match (axis.total_index(), axis.cumulative) {
        (Some(t), _) => (take(&a.result, t), take(&a.rel_error, t)),
        (None, true) => {
            let last = axis.n_bins() - 1;
            (take(&a.result, last), take(&a.rel_error, last))
        }
        (None, false) => {
            let n = axis.n_bins();
            values::sum_along(
                a.result.slice_axis(Dim(dim), Slice::from(0..n)),
                a.rel_error.slice_axis(Dim(dim), Slice::from(0..n)),
                Dim(dim),
                |_| 1.0,
            )
        }
    };
    let mut out = with_values(a, r, e);
    out.axes[dim] = axis.collapsed();
    if dim == F {
        out.numbers.clear();
        out.bins.clear();
    }
    out
}
