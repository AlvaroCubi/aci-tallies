//! Arithmetic, selection and integration of [`Tally`]s.
//!
//! Errors propagate as for independent quantities (see
//! [`crate::common::values`]). Where MCNP printed a Total bin, summing over
//! that axis returns it rather than recomputing it: MCNP derives its error
//! from the histories, which accounts for the correlation between bins that
//! a quadrature sum can't.

use std::ops::Range;

use ndarray::{s, Array3, Array5, ArrayView3, Axis as Dim, Ix5};
use rayon::prelude::*;

use super::model::{Geometry, Tally};
use crate::common::axis::close;
use crate::common::select::{check, select_axis};
pub use crate::common::select::{OpsError, Select};
use crate::common::values;

/// Voxel volumes from `geometry`, shape `[i, j, k]`.
pub fn voxel_volumes(geometry: &Geometry) -> Array3<f64> {
    // aci-tallies-geom numbers voxels with k fastest: ndarray's standard
    // layout for [i, j, k].
    Array3::from_shape_vec(geometry.dims(), geometry.voxel_volumes()).expect("one volume per voxel")
}

/// Relative tolerance for "the same bins".
const BOUNDS_TOL: f64 = 1e-9;

fn same_mesh(a: &Geometry, b: &Geometry) -> bool {
    let (a1, a2, a3) = a.axes();
    let (b1, b2, b3) = b.axes();
    std::mem::discriminant(a) == std::mem::discriminant(b)
        && close(a1, b1, BOUNDS_TOL)
        && close(a2, b2, BOUNDS_TOL)
        && close(a3, b3, BOUNDS_TOL)
}

fn check_compatible(a: &Tally, b: &Tally) -> Result<(), OpsError> {
    if a.result.shape() != b.result.shape() {
        return Err(OpsError::ShapeMismatch(
            a.result.shape().to_vec(),
            b.result.shape().to_vec(),
        ));
    }
    if !a.energy.matches(&b.energy, BOUNDS_TOL) {
        return Err(OpsError::BinningMismatch("energy"));
    }
    if !a.time.matches(&b.time, BOUNDS_TOL) {
        return Err(OpsError::BinningMismatch("time"));
    }
    if !same_mesh(&a.geometry, &b.geometry) {
        return Err(OpsError::BinningMismatch("spatial"));
    }
    Ok(())
}

/// `a`'s metadata with new values; no warnings (they were about reading).
fn with_values(
    a: &Tally,
    result: Array5<f64>,
    rel_error: Array5<f64>,
    volume: Option<Array3<f64>>,
) -> Tally {
    Tally {
        id: a.id,
        particle: a.particle.clone(),
        comment: a.comment.clone(),
        geometry: a.geometry.clone(),
        geometry_from_input: a.geometry_from_input,
        energy_weighted: a.energy_weighted,
        energy: a.energy.clone(),
        time: a.time.clone(),
        out_format: a.out_format,
        result,
        rel_error,
        volume,
        warnings: Vec::new(),
    }
}

/// `a + b` per value. They must have the same bins and mesh.
pub fn add(a: &Tally, b: &Tally) -> Result<Tally, OpsError> {
    linear(a, b, 1.0)
}

/// `a - b` per value. They must have the same bins and mesh.
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
    Ok(with_values(a, r, e, a.volume.clone()))
}

/// `a * b` per value. They must have the same bins and mesh.
pub fn mul(a: &Tally, b: &Tally) -> Result<Tally, OpsError> {
    product(a, b, false)
}

/// `a / b` per value (`0 / 0` is 0). They must have the same bins and mesh.
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
    Ok(with_values(a, r, e, a.volume.clone()))
}

/// Every value times `factor`, an exact constant: relative errors don't
/// change. (`a + a` is not `2 a`: it treats the two as independent.)
pub fn scale(a: &Tally, factor: f64) -> Tally {
    with_values(a, &a.result * factor, a.rel_error.clone(), a.volume.clone())
}

/// Sum over the energy bins: MCNP's Total bin when it printed one, else
/// the bins added up.
pub fn integrate_energy(a: &Tally) -> Tally {
    integrate(a, 0)
}

/// Sum over the time bins: MCNP's Total bin when it printed one, else the
/// bins added up.
pub fn integrate_time(a: &Tally) -> Tally {
    integrate(a, 1)
}

fn integrate(a: &Tally, dim: usize) -> Tally {
    let axis = if dim == 0 { &a.energy } else { &a.time };
    let (r, e) = match axis.total_index() {
        Some(t) => (
            take(&a.result, dim, t..t + 1),
            take(&a.rel_error, dim, t..t + 1),
        ),
        None => {
            let n = axis.n_bins();
            values::sum_along(
                a.result.slice_axis(Dim(dim), (0..n).into()),
                a.rel_error.slice_axis(Dim(dim), (0..n).into()),
                Dim(dim),
                |_| 1.0,
            )
        }
    };
    let mut out = with_values(a, r, e, a.volume.clone());
    let collapsed = axis.collapsed();
    if dim == 0 {
        out.energy = collapsed;
    } else {
        out.time = collapsed;
    }
    out
}

fn take(a: &Array5<f64>, dim: usize, range: Range<usize>) -> Array5<f64> {
    a.slice_axis(Dim(dim), range.into()).to_owned()
}

/// Integrate over the mesh: `sum(result * voxel volume)` for every energy
/// and time bin (Total ones included), leaving one voxel spanning the mesh.
/// Volumes are the file's (`out=cf`) or else [`voxel_volumes`]; they are
/// taken as exact and the voxels as independent.
pub fn integrate_space(a: &Tally) -> Tally {
    let volume = a
        .volume
        .clone()
        .unwrap_or_else(|| voxel_volumes(&a.geometry));
    let weights: Vec<f64> = volume.iter().copied().collect();
    let [ne, nt, ni, nj, nk] = shape5(&a.result);
    // Copied only if not already in standard layout (never, as read).
    let (values, errors) = (
        a.result.as_standard_layout(),
        a.rel_error.as_standard_layout(),
    );
    let shape = (ne * nt, ni * nj * nk);
    let (values, errors) = (rows(&values, shape), rows(&errors, shape));
    let sums: Vec<(f64, f64)> = (0..ne * nt)
        .into_par_iter()
        .map(|n| {
            let (v, e) = (values.row(n), errors.row(n));
            let (mut total, mut variance) = (0.0, 0.0);
            for ((&v, &e), &w) in v.iter().zip(e.iter()).zip(&weights) {
                total += w * v;
                variance += (w * v * e).powi(2);
            }
            (total, values::relative(variance.sqrt(), total))
        })
        .collect();
    let five = |f: fn(&(f64, f64)) -> f64| {
        Array5::from_shape_vec((ne, nt, 1, 1, 1), sums.iter().map(f).collect())
            .expect("one value per bin")
    };
    let total = Array3::from_elem((1, 1, 1), weights.iter().sum::<f64>());
    let mut out = with_values(a, five(|s| s.0), five(|s| s.1), Some(total));
    out.geometry = select_mesh(&a.geometry, None, None, None, true);
    out
}

/// `x` (in standard layout) as rows of `shape`.
fn rows<'a>(
    x: &'a ndarray::CowArray<'_, f64, Ix5>,
    shape: (usize, usize),
) -> ndarray::ArrayView2<'a, f64> {
    x.view()
        .into_shape_with_order(shape)
        .expect("standard layout")
}

fn shape5(a: &Array5<f64>) -> [usize; 5] {
    let d = a.dim();
    [d.0, d.1, d.2, d.3, d.4]
}

/// The mesh with bins `i`, `j`, `k` of its axes (all when `None`), or
/// collapsed to one voxel spanning it.
fn select_mesh(
    g: &Geometry,
    i: Option<&Range<usize>>,
    j: Option<&Range<usize>>,
    k: Option<&Range<usize>>,
    collapse: bool,
) -> Geometry {
    let cut = |b: &[f64], r: Option<&Range<usize>>| -> Vec<f64> {
        if collapse {
            vec![b[0], b[b.len() - 1]]
        } else {
            match r {
                Some(r) => b[r.start..=r.end].to_vec(),
                None => b.to_vec(),
            }
        }
    };
    match g {
        Geometry::Rectangular {
            x_bounds,
            y_bounds,
            z_bounds,
            transform,
        } => Geometry::Rectangular {
            x_bounds: cut(x_bounds, i),
            y_bounds: cut(y_bounds, j),
            z_bounds: cut(z_bounds, k),
            transform: *transform,
        },
        Geometry::Cylindrical {
            r_bounds,
            z_bounds,
            theta_bounds,
            origin,
            axis,
            vec,
        } => Geometry::Cylindrical {
            r_bounds: cut(r_bounds, i),
            z_bounds: cut(z_bounds, j),
            theta_bounds: cut(theta_bounds, k),
            origin: *origin,
            axis: *axis,
            vec: *vec,
        },
    }
}

/// Part of a tally: energy and time bins as `energy` and `time` select,
/// and index ranges `i`, `j`, `k` of the mesh (all when `None`), with its
/// bins and mesh cut to match.
pub fn isel(
    a: &Tally,
    energy: &Select,
    time: &Select,
    i: Option<Range<usize>>,
    j: Option<Range<usize>>,
    k: Option<Range<usize>>,
) -> Result<Tally, OpsError> {
    let [_, _, ni, nj, nk] = shape5(&a.result);
    let (er, energy_axis) = select_axis("energy", &a.energy, energy)?;
    let (tr, time_axis) = select_axis("time", &a.time, time)?;
    for (name, r, n) in [("i", &i, ni), ("j", &j, nj), ("k", &k, nk)] {
        if let Some(r) = r {
            check(name, r, n)?;
        }
    }
    let ir = i.clone().unwrap_or(0..ni);
    let jr = j.clone().unwrap_or(0..nj);
    let kr = k.clone().unwrap_or(0..nk);
    let cut = |x: &Array5<f64>| {
        x.slice(s![
            er.clone(),
            tr.clone(),
            ir.clone(),
            jr.clone(),
            kr.clone()
        ])
        .to_owned()
        .into_dimensionality::<Ix5>()
        .expect("rank 5")
    };
    let volume = a.volume.as_ref().map(|v: &Array3<f64>| {
        let v: ArrayView3<f64> = v.slice(s![ir.clone(), jr.clone(), kr.clone()]);
        v.to_owned()
    });
    let mut out = with_values(a, cut(&a.result), cut(&a.rel_error), volume);
    out.energy = energy_axis;
    out.time = time_axis;
    out.geometry = select_mesh(&a.geometry, i.as_ref(), j.as_ref(), k.as_ref(), false);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Axis;
    use crate::meshtal::OutFormat;
    use approx::assert_relative_eq;

    /// An `[ne, nt, 2, 1, 1]` tally on unit voxels (x from 0 to 2), values
    /// `f(e, t, i)` with relative error `err`.
    fn tally(energy: Axis, time: Axis, f: impl Fn(usize, usize, usize) -> f64, err: f64) -> Tally {
        let shape = (energy.len(), time.len(), 2, 1, 1);
        Tally {
            id: 1,
            particle: "neutron".into(),
            comment: None,
            geometry: Geometry::Rectangular {
                x_bounds: vec![0.0, 1.0, 2.0],
                y_bounds: vec![0.0, 1.0],
                z_bounds: vec![0.0, 1.0],
                transform: None,
            },
            geometry_from_input: false,
            energy_weighted: None,
            energy,
            time,
            out_format: OutFormat::Col,
            result: Array5::from_shape_fn(shape, |(e, t, i, _, _)| f(e, t, i)),
            rel_error: Array5::from_elem(shape, err),
            volume: None,
            warnings: Vec::new(),
        }
    }

    fn one() -> Axis {
        Axis::mcnp(vec![0.0, 1e36])
    }

    #[test]
    fn arithmetic() {
        let a = tally(one(), one(), |_, _, i| [10.0, 0.0][i], 0.1);
        let b = tally(one(), one(), |_, _, i| [5.0, 0.0][i], 0.2);
        let sum = add(&a, &b).unwrap();
        assert_relative_eq!(sum.result[[0, 0, 0, 0, 0]], 15.0);
        assert_relative_eq!(sum.rel_error[[0, 0, 0, 0, 0]], 1.0f64.hypot(1.0) / 15.0);
        assert_eq!(sum.rel_error[[0, 0, 1, 0, 0]], 0.0);
        let q = div(&mul(&a, &b).unwrap(), &b).unwrap();
        assert_relative_eq!(q.result[[0, 0, 0, 0, 0]], 10.0);
        assert_eq!(q.result[[0, 0, 1, 0, 0]], 0.0);
        let s = scale(&a, 2.0);
        assert_eq!(
            (s.result[[0, 0, 0, 0, 0]], s.rel_error[[0, 0, 0, 0, 0]]),
            (20.0, 0.1)
        );
        // Adding a tally to itself is not doubling it.
        assert!(add(&a, &a).unwrap().rel_error[[0, 0, 0, 0, 0]] < 0.1);
        let c = tally(Axis::mcnp(vec![0.0, 1.0, 2.0]), one(), |_, _, _| 1.0, 0.1);
        assert_eq!(
            add(&a, &c).unwrap_err(),
            OpsError::ShapeMismatch(vec![1, 1, 2, 1, 1], vec![3, 1, 2, 1, 1])
        );
        let d = tally(Axis::mcnp(vec![0.0, 2e36]), one(), |_, _, _| 1.0, 0.1);
        assert_eq!(
            add(&a, &d).unwrap_err(),
            OpsError::BinningMismatch("energy")
        );
    }

    #[test]
    fn integrating_energy_and_time() {
        // Two energy bins and MCNP's Total (whose value differs from the
        // sum, to tell which was used).
        let e = Axis::mcnp(vec![0.0, 1.0, 2.0]);
        let a = tally(
            e.clone(),
            one(),
            |e, _, i| [[1.0, 2.0], [3.0, 4.0], [9.5, 9.5]][e][i],
            0.1,
        );
        let t = integrate_energy(&a);
        assert_eq!(t.result.dim(), (1, 1, 2, 1, 1));
        assert_eq!(t.result[[0, 0, 0, 0, 0]], 9.5);
        assert_eq!(t.energy, Axis::bounded(vec![0.0, 2.0], false));
        // Without a Total: the sum, errors in quadrature.
        let a = isel(&a, &Select::Bins(0..2), &Select::All, None, None, None).unwrap();
        let t = integrate_energy(&a);
        assert_eq!(t.result[[0, 0, 0, 0, 0]], 4.0);
        assert_relative_eq!(t.rel_error[[0, 0, 0, 0, 0]], 0.1f64.hypot(0.3) / 4.0);
        let ti = tally(
            one(),
            Axis::mcnp(vec![0.0, 1.0, 2.0]),
            |_, t, _| [1.0, 2.0, 7.0][t],
            0.1,
        );
        assert_eq!(integrate_time(&ti).result[[0, 0, 1, 0, 0]], 7.0);
    }

    #[test]
    fn integrating_space() {
        let a = tally(one(), one(), |_, _, i| [2.0, 3.0][i], 0.1);
        let t = integrate_space(&a);
        assert_eq!(t.result.dim(), (1, 1, 1, 1, 1));
        assert_eq!(t.result[[0, 0, 0, 0, 0]], 5.0);
        assert_relative_eq!(t.rel_error[[0, 0, 0, 0, 0]], 0.2f64.hypot(0.3) / 5.0);
        assert_eq!(t.volume.unwrap()[[0, 0, 0]], 2.0);
        assert_eq!(t.geometry.dims(), (1, 1, 1));
    }

    #[test]
    fn selecting() {
        let e = Axis::mcnp(vec![0.0, 1.0, 2.0]);
        let a = tally(e, one(), |e, _, i| (10 * e + i) as f64, 0.1);
        let s = isel(&a, &Select::All, &Select::All, Some(1..2), None, None).unwrap();
        assert_eq!(s.result.dim(), (3, 1, 1, 1, 1));
        assert_eq!(s.geometry.axes().0, &[1.0, 2.0]);
        assert!(s.energy.total);
        let s = isel(&a, &Select::Bins(1..2), &Select::All, None, None, None).unwrap();
        assert_eq!(s.energy, Axis::bounded(vec![1.0, 2.0], false));
        assert_eq!(s.result[[0, 0, 1, 0, 0]], 11.0);
        let s = isel(&a, &Select::Total, &Select::All, None, None, None).unwrap();
        assert_eq!(s.result[[0, 0, 1, 0, 0]], 21.0);
        assert_eq!(s.energy, Axis::bounded(vec![0.0, 2.0], false));
        let err = |r| isel(&a, &r, &Select::All, None, None, None).unwrap_err();
        assert_eq!(
            err(Select::Bins(2..3)),
            OpsError::OutOfBounds {
                axis: "energy",
                start: 2,
                end: 3,
                len: 2
            }
        );
        assert_eq!(err(Select::Bins(1..1)), OpsError::Empty { axis: "energy" });
        assert_eq!(
            isel(&a, &Select::All, &Select::Total, None, None, None).unwrap_err(),
            OpsError::NoTotal { axis: "time" }
        );
    }

    #[test]
    fn volumes() {
        let g = Geometry::Rectangular {
            x_bounds: vec![0.0, 2.0, 5.0],
            y_bounds: vec![0.0, 3.0],
            z_bounds: vec![0.0, 1.0, 4.0],
            transform: None,
        };
        let v = voxel_volumes(&g);
        assert_eq!(v.dim(), (2, 1, 2));
        assert_relative_eq!(v[[1, 0, 1]], 27.0);
    }
}
