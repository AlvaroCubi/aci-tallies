//! Arithmetic on tally values with their relative errors.
//!
//! Errors combine as for independent quantities: absolute errors in
//! quadrature for sums, relative errors in quadrature for products and
//! quotients. Results of different runs, or of different tallies of one
//! run, are taken as independent, the usual assumption without their
//! covariance. A zero value with a zero error has relative error 0, as MCNP
//! prints it; a zero value with a nonzero error has an infinite one.

use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis, Zip};

/// Relative error of a value from its absolute error.
pub(crate) fn relative(abs: f64, value: f64) -> f64 {
    if abs == 0.0 {
        0.0
    } else {
        (abs / value).abs()
    }
}

/// `a + sign * b`.
pub(crate) fn sum<D: Dimension>(
    a: (ArrayView<f64, D>, ArrayView<f64, D>),
    b: (ArrayView<f64, D>, ArrayView<f64, D>),
    sign: f64,
) -> (Array<f64, D>, Array<f64, D>) {
    let mut result = Array::zeros(a.0.raw_dim());
    let mut error = Array::zeros(a.0.raw_dim());
    Zip::from(&mut result)
        .and(&mut error)
        .and(&a.0)
        .and(&a.1)
        .and(&b.0)
        .and(&b.1)
        .par_for_each(|r, e, &va, &ea, &vb, &eb| {
            *r = va + sign * vb;
            *e = relative((va * ea).hypot(vb * eb), *r);
        });
    (result, error)
}

/// `a * b`, or `a / b` (`0 / 0` is 0, as for a voxel neither scored in).
pub(crate) fn product<D: Dimension>(
    a: (ArrayView<f64, D>, ArrayView<f64, D>),
    b: (ArrayView<f64, D>, ArrayView<f64, D>),
    divide: bool,
) -> (Array<f64, D>, Array<f64, D>) {
    let mut result = Array::zeros(a.0.raw_dim());
    let mut error = Array::zeros(a.0.raw_dim());
    Zip::from(&mut result)
        .and(&mut error)
        .and(&a.0)
        .and(&a.1)
        .and(&b.0)
        .and(&b.1)
        .par_for_each(|r, e, &va, &ea, &vb, &eb| {
            *r = match divide {
                false => va * vb,
                true if va == 0.0 && vb == 0.0 => 0.0,
                true => va / vb,
            };
            *e = ea.hypot(eb);
        });
    (result, error)
}

/// The sum over `axis` of `values`, each weighted by `weight(index)`
/// (exact), keeping `axis` with length 1.
pub(crate) fn sum_along<D: RemoveAxis>(
    values: ArrayView<f64, D>,
    errors: ArrayView<f64, D>,
    axis: Axis,
    weight: impl Fn(usize) -> f64,
) -> (Array<f64, D>, Array<f64, D>) {
    let dim = values.raw_dim().remove_axis(axis);
    let mut total = Array::<f64, _>::zeros(dim.clone());
    let mut variance = Array::<f64, _>::zeros(dim);
    for (n, (v, e)) in values
        .axis_iter(axis)
        .zip(errors.axis_iter(axis))
        .enumerate()
    {
        let w = weight(n);
        Zip::from(&mut total)
            .and(&mut variance)
            .and(&v)
            .and(&e)
            .for_each(|t, var, &v, &e| {
                *t += w * v;
                *var += (w * v * e).powi(2);
            });
    }
    let errors = Zip::from(&total)
        .and(&variance)
        .map_collect(|&t, &var| relative(var.sqrt(), t));
    let keep = |a: Array<f64, D::Smaller>| {
        a.insert_axis(axis)
            .into_dimensionality::<D>()
            .expect("the same rank as before")
    };
    (keep(total), keep(errors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn quadrature() {
        let (a, ea) = (array![3.0, 0.0, 0.0, 2.0], array![0.1, 0.0, 0.0, 0.5]);
        let (b, eb) = (array![4.0, 0.0, 1.0, -2.0], array![0.2, 0.0, 0.5, 0.5]);
        let (r, e) = sum((a.view(), ea.view()), (b.view(), eb.view()), 1.0);
        assert_eq!(r, array![7.0, 0.0, 1.0, 0.0]);
        assert!((e[0] - (0.3f64.hypot(0.8) / 7.0)).abs() < 1e-15);
        assert_eq!(e[1], 0.0);
        assert_eq!(e[2], 0.5);
        assert_eq!(e[3], f64::INFINITY);
        let (r, e) = product((a.view(), ea.view()), (b.view(), eb.view()), true);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 0.0);
        assert_eq!(r[3], -1.0);
        assert!((e[0] - 0.1f64.hypot(0.2)).abs() < 1e-15);
        let (s, se) = sum_along(a.view(), ea.view(), Axis(0), |n| [1.0, 1.0, 1.0, 2.0][n]);
        assert_eq!(s, array![7.0]);
        assert!((se[0] - (0.3f64.hypot(2.0) / 7.0)).abs() < 1e-15);
    }
}
