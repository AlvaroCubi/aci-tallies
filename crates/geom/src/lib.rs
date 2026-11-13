//! Geometry of MCNP-style structured meshes, independent of what is scored
//! on them.
//!
//! The same mesh definition shows up in several MCNP outputs — FMESH
//! meshtal files, D1SUNED `meshinfo`/`cuvmsh` cell-under-voxel files — and
//! in tools built on top of them. This crate holds what they share: the
//! [`Geometry`] itself, the flat voxel numbering, voxel volumes and centres,
//! point location, VTK export (`vtk`, behind the default `vtk` feature) and
//! reading the exact mesh definition from an MCNP input (`mcnp_input`,
//! behind the `mcnp-input` feature).
//!
//! # Placement
//!
//! A mesh is defined in its own local frame and placed in the world by MCNP's
//! `TR=` keyword, a rotation plus a translation ([`Transform`]). A
//! rectangular mesh keeps it in its `transform` field; a cylindrical one
//! needs no extra field, since a rotated and translated cylinder is still a
//! cylinder: [`Geometry::transformed`] moves its `origin`, `axis` and `vec`.
//!
//! # Voxel numbering
//!
//! Voxels are numbered in `(outer, middle, inner)` = `(i, j, k)` order with
//! `k` varying fastest: flat index `(i * n_j + j) * n_k + k`. For a
//! rectangular mesh that is `(x, y, z)`, for a cylindrical one `(r, z,
//! theta)`. This is the order MCNP writes mesh tallies in and the order
//! D1SUNED numbers voxels in (`meshinfo` index = flat index + 1).
//!
//! # Validity
//!
//! [`Geometry`]'s fields are public and nothing stops a caller from building
//! one with empty, unsorted or non-finite bounds, or a degenerate cylinder
//! frame. Methods don't panic on such input but return meaningless numbers;
//! call [`Geometry::validate`] on any geometry that didn't come from a
//! trusted source.

use std::fmt;

#[cfg(feature = "mcnp-input")]
pub mod mcnp_input;
#[cfg(feature = "vtk")]
pub mod vtk;

/// The spatial mesh geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Rectangular {
        /// Bin boundaries along X, length = n_x + 1.
        x_bounds: Vec<f64>,
        /// Bin boundaries along Y, length = n_y + 1.
        y_bounds: Vec<f64>,
        /// Bin boundaries along Z, length = n_z + 1.
        z_bounds: Vec<f64>,
        /// Placement of the mesh's local X/Y/Z in the world; `None` when
        /// they are the world axes.
        transform: Option<Transform>,
    },
    Cylindrical {
        /// Bin boundaries along R, length = n_r + 1.
        r_bounds: Vec<f64>,
        /// Bin boundaries along the cylinder's Z axis, length = n_z + 1.
        z_bounds: Vec<f64>,
        /// Bin boundaries along Theta in revolutions, length = n_theta + 1.
        theta_bounds: Vec<f64>,
        origin: [f64; 3],
        axis: [f64; 3],
        vec: [f64; 3],
    },
}

impl Geometry {
    /// Number of voxels along each of the three spatial axes, in
    /// `(outer, middle, inner)` order. An axis with fewer than two bounds
    /// counts as zero voxels.
    pub fn dims(&self) -> (usize, usize, usize) {
        let (a, b, c) = self.axes();
        let n = |v: &[f64]| v.len().saturating_sub(1);
        (n(a), n(b), n(c))
    }

    /// Check that the geometry describes a real mesh: every axis has at
    /// least one bin, bounds are finite and strictly increasing, and for a
    /// cylinder radii are non-negative, theta lies within one revolution
    /// `[0, 1]`, and `axis`/`vec` define a frame (non-zero, not parallel).
    ///
    /// Zero-width bins are rejected: besides being meaningless, they are
    /// what a meshtal header produces when bins are finer than the two
    /// decimals MCNP prints bounds with.
    pub fn validate(&self) -> Result<(), GeometryError> {
        let names = match self {
            Geometry::Rectangular { .. } => ["x", "y", "z"],
            Geometry::Cylindrical { .. } => ["r", "z", "theta"],
        };
        let (a, b, c) = self.axes();
        for (axis, bounds) in names.into_iter().zip([a, b, c]) {
            if bounds.len() < 2 {
                return Err(GeometryError::TooFewBounds { axis });
            }
            if let Some(index) = bounds.iter().position(|v| !v.is_finite()) {
                return Err(GeometryError::NonFinite { axis, index });
            }
            if let Some(bin) = bounds.windows(2).position(|w| w[1] <= w[0]) {
                return Err(GeometryError::NotIncreasing {
                    axis,
                    bin,
                    lo: bounds[bin],
                    hi: bounds[bin + 1],
                });
            }
        }
        if let Geometry::Rectangular {
            transform: Some(t), ..
        } = self
        {
            t.validate()?;
        }
        if let Geometry::Cylindrical {
            origin, axis, vec, ..
        } = self
        {
            if a[0] < 0.0 {
                return Err(GeometryError::NegativeRadius(a[0]));
            }
            if c[0] < 0.0 || c[c.len() - 1] > 1.0 {
                return Err(GeometryError::ThetaOutOfRange(c[0], c[c.len() - 1]));
            }
            let finite = |v: &[f64; 3]| v.iter().all(|x| x.is_finite());
            if !(finite(origin) && finite(axis) && finite(vec)) {
                return Err(GeometryError::BadFrame("non-finite origin, axis or vec"));
            }
            let (na, nv) = (dot(*axis, *axis).sqrt(), dot(*vec, *vec).sqrt());
            if na == 0.0 || nv == 0.0 {
                return Err(GeometryError::BadFrame("zero-length axis or vec"));
            }
            let c = cross(*axis, *vec);
            if dot(c, c).sqrt() <= 1e-9 * na * nv {
                return Err(GeometryError::BadFrame("vec is parallel to axis"));
            }
        }
        Ok(())
    }

    /// Total number of voxels.
    pub fn n_voxels(&self) -> usize {
        let (a, b, c) = self.dims();
        a * b * c
    }

    /// The three bound vectors in `(outer, middle, inner)` order.
    pub fn axes(&self) -> (&[f64], &[f64], &[f64]) {
        match self {
            Geometry::Rectangular {
                x_bounds,
                y_bounds,
                z_bounds,
                ..
            } => (x_bounds, y_bounds, z_bounds),
            Geometry::Cylindrical {
                r_bounds,
                z_bounds,
                theta_bounds,
                ..
            } => (r_bounds, z_bounds, theta_bounds),
        }
    }

    /// Flat voxel index of `(i, j, k)`.
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        let (_, nj, nk) = self.dims();
        (i * nj + j) * nk + k
    }

    /// `(i, j, k)` of a flat voxel index.
    pub fn ijk(&self, index: usize) -> (usize, usize, usize) {
        let (_, nj, nk) = self.dims();
        (index / (nj * nk), (index / nk) % nj, index % nk)
    }

    /// Volume of one voxel.
    pub fn voxel_volume(&self, index: usize) -> f64 {
        let (i, j, k) = self.ijk(index);
        let (a, b, c) = self.axes();
        match self {
            Geometry::Rectangular { .. } => {
                (a[i + 1] - a[i]) * (b[j + 1] - b[j]) * (c[k + 1] - c[k])
            }
            // Annular sector: pi*(r1^2 - r0^2) is the full-revolution annulus
            // area; theta is already a fraction of one revolution.
            Geometry::Cylindrical { .. } => {
                std::f64::consts::PI
                    * (a[i + 1] * a[i + 1] - a[i] * a[i])
                    * (b[j + 1] - b[j])
                    * (c[k + 1] - c[k])
            }
        }
    }

    /// Volume of every voxel, in flat index order.
    pub fn voxel_volumes(&self) -> Vec<f64> {
        (0..self.n_voxels()).map(|n| self.voxel_volume(n)).collect()
    }

    /// World-space centre of one voxel: the midpoint of its bins along each
    /// axis (for a cylinder, mid-radius, mid-height and mid-angle — the same
    /// point MCNP prints as a voxel's coordinates).
    pub fn voxel_center(&self, index: usize) -> [f64; 3] {
        let (i, j, k) = self.ijk(index);
        let (a, b, c) = self.axes();
        let mid = |v: &[f64], n: usize| 0.5 * (v[n] + v[n + 1]);
        match self {
            Geometry::Rectangular { transform, .. } => {
                let local = [mid(a, i), mid(b, j), mid(c, k)];
                transform.map_or(local, |t| t.to_world(local))
            }
            Geometry::Cylindrical { .. } => self
                .cyl_frame()
                .expect("cylindrical geometry has a frame")
                .to_world(mid(a, i), mid(b, j), mid(c, k)),
        }
    }

    /// World-space centre of every voxel, in flat index order.
    pub fn voxel_centers(&self) -> Vec<[f64; 3]> {
        (0..self.n_voxels()).map(|n| self.voxel_center(n)).collect()
    }

    /// Flat index of the voxel containing `point`, or `None` outside the
    /// mesh. Bins are half-open `[lo, hi)` except the last one along each
    /// axis, which includes its upper bound.
    pub fn locate(&self, point: [f64; 3]) -> Option<usize> {
        let local = match self {
            Geometry::Rectangular { transform, .. } => {
                transform.map_or(point, |t| t.to_local(point))
            }
            Geometry::Cylindrical { .. } => {
                let frame = self.cyl_frame().expect("cylindrical geometry has a frame");
                let (r, z, theta) = frame.to_local(point);
                [r, z, theta]
            }
        };
        let (a, b, c) = self.axes();
        let i = find_bin(a, local[0])?;
        let j = find_bin(b, local[1])?;
        let k = match self {
            Geometry::Rectangular { .. } => find_bin(c, local[2])?,
            // Theta wraps: a point at angle t is also at t + 1 revolution.
            Geometry::Cylindrical { .. } => {
                find_bin(c, local[2]).or_else(|| find_bin(c, local[2] + 1.0))?
            }
        };
        Some(self.index(i, j, k))
    }

    /// The cylinder's frame, or `None` for a rectangular mesh.
    pub fn cyl_frame(&self) -> Option<CylFrame> {
        match self {
            Geometry::Rectangular { .. } => None,
            Geometry::Cylindrical {
                origin, axis, vec, ..
            } => Some(CylFrame::new(*origin, *axis, *vec)),
        }
    }

    /// The same mesh moved by `t` (applied after any placement it already
    /// has). A rectangular mesh composes `t` into its `transform`; a
    /// cylindrical one gets `t` applied to its `origin`, `axis` and `vec`,
    /// which can't describe a mirrored cylinder: a reflection of one is an
    /// error.
    pub fn transformed(&self, t: &Transform) -> Result<Geometry, GeometryError> {
        let mut out = self.clone();
        match &mut out {
            Geometry::Rectangular { transform, .. } => {
                *transform = Some(match transform {
                    Some(inner) => inner.then(t),
                    None => *t,
                });
            }
            Geometry::Cylindrical {
                origin, axis, vec, ..
            } => {
                if t.is_reflection() {
                    return Err(GeometryError::BadTransform(
                        "a reflection reverses the theta direction of a cylindrical mesh".into(),
                    ));
                }
                *origin = t.to_world(*origin);
                *axis = t.rotate(*axis);
                *vec = t.rotate(*vec);
            }
        }
        Ok(out)
    }
}

/// A rigid placement: `world = rotation · local + translation`.
///
/// `rotation` is row-major and orthonormal: its column `c` is local axis `c`
/// expressed in world coordinates. Its determinant may be -1 (a
/// reflection), which MCNP accepts on a `TR=`. With the `mcnp-input`
/// feature, one converts from the `migjorn::Affine` of a `TRn` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

/// How far from orthonormal [`Transform::validate`] lets a rotation be.
const ROTATION_TOL: f64 = 1e-6;

impl Transform {
    pub const IDENTITY: Transform = Transform {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    /// World point of a local point.
    pub fn to_world(&self, local: [f64; 3]) -> [f64; 3] {
        add(self.rotate(local), self.translation)
    }

    /// Local point of a world point.
    pub fn to_local(&self, world: [f64; 3]) -> [f64; 3] {
        let d = sub(world, self.translation);
        let r = &self.rotation;
        [
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
        ]
    }

    /// A local direction in world coordinates (rotation only).
    pub fn rotate(&self, v: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
    }

    /// `self` followed by `outer`: `outer.to_world(self.to_world(p))`.
    pub fn then(&self, outer: &Transform) -> Transform {
        let mut rotation = [[0.0; 3]; 3];
        for (row, out_row) in rotation.iter_mut().enumerate() {
            for (col, value) in out_row.iter_mut().enumerate() {
                *value = (0..3)
                    .map(|n| outer.rotation[row][n] * self.rotation[n][col])
                    .sum();
            }
        }
        Transform {
            rotation,
            translation: outer.to_world(self.translation),
        }
    }

    /// Whether the rotation is a reflection (determinant -1).
    pub fn is_reflection(&self) -> bool {
        det(&self.rotation) < 0.0
    }

    /// Check that the values are finite and `rotation` is orthonormal.
    pub fn validate(&self) -> Result<(), GeometryError> {
        let finite = self.translation.iter().all(|v| v.is_finite())
            && self.rotation.iter().flatten().all(|v| v.is_finite());
        if !finite {
            return Err(GeometryError::BadTransform("non-finite values".into()));
        }
        if orthonormality_error(&self.rotation) > ROTATION_TOL {
            return Err(GeometryError::BadTransform(
                "rotation is not orthonormal".into(),
            ));
        }
        Ok(())
    }
}

/// Same convention: column `k` of the rotation is auxiliary axis `k`.
#[cfg(feature = "mcnp-input")]
impl From<migjorn::Affine> for Transform {
    fn from(a: migjorn::Affine) -> Transform {
        Transform {
            rotation: a.rotation,
            translation: a.translation,
        }
    }
}

/// Largest entry of `RᵀR - I`.
fn orthonormality_error(r: &[[f64; 3]; 3]) -> f64 {
    let col = |c: usize| [r[0][c], r[1][c], r[2][c]];
    let mut worst: f64 = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            let expected = if a == b { 1.0 } else { 0.0 };
            worst = worst.max((dot(col(a), col(b)) - expected).abs());
        }
    }
    worst
}

fn det(r: &[[f64; 3]; 3]) -> f64 {
    dot(r[0], cross(r[1], r[2]))
}

/// Why [`Geometry::validate`] rejected a geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    /// An axis has fewer than two bounds (no bins).
    TooFewBounds { axis: &'static str },
    /// A bound is NaN or infinite.
    NonFinite { axis: &'static str, index: usize },
    /// Bin `bin` has zero or negative width.
    NotIncreasing {
        axis: &'static str,
        bin: usize,
        lo: f64,
        hi: f64,
    },
    /// A cylinder's first radial bound is negative.
    NegativeRadius(f64),
    /// A cylinder's theta bounds (first, last) leave `[0, 1]` revolutions.
    ThetaOutOfRange(f64, f64),
    /// A cylinder's origin/axis/vec don't define a frame.
    BadFrame(&'static str),
    /// A [`Transform`] isn't finite and orthonormal, or can't move the mesh.
    BadTransform(String),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometryError::TooFewBounds { axis } => {
                write!(f, "{axis} axis needs at least two bounds")
            }
            GeometryError::NonFinite { axis, index } => {
                write!(f, "{axis} bound {index} is not finite")
            }
            GeometryError::NotIncreasing { axis, bin, lo, hi } => {
                write!(f, "{axis} bin {bin} has non-positive width ({lo} to {hi})")
            }
            GeometryError::NegativeRadius(r) => write!(f, "negative radius bound {r}"),
            GeometryError::ThetaOutOfRange(lo, hi) => {
                write!(f, "theta bounds {lo} to {hi} leave [0, 1] revolutions")
            }
            GeometryError::BadFrame(why) => write!(f, "bad cylinder frame: {why}"),
            GeometryError::BadTransform(why) => write!(f, "bad transform: {why}"),
        }
    }
}

impl std::error::Error for GeometryError {}

/// Bin of `value` in sorted `bounds`, or `None` outside them.
fn find_bin(bounds: &[f64], value: f64) -> Option<usize> {
    let n = bounds.len().checked_sub(1)?;
    if n == 0 || !(value >= bounds[0] && value <= bounds[n]) {
        return None;
    }
    // Number of lower bounds <= value, minus one; clamp the closed top edge.
    let bin = bounds[..n].partition_point(|&b| b <= value) - 1;
    Some(bin)
}

/// Orthonormal frame of a cylindrical mesh, as MCNP defines it: `axis` is
/// the cylinder's Z, and theta = 0 lies along `vec` projected onto the
/// plane normal to `axis`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylFrame {
    pub origin: [f64; 3],
    /// Unit vector along the cylinder axis.
    pub e_axis: [f64; 3],
    /// Unit vector at theta = 0.
    pub e_ref: [f64; 3],
    /// Unit vector at theta = 1/4 revolution (`e_axis × e_ref`).
    pub e_perp: [f64; 3],
}

impl CylFrame {
    pub fn new(origin: [f64; 3], axis: [f64; 3], vec: [f64; 3]) -> Self {
        let e_axis = normalize(axis);
        let e_ref = normalize(sub(vec, scale(e_axis, dot(vec, e_axis))));
        let e_perp = cross(e_axis, e_ref);
        CylFrame {
            origin,
            e_axis,
            e_ref,
            e_perp,
        }
    }

    /// World point of local `(r, z, theta)`, theta in revolutions.
    pub fn to_world(&self, r: f64, z: f64, theta: f64) -> [f64; 3] {
        let (s, c) = (theta * std::f64::consts::TAU).sin_cos();
        let radial = add(scale(self.e_ref, r * c), scale(self.e_perp, r * s));
        add(add(self.origin, scale(self.e_axis, z)), radial)
    }

    /// Local `(r, z, theta)` of a world point, theta in `[0, 1)` revolutions.
    pub fn to_local(&self, point: [f64; 3]) -> (f64, f64, f64) {
        let d = sub(point, self.origin);
        let z = dot(d, self.e_axis);
        let x = dot(d, self.e_ref);
        let y = dot(d, self.e_perp);
        let theta = y.atan2(x) / std::f64::consts::TAU;
        let theta = if theta < 0.0 { theta + 1.0 } else { theta };
        ((x * x + y * y).sqrt(), z, theta)
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / dot(a, a).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn rect() -> Geometry {
        Geometry::Rectangular {
            x_bounds: vec![0.0, 1.0, 3.0],
            y_bounds: vec![0.0, 2.0],
            z_bounds: vec![-1.0, 0.0, 1.0, 2.0],
            transform: None,
        }
    }

    fn cyl(axis: [f64; 3], vec: [f64; 3]) -> Geometry {
        Geometry::Cylindrical {
            r_bounds: vec![0.0, 1.0, 2.0],
            z_bounds: vec![0.0, 10.0],
            theta_bounds: vec![0.0, 0.25, 1.0],
            origin: [1.0, 2.0, 3.0],
            axis,
            vec,
        }
    }

    #[test]
    fn index_round_trip_inner_axis_fastest() {
        let g = rect();
        assert_eq!(g.dims(), (2, 1, 3));
        assert_eq!(g.index(0, 0, 1), 1);
        assert_eq!(g.index(1, 0, 0), 3);
        for n in 0..g.n_voxels() {
            let (i, j, k) = g.ijk(n);
            assert_eq!(g.index(i, j, k), n);
        }
    }

    #[test]
    fn rectangular_volumes_and_centers() {
        let g = rect();
        assert_eq!(g.voxel_volumes(), vec![2.0, 2.0, 2.0, 4.0, 4.0, 4.0]);
        assert_eq!(g.voxel_center(4), [2.0, 1.0, 0.5]);
    }

    #[test]
    fn cylindrical_volumes_sum_to_cylinder() {
        let g = cyl([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let total: f64 = g.voxel_volumes().iter().sum();
        assert_relative_eq!(
            total,
            std::f64::consts::PI * 4.0 * 10.0,
            max_relative = 1e-12
        );
        // Inner annulus, first quarter.
        assert_relative_eq!(g.voxel_volume(0), std::f64::consts::PI * 10.0 / 4.0);
    }

    #[test]
    fn locate_matches_center_for_every_voxel() {
        for g in [
            rect(),
            cyl([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            cyl([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ] {
            for n in 0..g.n_voxels() {
                assert_eq!(g.locate(g.voxel_center(n)), Some(n), "{g:?} voxel {n}");
            }
        }
    }

    #[test]
    fn locate_outside_and_edges() {
        let g = rect();
        assert_eq!(g.locate([-0.1, 1.0, 0.0]), None);
        assert_eq!(g.locate([3.0, 2.0, 2.0]), Some(5)); // closed top corner
        assert_eq!(g.locate([1.0, 0.0, -1.0]), Some(3)); // lower edge belongs to upper bin
    }

    #[test]
    fn validate_accepts_good_and_rejects_bad_geometry() {
        assert_eq!(rect().validate(), Ok(()));
        assert_eq!(cyl([0.0, 0.0, 1.0], [1.0, 1.0, 0.0]).validate(), Ok(()));

        let rect_with = |x: Vec<f64>| Geometry::Rectangular {
            x_bounds: x,
            y_bounds: vec![0.0, 1.0],
            z_bounds: vec![0.0, 1.0],
            transform: None,
        };
        assert_eq!(
            rect_with(vec![0.0]).validate(),
            Err(GeometryError::TooFewBounds { axis: "x" })
        );
        assert_eq!(rect_with(vec![]).dims(), (0, 1, 1));
        assert!(matches!(
            rect_with(vec![0.5, 0.5, 1.0]).validate(),
            Err(GeometryError::NotIncreasing {
                axis: "x",
                bin: 0,
                ..
            })
        ));
        assert!(matches!(
            rect_with(vec![0.0, f64::NAN]).validate(),
            Err(GeometryError::NonFinite {
                axis: "x",
                index: 1
            })
        ));

        let parallel = cyl([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        assert!(matches!(
            parallel.validate(),
            Err(GeometryError::BadFrame(_))
        ));
        let mut wide = cyl([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        if let Geometry::Cylindrical { theta_bounds, .. } = &mut wide {
            theta_bounds.push(1.5);
        }
        assert!(matches!(
            wide.validate(),
            Err(GeometryError::ThetaOutOfRange(..))
        ));
    }

    #[test]
    fn cylinder_frame_follows_vec() {
        // Axis along +X, theta = 0 along +Z: a point at +Y is a quarter turn.
        let frame = CylFrame::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let (r, z, t) = frame.to_local([5.0, 0.0, 2.0]);
        assert_relative_eq!(r, 2.0);
        assert_relative_eq!(z, 5.0);
        assert_relative_eq!(t, 0.0);
        let p = frame.to_world(1.0, 0.0, 0.25);
        let (_, _, t) = frame.to_local(p);
        assert_relative_eq!(t, 0.25, epsilon = 1e-12);
        // e_axis x e_ref = X x Z = -Y.
        assert_relative_eq!(p[1], -1.0, epsilon = 1e-12);
    }

    /// A quarter turn about Z then about the new X, shifted.
    fn tilted() -> Transform {
        // x' = +Y, y' = +Z, z' = +X (proper: Y x Z = X), as columns.
        Transform {
            rotation: [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            translation: [1.0, 2.0, 3.0],
        }
    }

    #[test]
    fn transform_conventions() {
        let t = tilted();
        assert_eq!(t.to_world([1.0, 0.0, 0.0]), [1.0, 3.0, 3.0]);
        assert_eq!(t.to_world([0.0, 1.0, 0.0]), [1.0, 2.0, 4.0]);
        let back = t.to_local(t.to_world([0.3, -2.0, 5.0]));
        for (a, b) in back.iter().zip([0.3, -2.0, 5.0]) {
            assert_relative_eq!(*a, b, epsilon = 1e-15);
        }

        assert_eq!(t.validate(), Ok(()));
        assert!(!t.is_reflection());

        // A reflection is a valid placement; it can't move a cylinder.
        let mut mirror = t;
        mirror.rotation[0][2] = -1.0;
        assert_eq!(mirror.validate(), Ok(()));
        assert!(mirror.is_reflection());
        assert!(rect().transformed(&mirror).is_ok());
        assert!(matches!(
            cyl([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).transformed(&mirror),
            Err(GeometryError::BadTransform(_))
        ));

        let mut skew = t;
        skew.rotation[0][0] = 0.1;
        assert!(skew.validate().is_err());
    }

    #[test]
    fn transformed_meshes_locate_and_center_in_world() {
        let t = tilted();
        let mut cylinder = cyl([0.0, 0.0, 1.0], [1.0, 1.0, 0.0]);
        let moved_cyl = cylinder.transformed(&t).unwrap();
        let moved_rect = rect().transformed(&t).unwrap();
        assert_eq!(moved_rect.validate(), Ok(()));
        assert_eq!(moved_cyl.validate(), Ok(()));
        for (local, world) in [(rect(), moved_rect), (cylinder.clone(), moved_cyl)] {
            assert_eq!(local.voxel_volumes(), world.voxel_volumes());
            for n in 0..local.n_voxels() {
                let expected = t.to_world(local.voxel_center(n));
                let center = world.voxel_center(n);
                for (a, b) in center.iter().zip(expected) {
                    assert_relative_eq!(*a, b, epsilon = 1e-12);
                }
                assert_eq!(world.locate(center), Some(n));
            }
        }

        // Composition: moving twice is moving by the composed transform.
        let twice = rect().transformed(&t).unwrap().transformed(&t).unwrap();
        let composed = t.then(&t);
        if let Geometry::Rectangular { transform, .. } = &twice {
            assert_eq!(transform.as_ref(), Some(&composed));
        }
        let p = [0.5, 1.0, -0.5];
        let once = t.to_world(t.to_world(p));
        for (a, b) in composed.to_world(p).iter().zip(once) {
            assert_relative_eq!(*a, b, epsilon = 1e-12);
        }

        // A cylinder's frame moves with it.
        if let Geometry::Cylindrical { origin, .. } = &mut cylinder {
            *origin = [0.0; 3];
        }
        let frame = cylinder.transformed(&t).unwrap().cyl_frame().unwrap();
        assert_eq!(frame.origin, [1.0, 2.0, 3.0]);
        assert_relative_eq!(frame.e_axis[0], 1.0);
    }
}
