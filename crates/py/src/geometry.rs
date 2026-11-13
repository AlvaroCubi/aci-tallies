//! `aci_tallies.Geometry`: a mesh, from the `aci-tallies-geom` crate.

use std::path::PathBuf;

use aci_tallies_geom::{vtk, Geometry, Transform};
use ndarray::{Array1, Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// One of each of a mesh's three axes.
type Three<T> = (T, T, T);

/// A structured mesh: rectangular (x, y, z bins, maybe placed by a
/// transform) or cylindrical (r, z, theta bins about `origin`, `axis`,
/// `vec`). Voxel arrays are indexed `[i, j, k]` along `axis_names`.
#[pyclass(name = "Geometry", module = "aci_tallies", frozen)]
pub struct PyGeometry {
    pub geometry: Geometry,
}

#[pymethods]
impl PyGeometry {
    /// `"rectangular"` or `"cylindrical"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.geometry {
            Geometry::Rectangular { .. } => "rectangular",
            Geometry::Cylindrical { .. } => "cylindrical",
        }
    }

    /// The names of the `(i, j, k)` axes: `("x", "y", "z")` or `("r", "z",
    /// "theta")`.
    #[getter]
    fn axis_names(&self) -> (&'static str, &'static str, &'static str) {
        match self.geometry {
            Geometry::Rectangular { .. } => ("x", "y", "z"),
            Geometry::Cylindrical { .. } => ("r", "z", "theta"),
        }
    }

    /// The bin bounds along `(i, j, k)` (cm; theta in revolutions), in the
    /// mesh's own frame.
    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> Three<Bound<'py, PyArray1<f64>>> {
        let (a, b, c) = self.geometry.axes();
        let f = |x| PyArray1::from_slice(py, x);
        (f(a), f(b), f(c))
    }

    /// Voxels along `(i, j, k)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.geometry.dims()
    }

    /// A cylinder's origin, or None.
    #[getter]
    fn origin(&self) -> Option<[f64; 3]> {
        match &self.geometry {
            Geometry::Cylindrical { origin, .. } => Some(*origin),
            Geometry::Rectangular { .. } => None,
        }
    }

    /// A cylinder's axis, or None.
    #[getter]
    fn axis(&self) -> Option<[f64; 3]> {
        match &self.geometry {
            Geometry::Cylindrical { axis, .. } => Some(*axis),
            Geometry::Rectangular { .. } => None,
        }
    }

    /// The direction a cylinder's theta is measured from, or None.
    #[getter]
    fn vec(&self) -> Option<[f64; 3]> {
        match &self.geometry {
            Geometry::Cylindrical { vec, .. } => Some(*vec),
            Geometry::Rectangular { .. } => None,
        }
    }

    /// The 4x4 affine matrix placing a rectangular mesh in the world
    /// (`TR=`), or None. A cylinder's placement is in `origin`, `axis` and
    /// `vec`.
    #[getter]
    fn transform<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match &self.geometry {
            Geometry::Rectangular {
                transform: Some(t), ..
            } => Some(affine(t).into_pyarray(py)),
            _ => None,
        }
    }

    /// Voxel volumes (cm³), shape `(i, j, k)`.
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        Array3::from_shape_vec(self.geometry.dims(), self.geometry.voxel_volumes())
            .expect("one volume per voxel")
            .into_pyarray(py)
    }

    /// Voxel centres in world coordinates, shape `(i, j, k, 3)` (for a
    /// cylinder, mid-radius, mid-height and mid-angle: the points MCNP
    /// prints).
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        let (a, b, c) = self.geometry.dims();
        let flat: Vec<f64> = self
            .geometry
            .voxel_centers()
            .into_iter()
            .flatten()
            .collect();
        Array4::from_shape_vec((a, b, c, 3), flat)
            .expect("three coordinates per voxel")
            .into_pyarray(py)
    }

    /// The voxel holding each of `points` (world coordinates, shape `(n,
    /// 3)`), as `(i, j, k)` rows; `-1` for a point outside the mesh.
    fn locate<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let points = points.as_array();
        if points.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (n, 3)"));
        }
        let mut out = Array2::from_elem((points.nrows(), 3), -1i64);
        for (n, p) in points.outer_iter().enumerate() {
            if let Some(v) = self.geometry.locate([p[0], p[1], p[2]]) {
                let (i, j, k) = self.geometry.ijk(v);
                out[[n, 0]] = i as i64;
                out[[n, 1]] = j as i64;
                out[[n, 2]] = k as i64;
            }
        }
        Ok(out.into_pyarray(py))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.geometry == other.geometry
    }

    fn __repr__(&self) -> String {
        let (a, b, c) = self.geometry.dims();
        let (na, nb, nc) = self.axis_names();
        let placed = match &self.geometry {
            Geometry::Rectangular {
                transform: Some(_), ..
            } => ", transformed",
            _ => "",
        };
        format!(
            "Geometry({}, {na}={a}, {nb}={b}, {nc}={c}{placed})",
            self.kind()
        )
    }

    /// The VTK grid the mesh is drawn on: a dict with `kind`
    /// (`"rectilinear"` with `x`/`y`/`z`, or `"structured"` with `dims` and
    /// `points` of shape `(n, 3)`), and `cell_voxels`, the flat (C-order)
    /// voxel behind each VTK cell.
    fn _vtk_layout<'py>(
        &self,
        py: Python<'py>,
        max_theta_step: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let vtk::GridLayout { grid, cell_voxels } = vtk::layout(&self.geometry, max_theta_step);
        let out = PyDict::new(py);
        match grid {
            vtk::Grid::Rectilinear { x, y, z } => {
                out.set_item("kind", "rectilinear")?;
                out.set_item("x", x.into_pyarray(py))?;
                out.set_item("y", y.into_pyarray(py))?;
                out.set_item("z", z.into_pyarray(py))?;
            }
            vtk::Grid::Structured { dims, points } => {
                let n = points.len();
                let flat: Vec<f64> = points.into_iter().flatten().collect();
                let points = Array2::from_shape_vec((n, 3), flat).expect("3 coordinates per point");
                out.set_item("kind", "structured")?;
                out.set_item("dims", (dims[0], dims[1], dims[2]))?;
                out.set_item("points", points.into_pyarray(py))?;
            }
        }
        let cells: Array1<i64> = cell_voxels.into_iter().map(|v| v as i64).collect();
        out.set_item("cell_voxels", cells.into_pyarray(py))?;
        Ok(out)
    }

    /// The file extension VTK files of this mesh take: `"vtr"` or `"vts"`.
    #[getter]
    fn _vtk_extension(&self) -> &'static str {
        vtk::extension(&self.geometry)
    }

    /// Write `arrays` (each of shape `(i, j, k)`) as cell data named
    /// `names` to a VTK XML file.
    fn _write_vtk(
        &self,
        py: Python<'_>,
        path: PathBuf,
        names: Vec<String>,
        arrays: Vec<PyReadonlyArray3<'_, f64>>,
        max_theta_step: f64,
        compress: bool,
    ) -> PyResult<()> {
        let expected = vtk::extension(&self.geometry);
        if path.extension().and_then(|e| e.to_str()) != Some(expected) {
            return Err(PyValueError::new_err(format!(
                "this mesh is written as .{expected}; got {}",
                path.display()
            )));
        }
        if names.len() != arrays.len() {
            return Err(PyValueError::new_err("one name per array"));
        }
        let dims = self.geometry.dims();
        let mut values = Vec::with_capacity(arrays.len());
        for (name, array) in names.iter().zip(&arrays) {
            let view = array.as_array();
            if view.dim() != dims {
                return Err(PyValueError::new_err(format!(
                    "array {name:?} has shape {:?}, the mesh is {dims:?}",
                    view.dim()
                )));
            }
            // Logical (C) order is the flat voxel order, whatever the strides.
            values.push(view.iter().copied().collect::<Vec<f64>>());
        }
        let options = vtk::VtkOptions {
            max_theta_step,
            compress,
        };
        py.allow_threads(|| {
            let cells: Vec<vtk::CellArray> = names
                .iter()
                .zip(&values)
                .map(|(name, v)| vtk::CellArray {
                    name,
                    values: vtk::CellValues::F64(v),
                })
                .collect();
            vtk::write_file(&path, &self.geometry, &cells, &options)
        })
        .map_err(|e| PyIOError::new_err(format!("{}: {e}", path.display())))
    }
}

fn affine(t: &Transform) -> Array2<f64> {
    let mut m = Array2::eye(4);
    for row in 0..3 {
        for col in 0..3 {
            m[[row, col]] = t.rotation[row][col];
        }
        m[[row, 3]] = t.translation[row];
    }
    m
}

pub fn geometry(py: Python<'_>, g: &Geometry) -> PyResult<Py<PyGeometry>> {
    Py::new(
        py,
        PyGeometry {
            geometry: g.clone(),
        },
    )
}
