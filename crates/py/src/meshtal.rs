//! `aci_tallies.meshtal`: meshtal files, read by parts.

use std::path::PathBuf;
use std::sync::Arc;

use aci_tallies::common::{Axis, Warning};
use aci_tallies::meshtal::{ops, Reader, Select, Tally, TallyInfo};
use ndarray::{s, Array5};
use numpy::{PyArray3, PyArray5};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::common::{
    call, input_arg, io_error, read_error, selection, view, warnings, PyAxis, PyReadWarning,
    Selection,
};
use crate::geometry::{geometry, PyGeometry};

/// An indexed meshtal file. Opening one reads where everything is (one
/// parallel pass over the file) and each tally's header; values are read
/// when asked for, so a file larger than memory can be read a tally, or a
/// bin, at a time. No file handle is kept: a file changed since it was
/// opened is reported rather than misread.
#[pyclass(name = "MeshtalFile", module = "aci_tallies.meshtal", frozen, mapping)]
pub struct PyMeshtalFile {
    reader: Arc<Reader>,
}

/// Open the meshtal file at `path`. With `input` (the MCNP input it came
/// from, as a path or an `aci_tallies.Input`), each tally's geometry and
/// energy and time bounds come from its FMESH card: exact, with `TR=`.
#[pyfunction]
#[pyo3(signature = (path, input = None))]
pub fn open_meshtal(
    py: Python<'_>,
    path: PathBuf,
    input: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyMeshtalFile> {
    let input = input_arg(py, input)?;
    let reader = py
        .allow_threads(|| match &input {
            Some(i) => Reader::open_with_input(&path, i),
            None => Reader::open(&path),
        })
        .map_err(io_error)?;
    Ok(PyMeshtalFile {
        reader: Arc::new(reader),
    })
}

impl PyMeshtalFile {
    fn tally_info(&self, id: u32) -> PyResult<&TallyInfo> {
        self.reader.tally(id).map_err(read_error)
    }
}

/// The indices an `energy=`/`time=` argument of `read` names (None: all).
fn indices(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
    axis: &Axis,
) -> PyResult<Option<(Vec<usize>, Axis)>> {
    match selection(name, value, axis.n_bins(), true)? {
        Selection::All => Ok(None),
        Selection::Total => {
            let t = axis.total_index().ok_or_else(|| {
                PyValueError::new_err(format!("{name}: this tally has no Total bin"))
            })?;
            Ok(Some((vec![t], axis.collapsed())))
        }
        Selection::Bins(r) => Ok(Some((r.clone().collect(), axis.select(r)))),
    }
}

#[pymethods]
impl PyMeshtalFile {
    /// The file's path.
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.reader.path().map(|p| p.to_path_buf())
    }

    /// The problem's title.
    #[getter]
    fn title(&self) -> &str {
        &self.reader.header().title
    }

    /// The number of histories the tallies are normalized to.
    #[getter]
    fn histories(&self) -> Option<f64> {
        self.reader.header().histories
    }

    /// The code that wrote the file and its version, as printed.
    #[getter]
    fn code(&self) -> (String, String) {
        let h = self.reader.header();
        (h.code.clone(), h.version.clone())
    }

    /// When the problem ran, as printed.
    #[getter]
    fn probid(&self) -> &str {
        &self.reader.header().probid
    }

    /// The numbers of the tallies that can be read, in file order.
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.reader.tallies().map(|t| t.id).collect()
    }

    /// What each tally that can be read is (no values read).
    #[getter]
    fn tallies(&self, py: Python<'_>) -> PyResult<Vec<PyTallyInfo>> {
        self.reader
            .tallies()
            .map(|t| PyTallyInfo::new(py, t.clone()))
            .collect()
    }

    /// The sections that can't be read: `(tally number, why)` (number 0
    /// when even that is unreadable).
    #[getter]
    fn failed(&self) -> Vec<(u32, String)> {
        self.reader
            .failed()
            .map(|(id, why)| (id, why.to_string()))
            .collect()
    }

    /// What tally `id` is, without reading its values.
    fn info(&self, py: Python<'_>, id: u32) -> PyResult<PyTallyInfo> {
        PyTallyInfo::new(py, self.tally_info(id)?.clone())
    }

    /// Whether the file is unchanged since it was opened.
    fn is_current(&self) -> bool {
        self.reader.is_current()
    }

    /// Tally `id`: all of it, or with `energy`/`time` (an index, or
    /// `"total"` for MCNP's Total bin) only those bins, reading only their
    /// part of the file. The result always has the five axes
    /// `[energy, time, i, j, k]`.
    #[pyo3(signature = (id, energy = None, time = None))]
    fn read(
        &self,
        py: Python<'_>,
        id: u32,
        energy: Option<&Bound<'_, PyAny>>,
        time: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyMeshTally> {
        let info = self.tally_info(id)?.clone();
        let e = indices("energy", energy, &info.energy)?;
        let t = indices("time", time, &info.time)?;
        let reader = &self.reader;
        let tally = if e.is_none() && t.is_none() {
            py.allow_threads(|| reader.read(id)).map_err(read_error)?
        } else {
            let (es, energy_axis) =
                e.unwrap_or_else(|| ((0..info.energy.len()).collect(), info.energy.clone()));
            let (ts, time_axis) =
                t.unwrap_or_else(|| ((0..info.time.len()).collect(), info.time.clone()));
            let [_, _, ni, nj, nk] = info.shape();
            py.allow_threads(|| -> PyResult<Tally> {
                let mut result = Array5::from_elem((es.len(), ts.len(), ni, nj, nk), f64::NAN);
                let mut rel_error = result.clone();
                let mut volume = None;
                let mut notes: Vec<Warning> = Vec::new();
                for (a, &e) in es.iter().enumerate() {
                    for (b, &t) in ts.iter().enumerate() {
                        let bin = reader.read_bin(id, e, t).map_err(read_error)?;
                        result.slice_mut(s![a, b, .., .., ..]).assign(&bin.result);
                        rel_error
                            .slice_mut(s![a, b, .., .., ..])
                            .assign(&bin.rel_error);
                        volume = volume.or(bin.volume);
                        for w in bin.warnings {
                            if !notes.contains(&w) {
                                notes.push(w);
                            }
                        }
                    }
                }
                Ok(Tally {
                    energy: energy_axis,
                    time: time_axis,
                    result,
                    rel_error,
                    volume,
                    warnings: notes,
                    ..info_tally(info)
                })
            })?
        };
        PyMeshTally::new(py, tally)
    }

    /// Every tally that can be read.
    fn read_all(&self, py: Python<'_>) -> PyResult<Vec<PyMeshTally>> {
        let reader = &self.reader;
        let all = py.allow_threads(|| reader.read_all()).map_err(read_error)?;
        all.into_iter().map(|t| PyMeshTally::new(py, t)).collect()
    }

    fn __len__(&self) -> usize {
        self.reader.tallies().count()
    }

    fn __contains__(&self, id: u32) -> bool {
        self.reader.tally(id).is_ok()
    }

    fn __getitem__(&self, py: Python<'_>, id: u32) -> PyResult<PyMeshTally> {
        self.read(py, id, None, None)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ids = self.ids().into_pyobject(py)?;
        Ok(ids.call_method0("__iter__")?.unbind())
    }

    fn keys(&self) -> Vec<u32> {
        self.ids()
    }

    fn __repr__(&self) -> String {
        let path = self
            .reader
            .path()
            .map_or("<memory>".to_string(), |p| p.display().to_string());
        format!("MeshtalFile({path:?}, tallies={:?})", self.ids())
    }

    fn _repr_html_(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        let py = slf.py();
        Ok(py
            .import("aci_tallies._html")?
            .call_method1("meshtal_file", (slf,))?
            .unbind())
    }
}

/// A tally with no values (the shape the arrays will be set to).
fn info_tally(info: TallyInfo) -> Tally {
    let empty = Array5::zeros((0, 0, 0, 0, 0));
    Tally {
        id: info.id,
        particle: info.particle,
        comment: info.comment,
        geometry: info.geometry,
        geometry_from_input: info.geometry_from_input,
        energy_weighted: info.energy_weighted,
        energy: info.energy,
        time: info.time,
        out_format: info.out_format,
        result: empty.clone(),
        rel_error: empty,
        volume: None,
        warnings: info.warnings,
    }
}

/// What a mesh tally is, without its values.
#[pyclass(name = "TallyInfo", module = "aci_tallies.meshtal", frozen)]
pub struct PyTallyInfo {
    info: TallyInfo,
    #[pyo3(get)]
    geometry: Py<PyGeometry>,
    #[pyo3(get)]
    energy: Py<PyAxis>,
    #[pyo3(get)]
    time: Py<PyAxis>,
}

impl PyTallyInfo {
    fn new(py: Python<'_>, info: TallyInfo) -> PyResult<PyTallyInfo> {
        Ok(PyTallyInfo {
            geometry: geometry(py, &info.geometry)?,
            energy: Py::new(
                py,
                PyAxis {
                    axis: info.energy.clone(),
                },
            )?,
            time: Py::new(
                py,
                PyAxis {
                    axis: info.time.clone(),
                },
            )?,
            info,
        })
    }
}

#[pymethods]
impl PyTallyInfo {
    #[getter]
    fn id(&self) -> u32 {
        self.info.id
    }

    #[getter]
    fn particle(&self) -> &str {
        &self.info.particle
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.info.comment.as_deref()
    }

    /// The `out=` layout the values are printed in.
    #[getter]
    fn out_format(&self) -> &'static str {
        self.info.out_format.name()
    }

    #[getter]
    fn geometry_from_input(&self) -> bool {
        self.info.geometry_from_input
    }

    #[getter]
    fn energy_weighted(&self) -> Option<bool> {
        self.info.energy_weighted
    }

    /// `(energy, time, i, j, k)`, Total bins included.
    #[getter]
    fn shape(&self) -> [usize; 5] {
        self.info.shape()
    }

    #[getter]
    fn warnings(&self) -> Vec<PyReadWarning> {
        warnings(&self.info.warnings)
    }

    fn __repr__(&self) -> String {
        format!(
            "TallyInfo(id={}, particle={:?}, shape={:?}, out={:?})",
            self.info.id,
            self.info.particle,
            self.info.shape(),
            self.info.out_format.name()
        )
    }
}

/// One mesh tally: values with relative errors on a mesh, in arrays of
/// shape `[energy, time, i, j, k]` (always five axes; an energy or time
/// axis has one more index at its end for MCNP's Total bin, when printed).
/// Its arrays are read-only views of what Rust read, not copies; the
/// operations make new tallies.
#[pyclass(name = "MeshTally", module = "aci_tallies.meshtal", frozen)]
pub struct PyMeshTally {
    tally: Tally,
    #[pyo3(get)]
    geometry: Py<PyGeometry>,
    #[pyo3(get)]
    energy: Py<PyAxis>,
    #[pyo3(get)]
    time: Py<PyAxis>,
}

impl PyMeshTally {
    pub fn new(py: Python<'_>, tally: Tally) -> PyResult<PyMeshTally> {
        Ok(PyMeshTally {
            geometry: geometry(py, &tally.geometry)?,
            energy: Py::new(
                py,
                PyAxis {
                    axis: tally.energy.clone(),
                },
            )?,
            time: Py::new(
                py,
                PyAxis {
                    axis: tally.time.clone(),
                },
            )?,
            tally,
        })
    }
}

enum Operand<'a> {
    Tally(&'a Tally),
    Scalar(f64),
}

fn operand<'a>(other: &'a Bound<'_, PyAny>) -> Option<Operand<'a>> {
    if let Ok(t) = other.downcast::<PyMeshTally>() {
        return Some(Operand::Tally(&t.get().tally));
    }
    if other.is_instance_of::<PyString>() {
        return None;
    }
    other.extract::<f64>().ok().map(Operand::Scalar)
}

fn ops_error(e: ops::OpsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn select(s: Selection) -> Select {
    match s {
        Selection::All => Select::All,
        Selection::Bins(r) => Select::Bins(r),
        Selection::Total => Select::Total,
    }
}

fn mesh_range(s: Selection) -> Option<std::ops::Range<usize>> {
    match s {
        Selection::Bins(r) => Some(r),
        _ => None,
    }
}

impl PyMeshTally {
    fn binary(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tallies: fn(&Tally, &Tally) -> Result<Tally, ops::OpsError>,
        scalar: Option<fn(&Tally, f64) -> Tally>,
    ) -> PyResult<PyObject> {
        let a = &self.tally;
        let out = match (operand(other), scalar) {
            (Some(Operand::Tally(b)), _) => {
                py.allow_threads(|| tallies(a, b)).map_err(ops_error)?
            }
            (Some(Operand::Scalar(f)), Some(scalar)) => py.allow_threads(|| scalar(a, f)),
            _ => return Ok(py.NotImplemented()),
        };
        Ok(PyMeshTally::new(py, out)?
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

#[pymethods]
impl PyMeshTally {
    #[getter]
    fn id(&self) -> u32 {
        self.tally.id
    }

    #[getter]
    fn particle(&self) -> &str {
        &self.tally.particle
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.tally.comment.as_deref()
    }

    /// The `out=` layout the values were printed in.
    #[getter]
    fn out_format(&self) -> &'static str {
        self.tally.out_format.name()
    }

    /// Whether the geometry and the energy and time bounds came from the
    /// MCNP input (exact, `TR=` applied) rather than the rounded header.
    #[getter]
    fn geometry_from_input(&self) -> bool {
        self.tally.geometry_from_input
    }

    /// Whether the tally is a `*FMESH` (MeV/cm² rather than 1/cm²); None
    /// when no input was given (only the input says).
    #[getter]
    fn energy_weighted(&self) -> Option<bool> {
        self.tally.energy_weighted
    }

    /// `(energy, time, i, j, k)`.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.tally.result.shape().to_vec()
    }

    /// The values, shape `[energy, time, i, j, k]` (read-only).
    #[getter]
    fn result<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray5<f64>>> {
        // Safety: the tally is frozen and owned by `slf`.
        unsafe { view(&slf.get().tally.result, slf.clone().into_any()) }
    }

    /// Relative errors (a fraction), shaped as `result` (read-only).
    #[getter]
    fn rel_error<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray5<f64>>> {
        unsafe { view(&slf.get().tally.rel_error, slf.clone().into_any()) }
    }

    /// Voxel volumes (cm³), shape `[i, j, k]`, when the file gives them
    /// (`out=cf`); else None (`geometry.volumes()` has them).
    #[getter]
    fn volume<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyArray3<f64>>>> {
        match &slf.get().tally.volume {
            Some(v) => Ok(Some(unsafe { view(v, slf.clone().into_any())? })),
            None => Ok(None),
        }
    }

    #[getter]
    fn warnings(&self) -> Vec<PyReadWarning> {
        warnings(&self.tally.warnings)
    }

    /// Everything but the values.
    fn info(&self, py: Python<'_>) -> PyResult<PyTallyInfo> {
        PyTallyInfo::new(py, self.tally.info())
    }

    /// Part of the tally: `energy` and `time` an index, a slice or
    /// `"total"` (MCNP's Total bin, as one bin spanning the axis); `i`,
    /// `j`, `k` an index or a slice. Omitted axes are kept whole, Total bin
    /// included; the five axes always stay.
    #[pyo3(signature = (energy = None, time = None, i = None, j = None, k = None))]
    fn isel(
        &self,
        py: Python<'_>,
        energy: Option<&Bound<'_, PyAny>>,
        time: Option<&Bound<'_, PyAny>>,
        i: Option<&Bound<'_, PyAny>>,
        j: Option<&Bound<'_, PyAny>>,
        k: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyMeshTally> {
        let t = &self.tally;
        let (ni, nj, nk) = t.geometry.dims();
        let e = select(selection("energy", energy, t.energy.n_bins(), true)?);
        let tm = select(selection("time", time, t.time.n_bins(), true)?);
        let i = mesh_range(selection("i", i, ni, false)?);
        let j = mesh_range(selection("j", j, nj, false)?);
        let k = mesh_range(selection("k", k, nk, false)?);
        let out = py
            .allow_threads(|| ops::isel(t, &e, &tm, i, j, k))
            .map_err(ops_error)?;
        PyMeshTally::new(py, out)
    }

    /// Sum over the energy bins: MCNP's Total bin when printed (its error
    /// comes from the histories), else the bins added, errors in
    /// quadrature.
    fn integrate_energy(&self, py: Python<'_>) -> PyResult<PyMeshTally> {
        let t = &self.tally;
        PyMeshTally::new(py, py.allow_threads(|| ops::integrate_energy(t)))
    }

    /// Sum over the time bins, as `integrate_energy`.
    fn integrate_time(&self, py: Python<'_>) -> PyResult<PyMeshTally> {
        let t = &self.tally;
        PyMeshTally::new(py, py.allow_threads(|| ops::integrate_time(t)))
    }

    /// Integrate over the mesh: `sum(result * volume)` per energy and time
    /// bin, leaving one voxel spanning the mesh. Volumes are the file's
    /// (`out=cf`) or else the geometry's.
    fn integrate_space(&self, py: Python<'_>) -> PyResult<PyMeshTally> {
        let t = &self.tally;
        PyMeshTally::new(py, py.allow_threads(|| ops::integrate_space(t)))
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.binary(py, other, ops::add, None)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.binary(py, other, ops::sub, None)
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.binary(py, other, ops::mul, Some(ops::scale))
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.binary(py, other, ops::mul, Some(ops::scale))
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.binary(py, other, ops::div, Some(|t, f| ops::scale(t, 1.0 / f)))
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<PyMeshTally> {
        PyMeshTally::new(py, ops::scale(&self.tally, -1.0))
    }

    /// A long table, one row per (energy, time, voxel): see
    /// `aci_tallies.meshtal` for the columns. Needs polars.
    fn to_polars(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._polars", "meshtal_frame", ())
    }

    /// Write `to_polars()` to a Parquet file. Needs polars.
    fn to_parquet(slf: &Bound<'_, Self>, path: PathBuf) -> PyResult<PyObject> {
        call(slf, "aci_tallies._polars", "meshtal_parquet", (path,))
    }

    /// An `xarray.Dataset` with named dimensions and bin coordinates. Needs
    /// xarray.
    fn to_xarray(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._xarray", "meshtal_dataset", ())
    }

    /// A PyVista grid with every (field, energy, time) as cell data; see
    /// `aci_tallies._vtk.to_pyvista`. Needs pyvista.
    #[pyo3(signature = (energy = None, time = None, fields = None, max_theta_step = None))]
    fn to_pyvista(
        slf: &Bound<'_, Self>,
        energy: Option<PyObject>,
        time: Option<PyObject>,
        fields: Option<PyObject>,
        max_theta_step: Option<f64>,
    ) -> PyResult<PyObject> {
        call(
            slf,
            "aci_tallies._vtk",
            "to_pyvista",
            (energy, time, fields, max_theta_step),
        )
    }

    /// Write the `to_pyvista` grid to a VTK file (`.vtr`, or `.vts` for a
    /// cylindrical or transformed mesh), without needing pyvista.
    #[pyo3(signature = (path, energy = None, time = None, fields = None, max_theta_step = None, compress = true))]
    fn to_vtk(
        slf: &Bound<'_, Self>,
        path: PathBuf,
        energy: Option<PyObject>,
        time: Option<PyObject>,
        fields: Option<PyObject>,
        max_theta_step: Option<f64>,
        compress: bool,
    ) -> PyResult<PyObject> {
        call(
            slf,
            "aci_tallies._vtk",
            "to_vtk",
            (path, energy, time, fields, max_theta_step, compress),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshTally(id={}, particle={:?}, shape={:?}, out={:?})",
            self.tally.id,
            self.tally.particle,
            self.tally.result.shape(),
            self.tally.out_format.name()
        )
    }

    fn _repr_html_(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._html", "mesh_tally", ())
    }
}
