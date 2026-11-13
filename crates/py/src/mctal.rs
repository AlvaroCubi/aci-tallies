//! `aci_tallies.mctal`: MCTAL files, read by parts.

use std::path::PathBuf;
use std::sync::Arc;

use aci_tallies::mctal::{
    ops, Detector, Kcode, Modifier, Reader, Select, Tally, TallyBin, TallyInfo, Tfc, DIMS, F,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::common::{
    call, input_arg, io_error, read_error, selection, view, warnings, PyAxis, PyReadWarning,
    Selection,
};

/// An indexed MCTAL file. Opening one reads where everything is (one
/// parallel pass) and each tally's header; values are read when asked for,
/// a tally, or a range of its cells, at a time. No file handle is kept.
#[pyclass(name = "MctalFile", module = "aci_tallies.mctal", frozen, mapping)]
pub struct PyMctalFile {
    reader: Arc<Reader>,
}

/// Open the MCTAL file at `path`. `input`, the MCNP input it came from (a
/// path or an `aci_tallies.Input`), is read for what the file doesn't say.
#[pyfunction]
#[pyo3(signature = (path, input = None))]
pub fn open_mctal(
    py: Python<'_>,
    path: PathBuf,
    input: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyMctalFile> {
    let input = input_arg(py, input)?;
    let reader = py
        .allow_threads(|| match &input {
            Some(i) => Reader::open_with_input(&path, i),
            None => Reader::open(&path),
        })
        .map_err(io_error)?;
    Ok(PyMctalFile {
        reader: Arc::new(reader),
    })
}

fn detector_name(d: Detector) -> Option<&'static str> {
    match d {
        Detector::None => None,
        Detector::Point => Some("point"),
        Detector::Ring => Some("ring"),
        Detector::PinholeImage => Some("pinhole image"),
        Detector::RectangularImage => Some("rectangular image"),
        Detector::CylindricalImage => Some("cylindrical image"),
    }
}

fn modifier_name(m: Modifier) -> &'static str {
    match m {
        Modifier::None => "",
        Modifier::Star => "*",
        Modifier::Plus => "+",
    }
}

#[pymethods]
impl PyMctalFile {
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.reader.path().map(|p| p.to_path_buf())
    }

    #[getter]
    fn title(&self) -> &str {
        &self.reader.header().title
    }

    /// Histories run.
    #[getter]
    fn histories(&self) -> u64 {
        self.reader.header().histories
    }

    /// The code that wrote the file and its version (empty when the file
    /// leaves them out).
    #[getter]
    fn code(&self) -> (String, String) {
        let h = self.reader.header();
        (h.code.clone(), h.version.clone())
    }

    #[getter]
    fn probid(&self) -> &str {
        &self.reader.header().probid
    }

    /// The dump the tallies are from.
    #[getter]
    fn dump(&self) -> u64 {
        self.reader.header().dump
    }

    /// Pseudorandom numbers used.
    #[getter]
    fn randoms(&self) -> u64 {
        self.reader.header().randoms
    }

    /// Number of perturbations (`PERT` cards).
    #[getter]
    fn perturbations(&self) -> usize {
        self.reader.header().perturbations
    }

    /// The numbers of the tallies that can be read, in file order.
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.reader.tallies().map(|t| t.id).collect()
    }

    /// What each tally that can be read is (no values read).
    #[getter]
    fn tallies(&self, py: Python<'_>) -> PyResult<Vec<PyMctalInfo>> {
        self.reader
            .tallies()
            .map(|t| PyMctalInfo::new(py, t.clone()))
            .collect()
    }

    /// The tallies that can't be read: `(tally number, why)`.
    #[getter]
    fn failed(&self) -> Vec<(u32, String)> {
        self.reader
            .failed()
            .map(|(id, why)| (id, why.to_string()))
            .collect()
    }

    /// Problems with the file as a whole.
    #[getter]
    fn warnings(&self) -> Vec<PyReadWarning> {
        warnings(self.reader.warnings())
    }

    /// A KCODE run's estimates per cycle, or None.
    #[getter]
    fn kcode(&self) -> PyResult<Option<PyKcode>> {
        match self.reader.kcode() {
            None => Ok(None),
            Some(Ok(k)) => Ok(Some(PyKcode { kcode: k.clone() })),
            Some(Err(e)) => Err(PyValueError::new_err(format!(
                "the kcode block can't be read: {e}"
            ))),
        }
    }

    /// What tally `id` is, without reading its values.
    fn info(&self, py: Python<'_>, id: u32) -> PyResult<PyMctalInfo> {
        PyMctalInfo::new(py, self.reader.tally(id).map_err(read_error)?.clone())
    }

    fn is_current(&self) -> bool {
        self.reader.is_current()
    }

    /// Tally `id`: all of it, or with `f` (an index or a slice) only those
    /// cell or surface bins, reading only their part of the file; with
    /// `perturbation` (from 1), that perturbation's values.
    #[pyo3(signature = (id, f = None, perturbation = None))]
    fn read(
        &self,
        py: Python<'_>,
        id: u32,
        f: Option<&Bound<'_, PyAny>>,
        perturbation: Option<usize>,
    ) -> PyResult<PyMctalTally> {
        let nf = self.reader.tally(id).map_err(read_error)?.axes[F].len();
        let range = match selection("f", f, nf, false)? {
            Selection::Bins(r) => Some(r),
            _ => None,
        };
        let reader = &self.reader;
        let tally = py
            .allow_threads(|| match (perturbation, range) {
                (Some(_), Some(_)) => Err(aci_tallies::ReadError::Unsupported {
                    id,
                    reason: "a perturbation is read whole".into(),
                }),
                (Some(n), None) => reader.read_perturbation(id, n),
                (None, Some(r)) => reader.read_f(id, r),
                (None, None) => reader.read(id),
            })
            .map_err(read_error)?;
        PyMctalTally::new(py, tally)
    }

    /// Every tally that can be read (unperturbed).
    fn read_all(&self, py: Python<'_>) -> PyResult<Vec<PyMctalTally>> {
        let reader = &self.reader;
        let all = py.allow_threads(|| reader.read_all()).map_err(read_error)?;
        all.into_iter().map(|t| PyMctalTally::new(py, t)).collect()
    }

    fn __len__(&self) -> usize {
        self.reader.tallies().count()
    }

    fn __contains__(&self, id: u32) -> bool {
        self.reader.tally(id).is_ok()
    }

    fn __getitem__(&self, py: Python<'_>, id: u32) -> PyResult<PyMctalTally> {
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
        format!("MctalFile({path:?}, tallies={:?})", self.ids())
    }

    fn _repr_html_(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._html", "mctal_file", ())
    }
}

/// The axes of a tally, by name, in order.
fn axes_dict<'py>(py: Python<'py>, axes: &[Py<PyAxis>; 8]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, axis) in DIMS.iter().zip(axes) {
        d.set_item(*name, axis.clone_ref(py))?;
    }
    Ok(d)
}

fn py_axes(py: Python<'_>, axes: &[aci_tallies::Axis; 8]) -> PyResult<[Py<PyAxis>; 8]> {
    let v: Vec<Py<PyAxis>> = axes
        .iter()
        .map(|a| Py::new(py, PyAxis { axis: a.clone() }))
        .collect::<PyResult<_>>()?;
    Ok(v.try_into().unwrap_or_else(|_| unreachable!("eight axes")))
}

/// The getters an MCTAL tally and its info share, on `$ty` whose field
/// `$t` is the tally or info.
macro_rules! tally_getters {
    ($ty:ty, $t:ident) => {
        #[pymethods]
        impl $ty {
            #[getter]
            fn id(&self) -> u32 {
                self.$t.id
            }

            /// The MCNP tally type: the last digit of its number (1 current, 2
            /// surface flux, 4 cell flux, 5 detector, 6 heating, 7 fission
            /// heating, 8 pulse height).
            #[getter]
            fn kind(&self) -> u32 {
                self.$t.id % 10
            }

            /// Particles scored, as MCNP designators (`"n"`, `"np"`, ...).
            #[getter]
            fn particles(&self) -> String {
                self.$t.particles.iter().collect()
            }

            /// `"point"`, `"ring"`, `"pinhole image"`, `"rectangular image"`,
            /// `"cylindrical image"`, or None.
            #[getter]
            fn detector(&self) -> Option<&'static str> {
                detector_name(self.$t.detector)
            }

            /// `"*"` for `*F`, `"+"` for `+F6`, else `""`.
            #[getter]
            fn modifier(&self) -> &'static str {
                modifier_name(self.$t.modifier)
            }

            #[getter]
            fn comment(&self) -> Option<&str> {
                self.$t.comment.as_deref()
            }

            /// The names of the eight axes, in order: `f d u s m c e t`.
            #[getter]
            fn dims(&self) -> [&'static str; 8] {
                DIMS
            }

            /// The axes by name (`f`, `d`, `u`, `s`, `m`, `c`, `e`, `t`).
            #[getter]
            fn axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                axes_dict(py, &self.axes)
            }

            /// The cell or surface number of each `f` bin, None where MCTAL
            /// gives none (a group, the T bin, several cells at a bin's first
            /// level; detectors). Lower levels are never given.
            #[getter]
            fn numbers(&self) -> Option<Vec<Option<i64>>> {
                (!self.$t.numbers.is_empty()).then(|| {
                    self.$t
                        .numbers
                        .iter()
                        .map(|n| n.map(|n| n.number))
                        .collect()
                })
            }

            /// What each `f` bin is, from the tally's card in the MCNP input (when
            /// given to `open`, and it matches the file); else None.
            #[getter]
            fn bins(&self) -> Option<Vec<PyBin>> {
                (!self.$t.bins.is_empty()).then(|| {
                    self.$t
                        .bins
                        .iter()
                        .map(|b| PyBin { bin: b.clone() })
                        .collect()
                })
            }

            /// The macrobody facet of each `f` bin's surface, where it has one.
            #[getter]
            fn facets(&self) -> Option<Vec<Option<u8>>> {
                let facets: Vec<Option<u8>> = self
                    .$t
                    .numbers
                    .iter()
                    .map(|n| n.and_then(|n| n.facet))
                    .collect();
                facets.iter().any(Option::is_some).then_some(facets)
            }

            /// A legacy TMESH mesh tally's grid: `{"geometry": 1|2|3, "bounds":
            /// (a, b, c)}`, or None.
            #[getter]
            fn mesh<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
                let Some(m) = &self.$t.mesh else {
                    return Ok(None);
                };
                let d = PyDict::new(py);
                d.set_item("geometry", m.geometry)?;
                let [a, b, c] = &m.bounds;
                d.set_item(
                    "bounds",
                    (
                        PyArray1::from_slice(py, a),
                        PyArray1::from_slice(py, b),
                        PyArray1::from_slice(py, c),
                    ),
                )?;
                Ok(Some(d))
            }

            /// The length of each axis, Total bins included.
            #[getter]
            fn shape(&self) -> Vec<usize> {
                self.$t.axes.iter().map(|a| a.len()).collect()
            }

            #[getter]
            fn warnings(&self) -> Vec<PyReadWarning> {
                warnings(&self.$t.warnings)
            }
        }
    };
}

/// What an MCTAL tally is, without its values.
#[pyclass(name = "TallyInfo", module = "aci_tallies.mctal", frozen)]
pub struct PyMctalInfo {
    info: TallyInfo,
    axes: [Py<PyAxis>; 8],
}

impl PyMctalInfo {
    fn new(py: Python<'_>, info: TallyInfo) -> PyResult<PyMctalInfo> {
        Ok(PyMctalInfo {
            axes: py_axes(py, &info.axes)?,
            info,
        })
    }
}

tally_getters!(PyMctalInfo, info);

#[pymethods]
impl PyMctalInfo {
    fn __repr__(&self) -> String {
        format!(
            "TallyInfo(id={}, shape={:?})",
            self.info.id,
            self.info.shape()
        )
    }
}

/// One MCTAL tally: values and relative errors over eight axes, `f d u s m
/// c e t` (always eight; an axis with MCNP's Total bin has it last). The
/// arrays are read-only views of what Rust read; operations make new
/// tallies.
#[pyclass(name = "Tally", module = "aci_tallies.mctal", frozen)]
pub struct PyMctalTally {
    tally: Tally,
    axes: [Py<PyAxis>; 8],
}

impl PyMctalTally {
    fn new(py: Python<'_>, tally: Tally) -> PyResult<PyMctalTally> {
        Ok(PyMctalTally {
            axes: py_axes(py, &tally.axes)?,
            tally,
        })
    }

    fn binary(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tallies: fn(&Tally, &Tally) -> Result<Tally, ops::OpsError>,
        scalar: Option<fn(&Tally, f64) -> Tally>,
    ) -> PyResult<PyObject> {
        let a = &self.tally;
        let out = if let Ok(b) = other.downcast::<PyMctalTally>() {
            let b = &b.get().tally;
            py.allow_threads(|| tallies(a, b))
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let (Some(scalar), false, Ok(f)) = (
            scalar,
            other.is_instance_of::<PyString>(),
            other.extract::<f64>(),
        ) {
            py.allow_threads(|| scalar(a, f))
        } else {
            return Ok(py.NotImplemented());
        };
        Ok(PyMctalTally::new(py, out)?
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

fn dim_index(name: &str) -> PyResult<usize> {
    DIMS.iter()
        .position(|d| *d == name)
        .ok_or_else(|| PyKeyError::new_err(format!("no axis {name:?}; the axes are {DIMS:?}")))
}

tally_getters!(PyMctalTally, tally);

#[pymethods]
impl PyMctalTally {
    /// The perturbation these values are of (its `PERT` number), or None.
    #[getter]
    fn perturbation(&self) -> Option<u32> {
        self.tally.perturbation
    }

    /// The values, shape `(f, d, u, s, m, c, e, t)` (read-only).
    #[getter]
    fn result<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        // Safety: the tally is frozen and owned by `slf`.
        unsafe { view(&slf.get().tally.result, slf.clone().into_any()) }
    }

    /// Relative errors (a fraction), shaped as `result` (read-only).
    #[getter]
    fn rel_error<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        unsafe { view(&slf.get().tally.rel_error, slf.clone().into_any()) }
    }

    /// The tally fluctuation chart, or None.
    #[getter]
    fn tfc(&self) -> Option<PyTfc> {
        self.tally.tfc.clone().map(|tfc| PyTfc { tfc })
    }

    /// Everything but the values.
    fn info(&self, py: Python<'_>) -> PyResult<PyMctalInfo> {
        PyMctalInfo::new(py, self.tally.info())
    }

    /// Part of the tally: for each axis named, an index, a slice, or
    /// `"total"` (its Total bin, as one bin spanning the axis). Omitted
    /// axes are kept whole; the eight axes always stay.
    #[pyo3(signature = (**selected))]
    fn isel(&self, py: Python<'_>, selected: Option<&Bound<'_, PyDict>>) -> PyResult<PyMctalTally> {
        let mut select: [Select; 8] = Default::default();
        if let Some(selected) = selected {
            for (name, value) in selected.iter() {
                let name: String = name.extract()?;
                let k = dim_index(&name)?;
                let axis = &self.tally.axes[k];
                select[k] = match selection(DIMS[k], Some(&value), axis.n_bins(), true)? {
                    Selection::All => Select::All,
                    Selection::Bins(r) => Select::Bins(r),
                    Selection::Total => Select::Total,
                };
            }
        }
        let t = &self.tally;
        let out = py
            .allow_threads(|| ops::isel(t, &select))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyMctalTally::new(py, out)
    }

    /// Sum over the axes named (`"e"`, or several: `"e", "t"`): MCNP's
    /// Total bin where printed, the last bin of a cumulative axis, else the
    /// bins added, errors in quadrature. Each axis is left with one bin.
    #[pyo3(signature = (*dims))]
    fn integrate(&self, py: Python<'_>, dims: Vec<String>) -> PyResult<PyMctalTally> {
        let dims: Vec<usize> = dims.iter().map(|d| dim_index(d)).collect::<PyResult<_>>()?;
        let t = &self.tally;
        let out = py.allow_threads(|| {
            let mut out = t.clone();
            for &d in &dims {
                out = ops::integrate(&out, d);
            }
            out
        });
        PyMctalTally::new(py, out)
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

    fn __neg__(&self, py: Python<'_>) -> PyResult<PyMctalTally> {
        PyMctalTally::new(py, ops::scale(&self.tally, -1.0))
    }

    /// A long table, one row per value; see `aci_tallies.mctal` for the
    /// columns. Needs polars.
    fn to_polars(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._polars", "mctal_frame", ())
    }

    /// Write `to_polars()` to a Parquet file. Needs polars.
    fn to_parquet(slf: &Bound<'_, Self>, path: PathBuf) -> PyResult<PyObject> {
        call(slf, "aci_tallies._polars", "mctal_parquet", (path,))
    }

    /// An `xarray.Dataset` on dimensions `f d u s m c e t`, with bin
    /// coordinates. Needs xarray.
    fn to_xarray(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._xarray", "mctal_dataset", ())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tally(id={}, shape={:?}{})",
            self.tally.id,
            self.tally.result.shape(),
            self.tally
                .perturbation
                .map_or(String::new(), |p| format!(", perturbation={p}"))
        )
    }

    fn _repr_html_(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._html", "mctal_tally", ())
    }
}

/// A tally fluctuation chart: how one bin's mean and error evolved with
/// the histories run.
#[pyclass(name = "Tfc", module = "aci_tallies.mctal", frozen)]
pub struct PyTfc {
    tfc: Tfc,
}

#[pymethods]
impl PyTfc {
    /// The bin charted: an index along each of the eight axes.
    #[getter]
    fn bin(&self) -> [usize; 8] {
        self.tfc.bin
    }

    #[getter]
    fn histories<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_slice(py, &self.tfc.histories)
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.tfc.mean)
    }

    #[getter]
    fn rel_error<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.tfc.rel_error)
    }

    /// Figure of merit, or None when the file leaves it out.
    #[getter]
    fn fom<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.tfc.fom.as_ref().map(|f| PyArray1::from_slice(py, f))
    }

    fn __repr__(&self) -> String {
        format!(
            "Tfc(bin={:?}, rows={})",
            self.tfc.bin,
            self.tfc.histories.len()
        )
    }
}

/// A KCODE run's estimates: one row per cycle, one column per name in
/// `columns` (`kcode["k_collision"]` gives a column).
#[pyclass(name = "Kcode", module = "aci_tallies.mctal", frozen)]
pub struct PyKcode {
    kcode: Kcode,
}

#[pymethods]
impl PyKcode {
    #[getter]
    fn cycles(&self) -> usize {
        self.kcode.cycles
    }

    /// Inactive (settling) cycles.
    #[getter]
    fn inactive(&self) -> usize {
        self.kcode.inactive
    }

    #[getter]
    fn columns(&self) -> Vec<&'static str> {
        Kcode::COLUMNS[..self.kcode.values.ncols().min(Kcode::COLUMNS.len())].to_vec()
    }

    /// All of it, shape `(cycles, columns)`.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.kcode.values.clone().into_pyarray(py)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let k = Kcode::COLUMNS
            .iter()
            .position(|c| *c == name)
            .filter(|&k| k < self.kcode.values.ncols())
            .ok_or_else(|| PyKeyError::new_err(format!("no column {name:?}")))?;
        Ok(self.kcode.values.column(k).to_owned().into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Kcode(cycles={}, inactive={})",
            self.kcode.cycles, self.kcode.inactive
        )
    }
}

/// One cell or surface bin of a tally, as MCNP makes it from the input:
/// `str(bin)` is how MCNP's output names it (`"(8<7[0 0 0]<3)"`, `"(1 2)"`,
/// `"T"`).
#[pyclass(name = "Bin", module = "aci_tallies.mctal", frozen, eq, hash)]
#[derive(PartialEq, Eq, Hash)]
pub struct PyBin {
    bin: TallyBin,
}

/// A cell or surface of a bin: `(number, facet, lattice element)`.
type Ref = (i64, Option<u8>, Option<[i64; 3]>);

#[pymethods]
impl PyBin {
    /// Whether this is the T bin (the union of the others).
    #[getter]
    fn total(&self) -> bool {
        self.bin == TallyBin::Total
    }

    /// The levels, from the one scored outwards; each a list of `(number,
    /// facet, element)` (several when the level is a union). Empty for the
    /// T bin.
    #[getter]
    fn levels(&self) -> Vec<Vec<Ref>> {
        match &self.bin {
            TallyBin::Levels(levels) => levels
                .iter()
                .map(|l| l.iter().map(|r| (r.number, r.facet, r.element)).collect())
                .collect(),
            TallyBin::Total => Vec::new(),
        }
    }

    fn __str__(&self) -> String {
        self.bin.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Bin({:?})", self.bin.to_string())
    }
}
