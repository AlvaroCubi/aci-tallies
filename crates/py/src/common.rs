//! What the Python classes of every format share.

use std::path::PathBuf;
use std::sync::Arc;

use aci_tallies::common::{Axis, Bins, Input, ReadError, Warning};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use numpy::{PyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyString, PyTuple};

pub fn read_error(e: ReadError) -> PyErr {
    match e {
        ReadError::Io(e) => PyIOError::new_err(e.to_string()),
        e @ ReadError::Changed(_) => PyIOError::new_err(e.to_string()),
        e @ ReadError::NoTally { .. } => PyKeyError::new_err(e.to_string()),
        e @ ReadError::NoBin { .. } => PyIndexError::new_err(e.to_string()),
        e @ ReadError::Unsupported { .. } => PyValueError::new_err(e.to_string()),
    }
}

pub fn io_error(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// `array`, owned by `owner`, as a read-only numpy array without a copy.
///
/// # Safety
/// `array` must live inside `owner` and never change while `owner` lives:
/// the owners here are frozen classes that never hand out mutable access.
pub unsafe fn view<'py, D: Dimension>(
    array: &ArrayBase<OwnedRepr<f64>, D>,
    owner: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, D>>> {
    let a = PyArray::borrow_from_array(array, owner);
    let kwargs = PyDict::new(a.py());
    kwargs.set_item("write", false)?;
    a.call_method("setflags", (), Some(&kwargs))?;
    Ok(a)
}

/// A problem found while reading that didn't stop it: the values it
/// affects are NaN.
#[pyclass(name = "ReadWarning", module = "aci_tallies", frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct PyReadWarning {
    /// The line of the file it is about (from 1), if any.
    #[pyo3(get)]
    line: Option<u64>,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl PyReadWarning {
    fn __str__(&self) -> String {
        Warning {
            line: self.line,
            message: self.message.clone(),
        }
        .to_string()
    }

    fn __repr__(&self) -> String {
        format!("ReadWarning({:?})", self.__str__())
    }
}

pub fn warnings(w: &[Warning]) -> Vec<PyReadWarning> {
    w.iter()
        .map(|w| PyReadWarning {
            line: w.line,
            message: w.message.clone(),
        })
        .collect()
}

/// The bins of one dimension of a tally (energy, time, ...), and whether
/// its last index is MCNP's Total bin.
#[pyclass(name = "Axis", module = "aci_tallies", frozen)]
pub struct PyAxis {
    pub axis: Axis,
}

#[pymethods]
impl PyAxis {
    /// Bin boundaries (`n_bins + 1` of them), or None when only the number
    /// of bins is known.
    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.axis.bounds().map(|b| PyArray1::from_slice(py, b))
    }

    /// The upper bound of each bin (`bounds[1:]` when all bounds are
    /// known; MCTAL files give only these), or None.
    #[getter]
    fn upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.axis.upper().map(|b| PyArray1::from_slice(py, b))
    }

    /// Whether each bin holds everything up to its upper bound (MCNP's `C`
    /// option) rather than only what falls within it.
    #[getter]
    fn cumulative(&self) -> bool {
        self.axis.cumulative
    }

    /// Number of bins, the Total one excluded.
    #[getter]
    fn n_bins(&self) -> usize {
        self.axis.n_bins()
    }

    /// Whether the last index is MCNP's Total bin.
    #[getter]
    fn total(&self) -> bool {
        self.axis.total
    }

    /// Index of the Total bin, or None.
    #[getter]
    fn total_index(&self) -> Option<usize> {
        self.axis.total_index()
    }

    /// Number of indices, the Total one included.
    fn __len__(&self) -> usize {
        self.axis.len()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.axis == other.axis
    }

    fn __repr__(&self) -> String {
        let bins = match &self.axis.bins {
            Bins::Bounds(b) if b.len() <= 6 => format!("bounds={b:?}"),
            Bins::Bounds(b) => format!(
                "bounds=[{:?}, {:?}, ..., {:?}] ({} bins)",
                b[0],
                b[1],
                b[b.len() - 1],
                b.len() - 1
            ),
            Bins::Upper(b) if b.len() <= 6 => format!("upper={b:?}"),
            Bins::Upper(b) => format!(
                "upper=[{:?}, {:?}, ..., {:?}] ({} bins)",
                b[0],
                b[1],
                b[b.len() - 1],
                b.len()
            ),
            Bins::Count(n) => format!("n_bins={n}"),
        };
        format!(
            "Axis({bins}, total={})",
            if self.axis.total { "True" } else { "False" }
        )
    }
}

/// The MCNP input a tally file came from, parsed once (with migjorn) to be
/// used for several files.
#[pyclass(name = "Input", module = "aci_tallies", frozen)]
pub struct PyInput {
    pub input: Arc<Input>,
}

#[pymethods]
impl PyInput {
    /// Read the MCNP input at `path`.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<PyInput> {
        let input = py.allow_threads(|| Input::read(&path)).map_err(io_error)?;
        Ok(PyInput {
            input: Arc::new(input),
        })
    }

    /// The input given as text.
    #[staticmethod]
    fn from_text(text: &str) -> PyInput {
        PyInput {
            input: Arc::new(Input::parse(text)),
        }
    }
}

/// An `input=` argument: a path, an `Input`, or None.
pub fn input_arg(py: Python<'_>, input: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Arc<Input>>> {
    let Some(input) = input.filter(|i| !i.is_none()) else {
        return Ok(None);
    };
    if let Ok(i) = input.downcast::<PyInput>() {
        return Ok(Some(i.get().input.clone()));
    }
    let path: PathBuf = input
        .extract()
        .map_err(|_| PyTypeError::new_err("input must be a path or an aci_tallies.Input"))?;
    let input = py.allow_threads(|| Input::read(&path)).map_err(io_error)?;
    Ok(Some(Arc::new(input)))
}

/// What to keep of an energy or time axis.
pub enum Selection {
    All,
    Bins(std::ops::Range<usize>),
    Total,
}

/// An index selector: None (all), an int, a slice (step 1), or for an
/// energy or time axis the string "total".
pub fn selection(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
    n_bins: usize,
    total_allowed: bool,
) -> PyResult<Selection> {
    let Some(value) = value.filter(|v| !v.is_none()) else {
        return Ok(Selection::All);
    };
    if let Ok(s) = value.downcast::<PyString>() {
        if total_allowed && s.to_str()? == "total" {
            return Ok(Selection::Total);
        }
        return Err(PyValueError::new_err(format!(
            "{name}: {:?} is not a selection (an int, a slice{})",
            s.to_str()?,
            if total_allowed { " or \"total\"" } else { "" }
        )));
    }
    if let Ok(slice) = value.downcast::<PySlice>() {
        let ix = slice.indices(n_bins as isize)?;
        if ix.step != 1 {
            return Err(PyValueError::new_err(format!(
                "{name}: slices take no step"
            )));
        }
        return Ok(Selection::Bins(
            ix.start as usize..ix.stop.max(ix.start) as usize,
        ));
    }
    let i: isize = value.extract().map_err(|_| {
        PyTypeError::new_err(format!("{name}: expected an int, a slice or \"total\""))
    })?;
    let i = if i < 0 { i + n_bins as isize } else { i };
    if i < 0 || i as usize >= n_bins {
        return Err(PyIndexError::new_err(format!(
            "{name}: index {i} out of range for {n_bins} bins"
        )));
    }
    Ok(Selection::Bins(i as usize..i as usize + 1))
}

/// `module.function(slf, *args)`.
pub fn call<'py>(
    slf: &Bound<'py, impl pyo3::PyClass>,
    module: &str,
    function: &str,
    args: impl IntoPyObject<'py, Target = PyTuple, Output = Bound<'py, PyTuple>, Error: Into<PyErr>>,
) -> PyResult<PyObject> {
    let py = slf.py();
    let args = args.into_pyobject(py).map_err(Into::into)?;
    let mut all = vec![slf.clone().into_any()];
    all.extend(args.iter());
    let all = PyTuple::new(py, all)?;
    Ok(py.import(module)?.getattr(function)?.call1(all)?.unbind())
}
