//! `aci_tallies.cuv`: D1SUNED cell-under-voxel files, read by parts.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use aci_tallies::common::Input;
use aci_tallies::cuv::{CuvInfo, CuvMesh, Cuvmsh, Meshinfo, Pairs, Spectra};
use aci_tallies::meshtal::FileHeader;
use aci_tallies::ReadError;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};

use crate::common::{call, input_arg, io_error, read_error, warnings, PyAxis, PyReadWarning};
use crate::geometry::{geometry, PyGeometry};
use crate::meshtal::PyMeshTally;

/// The files a `CuvFile` reads: a cuvmsh, a meshinfo, or both (one per
/// file, behind an `Arc`: their sizes don't matter).
#[allow(clippy::large_enum_variant)]
enum Files {
    Cuvmsh(Cuvmsh),
    Meshinfo(Meshinfo),
    Both(CuvMesh),
}

impl Files {
    fn header(&self) -> &FileHeader {
        match self {
            Files::Cuvmsh(f) => f.header(),
            Files::Meshinfo(f) => f.header(),
            Files::Both(b) => b.cuvmsh.header(),
        }
    }

    fn path(&self) -> Option<PathBuf> {
        match self {
            Files::Cuvmsh(f) => f.path(),
            Files::Meshinfo(f) => f.path(),
            Files::Both(b) => b.cuvmsh.path(),
        }
        .map(|p| p.to_path_buf())
    }

    fn tallies(&self) -> Vec<&CuvInfo> {
        match self {
            Files::Cuvmsh(f) => f.tallies().collect(),
            Files::Meshinfo(f) => f.tallies().collect(),
            Files::Both(b) => b.tallies().collect(),
        }
    }

    fn failed(&self) -> Vec<(u32, String)> {
        let mut out = Vec::new();
        let mut add = |it: &mut dyn Iterator<Item = (u32, &str)>| {
            out.extend(it.map(|(i, w)| (i, w.to_string())))
        };
        match self {
            Files::Cuvmsh(f) => add(&mut f.failed()),
            Files::Meshinfo(f) => add(&mut f.failed()),
            Files::Both(b) => {
                add(&mut b.cuvmsh.failed());
                add(&mut b.meshinfo.failed());
            }
        }
        out
    }

    fn tally(&self, id: u32) -> Result<&CuvInfo, ReadError> {
        match self {
            Files::Cuvmsh(f) => f.tally(id),
            Files::Meshinfo(f) => f.tally(id),
            Files::Both(b) => b.tally(id),
        }
    }

    fn is_current(&self) -> bool {
        match self {
            Files::Cuvmsh(f) => f.is_current(),
            Files::Meshinfo(f) => f.is_current(),
            Files::Both(b) => b.cuvmsh.is_current() && b.meshinfo.is_current(),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Files::Cuvmsh(_) => "cuvmsh",
            Files::Meshinfo(_) => "meshinfo",
            Files::Both(_) => "cuvmsh+meshinfo",
        }
    }

    fn read(&self, id: u32, voxels: Range<usize>) -> Result<Read, ReadError> {
        Ok(match self {
            Files::Cuvmsh(f) => Read {
                spectra: Some(f.read_voxels(id, voxels)?),
                decomposition: None,
            },
            Files::Meshinfo(f) => Read {
                spectra: None,
                decomposition: Some(f.read_voxels(id, voxels)?),
            },
            Files::Both(b) => {
                let p = b.read_voxels(id, voxels)?;
                Read {
                    spectra: Some(p.spectra),
                    decomposition: Some(p.decomposition),
                }
            }
        })
    }
}

struct Read {
    spectra: Option<Spectra>,
    decomposition: Option<aci_tallies::cuv::Decomposition>,
}

/// An indexed cuvmsh, meshinfo, or both: every CuV mesh of a D1SUNED run.
/// Opening reads where each mesh and its header are (one parallel pass);
/// pairs are read when asked for, a mesh or a run of voxels at a time.
#[pyclass(name = "CuvFile", module = "aci_tallies.cuv", frozen, mapping)]
pub struct PyCuvFile {
    files: Arc<Files>,
}

fn open_with(
    py: Python<'_>,
    input: Option<&Bound<'_, PyAny>>,
    open: impl FnOnce(Option<&Input>) -> std::io::Result<Files> + Send,
) -> PyResult<PyCuvFile> {
    let input = input_arg(py, input)?;
    let files = py
        .allow_threads(|| open(input.as_deref()))
        .map_err(io_error)?;
    Ok(PyCuvFile {
        files: Arc::new(files),
    })
}

/// Open the cuvmsh at `path` (each pair's spectrum). With `input` (the MCNP
/// input of the run, a path or an `aci_tallies.Input`), each mesh's
/// geometry and energy bounds come from its FMESH card: exact, with `TR=`.
#[pyfunction]
#[pyo3(signature = (path, input = None))]
pub fn open_cuvmsh(
    py: Python<'_>,
    path: PathBuf,
    input: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyCuvFile> {
    open_with(py, input, |i| {
        match i {
            Some(i) => Cuvmsh::open_with_input(&path, i),
            None => Cuvmsh::open(&path),
        }
        .map(Files::Cuvmsh)
    })
}

/// Open the meshinfo at `path` (each pair's material and density); `input`
/// as for `open_cuvmsh`.
#[pyfunction]
#[pyo3(signature = (path, input = None))]
pub fn open_meshinfo(
    py: Python<'_>,
    path: PathBuf,
    input: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyCuvFile> {
    open_with(py, input, |i| {
        match i {
            Some(i) => Meshinfo::open_with_input(&path, i),
            None => Meshinfo::open(&path),
        }
        .map(Files::Meshinfo)
    })
}

/// Open a run's cuvmsh and meshinfo together; reads check that both list
/// the same pairs and give each pair's material, density and spectrum.
#[pyfunction]
#[pyo3(signature = (cuvmsh, meshinfo, input = None))]
pub fn open_cuv_mesh(
    py: Python<'_>,
    cuvmsh: PathBuf,
    meshinfo: PathBuf,
    input: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyCuvFile> {
    open_with(py, input, |i| {
        CuvMesh::open(&cuvmsh, &meshinfo, i).map(Files::Both)
    })
}

/// The voxels a `voxels=` argument names: all of them, a slice or an index.
fn voxel_range(voxels: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Range<usize>> {
    let Some(v) = voxels else {
        return Ok(0..n);
    };
    if let Ok(s) = v.downcast::<PySlice>() {
        let i = s.indices(n as isize)?;
        if i.step != 1 {
            return Err(PyValueError::new_err("voxels: only a step of 1"));
        }
        let (a, b) = (i.start as usize, (i.stop.max(i.start)) as usize);
        return Ok(a..b);
    }
    let i: isize = v.extract()?;
    let i = if i < 0 { i + n as isize } else { i };
    if i < 0 || i as usize >= n {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "voxel {i} out of range for {n} voxels"
        )));
    }
    Ok(i as usize..i as usize + 1)
}

#[pymethods]
impl PyCuvFile {
    /// The (cuvmsh's) file path.
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.files.path()
    }

    /// `"cuvmsh"`, `"meshinfo"` or `"cuvmsh+meshinfo"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.files.kind()
    }

    #[getter]
    fn title(&self) -> &str {
        &self.files.header().title
    }

    /// The number of histories the spectra are normalized to (a cuvmsh's).
    #[getter]
    fn histories(&self) -> Option<f64> {
        self.files.header().histories
    }

    /// The code that wrote the file and its version, as printed.
    #[getter]
    fn code(&self) -> (String, String) {
        let h = self.files.header();
        (h.code.clone(), h.version.clone())
    }

    #[getter]
    fn probid(&self) -> &str {
        &self.files.header().probid
    }

    /// The numbers of the meshes that can be read, in file order.
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.files.tallies().iter().map(|t| t.id).collect()
    }

    /// What each mesh is (no pairs read).
    #[getter]
    fn tallies(&self, py: Python<'_>) -> PyResult<Vec<PyCuvInfo>> {
        self.files
            .tallies()
            .into_iter()
            .map(|t| PyCuvInfo::new(py, t.clone()))
            .collect()
    }

    /// The sections that can't be read: `(mesh number, why)`.
    #[getter]
    fn failed(&self) -> Vec<(u32, String)> {
        self.files.failed()
    }

    /// What mesh `id` is, without reading its pairs.
    fn info(&self, py: Python<'_>, id: u32) -> PyResult<PyCuvInfo> {
        PyCuvInfo::new(py, self.files.tally(id).map_err(read_error)?.clone())
    }

    /// Whether the files are unchanged since they were opened.
    fn is_current(&self) -> bool {
        self.files.is_current()
    }

    /// The pairs of mesh `id`: all of them, or of `voxels` (a slice or an
    /// index of the mesh's voxels, in flat order), reading only their part
    /// of the file.
    #[pyo3(signature = (id, voxels = None))]
    fn read(
        &self,
        py: Python<'_>,
        id: u32,
        voxels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCuvPairs> {
        let info = self.files.tally(id).map_err(read_error)?.clone();
        let range = voxel_range(voxels, info.n_voxels)?;
        let files = &self.files;
        let read = py
            .allow_threads(|| files.read(id, range))
            .map_err(read_error)?;
        PyCuvPairs::new(py, info, read)
    }

    /// Mesh `id` as a `MeshTally` (needs the cuvmsh): each voxel's flux,
    /// the sum of its cells' fluxes times their shares of its volume, what
    /// an FMESH without `out=cuv` scores. Errors are combined as if the
    /// cells were independent (they aren't): an estimate, usually low.
    fn to_mesh(&self, py: Python<'_>, id: u32) -> PyResult<PyMeshTally> {
        let files = &self.files;
        let tally = py
            .allow_threads(|| -> Result<_, ReadError> {
                let cuvmsh = match &**files {
                    Files::Cuvmsh(f) => f,
                    Files::Both(b) => &b.cuvmsh,
                    Files::Meshinfo(_) => return Ok(None),
                };
                cuvmsh.to_mesh(id).map(Some)
            })
            .map_err(read_error)?
            .ok_or_else(|| PyValueError::new_err("to_mesh needs the cuvmsh (spectra)"))?;
        PyMeshTally::new(py, tally)
    }

    fn __len__(&self) -> usize {
        self.files.tallies().len()
    }

    fn __contains__(&self, id: u32) -> bool {
        self.files.tally(id).is_ok()
    }

    fn __getitem__(&self, py: Python<'_>, id: u32) -> PyResult<PyCuvPairs> {
        self.read(py, id, None)
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
            .files
            .path()
            .map_or("<memory>".to_string(), |p| p.display().to_string());
        format!(
            "CuvFile({path:?}, kind={:?}, meshes={:?})",
            self.files.kind(),
            self.ids()
        )
    }

    fn _repr_html_(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._html", "cuv_file", ())
    }
}

fn energy_bins(info: &CuvInfo) -> String {
    info.energy
        .as_ref()
        .map_or("None".into(), |e| e.n_bins().to_string())
}

/// What a CuV mesh is, without its pairs.
#[pyclass(name = "CuvInfo", module = "aci_tallies.cuv", frozen)]
pub struct PyCuvInfo {
    info: CuvInfo,
    #[pyo3(get)]
    geometry: Py<PyGeometry>,
    /// The spectra's energy bins (MeV, no Total bin); None for a meshinfo.
    #[pyo3(get)]
    energy: Option<Py<PyAxis>>,
}

impl PyCuvInfo {
    fn new(py: Python<'_>, info: CuvInfo) -> PyResult<PyCuvInfo> {
        Ok(PyCuvInfo {
            geometry: geometry(py, &info.geometry)?,
            energy: info
                .energy
                .clone()
                .map(|axis| Py::new(py, PyAxis { axis }))
                .transpose()?,
            info,
        })
    }
}

#[pymethods]
impl PyCuvInfo {
    #[getter]
    fn id(&self) -> u32 {
        self.info.id
    }

    /// `neutron`, ...; None for a meshinfo.
    #[getter]
    fn particle(&self) -> Option<&str> {
        self.info.particle.as_deref()
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.info.comment.as_deref()
    }

    #[getter]
    fn geometry_from_input(&self) -> bool {
        self.info.geometry_from_input
    }

    #[getter]
    fn n_voxels(&self) -> usize {
        self.info.n_voxels
    }

    /// Whether the header was D1SUNED 3.x's for a cylinder (theta in
    /// radians, volumes 2π too large), corrected when read.
    #[getter]
    fn legacy_cylindrical(&self) -> bool {
        self.info.legacy_cylindrical
    }

    #[getter]
    fn warnings(&self) -> Vec<PyReadWarning> {
        warnings(&self.info.warnings)
    }

    fn __repr__(&self) -> String {
        format!(
            "CuvInfo(id={}, voxels={}, energy_bins={})",
            self.info.id,
            self.info.n_voxels,
            energy_bins(&self.info)
        )
    }
}

/// The (voxel, cell) pairs of a run of voxels of a CuV mesh, one array
/// element per pair (`voxel_volume`: per voxel). Arrays are read-only and
/// hold what Rust read, not copies. A cuvmsh gives `flux`, `rel_error`
/// (shape `(pair, energy)`), `total` and `total_error`; a meshinfo
/// `material` and `density`; the others are None.
#[pyclass(name = "CuvPairs", module = "aci_tallies.cuv", frozen)]
pub struct PyCuvPairs {
    info: CuvInfo,
    voxels: Range<usize>,
    n_pairs: usize,
    warnings: Vec<aci_tallies::Warning>,
    #[pyo3(get)]
    geometry: Py<PyGeometry>,
    #[pyo3(get)]
    energy: Option<Py<PyAxis>>,
    /// Each pair's voxel (flat index, as `Geometry.locate` numbers them).
    #[pyo3(get)]
    voxel: Py<PyArray1<u32>>,
    #[pyo3(get)]
    cell: Py<PyArray1<u32>>,
    /// The share of its voxel's volume each pair takes.
    #[pyo3(get)]
    fraction: Py<PyArray1<f64>>,
    /// Each pair's volume (cm³): its fraction of its voxel's.
    #[pyo3(get)]
    volume: Py<PyArray1<f64>>,
    /// Each voxel's volume (cm³), for the voxels read.
    #[pyo3(get)]
    voxel_volume: Py<PyArray1<f64>>,
    /// Material number (0: void).
    #[pyo3(get)]
    material: Option<Py<PyArray1<u32>>>,
    /// g/cm³, printed with 4 decimals.
    #[pyo3(get)]
    density: Option<Py<PyArray1<f64>>>,
    /// Flux per energy bin, per cm² and source particle.
    #[pyo3(get)]
    flux: Option<Py<PyArray2<f64>>>,
    #[pyo3(get)]
    rel_error: Option<Py<PyArray2<f64>>>,
    #[pyo3(get)]
    total: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    total_error: Option<Py<PyArray1<f64>>>,
}

fn frozen<'py, T: numpy::Element, D: ndarray::Dimension>(
    a: Bound<'py, numpy::PyArray<T, D>>,
) -> PyResult<Py<numpy::PyArray<T, D>>> {
    let kwargs = PyDict::new(a.py());
    kwargs.set_item("write", false)?;
    a.call_method("setflags", (), Some(&kwargs))?;
    Ok(a.unbind())
}

impl PyCuvPairs {
    fn new(py: Python<'_>, info: CuvInfo, read: Read) -> PyResult<PyCuvPairs> {
        let pairs: &Pairs = match (&read.spectra, &read.decomposition) {
            (Some(s), _) => &s.pairs,
            (None, Some(d)) => &d.pairs,
            (None, None) => unreachable!("a read reads something"),
        };
        let mut notes = Vec::new();
        for w in read
            .spectra
            .iter()
            .flat_map(|s| &s.warnings)
            .chain(read.decomposition.iter().flat_map(|d| &d.warnings))
        {
            if !notes.contains(w) {
                notes.push(w.clone());
            }
        }
        let voxels = pairs.voxels.clone();
        let volume = pairs.volume();
        let volume_len = volume.len();
        let one = |v: Vec<f64>| frozen(v.into_pyarray(py));
        let ints = |v: Vec<u32>| frozen(v.into_pyarray(py));
        let p = pairs.clone();
        let (flux, rel_error, total, total_error) = match read.spectra {
            Some(s) => (
                Some(frozen(s.flux.into_pyarray(py))?),
                Some(frozen(s.rel_error.into_pyarray(py))?),
                Some(one(s.total)?),
                Some(one(s.total_error)?),
            ),
            None => (None, None, None, None),
        };
        let (material, density) = match read.decomposition {
            Some(d) => (Some(ints(d.material)?), Some(one(d.density)?)),
            None => (None, None),
        };
        Ok(PyCuvPairs {
            geometry: geometry(py, &info.geometry)?,
            energy: info
                .energy
                .clone()
                .map(|axis| Py::new(py, PyAxis { axis }))
                .transpose()?,
            voxel: ints(p.voxel)?,
            cell: ints(p.cell)?,
            fraction: one(p.fraction)?,
            volume: one(volume)?,
            voxel_volume: one(p.voxel_volume)?,
            material,
            density,
            flux,
            rel_error,
            total,
            total_error,
            n_pairs: volume_len,
            info,
            voxels,
            warnings: notes,
        })
    }
}

#[pymethods]
impl PyCuvPairs {
    #[getter]
    fn id(&self) -> u32 {
        self.info.id
    }

    #[getter]
    fn particle(&self) -> Option<&str> {
        self.info.particle.as_deref()
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.info.comment.as_deref()
    }

    #[getter]
    fn geometry_from_input(&self) -> bool {
        self.info.geometry_from_input
    }

    /// The voxels read, `(start, stop)`.
    #[getter]
    fn voxels(&self) -> (usize, usize) {
        (self.voxels.start, self.voxels.end)
    }

    /// The mesh's voxel count (all of them, not only those read).
    #[getter]
    fn n_voxels(&self) -> usize {
        self.info.n_voxels
    }

    #[getter]
    fn warnings(&self) -> Vec<PyReadWarning> {
        warnings(&self.warnings)
    }

    fn info(&self, py: Python<'_>) -> PyResult<PyCuvInfo> {
        PyCuvInfo::new(py, self.info.clone())
    }

    fn __len__(&self) -> usize {
        self.n_pairs
    }

    /// One row per pair, or with `spectra=True` one per (pair, energy bin):
    /// see `aci_tallies.cuv` for the columns. Needs polars.
    #[pyo3(signature = (spectra = false))]
    fn to_polars(slf: &Bound<'_, Self>, spectra: bool) -> PyResult<PyObject> {
        call(slf, "aci_tallies._polars", "cuv_frame", (spectra,))
    }

    /// Write `to_polars(spectra)` to a Parquet file. Needs polars.
    #[pyo3(signature = (path, spectra = false))]
    fn to_parquet(slf: &Bound<'_, Self>, path: PathBuf, spectra: bool) -> PyResult<PyObject> {
        call(slf, "aci_tallies._polars", "cuv_parquet", (path, spectra))
    }

    /// An `xarray.Dataset` on dimensions `(pair, energy)`. Needs xarray.
    fn to_xarray(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._xarray", "cuv_dataset", ())
    }

    fn __repr__(&self) -> String {
        format!(
            "CuvPairs(id={}, voxels=({}, {}), pairs={}, energy_bins={})",
            self.info.id,
            self.voxels.start,
            self.voxels.end,
            self.n_pairs,
            energy_bins(&self.info)
        )
    }

    fn _repr_html_(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        call(slf, "aci_tallies._html", "cuv_pairs", ())
    }
}
