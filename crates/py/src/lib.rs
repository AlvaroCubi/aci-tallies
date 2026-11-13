//! The Python extension module `aci_tallies._native`; the package around
//! it (`python/aci_tallies`) is what users import.

mod common;
mod cuv;
mod geometry;
mod mctal;
mod meshtal;

use pyo3::prelude::*;

/// The format of the file at `path`, from its first bytes: `"meshtal"`,
/// `"mctal"`, `"cuvmsh"` or `"meshinfo"`, or None when it isn't one this
/// package reads.
#[pyfunction]
fn detect(path: std::path::PathBuf) -> PyResult<Option<&'static str>> {
    use std::io::Read;
    let mut head = Vec::with_capacity(64 << 10);
    std::fs::File::open(&path)
        .and_then(|f| f.take(64 << 10).read_to_end(&mut head))
        .map_err(common::io_error)?;
    Ok(aci_tallies::detect(&head).map(|f| match f {
        aci_tallies::Format::Meshtal => "meshtal",
        aci_tallies::Format::Mctal => "mctal",
        aci_tallies::Format::Cuvmsh => "cuvmsh",
        aci_tallies::Format::Meshinfo => "meshinfo",
    }))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(meshtal::open_meshtal, m)?)?;
    m.add_function(wrap_pyfunction!(mctal::open_mctal, m)?)?;
    m.add_function(wrap_pyfunction!(cuv::open_cuvmsh, m)?)?;
    m.add_function(wrap_pyfunction!(cuv::open_meshinfo, m)?)?;
    m.add_function(wrap_pyfunction!(cuv::open_cuv_mesh, m)?)?;
    m.add_class::<cuv::PyCuvFile>()?;
    m.add_class::<cuv::PyCuvInfo>()?;
    m.add_class::<cuv::PyCuvPairs>()?;
    m.add_class::<common::PyAxis>()?;
    m.add_class::<common::PyReadWarning>()?;
    m.add_class::<common::PyInput>()?;
    m.add_class::<geometry::PyGeometry>()?;
    m.add_class::<meshtal::PyMeshtalFile>()?;
    m.add_class::<meshtal::PyTallyInfo>()?;
    m.add_class::<meshtal::PyMeshTally>()?;
    m.add_class::<mctal::PyMctalFile>()?;
    // Named TallyInfo and Tally in aci_tallies.mctal, like meshtal's.
    m.add("MctalInfo", m.py().get_type::<mctal::PyMctalInfo>())?;
    m.add("MctalTally", m.py().get_type::<mctal::PyMctalTally>())?;
    m.add_class::<mctal::PyTfc>()?;
    m.add_class::<mctal::PyBin>()?;
    m.add_class::<mctal::PyKcode>()?;
    Ok(())
}
