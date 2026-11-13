//! MCNP meshtal files: the FMESH mesh tallies.
//!
//! ```no_run
//! use aci_tallies::meshtal::Reader;
//!
//! let file = Reader::open("meshtal")?;
//! for info in file.tallies() {
//!     println!("tally {}: {:?}", info.id, info.shape());
//! }
//! let tally = file.read(14)?; // every energy and time bin
//! let bin = file.read_bin(14, 0, 0)?; // only the first energy and time bin
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Every `out=` layout (`col`, `cf`, `ij`, `ik`, `jk`) is read, for
//! rectangular and cylindrical meshes, with energy and time bins and their
//! Total bins, as checked against MCNP 6.2 and 6.3 output.

mod columnar;
pub(crate) mod header;
pub(crate) mod input;
mod matrix;
pub(crate) mod model;
pub mod ops;
mod reader;

pub use model::{FileHeader, Geometry, OutFormat, Tally, TallyBin, TallyInfo, Transform};
pub use ops::{OpsError, Select};
pub use reader::{is_meshtal, Reader};

/// Every tally of the meshtal file at `path` (see [`Reader`]).
pub fn read(path: impl AsRef<std::path::Path>) -> Result<Vec<Tally>, crate::ReadError> {
    Reader::open(path)?.read_all()
}
