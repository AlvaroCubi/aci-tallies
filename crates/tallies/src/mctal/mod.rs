//! MCNP MCTAL files: the `F` tallies (cells, surfaces, detectors), with
//! their eight bin dimensions, fluctuation charts, perturbations and KCODE
//! estimates.
//!
//! ```no_run
//! use aci_tallies::mctal::{Reader, DIMS};
//!
//! let file = Reader::open("mctal")?;
//! for info in file.tallies() {
//!     println!("tally {}: {:?} over {DIMS:?}", info.id, info.shape());
//! }
//! let tally = file.read(4)?; // all of it
//! let some = file.read_f(4, 0..10)?; // only its first ten cells
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! What MCTAL can't say: a tally's `f` line gives a cell or surface number
//! only for bins whose first level is a single one (`0` for a group, the T
//! bin, or several cells at the first level), and nothing of the levels
//! below (`(8 < 7[2 0 0] < 3)` is just `8`). The format is checked against
//! MCNP 6.2 and 6.3 output.

mod header;
mod model;
pub mod ops;
mod reader;
mod vals;

pub use model::{
    BinRef, Detector, FileHeader, Kcode, MeshGrid, Modifier, Number, Tally, TallyBin, TallyInfo,
    Tfc, C, D, DIMS, E, F, M, PARTICLES, S, T, U,
};
pub use ops::{OpsError, Select};
pub use reader::{is_mctal, Reader};

/// Every tally of the MCTAL file at `path` (see [`Reader`]).
pub fn read(path: impl AsRef<std::path::Path>) -> Result<Vec<Tally>, crate::ReadError> {
    Reader::open(path)?.read_all()
}
