//! Readers for MCNP tally output files.
//!
//! - [`meshtal`]: FMESH mesh tallies (`meshtal` files).
//! - [`mctal`]: `F` tallies (MCTAL files).
//! - [`cuv`]: D1SUNED cell-under-voxel meshes (`cuvmsh` and `meshinfo`).
//!
//! Every reader indexes a file in one parallel pass and then reads what is
//! asked for (a tally, or part of one) from disk, so files much larger than
//! memory can be read by parts.

pub mod common;
pub mod cuv;
pub mod mctal;
pub mod meshtal;

pub use common::{Axis, Bins, ReadError, Warning};

/// The formats this crate reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Meshtal,
    Mctal,
    Cuvmsh,
    Meshinfo,
}

/// The format of a file from its first bytes (a few kB are enough), or
/// `None` when it is neither.
pub fn detect(head: &[u8]) -> Option<Format> {
    if mctal::is_mctal(head) {
        Some(Format::Mctal)
    } else if cuv::is_cuvmsh(head) {
        Some(Format::Cuvmsh)
    } else if cuv::is_meshinfo(head) {
        Some(Format::Meshinfo)
    } else if meshtal::is_meshtal(head) {
        Some(Format::Meshtal)
    } else {
        None
    }
}
