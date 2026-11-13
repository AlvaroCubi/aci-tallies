//! D1SUNED cell-under-voxel (CuV) meshes: `meshinfo` and `cuvmsh` files.
//!
//! An FMESH with `out=cuv` is not written to the meshtal file. D1SUNED
//! writes instead, for each of its voxels, the cells in it:
//!
//! - `meshinfo` ([`Meshinfo`]): each (voxel, cell) pair's material,
//!   density and share of the voxel's volume, and the voxel's volume.
//! - `cuvmsh` ([`Cuvmsh`]): the same pairs, each with its flux spectrum and
//!   relative errors (the cell's track length in the voxel over its volume
//!   there), and its total flux.
//!
//! Both list the pairs in the same order, void cells included, every mesh
//! of the run in one file. [`CuvMesh`] reads the two together.
//!
//! ```no_run
//! use aci_tallies::cuv::{CuvMesh, Cuvmsh};
//!
//! let file = Cuvmsh::open("cuvmsh")?;
//! let spectra = file.read(14)?; // every pair of mesh 14
//! let first = file.read_voxels(14, 0..1000)?; // or a run of voxels
//! let mesh = file.to_mesh(14)?; // voxel fluxes, as a meshtal tally
//!
//! let both = CuvMesh::open("cuvmsh", "meshinfo", None)?;
//! let part = both.read_voxels(14, 0..1000)?; // materials, densities and spectra
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The headers print the geometry with 2 decimals, the cylinder's frame and
//! the energy bounds with 3 significant digits, and leave out `TR=` (the
//! printed geometry and voxel centres are local; D1SUNED applies `TR=` when
//! it finds the cells). With the MCNP input (`open_with_input`), the
//! geometry and energy bounds are the `FMESH` card's, exact and placed.
//!
//! D1SUNED 4.1.3.2 crashed writing a cylindrical CuV mesh with 2 energy
//! bins ("output statement overflows record"); with 20 it ran.

mod header;
mod join;
mod model;
mod reader;

use std::io;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use ndarray::Array2;

pub use join::CuvMesh;
pub use model::{CuvInfo, CuvPart, Decomposition, Pairs, Spectra};

use self::header::FileKind;
use self::reader::Reader;
use crate::common::source::Source;
use crate::common::{scan, Input, ReadError};
use crate::meshtal::{FileHeader, Tally};

/// Whether `head` (the first few kB of a file) looks like a cuvmsh.
pub fn is_cuvmsh(head: &[u8]) -> bool {
    memchr::memmem::find(head, b"cell-under-voxel mesh tally").is_some()
}

/// Whether `head` (the first few kB of a file) looks like a meshinfo.
pub fn is_meshinfo(head: &[u8]) -> bool {
    memchr::memmem::find(head, b"Mesh tally number:").is_some()
        && memchr::memmem::find(head, b"Average cell number per voxel").is_some()
}

macro_rules! common_api {
    () => {
        /// The file's path (`None` when read from memory).
        pub fn path(&self) -> Option<&Path> {
            self.0.path()
        }

        /// Whether the file still has the size and modification time it
        /// had when indexed (reads fail with [`ReadError::Changed`]
        /// otherwise).
        pub fn is_current(&self) -> bool {
            self.0.is_current()
        }

        /// What the file says about the run.
        pub fn header(&self) -> &FileHeader {
            self.0.header()
        }

        /// The meshes that can be read, in file order.
        pub fn tallies(&self) -> impl Iterator<Item = &CuvInfo> {
            self.0.tallies()
        }

        /// The sections that can't be read: tally number (0 when even that
        /// is unreadable) and why.
        pub fn failed(&self) -> impl Iterator<Item = (u32, &str)> {
            self.0.failed()
        }

        /// Mesh `id`.
        pub fn tally(&self, id: u32) -> Result<&CuvInfo, ReadError> {
            self.0.tally(id)
        }
    };
}

/// An indexed `meshinfo` file.
pub struct Meshinfo(Reader);

impl Meshinfo {
    /// Index the meshinfo at `path`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Meshinfo> {
        Reader::open(FileKind::Meshinfo, path.as_ref(), None).map(Meshinfo)
    }

    /// [`Meshinfo::open`], each mesh's geometry from its `FMESH` card in
    /// `input` (exact, `TR=` applied). A mesh whose card is missing or
    /// doesn't match keeps the header's, with a warning.
    pub fn open_with_input(path: impl AsRef<Path>, input: &Input) -> io::Result<Meshinfo> {
        Reader::open(FileKind::Meshinfo, path.as_ref(), Some(input)).map(Meshinfo)
    }

    /// A meshinfo held in memory.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, input: Option<&Input>) -> Meshinfo {
        Meshinfo::build(Source::bytes(bytes), input, scan::CHUNK_BYTES)
    }

    pub(crate) fn build(source: Source, input: Option<&Input>, chunk: usize) -> Meshinfo {
        Meshinfo(
            Reader::build(FileKind::Meshinfo, source, input, chunk)
                .expect("reading memory doesn't fail"),
        )
    }

    common_api!();

    /// Every pair of mesh `id`.
    pub fn read(&self, id: u32) -> Result<Decomposition, ReadError> {
        let n = self.0.tally(id)?.n_voxels;
        self.read_voxels(id, 0..n)
    }

    /// The pairs of voxels `voxels` of mesh `id`, reading only their part
    /// of the file.
    pub fn read_voxels(&self, id: u32, voxels: Range<usize>) -> Result<Decomposition, ReadError> {
        let mut p = self.0.read_voxels(id, voxels.clone())?;
        Ok(Decomposition {
            id,
            pairs: p.take_pairs(voxels),
            material: p.material,
            density: p.density,
            warnings: p.warnings,
        })
    }
}

/// An indexed `cuvmsh` file.
pub struct Cuvmsh(Reader);

impl Cuvmsh {
    /// Index the cuvmsh at `path`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Cuvmsh> {
        Reader::open(FileKind::Cuvmsh, path.as_ref(), None).map(Cuvmsh)
    }

    /// [`Cuvmsh::open`], each mesh's geometry and energy bounds from its
    /// `FMESH` card in `input` (exact, `TR=` applied). A mesh whose card
    /// is missing or doesn't match keeps the header's, with a warning.
    pub fn open_with_input(path: impl AsRef<Path>, input: &Input) -> io::Result<Cuvmsh> {
        Reader::open(FileKind::Cuvmsh, path.as_ref(), Some(input)).map(Cuvmsh)
    }

    /// A cuvmsh held in memory.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, input: Option<&Input>) -> Cuvmsh {
        Cuvmsh::build(Source::bytes(bytes), input, scan::CHUNK_BYTES)
    }

    pub(crate) fn build(source: Source, input: Option<&Input>, chunk: usize) -> Cuvmsh {
        Cuvmsh(
            Reader::build(FileKind::Cuvmsh, source, input, chunk)
                .expect("reading memory doesn't fail"),
        )
    }

    common_api!();

    /// Every pair of mesh `id`.
    pub fn read(&self, id: u32) -> Result<Spectra, ReadError> {
        let n = self.0.tally(id)?.n_voxels;
        self.read_voxels(id, 0..n)
    }

    /// The pairs of voxels `voxels` of mesh `id`, reading only their part
    /// of the file.
    pub fn read_voxels(&self, id: u32, voxels: Range<usize>) -> Result<Spectra, ReadError> {
        let n_energy = self.0.tally(id)?.energy.as_ref().map_or(0, |e| e.n_bins());
        let mut p = self.0.read_voxels(id, voxels.clone())?;
        let pairs = p.take_pairs(voxels);
        let shape = (pairs.len(), n_energy);
        let array =
            |v: Vec<f64>| Array2::from_shape_vec(shape, v).expect("n_energy values per pair");
        Ok(Spectra {
            id,
            pairs,
            flux: array(p.flux),
            rel_error: array(p.rel_error),
            total: p.total,
            total_error: p.total_error,
            warnings: p.warnings,
        })
    }

    /// Mesh `id` as a meshtal tally: each voxel's flux, the sum over its
    /// cells of their flux times their share of its volume (see
    /// [`Spectra::voxel_flux`]).
    pub fn to_mesh(&self, id: u32) -> Result<Tally, ReadError> {
        let info = self.0.tally(id)?.clone();
        Ok(self.read(id)?.voxel_flux(&info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/d1suned");
        std::fs::read(dir.join(name)).unwrap()
    }

    /// Whatever the chunks the file is cut into, every read is the same.
    #[test]
    fn reads_do_not_depend_on_chunks() {
        let (c, m) = (fixture("cuvmsh"), fixture("meshinfo"));
        let reference = Cuvmsh::from_bytes(c.clone(), None);
        let reference_m = Meshinfo::from_bytes(m.clone(), None);
        for chunk in [1, 7, 64, 333, 1000, 4096] {
            let f = Cuvmsh::build(Source::bytes(c.clone()), None, chunk);
            let g = Meshinfo::build(Source::bytes(m.clone()), None, chunk);
            for id in [4, 14, 44] {
                let n = f.tally(id).unwrap().n_voxels;
                assert_eq!(
                    f.read(id).unwrap(),
                    reference.read(id).unwrap(),
                    "chunk {chunk}"
                );
                assert_eq!(g.read(id).unwrap(), reference_m.read(id).unwrap());
                let mid = f.read_voxels(id, 1..n - 1).unwrap();
                assert_eq!(mid.pairs.voxels, 1..n - 1);
                assert_eq!(mid.pairs.voxel_volume.len(), n - 2);
            }
        }
    }

    /// Precision warnings are counted over the whole read, once, however
    /// many segments it spans.
    #[test]
    fn warnings_are_counted_over_a_read() {
        let text = String::from_utf8(fixture("meshinfo")).unwrap();
        let low = text.replace("    7.8700        1", "    0.0050        1");
        let in_14 = text
            .split("Mesh tally number:")
            .nth(2)
            .unwrap()
            .matches("    7.8700        1")
            .count();
        assert!(in_14 > 2);
        for chunk in [64, 4096, scan::CHUNK_BYTES] {
            let f = Meshinfo::build(Source::bytes(low.clone().into_bytes()), None, chunk);
            let w = f.read(14).unwrap().warnings;
            let low: Vec<_> = w
                .iter()
                .filter(|w| w.message.contains("below 0.01"))
                .collect();
            assert_eq!(low.len(), 1, "chunk {chunk}: {w:?}");
            let count = format!("{in_14} (voxel, cell) pairs");
            assert!(low[0].message.starts_with(&count), "{}", low[0]);
        }
        let ok = Meshinfo::from_bytes(text.into_bytes(), None);
        assert!(ok.read(14).unwrap().warnings.is_empty());
    }
}
