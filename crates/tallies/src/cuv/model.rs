use std::ops::Range;

use ndarray::Array2;

use crate::common::{Axis, Warning};
use crate::meshtal::Geometry;

/// What a CuV mesh is, from its section's header: everything but the
/// pairs.
#[derive(Debug, Clone)]
pub struct CuvInfo {
    pub id: u32,
    /// The particle (`neutron`); a meshinfo doesn't say.
    pub particle: Option<String>,
    /// The tally's `FC` comment; only the cuvmsh has it.
    pub comment: Option<String>,
    pub geometry: Geometry,
    /// Whether `geometry` (and the energy bounds) came from the tally's
    /// `FMESH` card in the MCNP input (exact, `TR=` applied) rather than
    /// from the file's header, which rounds them and leaves out `TR=`.
    pub geometry_from_input: bool,
    /// The energy bins of the spectra (MeV, no Total bin); `None` for a
    /// meshinfo. A mesh without `EMESH` has one bin, from 0 to 1e36.
    pub energy: Option<Axis>,
    /// Voxels in the file: `geometry.n_voxels()` unless a warning says
    /// otherwise.
    pub n_voxels: usize,
    /// Whether the header was D1SUNED 3.x's for a cylinder (theta in
    /// radians, voxel volumes 2π too large), corrected when read.
    pub legacy_cylindrical: bool,
    /// Problems with the section as a whole; a read adds those of the pairs.
    pub warnings: Vec<Warning>,
}

/// The (voxel, cell) pairs of a run of voxels, in file order: for each
/// voxel, its cells.
#[derive(Debug, Clone, PartialEq)]
pub struct Pairs {
    /// The voxels read (flat indices, as [`Geometry::index`] numbers them).
    pub voxels: Range<usize>,
    /// Each voxel's volume (cm³) as the file gives it, one per voxel of
    /// `voxels`.
    pub voxel_volume: Vec<f64>,
    /// Each pair's voxel.
    pub voxel: Vec<u32>,
    pub cell: Vec<u32>,
    /// The share of its voxel's volume the cell takes.
    pub fraction: Vec<f64>,
}

impl Pairs {
    pub fn len(&self) -> usize {
        self.cell.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cell.is_empty()
    }

    /// The volume of each pair (cm³): its fraction of its voxel's.
    pub fn volume(&self) -> Vec<f64> {
        let v0 = self.voxels.start;
        self.voxel
            .iter()
            .zip(&self.fraction)
            .map(|(&v, f)| f * self.voxel_volume[v as usize - v0])
            .collect()
    }
}

/// What a meshinfo says of a run of voxels: each pair's material and
/// density.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    pub id: u32,
    pub pairs: Pairs,
    /// Material number of each pair (0 for void).
    pub material: Vec<u32>,
    /// Density of each pair (g/cm³, written with 4 decimals).
    pub density: Vec<f64>,
    pub warnings: Vec<Warning>,
}

/// What a cuvmsh says of a run of voxels: each pair's spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectra {
    pub id: u32,
    pub pairs: Pairs,
    /// Flux per energy bin, shape `[pair, energy]`, lowest energy first:
    /// the cell's track length in the voxel over its volume there, per
    /// source particle (1/cm²).
    pub flux: Array2<f64>,
    /// Relative errors (a fraction), shaped as `flux`.
    pub rel_error: Array2<f64>,
    /// Each pair's flux over all energies, and its relative error.
    pub total: Vec<f64>,
    pub total_error: Vec<f64>,
    pub warnings: Vec<Warning>,
}

/// A run of voxels of a CuV mesh, from its meshinfo and its cuvmsh: the
/// same pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct CuvPart {
    pub decomposition: Decomposition,
    pub spectra: Spectra,
}
