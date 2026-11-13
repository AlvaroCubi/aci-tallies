use ndarray::{Array3, Array5};

pub use super::header::FileHeader;
use crate::common::{Axis, Warning};

/// The spatial mesh a tally was scored on (shared with other tools through
/// the `aci-tallies-geom` crate).
pub use aci_tallies_geom::{Geometry, Transform};

/// Which `out=` layout a tally's values were printed in. It never changes
/// how they are modeled, only how they were read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutFormat {
    /// `out=col` (the default): one row per voxel, `X Y Z Result Rel Error`,
    /// with `Energy`/`Time` columns first when the mesh has those bins.
    Col,
    /// `out=cf`: `col` plus `Volume` and `Rslt * Vol` columns.
    Cf,
    /// `out=ij`: matrices of X across by Y down, one per Z bin (for a
    /// cylinder, R by Z per Theta bin: the same roles, in (R, Z, Theta)).
    Ij,
    /// `out=ik`: X by Z, one per Y bin.
    Ik,
    /// `out=jk`: Y by Z, one per X bin.
    Jk,
    /// Not a meshtal section: a D1SUNED cell-under-voxel mesh (`out=cuv`)
    /// summed over the cells of each voxel ([`crate::cuv`]).
    Cuv,
}

impl OutFormat {
    /// The name `out=` takes.
    pub fn name(self) -> &'static str {
        match self {
            OutFormat::Col => "col",
            OutFormat::Cf => "cf",
            OutFormat::Ij => "ij",
            OutFormat::Ik => "ik",
            OutFormat::Jk => "jk",
            OutFormat::Cuv => "cuv",
        }
    }
}

/// What a tally is, from its section's header: everything a [`Tally`] has
/// but the values.
#[derive(Debug, Clone)]
pub struct TallyInfo {
    pub id: u32,
    pub particle: String,
    /// Its `FC` comment, if any.
    pub comment: Option<String>,
    pub geometry: Geometry,
    /// Whether `geometry` and the energy and time bounds came from the
    /// tally's `FMESH` card in the MCNP input (exact, `TR=` applied) rather
    /// than from the meshtal header, which rounds them and leaves out `TR=`.
    pub geometry_from_input: bool,
    /// Whether the tally is a `*FMESH` (energy-weighted, MeV/cm² rather than
    /// 1/cm²). Only the input says: `None` without one.
    pub energy_weighted: Option<bool>,
    /// Energy bins (MeV). A mesh without `EMESH` has one, from 0 to 1e36.
    pub energy: Axis,
    /// Time bins (shakes). A mesh without `TMESH` has one.
    pub time: Axis,
    pub out_format: OutFormat,
    /// Problems with the section as a whole (a header that doesn't match
    /// the input, missing rows or blocks); a read adds those of the values.
    pub warnings: Vec<Warning>,
}

impl TallyInfo {
    /// The shape of [`Tally::result`]: `[energy, time, i, j, k]`, Total
    /// indices included.
    pub fn shape(&self) -> [usize; 5] {
        let (i, j, k) = self.geometry.dims();
        [self.energy.len(), self.time.len(), i, j, k]
    }
}

/// One mesh tally: its values and what they are.
#[derive(Debug, Clone)]
pub struct Tally {
    pub id: u32,
    pub particle: String,
    pub comment: Option<String>,
    pub geometry: Geometry,
    /// See [`TallyInfo::geometry_from_input`].
    pub geometry_from_input: bool,
    /// See [`TallyInfo::energy_weighted`].
    pub energy_weighted: Option<bool>,
    pub energy: Axis,
    pub time: Axis,
    pub out_format: OutFormat,
    /// Shape `[energy, time, i, j, k]`, always: a mesh without energy or
    /// time bins has length 1 there. Each of the first two axes has one
    /// more index at its end when MCNP printed a Total bin for it.
    pub result: Array5<f64>,
    /// Relative errors (a fraction, not a percent), shaped as `result`.
    pub rel_error: Array5<f64>,
    /// Voxel volumes (cm³, shape `[i, j, k]`) when the file gives them
    /// (`out=cf`); otherwise [`Geometry::voxel_volumes`] has them.
    pub volume: Option<Array3<f64>>,
    pub warnings: Vec<Warning>,
}

impl Tally {
    /// Everything but the values.
    pub fn info(&self) -> TallyInfo {
        TallyInfo {
            id: self.id,
            particle: self.particle.clone(),
            comment: self.comment.clone(),
            geometry: self.geometry.clone(),
            geometry_from_input: self.geometry_from_input,
            energy_weighted: self.energy_weighted,
            energy: self.energy.clone(),
            time: self.time.clone(),
            out_format: self.out_format,
            warnings: self.warnings.clone(),
        }
    }

    pub(crate) fn from_info(
        info: TallyInfo,
        result: Array5<f64>,
        rel_error: Array5<f64>,
        volume: Option<Array3<f64>>,
        warnings: Vec<Warning>,
    ) -> Tally {
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
            result,
            rel_error,
            volume,
            warnings,
        }
    }
}

/// One energy bin and time bin of a tally: a value per voxel.
#[derive(Debug, Clone)]
pub struct TallyBin {
    pub id: u32,
    /// The energy and time indices read (Total ones included, as in
    /// [`Tally::result`]).
    pub energy: usize,
    pub time: usize,
    /// Shape `[i, j, k]`.
    pub result: Array3<f64>,
    pub rel_error: Array3<f64>,
    /// When the file gives volumes (`out=cf`).
    pub volume: Option<Array3<f64>>,
    pub warnings: Vec<Warning>,
}
