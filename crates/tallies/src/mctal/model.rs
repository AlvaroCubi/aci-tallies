use ndarray::{Array2, ArrayD};

use crate::common::{Axis, Warning};

/// A bin of a tally card, as MCNP makes it from the input (migjorn's).
pub use migjorn::{BinRef, TallyBin};

/// The eight dimensions of an MCTAL tally, in the order of its values (the
/// last varies fastest): cell or surface (`f`), direct or flagged (`d`),
/// user (`u`), segment (`s`), multiplier (`m`), cosine (`c`), energy (`e`)
/// and time (`t`) bins.
pub const DIMS: [&str; 8] = ["f", "d", "u", "s", "m", "c", "e", "t"];

/// Index of each dimension in [`DIMS`].
pub const F: usize = 0;
pub const D: usize = 1;
pub const U: usize = 2;
pub const S: usize = 3;
pub const M: usize = 4;
pub const C: usize = 5;
pub const E: usize = 6;
pub const T: usize = 7;

/// MCNP 6's particles by number (1 = `n`), as designators: the flags line
/// of a tally gives one 0 or 1 for each.
pub const PARTICLES: [char; 37] = [
    'n', 'p', 'e', '|', 'q', 'u', 'v', 'f', 'h', 'l', '+', '-', 'x', 'y', 'o', '!', '<', '>', 'g',
    '/', 'z', 'k', '%', '^', 'b', '_', '~', 'c', 'w', '@', 'd', 't', 's', 'a', '*', '?', '#',
];

/// What the lines before the first tally say about the run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FileHeader {
    /// The code that wrote the file, as it names itself (`mcnp`, `mcnp6`,
    /// `d1suned`); empty when the file leaves it out.
    pub code: String,
    pub version: String,
    /// When the problem ran.
    pub probid: String,
    /// The dump the tallies are from.
    pub dump: u64,
    /// Histories run.
    pub histories: u64,
    /// Pseudorandom numbers used.
    pub randoms: u64,
    /// The problem's title (the input's first line).
    pub title: String,
    /// Tally numbers, in file order (each tally's own line says the same).
    pub tallies: Vec<u32>,
    /// Number of perturbations (`PERT` cards): each tally has their values
    /// after its own.
    pub perturbations: usize,
}

/// A detector tally's kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detector {
    /// Not a detector.
    None,
    Point,
    Ring,
    /// Radiography: pinhole (`FIP`), transmitted rectangular (`FIR`) and
    /// cylindrical (`FIC`) images, whose `s` and `c` bins are the image
    /// grid.
    PinholeImage,
    RectangularImage,
    CylindricalImage,
}

impl Detector {
    /// The detector of an MCTAL tally line's third number.
    pub(crate) fn from_code(code: i64) -> Option<Detector> {
        Some(match code {
            0 => Detector::None,
            1 => Detector::Point,
            2 => Detector::Ring,
            3 => Detector::PinholeImage,
            4 => Detector::RectangularImage,
            5 => Detector::CylindricalImage,
            _ => return None,
        })
    }

    pub fn is_image(self) -> bool {
        matches!(
            self,
            Detector::PinholeImage | Detector::RectangularImage | Detector::CylindricalImage
        )
    }
}

/// `*F` or `+F` on the tally card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modifier {
    None,
    /// `*F`: energy-weighted (`*F4`, `*F6` in jerks, ...).
    Star,
    /// `+F6`: collision heating of every particle.
    Plus,
}

/// A cell or surface number from a tally's `f` line: MCTAL gives one for
/// each bin whose first level is a single cell or surface; 0 for a group,
/// the T bin, or a bin of several cells at its first level. A
/// macrobody facet is `number.facet`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Number {
    pub number: i64,
    pub facet: Option<u8>,
}

/// A legacy `TMESH` mesh tally's grid (MCNPX-style mesh tallies, which
/// MCTAL files carry; FMESH tallies go to meshtal files instead). Its mesh
/// cells are the tally's `f` bins, numbered with the first axis varying
/// fastest (Fortran order).
#[derive(Debug, Clone, PartialEq)]
pub struct MeshGrid {
    /// 1 rectangular, 2 cylindrical, 3 spherical.
    pub geometry: u8,
    /// Bounds along the three axes.
    pub bounds: [Vec<f64>; 3],
}

/// The tally fluctuation chart: how one bin's mean and error evolved.
#[derive(Debug, Clone, PartialEq)]
pub struct Tfc {
    /// The bin, as an index along each of the eight dimensions.
    pub bin: [usize; 8],
    pub histories: Vec<u64>,
    pub mean: Vec<f64>,
    pub rel_error: Vec<f64>,
    /// Figure of merit (absent from files written with the `MCTAL` value
    /// of `PRDMP` negative).
    pub fom: Option<Vec<f64>>,
}

/// What a tally is: everything a [`Tally`] has but the values.
#[derive(Debug, Clone)]
pub struct TallyInfo {
    pub id: u32,
    /// Particles scored, as MCNP designators (`n`, `p`, ...).
    pub particles: Vec<char>,
    pub detector: Detector,
    pub modifier: Modifier,
    /// Its `FC` comment.
    pub comment: Option<String>,
    /// The bins along each of [`DIMS`].
    pub axes: [Axis; 8],
    /// The `f` line's numbers, one per `f` bin; empty when the file gives
    /// none (detectors).
    pub numbers: Vec<Option<Number>>,
    /// What each `f` bin is, from the tally's card in the MCNP input (when
    /// given, and it matches the file): every level of a repeated-structure
    /// bin, groups, the T bin. Empty otherwise.
    pub bins: Vec<TallyBin>,
    /// A `TMESH` mesh tally's grid.
    pub mesh: Option<MeshGrid>,
    pub warnings: Vec<Warning>,
}

impl TallyInfo {
    /// The shape of [`Tally::result`]: the length of each of [`DIMS`],
    /// Total bins included.
    pub fn shape(&self) -> [usize; 8] {
        std::array::from_fn(|k| self.axes[k].len())
    }

    /// Values in the tally.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The MCNP tally type: the last digit of its number (1 current, 2
    /// surface flux, 4 cell flux, 5 detector, 6 heating, 7 fission heating,
    /// 8 pulse height). Meaningless for a `TMESH` mesh tally.
    pub fn kind(&self) -> u32 {
        self.id % 10
    }
}

/// One MCTAL tally: values with relative errors over eight dimensions.
#[derive(Debug, Clone)]
pub struct Tally {
    pub id: u32,
    pub particles: Vec<char>,
    pub detector: Detector,
    pub modifier: Modifier,
    pub comment: Option<String>,
    pub axes: [Axis; 8],
    pub numbers: Vec<Option<Number>>,
    /// See [`TallyInfo::bins`].
    pub bins: Vec<TallyBin>,
    pub mesh: Option<MeshGrid>,
    /// The perturbation these are the values of (its `PERT` number), or
    /// `None` for the unperturbed ones.
    pub perturbation: Option<u32>,
    /// Shape [`TallyInfo::shape`]: eight axes, always, in the order of
    /// [`DIMS`]; an axis with a Total bin has it at its last index.
    pub result: ArrayD<f64>,
    /// Relative errors (a fraction), shaped as `result`.
    pub rel_error: ArrayD<f64>,
    pub tfc: Option<Tfc>,
    pub warnings: Vec<Warning>,
}

impl Tally {
    /// Everything but the values.
    pub fn info(&self) -> TallyInfo {
        TallyInfo {
            id: self.id,
            particles: self.particles.clone(),
            detector: self.detector,
            modifier: self.modifier,
            comment: self.comment.clone(),
            axes: self.axes.clone(),
            numbers: self.numbers.clone(),
            bins: self.bins.clone(),
            mesh: self.mesh.clone(),
            warnings: self.warnings.clone(),
        }
    }

    pub(crate) fn from_info(
        info: TallyInfo,
        perturbation: Option<u32>,
        result: ArrayD<f64>,
        rel_error: ArrayD<f64>,
        tfc: Option<Tfc>,
        warnings: Vec<Warning>,
    ) -> Tally {
        Tally {
            id: info.id,
            particles: info.particles,
            detector: info.detector,
            modifier: info.modifier,
            comment: info.comment,
            axes: info.axes,
            numbers: info.numbers,
            bins: info.bins,
            mesh: info.mesh,
            perturbation,
            result,
            rel_error,
            tfc,
            warnings,
        }
    }
}

/// A `KCODE` run's estimates, one row per cycle; see [`Kcode::COLUMNS`].
#[derive(Debug, Clone, PartialEq)]
pub struct Kcode {
    /// Cycles run.
    pub cycles: usize,
    /// Inactive (settling) cycles.
    pub inactive: usize,
    /// One row per cycle, one column per [`Kcode::COLUMNS`] entry.
    pub values: Array2<f64>,
}

impl Kcode {
    /// What each column is: the
    /// cycle's k estimates and prompt removal lifetimes, then running
    /// averages over the active cycles with their standard deviations, the
    /// combined k when the first (row - 1) cycles are skipped, the number
    /// of source histories, and the figure of merit.
    pub const COLUMNS: [&'static str; 19] = [
        "k_collision",
        "k_absorption",
        "k_track_length",
        "lifetime_collision",
        "lifetime_absorption",
        "average_k_collision",
        "sd_k_collision",
        "average_k_absorption",
        "sd_k_absorption",
        "average_k_track_length",
        "sd_k_track_length",
        "average_k_combined",
        "sd_k_combined",
        "k_combined_skipping",
        "sd_k_combined_skipping",
        "average_lifetime_combined",
        "sd_lifetime_combined",
        "source_histories",
        "figure_of_merit",
    ];
}
