//! The section headers of D1SUNED's `meshinfo` and `cuvmsh`.
//!
//! Both start a mesh with its number and the `Tally bin boundaries:` of the
//! FMESH as a meshtal file prints them (as seen in D1SUNED 4.1.3.2 output):
//!
//! ```text
//!  Mesh tally number:         4                   (meshinfo)
//!  Average cell number per voxel:  1.25000
//!
//!  Tally bin boundaries:
//!     X direction:     -5.00     -4.00 ...        (10 wide, 2 decimals)
//!     ...
//!
//!  Mesh Tally Number         4                    (cuvmsh)
//!      <FC comment, if any>
//!  This is a neutron cell-under-voxel mesh tally.
//!
//!  Tally bin boundaries:
//!     ...
//!     Energy bin number:  709
//!     Energy bin boundaries: 0.00E+00 1.05E-11 ...   (MeV, 3 digits)
//!
//!    INDEX         X         Y         Z    Volume   ncel
//! ```
//!
//! A cylinder replaces X/Y/Z with an `origin at … axis in … direction, VEC
//! direction …` line (`0.00E+00`, 10 wide) and R / Z / `Theta direction
//! (revolutions):` lines. D1SUNED 3.x's meshinfo writes theta in radians
//! despite the label, 10 wide with 8 decimals, which runs the numbers
//! together, and voxel volumes 2π too large; it is told by a last theta
//! bound above 1, and both are corrected. 4.x writes revolutions.

use crate::common::Warning;
use crate::meshtal::header::SectionHeader;
use crate::meshtal::Geometry;

/// Which of the two files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileKind {
    Meshinfo,
    Cuvmsh,
}

impl FileKind {
    pub fn name(self) -> &'static str {
        match self {
            FileKind::Meshinfo => "meshinfo",
            FileKind::Cuvmsh => "cuvmsh",
        }
    }

    /// The tally number of a line that starts a section.
    pub fn section_id(self, line: &str) -> Option<Result<u32, ()>> {
        let label = match self {
            FileKind::Meshinfo => "Mesh tally number:",
            FileKind::Cuvmsh => "Mesh Tally Number",
        };
        let rest = line.trim_start().strip_prefix(label)?;
        Some(
            rest.split_whitespace()
                .next()
                .and_then(|w| w.parse().ok())
                .ok_or(()),
        )
    }
}

/// A parsed section header.
pub(crate) struct CuvHeader {
    /// Geometry and energy bounds as printed (theta in revolutions); no
    /// time bins. A meshinfo has no energy bounds.
    pub header: SectionHeader,
    pub legacy_cylindrical: bool,
    /// Lines of the section before its first voxel line.
    pub lines: usize,
    pub warnings: Vec<Warning>,
}

/// Parse the header of a section from its first `lines` (those that belong
/// to it). `Err(None)` when the header doesn't end within them.
pub(crate) fn parse(kind: FileKind, lines: &[&str]) -> Result<CuvHeader, Option<String>> {
    let first = lines.first().ok_or(None)?;
    let id = match kind.section_id(first) {
        Some(Ok(id)) => id,
        _ => return Err(Some(format!("bad section start {first:?}"))),
    };
    let mut comment = None;
    let mut particle = None;
    let mut warnings = Vec::new();
    let (mut x, mut y, mut z, mut r, mut theta) = (None, None, None, None, None);
    let mut frame = None;
    let mut n_energy: Option<usize> = None;
    let mut energy = None;
    let mut in_bounds = false;
    let mut end = None;
    for (n, line) in lines.iter().enumerate().skip(1) {
        let t = line.trim();
        let bounds = |values: &str| {
            fixed_floats(values, 10).ok_or_else(|| Some(format!("bad bin boundaries {t:?}")))
        };
        if t.is_empty() {
            continue;
        }
        if !in_bounds {
            if t.starts_with("Tally bin boundaries:") {
                in_bounds = true;
            } else if let Some(rest) = t.strip_prefix("This is a") {
                particle = Some(
                    rest.split("cell-under-voxel")
                        .next()
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                );
            } else if kind == FileKind::Cuvmsh && particle.is_none() && comment.is_none() {
                comment = Some(t.to_string());
            }
            continue;
        }
        if t.starts_with("origin at") {
            frame =
                Some(parse_origin_line(t).ok_or_else(|| Some(format!("bad origin line {t:?}")))?);
        } else if t.starts_with("INDEX") {
            end = Some(n + 1);
            break;
        } else if let Some((label, values)) = line.split_once(':') {
            match label.trim() {
                "X direction" => x = Some(bounds(values)?),
                "Y direction" => y = Some(bounds(values)?),
                "Z direction" => z = Some(bounds(values)?),
                "R direction" => r = Some(bounds(values)?),
                l if l.starts_with("Theta direction") => theta = Some(bounds(values)?),
                "Energy bin number" => {
                    n_energy = Some(
                        values
                            .trim()
                            .parse()
                            .map_err(|_| Some(format!("bad energy bin number {t:?}")))?,
                    )
                }
                "Energy bin boundaries" => {
                    energy = Some(
                        values
                            .split_whitespace()
                            .map(|w| w.parse::<f64>().ok())
                            .collect::<Option<Vec<f64>>>()
                            .ok_or_else(|| Some(format!("bad energy bounds {t:?}")))?,
                    )
                }
                _ => {}
            }
        } else if kind == FileKind::Meshinfo {
            // The first voxel line.
            end = Some(n);
            break;
        }
    }
    let lines_before_data = end.ok_or(None)?;

    let mut legacy_cylindrical = false;
    let geometry = match (frame, r, theta) {
        (Some((origin, axis, vec)), Some(r), Some(mut theta)) => {
            let z = z.ok_or_else(|| Some("cylindrical mesh without Z direction".to_string()))?;
            if theta.last().copied().unwrap_or(0.0) > 1.0 + 1e-6 {
                legacy_cylindrical = true;
                for t in &mut theta {
                    *t /= std::f64::consts::TAU;
                    // 8 printed decimals of 2π leave 1.0000000001.
                    if (*t - t.round()).abs() < 1e-6 {
                        *t = t.round();
                    }
                }
                warnings.push(Warning::new(
                    "cylindrical theta bounds written in radians: converted to revolutions, \
                     and voxel volumes divided by 2π (D1SUNED 3.x)",
                ));
            }
            Geometry::Cylindrical {
                r_bounds: r,
                z_bounds: z,
                theta_bounds: theta,
                origin,
                axis,
                vec,
            }
        }
        (None, None, None) => Geometry::Rectangular {
            x_bounds: x.ok_or_else(|| Some("missing X direction".to_string()))?,
            y_bounds: y.ok_or_else(|| Some("missing Y direction".to_string()))?,
            z_bounds: z.ok_or_else(|| Some("missing Z direction".to_string()))?,
            transform: None,
        },
        _ => return Err(Some("incomplete cylindrical mesh header".into())),
    };

    let energy_bounds = match kind {
        FileKind::Meshinfo => Vec::new(),
        FileKind::Cuvmsh => {
            let bounds = energy.ok_or_else(|| Some("no energy bin boundaries".to_string()))?;
            let n = n_energy.ok_or_else(|| Some("no energy bin number".to_string()))?;
            if bounds.len() != n + 1 {
                return Err(Some(format!(
                    "{} energy bounds for {n} energy bins",
                    bounds.len()
                )));
            }
            if bounds.windows(2).any(|w| w[0] >= w[1]) {
                return Err(Some("energy bounds are not increasing".into()));
            }
            bounds
        }
    };
    if kind == FileKind::Cuvmsh && particle.is_none() {
        return Err(Some(
            "no \"This is a ... cell-under-voxel mesh tally.\" line".into(),
        ));
    }
    Ok(CuvHeader {
        header: SectionHeader {
            id,
            particle: particle.unwrap_or_default(),
            comment,
            geometry,
            energy_bounds,
            time_bounds: vec![0.0, 1e36],
        },
        legacy_cylindrical,
        lines: lines_before_data,
        warnings,
    })
}

/// Floats in fixed-width fields of `width` characters (Fortran `fw.d`), so
/// that values running into each other still split; whitespace-separated
/// when the text is not a whole number of fields.
pub(crate) fn fixed_floats(s: &str, width: usize) -> Option<Vec<f64>> {
    let s = s.trim_end();
    let bytes = s.as_bytes();
    let v: Option<Vec<f64>> = if bytes.len().is_multiple_of(width) {
        bytes
            .chunks(width)
            .map(|c| std::str::from_utf8(c).ok()?.trim().parse().ok())
            .collect()
    } else {
        s.split_whitespace().map(|w| w.parse().ok()).collect()
    };
    v.filter(|v| !v.is_empty())
}

type Frame = ([f64; 3], [f64; 3], [f64; 3]);

/// `origin at x y z axis in a b c direction, VEC direction u v w`.
fn parse_origin_line(line: &str) -> Option<Frame> {
    let three = |s: &str| -> Option<[f64; 3]> {
        let v: Vec<f64> = s
            .split_whitespace()
            .take(3)
            .map(|w| w.parse().ok())
            .collect::<Option<_>>()?;
        (v.len() == 3).then(|| [v[0], v[1], v[2]])
    };
    let rest = line.split_once("origin at")?.1;
    let (origin, rest) = rest.split_once("axis in")?;
    let (axis, rest) = rest.split_once("direction, VEC direction")?;
    Some((three(origin)?, three(axis)?, three(rest)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_splits_merged_values() {
        assert_eq!(
            fixed_floats("0.000000002.50000000", 10),
            Some(vec![0.0, 2.5])
        );
        assert_eq!(
            fixed_floats("   0.00000   1.00000", 10),
            Some(vec![0.0, 1.0])
        );
        assert_eq!(
            fixed_floats("  -1000.00-999999.99", 10),
            Some(vec![-1000.0, -999999.99])
        );
        assert_eq!(
            fixed_floats("     0.000     1.000   ", 10),
            Some(vec![0.0, 1.0])
        );
        assert_eq!(fixed_floats("  ", 10), None);
    }

    #[test]
    fn origin_line() {
        let l = "origin at   0.00E+00  0.00E+00  1.50E+01 axis in   0.00E+00  0.00E+00  1.00E+00 direction, VEC direction   1.00E+00  0.00E+00  0.00E+00";
        assert_eq!(
            parse_origin_line(l),
            Some(([0.0, 0.0, 15.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
        );
    }
}
