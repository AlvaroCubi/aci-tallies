//! Exact mesh definitions read from the `FMESH` and `TR` cards of an MCNP
//! input.
//!
//! A meshtal file only carries a rounded copy of each mesh (bounds with two
//! decimals, origin and axes with three significant digits) and says
//! nothing about a `TR=` placement. The input has the definition itself:
//! [`read_fmeshes`] rebuilds every FMESH from it the way MCNP does, with fine
//! bins split evenly within each coarse bin, the `TR` card applied, and the
//! energy and time bins MCNP prints in the meshtal file.
//!
//! Checked against MCNP 6.2 and 6.3: defaults (`ORIGIN` 0, `AXS` 0 0 1, `VEC`
//! 1 0 0, one fine bin per coarse bin) and linear fine energy and time bins
//! below a first bound of 0 MeV and -1e36 shakes. Reading the card itself
//! (`=` being optional, the `nR`/`nI`/`xM`/`nLOG` shortcuts) and building
//! a `TRn`/`*TRn` the way MCNP does is migjorn's.
//!
//! Cards pulled in with `READ FILE=` are not followed.
//!
//! The other way, [`FmeshCard::to_input`] writes a card (and its `TRn`) that
//! reads back the same, bounds to the bit, with coarse bins as round as give
//! them ([`coarse_bins`]): MCNP 6.2 scores the pencil-beam fixture's meshes,
//! written this way, exactly as it scores the cards they were read from.

use migjorn::{DataCard, Model};

use crate::{Geometry, Transform};

/// One FMESH card of an MCNP input.
#[derive(Debug, Clone, PartialEq)]
pub struct FmeshCard {
    /// The tally number: `FMESH14` is meshtal tally 14.
    pub id: u32,
    /// The particle designator as written (`n` in `FMESH14:n`).
    pub particle: Option<String>,
    /// `*FMESH`: the tally is energy-weighted (MeV/cm² rather than 1/cm²).
    pub energy_weighted: bool,
    /// The mesh in its own frame, before `TR=`: what the meshtal header
    /// prints, at full precision. A cylinder's `axis` is normalized and its
    /// `vec` made perpendicular to it, as MCNP prints them.
    pub local: Geometry,
    /// `TR=n`: the transform number and the transform it names.
    pub transform: Option<(u32, Transform)>,
    /// Energy bin bounds in MeV, starting at 0, when the card has `EMESH`.
    pub energy_bounds: Option<Vec<f64>>,
    /// Time bin bounds in shakes, starting at -1e36, when it has `TMESH`.
    pub time_bounds: Option<Vec<f64>>,
    /// The `OUT=` value as written, lowercased.
    pub out: Option<String>,
    /// 1-based line of the card in the input.
    pub line: usize,
}

impl FmeshCard {
    /// The mesh where MCNP scores it: [`FmeshCard::local`] moved by `TR=`.
    /// An error only for a cylinder with a reflecting `TR=`, which
    /// [`read_fmeshes`] already rejects.
    pub fn geometry(&self) -> Result<Geometry, crate::GeometryError> {
        match &self.transform {
            Some((_, t)) => self.local.transformed(t),
            None => Ok(self.local.clone()),
        }
    }

    /// The card as MCNP reads it, then its `TRn` card when it has one:
    /// [`read_fmeshes`] gives the card back (`line` aside), its bounds to
    /// the bit (see [`coarse_bins`]). Lines fit in 80 columns.
    ///
    /// The mesh must be one MCNP can be given: a tally number ending in 4,
    /// a valid geometry without a `transform` of its own (placement is
    /// `transform`), a cylinder's r, z and theta bounds from 0 (theta to 1
    /// revolution) and no reflecting `TR=` for it, energy bins from 0 and
    /// time bins from -1e36.
    pub fn to_input(&self) -> Result<String, FmeshError> {
        let fail = |message: String| FmeshError {
            id: self.id,
            line: self.line,
            message,
        };
        if self.id % 10 != 4 {
            return Err(fail("a mesh tally's number ends in 4".into()));
        }
        self.local
            .validate()
            .map_err(|e| fail(format!("invalid mesh: {e}")))?;
        let mut name = String::new();
        if self.energy_weighted {
            name.push('*');
        }
        name.push_str(&format!("FMESH{}", self.id));
        if let Some(p) = &self.particle {
            name.push_str(&format!(":{p}"));
        }
        let mut card = Card::new(name);
        match &self.local {
            Geometry::Rectangular {
                x_bounds,
                y_bounds,
                z_bounds,
                transform,
            } => {
                if transform.is_some() {
                    return Err(fail(
                        "the local mesh has a transform: place it with TR= instead".into(),
                    ));
                }
                card.word("GEOM=XYZ");
                card.list("ORIGIN", &[x_bounds[0], y_bounds[0], z_bounds[0]]);
                card.bins("IMESH", "IINTS", x_bounds);
                card.bins("JMESH", "JINTS", y_bounds);
                card.bins("KMESH", "KINTS", z_bounds);
            }
            Geometry::Cylindrical {
                r_bounds,
                z_bounds,
                theta_bounds,
                origin,
                axis,
                vec,
            } => {
                for (name, b) in [("r", r_bounds), ("z", z_bounds), ("theta", theta_bounds)] {
                    if b[0] != 0.0 {
                        return Err(fail(format!("a cylinder's {name} bounds start at 0")));
                    }
                }
                if theta_bounds.last() != Some(&1.0) {
                    return Err(fail("a cylinder's theta bounds end at 1 revolution".into()));
                }
                if matches!(&self.transform, Some((_, t)) if t.is_reflection()) {
                    return Err(fail("a cylinder can't be placed by a reflecting TR".into()));
                }
                card.word("GEOM=CYL");
                card.list("ORIGIN", origin);
                card.list("AXS", axis);
                card.list("VEC", vec);
                card.bins("IMESH", "IINTS", r_bounds);
                card.bins("JMESH", "JINTS", z_bounds);
                card.bins("KMESH", "KINTS", theta_bounds);
            }
        }
        for (mesh, ints, bounds, lo) in [
            ("EMESH", "EINTS", &self.energy_bounds, 0.0),
            ("TMESH", "TINTS", &self.time_bounds, -1e36),
        ] {
            if let Some(b) = bounds {
                if b.len() < 2 || b[0] != lo {
                    return Err(fail(format!(
                        "{mesh} bounds start at {} and have at least one bin",
                        number(lo)
                    )));
                }
                card.bins(mesh, ints, b);
            }
        }
        if let Some(out) = &self.out {
            card.word(&format!("OUT={}", out.to_ascii_uppercase()));
        }
        if let Some((n, _)) = &self.transform {
            card.word(&format!("TR={n}"));
        }
        let mut text = card.finish();
        if let Some((n, t)) = &self.transform {
            text.push_str(&transform_card(*n, t));
        }
        Ok(text)
    }
}

/// A `TRn` card placing local coordinates by `t`: the displacement, then
/// the cosines of local x', y' and z' with the world axes (the columns of
/// `t.rotation`), as MCNP reads them with m = 1. Lines fit in 80 columns.
pub fn transform_card(n: u32, t: &Transform) -> String {
    let mut card = Card::new(format!("TR{n}"));
    for v in t.translation {
        card.push(number(v));
    }
    for col in 0..3 {
        for row in t.rotation {
            card.push(number(row[col]));
        }
    }
    card.finish()
}

/// Columns of a line [`FmeshCard::to_input`] and [`transform_card`] write.
const WIDTH: usize = 80;

/// A card being written: words in lines of at most [`WIDTH`] columns, the
/// later ones continued after 5 blanks.
struct Card {
    lines: Vec<String>,
}

impl Card {
    fn new(name: String) -> Card {
        Card { lines: vec![name] }
    }

    fn push(&mut self, word: String) {
        let line = self.lines.last_mut().expect("a card has a line");
        if line.len() + 1 + word.len() <= WIDTH {
            line.push(' ');
            line.push_str(&word);
        } else {
            self.lines.push(format!("     {word}"));
        }
    }

    fn word(&mut self, word: &str) {
        self.push(word.into());
    }

    fn list(&mut self, keyword: &str, values: &[f64]) {
        self.word(keyword);
        for &v in values {
            self.push(number(v));
        }
    }

    /// `mesh` with the coarse bounds of `fine` and, unless every coarse
    /// bin is one fine bin, `ints` with their counts.
    fn bins(&mut self, mesh: &str, ints: &str, fine: &[f64]) {
        let (_, coarse, counts) = coarse_bins(fine);
        self.list(mesh, &coarse);
        if counts.iter().any(|&n| n != 1) {
            self.word(ints);
            for n in counts {
                self.push(n.to_string());
            }
        }
    }

    fn finish(self) -> String {
        let mut text = self.lines.join("\n");
        text.push('\n');
        text
    }
}

/// `v` in as few characters as give it back exactly: `0.25`, `1e36`,
/// `-1.5e-7`.
fn number(v: f64) -> String {
    let plain = v.to_string();
    let exp = format!("{v:e}");
    if exp.len() < plain.len() {
        exp
    } else {
        plain
    }
}

/// An FMESH card that could not be read.
#[derive(Debug, Clone, PartialEq)]
pub struct FmeshError {
    pub id: u32,
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for FmeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FMESH{} (line {}): {}", self.id, self.line, self.message)
    }
}

impl std::error::Error for FmeshError {}

/// Every FMESH card of the MCNP input `src`, in file order.
pub fn read_fmeshes(src: &str) -> Vec<Result<FmeshCard, FmeshError>> {
    fmeshes_of(&Model::parse(src))
}

/// Every FMESH card of an already parsed input, in file order.
pub fn fmeshes_of(model: &Model) -> Vec<Result<FmeshCard, FmeshError>> {
    model
        .data_cards()
        .filter(|card| {
            card.mnemonic()
                .is_some_and(|m| m.eq_ignore_ascii_case("fmesh"))
        })
        .filter_map(|card| {
            let id = u32::try_from(card.number()?).ok()?;
            let line = model.line_of(card.slot()).unwrap_or(0);
            Some(
                read_card(model, card, id, line).map_err(|message| FmeshError {
                    id,
                    line,
                    message,
                }),
            )
        })
        .collect()
}

/// Every keyword of an FMESH card (MCNP 6.2/6.3, including the ones this
/// module ignores), so that their values are not taken for keywords; the
/// last eight are D1SUNED's (4.1.3.2), such as the `sppts`/`spmode` of a
/// cell-under-voxel mesh (`out=cuv`).
const KEYWORDS: [&str; 30] = [
    "geom", "origin", "axs", "vec", "imesh", "iints", "jmesh", "jints", "kmesh", "kints", "emesh",
    "eints", "enorm", "tmesh", "tints", "tnorm", "factor", "out", "tr", "inc", "type", "kclear",
    "etag", "ttag", "tally", "spmode", "sppts", "cuvwrt", "fname", "check",
];

fn read_card(model: &Model, card: DataCard, id: u32, line: usize) -> Result<FmeshCard, String> {
    let entries = card.keyword_entries(&KEYWORDS).map_err(|e| e.to_string())?;
    // `keyword_entries` already refused a keyword given twice.
    let find = |keyword: &str| {
        entries
            .iter()
            .find(|e| KEYWORDS[e.keyword] == keyword)
            .map(|e| e.values)
    };
    let numbers = |keyword: &str| -> Result<Option<Vec<f64>>, String> {
        find(keyword)
            .map(|v| {
                v.numbers()
                    .map_err(|why| format!("{}: {why}", keyword.to_ascii_uppercase()))
            })
            .transpose()
    };
    let word = |keyword: &str| -> Result<Option<String>, String> {
        Ok(match find(keyword) {
            None => None,
            Some(v) => match v.words().collect::<Vec<_>>().as_slice() {
                [w] => Some(w.to_ascii_lowercase()),
                _ => return Err(format!("{} takes one value", keyword.to_ascii_uppercase())),
            },
        })
    };
    let triple = |keyword: &str, default: [f64; 3]| -> Result<[f64; 3], String> {
        match numbers(keyword)? {
            None => Ok(default),
            Some(v) => v.try_into().map_err(|v: Vec<f64>| {
                format!(
                    "{} takes 3 values, got {}",
                    keyword.to_ascii_uppercase(),
                    v.len()
                )
            }),
        }
    };
    // Coarse upper bounds and their fine bin counts, split from `lo`.
    let axis = |mesh: &str, ints: &str, lo: f64| -> Result<Option<Vec<f64>>, String> {
        let Some(coarse) = numbers(mesh)? else {
            if find(ints).is_some() {
                return Err(format!(
                    "{} without {}",
                    ints.to_ascii_uppercase(),
                    mesh.to_ascii_uppercase()
                ));
            }
            return Ok(None);
        };
        let counts = numbers(ints)?.unwrap_or_else(|| vec![1.0; coarse.len()]);
        checked_fine_bounds(lo, &coarse, &counts)
            .map(Some)
            .map_err(|why| {
                format!(
                    "{}/{}: {why}",
                    mesh.to_ascii_uppercase(),
                    ints.to_ascii_uppercase()
                )
            })
    };
    let required = |bounds: Option<Vec<f64>>, mesh: &str| {
        bounds.ok_or_else(|| format!("{} is missing", mesh.to_ascii_uppercase()))
    };

    let origin = triple("origin", [0.0; 3])?;
    let local = match word("geom")?.as_deref() {
        None | Some("xyz") | Some("rec") => Geometry::Rectangular {
            x_bounds: required(axis("imesh", "iints", origin[0])?, "imesh")?,
            y_bounds: required(axis("jmesh", "jints", origin[1])?, "jmesh")?,
            z_bounds: required(axis("kmesh", "kints", origin[2])?, "kmesh")?,
            transform: None,
        },
        Some("cyl") | Some("rzt") => {
            let axs = normalize(triple("axs", [0.0, 0.0, 1.0])?).ok_or("AXS is zero")?;
            let vec = triple("vec", [1.0, 0.0, 0.0])?;
            let along = dot(vec, axs);
            let vec = normalize([
                vec[0] - along * axs[0],
                vec[1] - along * axs[1],
                vec[2] - along * axs[2],
            ])
            .ok_or("VEC is parallel to AXS")?;
            let theta_bounds = required(axis("kmesh", "kints", 0.0)?, "kmesh")?;
            if theta_bounds.last() != Some(&1.0) {
                return Err("the last KMESH bound must be 1 revolution".into());
            }
            Geometry::Cylindrical {
                r_bounds: required(axis("imesh", "iints", 0.0)?, "imesh")?,
                z_bounds: required(axis("jmesh", "jints", 0.0)?, "jmesh")?,
                theta_bounds,
                origin,
                axis: axs,
                vec,
            }
        }
        Some(other) => return Err(format!("GEOM={other} is not a mesh MCNP supports")),
    };
    local.validate().map_err(|e| format!("invalid mesh: {e}"))?;

    let transform = match numbers("tr")? {
        None => None,
        Some(v) => {
            let n = match v.as_slice() {
                [n] if n.fract() == 0.0 && *n >= 1.0 => *n as u32,
                _ => return Err("TR takes one transform number".into()),
            };
            let t: Transform = model
                .transform(n.into())
                .ok_or_else(|| format!("TR={n}: no TR{n} card in the input"))?
                .affine()
                .map_err(|e| format!("TR={n}: {e}"))?
                .into();
            local.transformed(&t).map_err(|e| format!("TR={n}: {e}"))?;
            Some((n, t))
        }
    };

    Ok(FmeshCard {
        id,
        particle: card.particle().map(str::to_owned),
        energy_weighted: card.starred(),
        local,
        transform,
        energy_bounds: axis("emesh", "eints", 0.0)?,
        time_bounds: axis("tmesh", "tints", -1e36)?,
        out: word("out")?,
        line,
    })
}

/// Bounds from `lo` through each coarse upper bound, each coarse bin split
/// into `counts` equal fine bins: what MCNP makes of `IMESH`/`IINTS` (and the
/// other axes, `EMESH`/`EINTS`, `TMESH`/`TINTS`). `coarse` and `counts` are
/// taken in pairs; a count of 0 counts as 1.
///
/// This follows MCNP to the bit where it shows: a coarse bin of one fine
/// bin ends exactly at its coarse bound, while in a split one fine bound
/// `k` of `n` is `lo + (hi - lo) * k / n`, the last one included. The
/// difference is a rounding error, except below the -1e36 shakes that time
/// bins start at: MCNP prints `TMESH 10 100 TINTS 2 3` as `-1.00E+36
/// -5.00E+35 0.00E+00 4.00E+01 ...`, with 0 where 10 was asked for.
pub fn fine_bounds(mut lo: f64, coarse: &[f64], counts: &[u32]) -> Vec<f64> {
    let mut out = vec![lo];
    for (&hi, &n) in coarse.iter().zip(counts) {
        if n <= 1 {
            out.push(hi);
        } else {
            out.extend((1..=n as usize).map(|k| lo + (hi - lo) * k as f64 / n as f64));
        }
        lo = hi;
    }
    out
}

/// [`fine_bounds`] of the numbers of a card, which must be fine bin counts.
fn checked_fine_bounds(lo: f64, coarse: &[f64], counts: &[f64]) -> Result<Vec<f64>, String> {
    if coarse.len() != counts.len() {
        return Err(format!(
            "{} coarse bounds but {} fine bin counts",
            coarse.len(),
            counts.len()
        ));
    }
    let counts = counts
        .iter()
        .map(|&n| {
            if n.fract() != 0.0 || n < 1.0 || n > u32::MAX as f64 {
                Err(format!("{n} is not a fine bin count"))
            } else {
                Ok(n as u32)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(fine_bounds(lo, coarse, &counts))
}

/// The coarse bins of fine bounds, the inverse of [`fine_bounds`]: `lo`, the
/// coarse upper bounds and the fine bin count of each, with
/// `fine_bounds(lo, &coarse, &counts) == fine` to the bit.
///
/// Evenly spaced fine bounds are grouped into one coarse bin, as long as
/// MCNP's arithmetic gives them back exactly; any others are coarse bins of
/// one fine bin each. The last fine bound of a split coarse bin is MCNP's
/// arithmetic too, `0.00420000000000087` for a coarse `0.0042`: the coarse
/// bound is the roundest number (fewest significant digits) that gives it.
/// Where an even run can't be one coarse bin (`0 2 4 IINTS 200 200` can't
/// always be `0 4 IINTS 400`), it ends at its roundest bound that can.
/// `fine` must not be empty.
pub fn coarse_bins(fine: &[f64]) -> (f64, Vec<f64>, Vec<u32>) {
    let (mut coarse, mut counts) = (Vec::new(), Vec::new());
    let lo0 = fine.first().copied().unwrap_or(0.0);
    // Where the next coarse bin starts: the last coarse bound, which MCNP
    // splits from (not the fine bound it gave).
    let mut lo = lo0;
    let mut i = 0;
    while i + 1 < fine.len() {
        // The longest run that looks even.
        let step = fine[i + 1] - fine[i];
        let tol = 1e-9 * (fine[i].abs().max(fine[i + 1].abs()) + step.abs());
        let mut last = i + 1;
        while last + 1 < fine.len() && ((fine[last + 1] - fine[last]) - step).abs() <= tol {
            last += 1;
        }
        // The coarse bound ending the run at `j`, if one gives it exactly.
        let ends = |j: usize| {
            let n = (j - i) as u32;
            roundest_near(fine[j], lo)
                .find(|&hi| fine_bounds(lo, &[hi], &[n])[1..] == fine[i + 1..=j])
        };
        // All of it, or else at its roundest bounds first (the longest
        // first among as round ones); a run of one bin always ends exactly.
        let (j, hi) = match ends(last) {
            Some(hi) => (last, hi),
            None => {
                let mut by_roundness: Vec<(usize, usize)> = (i + 1..last)
                    .map(|j| {
                        (
                            digits(roundest_near(fine[j], lo).next().unwrap_or(fine[j])),
                            j,
                        )
                    })
                    .collect();
                by_roundness.sort_by_key(|&(d, j)| (d, std::cmp::Reverse(j)));
                by_roundness
                    .into_iter()
                    .find_map(|(_, j)| Some((j, ends(j)?)))
                    .unwrap_or((i + 1, fine[i + 1]))
            }
        };
        coarse.push(hi);
        counts.push((j - i) as u32);
        lo = hi;
        i = j;
    }
    (lo0, coarse, counts)
}

/// Significant digits of `v` written as briefly as gives it back.
fn digits(v: f64) -> usize {
    let e = format!("{v:e}");
    let mantissa = e.split('e').next().unwrap_or("");
    mantissa.chars().filter(char::is_ascii_digit).count()
}

/// Numbers within a rounding error of `v` (on the scale of `v` and `lo`),
/// roundest first: 0, then 1 to 17 significant digits; `v` itself last.
fn roundest_near(v: f64, lo: f64) -> impl Iterator<Item = f64> {
    let tol = 1e-12 * v.abs().max(lo.abs());
    let zero = (v != 0.0 && v.abs() <= tol).then_some(0.0);
    let decimals = (0..17).filter_map(move |d| {
        let r: f64 = format!("{v:.d$e}").parse().ok()?;
        (r != v && (r - v).abs() <= tol).then_some(r)
    });
    zero.into_iter().chain(decimals).chain(std::iter::once(v))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot(a, a).sqrt();
    (n > 0.0 && n.is_finite()).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const HEAD: &str = "t\n1 0 -1 imp:n=1\n2 0 1 imp:n=0\n\n1 so 100\n\nmode n\n";

    fn one(cards: &str) -> Result<FmeshCard, FmeshError> {
        let mut all = read_fmeshes(&format!("{HEAD}{cards}\n"));
        assert_eq!(all.len(), 1, "{all:?}");
        all.remove(0)
    }

    #[test]
    fn rectangular_with_fine_bins_and_defaults() {
        let c = one("fmesh14:n origin -10 -10 -10 imesh -5 2i 10 iints 2 2r 1\n     jmesh 10 kmesh=10 kints=4 out=ij").unwrap();
        assert_eq!(c.id, 14);
        assert_eq!(c.particle.as_deref(), Some("n"));
        assert!(!c.energy_weighted);
        assert_eq!(c.out.as_deref(), Some("ij"));
        assert_eq!(c.line, 8);
        let (x, y, z) = c.local.axes();
        assert_eq!(x, [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0]);
        assert_eq!(y, [-10.0, 10.0]);
        assert_eq!(z, [-10.0, -5.0, 0.0, 5.0, 10.0]);
        assert_eq!(c.energy_bounds, None);
        assert_eq!(c.geometry(), Ok(c.local.clone()));
    }

    #[test]
    fn energy_and_time_bins_match_what_mcnp_prints() {
        let c =
            one("*fmesh4:p imesh 1 jmesh 1 kmesh 1 emesh 1 10 eints 3 2 tmesh 10 100 tints 2 3")
                .unwrap();
        assert!(c.energy_weighted);
        let e = c.energy_bounds.unwrap();
        assert_eq!(e.len(), 6);
        assert_relative_eq!(e[1], 1.0 / 3.0);
        assert_eq!(e[4..], [5.5, 10.0]);
        // MCNP 6.2/6.3 print: -1.00E+36 -5.00E+35 0.00E+00 4.00E+01 7.00E+01 1.00E+02
        assert_eq!(
            c.time_bounds.unwrap(),
            vec![-1e36, -5e35, 0.0, 40.0, 70.0, 100.0]
        );
        // More MCNP output, one coarse bin each: an unsplit one keeps its
        // bound, a split one ends where its arithmetic lands.
        for (tmesh, at, value) in [("10 100 tints 3 2", 3, 0.0), ("0.125 100", 1, 0.125)] {
            let c = one(&format!("fmesh4:n imesh 1 jmesh 1 kmesh 1 tmesh {tmesh}")).unwrap();
            assert_eq!(c.time_bounds.unwrap()[at], value, "{tmesh}");
        }
    }

    #[test]
    fn cylinder_with_transform() {
        let c = one(
            "fmesh24:n geom=cyl origin 1 2 3 axs 0 0 2 vec 1 0 1\n     imesh 5 jmesh 10 kmesh 0.25 1 kints 1 3 tr 7\n*tr7 1 0 0 90 0 90 90 90 0 0 90 90",
        )
        .unwrap();
        let Geometry::Cylindrical {
            theta_bounds,
            axis,
            vec,
            ..
        } = &c.local
        else {
            panic!("cylinder expected")
        };
        assert_eq!(theta_bounds, &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!((*axis, *vec), ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]));
        let (n, t) = c.transform.unwrap();
        assert_eq!(n, 7);
        // x' = +Y, y' = +Z, z' = +X: the cylinder's axis (local Z) is world X.
        let frame = c.geometry().unwrap().cyl_frame().unwrap();
        for (a, b) in frame.origin.iter().zip([4.0, 1.0, 2.0]) {
            assert_relative_eq!(*a, b, epsilon = 1e-12);
        }
        assert_relative_eq!(frame.e_axis[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(t.rotation[1][0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn transforms_are_built_as_mcnp_builds_them() {
        let world = |tr: &str, geom: &str| {
            one(&format!(
                "fmesh4:n {geom} imesh 1 jmesh 1 kmesh 1 tr 5\n{tr}"
            ))
            .map(|c| c.transform.unwrap().1)
        };
        // Degrees with m = -1: the displacement is the world origin in local
        // coordinates.
        let t = world("*tr5 1 2 3 90 0 90 90 90 0 0 90 90 -1", "").unwrap();
        assert_eq!(
            t.rotation,
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(t.to_local([0.0; 3]), [1.0, 2.0, 3.0]);
        // One vector, two decimals, a reflection: MCNP 6.3 takes all three.
        let one_vector = world("tr5 0 0 0 0.48 0.6 0.64", "").unwrap();
        assert_eq!(one_vector.validate(), Ok(()));
        let rough = world(
            "tr5 0 0 0 0.81 0.51 -0.28 -0.45 0.86 0.25 0.36 -0.07 0.93",
            "",
        );
        assert_eq!(rough.unwrap().validate(), Ok(()));
        let mirror = "tr5 0 0 0 1 0 0 0 1 0 0 0 -1";
        assert!(world(mirror, "").unwrap().is_reflection());
        // A mirrored cylinder has no origin/axis/vec.
        let err = world(mirror, "geom=cyl").unwrap_err().message;
        assert!(err.contains("reverses the theta direction"), "{err}");
    }

    #[test]
    fn bad_cards_say_why() {
        let err = |cards: &str| one(cards).unwrap_err().message;
        assert!(err("fmesh4:n imesh 1 jmesh 1").contains("KMESH is missing"));
        assert!(err("fmesh4:n imesh 1 2 iints 1 jmesh 1 kmesh 1").contains("2 coarse bounds"));
        assert!(err("fmesh4:n imesh 1 jmesh 1 kmesh 1 tr 3").contains("no TR3"));
        assert!(err("fmesh4:n geom=sph imesh 1 jmesh 1 kmesh 1").contains("GEOM=sph"));
        assert!(err("fmesh4:n geom=cyl imesh 1 jmesh 1 kmesh 0.5").contains("must be 1"));
        assert!(err("fmesh4:n imesh 1 0.5 jmesh 1 kmesh 1").contains("invalid mesh"));
        assert!(err("fmesh4:n imesh 1 jmesh 1 kmesh 1 tr 3\ntr3 1 2 3 4").contains("TR=3"));
        assert!(err("fmesh4:n imesh 1 imesh 2 jmesh 1 kmesh 1").contains("twice"));
        assert!(err("fmesh4:n 7 imesh 1 jmesh 1 kmesh 1").contains("before any keyword"));
        assert!(err("fmesh4:n imesh 1 2j jmesh 1 kmesh 1").contains("IMESH"));
        assert!(err("fmesh4:n origin 0 0 imesh 1 jmesh 1 kmesh 1").contains("3 values"));
    }

    #[test]
    fn several_cards_in_file_order() {
        let all = read_fmeshes(&format!(
            "{HEAD}fmesh24:n imesh 1 jmesh 1 kmesh 1\nf4:n 1\nfmesh14:p imesh 1 jmesh 1\nfmesh4:n imesh 1 jmesh 1 kmesh 1\n"
        ));
        let ids: Vec<u32> = all
            .iter()
            .map(|c| match c {
                Ok(c) => c.id,
                Err(e) => e.id,
            })
            .collect();
        assert_eq!(ids, [24, 14, 4]);
        assert!(all[1].is_err());
    }

    /// `c` written and read back: the same card, bounds to the bit. A
    /// cylinder's axes are normalized again, and a transform built again
    /// from its cosines, so those may move by a rounding error.
    fn round_trip(c: &FmeshCard) -> String {
        let text = c.to_input().unwrap();
        for line in text.lines() {
            assert!(line.len() <= WIDTH, "{line}");
        }
        let back = one(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        let close = |a: &[f64], b: &[f64]| {
            assert_eq!(a.len(), b.len(), "{text}");
            for (x, y) in a.iter().zip(b) {
                assert_relative_eq!(*x, *y, epsilon = 1e-15);
            }
        };
        assert_eq!(
            (&back.particle, back.energy_weighted, back.id, &back.out),
            (&c.particle, c.energy_weighted, c.id, &c.out),
            "{text}"
        );
        assert_eq!(back.local.axes(), c.local.axes(), "{text}");
        assert_eq!(back.energy_bounds, c.energy_bounds, "{text}");
        assert_eq!(back.time_bounds, c.time_bounds, "{text}");
        if let (
            Geometry::Cylindrical {
                origin, axis, vec, ..
            },
            Geometry::Cylindrical {
                origin: o,
                axis: a,
                vec: v,
                ..
            },
        ) = (&back.local, &c.local)
        {
            assert_eq!(origin, o, "{text}");
            close(axis, a);
            close(vec, v);
        }
        match (&back.transform, &c.transform) {
            (None, None) => {}
            (Some((n, t)), Some((m, u))) => {
                assert_eq!(n, m);
                close(&t.translation, &u.translation);
                close(t.rotation.as_flattened(), u.rotation.as_flattened());
            }
            _ => panic!("transform lost: {text}"),
        }
        text
    }

    #[test]
    fn written_cards_read_back() {
        for cards in [
            "fmesh14:n origin -10 -10 -10 imesh -5 2i 10 iints 2 2r 1\n     jmesh 10 kmesh=10 kints=4 out=ij",
            "*fmesh4:p imesh 1 jmesh 1 kmesh 1 emesh 1 10 eints 3 2 tmesh 10 100 tints 2 3",
            "fmesh4:n imesh 1 jmesh 1 kmesh 1 tmesh 0.125 100",
            "fmesh24:n geom=cyl origin 1 2 3 axs 0 0 2 vec 1 0 1\n     imesh 5 jmesh 10 kmesh 0.25 1 kints 1 3 tr 7\n*tr7 1 0 0 90 0 90 90 90 0 0 90 90",
            "fmesh34 geom=rzt origin 0.1 0.2 0.3 axs 0.6 0.8 0 vec 0 0 1\n     imesh 1 3 jmesh 7 kmesh 1 kints 12",
            "fmesh44:n imesh 1 jmesh 1 kmesh 1 tr 5\ntr5 0 0 0 1 0 0 0 1 0 0 0 -1",
        ] {
            round_trip(&one(cards).unwrap());
        }
    }

    #[test]
    fn written_cards_look_like_mcnp_input() {
        let c = one("fmesh14:n origin -10 0 0 imesh -5 10 iints 2 3 jmesh 1 kmesh 1e-7 out=ij")
            .unwrap();
        assert_eq!(
            round_trip(&c),
            "FMESH14:n GEOM=XYZ ORIGIN -10 0 0 IMESH -5 10 IINTS 2 3 JMESH 1 KMESH 1e-7\n     OUT=IJ\n"
        );
        let t = one("*fmesh4:p imesh 1 jmesh 1 kmesh 1 tmesh 10 100 tints 2 3").unwrap();
        let words = round_trip(&t)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert!(words.contains("TMESH 0 40 100 TINTS 2 1 2"), "{words}");
        // Many uneven bins wrap (even ones would be one coarse bin); the
        // TR card follows.
        let squares: Vec<String> = (1..40).map(|k| (k * k).to_string()).collect();
        let many = one(&format!(
            "fmesh24:n imesh {}\n     jmesh 1 kmesh 1 tr 3\ntr3 1.5 0 0",
            squares.join(" ")
        ))
        .unwrap();
        let text = round_trip(&many);
        assert!(text.lines().count() > 2, "{text}");
        assert!(text.ends_with("TR3 1.5 0 0 1 0 0 0 1 0 0 0 1\n"), "{text}");
    }

    #[test]
    fn pencil_beam_meshes_read_back() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/pencil_beam_input.mcnp"
        );
        // Some of its meshes are wrong on purpose (MCNP refuses them too).
        let all: Vec<_> = read_fmeshes(&std::fs::read_to_string(path).unwrap())
            .into_iter()
            .filter_map(Result::ok)
            .collect();
        assert!(all.len() >= 4);
        for c in &all {
            round_trip(c);
        }
    }

    #[test]
    fn coarse_bins_give_fine_bounds_back_to_the_bit() {
        // A small LCG: no dependency for random numbers.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let lo = (next() - 0.5) * 10f64.powi((next() * 8.0) as i32 - 3);
            let n = 1 + (next() * 6.0) as usize;
            let mut coarse = Vec::new();
            let mut at = lo;
            for _ in 0..n {
                at += next() * 10f64.powi((next() * 6.0) as i32 - 3) + 1e-6;
                coarse.push(at);
            }
            let counts: Vec<u32> = (0..n).map(|_| 1 + (next() * 9.0) as u32).collect();
            let fine = fine_bounds(lo, &coarse, &counts);
            let (l, c, k) = coarse_bins(&fine);
            assert_eq!(fine_bounds(l, &c, &k), fine, "{lo} {coarse:?} {counts:?}");
            assert!(c.len() <= coarse.len().max(1) * 9);
        }
        // Even bins become one coarse bin; uneven ones stay apart.
        assert_eq!(
            coarse_bins(&[0.0, 1.0, 2.0, 3.0, 5.0]),
            (0.0, vec![3.0, 5.0], vec![3, 1])
        );
        assert_eq!(coarse_bins(&[2.0]), (2.0, vec![], vec![]));
        // Coarse bounds as typed, though MCNP's fine bounds miss them.
        for (lo, coarse, counts) in [
            (-11.1111, vec![0.0042, 13.3333], vec![2, 3]),
            (-0.3, vec![0.0, 0.7], vec![3, 7]),
            (-1e36, vec![0.0, 40.0, 100.0], vec![2, 1, 2]),
        ] {
            let fine = fine_bounds(lo, &coarse, &counts);
            assert_eq!(coarse_bins(&fine), (lo, coarse, counts), "{fine:?}");
        }
        // Even all along, but not one coarse bin to the bit: two again,
        // not one too long and the rest bin by bin.
        for (coarse, counts) in [
            (vec![2.0, 4.0], vec![200, 200]),
            (vec![30.0, 100.0], vec![300, 700]),
        ] {
            let fine = fine_bounds(0.0, &coarse, &counts);
            let (l, c, k) = coarse_bins(&fine);
            assert_eq!(fine_bounds(l, &c, &k), fine);
            assert!(c.len() <= 2, "{c:?} {k:?}");
        }
    }

    #[test]
    fn unwritable_cards_say_why() {
        let err = |c: FmeshCard| c.to_input().unwrap_err().message;
        let rect = one("fmesh14:n imesh 1 jmesh 1 kmesh 1").unwrap();
        assert!(err(FmeshCard {
            id: 15,
            ..rect.clone()
        })
        .contains("ends in 4"));
        let mut placed = rect.clone();
        if let Geometry::Rectangular { transform, .. } = &mut placed.local {
            *transform = Some(Transform::IDENTITY);
        }
        assert!(err(placed).contains("TR="));
        assert!(err(FmeshCard {
            energy_bounds: Some(vec![1.0, 2.0]),
            ..rect.clone()
        })
        .contains("EMESH"));
        let mut cyl = one("fmesh24:n geom=cyl imesh 1 jmesh 1 kmesh 1").unwrap();
        if let Geometry::Cylindrical { z_bounds, .. } = &mut cyl.local {
            z_bounds[0] = -1.0;
        }
        assert!(err(cyl.clone()).contains("z bounds start at 0"));
        let mirror = Transform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            translation: [0.0; 3],
        };
        let cyl = one("fmesh24:n geom=cyl imesh 1 jmesh 1 kmesh 1").unwrap();
        assert!(err(FmeshCard {
            transform: Some((2, mirror)),
            ..cyl
        })
        .contains("reflecting"));
    }
}
