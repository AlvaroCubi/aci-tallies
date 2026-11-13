//! Indexing and reading `meshinfo` and `cuvmsh` files.
//!
//! Opening a file indexes it in one parallel pass ([`crate::common::scan`],
//! sections start at their `Mesh Tally Number` line) and parses each
//! section's header. The first read of a mesh finds its voxels: one more
//! parallel pass over the mesh's part of the file counts the voxel lines in
//! each chunk, which cuts it into segments of whole voxels. A read of a run
//! of voxels parses only the segments holding them, in parallel.
//!
//! Voxels are numbered in file order, which is the geometry's flat order
//! (the inner axis fastest); the index printed on each voxel line is not
//! used, as it overflows its field above 9,999,999 voxels. Lines are told
//! apart by their shape:
//!
//! ```text
//! meshinfo  voxel   `      1   1.60000E+01    2`             i7,es12.5,i3
//!           cell    `        101    7.9300        1  1.00000` cell, density, material, fraction
//! cuvmsh    voxel   `      1   -50.000   -50.000    10.000 2.00000E+05   5`   index, centre, volume, ncel
//!           cell    `    407667 0.61293 7.53493E-12 1.80822E-02`   cell, fraction, total flux, its error
//!                   `         6 0.72400`              (a mesh with one energy bin)
//!           values  each cell's flux per energy bin, then their relative errors,
//!                   6 per line, each list on new lines
//! ```

use std::io;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use super::header::{self, CuvHeader, FileKind};
use super::model::{CuvInfo, Pairs};
use crate::common::scan::{self, Chunk};
use crate::common::source::{Opened, Source};
use crate::common::{float, Axis, Input, ReadError, Warning};
use crate::meshtal::header::read_file_header;
use crate::meshtal::input::{self, InputMeshes};
use crate::meshtal::{FileHeader, Geometry};

/// An indexed meshinfo or cuvmsh file.
pub(crate) struct Reader {
    kind: FileKind,
    source: Source,
    header: FileHeader,
    chunks: Vec<Chunk>,
    sections: Vec<Section>,
}

struct Section {
    id: u32,
    body: Result<Body, String>,
}

struct Body {
    info: CuvInfo,
    /// The geometry as the header prints it (local, theta in revolutions),
    /// which the voxel centres are checked against.
    printed: Geometry,
    /// The voxel lines and what follows them (bytes).
    data: Range<u64>,
    /// Lines of the file before `data`.
    data_line: u64,
    /// Energy bins of a cuvmsh (0 for a meshinfo).
    n_energy: usize,
    segments: OnceLock<Arc<Vec<Segment>>>,
}

/// Whole voxels, `voxels`, in bytes `bytes`.
#[derive(Debug, Clone)]
struct Segment {
    bytes: Range<u64>,
    voxels: Range<usize>,
    /// Lines of the file before it.
    line: u64,
}

/// What a read of a run of voxels found.
#[derive(Default)]
pub(crate) struct Parsed {
    pub voxel_volume: Vec<f64>,
    pub voxel: Vec<u32>,
    pub cell: Vec<u32>,
    pub fraction: Vec<f64>,
    // meshinfo
    pub material: Vec<u32>,
    pub density: Vec<f64>,
    // cuvmsh: per pair, `n_energy` values each
    pub flux: Vec<f64>,
    pub rel_error: Vec<f64>,
    pub total: Vec<f64>,
    pub total_error: Vec<f64>,
    pub warnings: Vec<Warning>,
    /// Pairs with a density in (0, 0.01) g/cm³.
    low_density: usize,
    /// The largest relative difference of a voxel's volume from the
    /// geometry's.
    worst_volume: f64,
}

impl Parsed {
    pub fn take_pairs(&mut self, voxels: Range<usize>) -> Pairs {
        Pairs {
            voxels,
            voxel_volume: std::mem::take(&mut self.voxel_volume),
            voxel: std::mem::take(&mut self.voxel),
            cell: std::mem::take(&mut self.cell),
            fraction: std::mem::take(&mut self.fraction),
        }
    }

    fn append(&mut self, o: Parsed) {
        self.voxel_volume.extend(o.voxel_volume);
        self.voxel.extend(o.voxel);
        self.cell.extend(o.cell);
        self.fraction.extend(o.fraction);
        self.material.extend(o.material);
        self.density.extend(o.density);
        self.flux.extend(o.flux);
        self.rel_error.extend(o.rel_error);
        self.total.extend(o.total);
        self.total_error.extend(o.total_error);
        self.warnings.extend(o.warnings);
        self.low_density += o.low_density;
        self.worst_volume = self.worst_volume.max(o.worst_volume);
    }
}

impl Reader {
    pub fn open(kind: FileKind, path: &Path, input: Option<&Input>) -> io::Result<Reader> {
        Reader::build(kind, Source::file(path)?, input, scan::CHUNK_BYTES)
    }

    pub(crate) fn build(
        kind: FileKind,
        source: Source,
        input: Option<&Input>,
        chunk: usize,
    ) -> io::Result<Reader> {
        let meshes = input.map(InputMeshes::of);
        let opened = source.open()?;
        let index = scan::index(&opened, chunk, |line| {
            let t = std::str::from_utf8(line).ok()?;
            kind.section_id(t).map(|_| ())
        })?;
        let preamble_end = index.marks.first().map_or(opened.len(), |m| m.start);
        let preamble = opened.read_upto(0..preamble_end.min(64 << 10))?;
        let header = read_file_header(&String::from_utf8_lossy(&preamble));
        let sections = (0..index.marks.len())
            .into_par_iter()
            .map(|n| {
                let m = &index.marks[n];
                let end = index.marks.get(n + 1).map_or(opened.len(), |m| m.start);
                index_section(kind, &opened, m.start..end, m.line, meshes.as_ref())
            })
            .collect::<io::Result<Vec<Section>>>()?;
        drop(opened);
        Ok(Reader {
            kind,
            source,
            header,
            chunks: index.chunks,
            sections,
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.source.path()
    }

    pub fn is_current(&self) -> bool {
        self.source.is_current()
    }

    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    pub fn tallies(&self) -> impl Iterator<Item = &CuvInfo> {
        self.sections
            .iter()
            .filter_map(|s| s.body.as_ref().ok().map(|b| &b.info))
    }

    pub fn failed(&self) -> impl Iterator<Item = (u32, &str)> {
        self.sections
            .iter()
            .filter_map(|s| s.body.as_ref().err().map(|e| (s.id, e.as_str())))
    }

    pub fn tally(&self, id: u32) -> Result<&CuvInfo, ReadError> {
        self.find(id).map(|b| &b.info)
    }

    fn find(&self, id: u32) -> Result<&Body, ReadError> {
        let section =
            self.sections
                .iter()
                .find(|s| s.id == id)
                .ok_or_else(|| ReadError::NoTally {
                    id: Some(id),
                    tallies: self.sections.iter().map(|s| s.id).collect(),
                })?;
        section
            .body
            .as_ref()
            .map_err(|reason| ReadError::Unsupported {
                id,
                reason: reason.clone(),
            })
    }

    /// Voxels `voxels` of mesh `id`.
    pub fn read_voxels(&self, id: u32, voxels: Range<usize>) -> Result<Parsed, ReadError> {
        let body = self.find(id)?;
        let n = body.info.n_voxels;
        if voxels.start > voxels.end || voxels.end > n {
            return Err(ReadError::NoBin {
                id,
                axis: "voxel",
                index: voxels.end.max(voxels.start),
                count: n,
            });
        }
        let opened = self.source.open()?;
        let segments = self.segments(id, body, &opened)?;
        let first = segments.partition_point(|s| s.voxels.end <= voxels.start);
        let last = segments.partition_point(|s| s.voxels.start < voxels.end);
        let parts = segments[first..last.max(first)]
            .par_iter()
            .map(|s| {
                let text = opened.read(s.bytes.clone())?;
                parse_segment(self.kind, body, &text, s, &voxels)
                    .map_err(|reason| ReadError::Unsupported { id, reason })
            })
            .collect::<Result<Vec<Parsed>, ReadError>>()?;
        let mut all = Parsed::default();
        for p in parts {
            all.append(p);
        }
        all.warnings
            .splice(0..0, body.info.warnings.iter().cloned());
        if all.low_density > 0 {
            all.warnings.push(Warning::new(format!(
                "{} (voxel, cell) pairs have densities below 0.01 g/cm³, printed with 4 \
                 decimals: their masses are uncertain by more than 1%",
                all.low_density
            )));
        }
        if all.worst_volume > 0.01 {
            all.warnings.push(Warning::new(format!(
                "voxel volumes differ from the geometry's by up to {:.2}%",
                100.0 * all.worst_volume
            )));
        }
        Ok(all)
    }

    /// The mesh's segments, found on its first read.
    fn segments(
        &self,
        id: u32,
        body: &Body,
        opened: &Opened<'_>,
    ) -> Result<Arc<Vec<Segment>>, ReadError> {
        if let Some(s) = body.segments.get() {
            return Ok(s.clone());
        }
        let data = body.data.clone();
        let chunks = &self.chunks;
        // Chunks overlapping the data.
        let k0 = chunks
            .partition_point(|c| c.at <= data.start)
            .saturating_sub(1);
        let k1 = chunks
            .partition_point(|c| c.at < data.end)
            .min(chunks.len() - 1);
        let found = (k0..k1)
            .into_par_iter()
            .map(|k| {
                let lo = chunks[k].at.max(data.start);
                let hi = chunks[k + 1].at.min(data.end);
                if lo >= hi {
                    return Ok(None);
                }
                let line0 = if chunks[k].at < data.start {
                    body.data_line
                } else {
                    chunks[k].lines
                };
                let text = opened.read(lo..hi)?;
                let mut first = None;
                let mut count = 0usize;
                let base = text.as_ptr() as usize;
                for (n, line) in scan::lines(&text).enumerate() {
                    if is_voxel_line(self.kind, line) {
                        if first.is_none() {
                            let at = line.as_ptr() as usize - base;
                            first = Some((lo + at as u64, line0 + n as u64));
                        }
                        count += 1;
                    }
                }
                Ok(first.map(|f| (f, count)))
            })
            .collect::<Result<Vec<_>, ReadError>>()?;
        let found: Vec<_> = found.into_iter().flatten().collect();
        let mut segments = Vec::with_capacity(found.len());
        let mut voxel = 0usize;
        for (n, &((at, line), count)) in found.iter().enumerate() {
            let end = found.get(n + 1).map_or(data.end, |f| f.0 .0);
            segments.push(Segment {
                bytes: at..end,
                voxels: voxel..voxel + count,
                line,
            });
            voxel += count;
        }
        if voxel != body.info.n_voxels {
            return Err(ReadError::Unsupported {
                id,
                reason: format!(
                    "the {} lists {voxel} voxels, the mesh has {}",
                    self.kind.name(),
                    body.info.n_voxels
                ),
            });
        }
        let segments = Arc::new(segments);
        let _ = body.segments.set(segments.clone());
        Ok(segments)
    }
}

/// Index the section in bytes `range`, which starts on line `line` (from 0).
fn index_section(
    kind: FileKind,
    opened: &Opened<'_>,
    range: Range<u64>,
    line: u64,
    meshes: Option<&InputMeshes>,
) -> io::Result<Section> {
    let size = range.end - range.start;
    let mut want = 64u64 << 10;
    loop {
        let n = want.min(size);
        let bytes = opened.read_upto(range.start..range.start + n)?;
        let whole = n == size;
        let cut = if whole {
            bytes.len()
        } else {
            memchr::memrchr(b'\n', &bytes).map_or(0, |p| p + 1)
        };
        let raw = &bytes[..cut];
        let text = String::from_utf8_lossy(raw);
        let lines: Vec<&str> = text.lines().collect();
        match header::parse(kind, &lines) {
            Ok(h) => {
                let start = range.start + scan::line_start(raw, h.lines) as u64;
                let id = h.header.id;
                let data_line = line + h.lines as u64;
                let body = body(kind, h, meshes, start..range.end, data_line);
                return Ok(Section { id, body: Ok(body) });
            }
            Err(None) if !whole => want *= 4,
            Err(why) => {
                let id = lines
                    .first()
                    .and_then(|l| kind.section_id(l))
                    .and_then(Result::ok)
                    .unwrap_or(0);
                let why = why.unwrap_or_else(|| "the header doesn't end".into());
                return Ok(Section {
                    id,
                    body: Err(format!("header parse failed: {why}")),
                });
            }
        }
    }
}

fn body(
    kind: FileKind,
    h: CuvHeader,
    meshes: Option<&InputMeshes>,
    data: Range<u64>,
    data_line: u64,
) -> Body {
    let printed = h.header.geometry.clone();
    let n_voxels = printed.n_voxels();
    let mut warnings = h.warnings;
    let prepared = input::prepare_from(h.header, meshes, kind.name());
    warnings.extend(prepared.warnings);
    let hdr = prepared.header;
    if !prepared.geometry_from_input {
        if matches!(hdr.geometry, Geometry::Cylindrical { .. }) {
            warnings.push(Warning::new(
                "cylinder origin, axis and VEC are printed with 3 significant digits: the \
                 mesh may be slightly misplaced (give the input for the exact ones)",
            ));
        }
        if kind == FileKind::Cuvmsh && hdr.energy_bounds.len() > 2 {
            warnings.push(Warning::new(
                "energy bounds are printed with 3 significant digits (give the input for \
                 the exact ones)",
            ));
        }
    }
    let n_energy = hdr.energy_bounds.len().saturating_sub(1);
    let energy =
        (kind == FileKind::Cuvmsh).then(|| Axis::bounded(hdr.energy_bounds.clone(), false));
    let particle = (kind == FileKind::Cuvmsh).then(|| hdr.particle.clone());
    Body {
        info: CuvInfo {
            id: hdr.id,
            particle,
            comment: hdr.comment,
            geometry: hdr.geometry,
            geometry_from_input: prepared.geometry_from_input,
            energy,
            n_voxels,
            legacy_cylindrical: h.legacy_cylindrical,
            warnings,
        },
        printed,
        data,
        data_line,
        n_energy,
        segments: OnceLock::new(),
    }
}

fn is_count(tok: &[u8]) -> bool {
    !tok.is_empty() && (tok.iter().all(u8::is_ascii_digit) || tok.iter().all(|&b| b == b'*'))
}

fn is_voxel_line(kind: FileKind, line: &[u8]) -> bool {
    let mut t = scan::tokens(line);
    let mut tok = [&b""[..]; 7];
    let mut n = 0;
    for w in t.by_ref() {
        if n == 7 {
            return false;
        }
        tok[n] = w;
        n += 1;
    }
    match kind {
        FileKind::Meshinfo => n == 3 && is_count(tok[0]) && is_count(tok[2]),
        FileKind::Cuvmsh => {
            n == 6 && is_count(tok[0]) && is_count(tok[5]) && !tok[1].contains(&b'E')
        }
    }
}

/// The voxels of `seg` in `want`.
fn parse_segment(
    kind: FileKind,
    body: &Body,
    text: &[u8],
    seg: &Segment,
    want: &Range<usize>,
) -> Result<Parsed, String> {
    let mut out = Parsed::default();
    let mut p = VoxelParser {
        kind,
        body,
        out: &mut out,
        voxel: None,
        cells: 0,
        values: Vec::new(),
        low_density: 0,
        worst_volume: 0.0,
    };
    let mut voxel = seg.voxels.start;
    let base = seg.line + 1;
    let mut keep = false;
    let mut voxel_line = base;
    for (n, line) in scan::lines(text).enumerate() {
        let line_no = base + n as u64;
        if scan::is_blank(line) {
            continue;
        }
        let at = |m: String| format!("line {line_no}: {m}");
        if is_voxel_line(kind, line) {
            if keep {
                p.finish().map_err(at)?;
            }
            keep = want.contains(&voxel);
            voxel_line = line_no;
            if keep {
                p.start(voxel, line).map_err(at)?;
            }
            voxel += 1;
        } else if keep {
            p.line(line).map_err(at)?;
        }
    }
    if keep {
        p.finish().map_err(|m| format!("line {voxel_line}: {m}"))?;
    }
    let (low, worst) = (p.low_density, p.worst_volume);
    out.low_density = low;
    out.worst_volume = worst;
    Ok(out)
}

struct VoxelParser<'a> {
    kind: FileKind,
    body: &'a Body,
    out: &'a mut Parsed,
    voxel: Option<(usize, Option<usize>)>,
    /// Cells of the current voxel.
    cells: usize,
    /// Its values, as read.
    values: Vec<f64>,
    low_density: usize,
    worst_volume: f64,
}

fn num(tok: &[u8], what: &str) -> Result<f64, String> {
    float::parse(tok).ok_or_else(|| format!("bad {what} {:?}", String::from_utf8_lossy(tok)))
}

fn int(tok: &[u8], what: &str) -> Result<u32, String> {
    std::str::from_utf8(tok)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad {what} {:?}", String::from_utf8_lossy(tok)))
}

impl VoxelParser<'_> {
    fn start(&mut self, voxel: usize, line: &[u8]) -> Result<(), String> {
        let tok: Vec<&[u8]> = scan::tokens(line).collect();
        let (volume, declared) = match self.kind {
            FileKind::Meshinfo => {
                let factor = if self.body.info.legacy_cylindrical {
                    1.0 / std::f64::consts::TAU
                } else {
                    1.0
                };
                (num(tok[1], "voxel volume")? * factor, tok[2])
            }
            FileKind::Cuvmsh => {
                self.check_centre(voxel, &tok[1..4])?;
                (num(tok[4], "voxel volume")?, tok[5])
            }
        };
        let geometric = self.body.info.geometry.voxel_volume(voxel);
        if geometric > 0.0 {
            self.worst_volume = self
                .worst_volume
                .max((volume - geometric).abs() / geometric);
        }
        self.out.voxel_volume.push(volume);
        let declared = std::str::from_utf8(declared)
            .ok()
            .and_then(|s| s.parse().ok());
        self.voxel = Some((voxel, declared));
        self.cells = 0;
        self.values.clear();
        Ok(())
    }

    fn check_centre(&self, voxel: usize, tok: &[&[u8]]) -> Result<(), String> {
        let g = &self.body.printed;
        let (i, j, k) = g.ijk(voxel);
        let (a, b, c) = g.axes();
        let mid = |v: &[f64], n: usize| 0.5 * (v[n] + v[n + 1]);
        let expected = [mid(a, i), mid(b, j), mid(c, k)];
        let cylinder = matches!(g, Geometry::Cylindrical { .. });
        for (axis, (t, e)) in tok.iter().zip(expected).enumerate() {
            let theta = cylinder && axis == 2;
            if theta && self.body.info.legacy_cylindrical {
                continue;
            }
            let tol = if theta { 0.0011 } else { 0.006 };
            let v = num(t, "voxel centre")?;
            if (v - e).abs() > tol {
                return Err(format!(
                    "voxel {} is centred at {:?} in the file, at {expected:?} in the mesh: \
                     voxels are not in the order expected",
                    voxel + 1,
                    tok.iter()
                        .map(|t| String::from_utf8_lossy(t))
                        .collect::<Vec<_>>()
                ));
            }
        }
        Ok(())
    }

    fn line(&mut self, line: &[u8]) -> Result<(), String> {
        let (voxel, _) = self.voxel.ok_or("data before the first voxel")?;
        let tok: Vec<&[u8]> = scan::tokens(line).collect();
        let cell_line = is_count(tok[0]) && self.values.is_empty();
        match self.kind {
            FileKind::Meshinfo => {
                if tok.len() != 4 || !is_count(tok[0]) {
                    return Err(format!(
                        "expected a cell line, got {:?}",
                        String::from_utf8_lossy(line)
                    ));
                }
                let density = num(tok[1], "density")?;
                if density > 0.0 && density < 0.01 {
                    self.low_density += 1;
                }
                self.out.voxel.push(voxel as u32);
                self.out.cell.push(int(tok[0], "cell number")?);
                self.out.density.push(density);
                self.out.material.push(int(tok[2], "material number")?);
                self.out.fraction.push(num(tok[3], "volume fraction")?);
                self.cells += 1;
            }
            FileKind::Cuvmsh if cell_line => {
                if tok.len() != 2 && tok.len() != 4 {
                    return Err(format!(
                        "expected a cell line, got {:?}",
                        String::from_utf8_lossy(line)
                    ));
                }
                self.out.voxel.push(voxel as u32);
                self.out.cell.push(int(tok[0], "cell number")?);
                self.out.fraction.push(num(tok[1], "volume fraction")?);
                if tok.len() == 4 {
                    self.out.total.push(num(tok[2], "total flux")?);
                    self.out.total_error.push(num(tok[3], "total flux error")?);
                } else {
                    // Filled from the one bin in `finish`.
                    self.out.total.push(f64::NAN);
                    self.out.total_error.push(f64::NAN);
                }
                self.cells += 1;
            }
            FileKind::Cuvmsh => {
                for t in tok {
                    self.values.push(num(t, "value")?);
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        let Some((voxel, declared)) = self.voxel.take() else {
            return Ok(());
        };
        if let Some(d) = declared {
            if d != self.cells {
                return Err(format!(
                    "voxel {} declares {d} cells but lists {}",
                    voxel + 1,
                    self.cells
                ));
            }
        }
        if self.kind == FileKind::Cuvmsh {
            let n = self.body.n_energy;
            if self.values.len() != 2 * n * self.cells {
                return Err(format!(
                    "voxel {}: {} values for {} cells of {n} energy bins (flux and error each)",
                    voxel + 1,
                    self.values.len(),
                    self.cells
                ));
            }
            let first = self.out.total.len() - self.cells;
            for (c, v) in self.values.chunks(2 * n).enumerate() {
                self.out.flux.extend_from_slice(&v[..n]);
                self.out.rel_error.extend_from_slice(&v[n..]);
                if self.out.total[first + c].is_nan() && n == 1 {
                    self.out.total[first + c] = v[0];
                    self.out.total_error[first + c] = v[1];
                }
            }
        }
        Ok(())
    }
}
