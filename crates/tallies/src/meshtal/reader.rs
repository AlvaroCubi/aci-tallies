//! Meshtal files read by parts.
//!
//! [`Reader::open`] indexes the file in one parallel pass
//! ([`crate::common::scan`]) and reads each section's header; the values
//! are read when asked for, from disk, straight into the arrays returned:
//! a few chunks of text per thread are in memory at a time, never a whole
//! section. In the columnar layouts the rows of one (energy, time) bin are
//! consecutive, and in the matrix ones each bin is a run of blocks, so
//! [`Reader::read_bin`] reads only that bin's part of the file.

use std::borrow::Cow;
use std::io;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array3, Array5};
use rayon::prelude::*;

use super::columnar::{self, Columns, RowsOut};
use super::header::{self, FileHeader};
use super::input::{self, InputMeshes};
use super::matrix::{self, Layout, Marked, Marker};
use super::model::{Tally, TallyBin, TallyInfo};
use crate::common::scan::{self, Chunk, Index, Mark};
use crate::common::source::{Opened, Source};
use crate::common::{Axis, Input, ReadError, Warning};

/// An indexed meshtal file.
pub struct Reader {
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
    info: TallyInfo,
    data: Data,
}

enum Data {
    Columnar {
        columns: Columns,
        /// Row (in the file) of the block's first data row.
        first_row: u64,
        /// Rows in the block, up to the section's end.
        rows: u64,
        /// Where the section ends (bytes).
        end: u64,
    },
    Matrix {
        layout: Layout,
        blocks: Vec<matrix::Block>,
    },
}

/// What a marked line is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `Mesh Tally Number N`: a section starts.
    Section,
    /// A column header row (`X Y Z Result Rel Error ...`).
    Columns,
    Matrix(Marker),
}

fn classify(line: &[u8]) -> Option<Kind> {
    let t = line.trim_ascii_start();
    let kind = if t.starts_with(b"Mesh Tally Number") {
        Kind::Section
    } else if t.starts_with(b"Energy Bin:") {
        Kind::Matrix(Marker::Energy { total: false })
    } else if t.starts_with(b"Total Energy Bin") {
        Kind::Matrix(Marker::Energy { total: true })
    } else if t.starts_with(b"Time Bin:") {
        Kind::Matrix(Marker::Time { total: false })
    } else if t.starts_with(b"Total Time Bin") {
        Kind::Matrix(Marker::Time { total: true })
    } else if let Some(name) = matrix::Name::of_outer_line(t) {
        Kind::Matrix(Marker::Outer(name))
    } else if memchr::memmem::find(t, b"Rel Error").is_some() {
        Kind::Columns
    } else {
        return None;
    };
    Some(kind)
}

/// Whether `head` (the first few kB of a file) looks like a meshtal file.
pub fn is_meshtal(head: &[u8]) -> bool {
    // A cuvmsh has the same preamble and section lines.
    !crate::cuv::is_cuvmsh(head)
        && (memchr::memmem::find(head, b"Number of histories used for normalizing tallies")
            .is_some()
            || memchr::memmem::find(head, b"Mesh Tally Number").is_some())
}

impl Reader {
    /// Index the meshtal file at `path`, each tally's geometry from its
    /// header.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Reader> {
        Reader::build(Source::file(path.as_ref())?, None, scan::CHUNK_BYTES)
    }

    /// [`Reader::open`], each tally's geometry and energy and time bounds
    /// from its `FMESH` card in `input` (the MCNP input the file came from)
    /// rather than from the rounded copy in the header: the only way to get
    /// exact bounds and a `TR=` placement. A tally whose card is missing or
    /// doesn't match the header at its print precision (another input)
    /// keeps the header geometry, with a warning.
    pub fn open_with_input(path: impl AsRef<Path>, input: &Input) -> io::Result<Reader> {
        Reader::build(Source::file(path.as_ref())?, Some(input), scan::CHUNK_BYTES)
    }

    /// The meshtal file `bytes`, held in memory.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, input: Option<&Input>) -> Reader {
        Reader::build(Source::bytes(bytes), input, scan::CHUNK_BYTES)
            .expect("reading memory doesn't fail")
    }

    pub(crate) fn build(source: Source, input: Option<&Input>, chunk: usize) -> io::Result<Reader> {
        let meshes = input.map(InputMeshes::of);
        let opened = source.open()?;
        let index = scan::index(&opened, chunk, classify)?;
        let starts: Vec<usize> = (0..index.marks.len())
            .filter(|&k| index.marks[k].kind == Kind::Section)
            .collect();
        let preamble_end = starts
            .first()
            .map_or(opened.len(), |&k| index.marks[k].start);
        let preamble = opened.read_upto(0..preamble_end.min(64 << 10))?;
        let header = header::read_file_header(&String::from_utf8_lossy(&preamble));

        let sections = (0..starts.len())
            .into_par_iter()
            .map(|n| {
                let k = starts[n];
                let next = starts.get(n + 1).copied().unwrap_or(index.marks.len());
                index_section(&opened, &index, k..next, meshes.as_ref())
            })
            .collect::<io::Result<Vec<Section>>>()?;
        drop(opened);
        Ok(Reader {
            source,
            header,
            chunks: index.chunks,
            sections,
        })
    }

    /// The file's path (`None` when read from memory).
    pub fn path(&self) -> Option<&Path> {
        self.source.path()
    }

    /// Whether the file still has the size and modification time it had
    /// when indexed (reads fail with [`ReadError::Changed`] otherwise).
    pub fn is_current(&self) -> bool {
        self.source.is_current()
    }

    /// What the file says about the run.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// The tallies that can be read, in file order.
    pub fn tallies(&self) -> impl Iterator<Item = &TallyInfo> {
        self.sections
            .iter()
            .filter_map(|s| s.body.as_ref().ok().map(|b| &b.info))
    }

    /// The sections that can't be read: tally number (0 when even that is
    /// unreadable) and why.
    pub fn failed(&self) -> impl Iterator<Item = (u32, &str)> {
        self.sections
            .iter()
            .filter_map(|s| s.body.as_ref().err().map(|e| (s.id, e.as_str())))
    }

    /// Tally `id`.
    pub fn tally(&self, id: u32) -> Result<&TallyInfo, ReadError> {
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

    /// Every tally that can be read, one after the other (each read in
    /// parallel).
    pub fn read_all(&self) -> Result<Vec<Tally>, ReadError> {
        self.tallies().map(|t| self.read(t.id)).collect()
    }

    /// The whole of tally `id`: every energy and time bin.
    pub fn read(&self, id: u32) -> Result<Tally, ReadError> {
        let body = self.find(id)?;
        let opened = self.source.open()?;
        let shape = body.info.shape();
        let voxels = shape[2] * shape[3] * shape[4];
        let n = shape.iter().product::<usize>() as u64;
        let mut values = self.read_values(&opened, body, 0..n, |_, _| true)?;
        values.warnings.extend(body.info.warnings.iter().cloned());
        let array = |v: Vec<f64>| Array5::from_shape_vec(shape, v).expect("one value per index");
        let volume = values.volume.map(|v| {
            Array3::from_shape_vec((shape[2], shape[3], shape[4]), v).expect("one volume per voxel")
        });
        debug_assert!(volume.as_ref().is_none_or(|v| v.len() == voxels));
        Ok(Tally::from_info(
            body.info.clone(),
            array(values.result),
            array(values.rel_error),
            volume,
            values.warnings,
        ))
    }

    /// Energy index `energy` and time index `time` of tally `id` (Total
    /// ones included, as in [`Tally::result`]), reading only that bin's
    /// part of the file.
    pub fn read_bin(&self, id: u32, energy: usize, time: usize) -> Result<TallyBin, ReadError> {
        let body = self.find(id)?;
        let shape = body.info.shape();
        let check = |axis, index, count| {
            if index < count {
                Ok(())
            } else {
                Err(ReadError::NoBin {
                    id,
                    axis,
                    index,
                    count,
                })
            }
        };
        check("energy", energy, shape[0])?;
        check("time", time, shape[1])?;
        let opened = self.source.open()?;
        let voxels = shape[2] * shape[3] * shape[4];
        let rows = columnar::bin_rows(energy, time, shape[1], voxels);
        let mut values = self.read_values(&opened, body, rows, |e, t| (e, t) == (energy, time))?;
        values.warnings.extend(body.info.warnings.iter().cloned());
        let dims = (shape[2], shape[3], shape[4]);
        let array = |v: Vec<f64>| Array3::from_shape_vec(dims, v).expect("one value per voxel");
        Ok(TallyBin {
            id,
            energy,
            time,
            result: array(values.result),
            rel_error: array(values.rel_error),
            volume: values.volume.map(array),
            warnings: values.warnings,
        })
    }

    /// Values `flat` of a tally (flat indices into its result, a whole
    /// number of (energy, time) bins), which are the bins `wanted` accepts.
    fn read_values(
        &self,
        opened: &Opened<'_>,
        body: &Body,
        flat: Range<u64>,
        wanted: impl Fn(usize, usize) -> bool + Sync,
    ) -> Result<Values, ReadError> {
        let shape = body.info.shape();
        let voxels = shape[2] * shape[3] * shape[4];
        let n = (flat.end - flat.start) as usize;
        let mut result = vec![f64::NAN; n];
        let mut rel_error = vec![f64::NAN; n];
        match &body.data {
            Data::Columnar {
                columns,
                first_row,
                rows,
                end,
            } => {
                let mut volume = vec![f64::NAN; if columns.has_volume() { voxels } else { 0 }];
                let out = RowsOut {
                    start: flat.start,
                    result: &mut result,
                    rel_error: &mut rel_error,
                    volume: &mut volume,
                };
                let warnings = self.read_rows(opened, columns, *first_row, *rows, *end, out)?;
                Ok(Values {
                    result,
                    rel_error,
                    volume: columns.has_volume().then_some(volume),
                    warnings,
                })
            }
            Data::Matrix { layout, blocks } => {
                let (ni, nj, nk) = (shape[2], shape[3], shape[4]);
                let dims = [ni, nj, nk];
                let (n_cols, n_rows) = (dims[layout.across], dims[layout.down]);
                let blocks: Vec<&matrix::Block> =
                    blocks.iter().filter(|b| wanted(b.energy, b.time)).collect();
                // Blocks are parsed into buffers of their own, so they go
                // in batches that bound those buffers' memory.
                let per_block = 16 * n_cols * n_rows;
                let batch = ((256usize << 20) / per_block.max(1)).max(1);
                let mut warnings = Vec::new();
                for part in blocks.chunks(batch) {
                    let parsed = part
                        .par_iter()
                        .map(|b| {
                            let text = opened.read(b.bytes.clone())?;
                            Ok(matrix::parse_block(&text, b, n_cols, n_rows))
                        })
                        .collect::<Result<Vec<_>, ReadError>>()?;
                    for (b, p) in part.iter().zip(parsed) {
                        let bin = ((b.energy * shape[1] + b.time) * voxels) as u64 - flat.start;
                        for row in 0..n_rows {
                            for col in 0..n_cols {
                                let [i, j, k] = layout.ijk(col, row, b.outer);
                                let at = bin as usize + (i * nj + j) * nk + k;
                                result[at] = p.result[row * n_cols + col];
                                rel_error[at] = p.rel_error[row * n_cols + col];
                            }
                        }
                        warnings.extend(p.warnings);
                    }
                }
                Ok(Values {
                    result,
                    rel_error,
                    volume: None,
                    warnings,
                })
            }
        }
    }

    /// Parse rows `out.start..` of a columnar block, in parallel, each
    /// thread reading the chunks it parses.
    fn read_rows(
        &self,
        opened: &Opened<'_>,
        columns: &Columns,
        first_row: u64,
        block_rows: u64,
        end: u64,
        out: RowsOut<'_>,
    ) -> Result<Vec<Warning>, ReadError> {
        // The rows the file has: `a..b`, as rows of the file.
        let a = first_row + out.start;
        let b = first_row + (out.start + out.result.len() as u64).min(block_rows);
        if a >= b {
            return Ok(Vec::new());
        }
        let chunks = &self.chunks;
        let range = scan::chunks_of_rows(chunks, a..b);
        let lens = range.clone().map(|k| {
            let lo = chunks[k].rows.max(a);
            let hi = chunks[k + 1].rows.min(b);
            hi.saturating_sub(lo) as usize
        });
        let parts = out.split(lens);
        let warnings = range
            .into_par_iter()
            .zip(parts)
            .map(|(k, part)| {
                let text = opened.read(chunks[k].at..chunks[k + 1].at.min(end))?;
                let first = chunks[k].rows as i64 - first_row as i64;
                Ok(columnar::parse_rows(
                    &text,
                    first,
                    chunks[k].lines,
                    columns,
                    part,
                ))
            })
            .collect::<Result<Vec<Vec<Warning>>, ReadError>>()?;
        Ok(warnings.into_iter().flatten().collect())
    }
}

struct Values {
    result: Vec<f64>,
    rel_error: Vec<f64>,
    volume: Option<Vec<f64>>,
    warnings: Vec<Warning>,
}

/// Read section `marks[range]` (its first mark the section's start): its
/// header, and where and how its values are.
fn index_section(
    opened: &Opened<'_>,
    index: &Index<Kind>,
    range: Range<usize>,
    meshes: Option<&InputMeshes>,
) -> io::Result<Section> {
    let marks = &index.marks[range.clone()];
    let start = marks[0].start;
    let (end, end_row) = index
        .marks
        .get(range.end)
        .map_or((opened.len(), index.rows()), |m| (m.start, m.row));
    let size = end - start;
    let mut want = 64u64 << 10;
    loop {
        let n = want.min(size);
        let bytes = opened.read_upto(start..start + n)?;
        let whole = n == size;
        let cut = if whole {
            bytes.len()
        } else {
            memchr::memrchr(b'\n', &bytes).map_or(0, |p| p + 1)
        };
        let raw = &bytes[..cut];
        let text = String::from_utf8_lossy(raw);
        match header::read_header(&text) {
            Ok((hdr, at)) if whole || at < text.len() => {
                // Where the data starts (counted in lines, which replacing
                // bad UTF-8 doesn't change).
                let line = text.as_bytes()[..at]
                    .iter()
                    .filter(|&&b| b == b'\n')
                    .count();
                let data_start = start + scan::line_start(raw, line) as u64;
                let id = hdr.id;
                let prepared = input::prepare(hdr, meshes);
                let body = index_body(opened, prepared, &marks[1..], data_start, end, end_row);
                return Ok(Section { id, body });
            }
            Ok(_) | Err(header::HeaderError::UnexpectedEnd(_)) if !whole => want *= 4,
            Ok(_) => unreachable!("a whole section's header is accepted"),
            Err(e) => {
                return Ok(Section {
                    id: tally_number(&text),
                    body: Err(format!("header parse failed: {e}")),
                })
            }
        }
    }
}

/// How the data of a section, from `data_start` to `end`, is laid out.
fn index_body(
    opened: &Opened<'_>,
    p: input::Prepared,
    marks: &[Mark<Kind>],
    data_start: u64,
    end: u64,
    end_row: u64,
) -> Result<Body, String> {
    let hdr = p.header;
    let mut warnings = Vec::new();
    let marks: Vec<&Mark<Kind>> = marks.iter().filter(|m| m.start >= data_start).collect();
    let first = marks.first().filter(|m| m.start == data_start);
    let info = |energy, time, out_format, warnings| TallyInfo {
        id: hdr.id,
        particle: hdr.particle.clone(),
        comment: hdr.comment.clone(),
        geometry: hdr.geometry.clone(),
        geometry_from_input: p.geometry_from_input,
        energy_weighted: p.energy_weighted,
        energy,
        time,
        out_format,
        warnings,
    };
    let io_error = |e: ReadError| format!("reading the data block: {e}");
    match first.map(|m| m.kind) {
        Some(Kind::Columns) => {
            let m = first.expect("matched");
            let row = opened.read(m.start..m.end).map_err(io_error)?;
            let columns = Columns::from_header_row(&row);
            let energy = Axis::mcnp(hdr.energy_bounds.clone());
            let time = Axis::mcnp(hdr.time_bounds.clone());
            let (ni, nj, nk) = hdr.geometry.dims();
            let expected = (energy.len() * time.len() * ni * nj * nk) as u64;
            let first_row = m.row + 1;
            let rows = end_row - first_row;
            warnings.extend(columnar::row_count_warning(rows, expected));
            warnings.extend(p.warnings.iter().cloned());
            let out_format = columns.out_format();
            Ok(Body {
                info: info(energy, time, out_format, warnings),
                data: Data::Columnar {
                    columns,
                    first_row,
                    rows,
                    end,
                },
            })
        }
        Some(Kind::Matrix(Marker::Energy { .. })) => {
            let all: Vec<Marked> = marks
                .iter()
                .filter_map(|m| match m.kind {
                    Kind::Matrix(marker) => Some(Marked {
                        marker,
                        start: m.start,
                        end: m.end,
                        line: m.line,
                    }),
                    _ => None,
                })
                .collect();
            let layout = find_layout(opened, &hdr.geometry, &all, end)?;
            let outer = layout.outer_name(&hdr.geometry);
            let markers: Vec<Marked> = all
                .into_iter()
                .filter(|m| !matches!(m.marker, Marker::Outer(name) if name != outer))
                .collect();
            let dims = hdr.geometry.dims();
            let blocks = matrix::index_blocks(
                &markers,
                end,
                hdr.n_energy_from_bounds(),
                hdr.n_time_from_bounds(),
                [dims.0, dims.1, dims.2][layout.outer],
            )?;
            warnings.extend(blocks.warnings);
            warnings.extend(p.warnings.iter().cloned());
            let energy = Axis::bounded(hdr.energy_bounds.clone(), blocks.energy_total);
            let time = Axis::bounded(hdr.time_bounds.clone(), blocks.time_total);
            Ok(Body {
                info: info(energy, time, layout.out_format(), warnings),
                data: Data::Matrix {
                    layout,
                    blocks: blocks.blocks,
                },
            })
        }
        _ if data_start >= end => Err("the section has no data".to_string()),
        _ => Err("the data block is neither columns nor matrices".to_string()),
    }
}

/// The layout, from the first block that has a `Tally Results:` line.
fn find_layout(
    opened: &Opened<'_>,
    geometry: &super::model::Geometry,
    markers: &[Marked],
    end: u64,
) -> Result<Layout, String> {
    for (n, m) in markers.iter().enumerate() {
        if !matches!(m.marker, Marker::Outer(_)) {
            continue;
        }
        let to = markers.get(n + 1).map_or(end, |next| next.start);
        let text: Cow<[u8]> = opened
            .read(m.end..to)
            .map_err(|e| format!("reading the data block: {e}"))?;
        if let Some(layout) = Layout::find(geometry, &text)? {
            return Ok(layout);
        }
    }
    Err("no \"Tally Results:\" line found in matrix data block".to_string())
}

/// The number on a section's first line (`Mesh Tally Number N`), or 0.
fn tally_number(section: &str) -> u32 {
    section
        .lines()
        .next()
        .and_then(|l| l.split("Mesh Tally Number").nth(1))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use std::path::PathBuf;

    fn file(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(rel)
    }

    fn same(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    /// Whatever the chunk size, every tally and every bin of each reads the
    /// same, from disk and from memory.
    fn agrees(msht: &str, input: Option<&str>) {
        let path = file(msht);
        let input = input.map(|i| Input::read(file(i)).unwrap());
        let reference = Reader::open(&path).unwrap();
        let reference = match &input {
            Some(i) => Reader::open_with_input(&path, i).unwrap(),
            None => reference,
        };
        let want = reference.read_all().unwrap();
        assert!(!want.is_empty());
        assert_eq!(reference.failed().count(), 0);
        let bytes = std::fs::read(&path).unwrap();
        for chunk in [1, 7, 300, 4096] {
            for source in [Source::file(&path).unwrap(), Source::bytes(bytes.clone())] {
                let reader = Reader::build(source, input.as_ref(), chunk).unwrap();
                let got = reader.read_all().unwrap();
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    assert_eq!(g.id, w.id);
                    assert!(same(
                        g.result.as_slice().unwrap(),
                        w.result.as_slice().unwrap()
                    ));
                    assert!(same(
                        g.rel_error.as_slice().unwrap(),
                        w.rel_error.as_slice().unwrap()
                    ));
                    assert_eq!(
                        g.warnings, w.warnings,
                        "{msht} tally {} chunk {chunk}",
                        w.id
                    );
                    assert_eq!(g.geometry, w.geometry);
                    assert_eq!((&g.energy, &g.time), (&w.energy, &w.time));
                    let shape = w.result.shape();
                    for e in 0..shape[0] {
                        for t in 0..shape[1] {
                            let bin = reader.read_bin(w.id, e, t).unwrap();
                            let want = w.result.slice(s![e, t, .., .., ..]).to_owned();
                            assert!(
                                same(bin.result.as_slice().unwrap(), want.as_slice().unwrap()),
                                "{msht}: tally {} bin ({e}, {t}), chunk {chunk}",
                                w.id
                            );
                            let want = w.rel_error.slice(s![e, t, .., .., ..]).to_owned();
                            assert!(same(
                                bin.rel_error.as_slice().unwrap(),
                                want.as_slice().unwrap()
                            ));
                            match (&bin.volume, &w.volume) {
                                (Some(b), Some(w)) => {
                                    assert!(same(b.as_slice().unwrap(), w.as_slice().unwrap()))
                                }
                                (None, None) => {}
                                _ => panic!("volume in one and not the other"),
                            }
                        }
                    }
                    assert!(matches!(
                        reader.read_bin(w.id, shape[0], 0),
                        Err(ReadError::NoBin { axis: "energy", .. })
                    ));
                }
                assert!(matches!(
                    reader.tally(999_999),
                    Err(ReadError::NoTally { .. })
                ));
            }
        }
    }

    #[test]
    fn reads_agree_whatever_the_chunks() {
        agrees("pencil_beam.msht", None);
        agrees("pencil_beam_input.msht", Some("pencil_beam_input.mcnp"));
        agrees("../../examples/meshtal", None);
    }

    #[test]
    fn file_header() {
        let r = Reader::open(file("pencil_beam.msht")).unwrap();
        let h = r.header();
        assert_eq!((h.code.as_str(), h.version.as_str()), ("mcnp6", "6"));
        assert_eq!(h.probid, "10/15/26 14:36:50");
        assert!(h.title.starts_with("Pencil beam through void"));
        assert_eq!(h.histories, Some(100.0));
        assert!(is_meshtal(
            &std::fs::read(file("pencil_beam.msht")).unwrap()[..4096]
        ));
        assert!(!is_meshtal(b"mcnp 6 07/27/23\n title\nntal 1\n"));
    }

    const HEAD: &str = " Mesh Tally Number 4\n neutron  mesh tally.\n\n Tally bin boundaries:\n    \
                        X direction:  0.00 1.00 2.00\n    Y direction:  0.00 1.00\n    \
                        Z direction:  0.00 1.00\n    Energy bin boundaries: 0.00E+00 1.00E+36\n\n   \
                        X         Y         Z     Result     Rel Error\n";

    /// A line longer than a chunk, and than the slack read past one.
    #[test]
    fn long_lines() {
        let pad = " ".repeat(300_000);
        let text = format!("{HEAD}  0.5{pad} 0.5 0.5 1.0E+00 0.1\n  1.5 0.5 0.5 3.0-104 0.2{pad}");
        for chunk in [1, 1000, scan::CHUNK_BYTES] {
            let r = Reader::build(Source::bytes(text.as_bytes().to_vec()), None, chunk).unwrap();
            let bin = r.read_bin(4, 0, 0).unwrap();
            assert!(bin.warnings.is_empty(), "{:?}", bin.warnings);
            assert_eq!(bin.result.as_slice().unwrap(), [1.0, 3.0e-104]);
        }
    }

    #[test]
    fn damaged_and_changed_files() {
        let dir = std::env::temp_dir().join(format!("aci-tallies-meshtal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.msht");
        std::fs::write(
            &p,
            format!("{HEAD}  0.5 0.5 0.5 1.0E+00 0.1\n  1.5 0.5 0.5 oops 0.2\n"),
        )
        .unwrap();
        let r = Reader::open(&p).unwrap();
        let bin = r.read_bin(4, 0, 0).unwrap();
        assert_eq!(bin.result[[0, 0, 0]], 1.0);
        assert!(bin.result[[1, 0, 0]].is_nan());
        assert_eq!(bin.warnings.len(), 1, "{:?}", bin.warnings);
        assert_eq!(bin.warnings[0].line, Some(12));
        assert!(bin.warnings[0].message.starts_with("row 1:"));
        // A changed file is not read with a stale index.
        std::fs::write(&p, format!("{HEAD}  0.5 0.5 0.5 2.0E+00 0.1\n")).unwrap();
        assert!(!r.is_current());
        assert!(matches!(r.read_bin(4, 0, 0), Err(ReadError::Changed(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unreadable_sections_are_listed_not_fatal() {
        let text = format!(
            "{HEAD}  0.5 0.5 0.5 1.0E+00 0.1\n  1.5 0.5 0.5 2.0E+00 0.2\n \
             Mesh Tally Number 14\n garbage\n Mesh Tally Number 24\n neutron  mesh tally.\n\n \
             Tally bin boundaries:\n    X direction:  0.00 1.00\n    Y direction:  0.00 1.00\n    \
             Z direction:  0.00 1.00\n    Energy bin boundaries: 0.00E+00 1.00E+36\n\n something else\n"
        );
        let r = Reader::from_bytes(text.into_bytes(), None);
        assert_eq!(r.tallies().map(|t| t.id).collect::<Vec<_>>(), [4]);
        let failed: Vec<(u32, &str)> = r.failed().collect();
        assert_eq!(failed.len(), 2, "{failed:?}");
        assert_eq!(failed[0].0, 14);
        assert_eq!(
            failed[1],
            (24, "the data block is neither columns nor matrices")
        );
        assert!(matches!(
            r.read(24),
            Err(ReadError::Unsupported { id: 24, .. })
        ));
        assert_eq!(r.read_all().unwrap().len(), 1);
    }
}
