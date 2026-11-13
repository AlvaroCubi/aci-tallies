//! MCTAL files read by parts.
//!
//! [`Reader::open`] indexes the file in one parallel pass
//! ([`crate::common::scan`]), reads the file header and each tally's header
//! (in parallel), and notes where each `vals` block is. Values are read when
//! asked for, straight from disk into the arrays returned. As `f` varies
//! slowest, the values of a range of `f` bins (cells, surfaces) are a range
//! of lines, which [`Reader::read_f`] reads alone.

use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

use super::header;
use super::model::{FileHeader, Kcode, Tally, TallyBin, TallyInfo, F};
use super::vals::{self, ValsOut};
use crate::common::scan::{self, Chunk, Index, Mark};
use crate::common::source::{Opened, Source};
use crate::common::{Input, ReadError, Warning};

/// An indexed MCTAL file.
pub struct Reader {
    source: Source,
    header: FileHeader,
    chunks: Vec<Chunk>,
    tallies: Vec<Entry>,
    kcode: Option<Result<Kcode, String>>,
    warnings: Vec<Warning>,
}

struct Entry {
    id: u32,
    body: Result<Body, String>,
}

struct Body {
    info: TallyInfo,
    /// The unperturbed values, then each perturbation's.
    blocks: Vec<Block>,
}

/// A `vals` block and the fluctuation chart after it.
struct Block {
    perturbation: Option<u32>,
    /// Row (in the file) of its first line of values.
    first_row: u64,
    /// Rows of values it has.
    rows: u64,
    /// Where its values end (bytes).
    end: u64,
    tfc: Option<Range<u64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Tally,
    Vals(Option<u32>),
    Tfc,
    Kcode,
}

/// The keyword lines MCNP writes at the start of a line (comments are
/// indented, values start with a space).
fn classify(line: &[u8]) -> Option<Kind> {
    let word =
        |w: &[u8]| line.starts_with(w) && line.get(w.len()).is_none_or(|b| b.is_ascii_whitespace());
    if word(b"tally") {
        Some(Kind::Tally)
    } else if word(b"vals") {
        let pert = std::str::from_utf8(&line[4..])
            .ok()
            .and_then(|rest| rest.trim().strip_prefix("pert"))
            .and_then(|n| n.trim().parse().ok());
        Some(Kind::Vals(pert))
    } else if word(b"tfc") {
        Some(Kind::Tfc)
    } else if word(b"kcode") {
        Some(Kind::Kcode)
    } else {
        None
    }
}

pub use super::header::is_mctal;

impl Reader {
    /// Index the MCTAL file at `path`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Reader> {
        Reader::build(Source::file(path.as_ref())?, None, scan::CHUNK_BYTES)
    }

    /// [`Reader::open`], with the MCNP input the file came from (for what
    /// the file doesn't say about its bins).
    pub fn open_with_input(path: impl AsRef<Path>, input: &Input) -> io::Result<Reader> {
        Reader::build(Source::file(path.as_ref())?, Some(input), scan::CHUNK_BYTES)
    }

    /// The MCTAL file `bytes`, held in memory.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, input: Option<&Input>) -> Reader {
        Reader::build(Source::bytes(bytes), input, scan::CHUNK_BYTES)
            .expect("reading memory doesn't fail")
    }

    pub(crate) fn build(source: Source, input: Option<&Input>, chunk: usize) -> io::Result<Reader> {
        let cards = input.map(tally_cards);
        let opened = source.open()?;
        let index = scan::index(&opened, chunk, classify)?;
        let marks = &index.marks;
        let tallies: Vec<usize> = (0..marks.len())
            .filter(|&k| marks[k].kind == Kind::Tally)
            .collect();
        let kcode = marks.iter().position(|m| m.kind == Kind::Kcode);
        let end_of = |k: usize| marks.get(k).map_or(opened.len(), |m| m.start);

        let head_end = tallies
            .first()
            .or(kcode.as_ref())
            .map_or(opened.len(), |&k| marks[k].start);
        let head = opened.read(0..head_end)?;
        let header = header::read_file_header(&String::from_utf8_lossy(&head)).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not an MCTAL file: {e}"),
            )
        })?;
        let mut warnings = Vec::new();
        if header.tallies.len() != tallies.len() {
            warnings.push(Warning::new(format!(
                "the ntal line lists {} tallies, the file has {}",
                header.tallies.len(),
                tallies.len()
            )));
        }

        let entries = (0..tallies.len())
            .into_par_iter()
            .map(|n| {
                let k = tallies[n];
                let next = tallies.get(n + 1).copied().or(kcode).unwrap_or(marks.len());
                let id = header.tallies.get(n).copied().unwrap_or(0);
                let mut body = index_tally(&opened, &index, k..next, id, header.perturbations)?;
                if let (Ok(body), Some(cards)) = (&mut body, &cards) {
                    label(&mut body.info, cards.get(&id));
                }
                Ok(Entry { id, body })
            })
            .collect::<Result<Vec<Entry>, ReadError>>()?;

        let kcode = match kcode {
            Some(k) => {
                let text = opened.read(marks[k].start..end_of(k + 1))?;
                Some(header::read_kcode(&String::from_utf8_lossy(&text)))
            }
            None => None,
        };
        drop(opened);
        Ok(Reader {
            source,
            header,
            chunks: index.chunks,
            tallies: entries,
            kcode,
            warnings,
        })
    }

    /// The file's path (`None` when read from memory).
    pub fn path(&self) -> Option<&Path> {
        self.source.path()
    }

    /// Whether the file is unchanged since it was indexed.
    pub fn is_current(&self) -> bool {
        self.source.is_current()
    }

    /// What the file says about the run.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Problems with the file as a whole.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// The tallies that can be read, in file order.
    pub fn tallies(&self) -> impl Iterator<Item = &TallyInfo> {
        self.tallies
            .iter()
            .filter_map(|t| t.body.as_ref().ok().map(|b| &b.info))
    }

    /// The tallies that can't be read, and why.
    pub fn failed(&self) -> impl Iterator<Item = (u32, &str)> {
        self.tallies
            .iter()
            .filter_map(|t| t.body.as_ref().err().map(|e| (t.id, e.as_str())))
    }

    /// The `KCODE` estimates per cycle, if the run was a KCODE one (an
    /// error if its block couldn't be read).
    pub fn kcode(&self) -> Option<Result<&Kcode, &str>> {
        self.kcode
            .as_ref()
            .map(|k| k.as_ref().map_err(String::as_str))
    }

    /// Tally `id`.
    pub fn tally(&self, id: u32) -> Result<&TallyInfo, ReadError> {
        self.find(id).map(|b| &b.info)
    }

    fn find(&self, id: u32) -> Result<&Body, ReadError> {
        let entry = self
            .tallies
            .iter()
            .find(|t| t.id == id)
            .ok_or_else(|| ReadError::NoTally {
                id: Some(id),
                tallies: self.tallies.iter().map(|t| t.id).collect(),
            })?;
        entry
            .body
            .as_ref()
            .map_err(|reason| ReadError::Unsupported {
                id,
                reason: reason.clone(),
            })
    }

    /// Every tally that can be read (unperturbed values).
    pub fn read_all(&self) -> Result<Vec<Tally>, ReadError> {
        self.tallies().map(|t| self.read(t.id)).collect()
    }

    /// Tally `id`: all of it.
    pub fn read(&self, id: u32) -> Result<Tally, ReadError> {
        let n = self.tally(id)?.axes[F].len();
        self.read_part(id, 0, 0..n)
    }

    /// Tally `id`, only its `f` bins `f` (cells, surfaces, detectors):
    /// only their part of the file is read.
    pub fn read_f(&self, id: u32, f: Range<usize>) -> Result<Tally, ReadError> {
        self.read_part(id, 0, f)
    }

    /// Tally `id` as perturbation `n` (counting from 1, in file order) left
    /// it.
    pub fn read_perturbation(&self, id: u32, n: usize) -> Result<Tally, ReadError> {
        let count = self.find(id)?.blocks.len() - 1;
        if n == 0 || n > count {
            return Err(ReadError::NoBin {
                id,
                axis: "perturbation",
                index: n,
                count,
            });
        }
        let nf = self.tally(id)?.axes[F].len();
        self.read_part(id, n, 0..nf)
    }

    fn read_part(&self, id: u32, block: usize, f: Range<usize>) -> Result<Tally, ReadError> {
        let body = self.find(id)?;
        let info = &body.info;
        let nf = info.axes[F].len();
        if f.start >= f.end || f.end > nf {
            return Err(ReadError::NoBin {
                id,
                axis: "f",
                index: f.end.max(f.start),
                count: nf,
            });
        }
        let b = &body.blocks[block];
        let opened = self.source.open()?;
        let per_f = (info.len() / nf) as u64;
        let values = (f.start as u64 * per_f)..(f.end as u64 * per_f);
        let n = (values.end - values.start) as usize;
        let mut result = vec![f64::NAN; n];
        let mut rel_error = vec![f64::NAN; n];
        let out = ValsOut {
            start: values.start,
            result: &mut result,
            rel_error: &mut rel_error,
        };
        let mut warnings = self.read_vals(&opened, b, info.len() as u64, out)?;
        let tfc = match &b.tfc {
            Some(r) => {
                let text = opened.read(r.clone())?;
                match header::read_tfc(&String::from_utf8_lossy(&text)) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        warnings.push(Warning::new(format!("fluctuation chart: {e}")));
                        None
                    }
                }
            }
            None => None,
        };
        warnings.extend(info.warnings.iter().cloned());

        let mut info = info.clone();
        if f != (0..nf) {
            info.axes[F] = info.axes[F].select(f.clone());
            if !info.numbers.is_empty() {
                info.numbers = info.numbers[f.clone()].to_vec();
            }
            if !info.bins.is_empty() {
                info.bins = info.bins[f.clone()].to_vec();
            }
        }
        let shape = info.shape();
        let array =
            |v: Vec<f64>| ArrayD::from_shape_vec(IxDyn(&shape), v).expect("one value per bin");
        Ok(Tally::from_info(
            info,
            b.perturbation,
            array(result),
            array(rel_error),
            // Its bin is an index into the whole tally.
            tfc.filter(|_| f == (0..nf)),
            warnings,
        ))
    }

    /// Parse values `out.start..` of block `b` (of `total` values), in
    /// parallel, each thread reading the chunks it parses.
    fn read_vals(
        &self,
        opened: &Opened<'_>,
        b: &Block,
        total: u64,
        out: ValsOut<'_>,
    ) -> Result<Vec<Warning>, ReadError> {
        let (start, end) = (out.start, out.start + out.result.len() as u64);
        // The lines holding them: `lo..hi`, as rows of the file.
        let lo = b.first_row + start / 4;
        let hi = b.first_row + end.div_ceil(4).min(b.rows);
        let mut warnings = Vec::new();
        if end.div_ceil(4) > b.rows {
            warnings.push(Warning::new(format!(
                "{} lines of values where {} were due; the rest are NaN",
                b.rows,
                total.div_ceil(4)
            )));
        }
        if lo >= hi {
            return Ok(warnings);
        }
        let chunks = &self.chunks;
        let range = scan::chunks_of_rows(chunks, lo..hi);
        // The values each chunk's lines hold.
        let value_of = |row: u64| ((row - b.first_row) * 4).clamp(start, end);
        let lens = range.clone().map(|k| {
            let a = value_of(chunks[k].rows.clamp(lo, hi));
            let z = value_of(chunks[k + 1].rows.clamp(lo, hi));
            (z - a) as usize
        });
        let parts = out.split(lens);
        let found = range
            .into_par_iter()
            .zip(parts)
            .map(|(k, part)| {
                let text = opened.read(chunks[k].at..chunks[k + 1].at.min(b.end))?;
                let first = chunks[k].rows as i64 - b.first_row as i64;
                Ok(vals::parse_vals(&text, first, chunks[k].lines, total, part))
            })
            .collect::<Result<Vec<Vec<Warning>>, ReadError>>()?;
        warnings.extend(found.into_iter().flatten());
        Ok(warnings)
    }
}

/// Read tally `marks[range]` (its first mark its `tally` line): its header,
/// and where its values and fluctuation charts are.
fn index_tally(
    opened: &Opened<'_>,
    index: &Index<Kind>,
    range: Range<usize>,
    id: u32,
    perturbations: usize,
) -> Result<Result<Body, String>, ReadError> {
    let marks: &[Mark<Kind>] = &index.marks[range.clone()];
    let next = index.marks.get(range.end);
    let (end, end_row) = next.map_or((opened.len(), index.rows()), |m| (m.start, m.row));
    let tally = &marks[0];
    let Some(v) = marks.iter().position(|m| matches!(m.kind, Kind::Vals(_))) else {
        return Ok(Err("no vals line".to_string()));
    };
    let text = opened.read(tally.start..marks[v].start)?;
    let mut info =
        match header::read_tally_header(&String::from_utf8_lossy(&text), id, tally.line + 1) {
            Ok(info) => info,
            Err(e) => return Ok(Err(e)),
        };
    let lines_due = (info.len() as u64).div_ceil(4);
    let mut blocks = Vec::new();
    for (k, m) in marks.iter().enumerate().skip(v) {
        let Kind::Vals(perturbation) = m.kind else {
            continue;
        };
        let after = marks.get(k + 1);
        let (block_end, block_end_row) = after.map_or((end, end_row), |a| (a.start, a.row));
        let tfc = after
            .filter(|a| a.kind == Kind::Tfc)
            .map(|a| a.start..marks.get(k + 2).map_or(end, |b| b.start));
        let rows = block_end_row - m.row - 1;
        if rows > lines_due {
            info.warnings.push(Warning::at(
                m.line + 1,
                format!(
                    "{rows} lines of values where {lines_due} were due; the extra ones are ignored"
                ),
            ));
        }
        blocks.push(Block {
            perturbation,
            first_row: m.row + 1,
            rows: rows.min(lines_due),
            end: block_end,
            tfc,
        });
    }
    if blocks.len() != perturbations + 1 {
        info.warnings.push(Warning::at(
            tally.line + 1,
            format!(
                "{} vals blocks where the header's {perturbations} perturbations make {}",
                blocks.len(),
                perturbations + 1
            ),
        ));
    }
    Ok(Ok(Body { info, blocks }))
}

/// The bins of each tally card of `input`, by tally number (the first card
/// of a number; MCNP rejects two).
fn tally_cards(input: &Input) -> HashMap<u32, (usize, Result<Vec<TallyBin>, String>)> {
    let model = input.model();
    let mut out = HashMap::new();
    for card in model.data_cards() {
        if !card.mnemonic().is_some_and(|m| m.eq_ignore_ascii_case("f")) {
            continue;
        }
        let Some(id) = card.number().and_then(|n| u32::try_from(n).ok()) else {
            continue;
        };
        let line = model.line_of(card.slot()).unwrap_or(0);
        out.entry(id)
            .or_insert_with(|| (line, card.tally_bins().map_err(|e| e.message.clone())));
    }
    out
}

/// The number MCTAL gives bin `bin` of `bins`: the cell or surface of its
/// first level when that is a single one, and for the T bin the one every
/// bin's first level shares, if any (as MCNP 6.2 and 6.3 write it).
fn first_number(bin: &TallyBin, bins: &[TallyBin]) -> Option<i64> {
    match bin {
        TallyBin::Total => {
            let mut all = bins.iter().filter_map(|b| match b {
                TallyBin::Levels(levels) => levels.first(),
                TallyBin::Total => None,
            });
            let first = all.next()?.as_slice();
            let [one] = first else { return None };
            all.all(|l| l.as_slice() == first).then_some(one.number)
        }
        _ => bin.first().map(|r| r.number),
    }
}

/// Give `info` the bins of its card, when they are the file's: as many,
/// and the same first cell or surface where the file gives one.
fn label(info: &mut TallyInfo, card: Option<&(usize, Result<Vec<TallyBin>, String>)>) {
    let id = info.id;
    let problem = match card {
        // Detectors, TMESH: no cell or surface bins to read.
        _ if info.detector != super::model::Detector::None || info.mesh.is_some() => return,
        None => format!("the input has no F{id} card"),
        Some((line, Err(e))) => format!("the input's F{id} card (line {line}) can't be read: {e}"),
        Some((line, Ok(bins))) => {
            let n = info.axes[F].len();
            let firsts: Vec<Option<i64>> = bins.iter().map(|b| first_number(b, bins)).collect();
            let disagree = info
                .numbers
                .iter()
                .zip(&firsts)
                .position(|(num, first)| num.map(|n| n.number) != *first);
            if bins.len() != n {
                format!(
                    "the input's F{id} card (line {line}) makes {} bins, the file has {n}",
                    bins.len()
                )
            } else if let Some(k) = disagree {
                format!(
                    "bin {k} of the input's F{id} card (line {line}) is {}, the file says {:?}",
                    bins[k],
                    info.numbers[k].map(|n| n.number)
                )
            } else {
                info.bins = bins.clone();
                return;
            }
        }
    };
    info.warnings.push(Warning::new(format!(
        "{problem}; the bins are known only by number"
    )));
}
