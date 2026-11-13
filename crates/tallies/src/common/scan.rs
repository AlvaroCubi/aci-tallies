//! One pass over a file to find where things are in it, without holding it
//! in memory.
//!
//! The file is cut into chunks of whole lines, about [`CHUNK_BYTES`] each,
//! scanned in parallel (each thread reads its own chunk). For every chunk
//! the index keeps its byte offset and how many lines and non-blank lines
//! ("rows") come before it: a few hundred kB for a 10 GB file. Lines that
//! start (after spaces) with a letter are shown to a per-format classifier,
//! and the ones it recognizes become [`Mark`]s. Data rows start with a
//! digit, a sign or a point, so they cost a count and nothing else.
//!
//! Lines follow [`str::lines`]: split at `\n`, a trailing `\r` dropped. A
//! blank line (only ASCII whitespace) is never a row.

use std::ops::Range;

use rayon::prelude::*;

use super::source::Opened;

/// Bytes per chunk: far above rayon's per-task cost, while a multi-GB
/// section still splits into thousands of units for work stealing. Also
/// the unit later reads are done in, so it bounds the memory each thread
/// needs for text.
pub(crate) const CHUNK_BYTES: usize = 1 << 20;

/// The lines of `text`, as [`str::lines`] splits them.
pub(crate) fn lines(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        Some(match memchr::memchr(b'\n', rest) {
            Some(p) => {
                let line = &rest[..p];
                rest = &rest[p + 1..];
                line.strip_suffix(b"\r").unwrap_or(line)
            }
            None => std::mem::take(&mut rest),
        })
    })
}

pub(crate) fn is_blank(line: &[u8]) -> bool {
    line.iter().all(u8::is_ascii_whitespace)
}

/// The whitespace-separated tokens of `line`.
pub(crate) fn tokens(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(u8::is_ascii_whitespace)
        .filter(|t| !t.is_empty())
}

/// Byte offset of line `i` of `text` (counting blank ones), or the end of
/// `text` when it has no such line.
pub(crate) fn line_start(text: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    memchr::memchr_iter(b'\n', text)
        .nth(i - 1)
        .map_or(text.len(), |p| p + 1)
}

/// Where a chunk starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    /// Byte offset.
    pub at: u64,
    /// Rows (non-blank lines) before it.
    pub rows: u64,
    /// Lines before it.
    pub lines: u64,
}

/// A line the classifier recognized.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mark<K> {
    pub kind: K,
    /// Where the line starts, and where the next one does (bytes).
    pub start: u64,
    pub end: u64,
    /// Rows before it.
    pub row: u64,
    /// Lines before it (so it is line `line + 1` of the file).
    pub line: u64,
}

pub(crate) struct Index<K> {
    /// Every chunk, in order, then one for the end of the file.
    pub chunks: Vec<Chunk>,
    pub marks: Vec<Mark<K>>,
}

impl<K> Index<K> {
    /// Rows in the file.
    pub fn rows(&self) -> u64 {
        self.chunks.last().map_or(0, |c| c.rows)
    }
}

/// The chunks holding rows `rows` (rows of the file), as indices into
/// `chunks` (which ends with the end of the file, never in the result).
pub(crate) fn chunks_of_rows(chunks: &[Chunk], rows: Range<u64>) -> Range<usize> {
    if rows.start >= rows.end {
        return 0..0;
    }
    let c0 = chunks.partition_point(|c| c.rows <= rows.start) - 1;
    let c1 = chunks.partition_point(|c| c.rows < rows.end);
    c0..c1.min(chunks.len() - 1)
}

/// Index `src` in chunks of about `chunk` bytes. `classify` sees each line
/// whose first non-blank byte is a letter (without its line ending).
pub(crate) fn index<K: Send>(
    src: &Opened<'_>,
    chunk: usize,
    classify: impl Fn(&[u8]) -> Option<K> + Sync,
) -> std::io::Result<Index<K>> {
    let len = src.len();
    let chunk = chunk.max(1) as u64;
    let n = len.div_ceil(chunk);
    let scans: Vec<std::io::Result<Option<Scanned<K>>>> = (0..n)
        .into_par_iter()
        .map(|k| scan_part(src, k * chunk, ((k + 1) * chunk).min(len), &classify))
        .collect();
    let mut chunks = Vec::with_capacity(scans.len() + 1);
    let mut marks = Vec::new();
    let (mut rows, mut lines) = (0u64, 0u64);
    for s in scans {
        let Some((at, scan)) = s? else { continue };
        chunks.push(Chunk { at, rows, lines });
        marks.extend(scan.marks.into_iter().map(|m| Mark {
            kind: m.kind,
            start: at + m.start as u64,
            end: at + m.end as u64,
            row: rows + m.row as u64,
            line: lines + m.line as u64,
        }));
        rows += scan.rows as u64;
        lines += scan.lines as u64;
    }
    chunks.push(Chunk {
        at: len,
        rows,
        lines,
    });
    Ok(Index { chunks, marks })
}

/// A chunk's offset and scan.
type Scanned<K> = (u64, Scan<K>);

/// The lines starting in bytes `lo..hi` (a line belongs to the chunk where
/// it starts), scanned; `None` when no line starts there.
fn scan_part<K>(
    src: &Opened<'_>,
    lo: u64,
    hi: u64,
    classify: &(impl Fn(&[u8]) -> Option<K> + Sync),
) -> std::io::Result<Option<Scanned<K>>> {
    let len = src.len();
    // From the byte before `lo` (to tell whether a line starts at `lo`) to
    // a little past `hi` (to finish the last line).
    let from = lo.saturating_sub(1);
    let mut want = (hi - from) + (64 << 10);
    loop {
        let to = (from + want).min(len);
        let buf = src.read_upto(from..to)?;
        let line_at = |at: u64| -> Option<u64> {
            // The first line start at or after `at`.
            if at == 0 {
                return Some(0);
            }
            memchr::memchr(b'\n', &buf[(at - 1 - from) as usize..]).map(|p| at + p as u64)
        };
        let start = line_at(lo);
        let end = if hi == len { Some(len) } else { line_at(hi) };
        match (start, end) {
            (Some(start), _) if start >= hi => return Ok(None),
            (Some(start), Some(end)) => {
                let bytes = &buf[(start - from) as usize..(end - from) as usize];
                return Ok(Some((start, scan(bytes, classify))));
            }
            // A line runs past what was read: read more.
            _ if to < len => want *= 4,
            _ => {
                let start = start.unwrap_or(len);
                if start >= hi {
                    return Ok(None);
                }
                let bytes = &buf[(start - from) as usize..];
                return Ok(Some((start, scan(bytes, classify))));
            }
        }
    }
}

struct Scan<K> {
    rows: usize,
    lines: usize,
    marks: Vec<LocalMark<K>>,
}

struct LocalMark<K> {
    kind: K,
    start: usize,
    end: usize,
    row: usize,
    line: usize,
}

fn scan<K>(text: &[u8], classify: &impl Fn(&[u8]) -> Option<K>) -> Scan<K> {
    let mut s = Scan {
        rows: 0,
        lines: 0,
        marks: Vec::new(),
    };
    let base = text.as_ptr() as usize;
    for line in lines(text) {
        let first = line.iter().position(|b| !b.is_ascii_whitespace());
        if let Some(first) = first {
            if line[first].is_ascii_alphabetic() {
                if let Some(kind) = classify(line) {
                    let start = line.as_ptr() as usize - base;
                    let mut end = start + line.len();
                    if text.get(end) == Some(&b'\r') {
                        end += 1;
                    }
                    if text.get(end) == Some(&b'\n') {
                        end += 1;
                    }
                    s.marks.push(LocalMark {
                        kind,
                        start,
                        end,
                        row: s.rows,
                        line: s.lines,
                    });
                }
            }
            s.rows += 1;
        }
        s.lines += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source::Source;

    #[test]
    fn lines_like_str_lines() {
        for text in ["", "a", "a\n", "a\r\nb", "\n\n x\r\n", "a\nb\n\n", "a\r"] {
            let ours: Vec<&[u8]> = lines(text.as_bytes()).collect();
            let std: Vec<&[u8]> = text.lines().map(str::as_bytes).collect();
            assert_eq!(ours, std, "{text:?}");
        }
    }

    #[test]
    fn marks_and_counts_whatever_the_chunk_size() {
        let text = " Mesh 14\r\n\n  X  Rel\n 1 2 3\n\n 4 5\nMesh 2";
        let src = Source::bytes(text.as_bytes().to_vec());
        let opened = src.open().unwrap();
        let classify = |l: &[u8]| l.trim_ascii_start().first().copied();
        for chunk in 1..text.len() + 2 {
            let ix = index(&opened, chunk, classify).unwrap();
            assert_eq!(ix.rows(), 5, "chunk {chunk}");
            assert_eq!(ix.chunks.last().unwrap().lines, 7);
            let m: Vec<_> = ix
                .marks
                .iter()
                .map(|m| (m.kind, m.start, m.end, m.row, m.line))
                .collect();
            assert_eq!(
                m,
                [
                    (b'M', 0, 10, 0, 0),
                    (b'X', 11, 20, 1, 2),
                    (b'M', 33, 39, 4, 6)
                ],
                "chunk {chunk}"
            );
            for w in ix.chunks.windows(2) {
                assert!(w[0].at < w[1].at);
            }
            let c = chunks_of_rows(&ix.chunks, 2..3);
            let rows_in: u64 = ix.chunks[c.end].rows - ix.chunks[c.start].rows;
            assert!(ix.chunks[c.start].rows <= 2 && rows_in >= 1);
        }
        assert_eq!(line_start(text.as_bytes(), 2), 11);
        assert_eq!(line_start(text.as_bytes(), 99), text.len());
    }
}
