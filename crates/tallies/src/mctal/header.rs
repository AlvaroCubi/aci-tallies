//! The text of an MCTAL file other than its values: the file header, each
//! tally's header (particles, comment, bins), its fluctuation chart and the
//! KCODE block. As MCNP 6.2 and 6.3 write them (checked against both),
//! with older forms read where they differ: MCNP5's
//! particle number instead of a flags line, and the header without code,
//! version and date that a negative `MCTAL` value on `PRDMP` gives.

use ndarray::Array2;

use super::model::{
    Detector, FileHeader, Kcode, MeshGrid, Modifier, Number, TallyInfo, Tfc, DIMS, PARTICLES,
};
use crate::common::{Axis, Bins, Warning};

fn int(tok: &str) -> Option<i64> {
    tok.parse().ok()
}

fn number(tok: &str) -> Option<f64> {
    crate::common::float::parse(tok.as_bytes())
}

/// Read the lines before the first tally.
pub(crate) fn read_file_header(text: &str) -> Result<FileHeader, String> {
    let mut lines = text.lines();
    let mut h = FileHeader::default();
    let first = lines.next().ok_or("the file is empty")?;
    let words: Vec<&str> = first.split_whitespace().collect();
    // The last three are the dump, histories and random numbers; before
    // them the code, version and date, when the file has them.
    if words.len() < 3 {
        return Err(format!("unreadable first line {first:?}"));
    }
    let (names, counts) = words.split_at(words.len() - 3);
    let count = |s: &str| -> Result<u64, String> {
        s.trim_end_matches('.')
            .parse()
            .map_err(|_| format!("unreadable first line {first:?}"))
    };
    h.dump = count(counts[0])?;
    h.histories = count(counts[1])?;
    h.randoms = count(counts[2]).unwrap_or(0);
    if let Some((code, rest)) = names.split_first() {
        h.code = code.to_string();
        if let Some((version, date)) = rest.split_first() {
            h.version = version.to_string();
            h.probid = date.join(" ");
        }
    }
    h.title = lines.next().unwrap_or_default().trim().to_string();
    let ntal = lines.next().unwrap_or_default();
    let words: Vec<&str> = ntal.split_whitespace().collect();
    if words.first() != Some(&"ntal") {
        return Err(format!("expected the ntal line, found {ntal:?}"));
    }
    let n = words
        .get(1)
        .and_then(|w| w.parse::<usize>().ok())
        .ok_or_else(|| format!("unreadable ntal line {ntal:?}"))?;
    if let Some(p) = words.iter().position(|w| *w == "npert") {
        h.perturbations = words.get(p + 1).and_then(|w| w.parse().ok()).unwrap_or(0);
    }
    for line in lines {
        for w in line.split_whitespace() {
            h.tallies.push(
                w.parse()
                    .map_err(|_| format!("unreadable tally number {w:?}"))?,
            );
        }
    }
    if h.tallies.len() != n {
        return Err(format!(
            "ntal says {n} tallies but {} numbers follow",
            h.tallies.len()
        ));
    }
    Ok(h)
}

/// A line that starts one of the eight bin dimensions: its index in
/// [`DIMS`], and its flag (`t` Total, `c` cumulative).
fn bin_keyword(line: &str) -> Option<(usize, Option<char>)> {
    if !line.starts_with(|c: char| c.is_ascii_lowercase()) {
        return None;
    }
    let word = line.split_whitespace().next()?;
    let mut chars = word.chars();
    let first = chars.next()?;
    let dim = DIMS.iter().position(|d| d.starts_with(first))?;
    let flag = chars.next();
    match (flag, chars.next()) {
        (None, None) => Some((dim, None)),
        (Some(f @ ('t' | 'c')), None) => Some((dim, Some(f))),
        _ => None,
    }
}

/// Whether `line` is a particle flags line: one `0` or `1` per particle.
fn flags_line(line: &str) -> bool {
    line.starts_with(' ')
        && line.split_whitespace().count() > 0
        && line.split_whitespace().all(|w| w == "0" || w == "1")
}

/// Read a tally's header: its lines from `tally` to before `vals`.
/// `line` is the line number (from 1) of its `tally` line, for warnings.
pub(crate) fn read_tally_header(text: &str, id: u32, line: u64) -> Result<TallyInfo, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut warnings = Vec::new();
    let head: Vec<i64> = lines[0]
        .split_whitespace()
        .skip(1)
        .map(|w| int(w).ok_or_else(|| format!("unreadable tally line {:?}", lines[0])))
        .collect::<Result<_, _>>()?;
    if head.first().copied() != Some(i64::from(id)) && head.len() >= 4 {
        warnings.push(Warning::at(
            line,
            format!("the tally line says {}, the ntal list {id}", head[0]),
        ));
    }
    let mut at = 1;
    let mut particles = Vec::new();
    match head.get(1).copied() {
        Some(n) if n < 0 => {
            let mut flags = Vec::new();
            while at < lines.len() && flags_line(lines[at]) {
                flags.extend(lines[at].split_whitespace().map(|w| w == "1"));
                at += 1;
            }
            particles = flags
                .iter()
                .zip(PARTICLES)
                .filter_map(|(&on, p)| on.then_some(p))
                .collect();
            if particles.len() as i64 != -n {
                warnings.push(Warning::at(
                    line,
                    format!(
                        "{} particles flagged, the tally line says {}",
                        particles.len(),
                        -n
                    ),
                ));
            }
        }
        // MCNP5: 1 n, 2 p, 4 e, summed.
        Some(code) => {
            for (bit, p) in [(1, 'n'), (2, 'p'), (4, 'e')] {
                if code & bit != 0 {
                    particles.push(p);
                }
            }
        }
        None => return Err(format!("unreadable tally line {:?}", lines[0])),
    }
    let detector_code = head.get(2).copied().unwrap_or(0);
    let modifier = match head.get(3).copied().unwrap_or(0) {
        1 => Modifier::Star,
        2 => Modifier::Plus,
        _ => Modifier::None,
    };

    // The comment: lines up to the first bin line, 75 characters each
    // after 5 spaces (a long comment is cut anywhere, mid-word too).
    let mut comment = String::new();
    while at < lines.len() && bin_keyword(lines[at]).is_none() {
        let l = lines[at];
        let piece = l.get(5..).unwrap_or("");
        comment.push_str(piece);
        if piece.len() < 75 {
            comment.extend(std::iter::repeat_n(' ', 75 - piece.len()));
        }
        at += 1;
    }
    let comment = comment.trim();
    let comment = (!comment.is_empty()).then(|| comment.to_string());

    // The bins: a keyword line for each dimension, in order, then any
    // numbers it lists.
    // (dimension, flag, the keyword line's words, the numbers after it)
    type Found<'a> = (usize, Option<char>, Vec<&'a str>, Vec<&'a str>);
    let mut found: Vec<Found> = Vec::new();
    for (n, l) in lines.iter().enumerate().skip(at) {
        match bin_keyword(l) {
            Some((dim, flag)) => {
                if dim != found.len() {
                    return Err(format!(
                        "line {}: expected the {} bins, found {l:?}",
                        line + n as u64,
                        DIMS.get(found.len()).unwrap_or(&"no more")
                    ));
                }
                let words: Vec<&str> = l.split_whitespace().skip(1).collect();
                found.push((dim, flag, words, Vec::new()));
            }
            None => match found.last_mut() {
                Some(last) => last.3.extend(l.split_whitespace()),
                None => return Err(format!("line {}: unexpected {l:?}", line + n as u64)),
            },
        }
    }
    if found.len() != 8 {
        return Err(format!(
            "the {} bins are missing",
            DIMS.get(found.len()).unwrap_or(&"?")
        ));
    }

    let detector = match Detector::from_code(detector_code) {
        Some(d) => d,
        None if head.len() == 3 => Detector::None, // TMESH: geometry here
        None => {
            warnings.push(Warning::at(
                line,
                format!("unknown detector type {detector_code}"),
            ));
            Detector::None
        }
    };
    let mut numbers = Vec::new();
    let mut mesh = None;
    let mut axes: Vec<Axis> = Vec::with_capacity(8);
    for (dim, flag, words, values) in found {
        let count = words
            .first()
            .and_then(|w| int(w))
            .filter(|&n| n >= 0)
            .ok_or_else(|| format!("unreadable {} bins line", DIMS[dim]))?
            as usize;
        if dim == 0 && words.len() >= 5 {
            // A TMESH mesh tally: `f cells 0 n1 n2 n3`, then the grid.
            let n: Vec<usize> = words[2..5].iter().filter_map(|w| w.parse().ok()).collect();
            let v: Vec<f64> = values.iter().filter_map(|w| number(w)).collect();
            if n.len() == 3 && v.len() == n.iter().map(|k| k + 1).sum::<usize>() {
                let (a, rest) = v.split_at(n[0] + 1);
                let (b, c) = rest.split_at(n[1] + 1);
                mesh = Some(MeshGrid {
                    geometry: (-head.get(2).copied().unwrap_or(-1)).clamp(1, 3) as u8,
                    bounds: [a.to_vec(), b.to_vec(), c.to_vec()],
                });
            } else {
                warnings.push(Warning::at(line, "unreadable TMESH mesh grid"));
            }
            axes.push(Axis {
                bins: Bins::Count(count.max(1)),
                total: false,
                cumulative: false,
            });
            continue;
        }
        if dim == 0 {
            if !values.is_empty() {
                let parsed: Option<Vec<Option<Number>>> =
                    values.iter().map(|w| f_number(w)).collect();
                match parsed {
                    Some(list) if list.len() == count => numbers = list,
                    _ => warnings.push(Warning::at(
                        line,
                        format!(
                            "{} cell or surface numbers for {count} f bins; left out",
                            values.len()
                        ),
                    )),
                }
            }
            axes.push(Axis {
                bins: Bins::Count(count.max(1)),
                total: false,
                cumulative: false,
            });
            continue;
        }
        let total = flag == Some('t');
        let cumulative = flag == Some('c');
        // `e 0`: one bin, the whole range (MCNP prints 0 for it).
        let n_bins = if count == 0 {
            1
        } else {
            count - usize::from(total)
        };
        let v: Option<Vec<f64>> = values.iter().map(|w| number(w)).collect();
        let bins = match v {
            Some(v) if v.is_empty() => Bins::Count(n_bins),
            Some(v) if v.len() == n_bins => Bins::Upper(v),
            Some(v) if v.len() == n_bins + 1 => Bins::Bounds(v),
            _ => {
                warnings.push(Warning::at(
                    line,
                    format!(
                        "{} {} bounds for {n_bins} bins; left out",
                        values.len(),
                        DIMS[dim]
                    ),
                ));
                Bins::Count(n_bins)
            }
        };
        axes.push(Axis {
            bins,
            total: total && count > 0,
            cumulative,
        });
    }
    let axes: [Axis; 8] = axes.try_into().expect("eight dimensions");
    Ok(TallyInfo {
        id,
        particles,
        detector,
        modifier,
        comment,
        axes,
        numbers,
        bins: Vec::new(),
        mesh,
        warnings,
    })
}

/// A number on an `f` line: a cell or surface (`0` for none), or a
/// macrobody facet `surface.facet`.
fn f_number(word: &str) -> Option<Option<Number>> {
    let (n, facet) = match word.split_once('.') {
        Some((n, f)) => (n, Some(f.parse::<u8>().ok()?)),
        None => (word, None),
    };
    let number: i64 = n.parse().ok()?;
    Some((number != 0).then_some(Number { number, facet }))
}

/// Read a fluctuation chart: its `tfc` line and rows.
pub(crate) fn read_tfc(text: &str) -> Result<Tfc, String> {
    let mut lines = text.lines();
    let head: Vec<i64> = lines
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .skip(1)
        .filter_map(int)
        .collect();
    if head.len() != 9 {
        return Err("unreadable tfc line".into());
    }
    let bin: [usize; 8] = std::array::from_fn(|k| (head[k + 1].max(1) - 1) as usize);
    let mut tfc = Tfc {
        bin,
        histories: Vec::new(),
        mean: Vec::new(),
        rel_error: Vec::new(),
        fom: None,
    };
    let mut fom = Vec::new();
    for l in lines.take(head[0].max(0) as usize) {
        let w: Vec<&str> = l.split_whitespace().collect();
        let nps = w.first().and_then(|w| w.parse().ok());
        let values: Option<Vec<f64>> = w[1.min(w.len())..].iter().map(|w| number(w)).collect();
        match (nps, values) {
            (Some(nps), Some(v)) if v.len() == 2 || v.len() == 3 => {
                tfc.histories.push(nps);
                tfc.mean.push(v[0]);
                tfc.rel_error.push(v[1]);
                if let Some(&f) = v.get(2) {
                    fom.push(f);
                }
            }
            _ => return Err(format!("unreadable tfc row {l:?}")),
        }
    }
    if !fom.is_empty() {
        tfc.fom = Some(fom);
    }
    Ok(tfc)
}

/// Read the KCODE block: `kcode cycles inactive columns`, then the values.
pub(crate) fn read_kcode(text: &str) -> Result<Kcode, String> {
    let mut lines = text.lines();
    let head: Vec<usize> = lines
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .skip(1)
        .filter_map(|w| w.parse().ok())
        .collect();
    let [cycles, inactive, columns] = head[..] else {
        return Err("unreadable kcode line".into());
    };
    let values: Option<Vec<f64>> = lines.flat_map(str::split_whitespace).map(number).collect();
    let values = values.ok_or("unreadable kcode values")?;
    if columns == 0 || values.len() % columns != 0 {
        return Err(format!(
            "{} kcode values, not rows of {columns}",
            values.len()
        ));
    }
    let rows = values.len() / columns;
    Ok(Kcode {
        cycles,
        inactive,
        values: Array2::from_shape_vec((rows, columns), values).expect("whole rows"),
    })
}

/// Whether `head` (the first few kB of a file) looks like an MCTAL file:
/// an `ntal` line third, after the header and title lines.
pub fn is_mctal(head: &[u8]) -> bool {
    head.split(|&b| b == b'\n')
        .nth(2)
        .is_some_and(|l| l.starts_with(b"ntal "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords() {
        assert_eq!(bin_keyword("f        4"), Some((0, None)));
        assert_eq!(bin_keyword("et       4"), Some((6, Some('t'))));
        assert_eq!(bin_keyword("ec       3"), Some((6, Some('c'))));
        assert_eq!(bin_keyword("t        0"), Some((7, None)));
        assert_eq!(bin_keyword("     f comment"), None);
        assert_eq!(bin_keyword("tally 4"), None);
        assert_eq!(bin_keyword("tfc 1"), None);
    }

    #[test]
    fn f_numbers() {
        assert_eq!(
            f_number("12"),
            Some(Some(Number {
                number: 12,
                facet: None
            }))
        );
        assert_eq!(f_number("0"), Some(None));
        assert_eq!(
            f_number("101.3"),
            Some(Some(Number {
                number: 101,
                facet: Some(3)
            }))
        );
        assert_eq!(f_number("x"), None);
    }

    #[test]
    fn headers_with_and_without_code() {
        let h = read_file_header(
            "mcnp       6.mpi 10/23/26 10:42:57     2              10               0\n \
             a title\nntal     3 npert 2\n    1    2\n    4\n",
        )
        .unwrap();
        assert_eq!(
            (h.code.as_str(), h.version.as_str(), h.probid.as_str()),
            ("mcnp", "6.mpi", "10/23/26 10:42:57")
        );
        assert_eq!(
            (h.dump, h.histories, h.randoms, h.perturbations),
            (2, 10, 0, 2)
        );
        assert_eq!(h.tallies, [1, 2, 4]);
        let h = read_file_header("                                       2          500000        16304559\n t\nntal 1\n 5\n").unwrap();
        assert_eq!((h.code.as_str(), h.histories), ("", 500000));
        assert!(read_file_header("x\ny\nntal 2\n 1\n").is_err());
        assert!(is_mctal(b"mcnp 6\n title\nntal     1\n"));
        assert!(!is_mctal(
            b"mcnp6 version 6\n title\n Number of histories\n"
        ));
    }

    #[test]
    fn a_comment_cut_mid_word() {
        let piece = "x".repeat(70) + "abcde";
        let text = format!("tally    4                   -1    0    0\n 1 0 0\n     {piece}\n     fgh\nf        1\n      3\nd 1\nu 0\ns 0\nm 0\nc 0\ne 0\nt 0\n");
        let t = read_tally_header(&text, 4, 1).unwrap();
        assert_eq!(t.comment.unwrap(), format!("{}abcdefgh", "x".repeat(70)));
        assert_eq!(t.particles, ['n']);
        assert_eq!(
            t.numbers,
            [Some(Number {
                number: 3,
                facet: None
            })]
        );
        assert!(t.warnings.is_empty(), "{:?}", t.warnings);
    }
}
