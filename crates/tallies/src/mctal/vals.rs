//! A tally's `vals` block: (value, relative error) pairs, four to a line,
//! in the order of the tally's flat index. Line `r` of a block holds values
//! `4r..4r + 4`, so any range of values is a range of lines, read and
//! parsed on its own.

use crate::common::{float, scan, Warning};

/// Where parsed values go: values `start..start + result.len()` of the
/// block (`NaN`-filled beforehand).
pub(crate) struct ValsOut<'a> {
    pub start: u64,
    pub result: &'a mut [f64],
    pub rel_error: &'a mut [f64],
}

impl<'a> ValsOut<'a> {
    /// Cut into consecutive parts of `lens` values each.
    pub fn split(self, lens: impl IntoIterator<Item = usize>) -> Vec<ValsOut<'a>> {
        let ValsOut {
            mut start,
            mut result,
            mut rel_error,
        } = self;
        let mut out = Vec::new();
        for n in lens {
            let (r, rest_r) = result.split_at_mut(n);
            let (e, rest_e) = rel_error.split_at_mut(n);
            out.push(ValsOut {
                start,
                result: r,
                rel_error: e,
            });
            start += n as u64;
            (result, rel_error) = (rest_r, rest_e);
        }
        out
    }
}

/// Parse the lines of `text` that hold values of `out`, given that its
/// first non-blank line is line `first_row` of the block (negative before
/// the block: only counted) and line `first_line + 1` of the file; the
/// block has `total` values. A damaged line leaves its values `NaN`, with
/// a warning, and no other line is affected.
pub(crate) fn parse_vals(
    text: &[u8],
    first_row: i64,
    first_line: u64,
    total: u64,
    out: ValsOut<'_>,
) -> Vec<Warning> {
    let mut warnings = Vec::new();
    let (start, end) = (out.start, out.start + out.result.len() as u64);
    if start >= end {
        return warnings;
    }
    let mut row = first_row;
    for (n, line) in scan::lines(text).enumerate() {
        if scan::is_blank(line) {
            continue;
        }
        let r = row;
        row += 1;
        if r < 0 {
            continue;
        }
        let first = r as u64 * 4;
        if first + 4 <= start {
            continue;
        }
        if first >= end {
            break;
        }
        let expected = (total.saturating_sub(first)).min(4) as usize;
        let mut tokens = scan::tokens(line);
        let mut bad = false;
        let mut found = 0;
        for i in 0..4 {
            let pair = (tokens.next(), tokens.next());
            let v = first + i as u64;
            match pair {
                (Some(a), Some(b)) => {
                    found += 1;
                    if !(start..end).contains(&v) {
                        continue;
                    }
                    match (float::parse(a), float::parse(b)) {
                        (Some(a), Some(b)) => {
                            let k = (v - start) as usize;
                            out.result[k] = a;
                            out.rel_error[k] = b;
                        }
                        _ => bad = true,
                    }
                }
                (None, None) => break,
                _ => bad = true,
            }
        }
        if tokens.next().is_some() || found != expected {
            bad = true;
        }
        if bad {
            warnings.push(Warning::at(
                first_line + n as u64 + 1,
                format!(
                    "values {first}..{}: expected {expected} value and error pairs: {:?}",
                    first + expected as u64,
                    String::from_utf8_lossy(line).trim()
                ),
            ));
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(
        text: &str,
        first_row: i64,
        total: u64,
        range: std::ops::Range<u64>,
    ) -> (Vec<f64>, Vec<Warning>) {
        let n = (range.end - range.start) as usize;
        let mut r = vec![f64::NAN; n];
        let mut e = vec![f64::NAN; n];
        let w = parse_vals(
            text.as_bytes(),
            first_row,
            0,
            total,
            ValsOut {
                start: range.start,
                result: &mut r,
                rel_error: &mut e,
            },
        );
        (r, w)
    }

    #[test]
    fn lines_to_values() {
        let text =
            "  1.0E+00 0.1  2.0E+00 0.1  3.0E+00 0.1  4.0E+00 0.1\n  5.0E+00 0.1  6.0-113 0.0\n";
        let (r, w) = parse(text, 0, 6, 0..6);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(r, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0e-113]);
        let (r, _) = parse(text, 0, 6, 3..5);
        assert_eq!(r, [4.0, 5.0]);
        // Starting before the block: those lines are only counted.
        let (r, _) = parse(&format!("vals\n{text}"), -1, 6, 4..6);
        assert_eq!(r, [5.0, 6.0e-113]);
    }

    #[test]
    fn damage_stays_on_its_line() {
        let text = "  1.0E+00 0.1  2.0E+00 0.1  3.0E+00 0.1\n  5.0E+00 0.1  6.0E+00 x\n";
        let (r, w) = parse(text, 0, 6, 0..6);
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!(w[0].line, Some(1));
        assert_eq!(&r[..3], [1.0, 2.0, 3.0]);
        assert!(r[3].is_nan() && r[5].is_nan());
        assert_eq!(r[4], 5.0);
    }
}
