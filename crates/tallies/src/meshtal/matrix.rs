//! The matrix layouts (`out=ij`, `out=ik`, `out=jk`).
//!
//! Confirmed against MCNP 6.2 and 6.3 output, rectangular and cylindrical:
//!
//! ```text
//! Energy Bin:  <lo> - <hi> MeV          (or "Total Energy Bin")
//! Time Bin:  <lo> - <hi> shakes         (only with TMESH; or "Total Time Bin")
//! <O> bin:  <lo> - <hi>                 (one block per bin of the outer axis O)
//!      Tally Results:  <A> (across) by <D> (down)
//!      <A bin centres>
//!      <D centre> <value> ...           (one row per D bin)
//!      Relative Errors
//!      ...the same shape...
//! ```
//!
//! `A`, `D` and `O` are the mesh's three axes, named `X`/`Y`/`Z` or
//! `R`/`Z`/`Theta`: `ij` loops over the third axis, `ik` the second, `jk`
//! the first. The roles are read from the `(across) by (down)` label, since
//! `Z` is the outer axis of a rectangular `ij` but the down axis of a
//! cylindrical one. MCNP's Total blocks (when an axis has more than one
//! bin) become one more index at the end of that axis, as in the columnar
//! layouts.
//!
//! Blocks are found from their marker lines when the file is indexed, each
//! spanning up to the next marker, so a damaged block only affects itself,
//! and each is read and parsed on its own.

use std::ops::Range;

use crate::common::{float, scan, Warning};

use super::model::{Geometry, OutFormat};

/// MCNP's names for the mesh axes in matrix labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Name {
    X,
    Y,
    Z,
    R,
    Theta,
}

impl Name {
    const ALL: [(Name, &'static [u8]); 5] = [
        (Name::X, b"X"),
        (Name::Y, b"Y"),
        (Name::Z, b"Z"),
        (Name::R, b"R"),
        (Name::Theta, b"Theta"),
    ];

    /// The name an outer-bin line (`Z bin:`, `Theta bin (revolutions):`)
    /// starts with.
    pub fn of_outer_line(trimmed: &[u8]) -> Option<Name> {
        Name::ALL.iter().find_map(|&(name, text)| {
            trimmed
                .strip_prefix(text)
                .filter(|rest| rest.starts_with(b" bin"))
                .map(|_| name)
        })
    }

    fn of_label(label: &str) -> Option<Name> {
        Name::ALL
            .iter()
            .find(|(_, text)| text == &label.as_bytes())
            .map(|&(name, _)| name)
    }
}

/// The names of a mesh's `(i, j, k)` axes.
fn names(geometry: &Geometry) -> [Name; 3] {
    match geometry {
        Geometry::Rectangular { .. } => [Name::X, Name::Y, Name::Z],
        Geometry::Cylindrical { .. } => [Name::R, Name::Z, Name::Theta],
    }
}

/// Which spatial axis (0 = i, 1 = j, 2 = k) runs across a matrix's
/// columns, down its rows, and over the blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    pub across: usize,
    pub down: usize,
    pub outer: usize,
}

impl Layout {
    pub fn out_format(self) -> OutFormat {
        match self.outer {
            2 => OutFormat::Ij,
            1 => OutFormat::Ik,
            _ => OutFormat::Jk,
        }
    }

    /// The `[i, j, k]` of column `col`, row `row` of outer block `outer`.
    pub fn ijk(self, col: usize, row: usize, outer: usize) -> [usize; 3] {
        let mut ijk = [0; 3];
        ijk[self.across] = col;
        ijk[self.down] = row;
        ijk[self.outer] = outer;
        ijk
    }

    /// From the `Tally Results:  <A> (across) by <D> (down)` line in
    /// `text`, if it has one: `Ok(None)` when it has none.
    pub fn find(geometry: &Geometry, text: &[u8]) -> Result<Option<Layout>, String> {
        let Some(line) =
            scan::lines(text).find(|l| memchr::memmem::find(l, b"Tally Results:").is_some())
        else {
            return Ok(None);
        };
        let line = String::from_utf8_lossy(line);
        let labels = line
            .split_once("Tally Results:")
            .and_then(|(_, rest)| rest.split_once("(across)"))
            .and_then(|(across, rest)| {
                let down = rest
                    .trim_start()
                    .strip_prefix("by")?
                    .split_once("(down)")?
                    .0;
                Some((across.trim(), down.trim()))
            });
        let names = names(geometry);
        let position = |label: &str| {
            let name = Name::of_label(label)?;
            names.iter().position(|n| *n == name)
        };
        match labels.and_then(|(a, d)| Some((position(a)?, position(d)?))) {
            Some((across, down)) if across != down => Ok(Some(Layout {
                across,
                down,
                outer: 3 - across - down,
            })),
            _ => Err(format!(
                "unrecognized matrix layout line for a {} mesh: {:?}",
                match geometry {
                    Geometry::Rectangular { .. } => "X/Y/Z",
                    Geometry::Cylindrical { .. } => "R/Z/Theta",
                },
                line.trim()
            )),
        }
    }

    pub fn outer_name(self, geometry: &Geometry) -> Name {
        names(geometry)[self.outer]
    }
}

/// A block-structure line of a section's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Marker {
    Energy { total: bool },
    Time { total: bool },
    Outer(Name),
}

/// A marker line where it is in the file: its bytes and how many lines
/// come before it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Marked {
    pub marker: Marker,
    pub start: u64,
    pub end: u64,
    pub line: u64,
}

/// One `(energy, time, outer bin)` block.
#[derive(Debug, Clone)]
pub(crate) struct Block {
    pub energy: usize,
    pub time: usize,
    pub outer: usize,
    /// From the line after its outer-bin marker to the next marker (or the
    /// section's end): never into a neighbour, however damaged it is.
    pub bytes: Range<u64>,
    /// Lines before `bytes`.
    pub line: u64,
}

pub(crate) struct Blocks {
    pub blocks: Vec<Block>,
    pub energy_total: bool,
    pub time_total: bool,
    pub warnings: Vec<Warning>,
}

/// Give every outer-bin block its `(energy, time, outer)` index by walking
/// the markers (only those of the layout's outer axis) in file order: an
/// energy marker starts the next energy bin (or the Total one) and resets
/// the time and outer counters, a time marker does the same one level
/// down. Blocks that don't fit the expected counts, or repeat one already
/// seen, are left out with a warning; expected blocks that never appear are reported and left NaN.
/// As every energy and time marker resets the counters below it, one
/// missing block never shifts later ones.
pub(crate) fn index_blocks(
    markers: &[Marked],
    data_end: u64,
    n_energy: usize,
    n_time: usize,
    n_outer: usize,
) -> Result<Blocks, String> {
    if !markers
        .iter()
        .any(|m| matches!(m.marker, Marker::Energy { .. }))
    {
        return Err("no \"Energy Bin:\" line found in matrix data block".to_string());
    }
    let has_time = markers
        .iter()
        .any(|m| matches!(m.marker, Marker::Time { .. }));
    let energy_total = markers
        .iter()
        .any(|m| m.marker == Marker::Energy { total: true });
    let time_total = markers
        .iter()
        .any(|m| m.marker == Marker::Time { total: true });
    let n_e_slots = n_energy + usize::from(energy_total);
    let n_t_slots = n_time + usize::from(time_total);

    let mut warnings = Vec::new();
    if !has_time && n_time > 1 {
        warnings.push(Warning::new(format!(
            "header declares {n_time} time bins but the matrix block has no \"Time Bin:\" markers"
        )));
    }

    let mut blocks = Vec::new();
    let mut seen = vec![false; n_e_slots * n_t_slots * n_outer];
    let mut unexpected = 0usize;
    let (mut next_e, mut next_t) = (0usize, 0usize);
    let mut energy: Option<usize> = None;
    let mut time: Option<usize> = if has_time { None } else { Some(0) };
    let mut outer = 0usize;

    for (n, m) in markers.iter().enumerate() {
        match m.marker {
            Marker::Energy { total } => {
                energy = Some(if total { n_energy } else { next_e });
                next_e += usize::from(!total);
                if has_time {
                    time = None;
                    next_t = 0;
                }
                outer = 0;
            }
            Marker::Time { total } => {
                time = Some(if total { n_time } else { next_t });
                next_t += usize::from(!total);
                outer = 0;
            }
            Marker::Outer(_) => {
                match (energy, time) {
                    (Some(e), Some(t))
                        if e < n_e_slots
                            && t < n_t_slots
                            && outer < n_outer
                            && !seen[(e * n_t_slots + t) * n_outer + outer] =>
                    {
                        seen[(e * n_t_slots + t) * n_outer + outer] = true;
                        blocks.push(Block {
                            energy: e,
                            time: t,
                            outer,
                            bytes: m.end..markers.get(n + 1).map_or(data_end, |next| next.start),
                            line: m.line + 1,
                        });
                    }
                    _ => unexpected += 1,
                }
                outer += 1;
            }
        }
    }

    if unexpected > 0 {
        warnings.push(Warning::new(format!(
            "{unexpected} matrix block(s) don't fit the header's {n_e_slots} energy x \
             {n_t_slots} time x {n_outer} outer bins and were ignored"
        )));
    }
    let missing: Vec<usize> = (0..seen.len()).filter(|&s| !seen[s]).collect();
    if let Some(&first) = missing.first() {
        let (e, t, o) = (
            first / (n_t_slots * n_outer),
            (first / n_outer) % n_t_slots,
            first % n_outer,
        );
        warnings.push(Warning::new(format!(
            "{} of {} matrix blocks missing (first: energy {e}, time {t}, outer bin {o}); \
             their voxels are NaN",
            missing.len(),
            seen.len()
        )));
    }

    Ok(Blocks {
        blocks,
        energy_total,
        time_total,
        warnings,
    })
}

/// A block's two matrices: `n_rows * n_cols` values each, addressed
/// `[row * n_cols + col]`.
pub(crate) struct Parsed {
    pub result: Vec<f64>,
    pub rel_error: Vec<f64>,
    pub warnings: Vec<Warning>,
}

/// Parse block `block` from its text.
pub(crate) fn parse_block(text: &[u8], block: &Block, n_cols: usize, n_rows: usize) -> Parsed {
    let mut lines = scan::lines(text)
        .enumerate()
        .map(|(n, l)| (block.line + n as u64 + 1, l));
    let mut warnings = Vec::new();
    let mut result = vec![f64::NAN; n_cols * n_rows];
    let mut rel_error = vec![f64::NAN; n_cols * n_rows];
    let at = BlockId(block.energy, block.time, block.outer);
    let mut matrix = |needle: &str, out: &mut [f64], warnings: &mut Vec<Warning>| {
        read_matrix(&mut lines, needle, n_cols, n_rows, at, warnings, out)
    };
    if matrix("Tally Results:", &mut result, &mut warnings) {
        matrix("Relative Errors", &mut rel_error, &mut warnings);
    }
    Parsed {
        result,
        rel_error,
        warnings,
    }
}

/// A block's position, for warnings.
#[derive(Clone, Copy)]
struct BlockId(usize, usize, usize);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "energy {}, time {}, outer bin {}",
            self.0, self.1, self.2
        )
    }
}

/// Skip to the header line of one matrix (`Tally Results:` or `Relative
/// Errors`) and past the column centres after it, then read `n_rows` rows
/// of a row centre and `n_cols` values into `out`. `false` when the header
/// line isn't there.
fn read_matrix<'a>(
    lines: &mut impl Iterator<Item = (u64, &'a [u8])>,
    needle: &str,
    n_cols: usize,
    n_rows: usize,
    at: BlockId,
    warnings: &mut Vec<Warning>,
    out: &mut [f64],
) -> bool {
    if !lines.any(|(_, l)| memchr::memmem::find(l, needle.as_bytes()).is_some()) {
        warnings.push(Warning::new(format!("{at}: missing \"{needle}\" block")));
        return false;
    }
    lines.next(); // column centres: the bounds come from the header
    for row in 0..n_rows {
        let Some((line_no, line)) = lines.next() else {
            warnings.push(Warning::new(format!(
                "{at}: \"{needle}\" ended early at row {row}"
            )));
            break;
        };
        let n_tokens = scan::tokens(line).count();
        if n_tokens != n_cols + 1 {
            warnings.push(Warning::at(
                line_no,
                format!(
                    "{at}: \"{needle}\" row {row} has {n_tokens} fields, expected {} \
                     (fixed-width columns run together?)",
                    n_cols + 1
                ),
            ));
            continue;
        }
        // The first token is the row centre, from the header too.
        for (col, tok) in scan::tokens(line).skip(1).enumerate() {
            match float::parse(tok) {
                Some(v) => out[row * n_cols + col] = v,
                None => warnings.push(Warning::at(
                    line_no,
                    format!(
                        "{at}: \"{needle}\" row {row} col {col} is not a number: {:?}",
                        String::from_utf8_lossy(tok)
                    ),
                )),
            }
        }
    }
    true
}

/// Round trips: render random arrays in MCNP's matrix text (random sizes,
/// layouts, energy and time bins and Total blocks), read them back, and
/// compare exactly. Values go through the same formatting on both sides,
/// so this checks the structure; `tests/` checks real MCNP output.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Source;
    use crate::meshtal::{Reader, Tally};
    use ndarray::Array5;
    use proptest::prelude::*;

    const LAYOUTS: [Layout; 3] = [
        Layout {
            across: 0,
            down: 1,
            outer: 2,
        }, // ij
        Layout {
            across: 0,
            down: 2,
            outer: 1,
        }, // ik
        Layout {
            across: 1,
            down: 2,
            outer: 0,
        }, // jk
    ];

    #[derive(Debug, Clone, Copy)]
    struct Case {
        dims: [usize; 3],
        cylindrical: bool,
        layout: Layout,
        n_energy: usize,
        /// `None`: no TMESH (no time markers at all).
        n_time: Option<usize>,
    }

    impl Case {
        fn e_total(&self) -> bool {
            self.n_energy > 1
        }

        fn t_total(&self) -> bool {
            self.n_time.is_some_and(|n| n > 1)
        }

        fn shape(&self) -> (usize, usize, usize, usize, usize) {
            (
                self.n_energy + usize::from(self.e_total()),
                self.n_time.unwrap_or(1) + usize::from(self.t_total()),
                self.dims[0],
                self.dims[1],
                self.dims[2],
            )
        }

        fn index(&self, e: usize, t: usize, col: usize, row: usize, outer: usize) -> [usize; 5] {
            let [i, j, k] = self.layout.ijk(col, row, outer);
            [e, t, i, j, k]
        }

        fn header(&self) -> String {
            let b = |n: usize| {
                (0..=n)
                    .map(|i| format!("{i}.00"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let [n1, n2, n3] = self.dims;
            let mut s = " Mesh Tally Number 4\n neutron  mesh tally.\n\n Tally bin boundaries:\n"
                .to_string();
            if self.cylindrical {
                s += "  origin at 0.00E+00 0.00E+00 0.00E+00 axis in 0.00E+00 0.00E+00 1.00E+00 \
                      direction, VEC direction 1.00E+00 0.00E+00 0.00E+00\n";
                let theta = (0..=n3)
                    .map(|i| format!("{:.4}", i as f64 / n3 as f64))
                    .collect::<Vec<_>>();
                s += &format!(
                    "    R direction: {}\n    Z direction: {}\n    \
                               Theta direction (revolutions): {}\n",
                    b(n1),
                    b(n2),
                    theta.join(" ")
                );
            } else {
                s += &format!(
                    "    X direction: {}\n    Y direction: {}\n    Z direction: {}\n",
                    b(n1),
                    b(n2),
                    b(n3)
                );
            }
            if let Some(n) = self.n_time {
                s += &format!("    Time bin boundaries: {}\n", b(n));
            }
            s += &format!("    Energy bin boundaries: {}\n\n", b(self.n_energy));
            s
        }

        fn render(&self, result: &Array5<f64>, rel_error: &Array5<f64>) -> String {
            let names = if self.cylindrical {
                ["R", "Z", "Theta"]
            } else {
                ["X", "Y", "Z"]
            };
            let l = self.layout;
            let (n_e, n_t, ..) = self.shape();
            let mut s = self.header();
            for e in 0..n_e {
                if self.e_total() && e == n_e - 1 {
                    s += "Total Energy Bin\n\n";
                } else {
                    s += "Energy Bin:  0.00E+00 - 1.00E+36 MeV\n\n";
                }
                for t in 0..n_t {
                    if self.n_time.is_some() {
                        if self.t_total() && t == n_t - 1 {
                            s += "\nTotal Time Bin\n\n";
                        } else {
                            s += "\nTime Bin: -1.00E+36 - 1.00E+00 shakes\n\n";
                        }
                    }
                    for outer in 0..self.dims[l.outer] {
                        s += &format!(
                            "  {} bin:  {outer}.0   -   {}.0\n\n",
                            names[l.outer],
                            outer + 1
                        );
                        for (is_result, values) in [(true, result), (false, rel_error)] {
                            if is_result {
                                s += &format!(
                                    "     Tally Results:  {} (across) by {} (down)\n",
                                    names[l.across], names[l.down]
                                );
                            } else {
                                s += "     Relative Errors\n";
                            }
                            s += "      0.5 1.5 (column centres, not read)\n";
                            for row in 0..self.dims[l.down] {
                                s += &format!("  {row}.0");
                                for col in 0..self.dims[l.across] {
                                    let v = values[self.index(e, t, col, row, outer)];
                                    s += &if is_result {
                                        format!(" {v:.5E}")
                                    } else {
                                        format!(" {v:.5}")
                                    };
                                }
                                s.push('\n');
                            }
                            s.push('\n');
                        }
                        s.push('\n');
                    }
                }
            }
            s
        }
    }

    fn splitmix64(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn synthetic(case: &Case, seed: u64) -> (Array5<f64>, Array5<f64>) {
        let mut state = seed;
        let result = Array5::from_shape_fn(case.shape(), |_| {
            format!("{:.5E}", (splitmix64(&mut state) - 0.5) * 2.0e5)
                .parse()
                .unwrap()
        });
        let rel_error = Array5::from_shape_fn(case.shape(), |_| {
            format!("{:.5}", splitmix64(&mut state)).parse().unwrap()
        });
        (result, rel_error)
    }

    fn case_strategy(max_dim: usize, max_energy: usize) -> impl Strategy<Value = Case> {
        (
            [1..=max_dim, 1..=max_dim, 1..=max_dim],
            any::<bool>(),
            0..3usize,
            1..=max_energy,
            prop::option::of(1..=3usize),
        )
            .prop_map(|(dims, cylindrical, l, n_energy, n_time)| Case {
                dims,
                cylindrical,
                layout: LAYOUTS[l],
                n_energy,
                n_time,
            })
    }

    fn read(text: &str, chunk: usize) -> Tally {
        let r = Reader::build(Source::bytes(text.as_bytes().to_vec()), None, chunk).unwrap();
        if let Some((id, why)) = r.failed().next() {
            panic!("tally {id}: {why}");
        }
        r.read(4).unwrap()
    }

    fn round_trip(case: Case, seed: u64, chunk: usize) {
        let (result, rel_error) = synthetic(&case, seed);
        let t = read(&case.render(&result, &rel_error), chunk);
        assert!(t.warnings.is_empty(), "{:?}", t.warnings);
        assert_eq!(t.out_format, case.layout.out_format());
        assert!(t.volume.is_none());
        assert_eq!(
            (t.energy.total, t.time.total),
            (case.e_total(), case.t_total())
        );
        assert_eq!(t.result, result);
        assert_eq!(t.rel_error, rel_error);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        #[test]
        fn round_trips(case in case_strategy(4, 3), seed in any::<u64>(), chunk in 1usize..2000) {
            round_trip(case, seed, chunk);
        }
    }

    proptest! {
        // Hundreds of blocks.
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        #[test]
        fn round_trips_many_blocks(case in case_strategy(12, 4), seed in any::<u64>()) {
            round_trip(case, seed, 1 << 20);
        }
    }

    fn nth_containing(lines: &[String], needle: &str, n: usize) -> usize {
        lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(needle))
            .nth(n)
            .map(|(i, _)| i)
            .unwrap_or_else(|| panic!("no {n}th line containing {needle:?}"))
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

        /// Damage to one block only affects that block.
        #[test]
        fn damage_stays_local(case in case_strategy(3, 2), seed in any::<u64>(),
                              pick in any::<prop::sample::Index>(), kind in 0u8..3) {
            let (result, rel_error) = synthetic(&case, seed);
            let text = case.render(&result, &rel_error);
            let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
            let (n_e, n_t, ..) = case.shape();
            let n_outer = case.dims[case.layout.outer];
            let n_blocks = n_e * n_t * n_outer;
            // A lone block without its layout line leaves no layout to read.
            prop_assume!(n_blocks > 1 || kind != 0);
            let b = pick.index(n_blocks);
            let (e, t, o) = (b / (n_t * n_outer), (b / n_outer) % n_t, b % n_outer);
            let header = nth_containing(&lines, "Tally Results:", b);
            let first_row = header + 2;
            let (n_rows, n_cols) = (case.dims[case.layout.down], case.dims[case.layout.across]);
            match kind {
                0 => {
                    for _ in 0..n_rows + 2 {
                        lines.remove(header);
                    }
                }
                1 => {
                    let short: Vec<&str> = lines[first_row].split_whitespace().take(n_cols).collect();
                    lines[first_row] = short.join(" ");
                }
                _ => {
                    let mut tokens: Vec<String> = lines[first_row].split_whitespace().map(str::to_string).collect();
                    tokens[1] = "notanumber".to_string();
                    lines[first_row] = tokens.join(" ");
                }
            }
            let t_read = read(&lines.join("\n"), 1 << 20);
            let prefix = format!("energy {e}, time {t}, outer bin {o}: ");
            prop_assert_eq!(t_read.warnings.len(), 1, "{:?}", t_read.warnings);
            prop_assert!(t_read.warnings[0].message.starts_with(&prefix), "{:?}", t_read.warnings);
            for ee in 0..n_e {
                for tt in 0..n_t {
                    for oo in 0..n_outer {
                        if (ee, tt, oo) == (e, t, o) {
                            continue;
                        }
                        for row in 0..n_rows {
                            for col in 0..n_cols {
                                let idx = case.index(ee, tt, col, row, oo);
                                prop_assert_eq!(t_read.result[idx], result[idx]);
                                prop_assert_eq!(t_read.rel_error[idx], rel_error[idx]);
                            }
                        }
                    }
                }
            }
        }
    }

    fn rect(dims: [usize; 3], n_energy: usize) -> Case {
        Case {
            dims,
            cylindrical: false,
            layout: LAYOUTS[0],
            n_energy,
            n_time: None,
        }
    }

    fn failure(text: &str) -> String {
        let r = Reader::from_bytes(text.as_bytes().to_vec(), None);
        let failed: Vec<(u32, &str)> = r.failed().collect();
        assert_eq!(failed.len(), 1, "{failed:?}");
        failed[0].1.to_string()
    }

    #[test]
    fn unreadable_layouts() {
        let head = rect([1, 1, 1], 1).header();
        let why = failure(&format!(
            "{head}Energy Bin:  0.00E+00 - 1.00E+36 MeV\n  Theta bin (revolutions):  0.0 - 1.0\n     \
             Tally Results:  R (across) by Z (down)\n"
        ));
        assert!(why.contains("unrecognized matrix layout"), "{why}");
        let why = failure(&format!(
            "{head}Energy Bin:  0.00E+00 - 1.00E+36 MeV\n  Z bin:  0.0 - 1.0\n"
        ));
        assert!(why.contains("no \"Tally Results:\" line"), "{why}");
    }

    #[test]
    fn a_missing_block_does_not_shift_later_ones() {
        // Energy bin 0 loses its second block; energy bin 1 and the Total
        // still land at their own indices.
        let case = rect([2, 2, 2], 2);
        let (result, rel_error) = synthetic(&case, 11);
        let text = case.render(&result, &rel_error);
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        let second = nth_containing(&lines, "Z bin:", 1);
        let next_energy = nth_containing(&lines, "Energy Bin:", 1);
        lines.drain(second..next_energy);
        let t = read(&lines.join("\n"), 1 << 20);
        assert_eq!(t.warnings.len(), 1, "{:?}", t.warnings);
        assert!(t.warnings[0]
            .message
            .contains("1 of 6 matrix blocks missing (first: energy 0, time 0, outer bin 1)"));
        for e in 0..3 {
            for o in 0..2 {
                for row in 0..2 {
                    for col in 0..2 {
                        let idx = case.index(e, 0, col, row, o);
                        if (e, o) == (0, 1) {
                            assert!(t.result[idx].is_nan());
                        } else {
                            assert_eq!(t.result[idx], result[idx]);
                        }
                    }
                }
            }
        }
    }
}
