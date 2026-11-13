//! MCTAL files written by MCNP 6.2 and 6.3 for `tests/fixtures/mctal/`:
//! `probe.mcnp` fires one 14 MeV neutron ray along +x through void, so every
//! value is known exactly; `kcode.mcnp` fixes the layout of what only a run
//! with materials writes (KCODE, perturbations, detectors).

mod common;

use aci_tallies::common::{Axis, Bins};
use aci_tallies::mctal::{self, Detector, Modifier, Number, Reader, Select, Tally, DIMS, E, F};
use ndarray::IxDyn;

fn reader(name: &str) -> Reader {
    let r = Reader::open(common::fixture(&format!("mctal/{name}"))).unwrap();
    assert!(r.warnings().is_empty(), "{:?}", r.warnings());
    assert_eq!(
        r.failed().count(),
        0,
        "{:?}",
        r.failed().collect::<Vec<_>>()
    );
    r
}

/// The values of a tally, flat, as the file lists them.
fn flat(t: &Tally) -> Vec<f64> {
    t.result.iter().copied().collect()
}

fn close(a: &[f64], b: &[f64]) -> bool {
    // MCNP prints 6 significant digits.
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 5e-6 * y.abs())
}

const CELL: f64 = 20.0 / 8000.0; // slab chord / slab volume

fn numbers(list: &[i64]) -> Vec<Option<Number>> {
    list.iter()
        .map(|&n| {
            (n != 0).then_some(Number {
                number: n,
                facet: None,
            })
        })
        .collect()
}

#[test]
fn every_value_of_the_probe() {
    for name in ["probe.m", "probe_63.m"] {
        let r = reader(name);
        let h = r.header();
        assert_eq!((h.dump, h.histories, h.perturbations), (2, 100, 0));
        assert!(h.title.starts_with("MCTAL probe"));
        assert_eq!(h.tallies, [1, 2, 4, 11, 14, 24, 34, 44, 54, 64, 74, 84, 94]);
        assert!(r.kcode().is_none());
        let read = |id| {
            let t = r.read(id).unwrap();
            assert!(t.warnings.is_empty(), "{name} tally {id}: {:?}", t.warnings);
            assert_eq!(t.particles, ['n']);
            assert_eq!((t.detector, t.modifier), (Detector::None, Modifier::None));
            t
        };
        let expect = [
            // F1 on surfaces 11, 12 with cosine bins (-1, 0], (0, 1], Total.
            (1, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            // F2: 1 / (cos * area) with SD 400.
            (2, vec![1.0 / 400.0; 2]),
            (4, vec![CELL; 4]),
            // FS -40 on surfaces 11, 12: the ray is on the + side of y = 0.
            (11, vec![0.0, 1.0, 0.0, 1.0]),
            // (1 2) (3 4) 1 T.
            (14, vec![CELL; 4]),
            // 8 and 9 over all four elements, then each element's 8.
            (24, vec![12.0 / 300.0, 8.0 / 1700.0, 0.01, 0.01, 0.01, 0.01]),
            (34, vec![0.01, 0.01, 0.01, 2.0 / 1700.0]),
            // The T bin of repeated structures: tracks in 8 and 9, over
            // one element's volume.
            (
                44,
                vec![
                    12.0 / 300.0,
                    8.0 / 1700.0,
                    0.01,
                    0.01,
                    12.0 / 300.0,
                    20.0 / 2000.0,
                ],
            ),
            (
                54,
                vec![
                    CELL,
                    2.0 * CELL,
                    CELL * 1e-110,
                    CELL,
                    2.0 * CELL,
                    CELL * 1e-110,
                ],
            ),
            (74, vec![0.0, 0.0, CELL, 0.0, 0.0, CELL]),
            (84, vec![0.0, 0.0, CELL, 0.0, 0.0, CELL]),
            // FT INC: no collisions in void, all in user bin 0.
            (94, vec![CELL, 0.0, CELL, CELL, 0.0, CELL]),
        ];
        for (id, want) in expect {
            assert!(
                close(&flat(&read(id)), &want),
                "{name} tally {id}: {:?}",
                flat(&read(id))
            );
        }
        // Energy (1, 10, 20, Total) by time (1e2, 1e5, Total): a 14 MeV
        // neutron (51 cm/shake) is in energy bin 2 and crosses every slab
        // within 3 shakes.
        let t = read(64);
        assert_eq!(t.result.shape(), [4, 1, 1, 1, 1, 1, 4, 3]);
        for f in 0..4 {
            for e in 0..4 {
                for tm in 0..3 {
                    let want = if [2, 3].contains(&e) && [0, 2].contains(&tm) {
                        CELL
                    } else {
                        0.0
                    };
                    assert_eq!(
                        t.result[IxDyn(&[f, 0, 0, 0, 0, 0, e, tm])],
                        want,
                        "{f} {e} {tm}"
                    );
                }
            }
        }
        assert_eq!(
            t.axes[E],
            Axis {
                bins: Bins::Upper(vec![1.0, 10.0, 20.0]),
                total: true,
                cumulative: false
            }
        );
        assert_eq!(t.axes[7].upper(), Some(&[100.0, 1e5][..]));
        assert!(t.axes[7].total);

        // What the f lines say.
        assert_eq!(read(4).numbers, numbers(&[1, 2, 3, 4]));
        assert_eq!(read(14).numbers, numbers(&[0, 0, 1, 0]));
        assert_eq!(read(24).numbers, numbers(&[8, 9, 8, 8, 8, 8]));
        assert_eq!(read(44).numbers, numbers(&[8, 9, 8, 8, 8, 0]));
        assert_eq!(read(1).numbers, numbers(&[11, 12]));
        // The other bins.
        assert_eq!(read(1).axes[5].upper(), Some(&[0.0, 1.0][..]));
        assert!(read(1).axes[5].total);
        assert_eq!(read(11).axes[3].bins, Bins::Count(2));
        assert_eq!(read(54).axes[4].bins, Bins::Count(3));
        assert!(!read(74).axes[E].total);
        assert!(read(84).axes[E].cumulative);
        assert_eq!(
            read(94).axes[2],
            Axis {
                bins: Bins::Upper(vec![0.0, 1.0]),
                total: true,
                cumulative: false
            }
        );
        let c = read(4).comment;
        assert_eq!(c.as_deref(), Some("plain cells"));
        // The fluctuation chart: tfc bin of tally 64 is its last.
        let tfc = read(64).tfc.unwrap();
        assert_eq!(tfc.bin, [0, 0, 0, 0, 0, 0, 3, 2]);
        assert_eq!(tfc.histories, [100]);
        assert_eq!(tfc.mean, [CELL]);
        assert!(tfc.fom.is_some());
    }
}

#[test]
fn parts_and_operations() {
    let r = reader("probe.m");
    let whole = r.read(24).unwrap();
    for (a, b) in [(0, 6), (0, 1), (2, 5), (5, 6)] {
        let part = r.read_f(24, a..b).unwrap();
        assert_eq!(part.result.shape()[F], b - a);
        assert_eq!(flat(&part), flat(&whole)[a..b]);
        assert_eq!(part.numbers, whole.numbers[a..b]);
    }
    assert!(r.read_f(24, 2..2).is_err());
    assert!(r.read_f(24, 0..7).is_err());

    let t = r.read(64).unwrap();
    // Summing energy gives MCNP's Total bin.
    let e = mctal::ops::integrate(&t, E);
    assert_eq!(e.result.shape(), [4, 1, 1, 1, 1, 1, 1, 3]);
    assert_eq!(e.result[IxDyn(&[0, 0, 0, 0, 0, 0, 0, 0])], CELL);
    assert_eq!(e.result[IxDyn(&[0, 0, 0, 0, 0, 0, 0, 1])], 0.0);
    assert_eq!(e.axes[E].upper(), Some(&[20.0][..]));
    let total = mctal::ops::isel(
        &t,
        &std::array::from_fn(|k| if k == E { Select::Total } else { Select::All }),
    )
    .unwrap();
    assert_eq!(flat(&total), flat(&e));
    // Cumulative bins: the last is the sum.
    let c = mctal::ops::integrate(&r.read(84).unwrap(), E);
    assert_eq!(flat(&c), [CELL, CELL]);
    // Without a Total: the bins added up.
    let n = mctal::ops::integrate(&r.read(74).unwrap(), E);
    assert_eq!(flat(&n), [CELL, CELL]);
    let s = mctal::ops::add(&r.read(4).unwrap(), &r.read(4).unwrap()).unwrap();
    assert_eq!(flat(&s), [2.0 * CELL; 4]);
    assert!(mctal::ops::add(&r.read(4).unwrap(), &r.read(14).unwrap()).is_ok());
    assert!(mctal::ops::add(&r.read(4).unwrap(), &r.read(54).unwrap()).is_err());
}

#[test]
fn kcode_perturbations_and_detectors() {
    for name in ["kcode.m", "kcode_63.m"] {
        let r = reader(name);
        assert_eq!(r.header().perturbations, 2);
        assert_eq!(r.header().tallies, [4, 5, 6, 14, 15]);
        let k = r.kcode().unwrap().unwrap();
        assert_eq!((k.cycles, k.inactive), (8, 3));
        assert_eq!(k.values.dim(), (8, 19));
        assert_eq!(
            k.values[[0, 17]],
            500.0,
            "source histories of the first cycle"
        );
        assert!(
            k.values[[7, 11]] > 0.5,
            "an average k after the active cycles"
        );

        let t = r.read(4).unwrap();
        assert_eq!(t.result.shape(), [1, 1, 1, 1, 1, 1, 4, 1]);
        assert!(t.warnings.is_empty(), "{:?}", t.warnings);
        assert_eq!(t.perturbation, None);
        for n in 1..=2 {
            let p = r.read_perturbation(4, n).unwrap();
            assert_eq!(p.perturbation, Some(n as u32));
            assert_eq!(p.result.shape(), t.result.shape());
            assert!(p.tfc.is_some());
        }
        assert!(r.read_perturbation(4, 3).is_err());
        // Less dense, lower flux; denser, higher (each its own PERT card).
        let total = |t: &Tally| t.result[IxDyn(&[0, 0, 0, 0, 0, 0, 3, 0])];
        assert!(total(&r.read_perturbation(4, 1).unwrap()) != total(&t));

        let det = r.read(5).unwrap();
        assert_eq!(det.detector, Detector::Point);
        assert_eq!(det.result.shape(), [2, 2, 1, 1, 1, 1, 1, 1]);
        assert!(det.numbers.is_empty());
        assert_eq!(r.tally(15).unwrap().detector, Detector::Ring);
        assert_eq!(r.tally(14).unwrap().modifier, Modifier::Star);
        let heating = r.tally(6).unwrap();
        assert_eq!(heating.modifier, Modifier::Plus);
        // +F6 flags every particle of the problem: here, neutrons.
        assert_eq!(heating.particles, ['n']);
    }
}

#[test]
fn detects_formats() {
    let head = |p: &str| std::fs::read(common::fixture(p)).unwrap()[..2000].to_vec();
    use aci_tallies::{detect, Format};
    assert_eq!(detect(&head("mctal/probe.m")), Some(Format::Mctal));
    assert_eq!(detect(&head("mctal/kcode_63.m")), Some(Format::Mctal));
    assert_eq!(detect(&head("pencil_beam.msht")), Some(Format::Meshtal));
    assert_eq!(detect(&head("pencil_beam.mcnp")), None);
    assert_eq!(DIMS.len(), 8);
}

/// Real MCTAL files from other tools' tests, when present on this machine:
/// every tally reads, with no NaN and no warning.
#[test]
fn other_real_files() {
    let dir = std::path::Path::new(env!("HOME")).join("Coding/F4Enix/tests/resources/mctal");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: {} not found", dir.display());
        return;
    };
    let mut read = 0;
    for entry in entries {
        let path = entry.unwrap().path();
        let Ok(head) = std::fs::read(&path) else {
            continue;
        };
        if !mctal::is_mctal(&head[..head.len().min(4096)]) {
            continue;
        }
        let r = Reader::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            r.failed().count(),
            0,
            "{}: {:?}",
            path.display(),
            r.failed().collect::<Vec<_>>()
        );
        // Some are excerpts, with fewer tallies than their ntal line says.
        for w in r.warnings() {
            eprintln!("{}: {w}", path.display());
        }
        for t in r.read_all().unwrap() {
            assert!(
                t.warnings.is_empty(),
                "{} tally {}: {:?}",
                path.display(),
                t.id,
                t.warnings
            );
            assert!(
                t.result.iter().all(|v| v.is_finite()),
                "{} tally {}",
                path.display(),
                t.id
            );
            read += 1;
        }
    }
    assert!(read > 0);
}

/// With the input, each tally's bins come from its F card: their labels
/// are the ones MCNP's output file prints (`T` for its "union total").
#[test]
fn bins_from_the_input() {
    use aci_tallies::common::Input;
    // (fixture, [(tally, MCNP's labels)])
    type Case<'a> = (&'a str, &'a [(u32, &'a [&'a str])]);
    let cases: [Case; 2] = [
        (
            "probe",
            &[
                (1, &["11", "12"]),
                (4, &["1", "2", "3", "4"]),
                (14, &["(1 2)", "(3 4)", "1", "T"]),
                (
                    24,
                    &[
                        "8",
                        "9",
                        "(8<7[0 0 0]<3)",
                        "(8<7[1 0 0]<3)",
                        "(8<7[2 0 0]<3)",
                        "(8<7[3 0 0]<3)",
                    ],
                ),
                (
                    34,
                    &[
                        "(8<7[0 0 0]<3)",
                        "(8<7[1 0 0]<3)",
                        "(8<7[2 0 0]<3)",
                        "(9<7[2 0 0]<3)",
                    ],
                ),
                (
                    44,
                    &[
                        "(8<3)",
                        "(9<3)",
                        "(8<7[1 0 0]<3)",
                        "(8<7[3 0 0]<3)",
                        "(8<3)",
                        "T",
                    ],
                ),
                (94, &["1", "2"]),
            ],
        ),
        (
            "levels",
            &[
                (4, &["(8 9<3)"]),
                (14, &["(8<7[0 0 0] 7[1 0 0]<3)"]),
                (
                    24,
                    &[
                        "(8<7[0 0 0]<3)",
                        "(9<7[0 0 0]<3)",
                        "(8<7[1 0 0]<3)",
                        "(9<7[1 0 0]<3)",
                        "(8<7[0 1 0]<3)",
                        "(9<7[0 1 0]<3)",
                        "(8<7[1 1 0]<3)",
                        "(9<7[1 1 0]<3)",
                    ],
                ),
                (
                    34,
                    &[
                        "(8<7[0 0 0]<3)",
                        "(8<7[1 0 0]<3)",
                        "(8<7[0 1 0]<3)",
                        "(8<7[1 1 0]<3)",
                        "T",
                    ],
                ),
                (44, &["(8<7[0 0 0]<3)", "(8<7[1 1 0]<3)", "(9<3)"]),
                (54, &["(8<7<3)"]),
                (
                    64,
                    &[
                        "(8<7[0 0 0]<3)",
                        "(8<7[1 0 0]<3)",
                        "(8<7[0 1 0]<3)",
                        "(8<7[1 1 0]<3)",
                    ],
                ),
                (
                    84,
                    &["(8<7[0 0 0]<3)", "(9<7[0 0 0]<3)", "(8<7[1 0 0]<3)", "T"],
                ),
                (94, &["1", "(2 4)", "T"]),
            ],
        ),
    ];
    for (name, tallies) in cases {
        let input = Input::read(common::fixture(&format!("mctal/{name}.mcnp"))).unwrap();
        let r =
            Reader::open_with_input(common::fixture(&format!("mctal/{name}.m")), &input).unwrap();
        for &(id, want) in tallies {
            let t = r.read(id).unwrap();
            assert!(t.warnings.is_empty(), "{name} {id}: {:?}", t.warnings);
            let got: Vec<String> = t.bins.iter().map(ToString::to_string).collect();
            assert_eq!(got, want, "{name} tally {id}");
        }
        // Without the input: numbers only.
        assert!(Reader::open(common::fixture(&format!("mctal/{name}.m")))
            .unwrap()
            .read(24)
            .unwrap()
            .bins
            .is_empty());
    }
    // The structure behind a label.
    let input = Input::read(common::fixture("mctal/probe.mcnp")).unwrap();
    let r = Reader::open_with_input(common::fixture("mctal/probe.m"), &input).unwrap();
    let t = r.read_f(24, 3..4).unwrap();
    let aci_tallies::mctal::TallyBin::Levels(levels) = &t.bins[0] else {
        panic!()
    };
    let numbers: Vec<Vec<(i64, Option<[i64; 3]>)>> = levels
        .iter()
        .map(|l| l.iter().map(|r| (r.number, r.element)).collect())
        .collect();
    assert_eq!(
        numbers,
        [vec![(8, None)], vec![(7, Some([1, 0, 0]))], vec![(3, None)]]
    );

    // An input that is not the file's: a warning, and no bins.
    let other = Input::read(common::fixture("mctal/levels.mcnp")).unwrap();
    let r = Reader::open_with_input(common::fixture("mctal/probe.m"), &other).unwrap();
    let t = r.read(24).unwrap();
    assert!(t.bins.is_empty());
    assert!(
        t.warnings[0]
            .message
            .contains("makes 8 bins, the file has 6"),
        "{:?}",
        t.warnings
    );
    assert!(r.read(1).unwrap().warnings[0]
        .message
        .contains("no F1 card"));
}
