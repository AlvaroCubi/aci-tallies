use aci_tallies::meshtal::{Geometry, OutFormat, Tally};
use approx::{assert_abs_diff_eq, assert_relative_eq};
use std::collections::HashMap;
use std::path::PathBuf;

fn example_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/meshtal")
}

fn parse_example() -> HashMap<u32, Tally> {
    let reader = aci_tallies::meshtal::Reader::open(example_path())
        .expect("examples/meshtal should be readable");
    assert_eq!(reader.failed().count(), 0);
    reader
        .read_all()
        .unwrap()
        .into_iter()
        .map(|t| (t.id, t))
        .collect()
}

fn expect_tally(tallies: &HashMap<u32, Tally>, id: u32) -> &Tally {
    tallies
        .get(&id)
        .unwrap_or_else(|| panic!("tally {id} missing"))
}

#[test]
fn parses_all_nine_example_sections() {
    let sections = parse_example();
    let mut ids: Vec<u32> = sections.keys().copied().collect();
    ids.sort();
    assert_eq!(ids, vec![14, 24, 34, 44, 54, 64, 74, 84, 94]);
}

#[test]
fn tally_14_plain_rectangular_col_format() {
    let sections = parse_example();
    let t = expect_tally(&sections, 14);
    assert_eq!(t.particle, "neutron");
    assert_eq!(t.out_format, OutFormat::Col);
    assert!(t.comment.is_none());
    assert!(
        t.warnings.is_empty(),
        "unexpected warnings: {:?}",
        t.warnings
    );

    match &t.geometry {
        Geometry::Rectangular {
            x_bounds,
            y_bounds,
            z_bounds,
            transform: None,
        } => {
            assert_eq!(x_bounds.len() - 1, 5);
            assert_eq!(y_bounds.len() - 1, 5);
            assert_eq!(z_bounds.len() - 1, 5);
        }
        other => panic!("expected rectangular geometry, got {other:?}"),
    }

    assert!(!t.energy.total);
    assert!(!t.time.total);
    // First data row: -8.000 -8.000 -8.000 4.69512E-04 1.07701E-02
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 4.69512E-04, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[0, 0, 0, 0, 0]],
        1.07701E-02,
        max_relative = 1e-6
    );
    // Center voxel row: 0.000 0.000 0.000 3.82462E-02 1.35894E-04 (i=j=k=2, the middle bin)
    assert_relative_eq!(t.result[[0, 0, 2, 2, 2]], 3.82462E-02, max_relative = 1e-6);
}

#[test]
fn tally_24_non_uniform_spacing_and_fc_comment() {
    let sections = parse_example();
    let t = expect_tally(&sections, 24);
    assert_eq!(t.comment.as_deref(), Some("Simple FMESH"));
    match &t.geometry {
        Geometry::Rectangular {
            x_bounds,
            y_bounds,
            z_bounds,
            transform: None,
        } => {
            assert_eq!(x_bounds.len() - 1, 5);
            assert_eq!(y_bounds.len() - 1, 6);
            assert_eq!(z_bounds.len() - 1, 7);
        }
        other => panic!("expected rectangular geometry, got {other:?}"),
    }
    // First data row: -8.000 -8.250 -8.429 4.41649E+13 1.24624E-02
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 4.41649E+13, max_relative = 1e-6);
}

#[test]
fn tally_34_energy_discretized() {
    let sections = parse_example();
    let t = expect_tally(&sections, 34);
    assert_eq!(t.out_format, OutFormat::Col);
    assert_eq!(t.energy.n_bins(), 10);
    assert!(t.energy.total);
    assert!(!t.time.total);
    assert_eq!(t.result.dim(), (11, 1, 5, 5, 5));
    // First row: Energy=1.000E-06  X=-8.000 Y=-8.000 Z=-8.000 Result=7.41491E+11 RelErr=6.96947E-02
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 7.41491E+11, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[0, 0, 0, 0, 0]],
        6.96947E-02,
        max_relative = 1e-6
    );
    // MCNP's automatic Total energy block, first row:
    //   Total -8.000 -8.000 -8.000 8.67991E+11 6.73371E-02
    let total_idx = t.energy.total_index().expect("tally 34 has a Total block");
    assert_eq!(total_idx, 10);
    assert_relative_eq!(t.result[[10, 0, 0, 0, 0]], 8.67991E+11, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[10, 0, 0, 0, 0]],
        6.73371E-02,
        max_relative = 1e-6
    );
}

#[test]
fn tally_44_ij_matches_tally_14_col() {
    // Tally 44 (out=ij) and tally 14 (out=col) share the same 5x5x5 geometry
    // and the same underlying run, so a correct ij reconstruction must
    // produce numerically identical arrays to the plain columnar parse.
    let sections = parse_example();
    let col = expect_tally(&sections, 14);
    let ij = expect_tally(&sections, 44);
    assert_eq!(ij.out_format, OutFormat::Ij);
    assert!(
        ij.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ij.warnings
    );
    assert_eq!(col.result.shape(), ij.result.shape());
    for (a, b) in col.result.iter().zip(ij.result.iter()) {
        assert_relative_eq!(*a, *b, max_relative = 1e-6);
    }
    // The ij matrix format prints Rel Error to a fixed 5 decimal places
    // (e.g. "0.00014") rather than the columnar format's full scientific
    // precision ("1.35894E-04"), so an absolute (quantization) tolerance is
    // the right check here, not a relative one.
    for (a, b) in col.rel_error.iter().zip(ij.rel_error.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn tally_54_ik_matches_tally_14_col() {
    let sections = parse_example();
    let col = expect_tally(&sections, 14);
    let ik = expect_tally(&sections, 54);
    assert_eq!(ik.out_format, OutFormat::Ik);
    assert!(
        ik.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ik.warnings
    );
    assert_eq!(col.result.shape(), ik.result.shape());
    for (a, b) in col.result.iter().zip(ik.result.iter()) {
        assert_relative_eq!(*a, *b, max_relative = 1e-6);
    }
    for (a, b) in col.rel_error.iter().zip(ik.rel_error.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn tally_64_cf_format_includes_volume() {
    let sections = parse_example();
    let t = expect_tally(&sections, 64);
    assert_eq!(t.out_format, OutFormat::Cf);
    let volume = t.volume.as_ref().expect("cf format should populate volume");
    // First row: ... 4.69512E-04 1.07701E-02 6.40000E+01 3.00488E-02
    assert_relative_eq!(volume[[0, 0, 0]], 64.0, max_relative = 1e-6);
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 4.69512E-04, max_relative = 1e-6);
    // Volume should be uniform (64.0) across every voxel in this example.
    for v in volume.iter() {
        assert_relative_eq!(*v, 64.0, max_relative = 1e-6);
    }
}

#[test]
fn tally_74_cylindrical_geometry() {
    let sections = parse_example();
    let t = expect_tally(&sections, 74);
    match &t.geometry {
        Geometry::Cylindrical {
            r_bounds,
            z_bounds,
            theta_bounds,
            origin,
            axis,
            vec,
        } => {
            assert_eq!(r_bounds.len() - 1, 10);
            assert_eq!(z_bounds.len() - 1, 10);
            assert_eq!(theta_bounds.len() - 1, 10);
            assert_relative_eq!(origin[2], -10.0, max_relative = 1e-6);
            assert_relative_eq!(axis[2], 1.0, max_relative = 1e-6);
            assert_relative_eq!(vec[0], 1.0, max_relative = 1e-6);
        }
        other => panic!("expected cylindrical geometry, got {other:?}"),
    }
    // First row: R=0.500 Z=1.000 Th=0.050 Result=1.09801E-03 RelErr=5.13735E-02
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 1.09801E-03, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[0, 0, 0, 0, 0]],
        5.13735E-02,
        max_relative = 1e-6
    );
}

#[test]
fn tally_84_time_discretized_no_energy_total() {
    let sections = parse_example();
    let t = expect_tally(&sections, 84);
    assert_eq!(t.out_format, OutFormat::Col);
    assert!(
        t.warnings.is_empty(),
        "unexpected warnings: {:?}",
        t.warnings
    );
    assert_eq!(t.energy.n_bins(), 1);
    assert_eq!(t.time.n_bins(), 3);
    assert!(!t.energy.total);
    assert!(t.time.total);
    assert_eq!(t.result.dim(), (1, 4, 5, 5, 5));

    // First data row: Time=1.000E+08 X=-8.000 Y=-8.000 Z=-8.000
    // Result=4.69512E-04 RelErr=1.07701E-02
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 4.69512E-04, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[0, 0, 0, 0, 0]],
        1.07701E-02,
        max_relative = 1e-6
    );

    // MCNP's automatic Total time block, first row (identical to the first
    // real time bin here, since the later time bins are all zero for this
    // static source): Total -8.000 -8.000 -8.000 4.69512E-04 1.07701E-02
    let total_idx = t.time.total_index().expect("tally 84 has a Total block");
    assert_eq!(total_idx, 3);
    assert_relative_eq!(t.result[[0, 3, 0, 0, 0]], 4.69512E-04, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[0, 3, 0, 0, 0]],
        1.07701E-02,
        max_relative = 1e-6
    );
}

#[test]
fn tally_94_energy_and_time_cross_product_total() {
    let sections = parse_example();
    let t = expect_tally(&sections, 94);
    assert_eq!(t.out_format, OutFormat::Col);
    assert!(
        t.warnings.is_empty(),
        "unexpected warnings: {:?}",
        t.warnings
    );
    assert_eq!(t.energy.n_bins(), 10);
    assert_eq!(t.time.n_bins(), 3);
    assert!(t.energy.total);
    assert!(t.time.total);
    // 11 * 4 * 5 * 5 * 5 = 5500, matching the file's exact row count for
    // this section (verified against examples/meshtal directly).
    assert_eq!(t.result.dim(), (11, 4, 5, 5, 5));

    // First data row: Energy=1.000E-06 Time=1.000E+08 X=Y=Z=-8.000
    // Result=7.41491E-06 RelErr=6.96947E-02
    assert_relative_eq!(t.result[[0, 0, 0, 0, 0]], 7.41491E-06, max_relative = 1e-6);
    assert_relative_eq!(
        t.rel_error[[0, 0, 0, 0, 0]],
        6.96947E-02,
        max_relative = 1e-6
    );

    // Grand Energy=Total, Time=Total block: first row (X=Y=Z=-8.000):
    //   Total Total -8.000 -8.000 -8.000 8.67991E-06 6.73371E-02
    // and last row of the file (X=Y=Z=8.000):
    //   Total Total  8.000  8.000  8.000 7.88808E-06 7.09541E-02
    let e_total = t
        .energy
        .total_index()
        .expect("tally 94 has an energy Total");
    let t_total = t.time.total_index().expect("tally 94 has a time Total");
    assert_eq!((e_total, t_total), (10, 3));
    assert_relative_eq!(
        t.result[[e_total, t_total, 0, 0, 0]],
        8.67991E-06,
        max_relative = 1e-6
    );
    assert_relative_eq!(
        t.rel_error[[e_total, t_total, 0, 0, 0]],
        6.73371E-02,
        max_relative = 1e-6
    );
    assert_relative_eq!(
        t.result[[e_total, t_total, 4, 4, 4]],
        7.88808E-06,
        max_relative = 1e-6
    );
    assert_relative_eq!(
        t.rel_error[[e_total, t_total, 4, 4, 4]],
        7.09541E-02,
        max_relative = 1e-6
    );
}
