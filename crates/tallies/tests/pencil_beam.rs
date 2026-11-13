//! Exact physics checks against `tests/fixtures/pencil_beam.msht` (MCNP 6.3).
//!
//! Every history in that run is the same 14 MeV neutron flying along one
//! straight ray through void, so each voxel's flux is exactly
//! `chord length / voxel volume` with zero variance. These tests recompute
//! the chords analytically from the *parsed* geometry and compare them with
//! MCNP's numbers, which checks, in one go: the header parser, every data
//! layout (col, cf, ij, ik, jk, with energy/time/Total blocks), and the
//! geometry conventions in `aci-tallies-geom` (voxel numbering, volumes,
//! point location, and the cylinder frame built from AXS/VEC).

use aci_tallies::meshtal::{OutFormat, Tally};
use ndarray::s;
use std::collections::HashMap;

mod common;

use common::{by_id, chords, total_slice, PRINT_TOL};

fn tallies() -> HashMap<u32, Tally> {
    by_id(
        aci_tallies::meshtal::read(common::fixture("pencil_beam.msht"))
            .expect("fixture should be readable"),
    )
}

#[test]
fn every_section_parses_cleanly() {
    let tallies = tallies();
    assert_eq!(tallies.len(), 11);
    for t in tallies.values() {
        assert!(t.warnings.is_empty(), "tally {}: {:?}", t.id, t.warnings);
        assert!(t.result.iter().all(|v| v.is_finite()), "tally {}", t.id);
        assert!(
            t.rel_error.iter().all(|&v| v == 0.0),
            "tally {}: zero-variance run",
            t.id
        );
    }
    let formats: Vec<(u32, OutFormat)> = [
        (1004, OutFormat::Cf),
        (1014, OutFormat::Cf),
        (1024, OutFormat::Ij),
        (1034, OutFormat::Ik),
        (1044, OutFormat::Jk),
        (1054, OutFormat::Col),
        (1064, OutFormat::Col),
        (1074, OutFormat::Ij),
        (1084, OutFormat::Ik),
        (1094, OutFormat::Jk),
        (1104, OutFormat::Ik),
    ]
    .into();
    for (id, format) in formats {
        assert_eq!(tallies[&id].out_format, format, "tally {id}");
    }
}

#[test]
fn flux_is_chord_over_volume_in_every_voxel() {
    for t in tallies().values() {
        let chord = chords(&t.geometry);
        let volume = t.geometry.voxel_volumes();
        let result = total_slice(t);
        let scale = result.iter().copied().fold(0.0, f64::max);
        assert!(scale > 0.0, "tally {}: the beam must cross the mesh", t.id);
        for v in 0..result.len() {
            let expected = chord[v] / volume[v];
            assert!(
                (result[v] - expected).abs() <= PRINT_TOL * scale,
                "tally {} voxel {v} {:?}: MCNP {} vs chord/volume {expected}",
                t.id,
                t.geometry.ijk(v),
                result[v]
            );
        }
    }
}

#[test]
fn cf_volumes_match_geometry() {
    let tallies = tallies();
    for id in [1004, 1014] {
        let t = &tallies[&id];
        let mcnp = t.volume.as_ref().expect("out=cf has volumes");
        for (n, (&a, b)) in mcnp.iter().zip(t.geometry.voxel_volumes()).enumerate() {
            assert!(
                (a - b).abs() <= PRINT_TOL * b,
                "tally {id} voxel {n}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn all_layouts_of_one_mesh_agree() {
    let tallies = tallies();
    for group in [
        [1014, 1024, 1034, 1044, 1054].as_slice(),
        &[1064, 1074, 1084, 1094],
    ] {
        let reference = &tallies[&group[0]];
        for id in &group[1..] {
            let t = &tallies[id];
            assert_eq!(t.geometry, reference.geometry, "tally {id}");
            assert_eq!(t.result.shape(), reference.result.shape(), "tally {id}");
            for (a, b) in t.result.iter().zip(reference.result.iter()) {
                assert!(
                    (a - b).abs() <= PRINT_TOL * b.abs(),
                    "tally {id} vs {}: {a} vs {b}",
                    group[0]
                );
            }
        }
    }
}

#[test]
fn energy_and_time_bins_add_up_to_their_totals() {
    let tallies = tallies();
    for id in [1064, 1074, 1084, 1094, 1104] {
        let t = &tallies[&id];
        assert!(t.time.total, "tally {id}");
        let ne = t.result.shape()[0];
        let nt = t.result.shape()[1];
        assert_eq!(nt, t.time.n_bins() + 1, "tally {id}");
        let scale = t.result.iter().copied().fold(0.0, f64::max);
        for e in 0..ne {
            let total = t.result.slice(s![e, nt - 1, .., .., ..]);
            let sum = t
                .result
                .slice(s![e, ..nt - 1, .., .., ..])
                .sum_axis(ndarray::Axis(0));
            for (a, b) in sum.iter().zip(total.iter()) {
                assert!(
                    (a - b).abs() <= 3.0 * PRINT_TOL * scale,
                    "tally {id} energy {e}: {a} vs {b}"
                );
            }
        }
        if t.energy.total {
            assert_eq!(ne, t.energy.n_bins() + 1, "tally {id}");
            // The beam is 14 MeV: only the [13.5, 20] MeV group scores, and
            // it must equal the Total group.
            let hit = t
                .energy
                .bounds()
                .unwrap()
                .windows(2)
                .position(|w| w[0] < 14.0 && 14.0 <= w[1])
                .unwrap();
            for e in 0..t.energy.n_bins() {
                let group = t.result.slice(s![e, .., .., .., ..]);
                if e == hit {
                    assert_eq!(
                        group,
                        t.result.slice(s![ne - 1, .., .., .., ..]),
                        "tally {id}"
                    );
                } else {
                    assert!(group.iter().all(|&v| v == 0.0), "tally {id} energy {e}");
                }
            }
        }
    }
}
