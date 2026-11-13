//! Exact physics checks against `tests/fixtures/pencil_beam_input.msht`
//! (MCNP 6.3; 6.2 writes the same numbers), whose meshes the meshtal header
//! can't describe: `TR=` placements (reflections, one-vector and rounded
//! `TRn` cards included), unround origins and axes, bins finer than the
//! printed 0.01 cm, unround energy/time bins. With the geometry read from
//! `pencil_beam_input.mcnp`, every voxel must score exactly chord / volume,
//! as in `pencil_beam.rs`; with the header's alone, none of the tallies
//! does. Tally 2104, a mirrored cylinder, is the one the input can't fix.

use aci_tallies::common::Input;
use aci_tallies::meshtal::{Geometry, Reader, Tally};
use std::collections::HashMap;

mod common;

use common::{by_id, fixture, worst_voxel, PRINT_TOL};

const IDS: [u32; 10] = [2004, 2014, 2024, 2034, 2044, 2054, 2064, 2074, 2084, 2094];
/// A cylinder under a reflecting `TR=`, which `Geometry` can't describe.
const MIRRORED: u32 = 2104;
/// `*FMESH2064` scores energy times flux, and every history is 14 MeV.
fn scale(id: u32) -> f64 {
    if id == 2064 {
        14.0
    } else {
        1.0
    }
}

fn meshtal() -> String {
    std::fs::read_to_string(fixture("pencil_beam_input.msht")).unwrap()
}

fn input() -> String {
    std::fs::read_to_string(fixture("pencil_beam_input.mcnp")).unwrap()
}

fn with_input(input: &str) -> HashMap<u32, Tally> {
    let input = Input::parse(input);
    by_id(
        Reader::from_bytes(meshtal().into_bytes(), Some(&input))
            .read_all()
            .unwrap(),
    )
}

fn largest(t: &Tally) -> f64 {
    t.result.iter().copied().fold(0.0, f64::max)
}

#[test]
fn every_tally_takes_its_geometry_from_the_input() {
    let tallies = with_input(&input());
    assert_eq!(tallies.len(), IDS.len() + 1);
    for id in IDS {
        let t = &tallies[&id];
        assert!(t.warnings.is_empty(), "tally {id}: {:?}", t.warnings);
        assert!(t.geometry_from_input, "tally {id}");
        assert_eq!(t.energy_weighted, Some(id == 2064), "tally {id}");
        assert_eq!(t.geometry.validate(), Ok(()), "tally {id}");
    }
    // The same through the file-based entry point.
    let from_files = by_id(
        Reader::open_with_input(
            fixture("pencil_beam_input.msht"),
            &Input::read(fixture("pencil_beam_input.mcnp")).unwrap(),
        )
        .unwrap()
        .read_all()
        .unwrap(),
    );
    assert_eq!(from_files[&2004].geometry, tallies[&2004].geometry);

    // The mirrored cylinder keeps its header geometry, and says why.
    let t = &tallies[&MIRRORED];
    assert!(!t.geometry_from_input);
    let message = &t.warnings[0].message;
    assert!(
        message.contains("reverses the theta direction")
            && message.contains("geometry is from the meshtal header"),
        "{message}"
    );
}

#[test]
fn flux_is_chord_over_volume_with_the_input_geometry() {
    let tallies = with_input(&input());
    for id in IDS {
        let t = &tallies[&id];
        let (voxel, mcnp, expected) = worst_voxel(t, scale(id));
        assert!(
            (mcnp - expected).abs() <= PRINT_TOL * largest(t),
            "tally {id} voxel {voxel} {:?}: MCNP {mcnp} vs chord/volume {expected}",
            t.geometry.ijk(voxel)
        );
    }
}

#[test]
fn the_header_geometry_alone_is_wrong() {
    let tallies = by_id(
        Reader::from_bytes(meshtal().into_bytes(), None)
            .read_all()
            .unwrap(),
    );
    for id in IDS {
        let t = &tallies[&id];
        assert!(!t.geometry_from_input);
        assert_eq!(t.energy_weighted, None);
        let (_, mcnp, expected) = worst_voxel(t, scale(id));
        assert!(
            (mcnp - expected).abs() > 100.0 * PRINT_TOL * largest(t),
            "tally {id}: the header geometry should not reproduce MCNP"
        );
    }
    // The 0.001 cm bin collapses to zero width at two printed decimals.
    assert!(tallies[&2004].warnings[0]
        .message
        .contains("invalid geometry from the meshtal header"));
}

#[test]
fn cf_volumes_match_the_input_geometry() {
    let tallies = with_input(&input());
    for id in [2004, 2034] {
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
fn energy_and_time_bins_come_from_the_input() {
    let t = &with_input(&input())[&2054];
    let e = &t.energy.bounds().unwrap();
    assert_eq!(e.len(), 7);
    assert_eq!(e[0], 0.0);
    assert_eq!(e[1], 1.23456 / 2.0);
    assert_eq!(e[6], 20.2468);
    assert_eq!(t.time.bounds().unwrap()[..2], [-1e36, 0.123456]);
    assert_eq!(t.time.bounds().unwrap().len(), 4);
    let Geometry::Rectangular { transform, .. } = &t.geometry else {
        panic!("rectangular mesh expected")
    };
    assert!(transform.is_some());
}

#[test]
fn an_input_that_does_not_match_is_ignored_with_a_warning() {
    // A different mesh for 2004, and no 2014 at all.
    let src = input()
        .replace("IMESH -3.001 -3 2I 12.777", "IMESH -3.001 -3 2I 12.9")
        .replace("FMESH2014:N", "FMESH2994:N");
    let tallies = with_input(&src);
    let header = by_id(
        Reader::from_bytes(meshtal().into_bytes(), None)
            .read_all()
            .unwrap(),
    );
    for (id, why) in [
        (2004, "x bound 3 is 2.26 in the header, 2.3 in the input"),
        (2014, "no FMESH2014 card"),
    ] {
        let t = &tallies[&id];
        assert!(!t.geometry_from_input, "tally {id}");
        assert_eq!(t.geometry, header[&id].geometry, "tally {id}");
        let warning = t
            .warnings
            .iter()
            .find(|w| w.message.contains(why))
            .unwrap_or_else(|| panic!("tally {id}: {:?}", t.warnings));
        assert!(warning
            .message
            .contains("geometry is from the meshtal header"));
    }
    // An unreadable card says why.
    let broken = input().replace("TR=3 OUT=IJ", "TR=9 OUT=IJ");
    let t = &with_input(&broken)[&2054];
    assert!(!t.geometry_from_input);
    assert!(
        t.warnings[0].message.contains("no TR9 card"),
        "{:?}",
        t.warnings
    );
}
