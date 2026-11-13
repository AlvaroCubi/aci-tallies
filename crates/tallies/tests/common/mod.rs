//! The pencil-beam fixtures: `tests/fixtures/pencil_beam*.mcnp` fire every
//! history along one straight ray through void, so each voxel's flux is
//! exactly `chord length / voxel volume` with zero variance. [`chords`]
//! recomputes the chords analytically from a parsed geometry.

#![allow(dead_code)] // each test binary uses its own subset

use aci_tallies::meshtal::{Geometry, Tally};
use ndarray::s;
use std::collections::HashMap;
use std::path::PathBuf;

/// The SDEF of the fixtures.
pub const START: [f64; 3] = [-56.3597402157, -26.9768830971, -30.9828571187];
pub const DIR: [f64; 3] = [0.8151391459, 0.3668126157, 0.4483265303];
/// Radius of the void sphere the particle leaves through.
const WORLD_RADIUS: f64 = 200.0;
/// MCNP prints results with 6 significant digits.
pub const PRINT_TOL: f64 = 1e-5;

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

/// Every tally of a read fixture, by id.
pub fn by_id(tallies: Vec<Tally>) -> HashMap<u32, Tally> {
    tallies.into_iter().map(|t| (t.id, t)).collect()
}

/// The spatial slice at the last energy and time index: the Total slice
/// when MCNP printed one, else the single real bin.
pub fn total_slice(t: &Tally) -> Vec<f64> {
    let (ne, nt) = (t.result.shape()[0], t.result.shape()[1]);
    t.result
        .slice(s![ne - 1, nt - 1, .., .., ..])
        .iter()
        .copied()
        .collect()
}

/// The voxel where MCNP's flux differs most from chord / volume, as
/// `(voxel, MCNP, chord / volume)`, judged relative to the largest flux.
/// `scale` multiplies the expected value (14 for a `*FMESH` of 14 MeV).
pub fn worst_voxel(t: &Tally, scale: f64) -> (usize, f64, f64) {
    let chord = chords(&t.geometry);
    let volume = t.geometry.voxel_volumes();
    let result = total_slice(t);
    let mut worst = (0, 0.0, 0.0);
    let mut worst_diff = -1.0;
    for v in 0..result.len() {
        let expected = scale * chord[v] / volume[v];
        let diff = (result[v] - expected).abs();
        if diff > worst_diff {
            worst_diff = diff;
            worst = (v, result[v], expected);
        }
    }
    worst
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn at(t: f64) -> [f64; 3] {
    [
        START[0] + t * DIR[0],
        START[1] + t * DIR[1],
        START[2] + t * DIR[2],
    ]
}

/// Track length of the beam inside every voxel, from the ray's exact
/// crossings with every bin surface (planes, cylinders, theta half-planes).
pub fn chords(geometry: &Geometry) -> Vec<f64> {
    let b = dot(START, DIR);
    let t_end = -b + (b * b - dot(START, START) + WORLD_RADIUS * WORLD_RADIUS).sqrt();
    let mut ts = vec![0.0, t_end];
    let (a1, a2, a3) = geometry.axes();
    match geometry {
        Geometry::Rectangular { transform, .. } => {
            // Planes are axis-aligned in the mesh's own frame.
            let (start, dir) = match transform {
                None => (START, DIR),
                Some(t) => {
                    let origin = t.to_local([0.0; 3]);
                    let tip = t.to_local(DIR);
                    let dir = [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]];
                    (t.to_local(START), dir)
                }
            };
            for (axis, bounds) in [a1, a2, a3].into_iter().enumerate() {
                ts.extend(bounds.iter().map(|&v| (v - start[axis]) / dir[axis]));
            }
        }
        Geometry::Cylindrical { .. } => {
            let frame = geometry.cyl_frame().expect("cylinder has a frame");
            let rel = [
                START[0] - frame.origin[0],
                START[1] - frame.origin[1],
                START[2] - frame.origin[2],
            ];
            let (px, py, pz) = (
                dot(rel, frame.e_ref),
                dot(rel, frame.e_perp),
                dot(rel, frame.e_axis),
            );
            let (dx, dy, dz) = (
                dot(DIR, frame.e_ref),
                dot(DIR, frame.e_perp),
                dot(DIR, frame.e_axis),
            );
            for &r in a1 {
                let (qa, qb, qc) = (
                    dx * dx + dy * dy,
                    2.0 * (px * dx + py * dy),
                    px * px + py * py - r * r,
                );
                let disc = qb * qb - 4.0 * qa * qc;
                if disc >= 0.0 {
                    ts.push((-qb - disc.sqrt()) / (2.0 * qa));
                    ts.push((-qb + disc.sqrt()) / (2.0 * qa));
                }
            }
            ts.extend(a2.iter().map(|&z| (z - pz) / dz));
            for &theta in a3 {
                let (sin, cos) = (theta * std::f64::consts::TAU).sin_cos();
                ts.push((py * cos - px * sin) / (dx * sin - dy * cos));
            }
        }
    }
    ts.retain(|t| t.is_finite() && (0.0..=t_end).contains(t));
    ts.sort_by(f64::total_cmp);

    let mut out = vec![0.0; geometry.n_voxels()];
    for w in ts.windows(2) {
        if let Some(v) = geometry.locate(at(0.5 * (w[0] + w[1]))) {
            out[v] += w[1] - w[0];
        }
    }
    out
}
