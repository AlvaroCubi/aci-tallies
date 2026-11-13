//! D1SUNED cell-under-voxel files: `tests/fixtures/d1suned/` is a D1SUNED
//! 4.1.3.2 run of a synthetic problem (see its README) with three CuV
//! meshes in one cuvmsh and meshinfo (4: cylinder, 20 energy bins; 14:
//! box, 4 bins; 44: box under `TR=1`, no EMESH), and FMESH24, mesh 14
//! without `out=cuv`, in the meshtal.

use aci_tallies::common::Input;
use aci_tallies::cuv::{CuvMesh, Cuvmsh, Meshinfo};
use aci_tallies::meshtal::{Geometry, Reader};
use aci_tallies::{detect, Format};

mod common;

use common::fixture;

fn d1s(name: &str) -> std::path::PathBuf {
    fixture(&format!("d1suned/{name}"))
}

fn input() -> Input {
    Input::read(d1s("input")).unwrap()
}

#[test]
fn every_mesh_of_one_file() {
    let f = Cuvmsh::open(d1s("cuvmsh")).unwrap();
    assert_eq!(f.failed().count(), 0);
    let ids: Vec<u32> = f.tallies().map(|t| t.id).collect();
    assert_eq!(ids, [4, 14, 44]);
    assert_eq!(f.header().histories, Some(20000.0));
    let bins: Vec<usize> = f
        .tallies()
        .map(|t| t.energy.as_ref().unwrap().n_bins())
        .collect();
    assert_eq!(bins, [20, 4, 1]);
    let t14 = f.tally(14).unwrap();
    assert_eq!(
        t14.comment.as_deref(),
        Some("CuV rectangular, 4 energy bins")
    );
    assert_eq!(t14.particle.as_deref(), Some("neutron"));
    assert_eq!(t14.n_voxels, 8);
    assert_eq!(f.tally(4).unwrap().comment, None);

    let s = f.read(14).unwrap();
    assert_eq!(s.pairs.len(), 32);
    assert_eq!(s.flux.shape(), [32, 4]);
    assert_eq!(&s.pairs.cell[..5], [3, 1, 99, 2, 3]);
    assert_eq!(&s.pairs.fraction[..2], [0.579, 0.336]);
    assert_eq!(s.pairs.voxel_volume[..2], [2e5, 4e5]);
    assert_eq!(
        s.flux.row(0).to_vec(),
        [0.0, 1.09900e-7, 1.52172e-6, 4.79291e-6]
    );
    assert_eq!(s.rel_error[[0, 1]], 3.58860e-1);
    assert_eq!((s.total[0], s.total_error[0]), (6.42454e-6, 4.79452e-2));
    // The void graveyard cell scores nothing.
    assert!(s.flux.row(2).iter().all(|&x| x == 0.0));

    // One energy bin: cell lines without totals; the bin is the total.
    let s = f.read(44).unwrap();
    assert_eq!(s.flux.shape(), [12, 1]);
    for p in 0..12 {
        assert_eq!(s.total[p], s.flux[[p, 0]]);
        assert_eq!(s.total_error[p], s.rel_error[[p, 0]]);
    }

    let e = f.read(99).unwrap_err().to_string();
    assert!(e.contains("no tally 99"), "{e}");
}

#[test]
fn parts_are_the_whole() {
    let f = Cuvmsh::open(d1s("cuvmsh")).unwrap();
    for id in [4, 14, 44] {
        let whole = f.read(id).unwrap();
        let n = f.tally(id).unwrap().n_voxels;
        for cut in 0..=n {
            let a = f.read_voxels(id, 0..cut).unwrap();
            let b = f.read_voxels(id, cut..n).unwrap();
            assert_eq!(a.pairs.voxels, 0..cut);
            let mut cells = a.pairs.cell.clone();
            cells.extend(&b.pairs.cell);
            assert_eq!(cells, whole.pairs.cell, "mesh {id} cut at {cut}");
            let mut flux: Vec<f64> = a.flux.iter().copied().collect();
            flux.extend(b.flux.iter());
            assert_eq!(flux, whole.flux.iter().copied().collect::<Vec<_>>());
            assert_eq!(a.pairs.voxel_volume.len() + b.pairs.voxel_volume.len(), n);
        }
    }
    let e = f.read_voxels(14, 0..9).unwrap_err().to_string();
    assert!(e.contains("voxel"), "{e}");
}

#[test]
fn meshinfo_and_cuvmsh_together() {
    let m = Meshinfo::open(d1s("meshinfo")).unwrap();
    assert_eq!(m.tallies().map(|t| t.id).collect::<Vec<_>>(), [4, 14, 44]);
    assert_eq!(m.header().code, "mcnp");
    assert_eq!(m.header().version, "6.mpi");
    assert!(m.tally(14).unwrap().energy.is_none());

    let both = CuvMesh::open(d1s("cuvmsh"), d1s("meshinfo"), None).unwrap();
    let part = both.read(14).unwrap();
    let d = &part.decomposition;
    assert_eq!(d.pairs, part.spectra.pairs);
    assert_eq!(&d.material[..5], [0, 1, 0, 2, 0]);
    assert_eq!(&d.density[..5], [0.0, 7.87, 0.0, 1.0, 0.0]);
    let volume = d.pairs.volume();
    assert!((volume[0] - 0.579 * 2e5).abs() < 1e-6);
    // The 4 meshes' pairs sum to the voxels' volumes.
    let all = both.read(4).unwrap();
    assert_eq!(all.decomposition.pairs.len(), 7);
}

#[test]
fn pairs_of_another_run_are_refused() {
    let cuvmsh = std::fs::read_to_string(d1s("cuvmsh")).unwrap();
    let meshinfo = std::fs::read(d1s("meshinfo")).unwrap();
    // Voxel 1 of mesh 14 lists cell 1 as cell 7.
    let other = cuvmsh.replacen(
        "         1 0.33600 4.79822E-05",
        "         7 0.33600 4.79822E-05",
        1,
    );
    assert_ne!(other, cuvmsh);
    let both = CuvMesh {
        cuvmsh: Cuvmsh::from_bytes(other.into_bytes(), None),
        meshinfo: Meshinfo::from_bytes(meshinfo, None),
    };
    assert!(both.read(4).is_ok());
    let e = both.read(14).unwrap_err().to_string();
    assert!(e.contains("same run"), "{e}");
}

#[test]
fn broken_files_are_refused() {
    let cuvmsh = std::fs::read_to_string(d1s("cuvmsh")).unwrap();
    // A spectrum value lost.
    let short = cuvmsh.replacen(
        " 0.00000E+00 1.09900E-07 1.52172E-06 4.79291E-06",
        " 0.00000E+00 1.09900E-07 1.52172E-06",
        1,
    );
    let f = Cuvmsh::from_bytes(short.into_bytes(), None);
    let e = f.read(14).unwrap_err().to_string();
    assert!(e.contains("values"), "{e}");
    assert!(f.read(4).is_ok());
    // Voxels out of the geometry's order.
    let swapped = cuvmsh.replacen(
        "   -50.000   -50.000    10.000",
        "   -50.000   -50.000    40.000",
        1,
    );
    let e = Cuvmsh::from_bytes(swapped.into_bytes(), None)
        .read(14)
        .unwrap_err()
        .to_string();
    assert!(e.contains("order"), "{e}");
    // A cell more than declared.
    let meshinfo = std::fs::read_to_string(d1s("meshinfo")).unwrap();
    let extra = meshinfo.replacen(
        "          2    1.0000        2  1.00000\n",
        "          2    1.0000        2  1.00000\n          3    0.0000        0  0.00000\n",
        1,
    );
    let e = Meshinfo::from_bytes(extra.into_bytes(), None)
        .read(4)
        .unwrap_err()
        .to_string();
    assert!(e.contains("declares 1 cells but lists 2"), "{e}");
}

#[test]
fn exact_geometry_from_the_input() {
    let input = input();
    let f = Cuvmsh::open_with_input(d1s("cuvmsh"), &input).unwrap();
    for t in f.tallies() {
        assert!(t.geometry_from_input, "mesh {}: {:?}", t.id, t.warnings);
        assert!(t.warnings.is_empty(), "mesh {}: {:?}", t.id, t.warnings);
    }
    let e4 = f
        .tally(4)
        .unwrap()
        .energy
        .as_ref()
        .unwrap()
        .bounds()
        .unwrap()
        .to_vec();
    assert_eq!(e4[1], 1e-8);
    assert_eq!(e4[19], 7.0711); // printed 7.07E+00
    let Geometry::Rectangular { transform, .. } = &f.tally(44).unwrap().geometry else {
        panic!("box")
    };
    assert!(transform.is_some());
    // Without it: the printed values, and warnings saying so.
    let g = Cuvmsh::open(d1s("cuvmsh")).unwrap();
    assert_eq!(
        g.tally(4)
            .unwrap()
            .energy
            .as_ref()
            .unwrap()
            .bounds()
            .unwrap()[19],
        7.07
    );
    assert!(g.tally(4).unwrap().warnings.len() == 2);
    let m = Meshinfo::open_with_input(d1s("meshinfo"), &input).unwrap();
    assert!(m
        .tallies()
        .all(|t| t.geometry_from_input && t.warnings.is_empty()));
    let both = CuvMesh::open(d1s("cuvmsh"), d1s("meshinfo"), Some(&input)).unwrap();
    assert!(both.read(44).is_ok());
}

#[test]
fn voxel_flux_is_the_plain_fmesh() {
    let input = input();
    let f = Cuvmsh::open_with_input(d1s("cuvmsh"), &input).unwrap();
    let cuv = f.to_mesh(14).unwrap();
    let twin = Reader::open_with_input(d1s("meshtal"), &input)
        .unwrap()
        .read(24)
        .unwrap();
    assert_eq!(cuv.geometry, twin.geometry);
    assert_eq!(cuv.energy, twin.energy);
    assert_eq!(cuv.result.shape(), twin.result.shape());
    // The values to MCNP's 6 printed digits; the errors, combined as if
    // the cells were independent, a little low (2-18% here).
    let mut worst = 0.0f64;
    for (n, (a, b)) in cuv.result.iter().zip(twin.result.iter()).enumerate() {
        worst = worst.max((a - b).abs() / b);
        let ratio = cuv.rel_error.as_slice().unwrap()[n] / twin.rel_error.as_slice().unwrap()[n];
        assert!((0.8..=1.0).contains(&ratio), "{n}: error ratio {ratio}");
    }
    assert!(worst < 1e-5, "{worst}");
    assert_eq!(cuv.volume.as_ref().unwrap()[[0, 0, 0]], 2e5);
}

#[test]
fn legacy_cylinders_are_corrected() {
    // Made from `meshinfo` by `make_legacy.py` (see the README).
    let old = Meshinfo::open(d1s("legacy_3.1.4_cyl/meshinfo")).unwrap();
    let new = Meshinfo::open(d1s("meshinfo")).unwrap();
    let (o, n) = (old.tally(4).unwrap(), new.tally(4).unwrap());
    assert!(o.legacy_cylindrical && !n.legacy_cylindrical);
    assert!(o.warnings.iter().any(|w| w.message.contains("radians")));
    assert_eq!(o.geometry, n.geometry);
    let (a, b) = (old.read(4).unwrap(), new.read(4).unwrap());
    assert_eq!(a.pairs.cell, b.pairs.cell);
    assert_eq!(a.pairs.fraction, b.pairs.fraction);
    for (x, y) in a.pairs.voxel_volume.iter().zip(&b.pairs.voxel_volume) {
        assert!((x - y).abs() <= 1e-5 * y, "{x} {y}");
    }
}

#[test]
fn formats_are_told_apart() {
    let head = |name: &str| {
        let b = std::fs::read(d1s(name)).unwrap();
        b[..b.len().min(4096)].to_vec()
    };
    assert_eq!(detect(&head("cuvmsh")), Some(Format::Cuvmsh));
    assert_eq!(detect(&head("meshinfo")), Some(Format::Meshinfo));
    assert_eq!(detect(&head("meshtal")), Some(Format::Meshtal));
    assert_eq!(
        detect(&head("legacy_3.1.4_cyl/meshinfo")),
        Some(Format::Meshinfo)
    );
}
