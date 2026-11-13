"""D1SUNED cell-under-voxel files (`tests/fixtures/d1suned`, see its README)."""

from pathlib import Path

import aci_tallies
import numpy as np
import pytest
from aci_tallies import cuv

D1S = Path(__file__).parents[2] / "tests" / "fixtures" / "d1suned"
CUVMSH, MESHINFO, INPUT = (str(D1S / n) for n in ("cuvmsh", "meshinfo", "input"))


def test_open_tells_the_files_apart():
    assert aci_tallies.detect(CUVMSH) == "cuvmsh"
    assert aci_tallies.detect(MESHINFO) == "meshinfo"
    assert aci_tallies.detect(str(D1S / "meshtal")) == "meshtal"
    f = aci_tallies.open(CUVMSH)
    assert isinstance(f, cuv.CuvFile) and f.kind == "cuvmsh"
    assert f.ids == [4, 14, 44] and list(f) == [4, 14, 44] and 14 in f
    assert aci_tallies.open(MESHINFO).kind == "meshinfo"
    with pytest.raises(ValueError):
        cuv.open(str(D1S / "meshtal"))


def test_pairs_are_read_only_arrays():
    f = cuv.open(CUVMSH)
    p = f[14]
    assert len(p) == 32 and p.voxels == (0, 8)
    assert p.flux.shape == (32, 4) and p.energy.n_bins == 4
    assert list(p.cell[:5]) == [3, 1, 99, 2, 3]
    assert p.material is None and p.density is None
    assert not p.flux.flags.writeable and not p.cell.flags.writeable
    np.testing.assert_allclose(p.volume, p.fraction * p.voxel_volume[p.voxel])
    with pytest.raises(KeyError):
        f[99]


def test_a_run_of_voxels():
    f = cuv.open(CUVMSH)
    whole = f[14]
    part = f.read(14, voxels=slice(2, 5))
    assert part.voxels == (2, 5) and len(part.voxel_volume) == 3
    mask = (whole.voxel >= 2) & (whole.voxel < 5)
    np.testing.assert_array_equal(part.flux, whole.flux[mask])
    assert f.read(14, voxels=-1).voxels == (7, 8)
    with pytest.raises(IndexError):
        f.read(14, voxels=8)


def test_both_files_together():
    both = cuv.open_mesh(CUVMSH, MESHINFO, input=INPUT)
    assert both.kind == "cuvmsh+meshinfo"
    p = both[14]
    assert p.flux is not None and p.material is not None
    assert list(p.material[:5]) == [0, 1, 0, 2, 0]
    assert p.geometry_from_input


def test_exact_geometry_from_the_input():
    f = cuv.open(CUVMSH, input=INPUT)
    assert all(t.geometry_from_input and not t.warnings for t in f.tallies)
    assert f.info(4).energy.bounds[19] == 7.0711
    assert f.info(44).geometry.transform is not None
    assert cuv.open(CUVMSH).info(4).energy.bounds[19] == 7.07


def test_voxel_flux_is_the_plain_fmesh():
    f = cuv.open(CUVMSH, input=INPUT)
    mesh = f.to_mesh(14)
    twin = aci_tallies.open(str(D1S / "meshtal"), input=INPUT)[24]
    assert mesh.out_format == "cuv" and mesh.shape == twin.shape
    np.testing.assert_allclose(mesh.result, twin.result, rtol=1e-5)
    with pytest.raises(ValueError, match="cuvmsh"):
        cuv.open(MESHINFO).to_mesh(14)


def test_tables(tmp_path):
    pytest.importorskip("polars")
    p = cuv.open_mesh(CUVMSH, MESHINFO)[14]
    df = p.to_polars()
    assert df.columns == [
        "voxel",
        "x",
        "y",
        "z",
        "cell",
        "fraction",
        "volume",
        "material",
        "density",
        "total",
        "total_error",
    ]
    assert df.height == 32 and df["z"][1] == 10.0
    long = p.to_polars(spectra=True)
    assert long.height == 128 and long["energy_high"][3] == 14.5
    p.to_parquet(tmp_path / "p.parquet")
    with pytest.raises(ValueError, match="cuvmsh"):
        cuv.open(MESHINFO)[14].to_polars(spectra=True)


def test_xarray_and_vtk(tmp_path):
    pytest.importorskip("xarray")
    ds = cuv.open(CUVMSH)[4].to_xarray()
    assert ds.flux.dims == ("pair", "energy") and ds.sizes["energy"] == 20
    assert set(ds.coords) >= {"voxel", "cell", "r", "z", "theta"}
    cuv.open(CUVMSH).to_mesh(4).to_vtk(tmp_path / "m4.vts")
    assert (tmp_path / "m4.vts").stat().st_size > 0


def test_html():
    f = cuv.open_mesh(CUVMSH, MESHINFO)
    assert "CuvFile" in f._repr_html_() and "CuvPairs 14" in f[14]._repr_html_()
