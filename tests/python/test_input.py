from pathlib import Path

import aci_tallies
import numpy as np
import pytest
import pyvista as pv

FIXTURES = Path(__file__).parents[1] / "fixtures"
MESHTAL = FIXTURES / "pencil_beam_input.msht"
INPUT = FIXTURES / "pencil_beam_input.mcnp"


def _by_id(parsed):
    return {t.id: t for t in parsed.read_all()}


@pytest.fixture
def exact():
    return _by_id(aci_tallies.open(MESHTAL, input=INPUT))


@pytest.fixture(scope="module")
def header_only():
    return _by_id(aci_tallies.open(str(MESHTAL)))


def test_input_geometry_replaces_the_header(exact, header_only):
    assert set(exact) == set(range(2004, 2105, 10))
    # A cylinder under a reflecting TR= can't be described, so it keeps the
    # header's geometry.
    mirrored = exact.pop(2104)
    assert not mirrored.geometry_from_input
    assert "reverses the theta direction" in mirrored.warnings[0].message
    for tally_id, t in exact.items():
        assert t.geometry_from_input, tally_id
        assert t.warnings == [], tally_id
        assert t.energy_weighted == (tally_id == 2064)
        h = header_only[tally_id]
        assert not h.geometry_from_input
        assert h.energy_weighted is None
        assert h.geometry.transform is None
        # The header is the input rounded to two decimals.
        assert np.allclose(
            t.geometry.bounds[0], h.geometry.bounds[0], atol=0.0051, rtol=0
        )
    t = exact[2004]
    assert t.geometry.kind == "rectangular"
    assert t.geometry.bounds[0][:3] == pytest.approx([-12.3456, -3.001, -3.0])
    assert t.geometry.transform is not None and t.geometry.transform.shape == (4, 4)
    rotation = t.geometry.transform[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert t.geometry.transform[:3, 3] == pytest.approx([2.71828, -1.41421, 1.73205])
    # A cylinder's TR= moves its frame instead.
    assert exact[2034].geometry.transform is None
    assert exact[2034].geometry.origin != pytest.approx(
        header_only[2034].geometry.origin, abs=0.1
    )


def test_mismatched_input_keeps_the_header_with_a_warning(header_only):
    other = FIXTURES / "pencil_beam.mcnp"
    tallies = _by_id(aci_tallies.open(MESHTAL, input=other))
    t = tallies[2014]
    assert not t.geometry_from_input
    assert any("no FMESH2014 card" in w.message for w in t.warnings)
    assert np.array_equal(t.geometry.bounds[0], header_only[2014].geometry.bounds[0])


def test_transformed_mesh_is_a_structured_grid_in_the_world(exact, tmp_path):
    t = exact[2004]
    grid = t.to_pyvista()
    assert isinstance(grid, pv.StructuredGrid)
    x, y, z = t.geometry.bounds[0], t.geometry.bounds[1], t.geometry.bounds[2]
    corners = np.array([[x[-1], y[0], z[0], 1.0], [x[0], y[-1], z[-1], 1.0]])
    world = (t.geometry.transform @ corners.T).T[:, :3]
    nx = len(x)
    assert np.allclose(grid.points[nx - 1], world[0])
    assert np.allclose(grid.points[-nx], world[1])
    volumes = grid.compute_cell_sizes()["Volume"]
    assert np.all(volumes > 0)
    assert volumes.sum() == pytest.approx(np.ptp(x) * np.ptp(y) * np.ptp(z))

    # A reflected mesh is not turned inside out, and each cell still carries
    # its own voxel's value.
    m = exact[2074]
    assert np.linalg.det(m.geometry.transform[:3, :3]) < 0
    mirrored = m.to_pyvista()
    assert np.all(mirrored.compute_cell_sizes()["Volume"] > 0)
    centers = np.c_[mirrored.cell_centers().points, np.ones(mirrored.n_cells)]
    local = (np.linalg.inv(m.geometry.transform) @ centers.T).T[:, :3]
    ijk = [
        np.searchsorted(b, c) - 1
        for b, c in zip(
            (m.geometry.bounds[0], m.geometry.bounds[1], m.geometry.bounds[2]), local.T
        )
    ]
    assert np.array_equal(mirrored.cell_data["result"], m.result[0, 0][tuple(ijk)])

    out = tmp_path / "tally2004.vts"
    t.to_vtk(out)
    reloaded = pv.read(out)
    assert np.allclose(reloaded.points, grid.points)
    assert np.array_equal(reloaded.cell_data["result"], grid.cell_data["result"])
    with pytest.raises(ValueError, match=r"\.vts"):
        t.to_vtk(tmp_path / "tally2004.vtr")


def test_ops_keep_the_input_geometry(exact):
    t = exact[2004]
    for derived in [t + t, 2.0 * t, t.isel(i=slice(1, 4)), t.integrate_space()]:
        assert derived.geometry_from_input
        assert np.array_equal(derived.geometry.transform, t.geometry.transform)
    assert exact[2064].integrate_space().energy_weighted
