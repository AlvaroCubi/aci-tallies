from pathlib import Path

import aci_tallies
import numpy as np
import pytest
import pyvista as pv

EXAMPLE = Path(__file__).parents[2] / "examples" / "meshtal"


@pytest.fixture(scope="module")
def parsed():
    return {t.id: t for t in aci_tallies.open(str(EXAMPLE)).read_all()}


def _tally(parsed, tally_id):
    return parsed[tally_id]


def test_rectangular_grid_cell_ordering_matches_result_array(parsed):
    """Confirm our Fortran-order cell_data flatten actually lines up with
    PyVista/VTK's own cell ordering, by checking a specific known voxel's
    geometric center against its known coordinates and value.
    """
    t = _tally(parsed, 14)
    grid = t.to_pyvista()
    assert isinstance(grid, pv.RectilinearGrid)
    assert grid.n_cells == 5 * 5 * 5

    centers = grid.cell_centers().points
    # voxel (i=0, j=0, k=0) is Fortran-order flat index 0.
    assert centers[0] == pytest.approx((-8.0, -8.0, -8.0), abs=1e-6)
    assert grid.cell_data["result"][0] == pytest.approx(4.69512e-04, rel=1e-6)

    # voxel (i=2, j=2, k=2), the center voxel: flat index = 2 + 2*5 + 2*25 = 62.
    flat_idx = 2 + 2 * 5 + 2 * 25
    assert centers[flat_idx] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert grid.cell_data["result"][flat_idx] == pytest.approx(3.82462e-02, rel=1e-6)


def test_cylindrical_grid_points_lie_on_bin_boundaries(parsed):
    """Recompute (r, z, theta) of every grid point using nothing but
    `origin`/`axis`/`vec`, and check they land on the tally's bin bounds
    (theta on the subdivided bounds). Also checks the cells are not inside
    out: VTK cell volumes must all be positive.
    """
    t = _tally(parsed, 74)
    grid = t.to_pyvista(max_theta_step=0)
    assert isinstance(grid, pv.StructuredGrid)
    assert grid.dimensions == (11, 11, 11)  # (r, theta, z) point counts
    points = np.asarray(grid.points).reshape(11, 11, 11, 3)  # [z, theta, r]

    origin = np.array(t.geometry.origin)
    axis = np.array(t.geometry.axis) / np.linalg.norm(t.geometry.axis)
    vec = np.array(t.geometry.vec)
    e_ref = vec - np.dot(vec, axis) * axis
    e_ref /= np.linalg.norm(e_ref)
    e_perp = np.cross(axis, e_ref)

    rel = points - origin
    z_local = rel @ axis
    radial = rel - z_local[..., None] * axis
    r_local = np.linalg.norm(radial, axis=-1)
    theta_local = np.mod(np.arctan2(radial @ e_perp, radial @ e_ref) / (2 * np.pi), 1.0)

    r_bounds = np.asarray(t.geometry.bounds[0])
    z_bounds = np.asarray(t.geometry.bounds[1])
    theta_bounds = np.asarray(t.geometry.bounds[2])
    assert np.allclose(z_local, z_bounds[:, None, None], atol=1e-9)
    assert np.allclose(r_local, r_bounds[None, None, :], atol=1e-9)
    on_axis = np.broadcast_to(r_bounds[None, None, :] < 1e-9, theta_local.shape)
    expected = np.broadcast_to(np.mod(theta_bounds, 1.0)[None, :, None], theta_local.shape)
    diff = np.abs(theta_local[~on_axis] - expected[~on_axis])
    assert np.all(np.minimum(diff, 1.0 - diff) < 1e-9)

    assert np.all(grid.compute_cell_sizes()["Volume"] > 0)


def test_cylindrical_grid_value_at_known_world_point(parsed):
    t = _tally(parsed, 74)
    # 0.1-revolution theta bins split into 8 sub-cells of at most 1/72 rev.
    grid = t.to_pyvista()
    assert isinstance(grid, pv.StructuredGrid)
    assert grid.n_cells == 10 * 10 * 10 * 8

    # The cell for voxel (r=0, z=0, theta=0) spans R=[0,1], Z=[0,2] (local),
    # Th=[0.0, 0.1] rev — pick a world point safely inside it and confirm
    # the grid cell containing that point carries the expected result.
    theta_rad = 0.05 * 2 * np.pi
    point_inside = np.array([0.0, 0.0, -9.0]) + 0.5 * np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

    cell_id = grid.find_containing_cell(point_inside)
    assert cell_id != -1, "expected point to fall inside the cylindrical grid"
    assert grid.cell_data["result"][cell_id] == pytest.approx(1.09801e-03, rel=1e-6)

    # Subdivision only repeats values, and 5-degree flat-faced sub-cells
    # approximate the true annular sectors to sin(x)/x ~ 0.9987, so the
    # volume integral matches the exact one closely.
    r, z, th = (np.asarray(b) for b in (t.geometry.bounds[0], t.geometry.bounds[1], t.geometry.bounds[2]))
    exact_volume = (
        np.pi * np.diff(r**2)[:, None, None] * np.diff(z)[None, :, None] * np.diff(th)[None, None, :]
    )
    exact_sum = np.sum(t.result[0, 0] * exact_volume)
    grid_sum = np.sum(grid.cell_data["result"] * grid.compute_cell_sizes()["Volume"])
    assert grid_sum == pytest.approx(exact_sum, rel=2e-3)


@pytest.mark.parametrize(("tally_id", "suffix"), [(14, "vtr"), (34, "vtr"), (64, "vtr"), (74, "vts")])
@pytest.mark.parametrize("compress", [True, False])
def test_to_vtk_file_matches_to_pyvista(parsed, tmp_path, tally_id, suffix, compress):
    t = _tally(parsed, tally_id)
    out = tmp_path / f"tally{tally_id}.{suffix}"
    t.to_vtk(out, compress=compress)
    reloaded = pv.read(out)
    in_memory = t.to_pyvista()
    assert type(reloaded) is type(in_memory)
    assert reloaded.n_cells == in_memory.n_cells
    assert np.allclose(reloaded.points, in_memory.points)
    assert set(reloaded.cell_data.keys()) == set(in_memory.cell_data.keys())
    for name in in_memory.cell_data:
        assert np.array_equal(reloaded.cell_data[name], in_memory.cell_data[name]), name


def test_to_vtk_rejects_wrong_extension(parsed, tmp_path):
    with pytest.raises(ValueError, match=r"\.vtr"):
        _tally(parsed, 14).to_vtk(tmp_path / "tally14.vts")


def test_energy_index_out_of_range_raises(parsed):
    t = _tally(parsed, 14)
    with pytest.raises(ValueError):
        t.to_pyvista(energy=5)


def test_explicit_single_energy_index_keeps_plain_array_names(parsed):
    t = _tally(parsed, 34)  # 10 energy groups
    grid0 = t.to_pyvista(energy=0)
    grid1 = t.to_pyvista(energy=1)
    assert list(grid0.cell_data.keys()) == ["result", "rel_error"]
    assert not np.allclose(grid0.cell_data["result"], grid1.cell_data["result"])


def test_default_attaches_every_energy_group_including_total(parsed):
    t = _tally(parsed, 34)  # 10 real energy groups + a native Total
    grid = t.to_pyvista()
    expected_names = {f"result_e{i}" for i in range(10)} | {"result_etotal"}
    expected_names |= {f"rel_error_e{i}" for i in range(10)} | {"rel_error_etotal"}
    assert set(grid.array_names) == expected_names
    assert np.allclose(
        grid.cell_data["result_e3"], t.result[3, 0].flatten(order="F")
    )
    assert np.allclose(
        grid.cell_data["result_etotal"],
        t.result[t.energy.total_index, 0].flatten(order="F"),
    )


def test_default_attaches_every_energy_and_time_combination(parsed):
    t = _tally(parsed, 94)  # 10 energy groups x 3 time bins, both have a Total
    grid = t.to_pyvista(fields=["result"])  # narrow fields to keep this test readable
    assert len(grid.array_names) == 11 * 4  # (10 real + Total) x (3 real + Total)
    assert np.allclose(
        grid.cell_data["result_etotal_ttotal"],
        t.result[t.energy.total_index, t.time.total_index].flatten(order="F"),
    )


def test_explicit_sequence_narrows_default_selection(parsed):
    t = _tally(parsed, 34)
    grid = t.to_pyvista(energy=[0, "total"], fields=["result"])
    assert set(grid.array_names) == {"result_e0", "result_etotal"}


def test_narrowing_the_tally_first_narrows_default_output(parsed):
    t = _tally(parsed, 34)
    sliced = t.isel(energy=slice(0, 3))  # 3 real groups, isel drops the Total flag
    assert not sliced.energy.total
    grid = sliced.to_pyvista(fields=["result"])
    assert set(grid.array_names) == {"result_e0", "result_e1", "result_e2"}


def test_empty_index_sequence_raises(parsed):
    t = _tally(parsed, 34)
    with pytest.raises(ValueError):
        t.to_pyvista(energy=[])


def test_fields_none_attaches_all_available_arrays_by_default(parsed):
    t = _tally(parsed, 64)  # out=cf: has an explicit volume array
    grid = t.to_pyvista()
    assert sorted(grid.cell_data.keys()) == ["rel_error", "result", "volume"]


def test_fields_subset_attaches_only_requested_arrays(parsed):
    t = _tally(parsed, 64)
    grid = t.to_pyvista(fields=["result"])
    assert list(grid.cell_data.keys()) == ["result"]


def test_fields_unknown_name_raises(parsed):
    t = _tally(parsed, 14)
    with pytest.raises(ValueError):
        t.to_pyvista(fields=["bogus"])


def test_energy_index_total_selects_native_total_slice(parsed):
    t = _tally(parsed, 34)  # 10 real energy groups + a native Total
    grid = t.to_pyvista(energy="total")
    assert np.allclose(
        grid.cell_data["result"],
        t.result[t.energy.total_index, 0].flatten(order="F"),
    )


def test_energy_index_total_without_a_total_bin_raises(parsed):
    t = _tally(parsed, 14)  # single implicit energy group: no Total
    with pytest.raises(ValueError):
        t.to_pyvista(energy="total")


def test_time_index_selects_time_discretized_tally(parsed):
    t = _tally(parsed, 84)  # 3 real time bins + a Total; bins 1/2 are zero
    grid1 = t.to_pyvista(time=1)
    grid_total = t.to_pyvista(time="total")
    assert not np.allclose(grid1.cell_data["result"], grid_total.cell_data["result"])
    assert np.allclose(
        grid_total.cell_data["result"],
        t.result[0, t.time.total_index].flatten(order="F"),
    )


def test_energy_and_time_total_together_matches_grand_total_slice(parsed):
    t = _tally(parsed, 94)  # energy- and time-discretized, both have a Total
    grid = t.to_pyvista(energy="total", time="total")
    assert np.allclose(
        grid.cell_data["result"],
        t.result[t.energy.total_index, t.time.total_index].flatten(order="F"),
    )
