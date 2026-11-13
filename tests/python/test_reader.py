"""Reading by parts, and the API around it."""

from pathlib import Path

import aci_tallies
import numpy as np
import pytest
from aci_tallies import meshtal

FIXTURES = Path(__file__).parents[1] / "fixtures"
PENCIL = FIXTURES / "pencil_beam.msht"


@pytest.fixture(scope="module")
def f() -> meshtal.MeshtalFile:
    return meshtal.open(PENCIL)


def test_file_metadata(f: meshtal.MeshtalFile) -> None:
    assert f.title.startswith("Pencil beam through void")
    assert f.histories == 100.0
    assert f.code == ("mcnp6", "6")
    assert f.probid == "10/15/26 14:36:50"
    assert f.ids == [1004, 1014, 1024, 1034, 1044, 1054, 1064, 1074, 1084, 1094, 1104]
    assert len(f) == 11 and 1064 in f and 7 not in f
    assert list(f) == f.ids == f.keys()
    assert f.failed == []
    assert f.is_current()
    info = f.info(1064)
    assert info.shape == [4, 4, 10, 4, 5]
    assert info.out_format == "col"
    assert info.energy.total and info.time.total
    with pytest.raises(KeyError, match="no tally 7"):
        f[7]


def _index(selected: int | str, axis: aci_tallies.Axis) -> int:
    if isinstance(selected, int):
        return selected
    assert axis.total_index is not None
    return axis.total_index


def test_bins_read_alone_are_slices_of_the_whole(f: meshtal.MeshtalFile) -> None:
    for tally_id in (1064, 1074, 1084, 1094):  # col, ij, ik, jk
        whole = f[tally_id]
        for energy in (0, 2, "total"):
            for time in (1, "total", None):
                part = f.read(tally_id, energy=energy, time=time)
                e = _index(energy, whole.energy)
                t = (
                    slice(None)
                    if time is None
                    else slice(_index(time, whole.time), _index(time, whole.time) + 1)
                )
                want = whole.result[e : e + 1, t]
                assert np.array_equal(part.result, want), (tally_id, energy, time)
                assert np.array_equal(part.rel_error, whole.rel_error[e : e + 1, t])
        total = f.read(tally_id, energy="total")
        assert total.energy.n_bins == 1 and not total.energy.total
        assert total.energy.bounds is not None
        assert total.energy.bounds.tolist() == [0.0, 20.0]
    with pytest.raises(ValueError, match="no Total bin"):
        f.read(1054, energy="total")
    with pytest.raises(IndexError):
        f.read(1064, energy=3)


def test_arrays_are_read_only_views(f: meshtal.MeshtalFile) -> None:
    t = f[1004]
    for a in (t.result, t.rel_error, t.volume):
        assert a is not None
        assert not a.flags.writeable
        assert a.base is not None  # a view of what Rust read, not a copy
    with pytest.raises(ValueError):
        t.result[0, 0, 0, 0, 0] = 1.0
    assert t.volume is not None and t.volume.shape == (10, 4, 5)
    # The views keep the tally alive.
    r = f[1004].result
    assert r.sum() == t.result.sum()


def test_isel_takes_ints_slices_and_total(f: meshtal.MeshtalFile) -> None:
    t = f[1064]
    assert t.isel(energy=1, time=-1).shape == [1, 1, 10, 4, 5]
    assert t.isel(i=slice(2, 5), k=-1).shape == [4, 4, 3, 4, 1]
    assert np.array_equal(t.isel(energy="total").result, t.integrate_energy().result)
    assert np.array_equal(t.isel(energy="total").result[0], t.result[-1])
    with pytest.raises(ValueError, match="no step"):
        t.isel(i=slice(0, 4, 2))
    with pytest.raises(ValueError, match="not a selection"):
        t.isel(i="total")  # type: ignore[arg-type]


def test_input_is_reused_and_formats_detected() -> None:
    inp = aci_tallies.Input(FIXTURES / "pencil_beam_input.mcnp")
    a = meshtal.open(FIXTURES / "pencil_beam_input.msht", input=inp)
    b = meshtal.open(
        FIXTURES / "pencil_beam_input.msht",
        input=str(FIXTURES / "pencil_beam_input.mcnp"),
    )
    assert a.info(2004).geometry == b.info(2004).geometry
    assert a.info(2004).geometry_from_input
    assert aci_tallies.detect(PENCIL) == "meshtal"
    assert aci_tallies.detect(FIXTURES / "pencil_beam.mcnp") is None
    with pytest.raises(ValueError, match="not a file aci_tallies reads"):
        aci_tallies.open(FIXTURES / "pencil_beam.mcnp")


def test_geometry(f: meshtal.MeshtalFile) -> None:
    g = f.info(1004).geometry
    assert g.kind == "rectangular" and g.axis_names == ("x", "y", "z")
    assert g.shape == (10, 4, 5)
    v = g.volumes()
    assert v.shape == (10, 4, 5) and np.all(v > 0)
    c = g.centers()
    assert c.shape == (10, 4, 5, 3)
    ijk = g.locate(c.reshape(-1, 3))
    assert np.array_equal(ijk, np.argwhere(np.ones(g.shape)))
    assert g.locate(np.array([[1e9, 0.0, 0.0]])).tolist() == [[-1, -1, -1]]
    cyl = f.info(1014).geometry
    assert cyl.kind == "cylindrical" and cyl.axis_names == ("r", "z", "theta")
    assert cyl.origin is not None and cyl.transform is None


def test_arithmetic_returns_not_implemented_for_strangers(
    f: meshtal.MeshtalFile,
) -> None:
    t = f[1054]
    with pytest.raises(TypeError):
        t + 1.0  # type: ignore[operator]
    with pytest.raises(TypeError):
        t * "x"  # type: ignore[operator]
    assert np.array_equal((-t).result, -t.result)
    with pytest.raises(ValueError, match="different shapes"):
        t + f[1064]


def test_xarray_and_html(f: meshtal.MeshtalFile) -> None:
    pytest.importorskip("xarray")
    t = f[1064]
    ds = t.to_xarray()
    assert ds["result"].dims == ("energy", "time", "x", "y", "z")
    assert bool(ds["energy_is_total"][-1])
    assert np.array_equal(ds["result"].values, t.result)
    assert "MeshTally 1064" in t._repr_html_()
    page = f._repr_html_()
    assert "1104" in page and "Pencil beam" in page
