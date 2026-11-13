"""MCTAL files, against MCNP 6.2/6.3 output (see crates/tallies/tests/mctal_probe.rs
for the exact values)."""

import logging
from pathlib import Path

import aci_tallies
import numpy as np
import pytest
from aci_tallies import mctal

FIXTURES = Path(__file__).parents[1] / "fixtures" / "mctal"
CELL = 20.0 / 8000.0


@pytest.fixture(scope="module")
def probe() -> mctal.MctalFile:
    f = aci_tallies.open(FIXTURES / "probe.m")
    assert isinstance(f, mctal.MctalFile)
    return f


def test_file(probe: mctal.MctalFile) -> None:
    assert probe.ids == [1, 2, 4, 11, 14, 24, 34, 44, 54, 64, 74, 84, 94]
    assert probe.histories == 100 and probe.dump == 2 and probe.perturbations == 0
    assert probe.code == ("mcnp", "6.mpi")
    assert probe.kcode is None and probe.failed == [] and probe.warnings == []
    info = probe.info(64)
    assert info.shape == [4, 1, 1, 1, 1, 1, 4, 3]
    assert info.dims == ["f", "d", "u", "s", "m", "c", "e", "t"]
    assert info.kind == 4 and info.particles == "n" and info.detector is None
    assert "<b>MctalFile</b>" in probe._repr_html_()


def test_values_and_bins(probe: mctal.MctalFile) -> None:
    t = probe[64]
    assert isinstance(t, mctal.Tally)
    assert not t.result.flags.writeable
    e = t.axes["e"]
    assert e.upper is not None and e.upper.tolist() == [1.0, 10.0, 20.0]
    assert e.bounds is None and e.total
    assert t.numbers == [1, 2, 3, 4]
    assert np.all(t.result[:, 0, 0, 0, 0, 0, 2:, 0::2] == CELL)
    assert t.result.sum() == pytest.approx(4 * 4 * CELL)
    assert probe[14].numbers == [None, None, 1, None]
    assert probe[54].result[0, 0, 0, 0, 2, 0, 0, 0] == pytest.approx(CELL * 1e-110)
    assert probe[84].axes["e"].cumulative
    tfc = t.tfc
    assert tfc is not None and tfc.bin == [0, 0, 0, 0, 0, 0, 3, 2]
    assert tfc.histories.tolist() == [100]


def test_parts_and_operations(probe: mctal.MctalFile) -> None:
    whole = probe[24]
    part = probe.read(24, f=slice(2, 5))
    assert np.array_equal(part.result, whole.result[2:5])
    assert part.numbers == [8, 8, 8]
    t = probe[64]
    summed = t.integrate("e", "t")
    assert summed.shape == [4, 1, 1, 1, 1, 1, 1, 1]
    assert np.all(summed.result == CELL)
    assert np.array_equal(t.isel(e="total", t="total").result, summed.result)
    assert t.isel(f=0, e=slice(0, 2)).shape == [1, 1, 1, 1, 1, 1, 2, 3]
    with pytest.raises(KeyError, match="no axis"):
        t.isel(x=0)
    doubled = probe[4] + probe[4]
    assert np.all(doubled.result == 2 * CELL)
    assert np.all((probe[4] * 3).result == 3 * CELL)
    with pytest.raises(ValueError, match="different"):
        probe[4] + probe[54]


def test_tables(probe: mctal.MctalFile) -> None:
    pytest.importorskip("polars")
    df = probe[64].to_polars()
    assert df.height == 4 * 4 * 3
    assert df.columns == [
        "f", "number", "e", "e_low", "e_high", "e_is_total",
        "t", "t_low", "t_high", "t_is_total", "result", "rel_error",
    ]
    row = df.filter((df["f"] == 1) & (df["e"] == 2) & (df["t"] == 0))
    assert row["number"].item() == 2 and row["result"].item() == CELL
    assert row["e_low"].item() == 10.0 and row["e_high"].item() == 20.0
    assert df.filter(df["e_is_total"])["e_low"].unique().to_list() == [0.0]


def test_xarray(probe: mctal.MctalFile) -> None:
    pytest.importorskip("xarray")
    ds = probe[64].to_xarray()
    assert ds["result"].dims == ("f", "d", "u", "s", "m", "c", "e", "t")
    assert ds["number"].values.tolist() == [1, 2, 3, 4]
    assert ds["e_high"].values.tolist() == [1.0, 10.0, 20.0, 20.0]


def test_kcode_and_perturbations() -> None:
    f = mctal.open(FIXTURES / "kcode.m")
    k = f.kcode
    assert k is not None and (k.cycles, k.inactive) == (8, 3)
    assert k.values.shape == (8, 19)
    assert k["source_histories"][0] == 500
    assert len(k.columns) == 19
    p = f.read(4, perturbation=2)
    assert p.perturbation == 2 and p.shape == f[4].shape
    with pytest.raises(IndexError):
        f.read(4, perturbation=3)
    assert f.info(5).detector == "point" and f.info(15).detector == "ring"
    assert f.info(14).modifier == "*" and f.info(6).modifier == "+"


F4ENIX = Path.home() / "Coding/F4Enix/tests/resources/mctal"


@pytest.mark.skipif(not F4ENIX.is_dir(), reason="F4Enix's test files are not here")
def test_same_values_as_f4enix() -> None:
    """F4Enix's own MCTAL reader, where it reads right: it reads values
    below 1e-99 (`1.60682-113`), and their errors, as 0, and lays out TMESH
    meshes otherwise."""
    reader = pytest.importorskip("f4enix.output.mctal")
    logging.disable(logging.CRITICAL)
    for path in sorted(F4ENIX.iterdir()):
        if path.is_dir() or aci_tallies.detect(path) != "mctal":
            continue
        ours = mctal.open(path)
        for ref in reader.Mctal(path).tallies:
            t = ours[ref.tallyNumber]
            if t.mesh is not None:
                continue
            vals = ref.valsErrors[..., 0].reshape(tuple(t.shape))
            errs = ref.valsErrors[..., 1].reshape(tuple(t.shape))
            tiny = np.abs(t.result) < 1e-99
            assert np.array_equal(vals[~tiny], t.result[~tiny]), (path.name, ref.tallyNumber)
            assert np.array_equal(errs[~tiny], t.rel_error[~tiny]), (path.name, ref.tallyNumber)


def test_bins_from_the_input() -> None:
    f = mctal.open(FIXTURES / "levels.m", input=FIXTURES / "levels.mcnp")
    t = f[84]
    assert t.bins is not None
    assert [str(b) for b in t.bins] == ["(8<7[0 0 0]<3)", "(9<7[0 0 0]<3)", "(8<7[1 0 0]<3)", "T"]
    assert t.bins[2].levels == [[(8, None, None)], [(7, None, [1, 0, 0])], [(3, None, None)]]
    assert t.bins[3].total and t.bins[3].levels == []
    assert t.numbers == [8, 9, 8, None]
    pytest.importorskip("polars")
    df = t.to_polars()
    assert df["bin"].cast(str).to_list()[:2] == ["(8<7[0 0 0]<3)", "(9<7[0 0 0]<3)"]
    assert mctal.open(FIXTURES / "levels.m")[84].bins is None
