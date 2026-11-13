from pathlib import Path

import aci_tallies
import numpy as np
import polars as pl
import pytest

EXAMPLE = Path(__file__).parents[2] / "examples" / "meshtal"


@pytest.fixture(scope="module")
def parsed():
    return {t.id: t for t in aci_tallies.open(str(EXAMPLE)).read_all()}


def _tally(parsed, tally_id):
    return parsed[tally_id]


def test_to_polars_rectangular_shape_and_values(parsed):
    t = _tally(parsed, 14)
    df = t.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1 * 5 * 5 * 5
    # Neither axis has more than one bin, so their columns would hold one
    # value throughout: they're left out, as is a constant tally id.
    assert df.columns == ["x", "y", "z", "result", "rel_error"]
    assert "energy_is_total" not in df.columns
    assert "time_low" not in df.columns
    assert "time_high" not in df.columns
    assert "time_is_total" not in df.columns
    # The voxel at (-8, -8, -8) should carry the same value we've verified
    # elsewhere against the raw file.
    row = df.filter(
        (pl.col("x") == -8.0) & (pl.col("y") == -8.0) & (pl.col("z") == -8.0)
    )
    assert row.height == 1
    assert row["result"][0] == pytest.approx(4.69512e-04, rel=1e-6)


def test_to_polars_cylindrical_uses_r_z_theta(parsed):
    t = _tally(parsed, 74)
    df = t.to_polars()
    assert {"r", "z", "theta"} <= set(df.columns)
    assert "x" not in df.columns


def test_to_polars_cf_includes_volume(parsed):
    t = _tally(parsed, 64)
    df = t.to_polars()
    assert "volume" in df.columns
    assert (df["volume"] == 64.0).all()


def test_to_polars_energy_bounds_present(parsed):
    t = _tally(parsed, 34)  # energy-discretized, has a Total, no TMESH
    df = t.to_polars()
    assert t.energy.total
    assert "energy_is_total" in df.columns
    assert "time_low" not in df.columns  # no time discretization: left out
    assert df.height == 11 * 5 * 5 * 5
    real = df.filter(~pl.col("energy_is_total"))
    assert real.height == 10 * 5 * 5 * 5
    assert real["energy_low"].n_unique() == 10


def test_to_polars_time_only_includes_time_columns_without_energy_total(parsed):
    t = _tally(parsed, 84)  # time-discretized, no EMESH: no energy Total
    df = t.to_polars()
    assert t.time.total
    assert not t.energy.total
    assert "time_low" in df.columns and "time_is_total" in df.columns
    assert "energy_is_total" not in df.columns


def test_to_polars_time_bounds_and_total_rows(parsed):
    t = _tally(parsed, 94)  # energy- and time-discretized, both have a Total
    df = t.to_polars()
    assert t.energy.total and t.time.total
    assert {"energy_is_total", "time_low", "time_high", "time_is_total"} <= set(
        df.columns
    )
    assert df.height == 11 * 4 * 5 * 5 * 5
    grand_total = df.filter(pl.col("energy_is_total") & pl.col("time_is_total"))
    assert grand_total.height == 5 * 5 * 5
    row = grand_total.filter(
        (pl.col("x") == -8.0) & (pl.col("y") == -8.0) & (pl.col("z") == -8.0)
    )
    assert row.height == 1
    assert row["result"][0] == pytest.approx(8.67991e-06, rel=1e-6)
    # The grand-total row's bounds span the full real axis, not one bin.
    assert row["energy_low"][0] == pytest.approx(t.energy.bounds[0])
    assert row["energy_high"][0] == pytest.approx(t.energy.bounds[-1])
    assert row["time_low"][0] == pytest.approx(t.time.bounds[0])
    assert row["time_high"][0] == pytest.approx(t.time.bounds[-1])


def test_to_parquet_round_trips(parsed, tmp_path):
    t = _tally(parsed, 14)
    out = tmp_path / "tally14.parquet"
    t.to_parquet(out)
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.height == 1 * 5 * 5 * 5
    assert np.allclose(sorted(df["result"].to_numpy()), sorted(t.result.ravel()))
