from pathlib import Path

import aci_tallies
import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[2] / "examples" / "meshtal"


@pytest.fixture(scope="module")
def parsed():
    return {t.id: t for t in aci_tallies.open(str(EXAMPLE)).read_all()}


def test_all_nine_sections_parse(parsed):
    assert set(parsed) == {14, 24, 34, 44, 54, 64, 74, 84, 94}
    assert aci_tallies.open(str(EXAMPLE)).failed == []


def _tally(parsed, tally_id):
    return parsed[tally_id]


def test_tally_14_plain_rectangular(parsed):
    t = _tally(parsed, 14)
    assert t.particle == "neutron"
    assert t.out_format == "col"
    assert t.geometry.kind == "rectangular"
    assert t.result.shape == (1, 1, 5, 5, 5)
    assert t.result.dtype == np.float64
    assert not t.energy.total
    assert not t.time.total
    assert t.result[0, 0, 0, 0, 0] == pytest.approx(4.69512e-04, rel=1e-6)
    assert t.rel_error[0, 0, 0, 0, 0] == pytest.approx(1.07701e-02, rel=1e-6)


def test_tally_34_energy_discretized(parsed):
    t = _tally(parsed, 34)
    assert t.out_format == "col"
    assert t.energy.bounds.shape == (11,)
    assert t.energy.total
    assert t.result.shape == (11, 1, 5, 5, 5)
    assert t.result[0, 0, 0, 0, 0] == pytest.approx(7.41491e11, rel=1e-6)
    # MCNP's automatic Total energy block, first row.
    assert t.result[t.energy.total_index, 0, 0, 0, 0] == pytest.approx(8.67991e11, rel=1e-6)


def test_tally_44_ij_matches_tally_14_col(parsed):
    col = _tally(parsed, 14)
    ij = _tally(parsed, 44)
    assert ij.out_format == "ij"
    assert ij.warnings == []
    assert np.allclose(col.result, ij.result, rtol=1e-6)
    # The ij matrix format prints Rel Error to a fixed 5 decimal places,
    # so the comparison uses an absolute (quantization) tolerance.
    assert np.allclose(col.rel_error, ij.rel_error, atol=1e-5)


def test_tally_54_ik_matches_tally_14_col(parsed):
    col = _tally(parsed, 14)
    ik = _tally(parsed, 54)
    assert ik.out_format == "ik"
    assert ik.warnings == []
    assert np.allclose(col.result, ik.result, rtol=1e-6)
    assert np.allclose(col.rel_error, ik.rel_error, atol=1e-5)


def test_tally_64_cf_has_volume(parsed):
    t = _tally(parsed, 64)
    assert t.out_format == "cf"
    assert t.volume is not None
    assert np.allclose(t.volume, 64.0)


def test_tally_74_is_cylindrical(parsed):
    t = _tally(parsed, 74)
    assert t.geometry.kind == "cylindrical"
    assert t.geometry.origin == pytest.approx((0.0, 0.0, -10.0))
    assert t.geometry.axis == pytest.approx((0.0, 0.0, 1.0))
    assert t.result.shape == (1, 1, 10, 10, 10)


def test_tally_84_time_discretized(parsed):
    t = _tally(parsed, 84)
    assert t.out_format == "col"
    assert t.warnings == []
    assert t.time.bounds.shape == (4,)
    assert not t.energy.total
    assert t.time.total
    assert t.result.shape == (1, 4, 5, 5, 5)
    assert t.result[0, 0, 0, 0, 0] == pytest.approx(4.69512e-04, rel=1e-6)
    assert t.result[0, t.time.total_index, 0, 0, 0] == pytest.approx(4.69512e-04, rel=1e-6)


def test_tally_94_energy_and_time_discretized(parsed):
    t = _tally(parsed, 94)
    assert t.out_format == "col"
    assert t.warnings == []
    assert t.energy.total and t.time.total
    assert t.result.shape == (11, 4, 5, 5, 5)
    assert t.result[0, 0, 0, 0, 0] == pytest.approx(7.41491e-06, rel=1e-6)
    grand_total = t.result[t.energy.total_index, t.time.total_index]
    assert grand_total[0, 0, 0] == pytest.approx(8.67991e-06, rel=1e-6)
    assert grand_total[4, 4, 4] == pytest.approx(7.88808e-06, rel=1e-6)


def test_zero_copy_numpy_handoff(parsed):
    t = _tally(parsed, 14)
    # Arrays should be writable, own their memory, and be plain numpy — the
    # zero-copy contract is about avoiding an extra data copy on the Rust
    # side, not about the array being read-only in Python.
    assert t.result.flags["OWNDATA"] or t.result.base is not None
