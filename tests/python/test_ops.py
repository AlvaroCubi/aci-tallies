from pathlib import Path

import aci_tallies
import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[2] / "examples" / "meshtal"


@pytest.fixture(scope="module")
def parsed():
    return {t.id: t for t in aci_tallies.open(str(EXAMPLE)).read_all()}


def _tally(parsed, tally_id):
    return parsed[tally_id]


def test_add_matches_result_and_reduces_relative_error(parsed):
    t = _tally(parsed, 14)
    summed = t + t
    assert np.allclose(summed.result, t.result * 2)
    # Summing two independent measurements of the same value reduces
    # relative error by quadrature (1/sqrt(2)), unlike scaling.
    assert np.allclose(summed.rel_error, t.rel_error / np.sqrt(2))


def test_sub_self_is_zero(parsed):
    t = _tally(parsed, 14)
    diff = t - t
    assert np.allclose(diff.result, 0.0)


def test_scale_preserves_relative_error(parsed):
    t = _tally(parsed, 14)
    scaled = t * 3.0
    assert np.allclose(scaled.result, t.result * 3.0)
    assert np.allclose(scaled.rel_error, t.rel_error)
    tripled = 3.0 * t
    assert np.allclose(tripled.result, scaled.result)


def test_div_by_self_is_one_where_nonzero(parsed):
    t = _tally(parsed, 14)
    ratio = t / t
    nonzero = t.result != 0
    assert np.allclose(ratio.result[nonzero], 1.0)


def test_arithmetic_rejects_incompatible_shapes(parsed):
    t14 = _tally(parsed, 14)
    t74 = _tally(parsed, 74)
    with pytest.raises(ValueError, match="shape"):
        t14 + t74


def test_integrate_energy_falls_back_to_quadrature_without_total(parsed):
    t = _tally(parsed, 74)  # cylindrical, single implicit energy group: no Total
    assert not t.energy.total
    integrated = t.integrate_energy()
    assert integrated.result.shape == (1, *t.result.shape[1:])
    assert np.allclose(integrated.result[0], t.result.sum(axis=0))
    assert integrated.energy.bounds[0] == t.energy.bounds[0]
    assert integrated.energy.bounds[-1] == t.energy.bounds[-1]


def test_integrate_energy_prefers_native_total_over_quadrature(parsed):
    # Tally 34 has a real MCNP-computed Total energy block (10 real groups),
    # whose error propagation is not a plain quadrature sum of the real
    # groups' abs errors — integrate_energy must return that native slice
    # directly rather than re-deriving a different value.
    t = _tally(parsed, 34)
    assert t.energy.total
    integrated = t.integrate_energy()
    assert integrated.result.shape == (1, *t.result.shape[1:])
    assert not integrated.energy.total
    assert np.allclose(integrated.result[0], t.result[t.energy.total_index])
    assert np.allclose(integrated.rel_error[0], t.rel_error[t.energy.total_index])
    assert integrated.energy.bounds[0] == t.energy.bounds[0]
    assert integrated.energy.bounds[-1] == t.energy.bounds[-1]


def test_integrate_time_prefers_native_total_over_quadrature(parsed):
    t = _tally(parsed, 94)  # energy- and time-discretized, has a time Total
    assert t.time.total
    integrated = t.integrate_time()
    assert integrated.result.shape[1] == 1
    assert not integrated.time.total
    assert np.allclose(
        integrated.result[:, 0], t.result[:, t.time.total_index]
    )


def test_integrate_space_matches_manual_volume_weighting(parsed):
    t = _tally(parsed, 14)
    dx = np.diff(t.geometry.bounds[0])
    dy = np.diff(t.geometry.bounds[1])
    dz = np.diff(t.geometry.bounds[2])
    voxel_vol = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    expected_total = (t.result[0] * voxel_vol).sum()

    integrated = t.integrate_space()
    assert integrated.result.shape == (1, 1, 1, 1, 1)
    assert integrated.result[0, 0, 0, 0, 0] == pytest.approx(expected_total)


def test_integrate_space_uses_explicit_volume_when_present(parsed):
    t = _tally(parsed, 64)  # out=cf, has an explicit `volume` array
    expected_total = (t.result * t.volume).sum()
    integrated = t.integrate_space()
    assert integrated.result.sum() == pytest.approx(expected_total)


def test_isel_slices_result_and_bounds(parsed):
    t = _tally(parsed, 14)
    sliced = t.isel(i=slice(0, 2))
    assert sliced.result.shape == (t.result.shape[0], t.result.shape[1], 2, *t.result.shape[3:])
    assert np.allclose(sliced.result, t.result[:, :, 0:2, :, :])
    assert np.allclose(sliced.geometry.bounds[0], t.geometry.bounds[0][0:3])


def test_isel_slices_time_axis(parsed):
    t = _tally(parsed, 84)  # time-discretized: 3 real time bins + a Total
    sliced = t.isel(time=slice(1, 2))
    assert sliced.result.shape == (t.result.shape[0], 1, *t.result.shape[2:])
    assert not sliced.time.total
    assert np.allclose(sliced.result[:, 0], t.result[:, 1])
    assert np.allclose(sliced.time.bounds, t.time.bounds[1:3])


def test_isel_omitting_axis_keeps_total_slice(parsed):
    t = _tally(parsed, 34)
    assert t.energy.total
    kept_whole = t.isel(i=slice(0, 2))  # energy omitted entirely: kept whole, Total included
    assert kept_whole.energy.total
    assert kept_whole.result.shape[0] == t.result.shape[0]


def test_isel_rejects_empty_and_out_of_bounds_ranges(parsed):
    t = _tally(parsed, 14)
    with pytest.raises(ValueError, match="empty"):
        t.isel(i=slice(2, 2))
    # Slices clamp, as in numpy; an index out of range raises.
    assert t.isel(i=slice(0, 100)).shape == t.shape
    with pytest.raises(IndexError, match="out of range"):
        t.isel(i=100)
