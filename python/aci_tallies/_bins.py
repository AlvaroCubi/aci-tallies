"""Bins as table columns or coordinates."""

from __future__ import annotations

import numpy as np

from ._native import Axis


def centres(bounds: np.ndarray) -> np.ndarray:
    return (bounds[:-1] + bounds[1:]) / 2.0


def axis_columns(axis: Axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per index of `axis`: lower and upper bound, and whether it is the
    Total bin (which spans the whole axis)."""
    n = len(axis)
    low = np.full(n, np.nan)
    high = np.full(n, np.nan)
    bounds = axis.bounds
    if bounds is not None:
        low[: axis.n_bins] = bounds[:-1]
        high[: axis.n_bins] = bounds[1:]
        if axis.total:
            low[-1], high[-1] = bounds[0], bounds[-1]
    is_total = np.zeros(n, dtype=bool)
    if axis.total:
        is_total[-1] = True
    return low, high, is_total


# Where an MCTAL axis's first bin starts, when the file gives only upper
# bounds: 0 MeV, a cosine of -1, the beginning of time.
_FIRST_LOWER = {"e": 0.0, "c": -1.0, "t": -np.inf}


def mctal_axis_columns(name: str, axis: Axis) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """`axis_columns` for an MCTAL axis, which usually has only upper
    bounds; None when it has no bounds at all."""
    if axis.bounds is not None:
        return axis_columns(axis)
    upper = axis.upper
    if upper is None:
        return None
    n = len(axis)
    low = np.full(n, np.nan)
    high = np.full(n, np.nan)
    high[: axis.n_bins] = upper
    low[1 : axis.n_bins] = upper[:-1]
    if axis.cumulative:
        low[: axis.n_bins] = _FIRST_LOWER.get(name, np.nan)
    elif axis.n_bins:
        low[0] = _FIRST_LOWER.get(name, np.nan)
    if axis.total:
        low[-1], high[-1] = _FIRST_LOWER.get(name, np.nan), upper[-1]
    is_total = np.zeros(n, dtype=bool)
    if axis.total:
        is_total[-1] = True
    return low, high, is_total
