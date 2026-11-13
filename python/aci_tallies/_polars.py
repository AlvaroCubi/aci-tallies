"""Tallies as Polars tables (needs polars)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ._bins import axis_columns, centres, mctal_axis_columns
from ._extras import require

if TYPE_CHECKING:
    from ._native import CuvPairs, MctalTally, MeshTally


def _spread(a: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Each value of `a` `inner` times in a row, all that `outer` times."""
    return np.tile(np.repeat(a, inner), outer)


def meshtal_frame(tally: MeshTally) -> Any:
    """One row per (energy, time, voxel), in the order of `tally.result`.

    Each column is built at its final size directly (by repeating the small
    per-axis arrays), with no full-size index arrays alongside, and columns
    that would hold one value throughout are left out: an energy or time
    axis with one index, a `*_is_total` column of an axis with no Total.
    """
    pl = require("polars", "polars")
    ne, nt, ni, nj, nk = tally.shape
    voxels = ni * nj * nk
    columns: dict[str, np.ndarray] = {}
    for name, axis, outer, inner in (
        ("energy", tally.energy, 1, nt * voxels),
        ("time", tally.time, ne, voxels),
    ):
        if len(axis) > 1:
            low, high, is_total = axis_columns(axis)
            columns[f"{name}_low"] = _spread(low, inner, outer)
            columns[f"{name}_high"] = _spread(high, inner, outer)
            if axis.total:
                columns[f"{name}_is_total"] = _spread(is_total, inner, outer)
    names = tally.geometry.axis_names
    ca, cb, cc = (centres(b) for b in tally.geometry.bounds)
    columns[names[0]] = np.tile(np.repeat(ca, nj * nk), ne * nt)
    columns[names[1]] = np.tile(np.repeat(cb, nk), ne * nt * ni)
    columns[names[2]] = np.tile(cc, ne * nt * ni * nj)
    columns["result"] = tally.result.reshape(-1)
    columns["rel_error"] = tally.rel_error.reshape(-1)
    if tally.volume is not None:
        columns["volume"] = np.tile(tally.volume.reshape(-1), ne * nt)
    return pl.DataFrame(columns)


def meshtal_parquet(tally: MeshTally, path: str | os.PathLike[str]) -> None:
    meshtal_frame(tally).write_parquet(os.fspath(path))


def mctal_frame(tally: MctalTally) -> Any:
    """One row per value, in the order of `tally.result`, with columns only
    for axes of more than one index: the index (`f`, `d`, ...), bounds where
    known (`e_low`, `e_high`), `*_is_total` for an axis with a Total bin,
    and `number`, the cell or surface of each `f` bin where MCTAL gives it.
    """
    pl = require("polars", "polars")
    shape = tally.shape
    columns: dict[str, Any] = {}
    for k, name in enumerate(tally.dims):
        n = shape[k]
        if n <= 1:
            continue
        inner = int(np.prod(shape[k + 1 :]))
        outer = int(np.prod(shape[:k]))
        axis = tally.axes[name]
        columns[name] = _spread(np.arange(n, dtype=np.uint32), inner, outer)
        bins = mctal_axis_columns(name, axis)
        if bins is not None:
            low, high, _ = bins
            columns[f"{name}_low"] = _spread(low, inner, outer)
            columns[f"{name}_high"] = _spread(high, inner, outer)
        if axis.total:
            flag = np.zeros(n, dtype=bool)
            flag[-1] = True
            columns[f"{name}_is_total"] = _spread(flag, inner, outer)
        if name == "f" and tally.numbers is not None:
            numbers = pl.Series("number", tally.numbers, dtype=pl.Int64)
            columns["number"] = numbers.gather(columns["f"])
        if name == "f" and tally.bins is not None:
            # Categorical: one index per row, each label stored once.
            labels = pl.Series(
                "bin", [str(b) for b in tally.bins], dtype=pl.Categorical
            )
            columns["bin"] = labels.gather(columns["f"])
    columns["result"] = tally.result.reshape(-1)
    columns["rel_error"] = tally.rel_error.reshape(-1)
    return pl.DataFrame(columns)


def mctal_parquet(tally: MctalTally, path: str | os.PathLike[str]) -> None:
    mctal_frame(tally).write_parquet(os.fspath(path))


def _voxel_centres(pairs: CuvPairs) -> dict[str, np.ndarray]:
    """Each pair's voxel centre, local to the mesh, by axis name."""
    g = pairs.geometry
    ijk = np.unravel_index(pairs.voxel.astype(np.int64), g.shape)
    return {name: centres(b)[i] for name, b, i in zip(g.axis_names, g.bounds, ijk)}


def cuv_frame(pairs: CuvPairs, spectra: bool = False) -> Any:
    """One row per pair; with `spectra`, one per (pair, energy bin)."""
    pl = require("polars", "polars")
    if spectra:
        flux, rel_error, energy = pairs.flux, pairs.rel_error, pairs.energy
        if flux is None or rel_error is None or energy is None or energy.bounds is None:
            raise ValueError("spectra=True needs the cuvmsh")
        n_pairs, n_energy = flux.shape
        bounds = energy.bounds
        return pl.DataFrame(
            {
                "voxel": np.repeat(pairs.voxel, n_energy),
                "cell": np.repeat(pairs.cell, n_energy),
                "energy_low": np.tile(bounds[:-1], n_pairs),
                "energy_high": np.tile(bounds[1:], n_pairs),
                "flux": flux.reshape(-1),
                "rel_error": rel_error.reshape(-1),
            }
        )
    columns: dict[str, np.ndarray] = {"voxel": pairs.voxel, **_voxel_centres(pairs)}
    columns["cell"] = pairs.cell
    columns["fraction"] = pairs.fraction
    columns["volume"] = pairs.volume
    for name in ("material", "density", "total", "total_error"):
        value = getattr(pairs, name)
        if value is not None:
            columns[name] = value
    return pl.DataFrame(columns)


def cuv_parquet(
    pairs: CuvPairs, path: str | os.PathLike[str], spectra: bool = False
) -> None:
    cuv_frame(pairs, spectra).write_parquet(os.fspath(path))
