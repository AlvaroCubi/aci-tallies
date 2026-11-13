"""Tallies as xarray datasets (needs xarray)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._bins import axis_columns, centres, mctal_axis_columns
from ._extras import require

if TYPE_CHECKING:
    from ._native import CuvPairs, MctalTally, MeshTally


def meshtal_dataset(tally: MeshTally) -> Any:
    """`result` and `rel_error` on dimensions `(energy, time, <a>, <b>,
    <c>)`, the mesh axes named as `tally.geometry.axis_names` with their
    voxel centres as coordinates; energy and time indexed by bin, with
    `*_low`, `*_high` and `*_is_total` coordinates."""
    xr = require("xarray", "xarray")
    names = tally.geometry.axis_names
    dims = ("energy", "time", *names)
    coords: dict[str, Any] = {}
    for name, axis in (("energy", tally.energy), ("time", tally.time)):
        low, high, is_total = axis_columns(axis)
        coords[f"{name}_low"] = (name, low)
        coords[f"{name}_high"] = (name, high)
        coords[f"{name}_is_total"] = (name, is_total)
    for name, bounds in zip(names, tally.geometry.bounds):
        coords[name] = (name, centres(bounds))
        coords[f"{name}_low"] = (name, bounds[:-1])
        coords[f"{name}_high"] = (name, bounds[1:])
    data: dict[str, tuple[tuple[str, ...], Any]] = {
        "result": (dims, tally.result),
        "rel_error": (dims, tally.rel_error),
    }
    if tally.volume is not None:
        data["volume"] = (names, tally.volume)
    attrs = {
        "tally": tally.id,
        "particle": tally.particle,
        "comment": tally.comment or "",
        "out_format": tally.out_format,
        "geometry": tally.geometry.kind,
    }
    return xr.Dataset(data, coords=coords, attrs=attrs)


def mctal_dataset(tally: MctalTally) -> Any:
    """`result` and `rel_error` on dimensions `f d u s m c e t`, with
    `*_low`, `*_high` and `*_is_total` coordinates where known, and `number`
    (the cell or surface of each `f` bin, -1 where MCTAL gives none)."""
    xr = require("xarray", "xarray")
    dims = tuple(tally.dims)
    coords: dict[str, Any] = {}
    for name in dims:
        axis = tally.axes[name]
        bins = mctal_axis_columns(name, axis)
        if bins is not None:
            low, high, is_total = bins
            coords[f"{name}_low"] = (name, low)
            coords[f"{name}_high"] = (name, high)
            coords[f"{name}_is_total"] = (name, is_total)
        elif axis.total:
            flag = [False] * len(axis)
            flag[-1] = True
            coords[f"{name}_is_total"] = (name, flag)
    if tally.numbers is not None:
        coords["number"] = ("f", [-1 if n is None else n for n in tally.numbers])
    if tally.bins is not None:
        coords["bin"] = ("f", [str(b) for b in tally.bins])
    attrs = {
        "tally": tally.id,
        "particles": tally.particles,
        "comment": tally.comment or "",
        "detector": tally.detector or "",
        "modifier": tally.modifier,
    }
    data = {"result": (dims, tally.result), "rel_error": (dims, tally.rel_error)}
    return xr.Dataset(data, coords=coords, attrs=attrs)


def cuv_dataset(pairs: CuvPairs) -> Any:
    """Each pair's values on dimension `pair` (with `voxel`, `cell`,
    `fraction`, `volume` and the local voxel centre as coordinates), the
    spectra on `(pair, energy)` with `energy_low`/`energy_high`
    coordinates."""
    import numpy as np

    xr = require("xarray", "xarray")
    g = pairs.geometry
    ijk = np.unravel_index(pairs.voxel.astype(np.int64), g.shape)
    coords: dict[str, Any] = {
        "voxel": ("pair", pairs.voxel),
        "cell": ("pair", pairs.cell),
        "fraction": ("pair", pairs.fraction),
        "volume": ("pair", pairs.volume),
    }
    for name, bounds, i in zip(g.axis_names, g.bounds, ijk):
        coords[name] = ("pair", centres(bounds)[i])
    data: dict[str, tuple[tuple[str, ...], Any]] = {}
    for name in ("material", "density", "total", "total_error"):
        value = getattr(pairs, name)
        if value is not None:
            data[name] = (("pair",), value)
    if pairs.flux is not None and pairs.energy is not None and pairs.energy.bounds is not None:
        bounds = pairs.energy.bounds
        coords["energy_low"] = ("energy", bounds[:-1])
        coords["energy_high"] = ("energy", bounds[1:])
        data["flux"] = (("pair", "energy"), pairs.flux)
        data["rel_error"] = (("pair", "energy"), pairs.rel_error)
    attrs = {
        "tally": pairs.id,
        "particle": pairs.particle or "",
        "comment": pairs.comment or "",
        "voxels": list(pairs.voxels),
        "geometry": g.kind,
    }
    return xr.Dataset(data, coords=coords, attrs=attrs)
