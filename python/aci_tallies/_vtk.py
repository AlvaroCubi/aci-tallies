"""Mesh tallies as PyVista grids and VTK files for ParaView.

The grid layout (rectilinear coordinates, or a cylinder's world-space
points, and which voxel each VTK cell shows) comes from the Rust
`aci-tallies-geom` crate, which also writes the `.vtr`/`.vts` files of
`to_vtk`, so an in-memory grid and a written file are always the same.
Cylinders are laid out on (r, theta, z) grid axes, which gives cells
positive volume, and theta bins coarser than `max_theta_step` revolutions
are split into equal sub-cells (each repeating its voxel's values), so a
full-revolution bin draws as a cylinder, not a flat sheet. `to_vtk` needs
no pyvista.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._extras import require

if TYPE_CHECKING:
    from ._native import Axis, MeshTally

Index = int | Literal["total"]
Indices = Index | Sequence[Index] | None

DEFAULT_MAX_THETA_STEP = 1.0 / 72.0
"""Largest theta step (revolutions) drawn for a cylindrical mesh: 5 degrees."""


def _resolve(value: Indices, axis: Axis, name: str) -> list[tuple[int, str]]:
    """`(index, label)` for each index selected on `axis`: every one (Total
    included) for None, else those given (an int or `"total"`, or a
    sequence of them). Labels are the index, or `"total"`."""
    if value is None:
        out = [(i, str(i)) for i in range(axis.n_bins)]
        if axis.total:
            out.append((axis.n_bins, "total"))
        return out
    values = [value] if isinstance(value, (int, str)) else list(value)
    if not values:
        raise ValueError(f"{name}: no index selected")
    out = []
    for v in values:
        if v == "total":
            if not axis.total:
                raise ValueError(f"{name}='total', but this tally has no {name} Total bin")
            out.append((axis.n_bins, "total"))
        elif isinstance(v, int) and 0 <= v < axis.n_bins:
            out.append((v, str(v)))
        else:
            raise ValueError(f"{name}={v!r} is not one of the {axis.n_bins} {name} bin(s) or 'total'")
    return out


def _arrays(
    tally: MeshTally, energy: Indices, time: Indices, fields: Sequence[str] | None
) -> list[tuple[str, np.ndarray]]:
    """`(name, (i, j, k) array)` for each field and selected (energy,
    time): `result_e2_t1`, `rel_error_etotal`, ... with a suffix only for
    an axis of more than one selected index; `volume` once."""
    energies = _resolve(energy, tally.energy, "energy")
    times = _resolve(time, tally.time, "time")
    available = ["result", "rel_error"] + (["volume"] if tally.volume is not None else [])
    out = []
    for name in fields if fields is not None else available:
        if name not in available:
            raise ValueError(f"no field {name!r}; this tally has {available}")
        if name == "volume":
            assert tally.volume is not None
            out.append((name, tally.volume))
            continue
        values = getattr(tally, name)
        for e, el in energies:
            for t, tl in times:
                label = name
                if len(energies) > 1:
                    label += f"_e{el}"
                if len(times) > 1:
                    label += f"_t{tl}"
                out.append((label, values[e, t]))
    return out


def to_pyvista(
    tally: MeshTally,
    energy: Indices = None,
    time: Indices = None,
    fields: Sequence[str] | None = None,
    max_theta_step: float | None = None,
) -> Any:
    """A PyVista grid with, as cell data, every field (`result`,
    `rel_error`, and `volume` for `out=cf`) for every selected energy and
    time index (by default all, Total bins included). A rectangular mesh is
    a `RectilinearGrid`; a cylindrical or transformed one a `StructuredGrid`.
    To draw less, select (`energy=`, `time=`, `fields=`) or narrow the tally
    first (`isel`, `integrate_energy`)."""
    pv = require("pyvista", "vtk")
    step = DEFAULT_MAX_THETA_STEP if max_theta_step is None else max_theta_step
    arrays = _arrays(tally, energy, time, fields)
    layout = tally.geometry._vtk_layout(step)
    grid: Any
    if layout["kind"] == "rectilinear":
        grid = pv.RectilinearGrid(layout["x"], layout["y"], layout["z"])
    else:
        grid = pv.StructuredGrid()
        grid.points = layout["points"]
        grid.dimensions = layout["dims"]
    # Spatial arrays flatten (C order) to the voxel index `cell_voxels`
    # gives for each cell.
    cells = layout["cell_voxels"]
    for name, values in arrays:
        grid.cell_data[name] = np.ravel(values)[cells]
    return grid


def to_vtk(
    tally: MeshTally,
    path: str | os.PathLike[str],
    energy: Indices = None,
    time: Indices = None,
    fields: Sequence[str] | None = None,
    max_theta_step: float | None = None,
    compress: bool = True,
) -> None:
    """Write the `to_pyvista` grid to `path` (`.vtr` for a rectangular mesh,
    `.vts` for a cylindrical or transformed one), zlib-compressed unless
    `compress=False`. Written by Rust: needs no pyvista."""
    step = DEFAULT_MAX_THETA_STEP if max_theta_step is None else max_theta_step
    arrays = _arrays(tally, energy, time, fields)
    tally.geometry._write_vtk(
        os.fspath(path),
        [name for name, _ in arrays],
        [np.ascontiguousarray(values, dtype=np.float64) for _, values in arrays],
        step,
        compress,
    )
