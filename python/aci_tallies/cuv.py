"""D1SUNED cell-under-voxel (CuV) meshes: `cuvmsh` and `meshinfo` files.

An FMESH with `out=cuv` is not written to the meshtal file. D1SUNED writes
instead, for each voxel, the cells in it: the `meshinfo` gives each
(voxel, cell) pair's material, density and share of the voxel's volume, the
`cuvmsh` the same pairs with their flux spectra. Every CuV mesh of a run is
in the same two files.

>>> from aci_tallies import cuv
>>> f = cuv.open("cuvmsh", input="model.i")   # or a meshinfo; input optional
>>> f.ids                                     # mesh numbers
>>> p = f[14]                                 # CuvPairs: every pair of mesh 14
>>> p.cell, p.fraction, p.flux.shape          # flux: (pair, energy)
>>> part = f.read(14, voxels=slice(0, 1000))  # only those voxels' part of the file
>>> both = cuv.open_mesh("cuvmsh", "meshinfo")  # material, density and spectra
>>> f.to_mesh(14).to_vtk("mesh14")            # voxel fluxes, as a MeshTally

`CuvPairs` arrays have one element per pair, voxel by voxel in the mesh's
flat order (`voxel` is the flat index, as `Geometry.locate` numbers them),
except `voxel_volume`, one per voxel read. With `input`, the geometry and
energy bounds come from the FMESH card (exact, with `TR=`), not from the
headers, which round them and leave out `TR=`.

`to_polars()` gives one row per pair: `voxel`, the voxel centre (`x`, `y`,
`z` or `r`, `z`, `theta`, local to the mesh), `cell`, `fraction`, `volume`,
and what the files have of `material`, `density`, `total`, `total_error`.
`to_polars(spectra=True)` gives one row per (pair, energy bin) instead:
`voxel`, `cell`, `energy_low`, `energy_high`, `flux`, `rel_error`.
`to_xarray()` puts everything on dimensions `(pair, energy)`.
"""

from __future__ import annotations

import os

from ._native import (
    CuvFile,
    CuvInfo,
    CuvPairs,
    Input,
    detect,
    open_cuv_mesh,
    open_cuvmsh,
    open_meshinfo,
)

__all__ = ["CuvFile", "CuvInfo", "CuvPairs", "open", "open_mesh"]


def open(
    path: str | os.PathLike[str], input: str | os.PathLike[str] | Input | None = None
) -> CuvFile:
    """Open a cuvmsh or a meshinfo (told from its contents)."""
    kind = detect(path)
    if kind == "cuvmsh":
        return open_cuvmsh(path, input=input)
    if kind == "meshinfo":
        return open_meshinfo(path, input=input)
    raise ValueError(f"{os.fspath(path)!r} is neither a cuvmsh nor a meshinfo")


def open_mesh(
    cuvmsh: str | os.PathLike[str],
    meshinfo: str | os.PathLike[str],
    input: str | os.PathLike[str] | Input | None = None,
) -> CuvFile:
    """Open a run's cuvmsh and meshinfo together: reads give each pair's
    material, density and spectrum, once both files are found to list the
    same pairs."""
    return open_cuv_mesh(cuvmsh, meshinfo, input=input)
