"""Read MCNP tally files, however large, into NumPy arrays.

The work is done in Rust: opening a file indexes it in one parallel pass,
and values are read when asked for, a tally or a bin at a time, so files
much larger than memory can be used. Values are read-only NumPy views of
what Rust read; operations (`+`, `isel`, `integrate_energy`, ...) make new
tallies, with errors propagated.

>>> import aci_tallies
>>> f = aci_tallies.open("meshtal", input="model.i")  # input: optional
>>> tally = f[14]                     # read one tally
>>> tally.result.shape                # (energy, time, i, j, k)
>>> f.read(14, energy="total")        # read only MCNP's Total energy bin

Formats: `aci_tallies.meshtal` (FMESH mesh tallies), `aci_tallies.mctal`
(F tallies) and `aci_tallies.cuv` (D1SUNED cell-under-voxel meshes:
`cuvmsh` and `meshinfo`).
"""

from __future__ import annotations

import os

from . import cuv, mctal, meshtal
from ._native import Axis, Geometry, Input, ReadWarning, detect

__all__ = [
    "Axis",
    "Geometry",
    "Input",
    "ReadWarning",
    "cuv",
    "detect",
    "mctal",
    "meshtal",
    "open",
]


def open(
    path: str | os.PathLike[str], input: str | os.PathLike[str] | Input | None = None
) -> meshtal.MeshtalFile | mctal.MctalFile | cuv.CuvFile:
    """Open the tally file at `path`, whatever its format (told from its
    contents). `input` is the MCNP input it came from (a path or an
    `Input`), for what the file itself doesn't say: see each format's
    `open`."""
    kind = detect(path)
    if kind == "meshtal":
        return meshtal.open(path, input=input)
    if kind == "mctal":
        return mctal.open(path, input=input)
    if kind in ("cuvmsh", "meshinfo"):
        return cuv.open(path, input=input)
    raise ValueError(
        f"{os.fspath(path)!r} is not a file aci_tallies reads (meshtal, MCTAL, cuvmsh, meshinfo)"
    )
