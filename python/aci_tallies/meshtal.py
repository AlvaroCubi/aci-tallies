"""MCNP meshtal files: the FMESH mesh tallies.

>>> from aci_tallies import meshtal
>>> f = meshtal.open("meshtal")
>>> f.ids                      # tally numbers
>>> f.tallies                  # what each is (no values read)
>>> t = f[14]                  # MeshTally: every energy and time bin
>>> b = f.read(14, energy=0, time="total")   # only those bins

A `MeshTally`'s `result` and `rel_error` have shape `(energy, time, i, j,
k)`, always: a mesh without energy or time bins has length 1 there, and an
energy or time axis with more than one bin has one more index at its end
for MCNP's Total bin (`tally.energy.total`, `tally.energy.total_index`).
`i, j, k` run along `tally.geometry.axis_names`: x, y, z, or r, z, theta.

`to_polars()` gives one row per (energy, time, voxel), with columns
`energy_low`/`energy_high`/`energy_is_total` and the same for time (only
for an axis of more than one index), the voxel centre (`x`, `y`, `z` or
`r`, `z`, `theta`), `result`, `rel_error`, and `volume` for `out=cf`.
"""

from ._native import MeshtalFile, MeshTally, TallyInfo
from ._native import open_meshtal as open

__all__ = ["MeshTally", "MeshtalFile", "TallyInfo", "open"]
