# aci-tallies-geom

Geometry of MCNP-style structured meshes, rectangular and cylindrical: voxel
indexing, volumes, centres, point location, `TR=` transformations and VTK
export. Dependency-light, shared by
[`aci-tallies`](https://crates.io/crates/aci-tallies) and other mesh tools.

Features:

- `vtk` (default): VTK XML export (`aci_tallies_geom::vtk`).
- `mcnp-input`: exact mesh definitions from an MCNP input
  (`aci_tallies_geom::mcnp_input`).

License: EUPL-1.2.
