# D1SUNED cell-under-voxel fixtures

`input` is a synthetic test problem (an iron block with a water channel in
a void sphere, a 14 MeV point source, 2e4 histories) run with D1SUNED
4.1.3.2 (MCNP 6.2):

```
d1suned4132 i=input tasks 8
```

`cuvmsh` and `meshinfo` hold three CuV meshes (`out=cuv`): 4, a cylinder with
20 energy bins; 14, a box with 4 bins and an FC comment, several cells per
voxel, the void graveyard among them; 44, a box under `TR=1` without EMESH
(one bin: cell lines without totals). `meshtal` holds FMESH24, mesh 14
without `out=cuv`, scored with the same histories. Only `input`, `cuvmsh`,
`meshinfo` and `meshtal` are kept.

The cylinder has 20 energy bins and comes first because D1SUNED 4.1.3.2
crashed ("output statement overflows record") writing its `origin at` line
with 2 bins.

`legacy_3.1.4_cyl/meshinfo` is `meshinfo` as D1SUNED 3.x writes it, made by
`make_legacy.py`: 3.1.4 writes theta in radians and cylinder volumes 2π too
large, and otherwise the same file for the same run.
