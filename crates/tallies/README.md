# aci-tallies

Readers for MCNP tally output files, however large: meshtal files (FMESH mesh
tallies), MCTAL files (`F` tallies) and D1SUNED's cell-under-voxel files
(`cuvmsh`, `meshinfo`).

Opening a file indexes it in one parallel pass; values are read when asked
for, a tally or part of one at a time, straight from disk into `ndarray`
arrays. Files much larger than memory can be read by parts.

```rust
use aci_tallies::{common::Input, mctal, meshtal};

let input = Input::read("model.i")?;
let mesh = meshtal::Reader::open_with_input("meshtal", &input)?;
let bin = mesh.read_bin(14, 0, 0)?;            // one energy and time bin
let cells = mctal::Reader::open_with_input("mctal", &input)?;
let part = cells.read_f(4, 0..10)?;            // ten cells of tally 4
```

The MCNP input is optional; it gives what the output files leave out (exact
mesh bounds, `TR=` transformations, ...).

Also available from Python as [`aci-tallies`](https://pypi.org/project/aci-tallies/)
on PyPI. See the [repository](https://github.com/AlvaroCubi/aci-tallies) for
details.

License: EUPL-1.2.
