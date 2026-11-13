# aci-tallies

Read MCNP tally files, however large, into NumPy arrays: meshtal files (FMESH
mesh tallies), MCTAL files (`F` tallies) and D1SUNED's cell-under-voxel
files (`cuvmsh`, `meshinfo`), in Rust, from Python or Rust.

Install as `aci-tallies`, import as `aci_tallies`.

```python
import aci_tallies

f = aci_tallies.open("meshtal", input="model.i")   # or an MCTAL file; input optional
f.ids                          # tally numbers; f.tallies describes each, no values read
t = f[14]                      # read one tally
t.result, t.rel_error          # read-only NumPy views, no copy
f.read(14, energy="total")     # read only MCNP's Total energy bin of it
```

Opening a file indexes it in one parallel pass; values are read when asked
for, a tally or part of one at a time, straight from disk into the arrays
returned. A 50 GB file opens in well under a second, and only what you read
is ever in memory.

- `crates/tallies` (`aci-tallies`): the readers. `common` (file sources, the
  parallel scan, Fortran number parsing, bins, error propagation),
  `meshtal`, `mctal`, `cuv`.
- `crates/geom` (`aci-tallies-geom`): mesh geometry (voxels, volumes,
  centres, point location, VTK, `TR=`), dependency-light, shared with other
  mesh tools.
- `crates/py` and `python/aci_tallies`: the Python package. It needs only
  NumPy; `to_polars`, `to_xarray` and `to_pyvista` need their extras
  (`pip install 'aci-tallies[polars,xarray,vtk]'`, or `[all]`).

The MCNP input (`input=`, a path or an `aci_tallies.Input` to reuse) is
optional and read with [migjorn](https://crates.io/crates/migjorn): it gives
what the files leave out (below).

## meshtal files

`aci_tallies.meshtal.open(path, input=None)` → `MeshtalFile`, a mapping from
tally number to `MeshTally`.

- Every `out=` layout (`col`, `cf`, `ij`, `ik`, `jk`), rectangular and
  cylindrical meshes, energy and time bins and their Total bins.
- `result`/`rel_error` always have five axes, `(energy, time, i, j, k)`; a
  mesh without `EMESH`/`TMESH` has length 1 there. When an axis has more
  than one bin, MCNP's Total bin is one more index at its end
  (`t.energy.total`, `t.energy.total_index`), never dropped.
  `t.volume` (`out=cf`) is `(i, j, k)`.
- `t.geometry` is the mesh: `kind`, `axis_names` (`x y z` or `r z theta`),
  `bounds`, `origin`/`axis`/`vec`, `transform` (4x4, `TR=`), `volumes()`,
  `centers()`, `locate(points)`.
- `f.read(id, energy=, time=)` reads only those bins (an index or
  `"total"`); in the columnar layouts a bin's rows are consecutive, in the
  matrix ones a run of blocks, so nothing else is read.
- Operations run in Rust, in parallel, without copying their inputs: `+ -
  * /` (errors propagated as for independent quantities; `t + t` is not
  `2 * t`), `isel(energy=, time=, i=, j=, k=)` with ints, slices or
  `"total"`, `integrate_energy()`/`integrate_time()` (MCNP's Total bin when
  printed: its error comes from the histories, which summing can't
  reproduce), `integrate_space()`.
- `to_polars()`/`to_parquet()` (one row per value; an axis with one bin gets
  no columns), `to_xarray()`, `to_pyvista()` (every field and bin attached),
  `to_vtk(path)` (written by Rust, no PyVista needed).

In Rust, `aci_tallies_geom::mcnp_input` reads the `FMESH` cards of an
input exactly (`read_fmeshes`) and writes them (`FmeshCard::to_input`, with
the `TRn` card of a `TR=`): a written card reads back the same, bounds to
the bit, and MCNP scores it as the card it came from.

### What a meshtal file can't tell you

Found by running MCNP 6.2 and 6.3 on purpose-built inputs
(`tests/fixtures/pencil_beam*.mcnp`: one ray through void, so every voxel
scores exactly chord / volume). The input fixes the first two and the third:

- **Bounds are rounded** to 2 decimals (theta 3), origin and axes to 3
  significant digits: geometry rebuilt from them is off by up to ~0.2%
  (~2% for an unround cylinder origin), and bins finer than 0.01 cm
  collapse (with a warning). Values are unaffected.
- **`TR=` leaves no trace.** `tally.geometry_from_input` says which you got.
- **`*FMESH`** looks like `FMESH`; only the input says (`energy_weighted`).
- Bounds around ±1e5–1e6 cm overflow MCNP's fixed-width fields: such a
  section is listed in `f.failed`.

## MCTAL files

`aci_tallies.mctal.open(path, input=None)` → `MctalFile`, a mapping from
tally number to `Tally`.

- `result`/`rel_error` always have MCNP's eight axes, in the order the file
  lists values: `f d u s m c e t` (cells or surfaces, direct, user,
  segments, multipliers, cosines, energies, times). `t.axes["e"]` has an
  axis's bins: MCTAL gives upper bounds only (`Axis.upper`), and flags a
  Total bin (`total`, last index) or cumulative bins (`cumulative`).
- `f.read(id, f=slice(...))` reads only those cells: `f` varies slowest, so
  their values are a run of lines. `f.read(id, perturbation=n)` reads a
  `PERT` card's values; `f.kcode` has a KCODE run's estimates per cycle
  (a named column each); `t.tfc` the fluctuation chart.
- Detectors (point, ring, radiography grids), `*F`/`+F`, user bins, legacy
  TMESH mesh tallies (their cells are the `f` bins, first axis fastest).
- `isel(f=, e=, ...)`, `integrate("e", "t")`, `+ - * /`, `to_polars()`,
  `to_xarray()`.

### What an MCTAL file can't tell you

Found by running MCNP 6.2 and 6.3 on `tests/fixtures/mctal/*.mcnp`:

- **The `f` line names a bin only by its first level, and only when that
  is one cell or surface**: `(8 < 7[2 0 0] < 3)` is listed as `8`, a group
  `(1 2)` as `0`, and the T bin as `0` unless every bin starts with the same
  cell. `t.numbers` has these.
- With the input, **`t.bins` has what each bin is**: every level, groups,
  lattice elements, the T bin, expanded the way MCNP expands the F card
  (`str(bin)` is the name MCNP's output gives it, `"(8<7[1 0 0]<3)"`).
  They are checked against the file, and left out with a warning if they
  don't match. In a chain a level of several cells or a lattice range makes
  one bin each, innermost level fastest, first lattice index fastest; a
  parenthesized level is one union.
- Values below 1e-99 are printed without the `E` (`2.50000-113`), in MCTAL
  and meshtal files alike: read exactly here (some other readers take them
  for 0).

## CuV files (D1SUNED)

An FMESH with `out=cuv` (D1SUNED's cell-under-voxel mesh) is not written to
the meshtal file but to two files holding every CuV mesh of the run: the
`meshinfo` gives each (voxel, cell) pair's material, density and share of
the voxel's volume, the `cuvmsh` the same pairs with their flux spectra.

```python
from aci_tallies import cuv

f = cuv.open("cuvmsh", input="model.i")    # or a meshinfo; aci_tallies.open works too
p = f[14]                                  # CuvPairs: every (voxel, cell) pair of mesh 14
p.voxel, p.cell, p.fraction, p.flux        # flux: (pair, energy), per cm² and source particle
f.read(14, voxels=slice(0, 10_000))        # only those voxels' part of the file
both = cuv.open_mesh("cuvmsh", "meshinfo") # + material and density, pairs checked equal
f.to_mesh(14)                              # MeshTally of voxel fluxes: to_vtk, isel, ...
```

- Pairs come voxel by voxel in the mesh's flat order (`voxel`, as
  `Geometry.locate` numbers them); each voxel's printed centre is checked
  against the geometry, so a file in another order is refused.
- `to_polars()` (a row per pair; `spectra=True`: per pair and energy bin),
  `to_xarray()` (dimensions `pair`, `energy`). `to_mesh` sums each voxel's
  cells, flux × volume share: what the same FMESH without `out=cuv` scores
  (to its 6 printed digits, `tests/fixtures/d1suned`). Its errors are
  combined as if the cells were independent: they aren't, and the estimate
  ran 2–18% low there.
- The headers round the geometry like a meshtal header (and the energy
  bounds to 3 digits) and leave out `TR=`; `input=` gives the FMESH card's
  exact ones. D1SUNED's own keywords (`sppts`, `spmode`, ...) are known.
- D1SUNED 3.x meshinfo files of cylinders (theta in radians, volumes 2π too
  large) are corrected (`legacy_cylindrical`).
- D1SUNED 4.1.3.2 crashed writing a cylindrical CuV mesh with 2 energy
  bins ("output statement overflows record"); 20 bins ran.

## Reliability

A damaged part of a file never stops the rest from being read: a bad row
or line leaves its values NaN with a `ReadWarning` (with its line number),
an unreadable section or tally is listed in `f.failed`, and every value is
where it belongs whatever is missing around it. A file changed since it was
opened is reported, never misread. Checked against MCNP 6.2 and 6.3 output
with exact expected values, property tests of the chunked parsing, and, for
MCTAL, F4Enix's reader (identical wherever it reads right).

## Performance

On 96 cores, files in the OS cache:

| file | open | read all | peak memory |
|---|---|---|---|
| meshtal, 6.6 GB, 14 tallies | 0.06 s | 0.33 s | 1.6 GB (the arrays) |
| MCTAL, 2 GB, 104M values | 0.11 s | 0.22 s | 1.7 GB (the arrays) |

One bin of a 2.5M-voxel mesh tally: 0.01 s; ten cells of a 4M-cell MCTAL
tally: 1 ms. One core reads about 0.6–0.9 GB/s. The previous version read
the meshtal file in 3.3 s with 5.1 GB of memory (it held a section's text).
`cargo run --release --example meshtal -- <file> --all` (or `mctal`) times
a file and prints a checksum per tally.

## Rust

```rust
use aci_tallies::{common::Input, mctal, meshtal};

let input = Input::read("model.i")?;
let mesh = meshtal::Reader::open_with_input("meshtal", &input)?;
let bin = mesh.read_bin(14, 0, 0)?;            // one energy and time bin
let cells = mctal::Reader::open_with_input("mctal", &input)?;
let part = cells.read_f(4, 0..10)?;            // ten cells of tally 4
```

`aci_tallies::detect(head)` tells the formats apart from a file's first
bytes.

## Development

```bash
cargo test --workspace
python -m venv .venv && . .venv/bin/activate
pip install maturin && maturin develop --release   # the extension, into the venv
pip install -e ".[dev]"
pytest tests/python && mypy python/aci_tallies tests/python && ruff check python tests
```

`maturin develop` without `--release` builds a debug extension, 10–30x
slower at parsing. The fixtures under `tests/fixtures` are MCNP output; each
`.mcnp` says how to regenerate it (`tests/fixtures/d1suned`: its README).

## License

[EUPL-1.2](LICENSE).
