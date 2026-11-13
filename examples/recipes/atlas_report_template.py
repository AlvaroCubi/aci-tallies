"""Template for a per-run atlas report: one 2D slice image per tally.

Producing an atlas is a two-phase job. The *explore* phase — finding the
right value range, normalization, and slice location for each tally — is
open-ended, per-run, human-in-the-loop work; do it interactively (e.g. in a
notebook, the same way examples/quickstart.ipynb drives `to_pyvista`) rather
than trying to automate it. This script is the *freeze* phase: once you've
settled on those choices, copy this file per run, fill in MESHTAL_PATH and
TALLY_SETTINGS below with what you found, and run it:

    python atlas_report_template.py

TALLY_SETTINGS is also the record of what you decided — commit your copy of
this file alongside the run so the atlas is reproducible later.

Produces one PNG per configured tally in OUTPUT_DIR. If you're on a headless
machine with no X server, call `pyvista.start_xvfb()` before `main()` runs
(or install a VTK build with off-screen/EGL support).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import aci_tallies
import pyvista as pv

# --- Fill in after your exploration pass ------------------------------------

MESHTAL_PATH = Path("run.msht")
OUTPUT_DIR = Path("atlas")


@dataclass
class TallySettings:
    energy_index: int = 0
    slice_axis: str = "z"  # "x" | "y" | "z" — world-space axis to slice along
    slice_coord: float = 0.0  # world-space coordinate along slice_axis
    value_range: tuple[float, float] | None = None  # None = auto (data min/max)
    log_scale: bool = True  # note: breaks on zero/negative values in range
    normalize_by: float = 1.0  # e.g. a source normalization factor
    cmap: str = "viridis"


TALLY_SETTINGS: dict[int, TallySettings] = {
    14: TallySettings(slice_axis="z", slice_coord=0.0, value_range=(1e-6, 1e-2)),
    24: TallySettings(slice_axis="y", slice_coord=0.0, normalize_by=2.4e10),
}

# --- Rendering — usually shouldn't need to change ---------------------------

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_NORMAL = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_AXIS_VIEW = {"x": "view_yz", "y": "view_xz", "z": "view_xy"}


def render_tally_slice(
    tally: aci_tallies.meshtal.MeshTally, settings: TallySettings, out_path: Path
) -> None:
    """Render one world-space-axis-aligned slice of `tally` to `out_path`.

    Works for both CARTESIAN and CYLINDRICAL tallies unchanged: slicing is
    done in world space on the grid `to_pyvista` already built, not in the
    tally's own (r, z, theta) or (x, y, z) index space.
    """
    grid = tally.to_pyvista(energy=settings.energy_index)
    grid.cell_data["result"] = grid.cell_data["result"] * settings.normalize_by

    origin = list(grid.center)
    origin[_AXIS_INDEX[settings.slice_axis]] = settings.slice_coord
    sliced = grid.slice(normal=_AXIS_NORMAL[settings.slice_axis], origin=origin)
    if sliced.n_points == 0:
        raise ValueError(
            f"tally {tally.id}: slice_axis={settings.slice_axis!r} at "
            f"slice_coord={settings.slice_coord} doesn't intersect the mesh "
            f"(grid bounds: {grid.bounds}). Check TALLY_SETTINGS."
        )

    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.add_mesh(
        sliced,  # type: ignore[arg-type]
        scalars="result",
        cmap=settings.cmap,  # type: ignore[arg-type]
        clim=settings.value_range,
        log_scale=settings.log_scale,
        scalar_bar_args={"title": f"tally {tally.id} result"},
    )
    plotter.enable_parallel_projection()  # type: ignore[call-arg]
    getattr(plotter, _AXIS_VIEW[settings.slice_axis])()
    plotter.screenshot(str(out_path))
    plotter.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meshtal = aci_tallies.meshtal.open(MESHTAL_PATH)

    for tally_id, settings in TALLY_SETTINGS.items():
        if tally_id not in meshtal:
            raise ValueError(f"tally {tally_id} not found in {MESHTAL_PATH}")
        tally = meshtal[tally_id]  # reads only this tally
        out_path = OUTPUT_DIR / f"tally_{tally_id}.png"
        render_tally_slice(tally, settings, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
