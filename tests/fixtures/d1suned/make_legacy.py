"""Write `legacy_3.1.4_cyl/meshinfo`: `meshinfo` as D1SUNED 3.x writes it.

D1SUNED 3.1.4 isn't run here; its meshinfo differs from 4.1.3.2's for the
same cylinder in three ways only (seen comparing the two on one run): its
first line, a theta line in radians in `f10.8` fields that run together,
and cylinder voxel volumes 2π too large. Box meshes are unchanged.
Run from this directory: python make_legacy.py
"""

import math
import re
from pathlib import Path

lines = Path("meshinfo").read_text().splitlines(keepends=True)
probid = lines[0].split("probid =")[1]
out = [f"d1sune version 3.1.4 ld=10142021  probid ={probid}"]
radii = heights = None
for line in lines[1:]:
    if line.lstrip().startswith("Mesh tally number:"):
        radii = heights = None
    elif line.lstrip().startswith("R direction:"):
        radii = [float(v) for v in line.split(":")[1].split()]
    elif line.lstrip().startswith("Z direction:") and radii is not None:
        heights = [float(v) for v in line.split(":")[1].split()]
    elif line.lstrip().startswith("Theta direction (revolutions):"):
        turns = [float(v) for v in line.split(":")[1].split()]
        line = line.split(":")[0] + ":" + "".join(f"{2 * math.pi * t:10.8f}" for t in turns) + "\n"
    elif radii is not None and heights is not None:
        # A voxel line: index, volume, cell count (one theta bin).
        m = re.fullmatch(r"( *)(\d+)( +)(\S+)( +\d+\n)", line)
        if m:
            n = int(m.group(2)) - 1
            nz = len(heights) - 1
            i, k = divmod(n, nz)
            v = math.pi * (radii[i + 1] ** 2 - radii[i] ** 2) * (heights[k + 1] - heights[k])
            line = f"{m.group(1)}{m.group(2)}{m.group(3)}{2 * math.pi * v:.5E}{m.group(5)}"
    out.append(line)
Path("legacy_3.1.4_cyl").mkdir(exist_ok=True)
Path("legacy_3.1.4_cyl/meshinfo").write_text("".join(out))
