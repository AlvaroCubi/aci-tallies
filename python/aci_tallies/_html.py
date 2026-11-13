"""Rich display in notebooks (`_repr_html_`)."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._native import (
        Axis,
        CuvFile,
        CuvInfo,
        CuvPairs,
        MctalFile,
        MctalInfo,
        MctalTally,
        MeshtalFile,
        MeshTally,
        TallyInfo,
    )

_STYLE = (
    "<style>.aci-t table{border-collapse:collapse;font-size:0.9em}"
    ".aci-t th,.aci-t td{padding:2px 8px;text-align:left;border-bottom:1px solid #8884}"
    ".aci-t .w{color:#b55}</style>"
)


def _axis(axis: Axis, unit: str) -> str:
    if axis.n_bins == 1 and not axis.total:
        return "—"
    b = axis.bounds
    span = f" ({b[0]:.3g} – {b[-1]:.3g} {unit})" if b is not None else ""
    return f"{axis.n_bins}{' + Total' if axis.total else ''}{span}"


def _mesh(t: MeshTally | TallyInfo | CuvInfo | CuvPairs) -> str:
    g = t.geometry
    dims = " × ".join(f"{n}={d}" for n, d in zip(g.axis_names, g.shape))
    return f"{g.kind}{' (TR)' if g.transform is not None else ''}: {dims}"


def _row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def meshtal_file(f: MeshtalFile) -> str:
    head = [
        "tally",
        "particle",
        "mesh",
        "energy bins",
        "time bins",
        "out",
        "comment",
        "warnings",
    ]
    rows = []
    for t in f.tallies:
        rows.append(
            _row(
                [
                    str(t.id),
                    html.escape(t.particle),
                    _mesh(t),
                    _axis(t.energy, "MeV"),
                    _axis(t.time, "sh"),
                    t.out_format,
                    html.escape(t.comment or ""),
                    str(len(t.warnings)) if t.warnings else "",
                ]
            )
        )
    failed = "".join(
        f"<div class='w'>tally {i}: {html.escape(why)}</div>" for i, why in f.failed
    )
    title = html.escape(f.title)
    nps = f" · {f.histories:.6g} histories" if f.histories is not None else ""
    path = html.escape(str(f.path)) if f.path is not None else "&lt;memory&gt;"
    return (
        f"{_STYLE}<div class='aci-t'><b>MeshtalFile</b> {path}<br>{title}{nps}"
        f"<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(rows)}</table>"
        f"{failed}</div>"
    )


def mesh_tally(t: MeshTally) -> str:
    r, e = t.result, t.rel_error
    finite = np.isfinite(r)
    stats = ""
    if finite.any():
        nonzero = finite & (r != 0)
        stats = (
            f"<tr><td>result</td><td>min {np.nanmin(r):.4g}, max {np.nanmax(r):.4g}"
            f"{', ' + str(int((~finite).sum())) + ' NaN' if not finite.all() else ''}</td></tr>"
        )
        if nonzero.any():
            stats += (
                f"<tr><td>rel. error</td><td>max {np.max(e[nonzero]):.4g} where scored; "
                f"{int(nonzero.sum())} of {r.size} values scored</td></tr>"
            )
    warnings = "".join(
        f"<div class='w'>{html.escape(str(w))}</div>" for w in t.warnings[:10]
    )
    if len(t.warnings) > 10:
        warnings += f"<div class='w'>… {len(t.warnings) - 10} more</div>"
    rows = [
        ["particle", html.escape(t.particle)],
        ["comment", html.escape(t.comment or "")],
        ["mesh", _mesh(t)],
        ["energy bins", _axis(t.energy, "MeV")],
        ["time bins", _axis(t.time, "sh")],
        ["shape", str(tuple(t.shape))],
        ["out", t.out_format],
    ]
    if t.geometry_from_input:
        rows.append(["geometry", "from the MCNP input"])
    return (
        f"{_STYLE}<div class='aci-t'><b>MeshTally {t.id}</b><table>"
        f"{''.join(_row(r) for r in rows)}{stats}</table>{warnings}</div>"
    )


def _mctal_bins(t: MctalTally | MctalInfo) -> str:
    parts = []
    for name, n in zip(t.dims, t.shape):
        if n > 1:
            axis = t.axes[name]
            extra = (
                " incl. Total"
                if axis.total
                else (" cumulative" if axis.cumulative else "")
            )
            parts.append(f"{name}={n}{extra}")
    return ", ".join(parts) or "one value"


def mctal_file(f: MctalFile) -> str:
    head = ["tally", "type", "particles", "bins", "comment", "warnings"]
    rows = []
    for t in f.tallies:
        kind = f"F{t.kind}" if t.mesh is None else "TMESH"
        kind = (t.modifier + kind) + (f" ({t.detector})" if t.detector else "")
        rows.append(
            _row(
                [
                    str(t.id),
                    html.escape(kind),
                    html.escape(t.particles),
                    _mctal_bins(t),
                    html.escape(t.comment or ""),
                    str(len(t.warnings)) if t.warnings else "",
                ]
            )
        )
    notes = "".join(
        f"<div class='w'>tally {i}: {html.escape(why)}</div>" for i, why in f.failed
    )
    notes += "".join(f"<div class='w'>{html.escape(str(w))}</div>" for w in f.warnings)
    k = f.kcode
    kcode = (
        f" · KCODE: {k.cycles} cycles, {k.inactive} inactive" if k is not None else ""
    )
    pert = f" · {f.perturbations} perturbations" if f.perturbations else ""
    path = html.escape(str(f.path)) if f.path is not None else "&lt;memory&gt;"
    return (
        f"{_STYLE}<div class='aci-t'><b>MctalFile</b> {path}<br>{html.escape(f.title)}"
        f" · {f.histories:,} histories{kcode}{pert}"
        f"<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(rows)}</table>"
        f"{notes}</div>"
    )


def mctal_tally(t: MctalTally) -> str:
    r = t.result
    rows = [
        ["particles", html.escape(t.particles)],
        ["comment", html.escape(t.comment or "")],
        ["bins", _mctal_bins(t)],
        ["shape", str(tuple(t.shape))],
    ]
    if t.detector:
        rows.append(["detector", t.detector])
    if t.perturbation is not None:
        rows.append(["perturbation", str(t.perturbation)])
    if r.size:
        rows.append(["result", f"min {np.nanmin(r):.4g}, max {np.nanmax(r):.4g}"])
    warnings = "".join(
        f"<div class='w'>{html.escape(str(w))}</div>" for w in t.warnings[:10]
    )
    return (
        f"{_STYLE}<div class='aci-t'><b>Tally {t.id}</b><table>"
        f"{''.join(_row(r) for r in rows)}</table>{warnings}</div>"
    )


def cuv_file(f: CuvFile) -> str:
    head = [
        "mesh",
        "particle",
        "geometry",
        "voxels",
        "energy bins",
        "comment",
        "warnings",
    ]
    rows = [
        _row(
            [
                str(t.id),
                html.escape(t.particle or ""),
                _mesh(t),
                str(t.n_voxels),
                _axis(t.energy, "MeV") if t.energy is not None else "",
                html.escape(t.comment or ""),
                str(len(t.warnings)) if t.warnings else "",
            ]
        )
        for t in f.tallies
    ]
    failed = "".join(
        f"<div class='w'>mesh {i}: {html.escape(why)}</div>" for i, why in f.failed
    )
    nps = f" · {f.histories:.6g} histories" if f.histories is not None else ""
    path = html.escape(str(f.path)) if f.path is not None else "&lt;memory&gt;"
    return (
        f"{_STYLE}<div class='aci-t'><b>CuvFile</b> ({f.kind}) {path}<br>"
        f"{html.escape(f.title)}{nps}"
        f"<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(rows)}</table>"
        f"{failed}</div>"
    )


def cuv_pairs(p: CuvPairs) -> str:
    start, stop = p.voxels
    rows = [
        ["mesh", _mesh(p)],
        ["voxels read", f"{start} – {stop} of {p.n_voxels}"],
        ["pairs", str(len(p))],
        ["cells", str(len(np.unique(p.cell)))],
    ]
    if p.energy is not None:
        rows.append(["energy bins", _axis(p.energy, "MeV")])
    if p.total is not None and len(p):
        scored = p.total > 0
        rows.append(["scored", f"{int(scored.sum())} of {len(p)} pairs"])
    if p.material is not None:
        rows.append(
            ["materials", ", ".join(str(m) for m in np.unique(p.material)[:20])]
        )
    if p.geometry_from_input:
        rows.append(["geometry", "from the MCNP input"])
    warnings = "".join(
        f"<div class='w'>{html.escape(str(w))}</div>" for w in p.warnings[:10]
    )
    return (
        f"{_STYLE}<div class='aci-t'><b>CuvPairs {p.id}</b><table>"
        f"{''.join(_row(r) for r in rows)}</table>{warnings}</div>"
    )
