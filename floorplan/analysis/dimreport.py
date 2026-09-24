"""Builds the list of dimensions of the final plan.

Every wall length / thickness and every opening width gets a dimension.  Each one is
labelled with its origin:

* ``stated``   - written in the source drawing (value taken from the drawing text);
* ``derived``  - not written, but solved from other written dimensions (chains/overall);
* ``computed`` - not written, measured on the calibrated geometry.
"""
from __future__ import annotations

import math
from typing import Any

from ..log import Log
from ..model import Dimension, Plan
from .calibration import Calibration, finalize_derived

MATCH_TOL = 3.0  # mm


def _proj(p, u):
    return p[0] * u[0] + p[1] * u[1]


def _matches(stated: list[dict[str, Any]], a, b, value: float) -> dict[str, Any] | None:
    """Find a stated dimension measuring the same distance between the same points."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    u = (dx / L, dy / L)
    for sd in stated:
        ang = math.radians(sd["angle"])
        su = (math.cos(ang), math.sin(ang))
        if abs(abs(su[0] * u[0] + su[1] * u[1]) - 1) > 1e-3:
            continue
        if abs(sd["stated_mm"] - value) > MATCH_TOL:
            continue
        s1, s2 = sorted((_proj(sd["p1"], u), _proj(sd["p2"], u)))
        t1, t2 = sorted((_proj(a, u), _proj(b, u)))
        if abs(s1 - t1) <= MATCH_TOL * 2 and abs(s2 - t2) <= MATCH_TOL * 2:
            return sd
    return None


def build(plan: Plan, cal: Calibration, stated: list[dict[str, Any]], log: Log, fx=None, fy=None) -> list[Dimension]:
    dims: list[Dimension] = []
    # ---- dimensions written in the source (redrawn at their original place)
    for sd in stated:
        if sd["geom_mm"] < 1:
            continue
        dims.append(Dimension("source", sd["p1"], sd["p2"], round(sd["stated_mm"], 2), "stated",
                              offset=0.0, stated_text=sd["text"], deviation=round(sd["stated_mm"] - sd["geom_mm"], 2),
                              line_pos=sd["line_pos"]))
    worst = max((abs(d.deviation) for d in dims), default=0.0)
    derived = finalize_derived(cal, fx, fy)
    dims += derived

    n_stated = n_comp = 0
    for w in plan.walls:
        L = w.length
        dx, dy = (w.p2[0] - w.p1[0]) / L, (w.p2[1] - w.p1[1]) / L
        nx, ny = -dy, dx
        h = w.thickness / 2
        faces = [((w.p1[0] + nx * h, w.p1[1] + ny * h), (w.p2[0] + nx * h, w.p2[1] + ny * h)),
                 ((w.p1[0] - nx * h, w.p1[1] - ny * h), (w.p2[0] - nx * h, w.p2[1] - ny * h))]
        m = None
        for a, b in faces:
            m = _matches(stated, a, b, L)
            if m:
                break
        src = "stated" if m else "computed"
        n_stated += bool(m)
        n_comp += not m
        dims.append(Dimension("wall_length", faces[0][0], faces[0][1], round(L, 1), src, w.id, offset=w.thickness / 2 + 250,
                              stated_text=m["text"] if m else ""))
        dims.append(Dimension("wall_thickness", faces[0][0], faces[1][0], round(w.thickness, 1), "computed", w.id))
    for o in plan.openings:
        m = _matches(stated, o.p1, o.p2, o.width)
        src = "stated" if m else "computed"
        n_stated += bool(m)
        n_comp += not m
        dims.append(Dimension("opening_width", o.p1, o.p2, round(o.width, 1), src, o.id, offset=o.depth / 2 + 150,
                              stated_text=m["text"] if m else ""))
    # ---- overall size
    if plan.walls:
        x0, y0, x1, y1 = plan.bbox()
        dims.append(Dimension("overall", (x0, y0), (x1, y0), round(x1 - x0, 1), "computed", "overall", offset=-900))
        dims.append(Dimension("overall", (x1, y0), (x1, y1), round(y1 - y0, 1), "computed", "overall", offset=-900))
    log.info(f"Dimensions: {len(stated)} written in the drawing, {len(derived)} derived from other dimensions, "
             f"{n_stated} wall/opening sizes confirmed by written dimensions, {n_comp} computed from geometry")
    if stated:
        log.info(f"Largest difference between a written dimension and the final geometry: {worst:.1f} mm")
        if worst > 5:
            log.warn("Some written dimensions could not be honoured exactly (see the dimension report)")
    for d in derived:
        log.info(f"Missing dimension derived from others: {d.value:.1f} mm")
    return dims
