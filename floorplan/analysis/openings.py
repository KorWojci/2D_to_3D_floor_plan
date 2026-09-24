"""Classification of wall openings into doors, windows and passages."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from ..model import Arc, Drawing, Label, Opening, Settings
from . import keywords as kw
from .walls import ANG_TOL, GapCandidate, Line, Rect, WallLine, _endpoints, merge_collinear


OUTDOOR = re.compile(r"taras|balkon|balcon|balcony|terrace|terrass|loggia|patio|veranda|weranda|ogr[oó]d|garden|"
                     r"garten|jardin|jard[ií]n|giardino|deck|porch", re.I)


@dataclass
class OpeningContext:
    arcs: list[Arc]
    labels: list[Label]
    lines_all: np.ndarray  # merged (theta, v, s1, s2) of *all* segments
    E1: np.ndarray
    E2: np.ndarray
    cue: Any
    wall_lines: list[WallLine]
    stats: dict[str, int] = field(default_factory=dict)
    outdoor_labels: list[tuple[str, float, float]] = field(default_factory=list)
    footprint: Any = None  # filled outline of the building (for interior / exterior tests)

    @classmethod
    def build(cls, d: Drawing, s: Settings, merged_walls, lines: list[WallLine], tol: float) -> "OpeningContext":
        arcs = [a for a in d.arcs if 250 <= a.r <= 2000 and 20 <= a.sweep <= 200]
        labels = [l for l in d.labels if l.kind]
        other = [sg for sg in d.segments if not kw.is_ignored(sg.layer)]
        allm = merge_collinear(other, tol)
        arr = np.array(allm, dtype=float) if allm else np.zeros((0, 4))
        E1, E2 = _endpoints(arr) if len(arr) else (np.zeros((0, 2)), np.zeros((0, 2)))
        outdoor = [(t.text, t.x, t.y) for t in d.texts if OUTDOOR.search(t.text)]
        return cls(arcs, labels, arr, E1, E2, d.cue_provider, lines, outdoor_labels=outdoor,
                   footprint=_footprint(lines, s))


def _footprint(lines: list[WallLine], s: Settings):
    """Building outline with rooms filled: wall pieces, gaps closed, holes filled."""
    polys = [p.polygon(wl.frame) for wl in lines for p in wl.pieces]
    if not polys:
        return None
    r = s.max_opening * 0.55
    closed = unary_union(polys).buffer(r, join_style=2).buffer(-r, join_style=2)
    return unary_union([Polygon(g.exterior) for g in getattr(closed, "geoms", [closed]) if not g.is_empty])


def _swing_default(fr: Line, s1: float, s2: float, v2: float) -> dict[str, Any]:
    hinge = fr.world(s1, v2)
    a0 = math.degrees(math.atan2(fr.u[1], fr.u[0]))
    return {"hinge": hinge, "r": s2 - s1, "a0": a0 % 360, "a1": (a0 + 90) % 360, "assumed": True}


def classify_gap(g: GapCandidate, ctx: OpeningContext, s: Settings, require_evidence: bool = False) -> Optional[Opening]:
    fr = g.wl.frame
    w = g.s2 - g.s1
    depth = g.v2 - g.v1
    vm = (g.v1 + g.v2) / 2
    poly_pts = [fr.world(g.s1, g.v1), fr.world(g.s2, g.v1), fr.world(g.s2, g.v2), fr.world(g.s1, g.v2)]
    region = Polygon(poly_pts)
    p1, p2 = fr.world(g.s1, vm), fr.world(g.s2, vm)
    evidence: list[str] = []
    typ: Optional[str] = None
    swing = None

    # 1. named blocks / layers
    near = region.buffer(0.15 * w)
    kinds = {l.kind for l in ctx.labels if box(*l.bbox).intersects(near)}
    if "door" in kinds:
        typ = "door"
        evidence.append("door block/layer")
    elif "window" in kinds:
        typ = "window"
        evidence.append("window block/layer")

    # 2. door swing arcs
    hinges = [fr.world(sv, vv) for sv in (g.s1, g.s2) for vv in (g.v1, g.v2, vm)]
    reach = 0.6 * depth + 0.08 * w + 20
    best_arc = None
    for a in ctx.arcs:
        ratio = a.r / w
        if not (0.8 <= ratio <= 1.25 or 0.4 <= ratio <= 0.62):
            continue
        dmin = min(math.hypot(a.cx - hx, a.cy - hy) for hx, hy in hinges)
        if dmin <= reach and (best_arc is None or dmin < best_arc[0]):
            best_arc = (dmin, a)
    if best_arc and typ != "window":
        a = best_arc[1]
        typ = "door"
        evidence.append("door swing arc" + (" (double leaf)" if a.r / w < 0.7 else ""))
        swing = {"hinge": (a.cx, a.cy), "r": a.r, "a0": a.a0, "a1": a.a1, "assumed": False}

    # 3. glazing lines / fillers
    if typ is None:
        fill_cov = sum(max(0.0, min(f.s2, g.s2) - max(f.s1, g.s1)) for f in g.wl.fillers)
        if fill_cov >= 0.5 * w:
            typ = "window"
            evidence.append("glazing lines")
        elif len(ctx.lines_all):
            u, nn = fr.u, fr.n
            dth = np.abs(((ctx.lines_all[:, 0] - fr.theta) + 90) % 180 - 90)
            V = (ctx.E1 @ nn + ctx.E2 @ nn) / 2
            Sa, Sb = ctx.E1 @ u, ctx.E2 @ u
            ov = np.minimum(np.maximum(Sa, Sb), g.s2) - np.maximum(np.minimum(Sa, Sb), g.s1)
            m = (dth <= ANG_TOL) & (V >= g.v1 - 2) & (V <= g.v2 + 2) & (ov >= 0.6 * w)
            if m.any():
                typ = "window"
                evidence.append(f"{int(m.sum())} line(s) across the opening")

    # 4. bitmap ink
    cue = ctx.cue
    if typ is None and cue is not None and hasattr(cue, "door_swing"):
        sc, leaf, swing_found = 0.0, w, None
        need = 0.8 if g.source == "end" else 0.6
        for sv, other in ((g.s1, g.s2), (g.s2, g.s1)):
            into = 1 if other > sv else -1
            for vv, inset in [(vv, i) for vv in (g.v1, g.v2, vm) for i in (0.0, 0.04 * w)]:
                hinge = fr.world(sv + into * inset, vv)  # hinge may sit a little inside the jamb
                closed = fr.u * into
                # the leaf may be narrower than the opening (door + fixed side panel) or half of
                # it (double door)
                for f in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35):
                    v, side = cue.door_swing(hinge, f * w, closed, fr.n)
                    if v > sc + 0.05 or (v >= sc and f > leaf / w):
                        sc, leaf = v, f * w
                        a_closed = math.degrees(math.atan2(closed[1], closed[0])) % 360
                        a_open = math.degrees(math.atan2(fr.n[1] * side, fr.n[0] * side)) % 360
                        ccw = (a_open - a_closed) % 360 < 180
                        swing_found = {"hinge": hinge, "r": f * w, "a0": a_closed if ccw else a_open,
                                       "a1": a_open if ccw else a_closed, "assumed": False}
        gl = cue.glazing(p1, p2, fr.n, depth)
        if leaf < 0.85 * w and gl >= 0.6:
            need = max(need, 0.85)  # door + side panel next to glazing: demand a clear symbol
        if sc >= need and (sc >= gl or leaf < 0.95 * w):
            typ = "door"
            swing = swing_found
            evidence.append(f"door swing in image ({sc:.0%})" + (f", leaf {leaf:.0f} mm + side panel" if leaf < 0.85 * w and sc < gl + 0.5 else ""))
        elif gl >= 0.6:
            typ = "window"
            evidence.append(f"glazing in image ({gl:.0%})")

    # 4a. windows only exist in exterior walls: an interior opening with lines in it is a door
    #     (leaf drawn in the opening) unless a window block/layer says otherwise
    if typ == "window" and ctx.footprint is not None and "window block/layer" not in evidence:
        off = depth / 2 + 400
        outside = any(not ctx.footprint.contains(Point(p1[0] + fr.n[0] * sgn * off + (p2[0] - p1[0]) / 2,
                                                      p1[1] + fr.n[1] * sgn * off + (p2[1] - p1[1]) / 2))
                      for sgn in (1, -1))
        if not outside:
            typ = "door"
            evidence.append("interior wall - not a window")

    # 4b. room names next to the opening: an opening onto a terrace / balcony is a door
    if typ == "window" and ctx.outdoor_labels:
        cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        for name, lx, ly in ctx.outdoor_labels:
            dn = abs((lx - cx) * fr.n[0] + (ly - cy) * fr.n[1])
            da = abs((lx - cx) * fr.u[0] + (ly - cy) * fr.u[1])
            if dn <= 3000 and da <= max(w / 2, 1500):
                typ = "door"
                evidence.append(f"leads to '{name}' (terrace/balcony door)")
                break

    sill, head = None, None
    # 5. 3D model
    if cue is not None and hasattr(cue, "low_cover"):
        lc = cue.low_cover(p1, p2, fr.n, depth)
        if typ is None:
            typ = "window" if lc >= 0.5 else "door"
            evidence.append("3D model: " + ("wall below opening" if lc >= 0.5 else "open to the floor"))
        c = fr.world((g.s1 + g.s2) / 2, g.v1 + 0.27 * depth)
        prof = cue.vertical_profile(*c)
        if prof:
            target = s.slice_height
            iv = min(prof, key=lambda t: 0 if t[0] <= target <= t[1] else min(abs(t[0] - target), abs(t[1] - target)))
            sill, head = max(0.0, iv[0]), iv[1]
            if sill < 50:
                sill = 0.0
                if typ == "window":
                    typ = "door"
            elif typ == "door":
                typ = "window"
            evidence.append(f"3D model heights {sill:.0f}-{head:.0f} mm")

    if typ is None:
        if require_evidence:
            return None
        typ = "passage"  # a recess to door height without a door leaf
        evidence.append("gap in wall (no door symbol found)")
    if typ == "door" and swing is None:
        swing = _swing_default(fr, g.s1, g.s2, g.v2)
    if sill is None:
        if typ == "window":
            sill, head = s.window_sill, s.window_head
        else:
            sill, head = 0.0, s.door_height
    ctx.stats[typ] = ctx.stats.get(typ, 0) + 1
    return Opening("", typ, p1, p2, round(depth, 3), round(sill, 1), round(head, 1), poly_pts, "", evidence, swing)


def find_end_openings(lines: list[WallLine], ctx: OpeningContext, s: Settings, tol: float,
                      existing: list[Opening]) -> list[Opening]:
    """Openings between a free wall end and a perpendicular wall (only with evidence)."""
    items = [(wl, p) for wl in lines for p in wl.pieces]
    polys = [p.polygon(wl.frame) for wl, p in items]
    taken = [Polygon(o.polygon).buffer(tol) for o in existing]
    out: list[Opening] = []
    for k, (wl, p) in enumerate(items):
        fr = wl.frame
        for end in (0, 1):
            if p.extended[end]:
                continue
            if end == 1:
                probe = Rect(p.theta, p.v1 + tol, p.v2 - tol, p.s2 + tol, p.s2 + s.max_opening).polygon(fr)
            else:
                probe = Rect(p.theta, p.v1 + tol, p.v2 - tol, p.s1 - s.max_opening, p.s1 - tol).polygon(fr)
            best = None
            for m, q in enumerate(polys):
                if m == k or not q.intersects(probe):
                    continue
                inter = q.intersection(probe)
                ss = [fr.local(x, y)[0] for x, y in np.asarray(inter.exterior.coords)] if hasattr(inter, "exterior") else []
                if not ss:
                    continue
                gdist = (min(ss) - p.s2) if end == 1 else (p.s1 - max(ss))
                if best is None or gdist < best:
                    best = gdist
            if best is None or not (s.min_opening <= best <= s.max_opening):
                continue
            s1, s2 = (p.s2, p.s2 + best) if end == 1 else (p.s1 - best, p.s1)
            cand = GapCandidate(wl, s1, s2, p.v1, p.v2, "end", p)
            region = Rect(p.theta, p.v1, p.v2, s1, s2).polygon(fr)
            if any(t.intersection(region).area > 0.3 * region.area for t in taken):
                continue
            op = classify_gap(cand, ctx, s, require_evidence=True)
            if op and not op.evidence[0].startswith("gap in wall"):
                out.append(op)
                taken.append(region.buffer(tol))
    return out

