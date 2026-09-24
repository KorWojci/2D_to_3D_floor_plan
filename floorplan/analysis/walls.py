"""Wall and opening detection on a calibrated (millimetre) drawing.

Algorithm (works for CAD lines, PDF/SVG vectors, traced bitmaps and 3D sections):

1. choose the segments that can be wall faces (wall layers if the file has them);
2. merge collinear overlapping pieces;
3. every pair of parallel segments whose distance is a plausible wall thickness and
   that overlap along their length yields a candidate wall rectangle;
4. rectangles are grouped into *wall lines* (same direction, overlapping band);
   thin rectangles lying between thick ones are window glazing, not walls;
5. wall pieces are extended into corners / T-junctions (the region is bounded by the
   perpendicular wall and not yet covered by any wall);
6. gaps between consecutive pieces of a wall line, and sections between two jamb lines
   that contain glazing lines, are openings.  They are classified as door / window /
   passage using door swing arcs, glazing lines, block/layer names, bitmap ink or the
   3D model (see ``openings.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..log import Log
from ..model import Arc, Drawing, Opening, Seg, Settings, Wall
from . import keywords as kw

ANG_TOL = 2.0  # degrees


@dataclass
class Line:
    """Infinite line frame: point = s*u + v*n."""

    theta: float  # degrees in [0, 180)

    @property
    def u(self) -> np.ndarray:
        a = math.radians(self.theta)
        return np.array([math.cos(a), math.sin(a)])

    @property
    def n(self) -> np.ndarray:
        a = math.radians(self.theta)
        return np.array([-math.sin(a), math.cos(a)])

    def world(self, s: float, v: float) -> tuple[float, float]:
        p = s * self.u + v * self.n
        return float(p[0]), float(p[1])

    def local(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y])
        return float(p @ self.u), float(p @ self.n)


@dataclass
class Rect:
    theta: float
    v1: float
    v2: float
    s1: float
    s2: float
    kind: str = "thick"  # thick | thin | filler
    extended: list[bool] = field(default_factory=lambda: [False, False])

    @property
    def t(self) -> float:
        return self.v2 - self.v1

    def polygon(self, frame: Line) -> Polygon:
        return Polygon([frame.world(self.s1, self.v1), frame.world(self.s2, self.v1),
                        frame.world(self.s2, self.v2), frame.world(self.s1, self.v2)])


@dataclass
class WallLine:
    frame: Line
    rects: list[Rect]
    pieces: list[Rect] = field(default_factory=list)
    fillers: list[Rect] = field(default_factory=list)


# ------------------------------------------------------------------ selection


def _arc_chords(a: Arc, step: float = 3.0) -> list[Seg]:
    a0, sweep = a.a0, a.sweep
    angs = [a0]
    k = math.floor(a0 / step) + 1
    while k * step < a0 + sweep - 1e-6:
        angs.append(k * step)
        k += 1
    angs.append(a0 + sweep)
    pts = [a.point(t) for t in angs]
    return [Seg(p[0], p[1], q[0], q[1], a.layer, a.block) for p, q in zip(pts, pts[1:])]


def select_wall_segments(d: Drawing, s: Settings, log: Log) -> list[Seg]:
    segs = d.segments
    layers: dict[str, int] = {}
    for sg in segs:
        layers[sg.layer] = layers.get(sg.layer, 0) + 1
    wall_layers = {l for l in layers if kw.is_wall(l, s.wall_layers) and not kw.classify(l, s.door_layers, s.window_layers)}
    n_wall = sum(layers[l] for l in wall_layers)

    def opening_part(sg: Seg) -> bool:
        return bool(kw.classify(sg.layer, s.door_layers, s.window_layers) or kw.classify(sg.block, s.door_layers, s.window_layers))

    if wall_layers and n_wall >= 4:
        log.info(f"Using wall layer(s): {', '.join(sorted(wall_layers))} ({n_wall} segments)")
        out = [sg for sg in segs if sg.layer in wall_layers and not opening_part(sg)]
        arcs = [a for a in d.arcs if a.layer in wall_layers]
    else:
        if len(layers) > 1:
            log.info("No wall layer recognised - analysing all visible geometry "
                     "(set 'wall layers' in the options if walls are on a specific layer)")
        out = [sg for sg in segs if not kw.is_ignored(sg.layer) and not opening_part(sg)]
        arcs = [a for a in d.arcs if a.r >= 2000 and not kw.classify(a.layer) and not kw.classify(a.block)]
    for a in arcs:
        if a.r >= 1500 or wall_layers:
            out += _arc_chords(a)
    return out


# ------------------------------------------------------------------ merging


def merge_collinear(segs: list[Seg], tol: float) -> list[tuple[float, float, float, float]]:
    """Merge collinear, overlapping segments.  Returns (theta, v, s1, s2) tuples."""
    items = []
    for sg in segs:
        dx, dy = sg.x2 - sg.x1, sg.y2 - sg.y1
        L = math.hypot(dx, dy)
        if L < tol:
            continue
        th = math.degrees(math.atan2(dy, dx)) % 180.0
        if th > 180 - 0.05:
            th -= 180.0
        items.append((th, sg))
    items.sort(key=lambda t: t[0])
    groups: list[list[tuple[float, Seg]]] = []
    for th, sg in items:
        if groups and th - groups[-1][-1][0] <= 0.05:
            groups[-1].append((th, sg))
        else:
            groups.append([(th, sg)])
    out = []
    for g in groups:
        th = float(np.mean([t for t, _ in g]))
        fr = Line(th)
        u, n = fr.u, fr.n
        rows = []
        for _, sg in g:
            p1 = np.array([sg.x1, sg.y1])
            p2 = np.array([sg.x2, sg.y2])
            v = float(((p1 + p2) / 2) @ n)
            a, b = float(p1 @ u), float(p2 @ u)
            rows.append((v, min(a, b), max(a, b)))
        rows.sort()
        # cluster by offset
        clusters: list[list[tuple[float, float, float]]] = []
        for r in rows:
            if clusters and r[0] - clusters[-1][-1][0] <= tol * 0.5:
                clusters[-1].append(r)
            else:
                clusters.append([r])
        for c in clusters:
            v = float(np.mean([r[0] for r in c]))
            iv = sorted((r[1], r[2]) for r in c)
            cur = list(iv[0])
            for a, b in iv[1:]:
                if a <= cur[1] + tol * 0.5:
                    cur[1] = max(cur[1], b)
                else:
                    out.append((th, v, cur[0], cur[1]))
                    cur = [a, b]
            out.append((th, v, cur[0], cur[1]))
    return out


# ------------------------------------------------------------------ pairing


def _endpoints(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = np.radians(lines[:, 0])
    U = np.column_stack([np.cos(th), np.sin(th)])
    N = np.column_stack([-np.sin(th), np.cos(th)])
    E1 = lines[:, 2:3] * U + lines[:, 1:2] * N
    E2 = lines[:, 3:4] * U + lines[:, 1:2] * N
    return E1, E2


def pair_rects(lines: list[tuple[float, float, float, float]], s: Settings, tol: float) -> list[Rect]:
    """Candidate wall rectangles from pairs of parallel lines (vectorised per line)."""
    if not lines:
        return []
    arr = np.array(lines, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    E1, E2 = _endpoints(arr)
    th = arr[:, 0]
    n = len(arr)
    idx = np.arange(n)
    rects: list[Rect] = []
    for i in range(n):
        dth = np.abs(th - th[i])
        dth = np.minimum(dth, 180 - dth)
        js = np.nonzero((dth <= ANG_TOL) & (idx > i))[0]
        if len(js) == 0:
            continue
        fr = Line(th[i])
        u, nn = fr.u, fr.n
        vi, a1, b1 = arr[i, 1], arr[i, 2], arr[i, 3]
        vj1 = E1[js] @ nn
        vj2 = E2[js] @ nn
        vj = (vj1 + vj2) / 2
        dist = np.abs(vj - vi)
        sa = E1[js] @ u
        sb = E2[js] @ u
        lo = np.maximum(a1, np.minimum(sa, sb))
        hi = np.minimum(b1, np.maximum(sa, sb))
        ov = hi - lo
        ok = (
            (dist >= s.min_wall_thickness - tol)
            & (dist <= s.max_wall_thickness + tol)
            & (np.abs(vj1 - vj2) <= np.maximum(tol, 0.03 * dist))
            & (ov >= 0.9 * dist - tol)
            & (ov >= 2 * tol)
        )
        for k in np.nonzero(ok)[0]:
            v1, v2 = sorted((float(vi), float(vj[k])))
            rects.append(Rect(float(th[i]), v1, v2, float(lo[k]), float(hi[k])))
    return rects


# ------------------------------------------------------------------ grouping


def group_lines(rects: list[Rect], tol: float, max_open: float) -> list[WallLine]:
    rects = sorted(rects, key=lambda r: r.theta)
    fams: list[list[Rect]] = []
    for r in rects:
        if fams and r.theta - fams[-1][-1].theta <= ANG_TOL / 2:
            fams[-1].append(r)
        else:
            fams.append([r])
    if len(fams) > 1 and fams[0][0].theta + 180 - fams[-1][-1].theta <= ANG_TOL / 2:
        fams[0] = fams.pop() + fams[0]
    out: list[WallLine] = []
    for fam in fams:
        # weighted mean angle (handle wrap)
        angs = np.array([r.theta for r in fam])
        if angs.max() - angs.min() > 90:
            angs = np.where(angs > 90, angs - 180, angs)
        w = np.array([r.s2 - r.s1 for r in fam])
        th = float(np.average(angs, weights=w)) % 180
        fr = Line(th)
        # re-express every rect in the common frame
        for r in fam:
            if abs(r.theta - th) > 1e-12:  # also catches 180° flips (same line, reversed frame)
                old = Line(r.theta)
                pts = [old.world(r.s1, r.v1), old.world(r.s2, r.v1), old.world(r.s2, r.v2), old.world(r.s1, r.v2)]
                loc = [fr.local(*p) for p in pts]
                r.s1, r.s2 = min(p[0] for p in loc), max(p[0] for p in loc)
                r.v1, r.v2 = min(p[1] for p in loc), max(p[1] for p in loc)
                r.theta = th
        # union-find on band overlap and span proximity
        m = len(fam)
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        idx = sorted(range(m), key=lambda k: fam[k].v1)
        for ii, a in enumerate(idx):
            ra = fam[a]
            for b in idx[ii + 1:]:
                rb = fam[b]
                if rb.v1 >= ra.v2 - tol:
                    break
                if min(ra.v2, rb.v2) - max(ra.v1, rb.v1) > tol and \
                        max(ra.s1, rb.s1) - min(ra.s2, rb.s2) <= max_open:
                    parent[find(a)] = find(b)
        comps: dict[int, list[Rect]] = {}
        for k in range(m):
            comps.setdefault(find(k), []).append(fam[k])
        for rs in comps.values():
            out.append(WallLine(fr, rs))
    return out


def _merge_intervals(iv: list[tuple[float, float]], tol: float) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def classify_rects(wl: WallLine, tol: float) -> None:
    """Split rects of a wall line into thick walls and thin glazing fillers; build pieces."""
    tmax = max(r.t for r in wl.rects)
    thick = [r for r in wl.rects if r.t >= 0.7 * tmax - tol]
    spans = _merge_intervals([(r.s1, r.s2) for r in thick], tol)
    lo, hi = spans[0][0], spans[-1][1]
    pieces: list[Rect] = []
    thick_ids = {id(r) for r in thick}
    for r in wl.rects:
        if id(r) in thick_ids:
            continue
        inside_thick = any(r.s1 >= a - tol and r.s2 <= b + tol for a, b in spans)
        if inside_thick:
            continue  # layer line inside a wall
        if r.s1 >= lo - tol and r.s2 <= hi + tol:
            r.kind = "filler"
            wl.fillers.append(r)
        else:
            r.kind = "thick"  # thinner wall continuing the line
            thick.append(r)
    # merge thick rects with the same band into pieces
    thick.sort(key=lambda r: (round(r.v1 / max(tol, 1e-9)), round(r.v2 / max(tol, 1e-9)), r.s1))
    for r in thick:
        for p in pieces:
            if abs(p.v1 - r.v1) <= tol and abs(p.v2 - r.v2) <= tol and r.s1 <= p.s2 + tol and r.s2 >= p.s1 - tol:
                p.s1, p.s2 = min(p.s1, r.s1), max(p.s2, r.s2)
                break
        else:
            pieces.append(Rect(r.theta, r.v1, r.v2, r.s1, r.s2))
    # drop pieces contained in a thicker piece of the same line
    pieces.sort(key=lambda p: -p.t)
    kept: list[Rect] = []
    for p in pieces:
        if any(q.v1 - tol <= p.v1 and p.v2 <= q.v2 + tol and q.s1 - tol <= p.s1 and p.s2 <= q.s2 + tol for q in kept):
            continue
        kept.append(p)
    wl.pieces = kept


def _merge_pieces(wl: WallLine, tol: float) -> None:
    ps = sorted(wl.pieces, key=lambda p: p.s1)
    out: list[Rect] = []
    for p in ps:
        for q in out:
            if abs(p.v1 - q.v1) <= tol and abs(p.v2 - q.v2) <= tol and p.s1 <= q.s2 + tol and p.s2 >= q.s1 - tol:
                q.s1, q.s2 = min(p.s1, q.s1), max(p.s2, q.s2)
                q.extended = [q.extended[0] or p.extended[0], q.extended[1] or p.extended[1]]
                break
        else:
            out.append(p)
    wl.pieces = out


# ------------------------------------------------------------------ junctions


def extend_junctions(lines: list[WallLine], s: Settings, tol: float) -> int:
    """Extend wall pieces into corners and T-junctions."""
    items = [(wl, p) for wl in lines for p in wl.pieces]
    if not items:
        return 0
    L = s.max_wall_thickness + tol
    polys = [p.polygon(wl.frame) for wl, p in items]
    ext_polys = []
    for wl, p in items:
        e = Rect(p.theta, p.v1, p.v2, p.s1 - L, p.s2 + L)
        ext_polys.append(e.polygon(wl.frame))
    tree = STRtree(ext_polys)
    covered = unary_union(polys)
    count = 0
    order = sorted(range(len(items)), key=lambda k: -(items[k][1].s2 - items[k][1].s1))
    for k in order:
        wl, p = items[k]
        fr = wl.frame
        for end in (0, 1):
            if end == 0:
                probe = Rect(p.theta, p.v1 + tol * 0.1, p.v2 - tol * 0.1, p.s1 - L, p.s1)
            else:
                probe = Rect(p.theta, p.v1 + tol * 0.1, p.v2 - tol * 0.1, p.s2, p.s2 + L)
            pp = probe.polygon(fr)
            best: Optional[float] = None
            for m in tree.query(pp):
                wl2, p2 = items[m]
                if wl2 is wl:
                    continue
                dth = abs(((wl2.frame.theta - fr.theta) + 90) % 180 - 90)
                if dth < 20:
                    continue
                inter = pp.intersection(ext_polys[m])
                if inter.is_empty or inter.area < tol * tol:
                    continue
                ss = [fr.local(x, y)[0] for x, y in shapely.get_coordinates(inter)]
                near = min(ss) if end == 1 else max(ss)
                far = max(ss) if end == 1 else min(ss)
                edge = p.s2 if end == 1 else p.s1
                if abs(near - edge) > tol * 2:
                    continue
                # the perpendicular wall must actually reach this wall's band
                reach = Rect(p.theta, p.v1 - tol, p.v2 + tol, min(edge, far) - tol, max(edge, far) + tol).polygon(fr)
                if polys[m].distance(reach) > tol * 2:
                    continue
                best = far if best is None else (max(best, far) if end == 1 else min(best, far))
            if best is None:
                continue
            region = (Rect(p.theta, p.v1, p.v2, p.s2, best) if end == 1 else Rect(p.theta, p.v1, p.v2, best, p.s1)).polygon(fr)
            if region.area < tol * tol:
                continue
            uncovered = region.difference(covered).area
            if uncovered < 0.2 * region.area:
                continue
            if end == 1:
                p.s2 = best
            else:
                p.s1 = best
            p.extended[end] = True
            covered = covered.union(region)
            count += 1
    for wl in lines:
        _merge_pieces(wl, tol)
    return count


# ------------------------------------------------------------------ main entry


@dataclass
class GapCandidate:
    wl: WallLine
    s1: float
    s2: float
    v1: float
    v2: float
    source: str  # 'gap' | 'section' | 'filler' | 'end'
    host: Optional[Rect] = None


def find_gaps(lines: list[WallLine], s: Settings, tol: float) -> list[GapCandidate]:
    out = []
    for wl in lines:
        ps = sorted(wl.pieces, key=lambda p: p.s1)
        if not ps:
            continue
        cur = ps[0]
        for b in ps[1:]:
            g = b.s1 - cur.s2
            if g > tol and s.min_opening - tol <= g <= s.max_opening + tol and \
                    min(cur.v2, b.v2) - max(cur.v1, b.v1) > tol:
                out.append(GapCandidate(wl, cur.s2, b.s1, min(cur.v1, b.v1), max(cur.v2, b.v2), "gap", cur))
            if b.s2 > cur.s2:
                cur = b
    return out


def find_sill_sections(lines: list[WallLine], merged: list[tuple[float, float, float, float]], s: Settings,
                       tol: float) -> list[GapCandidate]:
    """Windows drawn with lines across the opening (face lines + glazing between two jambs)."""
    if not merged:
        return []
    arr = np.array(merged, dtype=float)
    E1, E2 = _endpoints(arr)
    out = []
    for wl in lines:
        fr = wl.frame
        u, nn = fr.u, fr.n
        dth = np.abs(((arr[:, 0] - fr.theta) + 90) % 180 - 90)
        S1, S2 = E1 @ u, E2 @ u
        V1, V2 = E1 @ nn, E2 @ nn
        perp = np.abs(dth - 90) <= ANG_TOL
        par = dth <= ANG_TOL
        for p in list(wl.pieces):
            T = p.t
            if p.s2 - p.s1 < s.min_opening:
                continue
            sm = (S1 + S2) / 2
            cov = np.minimum(np.maximum(V1, V2), p.v2) - np.maximum(np.minimum(V1, V2), p.v1)
            jm = perp & (sm > p.s1 + tol) & (sm < p.s2 - tol) & (cov >= 0.8 * T)
            jambs = sorted(_merge_close(list(sm[jm]), tol))
            if len(jambs) < 2:
                continue
            vmid = (V1 + V2) / 2
            margin = max(tol, 0.08 * T)
            im = par & (vmid > p.v1 + margin) & (vmid < p.v2 - margin)
            internal = [(float(vmid[i]), float(min(S1[i], S2[i])), float(max(S1[i], S2[i]))) for i in np.nonzero(im)[0]]
            if not internal:
                continue
            bounds = [p.s1] + jambs + [p.s2]
            secs = list(zip(bounds, bounds[1:]))

            def offsets(a0, b0):
                return {round(v / tol) for v, a, b in internal if min(b, b0) - max(a, a0) > 0.5 * (b0 - a0)}

            for k, (j1, j2) in enumerate(secs):
                if k == 0 or k == len(secs) - 1:
                    continue  # a section must be bounded by jamb lines on both sides
                w = j2 - j1
                if not (s.min_opening - tol <= w <= s.max_opening + tol):
                    continue
                inside = [(v, a, b) for v, a, b in internal if min(b, j2) - max(a, j1) > 0]
                if not inside or max(min(b, j2) - max(a, j1) for _, a, b in inside) < 0.6 * w:
                    continue
                # lines running on through the jambs are layer lines of the wall, not glazing
                if any(a < j1 - 2 * tol or b > j2 + 2 * tol for _, a, b in inside):
                    continue
                # ... and so are lines repeated at the same offsets in both neighbouring sections
                vs = offsets(j1, j2)
                if vs & offsets(*secs[k - 1]) & offsets(*secs[k + 1]):
                    continue
                out.append(GapCandidate(wl, j1, j2, p.v1, p.v2, "section", p))
    return out


def _merge_close(vals: list[float], tol: float) -> list[float]:
    out: list[float] = []
    for v in sorted(vals):
        if out and v - out[-1] <= tol:
            continue
        out.append(v)
    return out


def split_piece_at(wl: WallLine, host: Rect, s1: float, s2: float, tol: float = 1.0) -> None:
    for p in wl.pieces:
        if abs(p.v1 - host.v1) <= tol and abs(p.v2 - host.v2) <= tol and p.s1 <= s1 + tol and p.s2 >= s2 - tol:
            break
    else:
        return
    wl.pieces.remove(p)
    if s1 - p.s1 > tol:
        wl.pieces.append(Rect(p.theta, p.v1, p.v2, p.s1, s1, extended=[p.extended[0], False]))
    if p.s2 - s2 > tol:
        wl.pieces.append(Rect(p.theta, p.v1, p.v2, s2, p.s2, extended=[False, p.extended[1]]))


def pieces_to_walls(lines: list[WallLine]) -> list[Wall]:
    walls = []
    n = 0
    for wl in lines:
        fr = wl.frame
        for p in sorted(wl.pieces, key=lambda p: p.s1):
            if p.s2 - p.s1 <= 1e-6:
                continue
            n += 1
            vm = (p.v1 + p.v2) / 2
            poly = [fr.world(p.s1, p.v1), fr.world(p.s2, p.v1), fr.world(p.s2, p.v2), fr.world(p.s1, p.v2)]
            walls.append(Wall(f"W{n}", fr.world(p.s1, vm), fr.world(p.s2, vm), round(p.t, 3), poly))
    return walls


def detect(d: Drawing, s: Settings, log: Log) -> tuple[list[Wall], list[Opening], list[WallLine]]:
    from .openings import OpeningContext, classify_gap, find_end_openings

    tol = max(1.0, (d.pixel_size * 1.5) if d.raster else 1.0)
    segs = select_wall_segments(d, s, log)
    merged = merge_collinear(segs, tol)
    log.info(f"{len(segs)} candidate wall segments -> {len(merged)} after merging collinear pieces")
    rects = pair_rects(merged, s, tol)
    log.info(f"{len(rects)} wall-face pairs with thickness {s.min_wall_thickness:g}-{s.max_wall_thickness:g} mm")
    if not rects:
        return [], [], []
    lines = group_lines(rects, tol, s.max_opening)
    for wl in lines:
        classify_rects(wl, tol)
    n_fill = sum(len(wl.fillers) for wl in lines)
    ext = extend_junctions(lines, s, tol)
    log.info(f"{len(lines)} wall lines, {sum(len(wl.pieces) for wl in lines)} wall pieces; "
             f"{ext} corner/T-junction fills; {n_fill} glazing fillers")

    ctx = OpeningContext.build(d, s, merged, lines, tol)
    openings: list[Opening] = []
    cands = find_gaps(lines, s, tol)
    sections = find_sill_sections(lines, merged, s, tol)
    for g in cands + sections:
        op = classify_gap(g, ctx, s, require_evidence=(g.source == "section"))
        if op is None:
            continue
        if g.source == "section":
            split_piece_at(g.wl, g.host, g.s1, g.s2, tol)
        openings.append(op)
    openings += find_end_openings(lines, ctx, s, tol, openings)
    # drop tiny slivers left after splitting
    for wl in lines:
        wl.pieces = [p for p in wl.pieces if p.s2 - p.s1 > tol]
    walls = pieces_to_walls(lines)
    counters: dict[str, int] = {}
    for op in sorted(openings, key=lambda o: (o.type, -round(o.p1[1]), round(o.p1[0]))):
        counters[op.type] = counters.get(op.type, 0) + 1
        op.id = f"{ {'door': 'D', 'window': 'W', 'passage': 'P'}[op.type] }{counters[op.type]}"
    _attach_host(openings, walls)
    return walls, openings, lines


def _attach_host(openings: list[Opening], walls: list[Wall]) -> None:
    if not walls:
        return
    polys = [Polygon(w.polygon) for w in walls]
    for op in openings:
        c = Polygon(op.polygon)
        best = min(range(len(walls)), key=lambda i: polys[i].distance(c))
        op.wall_id = walls[best].id


def wall_union(walls: list[Wall]):
    return unary_union([Polygon(w.polygon).buffer(0) for w in walls]) if walls else Polygon()

