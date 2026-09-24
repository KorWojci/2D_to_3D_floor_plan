"""Scale / unit calibration driven by the dimensions written in the drawing.

The numbers written on a floor plan are treated as the truth.  The pipeline is:

1. collect dimensions: real DIMENSION entities and numeric texts sitting on dimension
   lines (for PDF/SVG/exploded CAD drawings);
2. ratio  r = stated value / drawn length  for every dimension -> robust median, outliers
   are reported and ignored;
3. the unit of the stated numbers (mm / cm / m / in / ft) is taken from explicit unit
   suffixes, the file header, or chosen by plausibility of the resulting flat size;
4. per axis, all horizontal (resp. vertical) dimensions form a system of equations
   ``x_j - x_i = value`` over their tick positions.  Least squares gives the true position
   of every tick; chains with one missing member are solved exactly (e.g. overall 10 000,
   parts 3 000 + ? + 4 000  ->  ? = 3 000).  A monotone piecewise-linear map from drawn
   coordinates to true coordinates is built from the ticks, so walls end up with exactly
   the stated lengths even when the drawing is not perfectly to scale;
5. without any dimensions the header units, a user-provided scale / known size, the paper
   scale (1:N) or - as last resort - a plausibility estimate is used (flagged in the log).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..log import Log
from ..model import DimEntity, Dimension, Drawing, Settings
from .numbers import parse_dimension_text

UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}
TYPICAL_FLAT_MM = 12000.0


@dataclass
class AxisMap:
    """Monotone piecewise-linear map raw -> display units with linear extrapolation."""

    slope: float
    raws: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vals: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fwd(self, u: float) -> float:
        if len(self.raws) < 2:
            if len(self.raws) == 1:
                return float(self.vals[0] + self.slope * (u - self.raws[0]))
            return self.slope * u
        if u <= self.raws[0]:
            return float(self.vals[0] + self.slope * (u - self.raws[0]))
        if u >= self.raws[-1]:
            return float(self.vals[-1] + self.slope * (u - self.raws[-1]))
        return float(np.interp(u, self.raws, self.vals))

    def inv(self, v: float) -> float:
        if len(self.raws) < 2:
            if len(self.raws) == 1:
                return float(self.raws[0] + (v - self.vals[0]) / self.slope)
            return v / self.slope
        if v <= self.vals[0]:
            return float(self.raws[0] + (v - self.vals[0]) / self.slope)
        if v >= self.vals[-1]:
            return float(self.raws[-1] + (v - self.vals[-1]) / self.slope)
        return float(np.interp(v, self.vals, self.raws))


@dataclass
class Calibration:
    k: float  # mm per display unit
    r: float  # display units per raw unit
    xmap: AxisMap
    ymap: AxisMap
    method: str
    confidence: str  # 'dimensions' | 'header' | 'user' | 'estimated'
    unit_label: str
    info: dict[str, Any] = field(default_factory=dict)
    dims: list[dict[str, Any]] = field(default_factory=list)  # processed source dimensions (mm)
    derived: list[Dimension] = field(default_factory=list)

    @property
    def scale(self) -> float:
        return self.k * self.r  # mm per raw unit (nominal)

    def fx(self, x: float) -> float:
        return self.k * self.xmap.fwd(x)

    def fy(self, y: float) -> float:
        return self.k * self.ymap.fwd(y)

    def inv_fx(self, x: float) -> float:
        return self.xmap.inv(x / self.k)

    def inv_fy(self, y: float) -> float:
        return self.ymap.inv(y / self.k)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "mm_per_drawing_unit": self.scale,
            "dimension_unit": self.unit_label,
            **self.info,
        }


# ------------------------------------------------------------------ text dimensions


def extract_text_dimensions(d: Drawing, log: Log) -> int:
    """Find numeric texts that sit on a dimension line and turn them into DimEntity."""
    cands = []
    for t in d.texts:
        v, vmm = parse_dimension_text(t.text)
        if v is not None:
            cands.append((t, v, vmm))
    if not cands or not d.segments:
        return 0
    S = np.array([[s.x1, s.y1, s.x2, s.y2] for s in d.segments], dtype=float)
    P1, P2 = S[:, :2], S[:, 2:]
    V = P2 - P1
    L = np.hypot(V[:, 0], V[:, 1])
    ok = L > 1e-9
    U = np.zeros_like(V)
    U[ok] = V[ok] / L[ok, None]
    ang = np.degrees(np.arctan2(U[:, 1], U[:, 0])) % 180
    bb = d.bbox()
    ext = max(bb[2] - bb[0], bb[3] - bb[1]) if bb else 1.0
    added = 0
    used_lines: set[tuple] = set()
    dim_segs: set[int] = set()
    for t, v, vmm in cands:
        h = t.height if t.height > 0 else ext * 0.01
        ta = t.angle % 180
        da = np.abs((ang - ta + 90) % 180 - 90)
        rel = np.array([t.x, t.y]) - P1
        s_along = rel[:, 0] * U[:, 0] + rel[:, 1] * U[:, 1]
        dist = np.abs(rel[:, 0] * U[:, 1] - rel[:, 1] * U[:, 0])
        m = ok & (da < 4) & (dist < 1.6 * h) & (s_along > -0.2 * h) & (s_along < L + 0.2 * h) & (L > 1.5 * h)
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            continue
        i = idx[np.argmin(dist[idx])]
        u = U[i]
        n = np.array([-u[1], u[0]])
        base = P1[i]
        # crossing (extension / tick) lines along this dimension line
        rel1 = P1 - base
        rel2 = P2 - base
        o1 = rel1 @ n
        o2 = rel2 @ n
        cross = ok & (np.abs(U @ u) < 0.5) & (np.minimum(o1, o2) <= 0.6 * h) & (np.maximum(o1, o2) >= -0.6 * h)
        cross[i] = False
        ticks = []
        for j in np.nonzero(cross)[0]:
            # intersection of segment j with the dim line (offset 0)
            if abs(o2[j] - o1[j]) < 1e-12:
                continue
            f = o1[j] / (o1[j] - o2[j])
            p = P1[j] + f * (P2[j] - P1[j])
            sj = float((p - base) @ u)
            if -0.3 * h <= sj <= L[i] + 0.3 * h:
                ticks.append(sj)
        st = float(s_along[i])
        left = [s for s in ticks if s < st - 0.3 * h]
        right = [s for s in ticks if s > st + 0.3 * h]
        a = max(left) if left else 0.0
        b = min(right) if right else float(L[i])
        if b - a < 0.25 * h:
            continue
        key = (i, round(a, 6), round(b, 6))
        if key in used_lines:
            continue
        used_lines.add(key)
        # mark the dimension line and its extension / tick lines so they never become walls
        dim_segs.add(int(i))
        for j in np.nonzero(cross)[0]:
            if min(abs(o1[j]), abs(o2[j])) <= 2.5 * h:
                dim_segs.add(int(j))
        p1 = base + a * u
        p2 = base + b * u
        d.dims.append(
            DimEntity((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), float(math.degrees(math.atan2(u[1], u[0])) % 180),
                      t.text, v, vmm, (float(t.x), float(t.y)), "text")
        )
        added += 1
    for k in dim_segs:
        d.segments[k].layer = "_dim_ " + d.segments[k].layer
    return added


# ------------------------------------------------------------------ helpers


def _axis_of(dim: DimEntity) -> Optional[str]:
    a = dim.angle % 180
    if min(a, 180 - a) < 1.0:
        return "x"
    if abs(a - 90) < 1.0:
        return "y"
    return None


def _plausibility(extent_mm: float) -> float:
    return abs(math.log(max(extent_mm, 1e-9) / TYPICAL_FLAT_MM))


def _choose_unit(values: list[float], extent_display: float, imperial: bool, log: Log) -> tuple[str, float]:
    cands = ["mm", "cm", "m"] + (["in", "ft"] if imperial else [])
    frac = [v for v in values if abs(v - round(v)) > 1e-6]
    if values and len(frac) >= 0.5 * len(values) and max(values) < 100:
        # "3.45" style numbers: metres (or feet on imperial drawings)
        cands = ["m"] + (["ft"] if imperial else [])
    ok = [(u, UNITS[u]) for u in cands if 1500 <= extent_display * UNITS[u] <= 200000]
    pool = ok or [(u, UNITS[u]) for u in cands]
    u, k = min(pool, key=lambda c: _plausibility(extent_display * c[1]))
    return u, k


def _consensus(ratios: np.ndarray, weights: np.ndarray, tol: float) -> float:
    """Ratio agreed on by most dimensions (weighted by length) - robust against mis-read
    numbers and room-area labels, unlike a plain median when many values are wrong."""
    best, best_w = float(np.median(ratios)), -1.0
    for r in ratios:
        inl = np.abs(ratios / r - 1) <= tol
        w = float(weights[inl].sum())
        if w > best_w:
            best_w, best = w, float(np.median(ratios[inl]))
    return best


def _area_scale(d: Drawing, log: Log) -> Optional[float]:
    """mm per pixel from room area labels (area written in the room / its pixel area)."""
    samples = d.meta.get("room_areas") or []
    vals = np.array([math.sqrt(a * 1e6 / px) for _, a, px in samples if px > 0])
    if len(vals) < 2:
        return None
    s = _consensus(vals, np.ones(len(vals)), 0.05)
    agree = int((np.abs(vals / s - 1) <= 0.05).sum())
    if agree < 2 or agree < 0.4 * len(vals):
        log.info(f"Room area labels do not agree on a scale ({len(vals)} room(s)) - not used")
        return None
    log.info(f"Room area labels: {agree} of {len(vals)} room(s) agree on {s:.4g} mm/px")
    return s


def _snap_k(k: float) -> Optional[tuple[str, float]]:
    for name, val in UNITS.items():
        if abs(k / val - 1) < 0.03:
            return name, val
    return None


# ------------------------------------------------------------------ main


def calibrate(d: Drawing, settings: Settings, log: Log, forced_scale: Optional[float] = None) -> Calibration:
    bb = d.bbox() or (0, 0, 1, 1)
    extent_raw = max(bb[2] - bb[0], bb[3] - bb[1], 1e-9)

    if forced_scale:  # from a known overall size (second pass)
        cal = Calibration(1.0, forced_scale, AxisMap(forced_scale), AxisMap(forced_scale), "known overall size", "user", "mm")
        cal.info["note"] = "scaled so that the walls' bounding box matches the size you entered"
        return cal

    # (bitmaps: dimension texts were already measured on the image by the OCR step)
    n_txt = extract_text_dimensions(d, log) if d.texts and not d.raster else 0
    if n_txt:
        log.info(f"Recognised {n_txt} dimension(s) written as text on dimension lines")
    dims = [m for m in d.dims if m.value and m.geom_length > extent_raw * 1e-4]
    log.info(f"{len(dims)} usable dimension(s) found in the drawing")

    user_unit = settings.units if settings.units in UNITS else None
    area_s = _area_scale(d, log) if d.raster else None

    if dims:
        # dimensions may be written in different units ("3,35 m" next to "335"): compare
        # explicit-unit values in mm and plain numbers in display units separately
        ratios = np.array([m.value / m.geom_length for m in dims])
        med = _consensus(ratios, np.array([m.geom_length for m in dims]), 0.04 if d.raster else 0.02)
        good = []
        rejected = []
        for m, rr in zip(dims, ratios):
            tol = 0.12 if m.source == "entity" else 0.03
            (good if abs(rr / med - 1) <= tol else rejected).append(m)
        for m in rejected:
            m.source += ":rejected"
        for m in rejected[:8]:
            log.warn(f"Dimension '{m.text}' disagrees with the drawing scale ({m.value / m.geom_length / med:.0%} of median) - ignored")
        if len(rejected) > 8:
            log.warn(f"... and {len(rejected) - 8} more inconsistent dimension(s) ignored")
        r = float(np.median([m.value / m.geom_length for m in good]))
        spread = float(np.max(np.abs(np.array([m.value / m.geom_length for m in good]) / r - 1))) if good else 0.0
        explicit = [m.value_mm / m.value for m in good if m.value_mm]
        if user_unit:
            unit, k = user_unit, UNITS[user_unit]
            how = "unit chosen by user"
        elif explicit:
            k = float(np.median(explicit))
            snap = _snap_k(k)
            unit, k = snap if snap else ("custom", k)
            how = "explicit unit in dimension text"
        elif d.unit_mm and _snap_k(d.unit_mm / r):
            unit, k = _snap_k(d.unit_mm / r)
            how = f"file header ({d.unit_name})"
        else:
            imperial = d.meta.get("measurement") == 0 or any("'" in m.text or '"' in m.text for m in good)
            unit, k = _choose_unit([m.value for m in good], extent_raw * r, imperial, log)
            how = "plausible flat size"
        log.info(f"Dimension values are in {unit} ({how}); drawing scale {k * r:.6g} mm per drawing unit")
        if spread > 0.005:
            log.info(f"Drawing is not exactly to scale (dimension/drawn ratios vary up to {spread:.1%}) - "
                     "geometry will be adjusted to the written dimensions")
        if area_s and (len(good) < 3 or abs(k * r / area_s - 1) > 0.06):
            log.warn(f"Scale from dimensions ({k * r:.4g} mm/px, {len(good)} dimension(s)) disagrees with the room "
                     f"areas written in the plan ({area_s:.4g} mm/px) - using the room areas")
            return Calibration(1.0, area_s, AxisMap(area_s), AxisMap(area_s), "room area labels", "dimensions", "mm")
        if area_s:
            log.info(f"Room area labels confirm the scale ({area_s:.4g} mm/px, {abs(k * r / area_s - 1):.1%} difference)")
        cal = Calibration(k, r, AxisMap(r), AxisMap(r), "dimensions", "dimensions", unit)
        cal.info["dimensions_used"] = len(good)
        # other scales supported by a group of dimensions (checked later against room areas)
        w_all = np.array([m.geom_length for m in dims])
        alts: list[float] = []
        for rr in sorted(set(np.round(ratios, 6)), key=lambda v: -w_all[np.abs(ratios / v - 1) <= 0.04].sum()):
            inl = np.abs(ratios / rr - 1) <= 0.04
            if inl.sum() >= 1 and abs(rr / r - 1) > 0.1 and all(abs(rr / a - 1) > 0.1 for a in alts) \
                    and w_all[inl].sum() >= 0.2 * w_all[np.abs(ratios / r - 1) <= 0.04].sum():
                alts.append(float(np.median(ratios[inl])))
        cal.info["alt_scales_mm"] = [round(k * a, 6) for a in alts[:3]]
        cal.info["dimensions_rejected"] = len(rejected)
        _fit_axes(cal, good, extent_raw, d.raster, log)
        return cal

    # ---- no dimensions
    if area_s and not settings.paper_scale:
        return Calibration(1.0, area_s, AxisMap(area_s), AxisMap(area_s), "room area labels", "dimensions", "mm")
    if user_unit and not d.meta.get("is_paper"):
        k = UNITS[user_unit]
        log.info(f"No dimensions - using drawing units chosen by user: {user_unit}")
        return Calibration(k, 1.0, AxisMap(1.0), AxisMap(1.0), "user units", "user", user_unit)
    if d.unit_mm and not d.meta.get("is_paper"):
        log.info(f"No dimensions - using file units ({d.unit_name or 'header'}): {d.unit_mm:g} mm per unit")
        if 1500 <= extent_raw * d.unit_mm <= 300000:
            return Calibration(1.0, d.unit_mm, AxisMap(d.unit_mm), AxisMap(d.unit_mm), "file units", "header", "mm")
        log.warn(f"Header units give an implausible size ({extent_raw * d.unit_mm / 1000:.1f} m) - estimating instead")
    if d.meta.get("is_paper"):
        pu = d.meta.get("paper_unit_mm", 1.0)
        if settings.paper_scale:
            s = pu * settings.paper_scale
            log.info(f"No dimensions - using paper scale 1:{settings.paper_scale:g} ({s:.5g} mm per unit)")
            return Calibration(1.0, s, AxisMap(s), AxisMap(s), f"paper scale 1:{settings.paper_scale:g}", "user", "mm")
        if not d.raster or d.meta.get("dpi"):
            best = min((10, 20, 25, 50, 75, 100, 200, 250, 500), key=lambda n: _plausibility(extent_raw * pu * n))
            s = pu * best
            log.warn(f"No dimensions and no scale given - ASSUMING paper scale 1:{best} (flat ≈ {extent_raw * s / 1000:.1f} m). "
                     "Enter the drawing scale or a known size for exact results.")
            return Calibration(1.0, s, AxisMap(s), AxisMap(s), f"assumed paper scale 1:{best}", "estimated", "mm")
    if d.raster:
        wpx = d.meta.get("wall_px_max") or 0
        if wpx > 2:
            s = 300.0 / wpx
            log.warn(f"No dimensions and no scale given - ASSUMING the thickest wall is 300 mm ({s:.3g} mm/px). "
                     "Enter a known overall size or the scan's scale for exact results.")
            return Calibration(1.0, s, AxisMap(s), AxisMap(s), "assumed exterior wall 300 mm", "estimated", "mm")
    best = min(UNITS.items(), key=lambda u: _plausibility(extent_raw * u[1]))
    log.warn(f"No units or dimensions found - ASSUMING drawing units are {best[0]} "
             f"(flat ≈ {extent_raw * best[1] / 1000:.1f} m). Set the units for exact results.")
    return Calibration(best[1], 1.0, AxisMap(1.0), AxisMap(1.0), f"assumed units {best[0]}", "estimated", best[0])


def _fit_axes(cal: Calibration, dims: list[DimEntity], extent_raw: float, raster: bool, log: Log) -> None:
    tol = extent_raw * (2e-3 if raster else 1e-4)
    r = cal.r
    for axis in ("x", "y"):
        ad = [m for m in dims if _axis_of(m) == axis]
        amap = cal.xmap if axis == "x" else cal.ymap
        ci = 0 if axis == "x" else 1
        if not ad:
            continue
        # --- cluster tick coordinates
        coords = sorted({round(c, 9) for m in ad for c in (m.p1[ci], m.p2[ci])})
        nodes: list[list[float]] = []
        for c in coords:
            if nodes and c - nodes[-1][-1] <= tol:
                nodes[-1].append(c)
            else:
                nodes.append([c])
        centers = np.array([float(np.mean(n)) for n in nodes])

        def node_of(c: float) -> int:
            return int(np.argmin(np.abs(centers - c)))

        edges = []
        for m in ad:
            i, j = node_of(m.p1[ci]), node_of(m.p2[ci])
            if i == j:
                continue
            if centers[i] > centers[j]:
                i, j = j, i
            edges.append((i, j, float(m.value), m))
        if not edges:
            continue
        # --- connected components (union find)
        parent = list(range(len(centers)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j, _, _ in edges:
            parent[find(i)] = find(j)
        comps: dict[int, list[int]] = {}
        for n in range(len(centers)):
            if any(n in (i, j) for i, j, _, _ in edges):
                comps.setdefault(find(n), []).append(n)
        raws: list[float] = []
        vals: list[float] = []
        comp_of: dict[int, int] = {}
        solved: dict[int, float] = {}
        for cid, members in comps.items():
            idx = {n: q for q, n in enumerate(members)}
            ce = [e for e in edges if e[0] in idx]
            A = np.zeros((len(ce) + 1, len(members)))
            b = np.zeros(len(ce) + 1)
            for q, (i, j, v, _) in enumerate(ce):
                A[q, idx[j]] = 1
                A[q, idx[i]] = -1
                b[q] = v
            A[-1, 0] = 1  # anchor
            b[-1] = r * centers[members[0]]
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            res = A[:-1] @ x - b[:-1]
            # align the component with the global scale (least squares offset)
            off = float(np.mean(r * centers[members] - x))
            x = x + off
            bad = [(ce[q][3], res[q]) for q in range(len(ce)) if abs(res[q]) * cal.k > 2.0]
            for m, rv in bad[:5]:
                log.warn(f"Dimension '{m.text}' conflicts with other {axis}-dimensions by {abs(rv) * cal.k:.1f} mm "
                         "(chain does not add up) - using best fit")
            for n in members:
                solved[n] = float(x[idx[n]])
                comp_of[n] = cid
        order = sorted(solved)
        raws = [float(centers[n]) for n in order]
        vals = [solved[n] for n in order]
        if raster:
            # bitmap measurements are only pixel-accurate: a piecewise map would amplify the
            # ±1 px noise, so fit one scale per axis (long dimensions weigh most)
            lens = np.array([abs(centers[j] - centers[i]) for i, j, _, _ in edges])
            vs = np.array([v for _, _, v, _ in edges])
            amap.slope = float((vs * lens).sum() / (lens * lens).sum())
            resid = np.abs(vs - amap.slope * lens) * cal.k
            log.info(f"{axis.upper()}-axis: {len(edges)} dimension(s), scale {amap.slope * cal.k:.4g} mm/px, "
                     f"largest residual {resid.max():.0f} mm")
        elif any(v2 <= v1 for v1, v2 in zip(vals, vals[1:])):
            log.warn(f"{axis}-dimensions are contradictory (non-monotone) - using uniform scale on this axis")
            continue
        else:
            amap.raws = np.array(raws)
            amap.vals = np.array(vals)
            worst_local = 0.0
            for (u1, v1), (u2, v2) in zip(zip(raws, vals), zip(raws[1:], vals[1:])):
                worst_local = max(worst_local, abs((v2 - v1) - r * (u2 - u1)) * cal.k)
            log.info(f"{axis.upper()}-axis: {len(ad)} dimension(s), {len(raws)} reference lines; "
                     f"largest correction between neighbouring lines {worst_local:.1f} mm")
            cal.info[f"{axis}_ticks"] = len(raws)
            cal.info[f"{axis}_max_correction_mm"] = round(worst_local, 2)
        # --- processed dims & derived (missing) chain members
        direct = {(i, j) for i, j, _, _ in edges}
        lines: dict[tuple[int, int], set[int]] = {}
        for i, j, v, m in edges:
            pos = m.line_pos[1 - ci] if m.line_pos else (m.p1[1 - ci] + m.p2[1 - ci]) / 2
            key = (comp_of[i], round(pos / max(tol * 50, 1e-9)))
            lines.setdefault(key, set()).update((i, j))
            cal.dims.append({"axis": axis, "i": i, "j": j, "value_disp": v, "text": m.text, "source": m.source,
                             "line_pos_raw": pos, "line": key})
        for key, nodeset in lines.items():
            seq = sorted(nodeset, key=lambda n: centers[n])
            linepos = float(np.mean([dd["line_pos_raw"] for dd in cal.dims if dd["axis"] == axis and dd["line"] == key]))
            for a, b in zip(seq, seq[1:]):
                if (a, b) in direct:
                    continue
                val = (solved[b] - solved[a]) * cal.k
                ua, ub = float(centers[a]), float(centers[b])
                cal.derived.append(
                    Dimension("chain", _pt(axis, ua, linepos), _pt(axis, ub, linepos), val, "derived",
                              stated_text=f"{(solved[b] - solved[a]):g}")
                )
        cal.info.setdefault("derived_raw", 0)
        cal.info["derived_raw"] = len(cal.derived)


def _pt(axis: str, along: float, across: float) -> tuple[float, float]:
    return (along, across) if axis == "x" else (across, along)


def finalize_derived(cal: Calibration, fx=None, fy=None) -> list[Dimension]:
    """Map derived chain dimensions (raw coords) to mm."""
    fx, fy = fx or cal.fx, fy or cal.fy
    out = []
    for dm in cal.derived:
        p1 = (fx(dm.p1[0]), fy(dm.p1[1]))
        p2 = (fx(dm.p2[0]), fy(dm.p2[1]))
        out.append(Dimension("chain", p1, p2, round(dm.value, 2), "derived", stated_text=dm.stated_text))
    return out


def stated_dims_mm(cal: Calibration, d: Drawing, fx=None, fy=None) -> list[dict[str, Any]]:
    """Every source dimension mapped to mm (for reports / matching).

    Points are projected onto the dimension line so they can be redrawn as-is."""
    fx, fy = fx or cal.fx, fy or cal.fy
    out = []
    for m in d.dims:
        if not m.value or m.source.endswith(":rejected"):
            continue
        p1 = (fx(m.p1[0]), fy(m.p1[1]))
        p2 = (fx(m.p2[0]), fy(m.p2[1]))
        lp = (fx(m.line_pos[0]), fy(m.line_pos[1])) if m.line_pos else None
        if lp is not None:
            ua = math.radians(m.angle)
            u = (math.cos(ua), math.sin(ua))
            p1 = (lp[0] + u[0] * ((p1[0] - lp[0]) * u[0] + (p1[1] - lp[1]) * u[1]),
                  lp[1] + u[1] * ((p1[0] - lp[0]) * u[0] + (p1[1] - lp[1]) * u[1]))
            p2 = (lp[0] + u[0] * ((p2[0] - lp[0]) * u[0] + (p2[1] - lp[1]) * u[1]),
                  lp[1] + u[1] * ((p2[0] - lp[0]) * u[0] + (p2[1] - lp[1]) * u[1]))
        stated = m.value_mm if m.value_mm else m.value * cal.k
        a = math.radians(m.angle)
        geom = abs((p2[0] - p1[0]) * math.cos(a) + (p2[1] - p1[1]) * math.sin(a))
        out.append({"p1": p1, "p2": p2, "angle": m.angle, "stated_mm": stated, "geom_mm": geom, "text": m.text,
                    "line_pos": lp, "source": m.source})
    return out


Transform = Callable[[float], float]
