"""Bitmap import (PNG, JPG, BMP, TIFF, WEBP, GIF and scanned PDFs).

Walls are recovered as the *thick* strokes of the drawing (morphological opening removes
thin lines such as dimension lines, door swings and window glazing).  The thin remainder
is kept as an ink mask that the opening classifier samples (door swing arcs, glazing).
If the walls are drawn hollow (outline only) the image is vectorised with a Hough transform
and processed like a CAD drawing.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..log import Log
from ..model import Drawing, ImportErrorUser, Seg, Settings, Text


def _load(path: Path) -> tuple[np.ndarray, Optional[float], Optional[np.ndarray]]:
    dpi = None
    data = np.fromfile(str(path), dtype=np.uint8)
    color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color is not None else None
    try:
        import pymupdf

        pix = pymupdf.Pixmap(str(path))
        if pix.xres and pix.xres not in (72, 96) and pix.xres > 0:
            dpi = float(pix.xres)
        if img is None:
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    except Exception:  # noqa: BLE001 - optional metadata
        pass
    if img is None:
        raise ImportErrorUser("Cannot decode image")
    return img, dpi, color


def _run_lengths(mask: np.ndarray, max_len: int = 60) -> np.ndarray:
    """Histogram of horizontal+vertical ink run lengths."""
    hist = np.zeros(max_len + 1, dtype=np.int64)
    for m in (mask, mask.T):
        m = m.astype(np.int8)
        pad = np.zeros((m.shape[0], 1), dtype=np.int8)
        dm = np.diff(np.hstack([pad, m, pad]), axis=1)
        starts = np.argwhere(dm == 1)
        ends = np.argwhere(dm == -1)
        lens = ends[:, 1] - starts[:, 1]
        lens = np.clip(lens, 0, max_len)
        hist += np.bincount(lens, minlength=max_len + 1)[: max_len + 1]
    return hist


class RasterCues:
    """Samples the thin-line ink mask for opening classification (coordinates in mm)."""

    def __init__(self, ink: np.ndarray, height: int):
        self.ink = ink
        self.h = height
        self.inv_fx = lambda x: x
        self.inv_fy = lambda y: y
        self.scale = 1.0  # mm per pixel

    def rebind(self, inv_fx, inv_fy, scale):
        self.inv_fx, self.inv_fy, self.scale = inv_fx, inv_fy, scale
        return self

    def _px(self, x, y):
        return self.inv_fx(x), self.h - self.inv_fy(y)

    def ink_ratio(self, pts) -> float:
        """Fraction of sample points (mm coords) lying on ink (with 1px tolerance)."""
        if len(pts) == 0:
            return 0.0
        hit = 0
        H, W = self.ink.shape
        for x, y in pts:
            c, r = self._px(x, y)
            ci, ri = int(round(c)), int(round(r))
            if 0 <= ri < H and 0 <= ci < W and self.ink[max(ri - 1, 0): ri + 2, max(ci - 1, 0): ci + 2].any():
                hit += 1
        return hit / len(pts)

    def door_swing(self, hinge, radius, closed_dir, normal) -> tuple[float, int, float]:
        """Score (0..1) for a door symbol hinged at ``hinge``: a quarter arc from the closed
        position toward the room AND the open leaf line along the normal (both must be drawn).
        Returns (score, side, inked share of the arc)."""
        best, best_side, best_ar = 0.0, 1, 0.0
        base = math.atan2(closed_dir[1], closed_dir[0])
        for side in (1, -1):
            n_ang = math.atan2(normal[1] * side, normal[0] * side)
            # sweep from closed direction toward the normal side
            delta = (n_ang - base + math.pi) % (2 * math.pi) - math.pi
            angs = np.linspace(base + 0.12 * delta, base + 0.88 * delta, 24)
            arc = [(hinge[0] + radius * math.cos(a), hinge[1] + radius * math.sin(a)) for a in angs]
            leaf = [(hinge[0] + radius * t * math.cos(n_ang), hinge[1] + radius * t * math.sin(n_ang))
                    for t in np.linspace(0.2, 0.9, 12)]
            # dashed swing arcs are common: a clearly drawn leaf lets a dashed arc count
            lr = self.ink_ratio(leaf)
            ar = self.ink_ratio(arc)
            sc = min(min(1.0, ar / (0.55 if lr >= 0.9 else 0.7)) if lr >= 0.8 else ar, lr)
            if sc > best:
                best, best_side, best_ar = sc, side, ar
        return best, best_side, best_ar

    def glazing(self, p1, p2, normal, depth) -> float:
        """Share of lines across the opening (parallel to the wall) that are inked."""
        best = 0.0
        for f in np.linspace(-0.4, 0.4, 17):
            ox, oy = normal[0] * depth * f, normal[1] * depth * f
            pts = [(p1[0] + (p2[0] - p1[0]) * t + ox, p1[1] + (p2[1] - p1[1]) * t + oy) for t in np.linspace(0.15, 0.85, 15)]
            best = max(best, self.ink_ratio(pts))
        return best


def dominant_angle(rings: list[np.ndarray]) -> float:
    """Main orientation of the drawing's edges, in degrees within [-45, 45)."""
    angs, lens = [], []
    for r in rings:
        d = np.diff(np.vstack([r, r[:1]]), axis=0)
        ln = np.hypot(d[:, 0], d[:, 1])
        ang = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 45) % 90 - 45  # fold to [-45, 45)
        angs.append(ang)
        lens.append(ln)
    if not angs:
        return 0.0
    ang, ln = np.concatenate(angs), np.concatenate(lens)
    hist, edges = np.histogram(ang, bins=90, range=(-45, 45), weights=ln)
    k = int(np.argmax(np.convolve(np.r_[hist[-1:], hist, hist[:1]], np.ones(3), "valid")))
    coarse = (edges[k] + edges[k + 1]) / 2
    near = np.abs((ang - coarse + 45) % 90 - 45) <= 1.5
    if not near.any():
        return 0.0
    theta = float(np.average(coarse + (ang[near] - coarse + 45) % 90 - 45, weights=ln[near]))
    return 0.0 if abs(theta) < 0.3 else theta


def rectilinear(pts: np.ndarray, theta0: float = 0.0, snap_deg: float = 10.0, min_edge: float = 1.6) -> list[tuple[float, float]]:
    """Regularise a traced polygon: edges within ``snap_deg`` of the drawing axes become exactly
    horizontal/vertical, pixel jaggies (steps shorter than ``min_edge``) are removed."""
    c, s_ = math.cos(math.radians(-theta0)), math.sin(math.radians(-theta0))
    P = [np.array([x * c - y * s_, x * s_ + y * c]) for x, y in pts]
    # 1. collapse very short edges
    changed = True
    while changed and len(P) > 4:
        changed = False
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if np.hypot(*(b - a)) < min_edge:
                P[i] = (a + b) / 2
                del P[(i + 1) % len(P)]
                changed = True
                break

    def cls(a, b):
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
        if min(ang, 180 - ang) <= snap_deg:
            return "H"
        if abs(ang - 90) <= snap_deg:
            return "V"
        return "F"

    # 2. merge consecutive edges of the same axis class
    changed = True
    while changed and len(P) > 4:
        changed = False
        n = len(P)
        for i in range(n):
            a, b, cc = P[i - 1], P[i], P[(i + 1) % n]
            k1, k2 = cls(a, b), cls(b, cc)
            if k1 == k2 and k1 != "F":
                del P[i]
                changed = True
                break
    n = len(P)
    if n < 3:
        return [(float(x), float(y)) for x, y in pts]
    # 3. each edge becomes a line; vertices are intersections of consecutive lines
    lines = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        k = cls(a, b)
        if k == "H":
            y = (a[1] + b[1]) / 2
            lines.append((np.array([a[0], y]), np.array([1.0, 0.0])))
        elif k == "V":
            x = (a[0] + b[0]) / 2
            lines.append((np.array([x, a[1]]), np.array([0.0, 1.0])))
        else:
            d = b - a
            lines.append((a, d / (np.hypot(*d) or 1)))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            q = P[i]
        else:
            t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
            q = p1 + t * d1
        out.append(q)
    c, s_ = math.cos(math.radians(theta0)), math.sin(math.radians(theta0))
    return [(float(x * c - y * s_), float(x * s_ + y * c)) for x, y in out]


def _line_kernel(angle_deg: float, length: int) -> np.ndarray:
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    dx, dy = math.cos(math.radians(angle_deg)) * c, -math.sin(math.radians(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def _clean(mask: np.ndarray, k: int, text_mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Drop small components and components lying inside OCR text boxes."""
    mask = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(k * k * 6, 0.00005 * H * W)
    if text_mask.any():
        in_text = np.bincount(lab[text_mask], minlength=n)
        keep &= ~(in_text >= 0.6 * np.maximum(stats[:, cv2.CC_STAT_AREA], 1))
    return keep[lab].astype(np.uint8)


def _drop_isolated_blobs(m: np.ndarray, ref: float) -> np.ndarray:
    """Keep the main wall network and separate parts that look like walls (elongated and
    reasonably long); compact isolated blobs are symbols (plants, icons, logos)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = np.zeros(n, bool)
    for j in range(1, n):
        w_, h_ = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
        keep[j] = j == main or (max(w_, h_) >= 3 * min(w_, h_) and max(w_, h_) >= 0.04 * ref) \
            or st[j, cv2.CC_STAT_AREA] >= 0.1 * st[main, cv2.CC_STAT_AREA]
    return keep[lab].astype(np.uint8)


def _wall_score(mask: np.ndarray, ref_extent: float) -> tuple[float, str]:
    """How much a mask looks like a building's walls: the area of its thin, elongated parts
    (walls) minus the area of filled blobs (furniture fills, room tints, symbols)."""
    if not mask.any():
        return 0.0, "empty"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    mean_dt = np.bincount(lab.ravel(), weights=dt.ravel(), minlength=n) / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
    good = bad = 0.0
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        thick = 4.0 * mean_dt[i]
        fill = area / max(w * h, 1)
        wall_like = fill <= 0.6 or max(w, h) >= 3 * min(w, h)
        if wall_like and 2 <= thick <= max(0.12 * max(w, h), 3):
            good += area
            boxes.append((x, y, x + w, y + h))
        else:
            bad += area
    if boxes:
        b = np.array(boxes)
        extent = max(b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min()) / max(ref_extent, 1)
    else:
        extent = 0.0
    info = f"{good / max(good + bad, 1):.0%} wall-like, spans {extent:.0%} of the drawing"
    score = (good - 0.5 * bad) * (1.0 if extent >= 0.4 else 0.2)
    return max(score, 0.0), info


def hatch_mask(inku: np.ndarray) -> Optional[np.ndarray]:
    """Walls filled with diagonal hatching (dense or sparse, single or crossed).

    Diagonal strokes mark a zone; inside it, the small paper cells enclosed by hatch lines
    and wall outlines are wall interior (room interiors are large cells and never count).
    The wall is the ink in the zone plus those cells; stray single lines are opened away."""
    # only thin line work can be hatching (any filled area contains "diagonal" runs)
    solid = cv2.morphologyEx(inku, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lines = inku & (cv2.dilate(solid, np.ones((3, 3), np.uint8)) == 0)
    # a thick axis-aligned stroke (window frame, wall outline) contains short slanted runs too
    lines = lines.astype(np.uint8)
    axis = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8)) \
        | cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((15, 1), np.uint8))
    slanted = lines & (axis == 0)
    diag = np.zeros(inku.shape, np.uint8)
    for ang in (25, 35, 45, 55, 65, 115, 125, 135, 145, 155):
        diag |= cv2.morphologyEx(slanted, cv2.MORPH_OPEN, _line_kernel(ang, 5))
    if diag.sum() < 200:
        return None
    zone = cv2.dilate(diag, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    bg = (inku == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, 4)
    inside = np.bincount(lab[zone], minlength=n)
    area = st[:, cv2.CC_STAT_AREA]
    cell = (area <= 600) & (inside >= 0.8 * area)
    cell[0] = False
    # a hatch cell is bordered mostly by diagonal strokes; cells between axis-aligned lines
    # (window frames, sills next to a hatched wall) are not wall
    lab_d = cv2.dilate(np.where(cell[lab], lab, 0).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    ring = (lab_d > 0) & (inku > 0)
    ring_all = np.bincount(lab_d[ring], minlength=n)
    ring_diag = np.bincount(lab_d[ring & (diag > 0)], minlength=n)
    cell &= ring_diag >= 0.25 * np.maximum(ring_all, 1)
    wall = cell[lab] | ((inku > 0) & zone)
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    wall = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    return wall


def hollow_mask(inku: np.ndarray, tmax: float, text_mask: Optional[np.ndarray] = None,
                tmin: float = 4.0) -> Optional[np.ndarray]:
    """Paper strips between two close parallel lines (outlined walls without fill).  Strips of
    walls meeting at junctions form one thin network, so shape is not restricted - only the
    width.  Strips holding text are the gaps between dimension lines, not walls."""
    bg = (inku == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, 4)
    dt = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
    maxdt = np.zeros(n)
    np.maximum.at(maxdt, lab.ravel(), dt.ravel())
    w, h, a = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    width = 2 * maxdt
    meandt = np.bincount(lab.ravel(), weights=dt.ravel(), minlength=n) / np.maximum(a, 1)
    strip = (width >= tmin) & (width <= tmax) & (2 * meandt <= 0.8 * tmax) & (np.maximum(w, h) >= 5 * np.maximum(width, 1))
    if text_mask is not None and text_mask.any():
        strip &= np.bincount(lab[text_mask], minlength=n) <= 0.02 * a
    strip[0] = False
    if not strip[1:].any():
        return None
    cells = strip[lab]
    wall = cells | ((inku > 0) & (cv2.dilate(cells.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _fills_opening(m: np.ndarray, box) -> bool:
    """A strip whose wall continues in line beyond both of its ends is the infill of a window
    or door opening (frame, glazing, sill) drawn in that wall - not a wall itself."""
    x, y, w, h = (int(v) for v in box[:4])
    t = min(w, h)
    L = max(2 * t, 6)
    if w >= h:
        a, b = m[y:y + h, max(0, x - L):x], m[y:y + h, x + w:x + w + L]
    else:
        a, b = m[max(0, y - L):y, x:x + w], m[y + h:y + h + L, x:x + w]
    return a.size > 0 and b.size > 0 and a.mean() >= 0.5 and b.mean() >= 0.5


def choose_wall_mask(img: np.ndarray, ink: np.ndarray, thin: int, text_mask: np.ndarray,
                     log: Log) -> tuple[np.ndarray, int, str]:
    """Walls are drawn in many styles.  Build one candidate mask per style and keep the one
    that looks most like a wall network:

    * ``black``: solid black walls, everything else grey or thin;
    * ``flat``: walls filled with one flat grey tone (lines and text are black);
    * ``hatch``: walls filled with diagonal hatching (single or crossed);
    * ``dark``: any thick dark stroke (generic fallback).
    """
    H, W = img.shape
    ys, xs = np.nonzero(ink)
    ref = max(xs.max() - xs.min(), ys.max() - ys.min()) if len(xs) else max(H, W)
    rect = lambda n: cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))  # noqa: E731
    cands: dict[str, tuple[np.ndarray, int]] = {}

    black = img <= 40
    if black.any():
        k = max(3, thin + 1)
        core = cv2.morphologyEx(black.astype(np.uint8), cv2.MORPH_OPEN, rect(k))
        # grow back over the anti-aliased edge up to mid-grey so thickness is unbiased
        cands["black"] = ((cv2.dilate(core, rect(3)) > 0) & (img < 128), k)

    hist = np.bincount(img.ravel() // 4, minlength=64)
    mid = hist.copy()
    mid[:11] = 0  # black
    mid[57:] = 0  # paper
    if mid.max() > 0.01 * img.size:
        tone = int(np.argmax(mid)) * 4 + 2
        k = max(3, thin + 2)
        core = cv2.morphologyEx((np.abs(img.astype(int) - tone) <= 10).astype(np.uint8), cv2.MORPH_OPEN, rect(k))
        cands["flat"] = ((cv2.dilate(core, rect(3)) > 0) & (img < (tone + 255) // 2), k)

    inku = ink.astype(np.uint8)
    # hatching is often drawn in light grey: detect it on a sensitive threshold
    bgv = float(np.median(img))
    hatch = hatch_mask(((img < 0.9 * bgv) | ink).astype(np.uint8))
    if hatch is not None:
        cands["hatch"] = (hatch, 3)

    k = max(3, 2 * thin + 1)
    cands["dark"] = (cv2.morphologyEx(inku, cv2.MORPH_OPEN, rect(k)), k)

    scored = {}
    for name, (m, kk) in cands.items():
        m = _clean(m, kk, text_mask, H, W)
        sc, info = _wall_score(m, ref)
        scored[name] = (sc, m, kk, info)
    best = max(v[0] for v in scored.values())
    for name in ("black", "flat", "hatch", "dark"):  # preference order among near-equal candidates
        if name in scored and scored[name][0] >= 0.8 * best:
            sc, m, kk, info = scored[name]
            break
    label = {"black": "solid black", "flat": "flat grey fill", "hatch": "hatching", "dark": "thick dark strokes"}[name]
    log.info(f"Walls recognised as {label} ({info}); "
             + ", ".join(f"{n}: {v[0] / max(best, 1):.0%}" for n, v in scored.items()))
    m = _drop_isolated_blobs(m, ref)
    # hollow walls (two parallel lines, nothing between) are not in any candidate: find them as
    # thin paper strips enclosed by lines, no thicker than the walls already found
    t_typ = 0.0
    # outlined (hollow) partitions accompany hatched structural walls; in drawings with solid
    # or grey-filled walls, double lines are railings, furniture or dimension gaps
    if m.any() and name == "hatch":
        dtm = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        t_typ = 4.0 * float(dtm[dtm > 0].mean())  # mean distance of a strip of width T is T/4
        hol = hollow_mask(inku, max(6.0, 1.3 * t_typ), text_mask, tmin=max(4.0, 0.3 * t_typ))
        if hol is not None:
            scored["hollow"] = (0.0, _clean(hol, 3, text_mask, H, W), 3, "")
    # plans often mix styles (grey exterior walls, hatched partitions): add wall-like parts of the
    # other styles that are joined to the chosen wall network
    added = 0
    for other in ("hatch", "black", "flat", "hollow"):
        if other == name or other not in scored or not scored[other][1].any():
            continue
        if not t_typ and m.any():
            dtm = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            t_typ = 4.0 * float(dtm[dtm > 0].mean())
        if other != "hollow" and scored[other][0] < 0.2 * best:
            continue  # this style is not really used in the drawing
        extra = (scored[other][1] > 0) & ~(m > 0)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)
        touch = cv2.dilate(m, rect(5)) > 0
        hit = np.bincount(lab2[touch & extra], minlength=n2)
        dt2 = cv2.distanceTransform(extra.astype(np.uint8), cv2.DIST_L2, 3)
        thick2 = 4.0 * np.bincount(lab2.ravel(), weights=dt2.ravel(), minlength=n2) / np.maximum(st2[:, cv2.CC_STAT_AREA], 1)
        for j in range(1, n2):
            w_, h_, a_ = st2[j, cv2.CC_STAT_WIDTH], st2[j, cv2.CC_STAT_HEIGHT], st2[j, cv2.CC_STAT_AREA]
            elongated = max(w_, h_) >= 4 * max(1, min(w_, h_)) or a_ < 0.5 * w_ * h_
            # thinner than a third of the walls: glazing, frames, sills - not a wall
            lo_t, hi_t = max(4.0, 0.35 * t_typ), (1.5 * t_typ if other == "hollow" else 1e9)
            if hit[j] > 0 and elongated and max(w_, h_) >= 8 * kk and lo_t <= thick2[j] <= hi_t \
                    and not _fills_opening(m, st2[j]):
                m[lab2 == j] = 1
                added += 1
    if added:
        log.info(f"Added {added} wall part(s) drawn in another style (e.g. hatched partitions)")
    return m, kk, name


def room_area_samples(texts: list[dict], walls: np.ndarray, log: Log) -> list[tuple[str, float, float]]:
    """Room area labels ("12,71 m²", "A: 11,10 m2") matched to the room they sit in.

    Rooms are the paper regions bounded by walls once door and window gaps are closed.
    Returns (label, area m², area px) for every labelled, closed room."""
    from ..analysis.numbers import parse_area_text, parse_dimension_text
    from .raster_dims import _area_label

    H, W = walls.shape
    ys, xs = np.nonzero(walls)
    if not len(xs):
        return []
    ref = max(xs.max() - xs.min(), ys.max() - ys.min())
    # close door and window gaps along the wall lines (horizontal / vertical line kernels up to
    # ~12 % of the plan): gaps are sealed without filling rooms, corners or corridors
    k = max(5, int(0.12 * ref))
    w8 = walls.astype(np.uint8)
    sealed = cv2.morphologyEx(w8, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)) | \
        cv2.morphologyEx(w8, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats((sealed == 0).astype(np.uint8), 4)
    border = np.zeros(n, bool)
    border[np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])] = True
    out = []
    for t in texts:
        a = parse_area_text(t["text"])
        if a is None:
            v, vmm = parse_dimension_text(t["text"])
            # "13,58 m" under a room name: the "²" was lost by OCR
            if vmm and 1000 <= vmm <= 200000 and _area_label(t, texts) and "," in t["text"]:
                a = v
        if a is None:
            continue
        j = lab[min(H - 1, int(t["cy"])), min(W - 1, int(t["cx"]))]
        if j == 0 or border[j]:
            continue
        out.append((t["text"], a, float(st[j, cv2.CC_STAT_AREA])))
    if out:
        log.info(f"Room area labels matched to {len(out)} closed room(s)")
    return out


def import_image(path: Path, settings: Settings, log: Log) -> Drawing:
    img, dpi, color = _load(path)
    if dpi:
        log.info(f"Image resolution metadata: {dpi:g} dpi")
    d = import_image_array(img, settings, log, dpi=dpi or settings.dpi, color=color)
    return d


def import_image_array(img: np.ndarray, settings: Settings, log: Log, dpi: Optional[float] = None,
                       color: Optional[np.ndarray] = None) -> Drawing:
    H, W = img.shape[:2]
    log.info(f"Raster image {W}×{H} px")
    if max(H, W) > 8000:
        f = 8000 / max(H, W)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if dpi:
            dpi *= f
        H, W = img.shape[:2]
        log.info(f"Downscaled to {W}×{H} px")
    if color is not None and color.shape[:2] != img.shape[:2]:
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_AREA)
    if color is not None and color.ndim == 3:
        # light, saturated colours (watermarks, room tints, colour-coded annotations in pastel)
        # are not drawing lines: treat them as paper.  Dark colours (red door swings) stay ink.
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        pale = (hsv[:, :, 1] > 70) & (img > 140)
        if pale.mean() > 0.001:
            # JPEG blurs the colour of the marks' edges: take the less saturated halo too
            near = cv2.dilate(pale.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            pale |= near & (hsv[:, :, 1] > 20) & (img > 140)
            img = img.copy()
            img[pale] = 255
            log.info(f"Ignoring light coloured marks ({pale.mean():.1%} of the image, e.g. watermarks)")
    if float(np.median(img)) < 128:
        img = 255 - img  # light-on-dark drawing -> dark-on-light
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    otsu, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink > 0
    hist = _run_lengths(ink)
    thin = int(np.argmax(hist[1:15]) + 1)
    # text first: OCR boxes are excluded from wall tracing (bold labels are as black as walls)
    from .raster_dims import find_dimensions, ocr

    texts = ocr(img, log)
    text_mask = np.zeros(img.shape, dtype=bool)
    for t in texts:
        if len(t["text"]) >= 2 or t["text"].isdigit():
            text_mask[max(0, int(t["y0"]) - 2):int(t["y1"]) + 3, max(0, int(t["x0"]) - 2):int(t["x1"]) + 3] = True
    walls, k, mode = choose_wall_mask(img, ink, thin, text_mask, log)
    wall_share = walls.sum() / max(ink.sum(), 1)

    d = Drawing(source_format="image", raster=True, pixel_size=1.0)
    fy = lambda r: H - r  # noqa: E731
    if wall_share > 0.08:
        log.info(f"Found solid wall strokes ({wall_share:.0%} of ink) - tracing wall outlines")
        dt = cv2.distanceTransform(walls, cv2.DIST_L2, 5)
        d.meta["wall_px_max"] = float(2 * np.percentile(dt[walls > 0], 97)) if walls.any() else 0.0
        contours, hier = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        eps = max(1.0, 0.6 * thin)
        approx = [cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float) for c in contours]
        theta0 = dominant_angle(approx)
        if abs(theta0) > 0.2:
            log.info(f"Drawing is rotated by {theta0:.1f}° - walls are snapped to that grid")

        def ring(c_idx):
            pts = rectilinear(approx[c_idx], theta0)
            return [(x + 0.5, fy(y + 0.5)) for x, y in pts]

        from shapely.geometry import Polygon

        fill_is_hole: list[bool] = []

        for i, c in enumerate(contours):
            if hier[0][i][3] != -1 or len(c) < 4:
                continue  # holes are handled with their parent
            outer = ring(i)
            holes = []
            j = hier[0][i][2]
            while j != -1:
                if len(contours[j]) >= 4:
                    holes.append(ring(j))
                j = hier[0][j][0]
            if len(outer) < 3:
                continue
            # the contour runs through boundary pixel centres - grow by half a pixel
            poly = Polygon(outer, [h for h in holes if len(h) >= 3]).buffer(0)
            poly = poly.buffer(0.5, join_style=2, mitre_limit=3.0)
            for g in getattr(poly, "geoms", [poly]):
                for hole, r in [(False, g.exterior)] + [(True, r) for r in g.interiors]:
                    pts = list(r.coords)
                    for a, b in zip(pts, pts[1:]):
                        d.segments.append(Seg(a[0], a[1], b[0], b[1], "WALL"))
                    d.fills.append(pts[:-1])
                    fill_is_hole.append(hole)
        d.meta["raster_mode"] = "solid"
        d.meta["fill_is_hole"] = fill_is_hole
    else:
        log.info("Walls appear to be drawn as outlines - vectorising lines (Hough transform)")
        skel = ink.astype(np.uint8) * 255
        lines = cv2.HoughLinesP(skel, 1, np.pi / 720, threshold=30, minLineLength=max(15, W // 100), maxLineGap=2)
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                d.segments.append(Seg(float(x1), fy(float(y1)), float(x2), fy(float(y2)), ""))
        d.meta["raster_mode"] = "outline"
    # thin lines (door swings, glazing) are often faint/anti-aliased: use a more sensitive threshold
    bg = float(np.median(img))
    faint = img < bg * 0.88
    residual = (ink | faint) & ~(cv2.dilate(walls, np.ones((3, 3), np.uint8)) > 0)
    # keep only line work: filled areas (room tints, furniture fills) are not door swings or glazing
    blobs = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    residual &= ~cv2.dilate(blobs.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    d.cue_provider = RasterCues(residual, H)
    # dimensions written on the bitmap (OCR)
    for t in texts:  # y up; used for room-name hints (terrace / balcony doors)
        d.texts.append(Text(t["cx"], H - t["cy"], t["text"], t["h"], 90.0 if t["vertical"] else 0.0, "OCR"))
    if texts:
        d.dims = find_dimensions(texts, (ink | faint) & ~walls.astype(bool), walls.astype(bool), log)
        d.meta["room_areas"] = room_area_samples(texts, walls, log)
    d.meta["image_size"] = (W, H)
    if dpi:
        d.meta["paper_unit_mm"] = 25.4 / dpi
        d.meta["is_paper"] = True
        d.meta["dpi"] = dpi
    log.info(f"Traced {len(d.segments)} wall outline segments")
    if not d.segments:
        raise ImportErrorUser("No walls found in the image")
    return d
