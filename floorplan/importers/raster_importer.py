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


def _load(path: Path) -> tuple[np.ndarray, Optional[float]]:
    dpi = None
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
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
    return img, dpi


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

    def door_swing(self, hinge, radius, closed_dir, normal) -> tuple[float, int]:
        """Score (0..1) for a door symbol hinged at ``hinge``: a quarter arc from the closed
        position toward the room AND the open leaf line along the normal (both must be drawn)."""
        best, best_side = 0.0, 1
        base = math.atan2(closed_dir[1], closed_dir[0])
        for side in (1, -1):
            n_ang = math.atan2(normal[1] * side, normal[0] * side)
            # sweep from closed direction toward the normal side
            delta = (n_ang - base + math.pi) % (2 * math.pi) - math.pi
            angs = np.linspace(base + 0.12 * delta, base + 0.88 * delta, 24)
            arc = [(hinge[0] + radius * math.cos(a), hinge[1] + radius * math.sin(a)) for a in angs]
            leaf = [(hinge[0] + radius * t * math.cos(n_ang), hinge[1] + radius * t * math.sin(n_ang))
                    for t in np.linspace(0.2, 0.9, 12)]
            sc = min(self.ink_ratio(arc), self.ink_ratio(leaf))
            if sc > best:
                best, best_side = sc, side
        return best, best_side

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


def import_image(path: Path, settings: Settings, log: Log) -> Drawing:
    img, dpi = _load(path)
    if dpi:
        log.info(f"Image resolution metadata: {dpi:g} dpi")
    d = import_image_array(img, settings, log, dpi=dpi or settings.dpi)
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
    if float(np.median(img)) < 128:
        img = 255 - img  # light-on-dark drawing -> dark-on-light
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    otsu, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink > 0
    hist = _run_lengths(ink)
    thin = int(np.argmax(hist[1:15]) + 1)
    # text first: OCR boxes are excluded from wall tracing (bold labels are as black as walls)
    from .raster_dims import find_dimensions, ocr

    texts = ocr(color if color is not None else img, log)
    text_mask = np.zeros(img.shape, dtype=bool)
    for t in texts:
        if len(t["text"]) >= 2 or t["text"].isdigit():
            text_mask[max(0, int(t["y0"]) - 2):int(t["y1"]) + 3, max(0, int(t["x0"]) - 2):int(t["x1"]) + 3] = True
    # Many plans draw walls solid black and everything else (text, furniture, dimension
    # lines) in grey.  If such a black class exists it is the most reliable wall mask and
    # allows a smaller kernel, so thin partitions survive.
    black = img <= 40
    if black.sum() >= 0.25 * ink.sum() and black.any():
        k = max(3, thin + 1)
        core = cv2.morphologyEx(black.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        # grow back over the anti-aliased edge up to mid-grey so thickness is unbiased
        walls = (cv2.dilate(core, np.ones((3, 3), np.uint8)) > 0) & (img < 128)
        walls = walls.astype(np.uint8)
        log.info(f"Walls drawn solid black - strokes thicker than {k - 1}px are walls (thin lines {thin}px)")
    else:
        k = max(3, 2 * thin + 1)
        log.info(f"Estimated thin stroke width {thin}px - wall strokes must be thicker than {k}px")
        walls = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(walls, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(k * k * 6, 0.00005 * H * W)
    if text_mask.any():
        # bold letters are as black as walls: drop components lying (mostly) inside OCR boxes
        in_text = np.bincount(lab[text_mask], minlength=n)
        keep &= ~(in_text >= 0.6 * np.maximum(stats[:, cv2.CC_STAT_AREA], 1))
    walls = keep[lab].astype(np.uint8)
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
            for x1, y1, x2, y2 in lines[:, 0, :]:
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
    d.meta["image_size"] = (W, H)
    if dpi:
        d.meta["paper_unit_mm"] = 25.4 / dpi
        d.meta["is_paper"] = True
        d.meta["dpi"] = dpi
    log.info(f"Traced {len(d.segments)} wall outline segments")
    if not d.segments:
        raise ImportErrorUser("No walls found in the image")
    return d
