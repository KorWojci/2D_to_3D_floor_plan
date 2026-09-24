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
from ..model import Drawing, ImportErrorUser, Seg, Settings


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

    def door_swing(self, hinge, radius, closed_dir, normal) -> float:
        """Score (0..1) that a quarter circle centred at hinge is drawn on either side."""
        best = 0.0
        base = math.atan2(closed_dir[1], closed_dir[0])
        for side in (1, -1):
            n_ang = math.atan2(normal[1] * side, normal[0] * side)
            # sweep from closed direction toward the normal side
            delta = (n_ang - base + math.pi) % (2 * math.pi) - math.pi
            angs = np.linspace(base + 0.12 * delta, base + 0.88 * delta, 24)
            pts = [(hinge[0] + radius * math.cos(a), hinge[1] + radius * math.sin(a)) for a in angs]
            best = max(best, self.ink_ratio(pts))
        return best

    def glazing(self, p1, p2, normal, depth) -> float:
        """Share of lines across the opening (parallel to the wall) that are inked."""
        best = 0.0
        for f in np.linspace(-0.4, 0.4, 17):
            ox, oy = normal[0] * depth * f, normal[1] * depth * f
            pts = [(p1[0] + (p2[0] - p1[0]) * t + ox, p1[1] + (p2[1] - p1[1]) * t + oy) for t in np.linspace(0.15, 0.85, 15)]
            best = max(best, self.ink_ratio(pts))
        return best


def import_image(path: Path, settings: Settings, log: Log) -> Drawing:
    img, dpi = _load(path)
    if dpi:
        log.info(f"Image resolution metadata: {dpi:g} dpi")
    d = import_image_array(img, settings, log, dpi=dpi or settings.dpi)
    return d


def import_image_array(img: np.ndarray, settings: Settings, log: Log, dpi: Optional[float] = None) -> Drawing:
    H, W = img.shape[:2]
    log.info(f"Raster image {W}×{H} px")
    if max(H, W) > 8000:
        f = 8000 / max(H, W)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if dpi:
            dpi *= f
        H, W = img.shape[:2]
        log.info(f"Downscaled to {W}×{H} px")
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink > 0
    if ink.mean() > 0.5:
        ink = ~ink  # light-on-dark image
    hist = _run_lengths(ink)
    thin = int(np.argmax(hist[1:15]) + 1)
    k = max(3, 2 * thin + 1)
    log.info(f"Estimated thin stroke width {thin}px - wall strokes must be thicker than {k}px")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    walls = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(walls, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(k * k * 6, 0.00005 * H * W)
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

        def ring(c):
            ap = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)
            return [(x + 0.5, fy(y + 0.5)) for x, y in ap]

        from shapely.geometry import Polygon

        for i, c in enumerate(contours):
            if hier[0][i][3] != -1 or len(c) < 4:
                continue  # holes are handled with their parent
            outer = ring(c)
            holes = []
            j = hier[0][i][2]
            while j != -1:
                if len(contours[j]) >= 4:
                    holes.append(ring(contours[j]))
                j = hier[0][j][0]
            if len(outer) < 3:
                continue
            # the contour runs through boundary pixel centres - grow by half a pixel
            poly = Polygon(outer, [h for h in holes if len(h) >= 3]).buffer(0)
            poly = poly.buffer(0.5, join_style=2, mitre_limit=3.0)
            for g in getattr(poly, "geoms", [poly]):
                for r in [g.exterior, *g.interiors]:
                    pts = list(r.coords)
                    for a, b in zip(pts, pts[1:]):
                        d.segments.append(Seg(a[0], a[1], b[0], b[1], "WALL"))
                d.fills.append(list(g.exterior.coords)[:-1])
        d.meta["raster_mode"] = "solid"
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
    light_on_dark = bg < 128
    if light_on_dark:
        faint = img > bg + 0.25 * (255 - bg)
    else:
        faint = img < bg * 0.88
    residual = (ink | faint) & ~(cv2.dilate(walls, np.ones((3, 3), np.uint8)) > 0)
    d.cue_provider = RasterCues(residual, H)
    d.meta["image_size"] = (W, H)
    if dpi:
        d.meta["paper_unit_mm"] = 25.4 / dpi
        d.meta["is_paper"] = True
        d.meta["dpi"] = dpi
    log.info(f"Traced {len(d.segments)} wall outline segments")
    if not d.segments:
        raise ImportErrorUser("No walls found in the image")
    return d
