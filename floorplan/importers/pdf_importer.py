"""PDF import (vector drawings via PyMuPDF; scanned pages fall back to raster tracing)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pymupdf

from ..analysis.numbers import clean_text
from ..log import Log
from ..model import Arc, Drawing, ImportErrorUser, Seg, Settings, Text
from .curves import fit_arc, polyline_arc

PT_MM = 25.4 / 72.0
RASTER_DPI = 300


def _dark(color) -> bool:
    return color is not None and len(color) >= 3 and sum(color[:3]) / 3 < 0.63


def _bezier(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n + 1)[:, None]
    P = [np.array([p.x, p.y]) for p in (p0, p1, p2, p3)]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]


def import_pdf(path: Path, settings: Settings, log: Log) -> Drawing:
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:  # noqa: BLE001
        raise ImportErrorUser(f"Cannot open PDF: {e}") from e
    if doc.page_count == 0:
        raise ImportErrorUser("PDF has no pages")
    # choose the page with the most vector content
    best, best_n = 0, -1
    for i, page in enumerate(doc):
        n = len(page.get_drawings())
        if n > best_n:
            best, best_n = i, n
    page = doc[best]
    log.info(f"PDF: {doc.page_count} page(s); using page {best + 1} ({best_n} vector paths)")
    H = page.rect.height

    if best_n < 20:
        log.info("Page has (almost) no vector geometry - treating it as a scanned drawing")
        from .raster_importer import import_image_array

        pix = page.get_pixmap(dpi=RASTER_DPI, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        d = import_image_array(img, settings, log, dpi=RASTER_DPI)
        d.source_format = "pdf (raster)"
        return d

    d = Drawing(source_format="pdf")
    fy = lambda y: H - y  # noqa: E731

    for path_d in page.get_drawings():
        filled = _dark(path_d.get("fill"))
        layer = path_d.get("layer") or ""
        ring: list[tuple[float, float]] = []
        run: list[tuple[float, float]] = []

        def flush():
            if len(run) >= 2:
                arc = polyline_arc(run)
                if arc:
                    d.arcs.append(Arc(*arc, layer=layer))
                else:
                    for a, b in zip(run, run[1:]):
                        d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
            run.clear()

        for it in path_d["items"]:
            op = it[0]
            if op != "l":
                flush()
            if op == "l":
                a, b = (it[1].x, fy(it[1].y)), (it[2].x, fy(it[2].y))
                if run and (abs(run[-1][0] - a[0]) > 1e-6 or abs(run[-1][1] - a[1]) > 1e-6):
                    flush()
                if not run:
                    run.append(a)
                run.append(b)
                if not ring:
                    ring.append(a)
                ring.append(b)
            elif op == "re":
                r = it[1]
                q = [(r.x0, fy(r.y0)), (r.x1, fy(r.y0)), (r.x1, fy(r.y1)), (r.x0, fy(r.y1))]
                for a, b in zip(q, q[1:] + q[:1]):
                    d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
                if filled:
                    d.fills.append(q)
            elif op == "qu":
                qd = it[1]
                q = [(p.x, fy(p.y)) for p in (qd.ul, qd.ur, qd.lr, qd.ll)]
                for a, b in zip(q, q[1:] + q[:1]):
                    d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
                if filled:
                    d.fills.append(q)
            elif op == "c":
                pts = [(x, fy(y)) for x, y in _bezier(*it[1:5])]
                arc = fit_arc(pts)
                if arc:
                    d.arcs.append(Arc(*arc, layer=layer))
                else:
                    for a, b in zip(pts, pts[1:]):
                        d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
                if not ring:
                    ring.append(pts[0])
                ring.append(pts[-1])
        flush()
        if path_d.get("closePath") and len(ring) > 2:
            a, b = ring[-1], ring[0]
            d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
        if filled and len(ring) > 2:
            d.fills.append(ring)

    # consecutive Bézier quarter-circles of one door swing: merge arcs sharing centre/radius
    d.arcs = _merge_arcs(d.arcs)

    td = page.get_text("dict")
    for block in td.get("blocks", []):
        for line in block.get("lines", []):
            txt = clean_text("".join(s["text"] for s in line["spans"]))
            if not txt:
                continue
            x0, y0, x1, y1 = line["bbox"]
            dx, dy = line.get("dir", (1, 0))
            ang = math.degrees(math.atan2(-dy, dx)) % 360
            size = max(s["size"] for s in line["spans"])
            d.texts.append(Text((x0 + x1) / 2, fy((y0 + y1) / 2), txt, size, ang))
    d.meta["paper_unit_mm"] = PT_MM
    d.meta["is_paper"] = True
    log.info(f"Read {len(d.segments)} segments, {len(d.arcs)} arcs, {len(d.texts)} texts, {len(d.fills)} filled regions")
    return d


def _merge_arcs(arcs: list[Arc]) -> list[Arc]:
    out: list[Arc] = []
    for a in sorted(arcs, key=lambda a: (round(a.cx, 1), round(a.cy, 1), round(a.r, 1), a.a0)):
        if out:
            b = out[-1]
            if abs(a.cx - b.cx) < 0.02 * b.r and abs(a.cy - b.cy) < 0.02 * b.r and abs(a.r - b.r) < 0.02 * b.r:
                if abs((a.a0 - b.a1 + 180) % 360 - 180) < 2:
                    b.a1 = a.a1
                    continue
                if abs((b.a0 - a.a1 + 180) % 360 - 180) < 2:
                    b.a0 = a.a0
                    continue
        out.append(Arc(a.cx, a.cy, a.r, a.a0, a.a1, a.layer, a.block))
    return out
