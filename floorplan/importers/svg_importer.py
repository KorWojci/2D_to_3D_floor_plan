"""SVG import via svgelements (transforms, <use>, CSS units handled by the library)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import svgelements as se

from ..analysis.keywords import classify
from ..analysis.numbers import clean_text
from ..log import Log
from ..model import Arc, Drawing, ImportErrorUser, Label, Seg, Settings, Text
from .curves import fit_arc

PX_MM = 25.4 / 96.0  # svgelements normalises lengths to CSS px (96 dpi)


def _dark(color) -> bool:
    if color is None or color.value is None:
        return False
    try:
        if color.alpha == 0:
            return False
        return (color.red + color.green + color.blue) / 3 < 160
    except AttributeError:
        return False


def import_svg(path: Path, settings: Settings, log: Log) -> Drawing:
    try:
        svg = se.SVG.parse(str(path), reify=True, ppi=96.0)
    except Exception as e:  # noqa: BLE001
        raise ImportErrorUser(f"Cannot parse SVG: {e}") from e
    d = Drawing(source_format="svg")
    height = float(svg.height) if svg.height else 0.0
    flip = lambda y: height - y  # noqa: E731  (SVG y axis points down)

    def group_names(el) -> str:
        names = []
        cur = el
        while cur is not None:
            v = cur.values if hasattr(cur, "values") else {}
            for key in ("id", "class", "inkscape:label", "{http://www.inkscape.org/namespaces/inkscape}label"):
                if v.get(key):
                    names.append(str(v[key]))
            cur = getattr(cur, "parent", None)
        return " ".join(names)

    for el in svg.elements():
        if isinstance(el, se.Text):
            txt = clean_text(el.text or "")
            if not txt:
                continue
            m = el.transform if el.transform is not None else se.Matrix()
            h = float(el.font_size or 12.0) * math.sqrt(abs(m.determinant)) if m else float(el.font_size or 12.0)
            ang = -math.degrees(math.atan2(m.b, m.a)) if m else 0.0
            bb = el.bbox()
            if bb:
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            else:
                cx, cy = float(el.x or 0), float(el.y or 0)
            d.texts.append(Text(cx, flip(cy), txt, h, ang % 360, group_names(el)))
            continue
        if not isinstance(el, se.Shape):
            continue
        try:
            p = se.Path(el)
            p.reify()
        except Exception:  # noqa: BLE001
            continue
        if len(p) == 0:
            continue
        layer = group_names(el)
        filled = _dark(el.fill)
        kind = classify(layer, settings.door_layers, settings.window_layers)
        pts_all: list[tuple[float, float]] = []
        for sub in p.as_subpaths():
            sub = se.Path(sub)
            ring: list[tuple[float, float]] = []
            for segm in sub:
                if isinstance(segm, se.Move):
                    ring.append((segm.end.x, flip(segm.end.y)))
                    continue
                if isinstance(segm, se.Close):
                    if ring and segm.end is not None:
                        a, b = ring[-1], (segm.end.x, flip(segm.end.y))
                        if a != b:
                            d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
                    continue
                if isinstance(segm, se.Line):
                    a = (segm.start.x, flip(segm.start.y))
                    b = (segm.end.x, flip(segm.end.y))
                    d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
                    ring.append(b)
                    continue
                # curves: sample, try to recognise a circular arc (door swings)
                ts = np.linspace(0, 1, 17)
                pts = [(segm.point(t).x, flip(segm.point(t).y)) for t in ts]
                arc = fit_arc(pts)
                if arc is not None:
                    cx, cy, r, a0, a1 = arc
                    d.arcs.append(Arc(cx, cy, r, a0, a1, layer))
                else:
                    for a, b in zip(pts, pts[1:]):
                        d.segments.append(Seg(a[0], a[1], b[0], b[1], layer))
                ring.append(pts[-1])
            if filled and len(ring) >= 3:
                d.fills.append(ring)
            pts_all += ring
        if kind and pts_all:
            xs = [q[0] for q in pts_all]
            ys = [q[1] for q in pts_all]
            d.labels.append(Label(layer, layer, (min(xs), min(ys), max(xs), max(ys)), kind))

    # svgelements normalises every length to CSS px, so one unit is 1/96 inch on paper.
    # The real-world scale comes from dimensions, the paper scale (1:N) or known sizes.
    d.meta["paper_unit_mm"] = PX_MM
    d.meta["is_paper"] = True
    log.info(
        f"Read {len(d.segments)} segments, {len(d.arcs)} arcs, {len(d.texts)} texts, "
        f"{len(d.fills)} filled regions, {len(d.labels)} labelled groups"
    )
    if not d.segments and not d.fills:
        raise ImportErrorUser("SVG contains no vector geometry (embedded bitmaps are not traced - export PNG instead)")
    return d
