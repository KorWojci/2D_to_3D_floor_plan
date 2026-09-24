"""2D exports: DXF (clean layered drawing, mm), DWG (via converter), SVG, PDF, PNG, JSON."""
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable

import ezdxf
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .. import dwg
from ..log import Log
from ..model import Dimension, Opening, Plan

DIM_LAYERS = {"stated": "DIMENSIONS", "derived": "DIMENSIONS_DERIVED", "computed": "DIMENSIONS_COMPUTED"}


# ------------------------------------------------------------------ geometry helpers


def _wall_union(plan: Plan):
    return unary_union([Polygon(w.polygon).buffer(0) for w in plan.walls]) if plan.walls else Polygon()


def _polys(g) -> list[Polygon]:
    if isinstance(g, Polygon):
        return [] if g.is_empty else [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    return dx / L, dy / L


def door_symbol(o: Opening) -> tuple[tuple, tuple, float, float, float]:
    """Return (hinge, leaf_end, radius, start_angle, end_angle) of the door swing."""
    sw = o.swing or {}
    ux, uy = _unit(o.p1, o.p2)
    if sw:
        h = tuple(sw["hinge"])
        r = float(sw["r"])
        a0, a1 = float(sw["a0"]), float(sw["a1"])
    else:
        nx, ny = -uy, ux
        h = (o.p1[0] + nx * o.depth / 2, o.p1[1] + ny * o.depth / 2)
        r = o.width
        a0 = math.degrees(math.atan2(uy, ux)) % 360
        a1 = (a0 + 90) % 360
    # leaf = the arc end most perpendicular to the wall
    def perp(a):
        return abs(math.cos(math.radians(a)) * ux + math.sin(math.radians(a)) * uy)

    la = a0 if perp(a0) < perp(a1) else a1
    leaf = (h[0] + r * math.cos(math.radians(la)), h[1] + r * math.sin(math.radians(la)))
    return h, leaf, r, a0, a1


def _dim_geometry(d: Dimension):
    """Points of the dimension line (a, b), text position and angle."""
    ux, uy = _unit(d.p1, d.p2)
    nx, ny = -uy, ux
    off = d.offset
    a = (d.p1[0] + nx * off, d.p1[1] + ny * off)
    b = (d.p2[0] + nx * off, d.p2[1] + ny * off)
    ang = math.degrees(math.atan2(uy, ux))
    if ang > 90.0001 or ang <= -90:
        ang += 180
    return a, b, ang % 360, (nx, ny)


def _dims_for_drawing(plan: Plan, kinds: Iterable[str]) -> list[Dimension]:
    kinds = set(kinds)
    out = []
    for d in plan.dimensions:
        if d.kind == "wall_thickness":
            continue
        if d.source == "stated" and d.kind != "source":
            continue  # already drawn as the source dimension
        if d.source in kinds and d.value > 0.5:
            out.append(d)
    return out


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


# ------------------------------------------------------------------ DXF


def export_dxf(plan: Plan, path: Path, log: Log, version: str = "R2018") -> Path:
    doc = ezdxf.new(version, setup=version != "R2000")
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LUNITS"] = 2
    for name, color in [("WALLS", 7), ("WALLS_HATCH", 8), ("DOORS", 1), ("WINDOWS", 5), ("OPENINGS", 30),
                        ("DIMENSIONS", 2), ("DIMENSIONS_DERIVED", 3), ("DIMENSIONS_COMPUTED", 4), ("LABELS", 6)]:
        doc.layers.add(name, color=color)
    ds = doc.dimstyles.new("FLOORPLAN")
    ds.dxf.dimtxt = 100
    ds.dxf.dimasz = 60
    ds.dxf.dimtsz = 40  # architectural ticks
    ds.dxf.dimexe = 60
    ds.dxf.dimexo = 40
    ds.dxf.dimgap = 30
    ds.dxf.dimdec = 0
    ds.dxf.dimtad = 1
    ds.dxf.dimtih = 0
    ds.dxf.dimtoh = 0
    ds.dxf.dimlfac = 1.0
    msp = doc.modelspace()
    walls = _wall_union(plan)
    for p in _polys(walls):
        rings = [list(p.exterior.coords)] + [list(r.coords) for r in p.interiors]
        for r in rings:
            msp.add_lwpolyline([(x, y) for x, y in r[:-1]], close=True, dxfattribs={"layer": "WALLS"})
        h = msp.add_hatch(color=8, dxfattribs={"layer": "WALLS_HATCH"})
        h.paths.add_polyline_path([(x, y) for x, y in rings[0][:-1]], is_closed=True, flags=1)
        for r in rings[1:]:
            h.paths.add_polyline_path([(x, y) for x, y in r[:-1]], is_closed=True, flags=0)
    for o in plan.openings:
        msp.add_lwpolyline(o.polygon, close=True, dxfattribs={"layer": "OPENINGS", "linetype": "CONTINUOUS"})
        ux, uy = _unit(o.p1, o.p2)
        nx, ny = -uy, ux
        if o.type == "window":
            for f in (-0.5, -0.08, 0.08, 0.5):
                off = f * o.depth
                msp.add_line((o.p1[0] + nx * off, o.p1[1] + ny * off), (o.p2[0] + nx * off, o.p2[1] + ny * off),
                             dxfattribs={"layer": "WINDOWS"})
        elif o.type == "door":
            h, leaf, r, a0, a1 = door_symbol(o)
            msp.add_line(h, leaf, dxfattribs={"layer": "DOORS"})
            msp.add_arc(h, r, a0, a1, dxfattribs={"layer": "DOORS"})
        c = ((o.p1[0] + o.p2[0]) / 2, (o.p1[1] + o.p2[1]) / 2)
        ang = math.degrees(math.atan2(uy, ux))
        if ang > 90.0001 or ang <= -90:
            ang += 180
        label = f"{o.id} {_fmt(o.width)}x{_fmt(o.head - o.sill)}" + (f" sill {_fmt(o.sill)}" if o.sill > 0 else "")
        msp.add_text(label, height=80, rotation=ang, dxfattribs={"layer": "LABELS"}).set_placement(
            (c[0] - nx * (o.depth / 2 + 120), c[1] - ny * (o.depth / 2 + 120)), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    n = 0
    for d in _dims_for_drawing(plan, ("stated", "derived", "computed")):
        layer = DIM_LAYERS[d.source]
        if version == "R2000":  # plain geometry: most compatible with DWG converters
            _simple_dim(msp, d, layer)
            n += 1
            continue
        try:
            dim = msp.add_aligned_dim(p1=d.p1, p2=d.p2, distance=d.offset, dimstyle="FLOORPLAN",
                                      text=_fmt(d.value), dxfattribs={"layer": layer})
            dim.render()
            n += 1
        except Exception as e:  # noqa: BLE001
            log.warn(f"Could not write dimension {d.value:.0f}: {e}")
    doc.saveas(path)
    log.info(f"DXF {version} written ({len(plan.walls)} walls, {len(plan.openings)} openings, {n} dimensions, units mm)")
    return path


def _simple_dim(msp, d: Dimension, layer: str) -> None:
    a, b, ang, (nx, ny) = _dim_geometry(d)
    ux, uy = _unit(d.p1, d.p2)
    at = {"layer": layer}
    msp.add_line(a, b, dxfattribs=at)
    t = 40
    for p, q in ((d.p1, a), (d.p2, b)):
        if d.offset:
            msp.add_line(p, (q[0] + nx * 60 * (1 if d.offset > 0 else -1), q[1] + ny * 60 * (1 if d.offset > 0 else -1)), dxfattribs=at)
        msp.add_line((q[0] - (ux + nx) * t, q[1] - (uy + ny) * t), (q[0] + (ux + nx) * t, q[1] + (uy + ny) * t), dxfattribs=at)
    c = ((a[0] + b[0]) / 2 + nx * 40, (a[1] + b[1]) / 2 + ny * 40)
    msp.add_text(_fmt(d.value), height=100, rotation=ang, dxfattribs=at).set_placement(
        c, align=ezdxf.enums.TextEntityAlignment.BOTTOM_CENTER)


def export_dwg(plan: Plan, path: Path, log: Log) -> Path | None:
    if not dwg.available()["dwg_write"]:
        log.warn("DWG export needs ODA File Converter or LibreDWG on the server - "
                 "download the DXF instead (it opens in every DWG application) or install a converter (see README)")
        return None
    tmp = path.with_suffix(".tmp.dxf")
    # LibreDWG writes DWG R2000 and reads R2000 DXF most reliably; ODA converts any version
    export_dxf(plan, tmp, Log(), version="R2000" if dwg.converter_name() == "LibreDWG" else "R2018")
    ok = dwg.dxf_to_dwg(tmp, path, log)
    tmp.unlink(missing_ok=True)
    if ok and dwg.verify_dwg(path, log):
        log.info(f"DWG written with {dwg.converter_name()}")
        return path
    log.warn("DWG conversion failed - use the DXF file instead")
    path.unlink(missing_ok=True)
    return None


# ------------------------------------------------------------------ SVG / PDF / PNG


def plan_to_svg(plan: Plan, dims: Iterable[str] = ("stated", "derived", "computed"), paper_scale: float = 50.0,
                title: str = "") -> str:
    x0, y0, x1, y1 = plan.bbox()
    for d in plan.dimensions:
        for p in (d.p1, d.p2):
            x0, y0, x1, y1 = min(x0, p[0]), min(y0, p[1]), max(x1, p[0]), max(y1, p[1])
    m = 1500.0
    W, H = (x1 - x0) + 2 * m, (y1 - y0) + 2 * m

    def X(x):
        return x - x0 + m

    def Y(y):
        return y1 - y + m

    def pts(seq):
        return " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in seq)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W / paper_scale:.1f}mm" height="{H / paper_scale:.1f}mm" '
        f'viewBox="0 0 {W:.1f} {H:.1f}" font-family="Helvetica, Arial, sans-serif">',
        "<style>.wall{fill:#2b2b2b;stroke:#000;stroke-width:6}.op{fill:#fff;stroke:#666;stroke-width:5}"
        ".win{stroke:#2a7fd4;stroke-width:12;fill:none}.door{stroke:#c0392b;stroke-width:10;fill:none}"
        ".dim line{stroke-width:6}.dim text{font-size:110px;text-anchor:middle}"
        ".stated line{stroke:#333}.stated text{fill:#111}.derived line{stroke:#1e8449}.derived text{fill:#1e8449}"
        ".computed line{stroke:#2471a3;stroke-dasharray:30 20}.computed text{fill:#2471a3}"
        ".lbl{font-size:90px;fill:#555;text-anchor:middle}</style>",
        f'<rect x="0" y="0" width="{W:.1f}" height="{H:.1f}" fill="#fff"/>',
    ]
    if title:
        out.append(f'<text x="{m / 3:.0f}" y="{m / 2:.0f}" font-size="200">{html.escape(title)}</text>')
    walls = _wall_union(plan)
    out.append('<g id="walls">')
    for p in _polys(walls):
        d = "M" + pts(p.exterior.coords) + "Z" + "".join("M" + pts(r.coords) + "Z" for r in p.interiors)
        out.append(f'<path class="wall" fill-rule="evenodd" d="{d}"/>')
    out.append("</g><g id=\"openings\">")
    for o in plan.openings:
        out.append(f'<polygon class="op" points="{pts(o.polygon)}"/>')
        ux, uy = _unit(o.p1, o.p2)
        nx, ny = -uy, ux
        if o.type == "window":
            for f in (-0.08, 0.08):
                a = (o.p1[0] + nx * f * o.depth, o.p1[1] + ny * f * o.depth)
                b = (o.p2[0] + nx * f * o.depth, o.p2[1] + ny * f * o.depth)
                out.append(f'<line class="win" x1="{X(a[0]):.1f}" y1="{Y(a[1]):.1f}" x2="{X(b[0]):.1f}" y2="{Y(b[1]):.1f}"/>')
        elif o.type == "door":
            h, leaf, r, a0, a1 = door_symbol(o)
            s = (h[0] + r * math.cos(math.radians(a0)), h[1] + r * math.sin(math.radians(a0)))
            e = (h[0] + r * math.cos(math.radians(a1)), h[1] + r * math.sin(math.radians(a1)))
            large = 1 if ((a1 - a0) % 360) > 180 else 0
            out.append(f'<line class="door" x1="{X(h[0]):.1f}" y1="{Y(h[1]):.1f}" x2="{X(leaf[0]):.1f}" y2="{Y(leaf[1]):.1f}"/>')
            # SVG y axis is flipped -> CCW arc becomes sweep-flag 0
            out.append(f'<path class="door" d="M{X(s[0]):.1f},{Y(s[1]):.1f} A{r:.1f},{r:.1f} 0 {large} 0 {X(e[0]):.1f},{Y(e[1]):.1f}"/>')
        c = ((o.p1[0] + o.p2[0]) / 2 - nx * (o.depth / 2 + 150), (o.p1[1] + o.p2[1]) / 2 - ny * (o.depth / 2 + 150))
        ang = math.degrees(math.atan2(uy, ux))
        if ang > 90.0001 or ang <= -90:
            ang += 180
        out.append(f'<text class="lbl" x="{X(c[0]):.1f}" y="{Y(c[1]):.1f}" '
                   f'transform="rotate({-ang:.2f} {X(c[0]):.1f} {Y(c[1]):.1f})">{html.escape(o.id)}</text>')
    out.append("</g>")
    for src in ("stated", "derived", "computed"):
        if src not in dims:
            continue
        out.append(f'<g class="dim {src}" id="dims-{src}">')
        for d in _dims_for_drawing(plan, (src,)):
            a, b, ang, (nx, ny) = _dim_geometry(d)
            out.append(f'<line x1="{X(a[0]):.1f}" y1="{Y(a[1]):.1f}" x2="{X(b[0]):.1f}" y2="{Y(b[1]):.1f}"/>')
            for p, q in ((d.p1, a), (d.p2, b)):
                if d.offset:
                    out.append(f'<line x1="{X(p[0]):.1f}" y1="{Y(p[1]):.1f}" x2="{X(q[0]):.1f}" y2="{Y(q[1]):.1f}" stroke-width="3"/>')
                t = 45
                ux, uy = _unit(d.p1, d.p2)
                out.append(f'<line x1="{X(q[0] - (ux + nx) * t):.1f}" y1="{Y(q[1] - (uy + ny) * t):.1f}" '
                           f'x2="{X(q[0] + (ux + nx) * t):.1f}" y2="{Y(q[1] + (uy + ny) * t):.1f}"/>')
            c = ((a[0] + b[0]) / 2 + nx * 50, (a[1] + b[1]) / 2 + ny * 50)
            if d.offset < 0:
                c = ((a[0] + b[0]) / 2 - nx * 170, (a[1] + b[1]) / 2 - ny * 170)
            out.append(f'<text x="{X(c[0]):.1f}" y="{Y(c[1]):.1f}" transform="rotate({-ang:.2f} {X(c[0]):.1f} {Y(c[1]):.1f})">'
                       f"{_fmt(d.value)}</text>")
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def export_svg(plan: Plan, path: Path, log: Log) -> Path:
    path.write_text(plan_to_svg(plan), encoding="utf-8")
    log.info("SVG written (scale 1:50 when printed)")
    return path


def _svg_to_pdf_bytes(svg: str) -> bytes:
    import pymupdf

    src = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    return src.convert_to_pdf()


def export_pdf(plan: Plan, path: Path, log: Log) -> Path:
    path.write_bytes(_svg_to_pdf_bytes(plan_to_svg(plan)))
    log.info("PDF written (vector, scale 1:50)")
    return path


def export_png(plan: Plan, path: Path, log: Log) -> Path:
    import pymupdf

    pdf = pymupdf.open("pdf", _svg_to_pdf_bytes(plan_to_svg(plan)))
    page = pdf[0]
    zoom = min(4.0, 3000.0 / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pix.save(str(path))
    log.info(f"PNG written ({pix.width}×{pix.height} px)")
    return path


def export_json(plan: Plan, path: Path, log: Log) -> Path:
    path.write_text(plan.to_json(), encoding="utf-8")
    log.info("JSON plan written (walls, openings, dimensions in mm)")
    return path
