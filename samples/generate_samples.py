"""Generate sample floor plans with a known ground truth (used by tests and for demos).

Flat 10.00 × 7.00 m, exterior walls 300 mm, interior walls 120 mm,
4 doors and 4 windows.  One chain dimension is intentionally missing (1900 -> 5000 on
the bottom chain = 3100 mm) so the app must derive it.

    python samples/generate_samples.py            # writes into samples/
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import ezdxf
from shapely.geometry import box
from shapely.ops import unary_union

OUT = Path(__file__).resolve().parent

EXT_T = 300.0
INT_T = 120.0
W, H = 10000.0, 7000.0

WALL_RECTS = [
    (0, 0, W, EXT_T), (0, H - EXT_T, W, H), (0, 0, EXT_T, H), (W - EXT_T, 0, W, H),  # exterior
    (5000, EXT_T, 5120, H - EXT_T),  # interior vertical
    (5120, 3500, W - EXT_T, 3620),  # interior horizontal
]
# (type, x1, y1, x2, y2) opening rectangles; hinge side / swing for doors
OPENINGS = [
    ("door", 1000, 0, 1900, EXT_T),
    ("door", 5000, 1000, 5120, 1800),
    ("door", 6000, 3500, 6800, 3620),
    ("door", 5000, 4500, 5120, 5300),
    ("window", 1500, H - EXT_T, 3000, H),
    ("window", 6500, H - EXT_T, 8000, H),
    ("window", W - EXT_T, 1000, W, 2400),
    ("window", 0, 3500, EXT_T, 4700),
]
# dimension chains: (axis, line position, [coordinates...], skip_index)
DIMS = [
    ("x", -1500, [0, W], None),
    ("x", -800, [0, 1000, 1900, 5000, 5120, W], 2),  # 1900->5000 missing
    ("x", H + 800, [0, 1500, 3000, 6500, 8000, W], None),
    ("y", -1500, [0, H], None),
    ("y", -800, [0, 3500, 4700, H], None),
    ("y", W + 800, [0, 1000, 2400, 3500, 3620, H], None),
]


def truth() -> dict:
    walls = unary_union([box(*r) for r in WALL_RECTS])
    ops = unary_union([box(*o[1:]) for o in OPENINGS])
    solid = walls.difference(ops)
    return {
        "walls_area": solid.area,
        "wall_polygon": solid,
        "openings": [{"type": o[0], "rect": o[1:], "width": max(o[3] - o[1], o[4] - o[2])} for o in OPENINGS],
        "size": (W, H),
        "derived": 3100.0,
    }


def _door_arc(o):
    """Hinge at an inner-face jamb corner, swing into the room, radius = width."""
    _, x1, y1, x2, y2 = o
    if x2 - x1 > y2 - y1:  # horizontal wall
        w = x2 - x1
        hinge = (x1, y2)
        return hinge, w, 0.0, 90.0, (x1, y2 + w)
    w = y2 - y1
    hinge = (x2, y1)
    return hinge, w, 0.0, 90.0, (x2 + w, y1)


def _window_lines(o):
    _, x1, y1, x2, y2 = o
    if x2 - x1 > y2 - y1:
        ym = (y1 + y2) / 2
        return [((x1, y1), (x2, y1)), ((x1, y2), (x2, y2)), ((x1, ym), (x2, ym))]
    xm = (x1 + x2) / 2
    return [((x1, y1), (x1, y2)), ((x2, y1), (x2, y2)), ((xm, y1), (xm, y2))]


def _boundary_segments(poly):
    geoms = getattr(poly, "geoms", [poly])
    for g in geoms:
        for ring in [g.exterior, *g.interiors]:
            c = list(ring.coords)
            for a, b in zip(c, c[1:]):
                yield a, b


def write_dxf(path: Path, layers: bool = True, sx: float = 1.0, sy: float = 1.0, unit_div: float = 1.0,
              insunits: int = 4) -> None:
    """sx/sy distort the geometry (drawing not to scale); unit_div converts mm to drawing units."""
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = insunits
    msp = doc.modelspace()
    L = (lambda n: n) if layers else (lambda n: "0")
    for n in ("WALLS", "DOORS", "WINDOWS", "DIMS"):
        if layers:
            doc.layers.add(n)
    T = lambda p: (p[0] * sx / unit_div, p[1] * sy / unit_div)  # noqa: E731
    t = truth()
    for a, b in _boundary_segments(t["wall_polygon"]):
        msp.add_line(T(a), T(b), dxfattribs={"layer": L("WALLS")})
    for o in OPENINGS:
        if o[0] == "door":
            hinge, r, a0, a1, leaf = _door_arc(o)
            h = T(hinge)
            rr = r * (sx if o[3] - o[1] > o[4] - o[2] else sy) / unit_div
            msp.add_arc(h, rr, a0, a1, dxfattribs={"layer": L("DOORS")})
            msp.add_line(h, T(leaf), dxfattribs={"layer": L("DOORS")})
        else:
            for a, b in _window_lines(o):
                msp.add_line(T(a), T(b), dxfattribs={"layer": L("WINDOWS")})
    for axis, pos, coords, skip in DIMS:
        for i, (c1, c2) in enumerate(zip(coords, coords[1:])):
            if i == skip:
                continue
            if axis == "x":
                p1, p2, base, ang = (c1, 0), (c2, 0), (0, pos), 0
            else:
                p1, p2, base, ang = (0, c1), (0, c2), (pos, 0), 90
            text = f"{(c2 - c1) / unit_div:g}" if (sx != 1 or sy != 1 or unit_div != 1) else "<>"
            d = msp.add_linear_dim(base=T(base), p1=T(p1), p2=T(p2), angle=ang, text=text,
                                   dimstyle="EZ_MM_100_H25_CM" if unit_div == 1 else "EZDXF",
                                   dxfattribs={"layer": L("DIMS")})
            d.render()
    doc.saveas(path)


def write_dxf_blocks(path: Path) -> None:
    """CAD-style variant: walls as closed polylines + solid hatch on layer 'A-WALL',
    doors and windows as block references (rotated / mirrored), no dimensions,
    units in metres."""
    doc = ezdxf.new("R2013")
    doc.header["$INSUNITS"] = 6  # metres
    for n in ("A-WALL", "A-DOOR", "A-GLAZ", "FURNITURE"):
        doc.layers.add(n)
    door = doc.blocks.new("DOOR_SINGLE")
    door.add_arc((0, 0), 1.0, 0, 90, dxfattribs={"layer": "0"})
    door.add_line((0, 0), (0, 1.0), dxfattribs={"layer": "0"})
    win = doc.blocks.new("WINDOW_1")
    for y in (0.0, 0.5, 1.0):
        win.add_line((0, y), (1.0, y), dxfattribs={"layer": "0"})
    msp = doc.modelspace()
    t = truth()
    for g in getattr(t["wall_polygon"], "geoms", [t["wall_polygon"]]):
        rings = [g.exterior, *g.interiors]
        for r in rings:
            msp.add_lwpolyline([(x / 1000, y / 1000) for x, y in list(r.coords)[:-1]], close=True,
                               dxfattribs={"layer": "A-WALL"})
        h = msp.add_hatch(color=8, dxfattribs={"layer": "A-WALL"})
        h.paths.add_polyline_path([(x / 1000, y / 1000) for x, y in list(g.exterior.coords)[:-1]], is_closed=True, flags=1)
        for r in g.interiors:
            h.paths.add_polyline_path([(x / 1000, y / 1000) for x, y in list(r.coords)[:-1]], is_closed=True, flags=0)
    for typ, x1, y1, x2, y2 in OPENINGS:
        horiz = x2 - x1 > y2 - y1
        w = (x2 - x1 if horiz else y2 - y1) / 1000
        T = (y2 - y1 if horiz else x2 - x1) / 1000
        if typ == "door":
            if horiz:  # hinge at (x1, y2), swing up
                msp.add_blockref("DOOR_SINGLE", (x1 / 1000, y2 / 1000), dxfattribs={"layer": "A-DOOR", "xscale": w, "yscale": w})
            else:  # hinge at (x2, y2), mirrored + rotated
                msp.add_blockref("DOOR_SINGLE", (x2 / 1000, y2 / 1000),
                                 dxfattribs={"layer": "A-DOOR", "xscale": -w, "yscale": w, "rotation": 90})
        else:
            if horiz:
                msp.add_blockref("WINDOW_1", (x1 / 1000, y1 / 1000), dxfattribs={"layer": "A-GLAZ", "xscale": w, "yscale": T})
            else:
                msp.add_blockref("WINDOW_1", (x2 / 1000, y1 / 1000),
                                 dxfattribs={"layer": "A-GLAZ", "xscale": w, "yscale": T, "rotation": 90})
    # some furniture that must not become walls: a 600 mm deep kitchen counter and a bed
    msp.add_lwpolyline([(0.3, 0.3), (2.3, 0.3), (2.3, 0.9), (0.3, 0.9)], close=True, dxfattribs={"layer": "FURNITURE"})
    msp.add_lwpolyline([(6.0, 4.0), (7.6, 4.0), (7.6, 6.0), (6.0, 6.0)], close=True, dxfattribs={"layer": "FURNITURE"})
    doc.saveas(path)


def plan_svg(scale: float = 50.0, dims: bool = True) -> str:
    """SVG drawn at 1:scale in CSS px (96 dpi), filled walls, text dimensions."""
    k = 96 / 25.4 / scale  # px per real mm
    m = 2500.0
    Wp, Hp = (W + 2 * m) * k, (H + 2 * m) * k
    X = lambda x: (x + m) * k  # noqa: E731
    Y = lambda y: (H + m - y) * k  # noqa: E731
    t = truth()
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wp * 25.4 / 96:.4f}mm" height="{Hp * 25.4 / 96:.4f}mm" viewBox="0 0 {Wp:.3f} {Hp:.3f}">',
           f'<rect width="{Wp:.3f}" height="{Hp:.3f}" fill="white"/>']
    for g in getattr(t["wall_polygon"], "geoms", [t["wall_polygon"]]):
        d = "M" + " L".join(f"{X(x):.3f},{Y(y):.3f}" for x, y in g.exterior.coords) + "Z"
        for r in g.interiors:
            d += "M" + " L".join(f"{X(x):.3f},{Y(y):.3f}" for x, y in r.coords) + "Z"
        out.append(f'<path d="{d}" fill="black" fill-rule="evenodd" stroke="black" stroke-width="0.3"/>')
    for o in OPENINGS:
        if o[0] == "door":
            hinge, r, a0, a1, leaf = _door_arc(o)
            s = (hinge[0] + r * math.cos(math.radians(a0)), hinge[1] + r * math.sin(math.radians(a0)))
            e = (hinge[0] + r * math.cos(math.radians(a1)), hinge[1] + r * math.sin(math.radians(a1)))
            out.append(f'<line x1="{X(hinge[0]):.3f}" y1="{Y(hinge[1]):.3f}" x2="{X(leaf[0]):.3f}" y2="{Y(leaf[1]):.3f}" stroke="black" stroke-width="0.6"/>')
            out.append(f'<path d="M{X(s[0]):.3f},{Y(s[1]):.3f} A{r * k:.3f},{r * k:.3f} 0 0 0 {X(e[0]):.3f},{Y(e[1]):.3f}" fill="none" stroke="black" stroke-width="0.4"/>')
        else:
            for a, b in _window_lines(o):
                out.append(f'<line x1="{X(a[0]):.3f}" y1="{Y(a[1]):.3f}" x2="{X(b[0]):.3f}" y2="{Y(b[1]):.3f}" stroke="black" stroke-width="0.4"/>')
    if dims:
        for axis, pos, coords, skip in DIMS:
            # one continuous dimension line with extension ticks, numbers above each part
            if axis == "x":
                out.append(f'<line x1="{X(coords[0]):.3f}" y1="{Y(pos):.3f}" x2="{X(coords[-1]):.3f}" y2="{Y(pos):.3f}" stroke="black" stroke-width="0.3"/>')
                for c in coords:
                    out.append(f'<line x1="{X(c):.3f}" y1="{Y(pos - 150):.3f}" x2="{X(c):.3f}" y2="{Y(pos + 150):.3f}" stroke="black" stroke-width="0.3"/>')
            else:
                out.append(f'<line x1="{X(pos):.3f}" y1="{Y(coords[0]):.3f}" x2="{X(pos):.3f}" y2="{Y(coords[-1]):.3f}" stroke="black" stroke-width="0.3"/>')
                for c in coords:
                    out.append(f'<line x1="{X(pos - 150):.3f}" y1="{Y(c):.3f}" x2="{X(pos + 150):.3f}" y2="{Y(c):.3f}" stroke="black" stroke-width="0.3"/>')
            for i, (c1, c2) in enumerate(zip(coords, coords[1:])):
                if i == skip:
                    continue
                mid = (c1 + c2) / 2
                fs = 150 * k
                if axis == "x":
                    out.append(f'<text x="{X(mid):.3f}" y="{Y(pos + 60):.3f}" font-size="{fs:.3f}" text-anchor="middle" font-family="Helvetica">{c2 - c1:g}</text>')
                else:
                    out.append(f'<text x="{X(pos - 60):.3f}" y="{Y(mid):.3f}" font-size="{fs:.3f}" text-anchor="middle" font-family="Helvetica" '
                               f'transform="rotate(-90 {X(pos - 60):.3f} {Y(mid):.3f})">{c2 - c1:g}</text>')
    out.append("</svg>")
    return "\n".join(out)


def write_all(out: Path = OUT) -> dict[str, Path]:
    import pymupdf

    out.mkdir(parents=True, exist_ok=True)
    files = {}
    files["dxf"] = out / "sample_flat.dxf"
    write_dxf(files["dxf"])
    files["dxf_nolayers"] = out / "sample_flat_nolayers.dxf"
    write_dxf(files["dxf_nolayers"], layers=False)
    files["dxf_offscale_cm"] = out / "sample_flat_offscale_cm.dxf"
    write_dxf(files["dxf_offscale_cm"], layers=True, sx=1.03, sy=0.98, unit_div=10.0, insunits=0)
    files["dxf_blocks_m"] = out / "sample_flat_blocks_m.dxf"
    write_dxf_blocks(files["dxf_blocks_m"])
    files["svg"] = out / "sample_flat.svg"
    files["svg"].write_text(plan_svg(), encoding="utf-8")
    files["pdf"] = out / "sample_flat.pdf"
    files["pdf"].write_bytes(pymupdf.open(stream=plan_svg().encode(), filetype="svg").convert_to_pdf())
    files["png"] = out / "sample_flat.png"
    pdf = pymupdf.open("pdf", pymupdf.open(stream=plan_svg(dims=False).encode(), filetype="svg").convert_to_pdf())
    pix = pdf[0].get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    pix.set_dpi(200, 200)
    pix.save(str(files["png"]))
    # 3D model of the ground truth: metres, Y-up (like typical OBJ exports from Blender/SketchUp)
    t = truth()
    scene = _truth_scene(t["wall_polygon"])
    files["obj"] = out / "sample_flat.obj"
    txt = scene.export(file_type="obj")
    files["obj"].write_text(txt if isinstance(txt, str) else txt.decode())
    (out / "sample_truth.json").write_text(json.dumps({"size": t["size"], "openings": t["openings"],
                                                       "walls_area": t["walls_area"]}, indent=1))
    return files


def _truth_scene(solid):
    import trimesh

    Hh = 2700.0
    meshes = [trimesh.creation.extrude_polygon(g, Hh) for g in getattr(solid, "geoms", [solid])]
    for typ, x1, y1, x2, y2 in OPENINGS:
        poly = box(x1, y1, x2, y2)
        if typ == "window":
            meshes.append(trimesh.creation.extrude_polygon(poly, 900.0))
        top = trimesh.creation.extrude_polygon(poly, Hh - 2100.0)
        top.apply_translation([0, 0, 2100.0])
        meshes.append(top)
    m = trimesh.util.concatenate(meshes)
    m.apply_scale(0.001)
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    return trimesh.Scene(m)


if __name__ == "__main__":
    for k, v in write_all().items():
        print(f"{k:16s} {v}")
