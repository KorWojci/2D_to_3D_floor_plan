"""DXF import (all versions supported by ezdxf).  DWG is converted to DXF first."""
from __future__ import annotations

import math
from pathlib import Path

import ezdxf
from ezdxf import path as ezpath
from ezdxf.math import bulge_to_arc

from ..analysis.keywords import classify
from ..analysis.numbers import clean_text, parse_dimension_text
from ..log import Log
from ..model import Arc, DimEntity, Drawing, ImportErrorUser, Label, Seg, Settings, Text

INSUNITS_MM = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 8: 25.4e-6, 9: 0.0254, 10: 914.4, 14: 100.0}
INSUNITS_NAME = {1: "in", 2: "ft", 4: "mm", 5: "cm", 6: "m", 8: "µin", 9: "mil", 10: "yd", 14: "dm"}
FLATTEN_TOL = 1e-3  # relative to drawing extent, see _flatten


def _read(path: Path, log: Log):
    try:
        return ezdxf.readfile(str(path))
    except ezdxf.DXFStructureError:
        log.warn("DXF structure error - trying recovery mode")
        from ezdxf import recover

        doc, auditor = recover.readfile(str(path))
        if auditor.has_errors:
            log.warn(f"Recovered DXF with {len(auditor.errors)} unfixable error(s)")
        return doc
    except IOError as e:
        raise ImportErrorUser(f"Cannot read DXF file: {e}") from e


class _Collector:
    def __init__(self, drawing: Drawing, settings: Settings, log: Log, flat_dist: float):
        self.d = drawing
        self.s = settings
        self.log = log
        self.flat = flat_dist
        self.skipped: dict[str, int] = {}
        self.hidden_layers: set[str] = set()

    # ---------------------------------------------------------------- helpers
    def seg(self, a, b, layer, block):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1e-9:
            self.d.segments.append(Seg(float(a[0]), float(a[1]), float(b[0]), float(b[1]), layer, block))

    def polyline_points(self, pts, layer, block, closed=False):
        for a, b in zip(pts, pts[1:]):
            self.seg(a, b, layer, block)
        if closed and len(pts) > 2:
            self.seg(pts[-1], pts[0], layer, block)

    def flatten(self, e, layer, block):
        try:
            p = ezpath.make_path(e)
        except (TypeError, ValueError):
            self.skipped[e.dxftype()] = self.skipped.get(e.dxftype(), 0) + 1
            return
        pts = [(v.x, v.y) for v in p.flattening(self.flat)]
        self.polyline_points(pts, layer, block)

    # ---------------------------------------------------------------- entities
    def add(self, e, block: str = "", depth: int = 0):
        t = e.dxftype()
        layer = e.dxf.get("layer", "0")
        if layer in self.hidden_layers:
            return
        if t == "LINE":
            self.seg(e.dxf.start, e.dxf.end, layer, block)
        elif t == "LWPOLYLINE":
            self.lwpolyline(e, layer, block)
        elif t == "POLYLINE":
            if e.is_2d_polyline:
                self.flatten(e, layer, block)
            else:
                self.skipped["POLYLINE(3D/mesh)"] = self.skipped.get("POLYLINE(3D/mesh)", 0) + 1
        elif t == "ARC":
            c = e.dxf.center
            sx = 1.0
            if e.dxf.get("extrusion", (0, 0, 1))[2] < 0:  # mirrored arc (OCS z = -1)
                sx = -1.0
                a0, a1 = 180 - e.dxf.end_angle, 180 - e.dxf.start_angle
            else:
                a0, a1 = e.dxf.start_angle, e.dxf.end_angle
            self.d.arcs.append(Arc(sx * c[0], c[1], e.dxf.radius, a0 % 360, a1 % 360, layer, block))
        elif t == "CIRCLE":
            pass  # columns / symbols - not used for walls
        elif t in ("ELLIPSE", "SPLINE"):
            self.flatten(e, layer, block)
        elif t in ("TEXT", "ATTRIB"):
            self.text(e, layer)
        elif t == "MTEXT":
            self.mtext(e, layer)
        elif t == "DIMENSION":
            self.dimension(e, layer)
        elif t == "INSERT":
            self.insert(e, block, depth)
        elif t == "HATCH":
            self.hatch(e, layer, block)
        elif t in ("SOLID", "TRACE"):
            pts = [e.dxf.get(f"vtx{i}") for i in range(4)]
            pts = [(p[0], p[1]) for p in pts if p is not None]
            if len(pts) == 4:
                pts = [pts[0], pts[1], pts[3], pts[2]]  # SOLID vertex order is Z-shaped
            self.d.fills.append(pts)
            self.polyline_points(pts, layer, block, closed=True)
        elif t in ("MLINE",):
            for ve in e.virtual_entities():
                self.add(ve, block, depth + 1)
        elif t in ("3DFACE", "MESH", "3DSOLID", "BODY", "REGION", "SURFACE"):
            self.skipped[t] = self.skipped.get(t, 0) + 1
        elif t in ("POINT", "VIEWPORT", "LEADER", "MULTILEADER", "IMAGE", "WIPEOUT", "XLINE", "RAY", "OLE2FRAME", "TABLE"):
            pass
        else:
            self.skipped[t] = self.skipped.get(t, 0) + 1

    def lwpolyline(self, e, layer, block):
        pts = list(e.get_points("xyb"))
        if not pts:
            return
        closed = e.closed
        width = e.dxf.get("const_width", 0.0) or 0.0
        n = len(pts)
        rng = range(n if closed else n - 1)
        for i in rng:
            x1, y1, b = pts[i]
            x2, y2, _ = pts[(i + 1) % n]
            if abs(b) > 1e-9:
                c, sa, ea, r = bulge_to_arc((x1, y1), (x2, y2), b)
                self.d.arcs.append(Arc(c.x, c.y, r, math.degrees(sa) % 360, math.degrees(ea) % 360, layer, block))
            elif width > 0:
                # a wide polyline is a wall drawn as a centre line - emit its outline
                dx, dy = x2 - x1, y2 - y1
                ln = math.hypot(dx, dy)
                if ln < 1e-9:
                    continue
                nx, ny = -dy / ln * width / 2, dx / ln * width / 2
                quad = [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]
                self.polyline_points(quad, layer, block, closed=True)
                self.d.fills.append(quad)
            else:
                self.seg((x1, y1), (x2, y2), layer, block)

    def _text_centre(self, ins, h, ang, n_chars, h_off=0.0, v_off=0.0):
        w = 0.6 * h * max(n_chars, 1)
        a = math.radians(ang)
        ux, uy = math.cos(a), math.sin(a)
        vx, vy = -uy, ux
        # h_off: 0 = insertion at left, 0.5 = centre, 1 = right; v_off: 0 bottom, .5 middle, 1 top
        cx = ins[0] + ux * w * (0.5 - h_off) + vx * h * (0.5 - v_off)
        cy = ins[1] + uy * w * (0.5 - h_off) + vy * h * (0.5 - v_off)
        return cx, cy

    def text(self, e, layer):
        txt = clean_text(e.dxf.get("text", ""))
        if not txt:
            return
        h = e.dxf.get("height", 1.0)
        ang = e.dxf.get("rotation", 0.0)
        halign = e.dxf.get("halign", 0)
        valign = e.dxf.get("valign", 0)
        if halign or valign:
            ins = e.dxf.get("align_point", e.dxf.insert)
            h_off = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.5}.get(halign, 0.0)
            v_off = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}.get(valign, 0.0)
            if halign == 4:
                v_off = 0.5
        else:
            ins, h_off, v_off = e.dxf.insert, 0.0, 0.0
        cx, cy = self._text_centre(ins, h, ang, len(txt), h_off, v_off)
        self.d.texts.append(Text(cx, cy, txt, h, ang % 360, layer))

    def mtext(self, e, layer):
        txt = clean_text(e.plain_text())
        if not txt:
            return
        h = e.dxf.get("char_height", 1.0)
        ang = e.get_rotation() if hasattr(e, "get_rotation") else e.dxf.get("rotation", 0.0)
        ap = e.dxf.get("attachment_point", 1)
        h_off = {0: 0.0, 1: 0.5, 2: 1.0}[(ap - 1) % 3]
        v_off = {0: 1.0, 1: 0.5, 2: 0.0}[(ap - 1) // 3]
        cx, cy = self._text_centre(e.dxf.insert, h, ang, len(txt), h_off, v_off)
        self.d.texts.append(Text(cx, cy, txt, h, ang % 360, layer))

    def dimension(self, e, layer):
        dt = e.dimtype & 7
        if dt not in (0, 1):
            return  # only linear / aligned dimensions carry lengths
        p1 = e.dxf.get("defpoint2")
        p2 = e.dxf.get("defpoint3")
        if p1 is None or p2 is None:
            return
        if dt == 0:
            ang = e.dxf.get("angle", 0.0)
        else:
            ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        try:
            meas = float(e.get_measurement())
        except Exception:  # noqa: BLE001 - ezdxf raises various errors for odd dims
            a = math.radians(ang)
            meas = abs((p2[0] - p1[0]) * math.cos(a) + (p2[1] - p1[1]) * math.sin(a))
        dimlfac = 1.0
        try:
            from ezdxf.entities import DimStyleOverride

            dimlfac = float(DimStyleOverride(e).get("dimlfac", 1.0) or 1.0)
        except Exception:  # noqa: BLE001
            pass
        raw = clean_text(e.dxf.get("text", ""))
        if raw in ("", "<>"):
            value, value_mm = meas * dimlfac, None
            text = f"{value:g}"
        elif raw.strip() == " ":
            return  # suppressed text
        else:
            text = raw.replace("<>", f"{meas * dimlfac:g}")
            value, value_mm = parse_dimension_text(text)
            if value is None:
                return
        self.d.dims.append(
            DimEntity((p1[0], p1[1]), (p2[0], p2[1]), ang % 180, text, value, value_mm,
                      (e.dxf.defpoint[0], e.dxf.defpoint[1]), "entity")
        )

    def hatch(self, e, layer, block):
        try:
            paths = ezpath.from_hatch(e)
        except Exception:  # noqa: BLE001
            return
        solid = bool(e.dxf.get("solid_fill", 0))
        for p in paths:
            pts = [(v.x, v.y) for v in p.flattening(self.flat)]
            if len(pts) < 3:
                continue
            if solid:
                self.d.fills.append(pts)
            self.polyline_points(pts, layer, block, closed=True)

    def insert(self, e, parent_block, depth):
        if depth > 8:
            return
        name = e.dxf.name
        block = parent_block or name
        n0s, n0a, n0f = len(self.d.segments), len(self.d.arcs), len(self.d.fills)
        try:
            ves = list(e.virtual_entities())
        except Exception as ex:  # noqa: BLE001
            self.log.warn(f"Cannot explode block '{name}': {ex}")
            return
        for ve in ves:
            ve_layer = ve.dxf.get("layer", "0")
            if ve_layer == "0":
                ve.dxf.layer = e.dxf.get("layer", "0")  # layer 0 inherits the INSERT's layer
            self.add(ve, block, depth + 1)
        if parent_block:
            return
        xs, ys = [], []
        for s in self.d.segments[n0s:]:
            xs += [s.x1, s.x2]
            ys += [s.y1, s.y2]
        for a in self.d.arcs[n0a:]:
            xs += [a.cx - a.r, a.cx + a.r]
            ys += [a.cy - a.r, a.cy + a.r]
        for f in self.d.fills[n0f:]:
            xs += [p[0] for p in f]
            ys += [p[1] for p in f]
        if xs:
            kind = classify(name, self.s.door_layers, self.s.window_layers) or classify(
                e.dxf.get("layer", ""), self.s.door_layers, self.s.window_layers
            )
            self.d.labels.append(Label(name, e.dxf.get("layer", "0"), (min(xs), min(ys), max(xs), max(ys)), kind))


def import_dxf(path: Path, settings: Settings, log: Log, source_format: str = "dxf") -> Drawing:
    doc = _read(path, log)
    log.info(f"DXF version {doc.dxfversion}, {len(doc.modelspace())} model-space entities")
    d = Drawing(source_format=source_format)
    insunits = doc.header.get("$INSUNITS", 0)
    if insunits in INSUNITS_MM:
        d.unit_mm = INSUNITS_MM[insunits]
        d.unit_name = INSUNITS_NAME[insunits]
        log.info(f"Drawing units from header ($INSUNITS={insunits}): {d.unit_name}")
    else:
        log.info("Drawing has no unit in header ($INSUNITS=0) - units will be inferred")
    d.meta["measurement"] = doc.header.get("$MEASUREMENT", None)

    try:
        from ezdxf import bbox as ezbbox

        ext = ezbbox.extents(doc.modelspace(), fast=True)
        size = max(ext.size.x, ext.size.y) if ext.has_data else 1000.0
    except Exception:  # noqa: BLE001
        size = 1000.0
    col = _Collector(d, settings, log, max(size * 2e-4, 1e-6))
    for layer in doc.layers:
        try:
            if layer.is_off() or layer.is_frozen():
                col.hidden_layers.add(layer.dxf.name)
        except AttributeError:
            pass
    if col.hidden_layers:
        log.info(f"Ignoring {len(col.hidden_layers)} hidden/frozen layer(s)")
    for e in doc.modelspace():
        col.add(e)
    if col.skipped:
        log.info("Skipped entity types: " + ", ".join(f"{k}×{v}" for k, v in sorted(col.skipped.items())))
    log.info(
        f"Read {len(d.segments)} segments, {len(d.arcs)} arcs, {len(d.texts)} texts, "
        f"{len(d.dims)} dimensions, {len(d.labels)} block references, {len(d.fills)} filled regions"
    )
    if not d.segments and any(k in col.skipped for k in ("3DFACE", "MESH", "3DSOLID")):
        raise ImportErrorUser("This DXF contains only 3D solids/meshes - export it as OBJ/STL/IFC and import that instead")
    return d
