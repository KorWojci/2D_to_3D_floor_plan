"""Core data structures shared by importers, analysis, 3D builder and exporters.

Two levels of representation are used:

* ``Drawing`` -- the raw, format-neutral content of an imported file (line segments,
  arcs, texts, dimension entities, block labels, filled regions) in the *source*
  coordinate units, Y axis pointing up.
* ``Plan`` -- the semantic result (walls, openings, dimensions) in millimetres.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

Point = tuple[float, float]


@dataclass
class Seg:
    x1: float
    y1: float
    x2: float
    y2: float
    layer: str = ""
    block: str = ""

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class Arc:
    cx: float
    cy: float
    r: float
    a0: float  # start angle, degrees, counter-clockwise
    a1: float  # end angle, degrees
    layer: str = ""
    block: str = ""

    @property
    def sweep(self) -> float:
        return (self.a1 - self.a0) % 360.0 or 360.0

    def point(self, ang_deg: float) -> Point:
        a = math.radians(ang_deg)
        return (self.cx + self.r * math.cos(a), self.cy + self.r * math.sin(a))


@dataclass
class Text:
    x: float  # insertion / centre point
    y: float
    text: str
    height: float = 0.0
    angle: float = 0.0  # degrees
    layer: str = ""


@dataclass
class DimEntity:
    """A dimension found in the source (DIMENSION entity or text next to a dim line)."""

    p1: Point  # first measured point (source units)
    p2: Point  # second measured point
    angle: float  # direction in which the distance is measured, degrees
    text: str = ""  # displayed text (may be empty -> measured value is shown)
    value: Optional[float] = None  # numeric displayed value (display units)
    value_mm: Optional[float] = None  # set if the text carried an explicit unit
    line_pos: Optional[Point] = None  # a point on the dimension line
    source: str = "entity"  # 'entity' | 'text'

    @property
    def geom_length(self) -> float:
        a = math.radians(self.angle)
        return abs((self.p2[0] - self.p1[0]) * math.cos(a) + (self.p2[1] - self.p1[1]) * math.sin(a))


@dataclass
class Label:
    """A named object (block reference, layer group) with its bounding box."""

    name: str
    layer: str
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    kind: str = ""  # 'door' | 'window' | '' (classified by name/layer keywords)


@dataclass
class Drawing:
    segments: list[Seg] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    dims: list[DimEntity] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    fills: list[list[Point]] = field(default_factory=list)  # filled closed regions
    unit_mm: Optional[float] = None  # millimetres per source unit, if known from the file
    unit_name: str = ""
    source_format: str = ""
    raster: bool = False  # geometry came from a bitmap (lower precision)
    pixel_size: float = 1.0  # size of one pixel in source units (raster only)
    meta: dict[str, Any] = field(default_factory=dict)
    # Optional helper supplied by importers that have extra knowledge about openings
    # (raster ink mask, 3D mesh sections).  Signature defined in analysis.cues.
    cue_provider: Any = None

    def bbox(self) -> Optional[tuple[float, float, float, float]]:
        xs: list[float] = []
        ys: list[float] = []
        for s in self.segments:
            xs += [s.x1, s.x2]
            ys += [s.y1, s.y2]
        for a in self.arcs:
            xs += [a.cx - a.r, a.cx + a.r]
            ys += [a.cy - a.r, a.cy + a.r]
        for f in self.fills:
            xs += [p[0] for p in f]
            ys += [p[1] for p in f]
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def transformed(self, fx: Callable[[float], float], fy: Callable[[float], float], scale: float) -> "Drawing":
        """Return a copy with every coordinate mapped (x->fx(x), y->fy(y)).

        ``scale`` is the average linear scale, used for radii and text heights.
        """
        d = Drawing(
            unit_mm=1.0,
            unit_name="mm",
            source_format=self.source_format,
            raster=self.raster,
            pixel_size=self.pixel_size * scale,
            meta=dict(self.meta),
            cue_provider=self.cue_provider,
        )
        d.segments = [Seg(fx(s.x1), fy(s.y1), fx(s.x2), fy(s.y2), s.layer, s.block) for s in self.segments]
        d.arcs = [Arc(fx(a.cx), fy(a.cy), a.r * scale, a.a0, a.a1, a.layer, a.block) for a in self.arcs]
        d.texts = [Text(fx(t.x), fy(t.y), t.text, t.height * scale, t.angle, t.layer) for t in self.texts]
        d.dims = [
            DimEntity(
                (fx(m.p1[0]), fy(m.p1[1])),
                (fx(m.p2[0]), fy(m.p2[1])),
                m.angle,
                m.text,
                m.value,
                m.value_mm,
                (fx(m.line_pos[0]), fy(m.line_pos[1])) if m.line_pos else None,
                m.source,
            )
            for m in self.dims
        ]
        d.labels = [
            Label(l.name, l.layer, (fx(l.bbox[0]), fy(l.bbox[1]), fx(l.bbox[2]), fy(l.bbox[3])), l.kind)
            for l in self.labels
        ]
        d.fills = [[(fx(x), fy(y)) for x, y in f] for f in self.fills]
        return d


# --------------------------------------------------------------------------- plan


@dataclass
class Wall:
    id: str
    p1: Point  # centre line start (mm)
    p2: Point  # centre line end
    thickness: float
    polygon: list[Point]  # footprint (rectangle for straight walls)

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])


@dataclass
class Opening:
    id: str
    type: str  # 'door' | 'window' | 'passage'
    p1: Point  # centre line start of the opening (mm)
    p2: Point  # centre line end
    depth: float  # = thickness of the host wall
    sill: float  # bottom of the recess above floor (0 for doors)
    head: float  # top of the recess above floor
    polygon: list[Point]
    wall_id: str = ""
    evidence: list[str] = field(default_factory=list)
    swing: Optional[dict[str, Any]] = None  # hinge, radius, start/end angles for 2D symbols

    @property
    def width(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])


@dataclass
class Dimension:
    kind: str  # 'wall_length' | 'wall_thickness' | 'opening_width' | 'chain' | 'overall'
    p1: Point
    p2: Point
    value: float  # mm
    source: str  # 'stated' (in file) | 'derived' (from other dims) | 'computed' (from geometry)
    ref: str = ""  # wall/opening id
    offset: float = 0.0  # suggested dimension-line offset for drawing (mm, signed, along normal)
    stated_text: str = ""
    deviation: float = 0.0  # stated value - geometry value (mm), when stated
    line_pos: Optional[Point] = None  # a point on the dimension line (source dimensions)


@dataclass
class Settings:
    wall_height: float = 2700.0
    door_height: float = 2100.0
    window_sill: float = 900.0
    window_head: float = 2100.0
    floor: bool = True  # floor placeholder under the whole flat
    floor_thickness: float = 0.0  # 0 -> flat surface at 0; >0 -> slab from -t to 0
    ceiling: bool = True  # ceiling placeholder at wall height
    ceiling_thickness: float = 0.0  # 0 -> flat surface at wall height; >0 -> slab above it
    min_wall_thickness: float = 60.0
    max_wall_thickness: float = 550.0
    min_opening: float = 350.0
    max_opening: float = 5000.0
    units: str = "auto"  # auto|mm|cm|m|in|ft
    paper_scale: Optional[float] = None  # N of 1:N for PDF/SVG/raster
    dpi: Optional[float] = None  # raster resolution (for paper_scale)
    known_width_mm: Optional[float] = None  # overall width of the walls' bounding box
    known_height_mm: Optional[float] = None
    wall_layers: str = ""  # comma separated substrings
    door_layers: str = ""
    window_layers: str = ""
    slice_height: float = 1300.0  # 3D import: section height above floor

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        s = cls()
        for k, v in (d or {}).items():
            if not hasattr(s, k) or v is None or v == "":
                continue
            cur = getattr(cls, k, None)
            try:
                if isinstance(cur, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, float) or k in ("paper_scale", "dpi", "known_width_mm", "known_height_mm"):
                    v = float(v)
                else:
                    v = str(v)
            except (TypeError, ValueError):
                continue
            setattr(s, k, v)
        return s


@dataclass
class Plan:
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
    wall_height: float = 2700.0
    calibration: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for w in self.walls for p in w.polygon] or [0.0]
        ys = [p[1] for w in self.walls for p in w.polygon] or [0.0]
        return min(xs), min(ys), max(xs), max(ys)

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "floorplan-json/1",
            "units": "mm",
            "wall_height": self.wall_height,
            "settings": asdict(self.settings),
            "walls": [asdict(w) for w in self.walls],
            "openings": [asdict(o) for o in self.openings],
            "dimensions": [asdict(d) for d in self.dimensions],
            "calibration": self.calibration,
            "warnings": self.warnings,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        def pt(p):
            return (float(p[0]), float(p[1]))

        p = cls()
        p.settings = Settings.from_dict(d.get("settings", {}))
        p.wall_height = float(d.get("wall_height", p.settings.wall_height))
        for w in d.get("walls", []):
            p.walls.append(Wall(w["id"], pt(w["p1"]), pt(w["p2"]), float(w["thickness"]), [pt(q) for q in w["polygon"]]))
        for o in d.get("openings", []):
            p.openings.append(
                Opening(
                    o["id"], o["type"], pt(o["p1"]), pt(o["p2"]), float(o["depth"]), float(o["sill"]),
                    float(o["head"]), [pt(q) for q in o["polygon"]], o.get("wall_id", ""),
                    list(o.get("evidence", [])), o.get("swing"),
                )
            )
        for m in d.get("dimensions", []):
            p.dimensions.append(
                Dimension(
                    m["kind"], pt(m["p1"]), pt(m["p2"]), float(m["value"]), m["source"], m.get("ref", ""),
                    float(m.get("offset", 0.0)), m.get("stated_text", ""), float(m.get("deviation", 0.0)),
                    pt(m["line_pos"]) if m.get("line_pos") else None,
                )
            )
        p.calibration = d.get("calibration", {})
        p.warnings = list(d.get("warnings", []))
        p.source = d.get("source", {})
        return p


class ImportErrorUser(Exception):
    """Raised for problems that should be shown to the user as-is."""
