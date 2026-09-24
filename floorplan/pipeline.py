"""End-to-end conversion: import -> calibrate -> detect -> dimensions -> 3D -> export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import exporters, importers
from .analysis import dimreport
from .analysis.calibration import Calibration, calibrate, stated_dims_mm
from .analysis.walls import detect
from .builder3d import build_scene
from .log import Log
from .model import Drawing, ImportErrorUser, Plan, Settings

DEFAULT_FORMATS = ("dxf", "dwg", "svg", "pdf", "glb", "obj", "stl", "ifc", "json")


def _apply(d: Drawing, fx, fy, ifx, ify, scale) -> Drawing:
    dm = d.transformed(fx, fy, scale)
    if dm.cue_provider is not None and hasattr(dm.cue_provider, "rebind"):
        dm.cue_provider.rebind(ifx, ify, scale)
    return dm


def _size_factors(walls, settings: Settings) -> tuple[float, float]:
    xs = [p[0] for w in walls for p in w.polygon]
    ys = [p[1] for w in walls for p in w.polygon]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    fw = settings.known_width_mm / bw if settings.known_width_mm else None
    fh = settings.known_height_mm / bh if settings.known_height_mm else None
    return (fw or fh), (fh or fw)


def _compose(fx, fy, ifx, ify, fw, fh):
    return (lambda x: fw * fx(x), lambda y: fh * fy(y), lambda x: ifx(x / fw), lambda y: ify(y / fh))


def _scale_result(walls, openings, fw: float, fh: float) -> None:
    """Scale detected geometry exactly (about the origin)."""
    import math

    def P(p):
        return (p[0] * fw, p[1] * fh)

    for w in walls:
        ux, uy = (w.p2[0] - w.p1[0]) / (w.length or 1), (w.p2[1] - w.p1[1]) / (w.length or 1)
        w.thickness = round(w.thickness * math.hypot(uy * fw, ux * fh), 3)
        w.p1, w.p2, w.polygon = P(w.p1), P(w.p2), [P(q) for q in w.polygon]
    for o in openings:
        ux, uy = (o.p2[0] - o.p1[0]) / (o.width or 1), (o.p2[1] - o.p1[1]) / (o.width or 1)
        o.depth = round(o.depth * math.hypot(uy * fw, ux * fh), 3)
        o.p1, o.p2, o.polygon = P(o.p1), P(o.p2), [P(q) for q in o.polygon]
        if o.swing:
            o.swing["hinge"] = P(o.swing["hinge"])
            o.swing["r"] = o.swing["r"] * (fw + fh) / 2


def analyse(path: Path, settings: Settings, log: Log) -> tuple[Plan, Drawing, Calibration]:
    log.step(f"Reading {path.name}")
    drawing = importers.load(path, settings, log)

    log.step("Calibrating scale and units from dimensions")
    cal = calibrate(drawing, settings, log)
    fx, fy, ifx, ify = cal.fx, cal.fy, cal.inv_fx, cal.inv_fy
    dmm = _apply(drawing, fx, fy, ifx, ify, cal.scale)

    log.step("Detecting walls, doors and windows")
    walls, openings, _ = detect(dmm, settings, log)

    if walls and (settings.known_width_mm or settings.known_height_mm):
        for attempt in range(3):
            fw, fh = _size_factors(walls, settings)
            if attempt == 0:
                log.info(f"Known overall size given: rescaling ×{fw:.5f} (X) ×{fh:.5f} (Y)")
                if cal.confidence == "dimensions" and max(abs(fw - 1), abs(fh - 1)) > 0.01:
                    log.warn(f"The size you entered differs from the drawing's own dimensions by "
                             f"{max(abs(fw - 1), abs(fh - 1)):.1%} - using your value")
            fx, fy, ifx, ify = _compose(fx, fy, ifx, ify, fw, fh)
            cal.k *= (fw + fh) / 2
            if max(abs(fw - 1), abs(fh - 1)) > 0.05 and attempt < 2:
                # large change: detect again so thickness/opening limits apply to the right scale
                dmm = _apply(drawing, fx, fy, ifx, ify, cal.scale)
                walls, openings, _ = detect(dmm, settings, log)
                continue
            _scale_result(walls, openings, fw, fh)
            break
        cal.method += " + known overall size"
        cal.confidence = "user"

    if not walls:
        raise ImportErrorUser(
            "No walls were found. Check that wall thickness limits match the drawing "
            f"({settings.min_wall_thickness:g}-{settings.max_wall_thickness:g} mm), the units/scale, "
            "or name the wall layer in the options."
        )
    plan = Plan(walls=walls, openings=openings, settings=settings)
    plan.wall_height = float(dmm.meta.get("wall_height_mm") or settings.wall_height)
    if dmm.meta.get("wall_height_mm"):
        log.info(f"Wall height taken from the 3D model: {plan.wall_height:.0f} mm")
    plan.calibration = cal.summary()
    # weld distance for the wall union (bitmap pieces only touch to within a pixel)
    plan.calibration["weld_mm"] = round(1.5 * dmm.pixel_size, 2) if dmm.raster else 0.5
    plan.source = {"file": path.name, "format": drawing.source_format}

    log.step("Building dimension report")
    stated = stated_dims_mm(cal, drawing, fx, fy) if drawing.dims else []
    plan.dimensions = dimreport.build(plan, cal, stated, log, fx, fy)
    n = {t: sum(o.type == t for o in openings) for t in ("door", "window", "passage")}
    log.info(f"Result: {len(walls)} walls, {n['door']} doors, {n['window']} windows, {n['passage']} passages")
    plan.warnings = log.warnings
    return plan, drawing, cal


def export_all(plan: Plan, formats: Iterable[str], out_dir: Path, stem: str, log: Log) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.step("Building 3D model")
    scene = build_scene(plan, log)
    files = []
    # previews used by the web page
    from .exporters.export2d import plan_to_svg
    from .exporters.export3d import export_glb

    (out_dir / "preview.svg").write_text(plan_to_svg(plan), encoding="utf-8")
    export_glb(scene, plan, out_dir / "preview.glb", Log())
    log.step("Exporting files")
    for fmt in formats:
        p = exporters.export(fmt, plan, scene, out_dir, stem, log)
        if p is not None and Path(p).exists():
            files.append({"format": fmt, "name": Path(p).name, "size": Path(p).stat().st_size})
    return files


def run(path: Path, settings: Settings, formats: Iterable[str], out_dir: Path, log: Log) -> dict[str, Any]:
    plan, _, _ = analyse(path, settings, log)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(plan.to_json(), encoding="utf-8")
    stem = Path(path).stem
    (out_dir / "stem.txt").write_text(stem, encoding="utf-8")
    files = export_all(plan, formats, out_dir, stem, log)
    return {"files": files, "summary": summary(plan)}


def summary(plan: Plan) -> dict[str, Any]:
    x0, y0, x1, y1 = plan.bbox()
    by_src: dict[str, int] = {}
    for d in plan.dimensions:
        by_src[d.source] = by_src.get(d.source, 0) + 1
    return {
        "walls": len(plan.walls),
        "doors": sum(o.type == "door" for o in plan.openings),
        "windows": sum(o.type == "window" for o in plan.openings),
        "passages": sum(o.type == "passage" for o in plan.openings),
        "size_mm": [round(x1 - x0, 1), round(y1 - y0, 1)],
        "wall_height": plan.wall_height,
        "calibration": plan.calibration,
        "dimensions": by_src,
        "openings": [
            {"id": o.id, "type": o.type, "width": round(o.width, 1), "sill": o.sill, "head": o.head,
             "depth": o.depth, "evidence": o.evidence}
            for o in plan.openings
        ],
        "walls_list": [{"id": w.id, "length": round(w.length, 1), "thickness": w.thickness} for w in plan.walls],
        "derived": [{"value": d.value, "p1": d.p1, "p2": d.p2} for d in plan.dimensions if d.source == "derived"],
        "warnings": plan.warnings,
    }


def load_plan(out_dir: Path) -> Plan:
    return Plan.from_dict(json.loads((out_dir / "plan.json").read_text(encoding="utf-8")))
