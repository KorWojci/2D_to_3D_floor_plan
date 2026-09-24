"""Export format registry."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import dwg
from ..log import Log
from ..model import Plan
from . import export2d, export3d

FORMATS_2D = {
    "dxf": ("AutoCAD DXF 2018 (2D, layered, with dimensions)", export2d.export_dxf),
    "dwg": ("AutoCAD DWG (2D, via ODA File Converter or LibreDWG)", export2d.export_dwg),
    "svg": ("SVG vector drawing", export2d.export_svg),
    "pdf": ("PDF vector drawing (1:50)", export2d.export_pdf),
    "png": ("PNG image", export2d.export_png),
    "json": ("JSON plan data (walls, openings, dimensions)", export2d.export_json),
}
FORMATS_3D = {
    "glb": ("glTF binary (.glb)", export3d.export_glb),
    "gltf": ("glTF (.gltf, embedded)", export3d.export_gltf),
    "obj": ("Wavefront OBJ", export3d.export_obj),
    "stl": ("STL (binary)", export3d.export_stl),
    "ply": ("PLY", export3d.export_ply),
    "off": ("OFF", export3d.export_off),
    "3mf": ("3MF", export3d.export_3mf),
    "dxf3d": ("DXF 3D mesh", export3d.export_dxf3d),
    "ifc": ("IFC4 BIM model (walls, openings, doors, windows)", export3d.export_ifc),
}
FILE_EXT = {"dxf3d": "_3d.dxf"}


def catalogue() -> dict:
    avail = dwg.available()
    return {
        "2d": {k: {"label": v[0], "available": k != "dwg" or avail["dwg_write"]} for k, v in FORMATS_2D.items()},
        "3d": {k: {"label": v[0], "available": True} for k, v in FORMATS_3D.items()},
    }


def export(fmt: str, plan: Plan, scene, out_dir: Path, stem: str, log: Log) -> Optional[Path]:
    fmt = fmt.lower()
    name = stem + (FILE_EXT.get(fmt) or ("_3d." + fmt if fmt in FORMATS_3D else "_plan." + fmt))
    path = out_dir / name
    try:
        if fmt in FORMATS_2D:
            res = FORMATS_2D[fmt][1](plan, path, log)
        elif fmt in FORMATS_3D:
            res = FORMATS_3D[fmt][1](scene, plan, path, log)
        else:
            log.warn(f"Unknown export format '{fmt}'")
            return None
    except Exception as e:  # noqa: BLE001 - one failing format must not kill the job
        log.error(f"Export to {fmt.upper()} failed: {e}")
        return None
    return res
