"""Importer registry: file extension -> reader producing a ``Drawing``."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .. import dwg
from ..log import Log
from ..model import Drawing, ImportErrorUser, Settings

VECTOR_2D = {".dxf": "AutoCAD DXF", ".dwg": "AutoCAD DWG", ".svg": "SVG", ".pdf": "PDF"}
RASTER_2D = {
    ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
    ".webp": "WebP", ".gif": "GIF",
}
MODELS_3D = {
    ".obj": "Wavefront OBJ", ".stl": "STL", ".ply": "PLY", ".off": "OFF", ".glb": "glTF binary",
    ".gltf": "glTF", ".3mf": "3MF", ".dae": "Collada", ".ifc": "IFC (BIM)",
}


def supported() -> dict[str, str]:
    out = {**VECTOR_2D, **RASTER_2D, **MODELS_3D}
    if not dwg.available()["dwg_read"]:
        out[".dwg"] = "AutoCAD DWG (needs ODA File Converter or LibreDWG - not installed)"
    return out


def load(path: Path, settings: Settings, log: Log) -> Drawing:
    ext = path.suffix.lower()
    if ext == ".dxf":
        from .dxf_importer import import_dxf

        return import_dxf(path, settings, log)
    if ext == ".dwg":
        from .dxf_importer import import_dxf

        if not dwg.available()["dwg_read"]:
            raise ImportErrorUser(
                "DWG import needs a DWG converter (ODA File Converter or LibreDWG's dwg2dxf) on the server. "
                "Install one (see README) or save the drawing as DXF."
            )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / (path.stem + ".dxf")
            log.info(f"Converting DWG to DXF with {dwg.converter_name()}")
            if not dwg.dwg_to_dxf(path, out, log):
                raise ImportErrorUser("DWG to DXF conversion failed")
            return import_dxf(out, settings, log, source_format="dwg")
    if ext == ".svg":
        from .svg_importer import import_svg

        return import_svg(path, settings, log)
    if ext == ".pdf":
        from .pdf_importer import import_pdf

        return import_pdf(path, settings, log)
    if ext in RASTER_2D:
        from .raster_importer import import_image

        return import_image(path, settings, log)
    if ext in MODELS_3D:
        from .mesh_importer import import_mesh

        return import_mesh(path, settings, log)
    raise ImportErrorUser(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(supported()))}")
