"""DWG <-> DXF conversion through external converters.

DWG is a closed format; no pure-Python writer exists.  Two converters are supported,
whichever is found first:

* ODA File Converter (free download from opendesign.com) - via ``ezdxf.addons.odafc``
* LibreDWG command line tools ``dwg2dxf`` / ``dxf2dwg`` (GPL, build from source)

Set ``FLOORPLAN_ODA_PATH`` to the ODAFileConverter executable if it is not on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .log import Log


def _oda() -> Optional[str]:
    exe = os.environ.get("FLOORPLAN_ODA_PATH") or shutil.which("ODAFileConverter")
    return exe if exe and Path(exe).exists() else None


def _libredwg(tool: str) -> Optional[str]:
    return shutil.which(tool)


def available() -> dict[str, bool]:
    return {
        "dwg_read": bool(_oda() or _libredwg("dwg2dxf")),
        "dwg_write": bool(_oda() or _libredwg("dxf2dwg")),
    }


def converter_name() -> str:
    if _oda():
        return "ODA File Converter"
    if _libredwg("dwg2dxf") or _libredwg("dxf2dwg"):
        return "LibreDWG"
    return "none"


def dwg_to_dxf(src: Path, dst: Path, log: Log) -> bool:
    oda = _oda()
    if oda:
        from ezdxf.addons import odafc

        odafc.win_exec_path = oda
        odafc.unix_exec_path = oda
        odafc.convert(str(src), str(dst), version="R2018", replace=True)
        return dst.exists()
    tool = _libredwg("dwg2dxf")
    if tool:
        r = subprocess.run([tool, "-y", "-o", str(dst), str(src)], capture_output=True, text=True, timeout=300)
        if r.returncode != 0 and not dst.exists():
            log.warn(f"dwg2dxf failed: {(r.stderr or r.stdout)[-400:]}")
        if dst.exists():
            _sanitize_libredwg_dxf(dst)
        return dst.exists()
    return False


def _sanitize_libredwg_dxf(path: Path) -> None:
    """LibreDWG emits some objects (e.g. ENDBLK) with handle 0, which strict DXF readers
    reject.  Drop those handle tags - the reader assigns fresh handles."""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "5" and i + 1 < len(lines) and lines[i + 1].strip() == "0" and \
                i >= 2 and lines[i - 2].strip() == "0":
            i += 2
            continue
        out.append(lines[i])
        i += 1
    path.write_text("\n".join(out), encoding="utf-8")


def dxf_to_dwg(src: Path, dst: Path, log: Log) -> bool:
    oda = _oda()
    if oda:
        from ezdxf.addons import odafc

        odafc.win_exec_path = oda
        odafc.unix_exec_path = oda
        odafc.convert(str(src), str(dst), version="R2018", replace=True)
        return dst.exists()
    tool = _libredwg("dxf2dwg")
    if tool:
        r = subprocess.run([tool, "-y", "-o", str(dst), str(src)], capture_output=True, text=True, timeout=300)
        if not dst.exists():
            log.warn(f"dxf2dwg failed: {(r.stderr or r.stdout)[-400:]}")
        return dst.exists() and dst.stat().st_size > 0
    return False


def verify_dwg(path: Path, log: Log) -> bool:
    """Read a freshly written DWG back and check that it contains the drawing."""
    with tempfile.TemporaryDirectory() as td:
        back = Path(td) / "check.dxf"
        try:
            if not dwg_to_dxf(path, back, Log()):
                return False
            import ezdxf

            doc = ezdxf.readfile(str(back))
            n = len(doc.modelspace())
        except Exception as e:  # noqa: BLE001
            log.warn(f"DWG verification failed: {e}")
            return False
    if n == 0:
        log.warn("DWG verification: file is empty after read-back")
        return False
    log.info(f"DWG verified by reading it back ({n} entities)")
    return True
