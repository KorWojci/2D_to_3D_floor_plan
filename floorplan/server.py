"""Web server: upload, live log (Server-Sent Events), previews and downloads.

Run:  uvicorn floorplan.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import dwg, exporters, importers
from .log import Log
from .model import ImportErrorUser, Settings
from .pipeline import DEFAULT_FORMATS, export_all, load_plan, run, summary

DATA_DIR = Path(os.environ.get("FLOORPLAN_DATA", Path(__file__).resolve().parent.parent / "data" / "jobs"))
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = int(os.environ.get("FLOORPLAN_MAX_UPLOAD_MB", "200")) * 1024 * 1024
JOB_TTL = float(os.environ.get("FLOORPLAN_JOB_TTL_H", "24")) * 3600
_ID = re.compile(r"^[a-f0-9]{32}$")

app = FastAPI(title="Floor plan 2D → 3D", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@dataclass
class Job:
    id: str
    dir: Path
    filename: str
    status: str = "queued"  # queued | running | done | error
    entries: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    summary: Optional[dict] = None
    error: str = ""
    created: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self) -> Log:
        return Log(self.entries.append)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "filename": self.filename, "status": self.status, "log": self.entries,
                "files": self.files, "summary": self.summary, "error": self.error}


JOBS: dict[str, Job] = {}


def _cleanup() -> None:
    now = time.time()
    for jid, job in list(JOBS.items()):
        if now - job.created > JOB_TTL and job.status in ("done", "error"):
            shutil.rmtree(job.dir, ignore_errors=True)
            JOBS.pop(jid, None)


def _job(jid: str) -> Job:
    if not _ID.match(jid) or jid not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[jid]


def _work(job: Job, src: Path, settings: Settings, formats: list[str]) -> None:
    log = job.log()
    job.status = "running"
    try:
        res = run(src, settings, formats, job.dir / "out", log)
        job.files = res["files"]
        job.summary = res["summary"]
        job.status = "done"
        log.step("Done")
    except ImportErrorUser as e:
        job.error = str(e)
        log.error(str(e))
        job.status = "error"
    except Exception as e:  # noqa: BLE001
        job.error = f"Internal error: {e}"
        log.error(job.error)
        log.info(traceback.format_exc(limit=4))
        job.status = "error"


def _parse_formats(formats: str) -> list[str]:
    known = set(exporters.FORMATS_2D) | set(exporters.FORMATS_3D)
    out = [f.strip().lower() for f in (formats or "").split(",") if f.strip()]
    out = [f for f in out if f in known]
    return out or list(DEFAULT_FORMATS)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/formats")
def formats() -> dict[str, Any]:
    return {
        "import": importers.supported(),
        "export": exporters.catalogue(),
        "default_export": list(DEFAULT_FORMATS),
        "dwg_converter": dwg.converter_name(),
    }


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), options: str = Form("{}"), formats: str = Form("")) -> JSONResponse:
    _cleanup()
    name = Path(file.filename or "drawing").name
    ext = Path(name).suffix.lower()
    if ext not in importers.supported():
        raise HTTPException(400, f"Unsupported file type '{ext}'")
    try:
        opts = json.loads(options or "{}")
        if not isinstance(opts, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "options must be a JSON object") from None
    jid = uuid.uuid4().hex
    jdir = DATA_DIR / jid
    (jdir / "in").mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)[:60] or "drawing"
    src = jdir / "in" / f"{safe_stem}{ext}"
    size = 0
    with src.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                shutil.rmtree(jdir, ignore_errors=True)
                raise HTTPException(413, f"File larger than {MAX_UPLOAD // 1024 // 1024} MB")
            fh.write(chunk)
    job = Job(jid, jdir, name)
    JOBS[jid] = job
    job.log().info(f"Uploaded {name} ({size / 1024:.1f} kB)")
    settings = Settings.from_dict(opts)
    threading.Thread(target=_work, args=(job, src, settings, _parse_formats(formats)), daemon=True).start()
    return JSONResponse({"id": jid})


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict[str, Any]:
    return _job(jid).public()


@app.get("/api/jobs/{jid}/events")
async def events(jid: str) -> StreamingResponse:
    job = _job(jid)

    async def gen():
        i = 0
        idle = 0
        while True:
            while i < len(job.entries):
                yield f"event: log\ndata: {json.dumps(job.entries[i])}\n\n"
                i += 1
            if job.status in ("done", "error") and i >= len(job.entries):
                yield f"event: {job.status}\ndata: {json.dumps(job.public())}\n\n"
                return
            await asyncio.sleep(0.15)
            idle += 1
            if idle % 100 == 0:
                yield ": keep-alive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/jobs/{jid}/export")
def export_more(jid: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = _job(jid)
    if job.status != "done":
        raise HTTPException(409, "Job is not finished")
    fmts = _parse_formats(",".join(payload.get("formats", [])))
    with job.lock:
        log = job.log()
        plan = load_plan(job.dir / "out")
        stem_file = job.dir / "out" / "stem.txt"
        stem = stem_file.read_text(encoding="utf-8") if stem_file.exists() else "plan"
        new = export_all(plan, fmts, job.dir / "out", stem, log)
        names = {f["name"] for f in job.files}
        job.files += [f for f in new if f["name"] not in names]
        for f in new:
            for g in job.files:
                if g["name"] == f["name"]:
                    g["size"] = f["size"]
        job.summary = summary(plan)
    return job.public()


def _file(job: Job, name: str) -> Path:
    allowed = {f["name"] for f in job.files} | {"preview.svg", "preview.glb", "plan.json"}
    if name not in allowed:
        raise HTTPException(404, "No such file")
    p = job.dir / "out" / name
    if not p.exists():
        raise HTTPException(404, "No such file")
    return p


@app.get("/api/jobs/{jid}/files/{name}")
def download(jid: str, name: str) -> FileResponse:
    job = _job(jid)
    p = _file(job, name)
    return FileResponse(p, filename=name)


@app.get("/api/jobs/{jid}/preview/{name}")
def preview(jid: str, name: str) -> FileResponse:
    job = _job(jid)
    if name not in ("preview.svg", "preview.glb"):
        raise HTTPException(404, "No such preview")
    p = _file(job, name)
    media = "image/svg+xml" if name.endswith(".svg") else "model/gltf-binary"
    return FileResponse(p, media_type=media)
