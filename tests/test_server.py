"""HTTP API test (upload -> log stream -> downloads -> re-export)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "samples"))


def test_upload_convert_download(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOORPLAN_DATA", str(tmp_path / "jobs"))
    import importlib

    import generate_samples as gs
    from fastapi.testclient import TestClient

    import floorplan.server as server

    importlib.reload(server)
    client = TestClient(server.app)
    assert client.get("/").status_code == 200
    fm = client.get("/api/formats").json()
    assert ".dxf" in fm["import"] and "glb" in fm["export"]["3d"]

    src = tmp_path / "flat.dxf"
    gs.write_dxf(src)
    with src.open("rb") as fh:
        r = client.post("/api/jobs", files={"file": ("flat.dxf", fh, "application/dxf")},
                        data={"options": '{"wall_height": 2600}', "formats": "dxf,glb,svg"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(200):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    assert j["summary"]["size_mm"] == [10000.0, 7000.0]
    assert j["summary"]["wall_height"] == 2600
    names = [f["name"] for f in j["files"]]
    assert names == ["flat_plan.dxf", "flat_3d.glb", "flat_plan.svg"]
    assert client.get(f"/api/jobs/{jid}/files/flat_3d.glb").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/..%2Fin%2Fflat.dxf").status_code == 404
    ev = client.get(f"/api/jobs/{jid}/events")
    assert "event: done" in ev.text
    r = client.post(f"/api/jobs/{jid}/export", json={"formats": ["stl"]})
    assert r.status_code == 200 and "flat_3d.stl" in [f["name"] for f in r.json()["files"]]
    bad = client.post("/api/jobs", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400
