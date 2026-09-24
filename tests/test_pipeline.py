"""End-to-end tests against generated samples with a known ground truth."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "samples"))

import generate_samples as gs  # noqa: E402

from floorplan import dwg  # noqa: E402
from floorplan.log import Log  # noqa: E402
from floorplan.model import Settings  # noqa: E402
from floorplan.pipeline import analyse, export_all  # noqa: E402

TRUE_DOORS = sorted(o[3] - o[1] if o[3] - o[1] > o[4] - o[2] else o[4] - o[2] for o in gs.OPENINGS if o[0] == "door")
TRUE_WINDOWS = sorted(o[3] - o[1] if o[3] - o[1] > o[4] - o[2] else o[4] - o[2] for o in gs.OPENINGS if o[0] == "window")


@pytest.fixture(scope="session")
def samples(tmp_path_factory) -> dict[str, Path]:
    return gs.write_all(tmp_path_factory.mktemp("samples"))


def _check(plan, tol_size=1.0, tol_open=1.0, area_tol=0.002):
    x0, y0, x1, y1 = plan.bbox()
    assert abs((x1 - x0) - gs.W) <= tol_size, (x1 - x0)
    assert abs((y1 - y0) - gs.H) <= tol_size, (y1 - y0)
    doors = sorted(o.width for o in plan.openings if o.type == "door")
    windows = sorted(o.width for o in plan.openings if o.type == "window")
    assert len(doors) == len(TRUE_DOORS), doors
    assert len(windows) == len(TRUE_WINDOWS), windows
    for a, b in zip(doors + windows, TRUE_DOORS + TRUE_WINDOWS):
        assert abs(a - b) <= tol_open, (a, b)
    # total wall footprint must match the truth
    from floorplan.analysis.walls import wall_union

    area = wall_union(plan.walls).area
    assert abs(area - gs.truth()["walls_area"]) / gs.truth()["walls_area"] < area_tol


@pytest.mark.parametrize("key", ["dxf", "dxf_nolayers", "dxf_offscale_cm", "svg", "pdf"])
def test_vector_formats_exact(samples, key):
    plan, _, cal = analyse(samples[key], Settings(), Log())
    _check(plan)
    assert cal.confidence == "dimensions"
    derived = [d.value for d in plan.dimensions if d.source == "derived"]
    assert any(abs(v - 3100) < 0.5 for v in derived), derived  # the missing chain dimension


def test_obj_model_with_heights(samples):
    plan, _, _ = analyse(samples["obj"], Settings(), Log())
    _check(plan)
    assert abs(plan.wall_height - 2700) < 1
    for o in plan.openings:
        assert abs(o.head - 2100) < 1
        assert abs(o.sill - (900 if o.type == "window" else 0)) < 1


def test_raster_with_paper_scale(samples):
    plan, _, _ = analyse(samples["png"], Settings(paper_scale=50), Log())
    _check(plan, tol_size=40, tol_open=15, area_tol=0.03)


def test_raster_with_known_size(samples):
    plan, _, _ = analyse(samples["png"], Settings(known_width_mm=10000, known_height_mm=7000), Log())
    _check(plan, tol_size=1, tol_open=20, area_tol=0.03)  # bitmap: ±3 px of 6.35 mm


def test_raster_without_scale_is_flagged(samples):
    log = Log()
    _, _, cal = analyse(samples["png"], Settings(dpi=None), log)
    # the PNG carries 200 dpi metadata but no paper scale -> assumed, with a warning
    assert cal.confidence == "estimated"
    assert any("ASSUM" in w for w in log.warnings)


def test_exports_and_roundtrip(samples, tmp_path):
    plan, _, _ = analyse(samples["dxf"], Settings(), Log())
    fmts = ["dxf", "svg", "pdf", "png", "json", "glb", "gltf", "obj", "stl", "ply", "off", "3mf", "dxf3d", "ifc"]
    files = export_all(plan, fmts, tmp_path, "flat", Log())
    assert {f["format"] for f in files} == set(fmts)
    # our own DXF must import back to the same plan
    plan2, _, _ = analyse(tmp_path / "flat_plan.dxf", Settings(), Log())
    _check(plan2)
    import trimesh

    glb = trimesh.load(tmp_path / "flat_3d.glb", force="scene")
    ext = glb.extents  # metres, Y-up
    assert abs(ext[0] - 10.0) < 1e-3 and abs(ext[1] - 2.7) < 1e-3 and abs(ext[2] - 7.0) < 1e-3
    stl = trimesh.load(tmp_path / "flat_3d.stl")
    assert stl.is_watertight
    assert abs(stl.extents[0] - 10000) < 1 and abs(stl.extents[2] - 2700) < 1
    import ifcopenshell

    ifc = ifcopenshell.open(str(tmp_path / "flat_3d.ifc"))
    assert len(ifc.by_type("IfcOpeningElement")) == 8
    assert len(ifc.by_type("IfcDoor")) == 4 and len(ifc.by_type("IfcWindow")) == 4
    # 3D outputs import back into the same plan (sections + ray casts through the recesses)
    for name in ("flat_3d.ifc", "flat_3d.glb", "flat_3d.ply", "flat_3d.3mf"):
        plan3, _, _ = analyse(tmp_path / name, Settings(), Log())
        _check(plan3)


@pytest.mark.skipif(not dwg.available()["dwg_write"], reason="no DWG converter installed")
def test_dwg_roundtrip(samples, tmp_path):
    plan, _, _ = analyse(samples["dxf"], Settings(), Log())
    files = export_all(plan, ["dwg"], tmp_path, "flat", Log())
    assert files and files[0]["name"].endswith(".dwg")
    plan2, _, _ = analyse(tmp_path / "flat_plan.dwg", Settings(), Log())
    _check(plan2)


def test_cad_blocks_hatch_metres(samples):
    """Walls as polylines + hatch, doors/windows as rotated/mirrored blocks, furniture layer."""
    plan, _, cal = analyse(samples["dxf_blocks_m"], Settings(), Log())
    _check(plan)
    assert cal.confidence == "header"
    assert all("block/layer" in " ".join(o.evidence) for o in plan.openings)


def test_floor_ceiling_placeholders(samples, tmp_path):
    """Placeholders are separate objects; slabs in a 3D model do not shift sill/head heights."""
    import trimesh

    plan, _, _ = analyse(samples["dxf"], Settings(floor_thickness=200, ceiling_thickness=250), Log())
    export_all(plan, ["glb", "obj"], tmp_path, "flat", Log())
    glb = trimesh.load(tmp_path / "flat_3d.glb", force="scene")
    assert {"walls", "floor", "ceiling"} <= set(glb.geometry)
    b = {k: g.bounds for k, g in glb.geometry.items()}  # mesh data in mm, Z-up (node transform -> m, Y-up)
    assert abs(b["floor"][0][2] + 200) < 1e-6 and abs(b["floor"][1][2]) < 1e-6
    assert abs(b["ceiling"][0][2] - 2700) < 1e-6 and abs(b["ceiling"][1][2] - 2950) < 1e-6
    plan2, _, _ = analyse(tmp_path / "flat_3d.obj", Settings(), Log())
    _check(plan2)
    assert abs(plan2.wall_height - 2700) < 1
    for o in plan2.openings:
        assert abs(o.head - 2100) < 1 and abs(o.sill - (900 if o.type == "window" else 0)) < 1
    plan3, _, _ = analyse(samples["dxf"], Settings(floor=False, ceiling=False), Log())
    export_all(plan3, ["glb"], tmp_path / "none", "flat", Log())
    assert set(trimesh.load(tmp_path / "none" / "flat_3d.glb", force="scene").geometry) == {"walls"}


def _has_ocr() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_ocr(), reason="rapidocr-onnxruntime not installed")
def test_real_estate_agent_bitmap():
    """samples/real/house_plan.webp: low-res bitmap, scale only from its written dimensions."""
    plan, _, cal = analyse(ROOT / "samples" / "real" / "house_plan.webp", Settings(), Log())
    assert cal.confidence == "dimensions"
    x0, _, x1, _ = plan.bbox()
    assert abs((x1 - x0) - 9000) < 60  # 3,35 m + 5,65 m (±1.5 px at 21 mm/px)
    kinds = [o.type for o in plan.openings]
    assert kinds.count("window") == 5
    assert kinds.count("door") >= 8
    terrace = [o for o in plan.openings if o.width > 4000]
    assert len(terrace) == 1 and terrace[0].type == "door" and abs(terrace[0].width - 4800) < 60


@pytest.mark.skipif(not _has_ocr(), reason="rapidocr-onnxruntime not installed")
def test_flat_with_grey_filled_walls():
    """samples/real/flat_gray_walls.webp: flat grey wall fill, dashed door swings, area labels."""
    plan, _, cal = analyse(ROOT / "samples" / "real" / "flat_gray_walls.webp", Settings(), Log())
    assert cal.confidence == "dimensions"
    x0, y0, x1, y1 = plan.bbox()
    assert abs((x1 - x0) - 9990) < 150 and abs((y1 - y0) - 7530) < 150
    kinds = [o.type for o in plan.openings]
    assert kinds.count("window") == 2
    assert kinds.count("door") >= 4


@pytest.mark.skipif(not _has_ocr(), reason="rapidocr-onnxruntime not installed")
def test_house_with_hatched_and_outlined_walls():
    """samples/real/house_hatched.jpg: cross-hatched insulation + outlined (hollow) walls; the
    written dimensions compete, room area labels ("A: 11,10 m²") settle the scale."""
    plan, _, _ = analyse(ROOT / "samples" / "real" / "house_hatched.jpg", Settings(), Log())
    x0, y0, x1, y1 = plan.bbox()
    assert abs((x1 - x0) - 17400) < 500 and abs((y1 - y0) - 11000) < 500
    kinds = [o.type for o in plan.openings]
    assert kinds.count("window") >= 5
    assert kinds.count("door") >= 6


@pytest.mark.skipif(not _has_ocr(), reason="rapidocr-onnxruntime not installed")
def test_house_with_watermark():
    """samples/real/house_watermark.jpg: pale colour watermark over a hatched plan (11,90 × 6,88 m)."""
    plan, _, _ = analyse(ROOT / "samples" / "real" / "house_watermark.jpg", Settings(), Log())
    x0, y0, x1, y1 = plan.bbox()
    assert abs((x1 - x0) - 11990) < 120 and abs((y1 - y0) - 6880) < 120
    kinds = [o.type for o in plan.openings]
    assert kinds.count("window") >= 5
    assert kinds.count("door") >= 6
