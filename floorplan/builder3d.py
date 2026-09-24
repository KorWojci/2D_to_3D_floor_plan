"""3D model generation: walls extruded to storey height with door and window recesses."""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .log import Log
from .model import Plan

WALL_COLOR = [228, 226, 220, 255]
FLOOR_COLOR = [196, 170, 130, 255]
CEILING_COLOR = [245, 245, 242, 255]


def _add_plate(scene: trimesh.Scene, footprint, walls, z: float, t: float, color, name: str, log: Log) -> None:
    """Floor / ceiling placeholder: a surface over the rooms at height ``z`` (t = 0) or a slab of
    thickness |t| over the whole footprint (t < 0 below z, t > 0 above z)."""
    if footprint.is_empty:
        return
    if t:
        z0, z1 = (z + t, z) if t < 0 else (z, z + t)
        parts = _extrude(footprint, z0, z1)
    else:
        parts = []
        for p in _polys(footprint.difference(walls)):
            v2, f = trimesh.creation.triangulate_polygon(p)
            v3 = np.column_stack([v2, np.full(len(v2), z)])
            m = trimesh.Trimesh(v3, f)
            if name == "ceiling":
                m.invert()  # the visible side of a ceiling faces down
            parts.append(m)
    mesh = trimesh.util.concatenate(parts) if parts else None
    if mesh is not None and len(mesh.faces):
        _paint(mesh, color, name)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        kind = f"slab {abs(t):g} mm" if t else "surface"
        log.info(f"{name.capitalize()} placeholder: {kind} at {z:g} mm, {footprint.area / 1e6:.1f} m² footprint")


def _paint(mesh: trimesh.Trimesh, rgba: list[int], name: str) -> None:
    mat = trimesh.visual.material.PBRMaterial(name=name, baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.9)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)


def _polys(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def _extrude(g, z0: float, z1: float) -> list[trimesh.Trimesh]:
    out = []
    for p in _polys(g):
        p = p.buffer(0)
        if p.area < 1.0:
            continue
        for q in _polys(p):
            m = trimesh.creation.extrude_polygon(q, z1 - z0)
            m.apply_translation([0, 0, z0])
            out.append(m)
    return out


def _union(meshes: list[trimesh.Trimesh], log: Log) -> trimesh.Trimesh:
    if not meshes:
        return trimesh.Trimesh()
    try:
        import manifold3d as mf

        ms = []
        for m in meshes:
            ms.append(mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, dtype=np.float32),
                                          tri_verts=np.asarray(m.faces, dtype=np.uint32))))
        res = mf.Manifold.batch_boolean(ms, mf.OpType.Add)
        out = res.to_mesh()
        mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
        if len(mesh.faces):
            return mesh
    except Exception as e:  # noqa: BLE001
        log.warn(f"Boolean union unavailable ({e}); exporting non-merged solids")
    return trimesh.util.concatenate(meshes)


def build_scene(plan: Plan, log: Log) -> trimesh.Scene:
    H = plan.wall_height
    walls = wall_geometry(plan)
    # z levels where the cross-section changes
    levels = {0.0, H}
    for o in plan.openings:
        levels.update(z for z in (o.sill, o.head) if 0 < z < H)
    levels = sorted(levels)
    pieces = []
    for z0, z1 in zip(levels, levels[1:]):
        zm = (z0 + z1) / 2
        solid = [walls]
        for o in plan.openings:
            if not (o.sill <= zm <= o.head):
                solid.append(Polygon(o.polygon).buffer(0))
        pieces += _extrude(unary_union(solid), z0, z1)
    wall_mesh = _union(pieces, log)
    _paint(wall_mesh, WALL_COLOR, "wall")
    scene = trimesh.Scene()
    scene.add_geometry(wall_mesh, node_name="walls", geom_name="walls")
    log.info(f"3D walls: {len(wall_mesh.faces)} triangles, watertight={wall_mesh.is_watertight}, "
             f"height {H:g} mm, {len(plan.openings)} recesses")
    st = plan.settings
    if st.floor or st.ceiling:
        footprint = building_footprint(plan)
        if st.floor:
            _add_plate(scene, footprint, walls, 0.0, -st.floor_thickness, FLOOR_COLOR, "floor", log)
        if st.ceiling:
            _add_plate(scene, footprint, walls, H, st.ceiling_thickness, CEILING_COLOR, "ceiling", log)
    return scene


def wall_geometry(plan: Plan):
    """Union of all wall pieces, welded: pieces traced from a bitmap touch only to within a
    pixel, which would leave hair-line slits in the model.  The weld distance is stored by the
    pipeline (1.5 px for bitmaps, 0.5 mm otherwise)."""
    u = unary_union([Polygon(w.polygon).buffer(0) for w in plan.walls]) if plan.walls else Polygon()
    c = float(plan.calibration.get("weld_mm", 0.5)) if plan.calibration else 0.5
    return u.buffer(c, join_style=2).buffer(-c, join_style=2) if c > 0 and not u.is_empty else u


def building_footprint(plan: Plan):
    """Outline of the flat (walls + openings, rooms filled)."""
    allp = unary_union([wall_geometry(plan)] + [Polygon(o.polygon).buffer(0) for o in plan.openings])
    c = max(2.0, float(plan.calibration.get("weld_mm", 0.5)) if plan.calibration else 2.0)
    allp = allp.buffer(c, join_style=2).buffer(-c, join_style=2)
    return unary_union([Polygon(p.exterior) for p in _polys(allp) if p.area > 1e4])


def single_mesh(scene: trimesh.Scene, solids_only: bool = False) -> trimesh.Trimesh:
    """All geometry as one mesh.  ``solids_only`` drops open surfaces (zero-thickness floor),
    which keeps files for 3D printing watertight."""
    geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
    if solids_only:
        geoms = [g for g in geoms if g.is_watertight] or geoms
    return trimesh.util.concatenate(geoms)
