"""3D model import (OBJ, STL, PLY, OFF, GLB/GLTF, 3MF, DAE, IFC).

The model is cut by a horizontal plane (default 1.3 m above the floor) which yields
the wall outlines with gaps for doors and windows.  A second, low cut (0.3 m) and
vertical ray casts tell doors from windows and measure real sill/head heights.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from ..log import Log
from ..model import Drawing, ImportErrorUser, Seg, Settings

UNIT_CANDIDATES = [("mm", 1.0), ("cm", 10.0), ("m", 1000.0), ("in", 25.4), ("ft", 304.8)]


def _load_mesh(path: Path, log: Log) -> trimesh.Trimesh:
    ext = path.suffix.lower()
    if ext == ".ifc":
        return _load_ifc(path, log)
    try:
        scene = trimesh.load(str(path), force="scene")
    except Exception as e:  # noqa: BLE001
        raise ImportErrorUser(f"Cannot read 3D file: {e}") from e
    geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)] if isinstance(scene, trimesh.Scene) else [scene]
    if not geoms:
        raise ImportErrorUser("3D file contains no triangle meshes")
    mesh = trimesh.util.concatenate(geoms)
    if ext in (".glb", ".gltf"):
        # glTF is Y-up by specification -> rotate to Z-up
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        log.info("glTF is Y-up - rotated to Z-up")
    return mesh


def _load_ifc(path: Path, log: Log) -> trimesh.Trimesh:
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError as e:
        raise ImportErrorUser("IFC support requires the 'ifcopenshell' package") from e
    f = ifcopenshell.open(str(path))
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    meshes = []
    types = ("IfcWall", "IfcWallStandardCase", "IfcColumn", "IfcCurtainWall", "IfcSlab", "IfcWindow", "IfcDoor")
    for t in types:
        for el in f.by_type(t):
            try:
                shape = ifcopenshell.geom.create_shape(st, el)
            except Exception:  # noqa: BLE001
                continue
            v = np.array(shape.geometry.verts).reshape(-1, 3)
            fc = np.array(shape.geometry.faces).reshape(-1, 3)
            if len(fc) and t not in ("IfcWindow", "IfcDoor"):
                meshes.append(trimesh.Trimesh(v * 1000.0, fc, process=False))  # IFC geometry in metres
    if not meshes:
        raise ImportErrorUser("IFC file has no wall geometry")
    log.info(f"IFC: {len(meshes)} building elements tessellated")
    m = trimesh.util.concatenate(meshes)
    m.metadata["unit_mm"] = 1.0
    return m


class MeshCues:
    """Opening classifier backed by the 3D model (low section + vertical rays)."""

    def __init__(self, mesh: trimesh.Trimesh, low_segments: list[Seg], z0: float, zmax: float, to_mm: float):
        self.mesh = mesh
        self.low = low_segments  # in source units (x, y)
        self.z0 = z0
        self.zmax = zmax
        self.to_mm = to_mm  # source unit -> mm (before calibration)
        self.inv_fx = lambda x: x
        self.inv_fy = lambda y: y
        self.scale = 1.0

    def rebind(self, inv_fx, inv_fy, scale):
        self.inv_fx, self.inv_fy, self.scale = inv_fx, inv_fy, scale
        return self

    def _hits_z(self, sx: float, sy: float) -> list[float]:
        """z of all triangles crossed by the vertical line through (sx, sy) (source units)."""
        if not hasattr(self, "_tri"):
            tri = np.asarray(self.mesh.triangles, dtype=float)
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            keep = np.abs(area) > 1e-12  # vertical faces project to nothing
            self._tri = tri[keep]
            self._area = area[keep]
        t, A = self._tri, self._area
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        w1 = ((b[:, 0] - sx) * (c[:, 1] - sy) - (c[:, 0] - sx) * (b[:, 1] - sy)) / A
        w2 = ((c[:, 0] - sx) * (a[:, 1] - sy) - (a[:, 0] - sx) * (c[:, 1] - sy)) / A
        w3 = 1 - w1 - w2
        inside = (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
        z = w1[inside] * a[inside, 2] + w2[inside] * b[inside, 2] + w3[inside] * c[inside, 2]
        return sorted(z.tolist())

    def vertical_profile(self, x, y):
        """Free vertical intervals (mm above floor) at plan point (x, y) [mm]."""
        zs_raw = self._hits_z(self.inv_fx(x), self.inv_fy(y))
        zs: list[float] = []
        for z in zs_raw:  # merge duplicates from shared triangle edges
            if not zs or z - zs[-1] > 1e-6 * max(1.0, abs(z)):
                zs.append(z)
        zs = [(z - self.z0) * self.scale for z in zs]
        top = (self.zmax - self.z0) * self.scale
        zs = [z for z in zs if -1.0 <= z <= top + 1.0]  # ignore floor / ceiling slabs
        if len(zs) % 2 == 1 and zs[0] < 1.0:
            zs = zs[1:]  # a zero-thickness floor surface, not the bottom of a solid
        # pair up hits: [enter, exit], [enter, exit] ... -> free gaps between them
        free = []
        prev = 0.0
        i = 0
        if zs and zs[0] < 1.0:
            prev = zs[1] if len(zs) > 1 else 0.0  # solid starting at the floor
            i = 2
        while i < len(zs) - 1:
            if zs[i] - prev > 1.0:
                free.append((prev, zs[i]))
            prev = zs[i + 1]
            i += 2
        if top - prev > 1.0:
            free.append((prev, top))
        return free

    def low_cover(self, p1, p2, normal, depth) -> float:
        """Share of the opening width that is solid in the low section (-> window sill)."""
        from shapely.geometry import LineString, Polygon

        cx = [p1[0] + normal[0] * depth / 2, p2[0] + normal[0] * depth / 2, p2[0] - normal[0] * depth / 2, p1[0] - normal[0] * depth / 2]
        cy = [p1[1] + normal[1] * depth / 2, p2[1] + normal[1] * depth / 2, p2[1] - normal[1] * depth / 2, p1[1] - normal[1] * depth / 2]
        poly = Polygon([(self.inv_fx(x), self.inv_fy(y)) for x, y in zip(cx, cy)]).buffer(1e-6)
        width = math.hypot(self.inv_fx(p2[0]) - self.inv_fx(p1[0]), self.inv_fy(p2[1]) - self.inv_fy(p1[1]))
        tot = 0.0
        for s in self.low:
            ls = LineString([(s.x1, s.y1), (s.x2, s.y2)])
            if ls.intersects(poly):
                tot += ls.intersection(poly).length
        return min(1.0, tot / max(2 * width, 1e-9))


def storey_levels(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Floor and ceiling level of the storey.  The largest horizontal surface near the bottom
    is the floor (top of a floor slab), the largest near the top the ceiling / wall tops, so
    slabs in the model do not shift the measured sill and head heights."""
    lo, hi = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    rng = hi - lo
    if rng <= 0:
        return lo, hi
    n = mesh.face_normals
    horiz = np.abs(n[:, 2]) > 0.99
    if not horiz.any():
        return lo, hi
    z = mesh.triangles_center[horiz, 2]
    a = mesh.area_faces[horiz]
    q = np.round(z / (rng * 1e-4)) * (rng * 1e-4)  # merge coplanar faces

    def pick(mask, prefer_high):
        if not mask.any():
            return None
        levels = {}
        for zz, aa in zip(q[mask], a[mask]):
            levels[zz] = levels.get(zz, 0.0) + aa
        best = max(levels.values())
        cands = [zz for zz, aa in levels.items() if aa >= 0.9 * best]
        return max(cands) if prefer_high else min(cands)

    floor = pick(q <= lo + 0.25 * rng, prefer_high=True)
    ceil_ = pick(q >= lo + 0.6 * rng, prefer_high=False)
    return (floor if floor is not None else lo), (ceil_ if ceil_ is not None else hi)


def import_mesh(path: Path, settings: Settings, log: Log) -> Drawing:
    mesh = _load_mesh(path, log)
    log.info(f"3D model: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    ext = mesh.extents
    unit_mm = mesh.metadata.get("unit_mm")
    if path.suffix.lower() not in (".glb", ".gltf", ".ifc"):
        up = int(np.argmin(ext))
        if up != 2:
            if up == 1:  # Y-up -> rotate +90° about X (y -> z)
                rot = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
            else:  # X-up -> rotate -90° about Y (x -> z)
                rot = trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 1, 0])
            mesh.apply_transform(rot)
            log.info(f"Up axis detected as {'XYZ'[up]} (smallest extent) - rotated to Z-up")
    ext = mesh.extents
    if unit_mm is None:
        if settings.units in ("mm", "cm", "m", "in", "ft"):
            unit_mm = dict(UNIT_CANDIDATES)[settings.units]
        else:
            # storey height of a flat is ~2.4-4 m
            best = min(UNIT_CANDIDATES, key=lambda u: abs(math.log(max(ext[2] * u[1], 1e-9) / 2800.0)))
            unit_mm = best[1]
            log.info(f"Model units guessed as {best[0]} (height {ext[2]:g} units ≈ {ext[2] * unit_mm / 1000:.2f} m)")
    zmin, zmax = storey_levels(mesh)
    if zmin > mesh.bounds[0][2] + 1e-9 or zmax < mesh.bounds[1][2] - 1e-9:
        log.info(f"Floor / ceiling slabs detected - storey from {zmin * unit_mm:.0f} to {zmax * unit_mm:.0f} mm")
    h_cut = zmin + min(settings.slice_height / unit_mm, (zmax - zmin) * 0.6)
    h_low = zmin + min(300.0 / unit_mm, (zmax - zmin) * 0.12)

    def cut(z):
        try:
            lines = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z])
        except Exception:  # noqa: BLE001
            return []
        return [Seg(float(a[0]), float(a[1]), float(b[0]), float(b[1]), "SECTION") for a, b in lines]

    segs = cut(h_cut)
    if not segs:
        raise ImportErrorUser("Horizontal section of the model is empty - is it a floor plan model?")
    low = cut(h_low)
    log.info(f"Section at {(h_cut - zmin) * unit_mm:.0f} mm: {len(segs)} edges; low section: {len(low)} edges")
    d = Drawing(source_format=path.suffix.lower().lstrip("."), unit_mm=unit_mm, unit_name="model")
    d.segments = _merge_collinear(segs)
    d.cue_provider = MeshCues(mesh, low, zmin, zmax, unit_mm)
    d.meta["wall_height_mm"] = (zmax - zmin) * unit_mm
    d.meta["from_3d"] = True
    return d


def _merge_collinear(segs: list[Seg]) -> list[Seg]:
    """Triangulated faces produce many collinear pieces; the wall detector merges them too,
    but dropping zero-length pieces here keeps things fast."""
    return [s for s in segs if s.length > 1e-9]
