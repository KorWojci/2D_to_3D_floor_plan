"""3D exports: GLB, glTF, OBJ, STL, PLY, OFF, 3MF, DXF (3D mesh) and IFC.

Units: millimetres and Z-up for every format except glTF/GLB, which by specification
are metres and Y-up (they are converted accordingly, so real-world size is preserved).
"""
from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import numpy as np
import trimesh

from ..log import Log
from ..model import Plan
from ..builder3d import single_mesh


def _gltf_scene(scene: trimesh.Scene) -> trimesh.Scene:
    s = scene.copy()
    m = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])  # Z-up -> Y-up
    m = trimesh.transformations.scale_matrix(0.001) @ m  # mm -> m
    s.apply_transform(m)
    return s


def export_glb(scene, plan, path: Path, log: Log) -> Path:
    path.write_bytes(_gltf_scene(scene).export(file_type="glb"))
    log.info("GLB written (metres, Y-up per glTF spec)")
    return path


def export_gltf(scene, plan, path: Path, log: Log) -> Path:
    from trimesh.exchange.gltf import export_gltf as _eg

    files = _eg(_gltf_scene(scene), embed_buffers=True)
    data = files.get("model.gltf") or next(iter(files.values()))
    path.write_bytes(data)
    log.info("glTF written (single file, embedded buffers)")
    return path


def export_obj(scene, plan, path: Path, log: Log) -> Path:
    txt = scene.export(file_type="obj", include_normals=True)
    if isinstance(txt, bytes):
        txt = txt.decode()
    path.write_text("# floor plan 3D model - units: millimetres, Z up\n" + txt)
    log.info("OBJ written (mm, Z-up)")
    return path


def export_stl(scene, plan, path: Path, log: Log) -> Path:
    single_mesh(scene, solids_only=True).export(str(path), file_type="stl")
    log.info("STL written (binary, mm)")
    return path


def export_ply(scene, plan, path: Path, log: Log) -> Path:
    single_mesh(scene).export(str(path), file_type="ply")
    log.info("PLY written (mm, with colours)")
    return path


def export_off(scene, plan, path: Path, log: Log) -> Path:
    single_mesh(scene).export(str(path), file_type="off")
    log.info("OFF written (mm)")
    return path


def export_3mf(scene, plan, path: Path, log: Log) -> Path:
    path.write_bytes(trimesh.Scene(single_mesh(scene, solids_only=True)).export(file_type="3mf"))
    log.info("3MF written (mm)")
    return path


def export_dxf3d(scene, plan, path: Path, log: Log) -> Path:
    doc = ezdxf.new("R2018")
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    for name, geom in scene.geometry.items():
        doc.layers.add(name.upper())
        v = np.asarray(geom.vertices)
        f = np.asarray(geom.faces)
        mesh = msp.add_mesh(dxfattribs={"layer": name.upper()})
        with mesh.edit_data() as md:
            md.vertices = [tuple(map(float, p)) for p in v]
            md.faces = [tuple(map(int, t)) for t in f]
    doc.saveas(path)
    log.info("3D DXF written (MESH entities, mm)")
    return path


# ------------------------------------------------------------------ IFC


def _wall_runs(plan: Plan):
    """Merge wall pieces interrupted by openings into continuous wall runs (IFC walls)."""
    walls = plan.walls
    parent = list(range(len(walls)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def frame(w):
        ux, uy = (w.p2[0] - w.p1[0]) / w.length, (w.p2[1] - w.p1[1]) / w.length
        if ux < -1e-9 or (abs(ux) <= 1e-9 and uy < 0):
            ux, uy = -ux, -uy
        return ux, uy

    host_of: dict[str, int] = {}
    for o in plan.openings:
        ou = ((o.p2[0] - o.p1[0]) / o.width, (o.p2[1] - o.p1[1]) / o.width)
        touching = []
        for i, w in enumerate(walls):
            ux, uy = frame(w)
            tol = 2.0 + 0.05 * w.thickness  # raster input is only pixel-accurate
            if abs(abs(ux * ou[0] + uy * ou[1]) - 1) > 2e-3 or abs(w.thickness - o.depth) > tol:
                continue
            nx, ny = -uy, ux
            if abs((o.p1[0] - w.p1[0]) * nx + (o.p1[1] - w.p1[1]) * ny) > tol:
                continue
            s = sorted([w.p1[0] * ux + w.p1[1] * uy, w.p2[0] * ux + w.p2[1] * uy])
            t = sorted([o.p1[0] * ux + o.p1[1] * uy, o.p2[0] * ux + o.p2[1] * uy])
            if abs(s[1] - t[0]) < tol or abs(t[1] - s[0]) < tol:
                touching.append(i)
        for i in touching[1:]:
            parent[find(i)] = find(touching[0])
        if touching:
            host_of[o.id] = touching[0]
    runs: dict[int, list[int]] = {}
    for i in range(len(walls)):
        runs.setdefault(find(i), []).append(i)
    out = []
    for members in runs.values():
        w0 = walls[members[0]]
        ux, uy = frame(w0)
        nx, ny = -uy, ux
        v = w0.p1[0] * nx + w0.p1[1] * ny
        ss = [p[0] * ux + p[1] * uy for i in members for p in (walls[i].p1, walls[i].p2)]
        s1, s2 = min(ss), max(ss)
        p1 = (s1 * ux + v * nx, s1 * uy + v * ny)
        out.append({"members": members, "p1": p1, "u": (ux, uy), "length": s2 - s1, "t": w0.thickness,
                    "name": "+".join(walls[i].id for i in members)})
    idx_run = {i: k for k, r in enumerate(out) for i in r["members"]}
    return out, {oid: idx_run[i] for oid, i in host_of.items()}


def export_ifc(scene, plan, path: Path, log: Log) -> Path | None:
    try:
        import ifcopenshell
        import ifcopenshell.api
        import ifcopenshell.api.aggregate
        import ifcopenshell.api.context
        import ifcopenshell.api.feature
        import ifcopenshell.api.geometry
        import ifcopenshell.api.root
        import ifcopenshell.api.spatial
        import ifcopenshell.api.unit
    except ImportError:
        log.warn("IFC export needs the 'ifcopenshell' package")
        return None
    f = ifcopenshell.api.project.create_file(version="IFC4") if hasattr(ifcopenshell.api, "project") else ifcopenshell.file(schema="IFC4")
    project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="Floor plan")
    lu = ifcopenshell.api.unit.add_si_unit(f, unit_type="LENGTHUNIT", prefix="MILLI")
    ifcopenshell.api.unit.assign_unit(f, units=[lu])
    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    body = ifcopenshell.api.context.add_context(f, context_type="Model", context_identifier="Body",
                                                target_view="MODEL_VIEW", parent=model)
    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingStorey", name="Flat")
    ifcopenshell.api.aggregate.assign_object(f, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(f, relating_object=site, products=[building])
    ifcopenshell.api.aggregate.assign_object(f, relating_object=building, products=[storey])

    def placement(obj, origin, u, z=0.0):
        m = np.eye(4)
        m[:3, 0] = [u[0], u[1], 0]
        m[:3, 1] = [-u[1], u[0], 0]
        m[:3, 2] = [0, 0, 1]
        m[:3, 3] = [origin[0], origin[1], z]
        ifcopenshell.api.geometry.edit_object_placement(f, product=obj, matrix=m, is_si=False)

    runs, host = _wall_runs(plan)
    H = plan.wall_height
    ifc_walls = []
    for r in runs:
        w = ifcopenshell.api.root.create_entity(f, ifc_class="IfcWall", name=r["name"])
        placement(w, r["p1"], r["u"])
        # representation sizes are given in SI metres (placements above are in project mm)
        rep = ifcopenshell.api.geometry.add_wall_representation(f, context=body, length=r["length"] / 1000, height=H / 1000,
                                                                 thickness=r["t"] / 1000, offset=-r["t"] / 2000)
        ifcopenshell.api.geometry.assign_representation(f, product=w, representation=rep)
        ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[w])
        ifc_walls.append(w)
    for o in plan.openings:
        u = ((o.p2[0] - o.p1[0]) / o.width, (o.p2[1] - o.p1[1]) / o.width)
        op = ifcopenshell.api.root.create_entity(f, ifc_class="IfcOpeningElement", name=f"{o.id} recess")
        placement(op, o.p1, u, o.sill)
        extra = 20.0
        rep = ifcopenshell.api.geometry.add_wall_representation(f, context=body, length=o.width / 1000,
                                                                 height=(o.head - o.sill) / 1000,
                                                                 thickness=(o.depth + 2 * extra) / 1000,
                                                                 offset=-(o.depth / 2 + extra) / 1000)
        ifcopenshell.api.geometry.assign_representation(f, product=op, representation=rep)
        if o.id in host:
            ifcopenshell.api.feature.add_feature(f, feature=op, element=ifc_walls[host[o.id]])
        cls = {"door": "IfcDoor", "window": "IfcWindow"}.get(o.type)
        if cls:
            el = ifcopenshell.api.root.create_entity(f, ifc_class=cls, name=o.id)
            el.OverallWidth = float(o.width)
            el.OverallHeight = float(o.head - o.sill)
            placement(el, o.p1, u, o.sill)
            ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[el])
            if o.id in host:
                ifcopenshell.api.feature.add_filling(f, opening=op, element=el)
    # floor / ceiling placeholders as IfcSlab (IFC needs a real thickness: 200 mm when the
    # placeholder is only a surface, placed below the floor / above the ceiling)
    from ..builder3d import building_footprint

    st = plan.settings
    fp = building_footprint(plan)
    rings = [g for g in getattr(fp, "geoms", [fp]) if not g.is_empty]
    n_slabs = 0
    for enabled, name, ptype, t, z in ((st.floor, "Floor", "FLOOR", st.floor_thickness or 200.0, -(st.floor_thickness or 200.0)),
                                       (st.ceiling, "Ceiling", "ROOF", st.ceiling_thickness or 200.0, H)):
        if not enabled:
            continue
        for g in rings:
            slab = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSlab", name=name, predefined_type=ptype)
            placement(slab, (0.0, 0.0), (1.0, 0.0), z)
            ring = [(round(x, 1), round(y, 1)) for x, y in g.simplify(1.0).exterior.coords]  # mm, closed
            curve = f.createIfcPolyline([f.createIfcCartesianPoint(pt) for pt in ring])
            profile = f.createIfcArbitraryClosedProfileDef("AREA", None, curve)
            solid = f.createIfcExtrudedAreaSolid(profile, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0))),
                                                 f.createIfcDirection((0.0, 0.0, 1.0)), float(t))
            rep = f.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid])
            ifcopenshell.api.geometry.assign_representation(f, product=slab, representation=rep)
            ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[slab])
            n_slabs += 1
    f.write(str(path))
    log.info(f"IFC4 written ({n_slabs} IfcSlab placeholder(s), {len(runs)} IfcWall, {len(plan.openings)} IfcOpeningElement with doors/windows)")
    return path
