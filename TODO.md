# TODO: 2D floor plan → 3D flat

Legend: `[x]` done and covered by tests · `[~]` done but only partly, or not tested
against real-world files · `[ ]` not done yet

Each step says **what** has to be done and **how**. Code references are relative to
`floorplan/`.

---

## Phase 0: Project skeleton
- [x] **Python package with one data model shared by every stage.** How: `model.py`
  defines `Drawing` (the raw imported content, in source units, Y up) and `Plan`
  (walls, openings and dimensions, in mm, JSON-serialisable).
- [x] **Logger that streams to the UI.** How: `log.py`. Its `Log(sink)` collects
  info/step/warn/error entries.
- [x] **Dependencies pinned.** How: `requirements.txt` and `requirements-dev.txt`.
- [x] **Sample drawings with a known ground truth.** How:
  `samples/generate_samples.py` writes the same flat as DXF (with layers, without
  layers, with blocks + hatch in metres, and in cm drawn 3 % off-scale) plus SVG, PDF,
  PNG and OBJ. It leaves out one chain dimension on purpose.

## Phase 1: Import (2D and 3D formats → `Drawing`)
- [x] **DXF.** How: `importers/dxf_importer.py` uses ezdxf and falls back to recover
  mode for damaged files. It explodes blocks recursively and records their name and
  bbox as `Label`s. It reads bulges, wide polylines, MLINE, hatches (solid → fill
  regions), TEXT/MTEXT (centre point computed from alignment) and linear/aligned
  DIMENSION (text override, `<>`, DIMLFAC). It skips frozen/off layers and reads the
  units from `$INSUNITS`.
- [x] **DWG.** How: `dwg.py` converts DWG to DXF with ODA File Converter or LibreDWG
  `dwg2dxf`, then fixes LibreDWG's handle-0 records.
- [x] **PDF vector.** How: `importers/pdf_importer.py` uses PyMuPDF `get_drawings()`.
  Bézier curves and flattened polylines are recognised as arcs (door swings), dark
  fills become wall regions, and text lines are read with their direction.
- [x] **PDF scanned.** How: pages with fewer than 20 vector paths are rendered at
  300 dpi and passed to the raster importer.
- [x] **SVG.** How: `importers/svg_importer.py` uses svgelements, which handles
  transforms, `<use>` and CSS units. Curves are fitted to arcs, and group ids/classes
  act as layer names.
- [x] **Raster images.** How: `importers/raster_importer.py`.
  - Otsu threshold, then the thin stroke width is estimated from run lengths.
  - A morphological opening keeps the thick wall strokes, which are traced as
    polygons (grown by half a pixel).
  - Faint thin lines are kept in an ink mask for the door/window cues.
  - Walls drawn as outlines fall back to a Hough transform.
  - DPI is read from the file metadata.
- [x] **3D models (OBJ/STL/PLY/OFF/GLB/glTF/3MF/DAE).** How:
  `importers/mesh_importer.py`.
  - Up axis: glTF is always Y-up; for other formats the smallest extent is taken as up.
  - Units are guessed from the storey height.
  - The model is cut at 1.3 m to get the plan, and cut again at 0.3 m.
  - Vertical ray casts (numpy, no rtree needed) measure sill and head heights.
- [x] **IFC.** How: ifcopenshell tessellates the walls, slabs and columns, and the result
  goes through the mesh path.
- [ ] **Multi-page PDF / multiple layouts.** How: let the user pick the page (UI select
  + `Settings.page`). Today the page with the most vector paths is used.
- [ ] **DXF paper space / viewports.** How: when model space is empty, read the layout
  and apply the viewport scale.
- [ ] **Native SketchUp (.skp), Revit (.rvt), ArchiCAD (.pln).** How: not possible
  without the vendor SDKs. Document the IFC/DWG export route for these tools instead.
- [ ] **HEIC / camera photos.** How: pillow-heif, plus perspective correction (find
  the paper quad with `cv2.findContours` and warp it) before raster tracing.

## Phase 2: Scale, units and dimensions (`analysis/calibration.py`)
- [x] **Parse dimension texts.** How: `analysis/numbers.py` handles mm/cm/m/in/ft,
  decimal comma, feet-inches with fractions and MTEXT codes. It rejects areas (m²),
  angles and labels.
- [x] **Text dimensions.** How: find a numeric text sitting on a parallel segment, then
  use the perpendicular tick and extension lines around the text as the measured points
  (this also works for chains drawn as one line). Those lines are marked as dimension
  geometry so they never become walls.
- [x] **Robust scale.** How: take the median of `value / drawn length`. Outliers
  (>12 % for DIMENSION entities, >3 % for text dimensions) are logged and ignored.
- [x] **Unit of the numbers.** How: in this order: explicit suffix, then the file header
  (if it agrees within 3 %), then the plausible flat size (with a decimal-format hint
  for metres). Imperial units are considered only for imperial drawings.
- [x] **Least-squares solver per axis.** How: all horizontal/vertical dimensions become
  equations `x_j − x_i = value` over clustered reference lines. The components are
  aligned to the global scale. A monotone piecewise-linear map (drawn → true
  coordinate) makes walls exactly as long as the written dimensions, even on plans
  that aren't drawn to scale. Chains that don't add up are reported.
- [x] **Missing dimensions derived from others** (chain gaps, overall minus parts).
  How: adjacent reference lines on one dimension line that have no dimension between
  them but are solved in the same component. They're shown in green and written to the
  DXF layer `DIMENSIONS_DERIVED`.
- [x] **No-dimension fallbacks.** How: file units, then user units, then paper scale 1:N
  (× DPI for scans), then the known overall width/depth (applied as an exact final
  scale), then an estimate that is always flagged with a warning.
- [x] **Dimension report.** How: `analysis/dimreport.py`. Each wall length and
  thickness and each opening width is labelled `stated`, `derived` or `computed`, along
  with the deviation between stated and final values.
- [x] **OCR for dimensions on bitmaps.** How: `importers/raster_dims.py`.
  - RapidOCR (pip, ONNX, reads rotated text) is used, with Tesseract as a fallback.
  - For each number, the dimension line is searched through or next to the text.
  - The line's extent ends at tick marks, at an arrow tip touching a wall face, or at the
    line end. Crossings by furniture lines are not ticks, and gaps under furniture fills
    are bridged.
  - Room-area labels (a number directly under a room name) are skipped.
  - The scale is a consensus (the largest agreeing cluster) and one scale per axis is
    fitted for bitmaps. Mis-measured dimensions are listed in the log.
- [~] **Interior arrow dimensions on bitmaps** are sometimes mis-measured: a line broken by
  more than about 2.5 text heights of furniture, or an arrow merged into the wall. Today
  they are rejected as outliers; the exterior chain dimensions carry the scale. How:
  follow the line past longer interruptions when it continues at the same row and the
  ink between is a filled area.
- [ ] **Angular / diagonal dimension chains.** How: extend the per-axis solver to any
  direction that has ≥2 parallel dimensions (group by angle instead of only x/y).
- [ ] **Room-area texts as a consistency check.** How: compare "12.5 m²" labels with
  the area of the polygon they sit in, and warn when they differ by more than 3 %.

## Phase 3: Walls and openings (`analysis/walls.py`, `analysis/openings.py`)
- [x] **Wall segment selection.** How: use the wall layer when one exists (multilingual
  keywords in `analysis/keywords.py`, or names the user enters). Otherwise use
  everything except door/window/dimension/furniture layers and blocks.
- [x] **Merge collinear pieces** so that triangulated sections and split lines don't matter.
- [x] **Wall rectangles from pairs of parallel faces.** How: the distance between faces
  must be within [min, max] thickness, and the overlap must be at least the thickness
  (this rejects pairs of jamb lines).
- [x] **Wall lines.** How: group the rectangles by direction and overlapping band. Thin
  rectangles between thick ones are glazing fillers; layered walls (insulation lines)
  are absorbed.
- [x] **Corners and T-junctions.** How: extend a wall end into a region bounded by a
  perpendicular wall, but only when that region isn't already covered.
- [x] **Openings.** Three sources:
  - gaps between consecutive wall pieces;
  - "sill sections" (between two jamb lines, with glazing lines that don't continue
    through the jambs), for windows drawn with lines across the opening;
  - free wall ends next to a perpendicular wall, accepted only with evidence.
- [x] **Door / window / passage classification.** How, in priority order:
  1. block/layer names;
  2. a door swing arc hinged at a jamb (single or double leaf);
  3. glazing lines;
  4. ink samples along the swing / glazing (bitmaps);
  5. the 3D model: wall below the opening means a window, then the ray-cast heights.

  With no evidence, openings up to 1.3 m are doors and wider ones are passages.
- [x] **Heights.** How: defaults are 2700 wall, 2100 door, 900/2100 window (editable in
  the UI). A 3D model supplies the real heights for each opening.
- [~] **Curved walls.** How: arcs on wall layers (r ≥ 1.5 m) are split into chords on a
  common 3° grid so the inner and outer faces pair up. Tested only on 3D sections, not
  on real curved CAD walls.
- [~] **Sliding doors vs windows.** Done so far:
  - An exterior "window" next to a room name such as taras / balkon / terrace / loggia /
    patio (OCR text or drawing text) becomes a door.
  - Wide doors without a drawn swing are drawn with a sliding symbol.

  Still to do: recognise the sliding-door symbol itself (two overlapping leaves offset in
  the gap).
- [x] **Door + fixed side panel** (balcony doors). How: leaf radii of 0.35 to 1.0 × the
  opening width are searched. The hinge may sit on either face or the centre line,
  slightly inside the jamb. Both the arc and the open leaf line must be drawn.
- [~] **Bitmap door swings that are only partly drawn** (for example the boiler-room door
  in `samples/real/house_plan.webp`) are read as a passage. How: accept an arc alone when
  it is long (≥ 70 %) and ends at the jamb.
- [ ] **Arched / non-rectangular openings, niches that don't go through the wall,
  columns.** How: store an opening profile (polygon in the wall plane) and build it
  with `manifold` booleans instead of the band extrusion.
- [ ] **Wall thickness steps within one wall line** (for example 300 → 250 sharing one
  face). How: split wall lines by band instead of using a single thick/thin ratio.
- [ ] **Furniture drawn on layer 0 without layers** (counters 600 deep fall just outside
  the default 550 max thickness). How: a classifier that uses the connectivity of the
  wall graph (real walls form a connected network ending at the outer boundary).
- [ ] **Stairs, shafts, balconies and railings.** How: detect them as separate element
  types and export them with their own layers or IFC classes.

## Phase 4: 3D model (`builder3d.py`)
- [x] **Watertight walls.** How: extrude the plan section for every height band (the
  band borders are all sill/head heights), then union the bands with manifold3d. The
  fallback is to concatenate them.
- [x] **Recesses.** How: doors get a lintel above the head; windows get a parapet below
  the sill and a lintel above the head.
- [x] **Floor** as a surface, or as a slab with a given thickness. It's left out of
  STL/3MF when it's a zero-thickness surface.
- [x] **Floor and ceiling placeholders.** Separate `floor` / `ceiling` objects (surface or
  slab, switchable in the settings, and shown or hidden in the 3D preview). They are
  IfcSlab in IFC. When a 3D model is imported, floor and ceiling slabs are detected and
  ignored, so sill and head heights are measured from the real floor.
- [ ] **Per-room floors and ceilings with room names** (from OCR or text). How: polygonise
  the room holes of the wall union and label each one with the text inside it.
- [ ] **Door leaves and window frames/glass** (optional furniture-level detail). How:
  parametric meshes placed from `Opening.swing` and the depth.
- [ ] **Per-room / sloped ceilings (attics).**

## Phase 5: Export
- [x] **2D: DXF 2018.** How: layers, solid hatch, door and window symbols, labels, and
  real DIMENSION entities for stated, derived and computed values. Units are mm.
- [x] **2D: DWG.** How: an R2000 DXF with plain-geometry dimensions goes through
  LibreDWG, or R2018 goes through ODA. The DWG is verified by converting it back.
- [x] **2D: SVG, PDF (vector 1:50), PNG, JSON.**
- [x] **3D: GLB/glTF** (metres, Y-up), **OBJ, STL, PLY, OFF, 3MF** (mm, Z-up), and
  **DXF 3D** (MESH).
- [x] **3D: IFC4.** How: wall pieces interrupted by openings are merged back into one
  IfcWall. That wall gets an IfcOpeningElement (void) plus an IfcDoor or IfcWindow
  (fill). Representation sizes are in SI units.
- [x] **Round-trip tests.** Our own DXF, DWG, IFC, GLB, PLY and 3MF import back into
  the same plan.
- [ ] **Collada (.dae) and FBX export.** How: pycollada writer (trimesh can read DAE but
  can't write it). FBX needs the Autodesk SDK or Blender in batch mode.
- [ ] **DWG test in AutoCAD / BricsCAD.** LibreDWG's writer is experimental. If users
  report problems, recommend ODA and make it the default in Docker, subject to its
  licence.
- [ ] **USDZ for iOS AR Quick Look.** How: `usd-core` Python package.

## Phase 6: Web app (`server.py`, `static/index.html`)
- [x] **FastAPI app.**
  - Upload endpoint with a size limit and an extension whitelist.
  - One background thread per job.
  - Live log over Server-Sent Events.
  - Job status JSON, downloads restricted to the job's own files (no path
    traversal), and re-export in more formats from the stored `plan.json`.
- [x] **Minimal page.**
  - Drag & drop upload, with collapsible options for heights, scale/units and
    detection.
  - Output formats chosen with checkboxes; unavailable ones are greyed out.
  - Colour-coded log.
  - 3D preview (three.js, vendored so it works offline), 2D preview with toggles for
    the dimension layers, and a report tab.
  - Download list, plus "export this format too".
- [x] **Job cleanup** after `FLOORPLAN_JOB_TTL_H`.
- [ ] **Manual correction in the 2D preview.** How: click an opening to change its type
  or heights, or draw a missing wall. Send the edits to `POST /api/jobs/{id}/plan`,
  then re-export.
- [ ] **Calibration by clicking.** How: pick two points on the preview and type the
  real distance. This is the same as `known_width_mm`, but for any segment.
- [ ] **Authentication and a persistent job store** (for multi-user deployment). How:
  a reverse proxy with auth, jobs in SQLite or Redis, and a worker queue (RQ/Celery)
  instead of threads.

## Phase 7: Quality
- [x] **pytest suite** (`tests/`).
  - Every sample format is checked against the ground truth: overall size, opening
    count, type and width, and wall area.
  - The derived dimension is checked.
  - Round-trips of the exports, and the HTTP API.
- [~] **Dockerfile** (builds LibreDWG). Written but not yet built: there was no Docker daemon in the
  development sandbox. Its LibreDWG and pip steps are the ones that were run and tested natively.
- [~] **A corpus of real-world plans** with hand-checked results. The first one is
  `samples/real/house_plan.webp`, an estate-agent bitmap.

  **Fixed for it:**
  - Rectilinear snapping of traced outlines. Slanted short faces had lost most piers.
  - A solid-black wall class, so thin partitions survive.
  - Bold text removed from the wall mask.
  - Solid junction blocks and columns kept as walls.
  - Gaps split at columns.
  - Windows only in exterior walls.
  - Openings up to 5 m.
  - Pixel seams welded.
  - SVG/PDF/PNG exports use presentation attributes: CSS was ignored by the renderers,
    which filled the door arcs black.

  **Next:** more plans (PDF exports from CAD, photos of paper plans), with an automated
  regression test per plan.
- [ ] **CI workflow** (GitHub Actions: `pip install -r requirements-dev.txt && pytest`).
- [ ] **Performance on large drawings.** Wall pairing is O(n²) within a direction
  family and is fine up to about 10k segments. Use an STRtree / sweep line for bigger
  inputs.
