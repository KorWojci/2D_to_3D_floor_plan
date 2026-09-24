# Floor plan 2D → 3D

Web app that turns a 2D floor plan into a 3D model of the flat. It builds **walls**,
**door recesses** and **window recesses**, and makes the **dimensions of the result match
the dimensions written in the drawing**. Dimensions that are missing are worked out from
the ones that are there.

```
upload ──► import ──► calibrate from dimensions ──► detect walls / doors / windows ──► 3D ──► export
            (any 2D/3D format)   (units, scale, missing dims)                                   (2D + 3D formats)
```

## Quick start

```bash
pip install -r requirements.txt
./run.sh                        # http://localhost:8000
```

or with Docker. The image also builds LibreDWG, so DWG import and export work
out of the box:

```bash
docker build -t floorplan . && docker run -p 8000:8000 floorplan
```

The page is laid out like Visual Studio Code in dark mode:
- **Side bar:** import a file (drag & drop), pick the output formats, download the results, and open the optional settings.
- **Editor tabs:** the 3D preview you can orbit, the 2D plan with dimension layers you can switch on and off, and a report.
- **OUTPUT panel:** the live log. Drag its top edge to resize it.
- **Status bar:** the job state, the overall size and how the scale was found.

`Ctrl+B` hides or shows the side bar, and `Ctrl+Enter` starts the conversion.

## Settings reference

Every setting is optional. In the web page they sit in the side bar sections **Heights &
placeholders**, **Scale & units** and **Detection**. Through the API, send them as the JSON
`options` field of `POST /api/jobs` using the key shown. All lengths are in millimetres.

### Heights & placeholders

| Setting (API key) | Default | What it does |
|---|---|---|
| Wall height (`wall_height`) | 2700 | Height of all walls, from the floor to the ceiling. 3D models you import use the height measured in the model instead. |
| Door height (`door_height`) | 2100 | Top of door and passage recesses. Above it, a lintel fills the wall up to the ceiling. |
| Window sill (`window_sill`) | 900 | Bottom of window recesses. The wall is solid below it. |
| Window head (`window_head`) | 2100 | Top of window recesses. A lintel fills the wall above it. |
| Floor placeholder (`floor`) | on | Adds a separate `floor` object under the whole flat. Turn it off to export walls only. |
| Floor thickness (`floor_thickness`) | 0 | **0**: a flat surface at floor level, covering the rooms. **More than 0**: a slab of that thickness below the floor, covering the whole footprint. |
| Ceiling placeholder (`ceiling`) | on | Adds a separate `ceiling` object at wall height. |
| Ceiling thickness (`ceiling_thickness`) | 0 | **0**: a flat surface facing down. **More than 0**: a slab of that thickness on top of the walls. |

The placeholders are separate objects named `floor` and `ceiling` in GLB/glTF/OBJ/3MF/3D
DXF, and IfcSlab (FLOOR / ROOF) in IFC. IFC can't hold a zero-thickness slab, so there it is
200 mm thick unless you set a thickness. STL/3MF leave out zero-thickness surfaces so the
print solid stays watertight. The **floor / ceiling** checkboxes in the 3D preview only
change the view, not the files; the ceiling starts hidden so you can look inside.

### Scale & units: only needed when the drawing has no readable dimensions

| Setting (API key) | Default | What it does |
|---|---|---|
| Units of numbers (`units`) | auto | The unit of the numbers written in the drawing (`mm`, `cm`, `m`, `in`, `ft`). If the drawing has no dimensions, this is the unit of the drawing's coordinates. On *auto*, the unit comes from explicit suffixes ("3,45 m"), the file header, or the flat size that is plausible. |
| Paper scale 1 : (`paper_scale`) | – | The drawing scale of a PDF, SVG or scan, e.g. `50` for 1:50. Combined with the paper size (PDF/SVG) or the DPI (bitmaps). |
| Overall width (`known_width_mm`) | – | The real outer width (X) of the walls. The whole plan is scaled so that its wall outline has exactly this width. This overrides every other scale source and warns you if it disagrees with the drawing's own dimensions. |
| Overall depth (`known_height_mm`) | – | The same for the Y direction. If you give only one of the two, it is used for both axes. |
| Image DPI (`dpi`) | from file | The resolution of a scan, used together with the paper scale. Normally read from the image file. |

### Detection

| Setting (API key) | Default | What it does |
|---|---|---|
| Min wall thickness (`min_wall_thickness`) | 60 | Two parallel lines closer than this are not a wall. Lower it for very thin partitions. |
| Max wall thickness (`max_wall_thickness`) | 550 | Two parallel lines further apart than this are not a wall. Raise it for thick old walls (600+). Lowering it helps if kitchen counters or wardrobes turn into walls. |
| Min opening width (`min_opening`) | 350 | Gaps in a wall narrower than this are closed, not treated as openings. |
| Max opening width (`max_opening`) | 5000 | Wider gaps are treated as the end of a wall. Raise it for very wide glazing. |
| Wall layers (`wall_layers`) | auto | Comma-separated parts of CAD layer names that hold walls, e.g. `A-WALL, MUR`. Common names in many languages are recognised automatically. |
| Door layers (`door_layers`) | auto | The same for doors (layer or block names). |
| Window layers (`window_layers`) | auto | The same for windows. |
| 3D cut height (`slice_height`) | 1300 | 3D imports only: the height above the floor where the model is cut to read the plan. It must pass through windows and doors (between the sill and head heights). |

### Output formats

The API equivalent is the `formats` form field, a comma-separated list: `dxf, dwg, svg,
pdf, png, json, glb, gltf, obj, stl, ply, off, 3mf, dxf3d, ifc`. A finished job can export
more formats later (`POST /api/jobs/{id}/export`) without analysing the drawing again.

Sample drawings with a known ground truth are in `samples/`. Regenerate them with
`python samples/generate_samples.py`.

## Supported formats

| Import | Notes |
|---|---|
| **DXF** (all versions) | lines, polylines (incl. bulges / wide polylines), arcs, splines, blocks (nested, mirrored), hatches, MLINE, TEXT/MTEXT, linear & aligned DIMENSIONs; frozen/off layers ignored |
| **DWG** | converted to DXF with ODA File Converter or LibreDWG (`dwg2dxf`) |
| **PDF** | vector drawings (lines, curves, fills, text); scanned pages are traced as images |
| **SVG** | paths, shapes, transforms, text |
| **PNG, JPG, BMP, TIFF, WEBP, GIF** | walls recognised in any common drawing style (solid black, flat grey fill, diagonal or cross hatching, outlined walls between two lines, mixed styles) and snapped to the drawing's axes; pale coloured watermarks ignored; door swings (solid or dashed), glazing and sliding doors read from the thin lines; **dimension texts read by OCR** (RapidOCR) and measured on their dimension lines; **room area labels** ("12,71 m²", "A: 11,10 m²") check and correct the scale; room names such as "taras"/"balcony" mark terrace doors |
| **OBJ, STL, PLY, OFF, GLB, glTF, 3MF, DAE, IFC** | the model is cut at 1.3 m; a low cut and vertical ray casts separate doors from windows and **measure real sill and head heights** |

| Export 2D | Export 3D |
|---|---|
| **DXF** 2018: layers WALLS, WALLS_HATCH, DOORS, WINDOWS, OPENINGS, LABELS, DIMENSIONS / _DERIVED / _COMPUTED | **GLB / glTF** (metres, Y-up as the spec requires) |
| **DWG**, via ODA File Converter or LibreDWG, read back to verify | **OBJ, STL, PLY, OFF, 3MF** (mm, Z-up; STL/3MF watertight) |
| **SVG, PDF** (vector, 1:50), **PNG** | **IFC4** (IfcWall + IfcOpeningElement + IfcDoor / IfcWindow, IfcSlab floor/ceiling) |
| **JSON**: walls, openings and dimensions in mm | **DXF 3D** (MESH entities) |

Every import format is converted to one clean, layered plan in millimetres. That plan
is then written out as DXF/DWG, so any input can be "mapped to DWG".

## Cleaned plan (bitmaps)

Before the model is built, a bitmap plan is reduced to **walls and dimensions only**:
furniture, fixtures, electrical symbols, labels and watermarks are dropped. The result is
shown in the `cleaned.png` tab and can be downloaded as `<name>_cleaned.png`. The log
lists the wall patterns found and their share, e.g.
`Wall patterns found: flat grey fill (83%), diagonal hatching (17%)`.

| Pattern | How it is recognised |
|---|---|
| solid black | pixels ≤ 40 grey, opened to drop thin lines |
| flat grey fill | the dominant mid-grey tone (±10) |
| diagonal / cross hatching | several parallel slanted strokes close together, not part of axis-aligned lines; the cells between them and the wall outlines are filled |
| outline only (two lines) | paper strips between two parallel lines, as thick as the other walls (only with hatched plans) |
| thick dark strokes | fallback for anything else drawn heavy |

Each style is scored as a wall network (thin, elongated, spanning the drawing); the best
one is the main style and wall parts of the other styles that join it are merged (grey
exterior walls + hatched partitions). Small pieces stuck to walls (radiator brackets,
labels) and frame/sill strips that continue a wall across a window are removed.
Door and window symbols are still read from the original image.

## How the dimensions are kept correct

1. **Dimensions are read.** These are real DIMENSION entities, plus numbers written on
   dimension lines in PDFs, SVGs and exploded CAD drawings. For those, the tick and
   extension lines around each number mark the measured points.
2. **The unit of the numbers** (mm, cm, m, in, ft) comes from an explicit suffix, the
   file header, or the flat size that is plausible.
3. **Each axis is solved as a system of equations.** Every horizontal or vertical
   dimension says `x_j − x_i = value`. Least squares gives the true position of every
   reference line. The drawing is then mapped onto those positions piece by piece, so a
   plan that isn't drawn to scale still comes out with exactly the written lengths.
   Dimensions that disagree are reported in the log.
4. **Missing dimensions are derived.** Suppose a chain reads 1000 + 900 + **?** + 120
   + 4880 and the overall dimension is 10000. The gap is solved from the other numbers
   (here 3100) and drawn in green on a separate DXF layer.
5. **Everything else is computed** from the calibrated geometry: every wall length and
   thickness, and every opening width. These values are marked *computed*.
6. **Drawings with no dimensions** use, in order: the file units, the paper scale (1:N)
   and DPI, or a known overall width or depth that you enter. If none of these exist,
   the app makes a clearly flagged estimate.

The report tab and the JSON export label every value as `stated`, `derived` or
`computed`.

## Accuracy (measured by the test suite on the sample flat)

| Input | Overall size | Openings |
|---|---|---|
| DXF (layers / no layers / blocks + hatch / cm, drawn 3 % off-scale) | exact | exact, all 4 doors + 4 windows |
| PDF, SVG (dimensions as text) | exact | exact |
| OBJ / GLB / PLY / 3MF / IFC | exact | exact, sill and head measured from the model |
| PNG 200 dpi with paper scale | ±0.4 % | ±1 px (≈ 6 mm) |
| PNG with a known overall size | exact | ±1 px |
| Flat, grey walls (`samples/real/flat_gray_walls.webp`) | scale from dimensions, confirmed by 3 room area labels | 2 windows, balcony door, swing doors with dashed arcs, entrance read as a passage |
| House, cross-hatched + outlined walls (`samples/real/house_hatched.jpg`) | competing dimension readings, scale settled by the room area labels | 6 windows, swing doors; a few thin single-line partitions are not found |
| House with a coloured watermark (`samples/real/house_watermark.jpg`) | 11.99 × 6.87 m against the dimensioned 11.90 × 6.88 m | 6 windows, swing doors; some thin hatched partitions read as openings, a few furniture outlines kept as walls |
| Real estate-agent plan (`samples/real/house_plan.webp`, 676×806 px, ≈21 mm/px) | 9.03 m against the dimensioned 9.00 m (scale from the plan's own dimensions via OCR) | found: 5 windows, 7 swing doors (1 balcony door with side panel), 1 sliding terrace door (4.8 m), 1 entrance door; 1 swing door (boiler room) read as an open passage, 1 real passage |

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

The code is organised like this:

```
floorplan/
  importers/   dxf, dwg (via dwg.py), svg, pdf, raster, mesh/ifc  -> Drawing (source units)
  analysis/    calibration (units, scale, dimension solver), walls, openings, dimreport
  builder3d.py extrusion + recesses, manifold boolean union
  exporters/   export2d (dxf, dwg, svg, pdf, png, json), export3d (glb, gltf, obj, stl, ply, off, 3mf, dxf3d, ifc)
  pipeline.py  analyse() + export_all()
  server.py    FastAPI app: POST /api/jobs, SSE /api/jobs/{id}/events, downloads, re-export
  static/      index.html (+ vendored three.js for the 3D preview)
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `FLOORPLAN_DATA` | `./data/jobs` | where uploads and results are stored |
| `FLOORPLAN_MAX_UPLOAD_MB` | 200 | upload limit |
| `FLOORPLAN_JOB_TTL_H` | 24 | finished jobs are deleted after this many hours |
| `FLOORPLAN_ODA_PATH` | – | path to `ODAFileConverter`, if it is not on PATH |

DWG support needs one of these converters:
- **ODA File Converter**, the most compatible option. It's a free download from
  opendesign.com. Put it on PATH or set `FLOORPLAN_ODA_PATH`.
- **LibreDWG** (GPL): build it from source as the `Dockerfile` does. It writes DWG R2000.

See [TODO.md](TODO.md) for the roadmap and known limitations.
