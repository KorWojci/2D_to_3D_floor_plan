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

The page has four parts. You **import** a file (drag & drop) and can open optional
settings. The **log** shows each step as it runs. The **preview** has three tabs: a 3D
view you can orbit, the 2D plan with dimension layers you can switch on and off, and a
report. **Downloads** lists the output files in the formats you picked, and you can add
more formats to a finished job.

Sample drawings with a known ground truth are in `samples/`. Regenerate them with
`python samples/generate_samples.py`.

## Supported formats

| Import | Notes |
|---|---|
| **DXF** (all versions) | lines, polylines (incl. bulges / wide polylines), arcs, splines, blocks (nested, mirrored), hatches, MLINE, TEXT/MTEXT, linear & aligned DIMENSIONs; frozen/off layers ignored |
| **DWG** | converted to DXF with ODA File Converter or LibreDWG (`dwg2dxf`) |
| **PDF** | vector drawings (lines, curves, fills, text); scanned pages are traced as images |
| **SVG** | paths, shapes, transforms, text |
| **PNG, JPG, BMP, TIFF, WEBP, GIF** | walls traced from thick strokes, door swings / glazing read from thin lines |
| **OBJ, STL, PLY, OFF, GLB, glTF, 3MF, DAE, IFC** | the model is cut at 1.3 m; a low cut and vertical ray casts separate doors from windows and **measure real sill and head heights** |

| Export 2D | Export 3D |
|---|---|
| **DXF** 2018: layers WALLS, WALLS_HATCH, DOORS, WINDOWS, OPENINGS, LABELS, DIMENSIONS / _DERIVED / _COMPUTED | **GLB / glTF** (metres, Y-up as the spec requires) |
| **DWG**, via ODA File Converter or LibreDWG, read back to verify | **OBJ, STL, PLY, OFF, 3MF** (mm, Z-up; STL/3MF watertight) |
| **SVG, PDF** (vector, 1:50), **PNG** | **IFC4** (IfcWall + IfcOpeningElement + IfcDoor / IfcWindow) |
| **JSON**: walls, openings and dimensions in mm | **DXF 3D** (MESH entities) |

Every import format is converted to one clean, layered plan in millimetres. That plan
is then written out as DXF/DWG, so any input can be "mapped to DWG".

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
