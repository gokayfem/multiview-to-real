# Conventions

The data contracts every stage depends on. Read this before writing a new tool or changing a stage.
`scripts/mvb2r/common.py` is the code form of this page; `templates/CONTRACT.md` is the short public version
copied into every case.

## World and camera

- **World:** Blender, Z up, metres, right-handed.
- **Camera:** OpenCV axes: +X right, +Y down, +Z forward.
  `w2c = diag(1, -1, -1, 1) · inverse(matrix_world)`, where `matrix_world` is Blender's camera-to-world matrix
  (Blender cameras look along -Z with +Y up).
- **Projection:** `u = fx · X / Z + cx`, `v = fy · Y / Z + cy` with (X, Y, Z) in camera coordinates. `P = K ·
  w2c[:3, :]`.
- **Pixels:** the origin is the top-left corner of the top-left pixel; u to the right, v down; the centre of
  pixel (i, j) is at (i + 0.5, j + 0.5). Row 0 is the top row of the image as a normal image reader returns it.
- **Depth:** `depth_z` is camera-axis Z in metres (distance along the optical axis, positive in front), never
  ray length. In a room the two differ by tens of centimetres (up to 0.91 m in the first run), so confusing them
  breaks every visibility test. `views` proves it per case: `depth_z - Zc` must be at float precision and the
  ray-length difference must be clearly non-zero. In Blender 5.0 the Cycles Z pass was verified to be camera-
  axis Z (median difference 1.5e-6 m); the check is run again per case rather than assumed.

## cameras.json

One entry per view: `name` (`view_01` ...), `camera` (the Blender object), `role`, `width`, `height`,
`pixel_aspect`, `lens_mm`, `sensor_width_mm`, `sensor_height_mm`, `sensor_fit`, `shift_x`, `shift_y`,
`clip_start`, `clip_end`, `K` (3x3), `c2w_blender` (4x4), `w2c_opencv` (4x4), `P` (3x4), plus a `convention`
text that repeats the rules above.

## Per-view passes (`views/view_XX/`)

| File | Type | Meaning |
|---|---|---|
| `clay.png` | 8-bit sRGB | grey beauty render (Standard view transform, no look), opaque |
| `depth_z.exr` | float32, one channel | camera-axis Z in metres; background is a large value (>= 1e9) |
| `position_world.exr` | float32 RGB | world XYZ of the first hit, metres |
| `normal_world.exr` | float32 RGB | world-space unit normal |
| `object_id.png` | 8 or 16-bit, one channel | object `pass_index` per pixel, 0 = background or sky, not anti-aliased |

The data passes are one sample at the pixel centre: no anti-aliasing across edges, no denoising.

EXR files are read and written by the tool's own reader (`mvb2r/exrio.py`: single-part scanline files, HALF or
FLOAT channels, compression NONE, ZIPS or ZIP), so no OpenEXR codec is needed in OpenCV or anywhere else. A
single-channel EXR written by Blender stores the value in a channel named `Y`; three channels come back as RGB.
Other EXR variants (tiled, multi-part, PIZ, DWA) are refused with a message that says how to re-save them.

## Objects

- One mesh object is one unit of everything downstream: one id, one UV map, one atlas, one row in every
  per-object table.
- `id` is Blender `pass_index`: unique, >= 1, stable for the life of the case, never reused. 0 means none.
- UV map `UVMap`: islands inside [0, 1]^2 with no overlapping texels. `new` audits every object and repairs
  overlaps (downloaded assets often have stacked or overlapping islands); a projection into an overlapping atlas
  writes two surfaces into one texel.
- Names are also file-name parts: letters, digits, space, `_`, `.`, `-` only, no `..`, a bounded length. Names
  that do not fit are renamed at ingest and rejected in a spec.

## Grid

`grid/clay_grid_master.png` holds the cells at calibration resolution (checked bit for bit against each clay
view); `grid/clay_grid_conditioning.png` is the same sheet at the generation size and is what is sent. Row-major cells, numbered from 1: with `rows x cols`, cell N has `x0 = ((N - 1) % cols) · cell_w`, `y0 = ((N -
1) // cols) · cell_h`, no gutters, no borders, no labels. `grid/layout.json` records the cell rectangles and
the view each cell holds. A separate annotated sheet is for people only and is never sent to the model.
Returned sheets are not assumed to be exact: `generate` measures the seams and stores per-cell rectangles.

## Case layout

```text
cases/<case>/
  case.json                 the only scene-specific file (below)
  CONTRACT.md               what will be judged, written before the first paid call
  input/                    the input copied with its sha256
  blockout.blend            grey copy: one UV map `UVMap` per object (0 overlap), stable pass_index ids
  views/                    cameras.json, objects.json, landmarks.json (optional), view_XX/{clay.png,depth_z.exr,position_world.exr,normal_world.exr,object_id.png}, verify/
  grid/                     clay grid master, conditioning copy, layout.json, references manifest
  gen/                      calls.jsonl, model.json, grid_edit/, cells/, refined/, round_K/, qa/
  targets/                  frozen: view_XX.png (calibration resolution), native paths, align_conf_XX.png, mask_<object>_XX.png, targets.json
  proj/                     <method>/<object>/{texture,source,confidence,coverage,unobserved,...}.png, coverage.json, rejection/, rerender/, novel/, score.json
  refit/                    refit.json and sheets (with blockout_refit.blend next to blockout.blend)
  pbr/                      shading/, delit/, albedo/, textures/, rig_fit.json, eval.json
  out/                      diagnostic.blend, final.blend, renders/, novel/, camera_move.mp4, tour_360.mp4, tour_360_coverage.mp4, tour.json, relight.jpg, relight.json, sheets/, report.md
```

A stage reads only files written by earlier stages and fails with a message that names the missing file and
the command that produces it. Cases live outside the repository (`cases/` is git-ignored).

## case.json

```json
{
  "name": "attic",
  "input": {"path": "input/attic.blend", "sha256": "…", "kind": "blend"},
  "look": "place, style, time of day, light — the agent writes this",
  "references": [{"path": "input/ref_person.png", "role": "identity", "applies_to": ["Person_*"], "note": "who/what it defines"}],
  "objects": [{"name": "Sofa", "id": 10, "description": "material, colour, finish", "class": "fabric",
               "detail_critical": false, "prefer_view": null, "refit": false, "opening": false}],
  "views": {"resolution": 1664, "cameras": [{"name": "view_01", "role": "frontal wide", "source": "existing"}],
            "auto": {"count": 6, "arc_degrees": 120, "closeup": null, "elevated": true}},
  "grid": {"rows": 2, "cols": 3},
  "model": {"provider": "fal", "edit_endpoint": "bytedance/seedream/v5.5/edit",
            "t2i_endpoint": "bytedance/seedream/v5.5/text-to-image", "generator": "fal"},
  "budget": {"max_calls": 30},
  "fusion": {"method": "auto"},
  "pbr": {"delight": "conservative"}
}
```

- `input.kind`: `blend`, `mesh` or `spec`.
- `class` is one of plaster, paint, wood, fabric, knit, leather, skin, hair, metal, plastic, stone, generic. It
  selects material priors (albedo range, roughness, specular, bump) in `mvb2r/material_priors.json`; the prior
  bounds an object's mean albedo and never replaces its projected texture.
- `detail_critical`: one coherent source view per connected region, never blended (faces, hands, text, logos).
  `prefer_view` names it; otherwise the view with the highest projected resolution wins.
- `refit`: the geometry may be carved to the generated silhouettes (hair, cloth, soft props).
- `opening`: an emissive opening such as a window to the sky; never projected onto.
- `group` (optional, per object): objects sharing a group are also checked as one union in the generation QA
  (for example all parts of a person).
- `model.generator`: `fal` or `mock` (the offline synthetic image model).
- `model.download_hosts`: host suffixes downloads may come from (default `fal.media`, `fal.run`,
  `storage.googleapis.com`). `model.pixel_cap`, `model.max_refs`, `model.size_param`, `model.image_param`:
  overrides for an endpoint whose schema `probe` cannot read.
- `grid.source`: `auto` (rows and columns chosen by `grid`) or `user` (yours are kept).
- `input.external_images`: images and libraries the input linked from outside its folder, and what was done with
  each (refused by default; `new --allow-external-images` trusts them).
- `budget.max_calls`: a hard stop on image-model calls for the case. It counts every started call, failed ones
  included; cache hits are free; a call that would exceed it is refused before it starts; `--max-calls` and the
  environment variable `MVB2R_MAX_CALLS` override it. There are no automatic retries.

## Run bookkeeping

- `logs/run_summary.json`: the stages `run` executed, their flags, and STALE outputs (files older than the
  targets they depend on, for example after a resync promotion); `run CASE --from project` refreshes them.
- `targets/targets.json` carries a `generation` counter that increases each time targets are replaced.
- A failed `new` leaves `case.failed.json` and can be re-run.
- `$MVB2R_BLENDER_TIMEOUT` sets the time limit of each Blender call; Blender 4.2 or later is required.
- A JSON scene spec must hold finite numbers, at most 2000 objects and at most 500 parts per object.

## Optional files the agent writes

- `views/landmarks.json`: named world points (head top, hands, feet ...) whose projected positions the QA checks
  against the generated outlines.
- `gen/qa/visual.json`: your visual verdict per view; it gates acceptance when the case's `acceptance.json` sets
  `visual_review_required`.
- `acceptance.json` in the case (copied from `templates/acceptance.json` by `new`): the acceptance thresholds,
  pixel values stated at 1664 px and scaled to the case's calibration width. Edit it before results exist.

## Colour

Images are stored as 8-bit sRGB PNG unless stated; fusion, levelling and de-lighting work in linear light.
Renders that are compared with targets use the Standard view transform with a fixed exposure for every view.

## Calls log

`gen/calls.jsonl`, one line per paid call: endpoint, sanitised arguments, input hashes, request id, the
returned dimensions read from the file, elapsed time and a cost note (`unknown` when the price is not
documented or is a `$0` placeholder). The key is never in it.

## Safety rules in code

- Blender is started with `--factory-startup --disable-autoexec`; `.blend` files are opened with scripts
  disabled. No tool runs a script stored in a file.
- Names from files are sanitised before they become paths; free text from files is cut to one printable line.
- Nothing from a model response is executed; downloads are size-limited and type-checked.
- `FAL_KEY` is read from the environment only.

## Honesty rules in code

- A failed call or render stops the stage. No stage writes a stand-in.
- Resamples are recorded with their factor and method; a resize is never called generated detail.
- Unobserved texels are filled from the same object and flagged in the `unobserved` map and `coverage.json`.
- The diagnostic `.blend` carries a note that it contains baked lighting.
