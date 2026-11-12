# Contract

Write this before the first paid call. It says what will be judged and how; do not change it after seeing
results. The full conventions are in `references/conventions.md`.

## Scene
- Input: (file name, kind, sha256 from case.json)
- Look: (case.json "look", copied here)
- References and their roles: (one line each)
- Cameras: (count, resolution, role of each view; existing or placed)
- Objects that are `detail_critical`, `refit` or `opening`: (names)

## Conventions every stage relies on
- World: Blender, Z up, metres. Camera: OpenCV (+X right, +Y down, +Z forward),
  `w2c = diag(1,-1,-1,1) · inverse(matrix_world)`, `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`.
- Pixels: origin at the top-left corner of the top-left pixel; pixel centres at +0.5; row 0 is the top row.
- `depth_z` is camera-axis Z in metres, never ray length; `views` proves it for this case.
- Grid: rows × cols, row-major, exact cells, no gutters; cell rectangles in `grid/layout.json`, measured again on
  the returned sheet.
- One non-overlapping UV map `UVMap` and one stable `pass_index` per object.

## Case layout
`case.json`, `input/`, `blockout.blend`, `views/`, `grid/`, `gen/`, `targets/`, `proj/`, `pbr/`, `out/` (see
`references/conventions.md`). Targets are frozen once written; every method comparison uses the same frozen set.

## Acceptance of a refined view (fixed now, never lowered later)
- Subject silhouette IoU against the clay: not more than ___ below the cell's own IoU.
- Mean contour distance: at most ___ px. Landmark offset: at most ___ px.
- Colour drift against the cell: at most ___ ΔE.
- Visual pass on full-resolution crops (faces, hands, feet, edges).
- At most ___ retry per view, each with a written reason; otherwise the cell, labelled as a resize.

## Budget
- Paid calls: at most ___ (`budget.max_calls`). Cost recorded per call; a `$0` placeholder is "unknown".

## Privacy and security (check before sharing this case)
- This case folder records local paths: `case.json` `input.source_dir`, the `--schema-file` path in
  `gen/model.json` if one was used, and file names in logs. Remove or redact them before sharing the folder.
- Images linked from the input `.blend` were packed only if they live inside the input's folder tree
  (`new --allow-external-images` widens this; say here if it was used and why).
- Only images from inside this folder are uploaded to the model provider; downloads are https only, from the
  provider's allowed hosts, size-limited and decoded as images.
- `FAL_KEY` is never written here.

## What success means
1. The re-renders from the source cameras match their targets on observed pixels, and the novel views and the
   camera move show no seams, ghosting or painted-on geometry that a fresh reviewer flags as severe.
2. The fusion method is chosen by leave-one-view-out and seam visibility, not by source-view reproduction.
3. Coverage is reported per object; inferred surfaces are plain and labelled; the coverage-tinted tour exists.
4. The diagnostic `.blend` says it contains lighting. The PBR `.blend` relights, and its residual baked shading
   is reported with a number.
5. Every size, resample, call and cost is in the report.

## Out of scope for this case
- (list: e.g. glass and mirrors, strands, animation, surfaces no camera sees, exact identity)

## Stop rule
- Resync is opt-in (`resync CASE` or `run --resync`; one paid call per view): at most ___ round(s), kept only if
  the consistency score improves and alignment holds; after a promotion, re-run consistency, project, score, pbr
  and render.
- Refit only objects marked `refit`, from several views' masks.
