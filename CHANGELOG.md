# Changelog

## [0.1.0] — 2026-11-12

First version.

- An agent skill that turns any grey 3D scene (a `.blend` with several, one or no cameras, a glTF/GLB, OBJ, FBX,
  USD, PLY or STL file, a JSON scene spec, or an idea the agent writes as a spec) into the same scene textured
  from several calibrated views of one generated photoreal multi-view sheet: exact clay views and passes from
  N cameras with numeric verification, one multi-reference grid edit, per-view refinement under a fixed
  acceptance rule, frozen targets with alignment confidence and per-object generated masks, a projector-free
  cross-view consistency score, visibility-aware fusion into each object's own UV atlas with several fusion
  methods, leave-one-view-out and seam-visibility scoring, an optional render-then-refine round, an optional
  silhouette-carving refit, a baked-appearance diagnostic `.blend` and a relightable PBR `.blend`.
- `mvb2r.py` CLI: `doctor`, `new`, `cameras`, `views`, `grid`, `probe`, `generate`, `refine`, `targets`,
  `consistency`, `project`, `score`, `resync`, `refit`, `pbr`, `render`, `critique`, `report` and `run`.
- Input-agnostic: everything scene-specific is written into the case's `case.json`; any fal multi-reference edit
  endpoint can be used after `probe` has checked its schema and the size it really returns; 2 to 9 views.
- Offline paths: `targets --images` for your own views and `--generator mock`, a synthetic image model with the
  defects of generated views, used by the end-to-end round-trip test.
- Tests (`python3 tests/run_tests.py`, 2026-11-12, macOS, Blender 5.0.1 found): 119 of 119 passed, 0 skipped. Two runs by hand: the spec example on the mock generator through every stage
  (about 70 s at 320 px, 4 automatic views), and the first run's real 15-object scene with its six cameras and
  existing generated images through `targets --images`, `consistency`, `project` and `score` (low-band agreement
  3.40 ΔE against the experiment's 3.38). Implemented but not yet exercised: `probe --check-size`, glTF, FBX, USD, PLY and STL inputs and `--group-by`.
- Found by the first CI run: `opencv-python-headless` 5.0 wheels have no OpenEXR codec, so every test that
  touched a depth, position or normal pass failed on the runner. EXR reading and writing is now built in
  (`mvb2r/exrio.py`, pure Python and numpy, bit-identical to OpenCV 4.13 on the pass files); CI runs the
  suite against OpenCV 4.x and 5.x.
- Fixed before release, found in review: `run` order (`probe` now runs before `grid`; `resync` is opt-in);
  the CI key check; a leave-one-view-out leak that favoured `register`; a flipped automatic camera arc;
  containment of external images and linked libraries (refused outside the input's folder unless
  `new --allow-external-images`); texture size now chosen from the geometry, with bleed padding around UV
  islands; `refit --apply` (keeps `blockout_pre_refit.blend`).
- Found on the second input (the bundled primitives spec, run with real Seedream calls), fixed before release: the
  image model returned one panel as a near-copy of another and it was fused. Every frozen view is now gated
  (object IoU, a reliably segmented object, the image must match its own clay better than any other view's, no
  near-duplicate of another view's image, no shifted panel); a failing view gets `usable: false` in
  `targets.json`, moves to `targets/rejected/` and is skipped by `consistency` and fusion; fewer than two usable
  views stops the stage. Re-run on that case, the duplicate is rejected (image matches the neighbour's clay
  at 0.686 against 0.365 for its own; 0.992 edge correlation with the neighbour's image) and the three remaining
  views re-render cleanly; consistency went from 12.1 to 9.2 ΔE (low band) and 0.04 to 0.06 (high band).
- The fusion `auto` default is `regions-freqsplit` (seam-aware labels for the detail, low band blended at σ 8 px
  at 1664 px), the winner of the seam bake-off on the first run's targets.
- The paid path was exercised once end to end on the second input (probe on the live schema through score).
- Requires Blender 4.2 or later (checked); developed and tested on 5.0.1.
- Fixed before release, found on the real scene: clay renders of an input whose world image used a relative
  path came out magenta (external images are now resolved from the input's folder, and a missing world image is
  replaced by a neutral grey world with a warning); the automatic texture size was too small for a
  detail-critical object (now chosen from the best view's pixels per texel); contact sheets were written for the
  first fusion method only.
- Known limits: no sun-direction search, white-balance fit or plain-material flattening in the PBR stage;
  `critique --judge` is reserved; the shipped acceptance thresholds are looser than the first run's.
- `run` order: views, probe, grid, generate, refine, targets, consistency, project, score, pbr, render, report.
  `resync` is opt-in (`run --resync` or by hand); `probe --check-size` is a separate paid step after `grid`.
- Platforms: developed and tested on macOS with Blender 5.0.1; Linux expected to work (CI runs the offline tests
  on Ubuntu without Blender); Windows untested.
- Ports from Blockout 2 Real: input ingestion, the JSON scene spec, the doctor, budgeted fal calls and the
  handling of untrusted input, the test runner and CI.
- Adapters for Claude Code, Codex, Cursor, Copilot, Gemini CLI and chat assistants.
- Built from a first full run on one room with a seated person (six views, 26 logged image-model calls, cost
  unknown because the documented price was a `$0` placeholder); its findings are in
  `skills/multiview-blockout-to-real/references/lessons.md`.
