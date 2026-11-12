# Changelog

## [Unreleased]

**Renamed to `multiview-to-real`** (was `multiview-blockout-to-real`): the GitHub repository, the skill and its
folder (`skills/multiview-to-real/`), the plugin and marketplace ids, the Gemini extension, the Copilot and
Cursor adapter files, the `--version` string and the download User-Agent. The display name (Multiview Blockout
2 Real), the CLI (`mvb2r.py`), the library package (`mvb2r`) and the case layout are unchanged, and nothing
else in the code changed: 46 lines in 23 files, each the name alone. An installed copy under the old name
keeps working but no longer updates; install again under the new name (README, Install). GitHub redirects the
old repository URL.

The findings of a third input, a rigid scene without people (two walls, a floor, an oak cabinet of five parts
and a terracotta block; six cameras; ten logged Seedream calls, cost unknown). That run was made with a
separate set of experiment scripts, not with this repository's CLI. This part is documentation only; the suite
was not re-run for it (no test reads the documents).

- README: a new section, "A third input", with four gallery images: the clay grid and the accepted sheet, a
  registration overlay, clay / target / unlit re-render / relightable render for two views, and a relight
  pair. It states which numbers come from that run's own tools and are not comparable with the tables of the
  first run.
- `references/lessons.md`: nineteen new entries marked *(third input)* and four existing entries extended
  with what was seen again. The main ones: every per-view refinement sat further from the geometry than the
  cell it came from; whole rigid objects were displaced, each by its own amount (up to 31 px at 2144 px), and
  a bounded map per object brought edges back to 1 to 2 px, which qualifies "no global warp" for rigid scenes;
  the same surface differed between views by a luminance factor of 0.3 to 3.4; the reference image's unwanted
  features were copied into every view; one sentence per part fixed grain direction across the sheet; a tone
  fitted per object turned one oak into five; reviewers saw only the frames that existed when their packets
  were made.
- `references/generation.md`, `consistency.md` and `validation.md`: short notes that point to those entries
  (check the references and state texture direction before the sheet call; a refinement can cost alignment;
  an exact-size sheet can have seams 8 px off the lattice; rigid objects and displacement; what a reviewer
  was given).
- Known limits: this repository has no stage that registers a generated view to the calibrated geometry per
  object. How its gates and scores behave on the third input has not been measured; running the CLI on that
  scene with `targets --images` is the open next step.

The findings of an exploration of video capture on the first run's room (a driving video of the grey scene
re-rendered by a reference-to-video model, its frames projected instead of six separate views). It was made
with experiment scripts tied to that scene, not with this repository's CLI. Documentation only: no stage
changed.

- `references/video-capture.md` (new): what was run, what each check showed (frames of one video agree with
  each other where separate stills do not; a zooming clip needs per-frame intrinsics; a driven video keeps the
  clay's outlines, so the proxy fits; dense stereo from frame pairs and its limits), the two projection passes
  with their numbers, and what a stage would need.
- `references/lessons.md`: eleven new entries under "Video capture", among them three diagnoses that were
  reported and then withdrawn after a second measurement.
- `references/validation.md`: section 6, "Checking a diagnosis" (two measures of different kinds, independent
  data, the right distance, known values first, the scale of a figure).
- `references/conventions.md`: the pixel under a projected point is `floor`, not a rounding; OpenCV's camera
  functions want `cx - 0.5`, `cy - 0.5`. This toolkit's own lookups were checked and already follow it.
- README: a section on the exploration with two gallery images, and two known limits (video capture is not
  implemented; `uv_blur` feathers across UV islands).
- Tests (`python3 tests/run_tests.py`, 2026-11-12, macOS, Blender 5.0.1 found): 231 of 231 passed, 0 skipped
  (run on the renamed tree with these documents in place).

## [0.2.0] — 2026-11-12

The fix rounds that followed an independent review of the first run, ported into the toolkit, reviewed and run
once on real data.

- **Gate policy.** `fusion.gate_policy`: `strict` (new default) or `legacy` (the 0.1.0 gates, pinned bit for
  bit). Strict erodes the object's own region (with a fallback for thin objects) and its generated mask, rejects
  every other object's mask, and keeps a hard and a soft margin around every nearer object's silhouette. Strict
  margins scale with the camera size in both directions, never below 1 px.
- **Single-source regions with a gap.** A detail-critical region takes one dominant camera; other cameras start
  only `handover_mm` away, and the gap is filled. `objects[].regions` defines regions inside a mesh (a sphere or
  a box in world metres, a preferred view, a feathered or hard cut).
- **Contact zones (experimental, opt-in).** `fusion.contacts`: where one surface passes under another object,
  it takes that object's photographed colour instead of a fill of its own, inside a spatial limit.
- **Leak count in `score`**, and a leave-one-view-out evaluation mask that does not depend on the gate policy or
  on any margin setting. `score` refuses to run when the policy, fill or view list differ from what `project`
  recorded.
- **Smooth fill.** A Gaussian pull-push replaces the box push-pull; plain wall-like classes fade to the median of
  their well-lit texels. `fusion.fill: "blocks"` with `gate_policy: "legacy"` reproduces 0.1.0 exactly.
- **Review at full size.** `critique` writes native-resolution crop strips on fixed boxes that are reused in
  every round (`--round NAME`, `--new-boxes`, `review.crops` in `case.json`); the reviewer gives one verdict per
  crop. The report lists the gate policy, the leak, the tour transition and the review rounds.
- **Tour.** The two segments are joined by a dip through black.
- **Direct-light patch synthesis (opt-in).** `pbr --synth-direct` with the surfaces named in
  `pbr.synth_objects` replaces the generated photo's own direct-light patch with texture copied from the shaded
  part of the same object, marks it inferred and reports the source size. The relight test's old-patch measure
  now compares each object with itself.
- **A missing sky image no longer renders magenta** in the relightable stages: every Blender script on that side
  replaces it with the clay stage's neutral grey and prints a warning.
- **Validation.** Unknown keys under `fusion` are an error that lists the allowed ones; every new value is
  type-checked and bounded (pixel settings at most 256, list lengths at most 64, distances finite).
- **Hardening found in a security review of this change:** nothing is deleted or written through a symbolic
  link planted in a case folder (`common.owned_path`; the first version of the new `critique` clean-up would
  have emptied the link's target; the JSON writer's temporary file and `out/report.md` are covered too); object names can no longer put a line of their own into error text; files
  named in `targets.json` are checked to be inside the case before they are copied.
- **Found in code review and fixed before release:** the gap around a single-source region could be refilled
  from another camera's low-confidence samples; `legacy` with `blocks` did not reproduce 0.1.0 for plaster and
  paint; the floor step kept a bright line where a patch met its UV chart's edge, and reported a patch as
  filled when no block of its source fitted; the neutral stand-in for a missing sky would have dropped the
  original world from saved files (it is now kept in the file).
- **Found on real data and fixed before release:** fixed-size margins swallowed a thin object at 320 px; the
  floor step skipped the floor, selected furniture by class, looked for Blender's direct-light channel under
  the wrong name and painted stacked strips; the magenta sky above.
- Tests (`python3 tests/run_tests.py`, 2026-11-12, macOS, Blender 5.0.1 found): 231 of 231 passed, 0 skipped
  (123 in 0.1.0).
- Runs by hand with this code and the first run's existing images: old against new gating on the 15-object
  room at 1664 px (leak 0.01% against 0; held-out error 13.82 against 14.12, higher only with the close-up
  held out; novel-view seam ratio 2.73 against 2.57), and the whole chain once at 512 px including
  `pbr --synth-direct`, `render --relight --tour`, `critique` and `report`. `strict` without regions or a
  contact zone was not measured on the real scene; no paid call was made for this release.
- Documentation: the first-run section of the README now carries the independent review, the three fix rounds
  and the reviewer's final verdict; `references/lessons.md` has fourteen new entries; the reference pages no
  longer repeat a builder's claim the review contradicted.
- Known limits added: texture work does not repair geometry; objects keep their photographed shading under new
  light; the floor step passes at full frame and fails close up; the leak count is blind to content displaced
  beyond the margins.

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
  `skills/multiview-to-real/references/lessons.md`.
