# Validation

Three kinds of evidence, in this order: the tools are right on inputs with known answers; the fusion predicts
views it did not see; and a reviewer who has not seen the work looks at the renders.

## 1. Round trip with a known answer

Before trusting projection on a new machine or after changing a projection stage, run a synthetic round trip:
a known texture on known geometry, rendered from the cameras, projected back, rendered again.

- `tests/test_roundtrip.py` does this on a tiny primitives scene with known textures, 4 cameras at 320 px and
  views perturbed the way generated views are (per-view gain and tint fields, smooth local warps of a few pixels,
  view-specific fine detail): it runs `project --method all` and `score` and checks texture PSNR against the
  truth, object leakage, UV orientation, single-source regions and coverage bookkeeping. A lit variant checks
  that de-lighting lowers the shading correlation. `tests/test_case.py` runs the core stages on the mock
  generator (`--generator mock`) in Blender. Both run when Blender is found and skip with a message otherwise.
- A full case on the mock generator (`new … --generator mock`, then `run`) exercises every stage end to end
  offline.
- What it catches: flipped axes, ray length used as depth, off-by-half-pixel sampling, colour bleeding between
  objects, texels painted from views that cannot see them, seams that appear even with perfect inputs.
- On a real case, the clean form of the same check is to project the clay renders themselves: the result must
  reproduce the clay views closely, with seams only where the clay's own lighting differs.

## 2. Leave-one-view-out

`score CASE` removes each view in turn, fuses from the others, renders into the held-out camera and compares
with the held-out target on pixels the remaining views observed (`seams.md`). It is the only measure here that
tests prediction rather than reproduction. Report it per object and per method next to the seam ratios,
sharpness retention and coverage. Expect the high band to stay low for every method: no fusion can predict
detail the held-out view invented on its own.

## 3. What to look at

Numbers are not enough. After `project`, `pbr` and `render`, look at full-resolution crops:

- the source-view re-renders next to their targets: faces, hands, feet, fabric, floors, edges against walls;
- novel views between source cameras (no target exists: fills and seams show here first);
- the camera move and the coverage-tinted tour: the measured and inferred fractions per frame;
- the relight sheet: the PBR scene changes, the diagnostic does not, the ghost of the old light is visible.

Post them labelled, images first.

## 4. The fresh reviewer

`critique CASE` packs `out/sheets/critique_in/` with one pair per camera, `clay_<view>.png` (the geometry
authority) and `photo_<view>.png` (the target), plus `render_<method>_<view>.png` re-renders when they exist,
and prints a prompt. Start a fresh subagent that may read only those files and nothing about the method or the
scores, give it the printed prompt, and store its JSON answer with `critique CASE --save REVIEW.json` (validated,
written to `out/sheets/critique.json`). Per view it says whether the geometry matches the clay and lists
problems with a box and a severity; overall it says where materials, colours and light disagree across views,
and for the re-renders where seams, smears and misplaced detail are. Never write the review yourself; a reviewer
who has seen the pipeline grades the pipeline. Novel views, the tour and the relight sheet are not packed:
show them to the user, or to a second fresh reviewer by hand. (`--judge` is reserved for a blind judge of
final renders and does nothing in this version.)

## 5. The report

`report CASE` writes `out/report.md` from the files that exist, and says which command produces any missing
section:

- the views and their verification; generation: sizes requested against sizes read from the returned files;
  the native and resized size of every target; the per-view acceptance table;
- the consistency table;
- the calls with their cost notes (unknown when the price is a placeholder);
- the fusion scores (`proj/score.json`) and per-object coverage: observed, soft and inferred fractions;
- the tour's mode and its measured and inferred fractions (`out/tour.json`);
- the PBR numbers: residual baked-shading correlation and the views whose de-light was skipped
  (`pbr/eval.json`), and the relight result (`out/relight.json`);
- stale-output warnings: after a `resync` promotion, every output older than the new targets is listed until
  `consistency`, `project`, `score`, `pbr` and `render` are re-run;
- the limits.

Add to your message what the files cannot say: the reviewer's list and what was done about it, and the limits
that apply to this case.

A case is done when the report is written and the user has seen the sheets; what to improve next is the
user's decision.
