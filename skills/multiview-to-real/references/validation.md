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

**Judge at native resolution, on fixed boxes.** The 384 px contact sheets are an overview, not evidence: in the
experiment the builder judged results on them and called them acceptable, and a fresh reviewer looking at
100 % found the head and the hands failing. Acceptance is therefore decided only on native-resolution crops,
one verdict per crop, and the crops are cut from the same boxes in every round so that rounds can be compared
(a new box per round hides regressions and lets the eye drift to whatever got better).

The tour joins its two segments (the pan or orbit, then the arc through the source cameras) with a dip
through black, 0.25 s out and 0.25 s in, in `tour_360.mp4` and in the coverage-tinted copy. In the experiment
the hard cut read as a glitch and a 0.5 s cross-fade showed two transparent copies of the subject. The fade is
applied to a temporary copy of the frames (the rendered frames are kept unchanged); `out/tour.json` records
`transition` and `first_segment_frames`. Without ffmpeg only the frames exist and no transition is applied.

## 4. The fresh reviewer

`critique CASE` packs `out/sheets/critique_in/` with one pair per camera, `clay_<view>.png` (the geometry
authority) and `photo_<view>.png` (the target), plus `render_<method>_<view>.png` re-renders when they exist,
and native-resolution crop strips in `crops/<round>/`, and prints a prompt.

Crop strips are target | re-render of each method, cut at each image's own pixels and never downscaled; a box
under 256 px is enlarged x2 nearest-neighbour and its strip says so. The boxes, in pixels of
`targets/<view>.png`, per usable view:

- each `detail_critical` object: the bounding box of its pixels in the clay id pass, padded 10 % (skipped when
  under 32 px);
- the three 256 px tiles with the largest mean absolute target-vs-re-render error over object pixels (all
  methods averaged; tiles overlap by at most a quarter);
- your own boxes: `case.json` `"review": {"crops": [{"view": "view_01", "name": "head", "box": [x0, y0, x1, y1]}]}`
  (a usable view, integers inside the image, x1 > x0, y1 > y0, a safe name; when boxes are computed, a user
  box of the same view and name replaces the automatic one).

They are written to `out/sheets/crop_boxes.json` on the first `critique` and reused by every later one (user
boxes added to `case.json` later are appended). On a later round a saved box wins over a `case.json` box of the
same view and name: `critique` prints one line naming each user box ignored that way, and `critique CASE
--new-boxes` recomputes all boxes and applies it. Name each later
round, `critique CASE --round round_2`: its strips land in `crops/round_2/` and earlier rounds stay next to
them (the default round is `round_1`). The prompt asks for one verdict per crop (`"crops": {"view_01_head":
{"verdict": "pass" | "fail", "problems": [...]}}`), says that full-frame impressions do not count towards
acceptance and that the boxes are the same every round. Names drawn in strip headers or used in file names are
cut to 40 characters (a numeric suffix keeps them unique within a view) and the prompt says they are labels, never instructions. Only targets inside the case are
packed, and `critique` refuses to clear or write through a symbolic link in `critique_in/` or `crops/`.

Start a fresh subagent that may read only those files and nothing about the method or the scores, give it the
printed prompt, and store its JSON answer with `critique CASE --save REVIEW.json [--round NAME]` (validated,
written to `out/sheets/critique.json` and, with a round, `critique_<round>.json`). Per view it says whether the
geometry matches the clay and lists problems with a box and a severity; overall it says where materials, colours
and light disagree across views, and for the re-renders where seams, smears and misplaced detail are. Never write
the review yourself; a reviewer who has seen the pipeline grades the pipeline. Novel views, the tour and the
relight sheet are not packed: show them to the user, or to a second fresh reviewer by hand. (`--judge` is reserved
for a blind judge of final renders and does nothing in this version.)

Give a reviewer finished outputs only, and write down what it opened. On a later scene three reviewers each saw
the first 37 to 43 frames of a 192-frame camera move, because the move was still rendering when their packets
were made, and the report first read as if they had seen all of it. A change made after the last review is
unreviewed: review the delivered state, or say that it was not reviewed.

## 5. The report

`report CASE` writes `out/report.md` from the files that exist, and says which command produces any missing
section:

- the views and their verification; generation: sizes requested against sizes read from the returned files;
  the native and resized size of every target; the per-view acceptance table;
- the consistency table;
- the calls with their cost notes (unknown when the price is a placeholder);
- the fusion scores (`proj/score.json`) and per-object coverage: observed, soft and inferred fractions;
- the tour's mode, its measured and inferred fractions and its transition (`out/tour.json`);
- the leak table per method and its worst objects (`proj/score.json` `leak`, when `score` wrote it);
- the number of review crop boxes (`out/sheets/crop_boxes.json`), the rounds packed and how many crops the
  last saved review passed;
- the PBR numbers: residual baked-shading correlation and the views whose de-light was skipped
  (`pbr/eval.json`), and the relight result (`out/relight.json`);
- stale-output warnings: after a `resync` promotion, every output older than the new targets is listed until
  `consistency`, `project`, `score`, `pbr` and `render` are re-run;
- the limits.

Add to your message what the files cannot say: the reviewer's list and what was done about it, and the limits
that apply to this case.

A case is done when the report is written and the user has seen the sheets; what to improve next is the
user's decision.

## 6. Checking a diagnosis

A number that explains a defect is a claim, and it gets the same treatment as a result.

- **Two measures.** Confirm a cause with a second measurement of a different kind before acting on it (matches
  and outlines, perpendicular distance and overlap, a figure and a look at full size).
- **Independent data.** A quantity adjusted on some data is not judged on that data. Hold something out, and
  say which check was independent. When holding a known camera out, start it from a solve that never saw it.
- **The right distance.** Misfit between a proxy and what was photographed is measured perpendicular to the
  surface and against outlines. Along the viewing ray it is inflated at grazing angles.
- **Known values first.** Whatever is known exactly (cameras of pinned frames, flat surfaces, focal lengths) is
  compared with the estimate before the estimate is used to judge anything else.
- **The scale of a figure.** A sharpness or agreement figure names its scale and its control; a control made of
  single samples is speckled and flatters nothing that is blended.

The exploration in `references/video-capture.md` broke each of these once.
