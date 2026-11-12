# Generation

All views are generated together in one multi-reference edit, then refined one by one under a fixed acceptance
rule. N independent edits of N clay views are not a multi-view method: each would be a separate
interpretation of the scene with nothing tying them together.

## The grid

`grid CASE` (after `probe`, which it needs) tiles the clay views into one image of `rows x cols` exact cells (row-major, no gutters, borders or
labels; `conventions.md`). Rows and columns are the fewest panels that hold the views without a very elongated
sheet; unavoidable empty panels are flat grey and the prompt says so. The cell size is the largest integer size
with the views' aspect whose whole sheet fits the verified pixel cap, sides multiples of 8. The master
(`clay_grid_master.png`) is at calibration resolution and checked bit for bit against each clay view; the
conditioning copy (`clay_grid_conditioning.png`) at the generation size is what is sent, and its resize is
recorded.

`grid/` also gets an annotated review sheet (labels, borders) for people. It is never sent to the model.

## The references, in order

The edit gets, in this order, as recorded in the references manifest:

1. the clay grid (defines the layout and every camera's framing);
2. the reference images from `case.json`, each with its role (identity, product, material, style);
3. each full-resolution clay view, in cell order.

The prompt names every image by its position and says what it is for: image 1 is the layout; each reference
image defines only what its `note` says (an identity reference, for example, only the face, hair and clothing of
one person); the remaining images are higher-resolution copies of the cells, top-left to bottom-right, to be
used as geometry references for their cells, not as extra panels. It asks for exactly `rows x cols` panels, the same moment and one light setup
fixed in world space, the same pose, scale, contact and occlusion as the clay, and no merged panels, borders,
text, extra people or moved furniture. `grid` checks that the references are distinct files (sha256). When the
endpoint's verified reference limit is exceeded, the per-view clay copies are dropped from the last view
backwards (the first views keep theirs), and `grid/references.json` lists what was dropped.

Two things to settle before the call, both from a later rigid scene (`lessons.md`, *third input*):

- **Look at each reference for features the geometry does not have.** A reference is copied closely. A darker
  band across a cabinet's doors in the appearance reference came back in all six views; an edit asking to
  remove it from the reference changed nothing. Replace the reference or accept the feature in writing.
- **State the direction of a directional texture for each part** (wood grain, brushed metal, weave, planks):
  "along the length of the top, horizontal on the doors, vertical on the side panels". Without it one sheet
  drew door grain horizontally in two panels and vertically in four; with it all six agreed.

## Probing sizes

`probe CASE` runs before `grid` (the grid's cell size comes from the pixel cap). It fetches the endpoint's schema
(or reads a saved one with `--schema-file`) and records in `gen/model.json` the parameters it accepts, the
reference limit and the pixel cap. It understands fal endpoints shaped like Seedream's: an `image_size`
parameter taking `{width, height}` and a list of image URLs. For any other schema it refuses and asks you to
set `model.pixel_cap`, `model.max_refs` and the size-parameter overrides in `case.json`. `probe CASE
--check-size` is a separate, explicit paid step after `grid`: one call at the largest in-spec size of the
sheet, recording what came back (implemented, not yet exercised). Rules from the first run:

- **Over-cap requests are not refused; they are downsized.** Asked for 4992 x 3328 (16.6 MP, against a
  documented cap of 4,624,220 pixels), the edit endpoint returned 2504 x 1672 with no error. The cells then had
  fractional boundaries (834.67 px).
- **The response does not tell you.** `images[].width` and `height` were null in every response. Read the size
  from the returned file, always.
- **The largest native size seen was 2628 x 1752** (4,604,256 pixels, 99.6% of the cap). Requests at or under
  the cap came back exactly as asked (2628 x 1752 sheets, 2144 x 2144 refinements).
- **Requesting more pixels adds no detail.** A resize after the fact is a resize, recorded with its factor and
  method, never called generated detail.
- **Only schema parameters.** No invented seed, strength, negative prompt, ControlNet or camera control.

## The edit

`generate CASE` makes one call at the largest verified native size. Then it measures the panel seams (the
peak of the mean adjacent-pixel difference within a few percent of each ideal seam), extracts each cell with its
measured rectangle and writes QA sheets. In the first run the seams of the accepted asset-based sheet were all
within 2 px of the ideal 876 px grid; the earlier, downsized sheet was off by up to 5.3 px. On a later sheet
returned at exactly the requested 2592 x 1728, the row seam was 8 px from the 864 px lattice, so an exact size
does not mean exact cells.

Look at the whole sheet and at every cell next to its clay view. Reject and regenerate (within the budget)
when panels are merged, an object moved, a person was added or the pose changed. A size probe at a slightly
different size once produced an open-mouthed close-up and a differently coloured table: a different call is a
different interpretation, so keep the accepted sheet unless a new one is better on what you can see.

## Refinement

`refine CASE` makes one edit per view at the largest verified square or view-aspect size (2144 x 2144 in the
first run; the code takes the largest size with the view's aspect under the verified cap), with these
references in order: the clay view, its cell, the accepted sheet, then the case's reference images (dropped
from the end at the model's limit, and said in the log). A retry is `refine CASE --views view_XX --attempt 2`. The prompt asks for a single full-frame photograph of that view, with the geometry of the
clay view, the appearance of the cell, and no new scene detail. Refinement adds real detail (weave, grain, knit,
skin, hair strands) because the model draws at a higher resolution, not because pixels were added afterwards.

A refinement is also a new picture, and it can sit further from the geometry than its cell. On a later rigid
scene every one of six did (global shift 5.7 to 11.3 px at 2144 px against 2.6 to 9.9 px for the cells in the
same units), and the views were no closer to each other in colour afterwards. That is what the acceptance rule
below is for; read its alignment numbers next to the cell's before keeping a refinement for its sharpness.

## Acceptance

The rule is written in the case before any refinement is looked at (`CONTRACT.md` and the case's
`acceptance.json`, whose shipped defaults are the first run's values: IoU margin 0.03, contour distance 12 px,
landmark offset 10 px, colour drift 8 ΔE, stated at 1664 px and scaled to the case) and is never lowered after a
failure. It is applied per object that covers a minimum fraction of the view and whose generated mask is
reliable enough; landmarks are checked when `views/landmarks.json` exists; your visual verdict
(`gen/qa/visual.json`) gates acceptance when `visual_review_required` is set. The first run used:

| Check | Threshold |
|---|---|
| Person (or subject) silhouette IoU against the clay | not more than 0.03 below the cell's own IoU |
| Mean contour distance | at most 12 px at the calibration resolution |
| Landmark offset (head top) | at most 10 px |
| Colour drift against the cell | at most 8 ΔE (CIE76) on the subject |
| Visual pass | full-resolution crops of face, hands, feet and edges: one person, plausible hands, no new objects |

Silhouettes come from GrabCut seeded by the clay object ids, with a band tight enough that dark shadows under
furniture do not join the subject (a 40 px band once let under-sofa shadow join the person and dropped the
IoU to 0.81). One of six refinements failed in the asset-based run (IoU 0.883 against 0.889 and a head 11.5 px
low); its one retry passed. A view that still fails falls back to its cell, upsampled and labelled as a resize.

No registration warp is applied unless two independent estimators (mask fit and edge-based ECC) agree within 2
px with a shift of at most 4 px and a scale within 1%. In the first run they never agreed: the residuals were
local (pose, silhouette), and a global warp would have moved correct pixels to fix wrong ones.

On a rigid scene the residuals have another shape. Whole objects were displaced, each by its own amount (up
to 31 px at 2144 px), and one bounded map per object, estimated against the calibrated passes and applied only
when sampling colour, brought the edges back to 1 to 2 px. This repository has no such stage. For a scene of
rigid objects with straight edges, overlay every object's clay edges on its target before `project` and
report the displacement per object (`lessons.md`).

## Targets

`targets CASE` freezes the accepted images: each is resampled to the calibration resolution (area
interpolation when downsampling; recorded), and the native file stays the source for projection. It writes:

- `align_conf_XX.png` (255 = trustworthy): lowered where the generated masks disagree with the clay ids,
  dilated and feathered, and capped near the subject's boundary where generated limbs are slimmer than the
  proxy.
- `mask_<object>_XX.png`: one generated mask per object of interest, disjoint, at the calibration resolution.
  Projection accepts pixels for an object only inside its own mask, so background beside a slimmer generated
  arm is not painted onto a wider proxy arm.
- `targets.json`: per view, the source and its sha256, native and frozen sizes, resample factor and method,
  checks, mask paths and their reliability.

Every frozen view is also gated (`acceptance.json` "cell"): its objects must overlap their clay ids (mean and
minimum object IoU), at least one object must be reliably segmented, the image must match its own clay better
than any other view's (a swapped or wrong panel), it must not be a near-duplicate of another view's image while
their clays differ, and its content must not be shifted. A failing view gets `usable: false` with its reasons,
moves to `targets/rejected/`, and is skipped by `consistency` and fusion; fewer than two usable views stops the
stage. This gate exists because the image model once returned a panel as a copy of its neighbour (`lessons.md`).

Once frozen, targets do not change; every comparison of methods uses the same frozen set.

## Budget

`budget.max_calls` in `case.json` is a hard stop: it counts every started call, failed ones included, cache
hits are free, and a call that would exceed it is refused before it starts (`--max-calls` and
`MVB2R_MAX_CALLS` override it). The first run made 26 calls in all: one text-to-image call
for the character reference, four sheet edits (including a size probe and a test), 21 per-view edits across
three rounds (including three retries). There are no automatic retries: a retry is a decision with a reason,
written in the log.
