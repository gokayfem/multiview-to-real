# Projection

`project CASE` writes the frozen targets into each object's own UV atlas. The geometry and the cameras are known
exactly, so there is no structure-from-motion and no camera estimate: every texel is projected through every
camera and accepted or rejected for a recorded reason.

## Texels

Each object's `UVMap` is rasterised at its atlas resolution into texel records: triangle, barycentric
coordinates, world position, world normal and the texel's world area. The atlas must have no overlapping
islands (`geometry.md`). The resolution is chosen per object from the geometry: for each view, the image area
covered by the triangles it sees divided by their UV area; the median over those triangles (the 85th percentile
for `detail_critical` objects) in the best view gives the texels needed, rounded to a power of two and clamped,
unless `case.json` `fusion.tex_size` fixes it. `project` prints the size and the source pixels per texel it
implies. Islands get bleed padding so filtering at their borders never reads empty texels; the lookup renderer
ignores texels that belong to no triangle. Outputs are written atomically. In the first run
2048² for most objects and 4096² for the skin and the sofa.

## Visibility tests (per view, per texel)

A texel's sample from a view is rejected, with its reason bit stored in `proj/rejection/`, when it is:

| Reason | Test |
|---|---|
| behind | camera-axis Z of the texel is not positive |
| off_image | the projection falls outside the frame (minus a border margin) |
| backface | the normal faces away from the camera |
| grazing | the cosine between the normal and the view ray is below about 0.15 |
| occluded / depth_ahead | the view's `depth_z` at the projection, interpolated in 1/Z, differs from the texel's Z by more than a tolerance that grows with distance and with the slope of the surface |
| wrong_object | the view's object id at the projection is not this object |
| edge_margin | within a hard margin of an id or depth discontinuity |
| border_margin | within a hard margin of the image border |
| self_edge | near the object's own silhouette (soft: kept with low confidence) |
| low_align | the target's alignment confidence is below a floor (0.3 in the first run) |
| mask_gate | rejected by the object's gate (`fusion.gate_policy`, below): outside its accepted pixels or inside another object's dilated generated mask |
| inconsistent | the sample's colour disagrees with the other views (soft: lowers the score, never a veto) |

Samples rejected only for `grazing` or `self_edge` are kept as low-confidence "soft" samples, used when nothing
better exists and counted separately in `coverage.json`.

The pixel margins are defined at 1664 px. The visibility margins (edges, border) and the `legacy` gate margins
scale up in proportion for larger cameras and stay at their 1664 px values for smaller ones. The `strict` gate
margins scale with the camera size in both directions and never go below 1 px: targets are resampled to the
calibration size, so their misregistration in pixels shrinks with it, and at 320 px unscaled margins removed a
fifth of a thin object's observed texels (0.36 to 0.29 of its surface) and cost it 1.7 dB against the truth.

Sampling taps never cross into another object's pixels or into pixels whose `position_world` disagrees with the
texel, so colour does not bleed across silhouettes.

**Generated masks are hard constraints.** The generated object and the proxy are never the same shape. Where the
proxy is wider than the generated object, the pixels just outside the generated silhouette show background; a
texel of the proxy that projects there would be painted with the wall. `case.json` `fusion.gate_policy` picks
the rule (`mvb2r/gating.py`); openings are never projected onto under either.

| rule (px at 1664; explicit case.json values used as given) | `strict` (default; defaults scale with the camera size both ways, never below 1 px) | `legacy` (defaults scale up only) |
|---|---|---|
| own clay region | eroded `erode_own` 2 px; 1 px if 2 px removes more than half of the object's pixels in that view, 0 px if 1 px still does (thin objects) | not eroded |
| object WITH its own generated mask | accept = own mask eroded `erode_gen` 2 px, AND NOT every other object's mask dilated `other_mask_dilate` 2 px | accept = own mask eroded 1 px |
| object WITHOUT one | reject the union of the other objects' masks dilated `reject_dilate` 8 px | dilated 4 px |
| occluder margin | reject pixels within `occ_margin` px of a NEARER other object's clay pixels (nearer = more than `occ_dz` 0.03 m closer); score x`occ_soft_factor` 0.3 out to `occ_soft` px. Without own mask 8 / 20 px, with one 2 / 6 px (`occ_margin_masked` / `occ_soft_masked`) | none |
| halo | none | score x`halo_factor` 0.3 within `halo_px` 6 px of the masked objects' clay |

An explicit `fusion.gate_dilate` (signed: negative erodes) or `objects[].gate_dilate` replaces the own-mask
erosion under both policies; `objects[].reject_dilate` replaces `reject_dilate` for one object (the first run
needed 4 px for a sofa under a seated figure: 8 px removed the cushion between the legs). `reject_dilate: 0`
still rejects a 1 px ring (the minimum under both policies). Every value is checked when the case is validated
(`validate_case`) and again by `project`, before any image is read: only known `fusion` keys (a misspelt key is
refused with the allowed list), `fusion.fill` one of `smooth` / `blocks`, finite, non-negative, pixel values at
most 256 (absolute), `occ_dz` at most 10 m, `within_mm` / `handover_mm` at most 1000, coordinates and radii at
most 1000 m, at most 64 contacts and 64 regions per object. Contact `surface` / `occluder` must name an object of
case.json `objects` or views/objects.json; a region `prefer_view` must name a camera of the case (one later
marked unusable in targets.json falls back to the object's `prefer_view` / the score rule, with a note). Error
messages quote names and values on one line, capped at 40 / 80 characters. `project` prints one line for every
setting it ignores: `regions` on an object that is not `detail_critical`, and `handover_mm` / `cut` /
`fusion.contacts` under `legacy`. `score` refuses a `proj/` made with another `gate_policy`, `fill` or set of
usable views than the case now gives (re-run `project`).
`legacy` reproduces the gates of the first version exactly (pinned by `tests/test_gating.py`).

Why `strict` is the default. In the first run's fix rounds (the experiment's own scripts, same targets and
fusion method; tables in `seams.md`) the original gate, which DILATED each generated mask by 2 px and protected
background objects only from the subject, let 4.3 % of the observed skin texels (231,691 of 5.44 M) take
sweater, hair or trouser pixels, 2.3 % of the knit top, 0.6 % of the trousers and 0.4 % of the shoes; a table, a
sofa rail and the shoes bled onto wall and floor. With the strict gate the leak count went to 0 for every
object, held-out PSNR (mean of five objects) rose from 28.17 to 28.29 dB and the novel-view seam step fell from
4.29 to 4.12. What the skin paid: sharpness 0.94 -> 0.85, observed area 0.237 -> 0.197 (more of it is smooth
same-object fill instead of wrong-object pixels).

This repository's `legacy` gate is NOT that original gate: it already eroded each mask by 1 px. Run with this
code on the same scene at 1664 px (method `regions-freqsplit`), `legacy` with the old fill against `strict` with
a face and two hand regions and a hairline contact zone:

| | `legacy`, `fill: blocks` | `strict`, regions, contact zone |
|---|---|---|
| leaked texels | 1,397 of 13.2 M (0.01 %) | 0 |
| held-out error, mean of six views (mean absolute x255) | 13.82 | 14.12 |
| held-out error with the close-up held out | 17.41 | 19.69 |
| held-out error, the other five views | 13.10 | 13.00 |
| novel-view seam ratio / texture seam ratio | 2.73 / 2.59 | 2.57 / 2.34 |
| sharpness kept | 1.00 | 1.00 |
| observed skin / floor / sofa | 0.209 / 0.247 / 0.425 | 0.192 / 0.243 / 0.416 |

So in this code base the leak count hardly separates the two, seams are lower with `strict`, and the mean
held-out error is slightly higher because of one view: with a single source for the face, the close-up is
predicted worse from the remaining cameras. On the crops the skin band under the hair and a dark smudge on the
neck are gone and there are fewer white fragments along the shoe rims; the hands and the three-quarter face
are unchanged. `strict` without regions or contact zones was not measured on this scene.

**Contact zones (experimental, opt-in).** `fusion.contacts: [{"surface": A, "occluder": B, "within_mm": 20,
"box": [[x0,y0,z0],[x1,y1,z1]]}]` (box optional, metres). Under `strict`, A's pixels whose 3D point lies within
`within_mm` of B's mesh (and inside `box`) ACCEPT B's generated-mask pixels (eroded 1 px) instead of rejecting
them; after the fill, A's unobserved texels there take B's median observed colour (weight 1 within 0.75 x
`within_mm`, 0 at 1.25 x). They stay unobserved / inferred in every map and count (`coverage.json`
`contact_fill` records them). Why: under strict, skin under a hairline that only ever saw hair pixels was
rejected and filled with skin colour, a pink band under the hair. Why `box`: without it the first attempt
painted dark bands down the cheek, because the hair hangs within 20 mm of it; the first run limited the zone to
above eye level and behind the head.

## The view score

Every accepted sample gets a score; fusion uses it to choose or weight views:

`s = cosθ · ppt / (ppt + ppt_half) · E(d_edge) · E(d_border) · align · Q(sharp)`

- `cosθ`: viewing angle.
- `ppt`: source pixels per texel, `sqrt(texel_area · fx · fy · cosθ / (Zc² · cosα))` with α the off-axis
  angle; it saturates, so a very close view does not win everything.
- `E(d)`: ramps from a floor (0.1) to 1 between a hard and a soft margin from id or depth edges and from the
  image border.
- `align`: the target's alignment confidence at the pixel.
- `Q(sharp)`: optional and down-weighted. Generated images are often sharpest where they invent detail, so
  sharpness is not evidence of truth (unlike for photographs).

`detail_critical` objects override the score with single-source regions (`mvb2r/handover.py`, `seams.md`).
Under `strict` each region gets ONE dominant camera: its `prefer_view` when that camera sees the region, else
the camera with the largest total score over it; its grazing / self-edge ("soft") samples count as valid inside
the region. Other cameras may supply only region texels at least `handover_mm` (default 5 mm, 3D) from what the
dominant camera covers; the band in between is filled. Regions are `objects[].regions: [{"name", "sphere":
[x, y, z, r] | "box": [[..], [..]], "prefer_view", "handover_mm", "cut"}]` in world metres, else one per connected
part. Label cells are half size (8 instead of 16 texels at 2048) on these objects. Under `legacy` each region
takes the first valid view of a priority list (`prefer_view`, then projected resolution). In the first run the
face, front neck, each hand and each shoe were regions; the face used `"cut": "hard"` and 12 mm. One source
and a gap remove a second camera's displaced copy next to the first. They do not make a photograph laid on a
proxy of a different shape look right from another angle: the independent review still found a doubled eye
and jaw in a three-quarter view after this change (`lessons.md`).

## Maps written per object

`proj/<method>/<object>/`: `texture.png` (the fused colour), `source.png` and `source_raw.png` (which view each
texel came from), `confidence.png`, `coverage.png` (number of valid views), `unobserved.png` (texels no view
accepted) and `soft_observed.png`. `proj/rejection/<object>/` holds the reason maps. `proj/coverage.json` records per object the observed, soft and inferred area fractions, the
area seen by two or more views and how often they disagree (max pairwise low-pass ΔE above 10), the seam ratio
per method, and the fill rule used.

## Inferred surfaces

Texels no view accepted are filled from the same object only and flagged in `unobserved.png`, by a smooth
Gaussian pull-push (`mvb2r/fill.py`): observed colours go down a Gaussian pyramid, and on the way up each finer
level is blended in with alpha = clip(support / 0.3, 0, 1), so the fill has no blocks. Far from any observation
it fades to a plain reference colour, so large never-seen surfaces carry no invented detail. For plain
wall-like classes (plaster, paint) that colour is the median of the well-lit observed texels (luminance between
the 50th and 90th percentile), and the fill is lifted to be no darker than it (at most x1.6; the lift fades in
over 96 texels and the fade to the colour takes 192, at a 2048 texture): the median of all texels includes
shadowed corners, and in the first run it made never-seen walls dark with halos around the observed parts. For
other classes it is the plain median. `fusion.fill: "blocks"` gives the box push-pull of the first version,
which showed blocks once stricter gating enlarged the holes. An object with almost nothing observed
takes the median of observed objects of the same class, else its class prior, and `coverage.json` says so. Openings (`opening: true`) are never projected; the
view through a window is the scene's own sky light.

## Outputs

`project` also builds `out/diagnostic.blend`: every object with its fused texture as an emission-style
appearance (switchable between fusion methods), packed, with a note that the textures contain the generated
lighting and shadows and are for alignment and source-view comparison only. It re-renders the source views
(`proj/<method>/rerender/view_XX.png`) and novel views between them (`proj/<method>/novel/`). A novel camera
that falls outside a detected room or mostly inside geometry is skipped, with the reason in
`proj/novel/cameras.json`; camera-move and tour frames are not checked this way. These consensus
re-renders are made by an exact UV lookup in numpy from Blender's UV, id and depth passes, not by rendering the
`.blend`; `resync` uses them. `--no-render` skips them, `--method all` (or `a,b`) runs several methods.

## Reading the first run

- **Coverage** (area-weighted, method C): floor 24%, back wall 28%, side walls 9 to 10%, ceiling 1%, wall
  behind the cameras 0%, sofa 41%, side table 37%, person parts 24% (skin) to 52% (trousers).
- **Disagreement where two views saw the same texel:** plain walls 0 to 3% of such texels above 10 ΔE; sofa 49%,
  skin 50%, knit top 34%, window frame 30%. These are the regions where any blend shows.
- **Runtime:** about 5.5 minutes to project fifteen objects with three methods from six 2144² views, three more
  for the diagnostic `.blend` and its re-renders, two for the evaluation.
