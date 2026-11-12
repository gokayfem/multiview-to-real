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
| mask_gate | outside this object's generated mask, or inside another object's dilated generated mask |
| inconsistent | the sample's colour disagrees with the other views (soft: lowers the score, never a veto) |

Samples rejected only for `grazing` or `self_edge` are kept as low-confidence "soft" samples, used when nothing
better exists and counted separately in `coverage.json`.

The pixel margins (edges, border, mask dilation, halo) are defined at 1664 px; for larger cameras they scale up
in proportion, for smaller ones they stay at their 1664 px values.

Sampling taps never cross into another object's pixels or into pixels whose `position_world` disagrees with the
texel, so colour does not bleed across silhouettes.

**Generated masks are hard constraints.** The generated object and the proxy are never the same shape. Where the
proxy is wider than the generated object, the pixels just outside the generated silhouette show background; a
texel of the proxy that projects there would be painted with the wall. So an object with its own generated mask
samples only inside it (dilated 2 px by default); objects without one reject the union of the other objects'
masks dilated 4 px, and get a low-confidence halo (score x0.3 within 6 px) around those objects' clay
silhouettes, so another view wins there. Openings are never projected onto. The first run used the same rule
(with a 14 px halo).

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

`detail_critical` objects override the score: each connected region takes one source view (`prefer_view`, or
the view with the highest projected resolution), with an ordered fallback, and is never blended. In the first
run the face and front neck came from the close-up, each hand and each shoe from one view.

## Maps written per object

`proj/<method>/<object>/`: `texture.png` (the fused colour), `source.png` and `source_raw.png` (which view each
texel came from), `confidence.png`, `coverage.png` (number of valid views), `unobserved.png` (texels no view
accepted) and `soft_observed.png`. `proj/rejection/<object>/` holds the reason maps. `proj/coverage.json` records per object the observed, soft and inferred area fractions, the
area seen by two or more views and how often they disagree (max pairwise low-pass ΔE above 10), the seam ratio
per method, and the fill rule used.

## Inferred surfaces

Texels no view accepted are filled from the same object only and flagged in `unobserved.png`: near observed
texels a push-pull fill continues the observed colours; farther away it fades to the object's robust median
colour, so large never-seen surfaces are plain, with no invented detail. An object with almost nothing observed
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
