# Consistency

Generated views of one scene are not photographs of one surface. Each is a separate interpretation, even when
all views come from one call. `consistency CASE` measures how far they agree, before any projection, so you know
where seams and blotches will come from.

## The score

For every ordered pair of views (A, B), every valid pixel of B is mapped into A through B's `position_world`
and A's `P`. A pixel counts only if:

- it lands inside A's frame and is visible in A (`|depth_z_A - Zc| < 1 cm + 1%`);
- it has the same object id in both views and is more than a few pixels from any id boundary in both;
- its alignment confidence is high in both views (the same pixel set is used for every image set compared);
- it is not an opening or background.

A is sampled bilinearly at the mapped point. Then, inside each object's region (normalised convolution, so
neighbours do not leak in):

- **low band** (Gaussian, sigma about 8 px): mean CIE76 ΔE. This is lighting and colour agreement, what makes
  brightness steps at seams.
- **high band** (difference of Gaussians, luminance): normalised cross-correlation (1 = the same detail in the
  same place, 0 = unrelated) and SSIM. This is "do the invented details line up", what makes double patterns
  and smeared fabric when views are mixed.
- **full band:** ΔE and SSIM.

Results are reported per object class and per view (mean over the pairs that involve the view), pixel-weighted.

Any image set can be scored with `consistency CASE --set NAME=PATTERN` (the pattern contains `{view}`; the
default is `targets/{view}.png`), restricted with `--views`.

**Sanity check.** Score the consensus re-renders of the projected scene (one shared texture), for example
`--set consensus=proj/regions/rerender/{view}.png`: they should score close to 1 in the high band and near 0 ΔE
in the low band. In the first run they scored 0.947 and 0.07; the residual is resampling and defocus.

## What the first run measured

Six views, 30 ordered pairs, 34.2 M samples. First targets (each view refined independently) against the
targets after one render-then-refine round:

| class | low-band ΔE | high-band correlation |
|---|---|---|
| floor | 6.92 → 3.79 | 0.207 → 0.489 |
| walls | 3.97 → 2.80 | 0.091 → 0.179 |
| sofa (linen weave) | 7.19 → 4.22 | 0.145 → 0.173 |
| table (grain) | 6.09 → 3.47 | 0.224 → 0.220 |
| skin | 6.93 → 6.38 | 0.210 → 0.256 |
| hair | 3.11 → 3.22 | -0.003 → 0.065 |
| top (knit) | 4.51 → 3.75 | 0.058 → 0.059 |
| trousers | 1.84 → 1.30 | 0.210 → 0.296 |
| shoes | 4.79 → 6.09 | 0.142 → 0.102 |
| all | 5.19 → 3.38 | 0.161 → 0.244 |

Reading it:

- **Fine detail barely agrees.** 0.16 between independently generated views against 0.95 for one shared
  texture. Seams are therefore not only a fusion problem: no fusion rule can make two different weaves line up.
- **The round fixed lighting more than detail.** Low-band ΔE fell 35% overall and 41 to 45% on floor, sofa and
  table. Coarse, high-contrast patterns (floor boards, plaster marks, trouser creases) were inherited from the
  re-render; linen weave, knit, table grain and hair were re-invented per view.
- **Small objects can get worse.** Shoes went from 4.79 to 6.09 ΔE (42 thousand samples).

## When to run `resync`

`resync` is opt-in: it is not part of a default `run` (use `resync CASE` or `run CASE --resync`), because it
spends one paid call per view and, when its targets are promoted, makes every later output stale until
`consistency`, `project`, `score`, `pbr` and `render` are re-run (the report warns about stale outputs). Run one
render-then-refine round when:

- the low band is large (several ΔE) on big plain objects such as floors, walls and upholstery, and the
  re-renders show brightness steps or pasted sun patches at seams; and
- alignment is good enough that the consensus re-render is a fair layout reference.

`resync` re-renders the fused scene from every camera, sends each re-render with the accepted target, the clay
view and the references to the image model with a prompt that makes the re-render the authority for where every
joint, grain and pattern is and asks only for seams, patch edges, brightness steps, smears and duplicates to be
repaired, and freezes new targets. It is kept only if (the shipped limits are in `templates/acceptance.json`, copied
into each case; the first run's stricter values are given here):

- the overall consistency improves and no view's low band gets worse by more than a set margin (0.5 ΔE in the
  first run);
- each view's high band improves;
- silhouette IoU stays within 0.02 of the previous targets, contour distance and landmark offset within the
  generation limits; and
- a visual pass shows the re-render's artefacts were removed, not copied.

In the first run all six views passed after one retry for the elevated wide view (its first attempt shifted
the person: IoU 0.852 to 0.813). Do not expect a second round to make weave or knit agree; stop after one unless
the numbers give a reason.

## What it does not measure

It does not say which view is right, only where views disagree. A consistent hallucination in all views scores
well. It ignores surfaces seen by only one view. It is computed on the frozen targets at calibration
resolution, so it cannot see detail that only exists in the native files.
