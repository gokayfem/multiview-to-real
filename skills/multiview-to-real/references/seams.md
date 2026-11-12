# Seams and fusion

Several views have valid samples for most observed texels. Fusion decides which view, or which mix of views,
each texel takes. Every method runs on the same frozen targets, cameras, geometry and masks, so they can be
compared. `project CASE --method NAME` runs one (`best`, `blend`, `regions`, `freqsplit`, `regions-freqsplit`,
`multiband`, `graphcut`, `register`; the letters `A`, `B`, `C` are accepted for the first three).
`fusion.method: "auto"` in `case.json` resolves to `regions-freqsplit`, the winner of the bake-off below
(per-class overrides are possible), and `detail_critical` objects always take one source per region. Run `score`
on your own case and pass `--method` explicitly when its numbers disagree.

## Why the classic method needs changes

Waechter, Moehrle and Goesele ("Let There Be Color!", ECCV 2014) select one view per face with a Markov random
field (data term: how good the view is; smoothness term: a cost for every seam), drop views whose colour
disagrees with the majority, then level colours globally and Poisson-blend a strip along each seam. It assumes
that all images observe one shared, roughly Lambertian radiance, that differences between views are exposure,
white balance and small misregistration, and that the majority of views is right. Generated views break each
assumption:

| Assumption | Why it breaks | Adaptation |
|---|---|---|
| views sample one radiance | each view is its own interpretation; shading, shadows and fine texture differ | never average views that disagree; blend only where low-pass colour agrees within a gate |
| sharpness means quality | generated images are sharp where they invent | score on geometry and alignment confidence; sharpness optional and down-weighted |
| the majority is right | a consistent hallucination can win; the only face may be in one view | agreement is a soft penalty, not a veto; `detail_critical` regions take one source |
| detail is shared | weave, knit and faces differ per view | one source per region for detail; blending (if any) only in the low band |
| global levelling across the scene | it moves one object's shading onto another | levelling per object, in linear light as gains, between adjacent regions of the same object, robust, clamped, with mismatched pairs rejected |

## The methods

- **A, best view (`best`).** Each region takes the highest-scoring valid view. Sharp, but every switch between views is
  a visible seam, and a view's baked shadow or sun patch is pasted where that view wins.
- **B, agreement-gated blend (`blend`).** A score-weighted blend of the views whose low-pass colour agrees with a
  reference; one source inside `detail_critical` regions. Lowest seam contrast, but fabric and plaster blur
  into blotches, because the details being averaged do not line up.
- **C, seam-aware labelling with levelling (`regions`).** Labels on small UV cells, chosen by alpha-expansion graph cuts
  (data term from the view score and agreement, Potts smoothness that also links cells across UV cuts by 3D
  proximity), then per-object gain levelling between adjacent labels and a narrow feather. One source per
  region keeps detail; the levelling hides most brightness steps. The default in the first run.
- **Frequency split (`freqsplit`).** The low band of all valid views is blended (in linear light, within each
  object's own pixels); the high band comes from the one labelled source per texel. Brightness steps vanish,
  detail stays single-source.
- **C labels + frequency split (`regions-freqsplit`, the `auto` default).** The high band from `regions`'
  seam-aware labels, the low band (σ 8 px at 1664 px) blended from every valid view. The bake-off winner.
- **Multi-band (`multiband`).** Burt and Adelson's blending, in log space: each band is blended with the label
  mask feathered by a width that grows with the band's scale.
- **Difference-aware labelling + multi-band (`graphcut`).** Labels from graph cuts with a pairwise cost that
  measures how different the two candidate views are on both sides of a cut (after Kwatra et al. and Lempitsky
  and Ivanov), so cuts go where the views already agree; then multi-band blending.
- **Registration first (`register`).** Each view is warped slightly (bounded, smooth optical flow) to the
  consensus, then multi-band. It can reduce double edges, and it can also move correct pixels, so it is scored
  like the rest.

## How methods are scored

Image metrics on the source views barely separate the methods, because each source view is rebuilt mostly from
its own pixels. In the first run the observed-pixel PSNR of the six re-renders was 27.6 dB for A, 27.6 for B and
27.0 for C, while the crops showed A pasting a sun patch and a person's shadow into other views and B blurring
the linen. Score instead:

- **Leave-one-view-out (LOO).** Remove one view, fuse from the others, render into the held-out camera by direct
  UV lookup, and compare with the held-out target on pixels that the other views observed (away from edges,
  inside the object's own generated mask, high alignment confidence). Repeat for every view. This is the
  only number that asks whether a method predicts a view it did not see.
- **Seam visibility.** In texture space and in novel views: the mean colour step across label boundaries,
  divided by the mean step between neighbouring texels inside regions, at a fine and a coarse scale. 1 means
  seams are as visible as ordinary texture.
- **Sharpness retention.** High-frequency energy of the fused texture relative to the sources, so a method
  cannot win by blurring.
- **Boundary length** per square metre: how fragmented the labelling is.

Choose by LOO first, then seam visibility in novel views, with sharpness as a floor, and confirm on full-
resolution crops. `score CASE [--methods a,b] [--no-loo]` writes `proj/score.json` and the contact sheets
(`out/sheets/view_XX_<method>.jpg`: clay, target, re-render; `view_XX_methods.jpg`: target and every method).
It reports LOO as the mean absolute error x255 in sRGB, full, low band and high band (lower is better); the
bake-off below used PSNR, so the two tables are not on the same scale. `register` falls back to `multiband` when
it cannot run (no consensus re-render).

On the repository's synthetic round trip (views that differ by gain, tint, small warps and invented detail),
LOO error is 9.04 for `register`, 9.01 `blend`, 9.06 `freqsplit`, 9.22 `multiband`, 9.75 `best` and 9.88
`regions`: the methods are close, and no method should be called best from LOO alone. An earlier lead for
`register` came from a measurement leak and is gone. On the first run's real scene at 512 px, `regions` and
`multiband` scored 9.51 and 9.58 LOO, 2.01 and 1.87 novel-view seam ratio.

## Bake-off results

Every candidate was run on the first run's frozen targets: first the independently refined set, then the
consistency-round set. Scores: leave-one-view-out PSNR (pixel-weighted; the close-up view dominates the pixel
count), the coarse seam step in four novel views (8-bit levels; lower is better) and sharpness retention.
Means over floor, back wall, sofa, knit top and skin:

| Method (bake-off name) | LOO PSNR (dB), first / second set | Novel-view coarse seam step, first / second | Sharpness, first set |
|---|---|---|---|
| A, best view | 25.02 / not run | 6.54 / not run | 0.83 |
| C, labelling + levelling (`regions`) | 25.57 / 27.92 | 5.31 / 4.76 | 0.76 |
| **C labels + low band blended at σ 8 px (C+S8, `regions-freqsplit`)** | **25.64 / 28.17** | 4.49 / 4.29 | 0.75 |
| best view + low band blended at σ 16 px (A+S16) | 25.58 / 28.11 | 4.47 / 4.04 | 0.79 |
| 4-band multi-band (MB) | 25.52 / 28.05 | 4.43 / 4.26 | 0.74 |

σ is stated at the 1664 px reference. On a four-object subset the gated blend (B)
scored 26.32 dB against 26.58 for the winner with sharpness 0.61 (it blurs), and difference-aware graph cuts
alone 26.17. Registration to the consensus before the split gave no real gain on the floor (24.39 against 24.40
dB) and took about ten times as long (477 s against 41 s).

**Per material.**

| Material | Finding |
|---|---|
| plaster walls | the split wins clearly (+0.7 dB on the first set, seams about 30% lower); on the second set C's LOO is 0.3 dB higher on the back wall but its rectangles stay visible |
| oak floor | the split wins (+0.5 dB first set, +0.4 dB second) |
| linen sofa | tie on the first set (view mean), +0.9 dB on the second |
| knit top | plain labelling better on the first set, the split better on the second (pixel-weighted 29.5 against 29.1) |
| skin, face, hands | within 0.1 dB: single source, never averaged |

**What fusion cannot fix.** The fine detail of the prediction correlates with the held-out view only at 0.02 to
0.03 on knit, 0.04 to 0.07 on plaster, 0.12 to 0.17 on linen and 0.14 to 0.39 on the floor (first set to
second set); a view's own re-render scores 0.34 to 0.6. The spread of the high-band error across all methods is
about 0.5 dB. No fusion can make one view's weave, knit or grain predict another's; the only levers are which
single view supplies the detail and where the cut goes. The consistency round helps where the image model
copies the coarse pattern: floor 0.14 to 0.39. The winner still shows detail discontinuities at label
boundaries on knit and linen, a seam at the hair crown, and seams at the chart borders of a many-island body
mesh.

**Caveat.** The second set was refined from a fused re-render of the first, so part of its agreement is
inherited and its absolute LOO is optimistic; rankings within a set are fair.

## Gate policy and coherent regions (fix rounds 2 and 3)

An independent review of the first delivery found wrong-object pixels on surfaces and a double-exposure look
where adjacent cameras met on one face. Two fix rounds changed the gates and the single-source regions, not the
fusion method (still C+S8). The repository's version: `fusion.gate_policy` (`"strict"` default, `"legacy"`;
table in `projection.md`), `mvb2r/handover.py` and `fusion.contacts` (experimental).

**Coherent regions with a hand-over band** (`detail_critical` objects under `strict`). One dominant camera per
region (`prefer_view` when it sees the region, else the largest total score); its soft samples count inside the
region; other cameras only on region texels at least `handover_mm` (default 5 mm, 3D distance) from the dominant
camera's coverage; the band between stays unobserved and is filled from the same object. In the low band the
hand-over is either feathered (`"cut": "feather"`, default, 16 texels at the 2048 reference) or cut hard
(`"cut": "hard"`: single source in both bands). Label cells are half size on these objects (8 instead of 16
texels: no rectangular patches on jaw and neck). Without `objects[].regions` every connected part is a region
with these defaults.

| region (first run) | setting | the builder's read | the independent reviewer, at 100% |
|---|---|---|---|
| hands, shoes | dominant camera, 5 mm, feather | ghost fingers and the pink finger on the trousers gone; parts of the back of the hands became smooth fill | trouser-coloured fingertips gone; fingers still rubbery and doubled, shoes read as slippers with white rims |
| face, front neck (round 2) | dominant camera, 5 mm, feather | no second jaw copy; her left cheek and under the jaw became smooth fill | introduced a pink band under the hair (below) |
| face, front neck (round 3) | dominant camera, 12 mm, `hard` | no second eye in any source or novel view; small lighter patches where fill meets observed skin | doubled eye and jaw still there in one three-quarter source view and one novel view; patched jaw and neck unchanged |
| hairline (round 3) | contact zone 20 mm, limited to above eye level / behind the head | the pink band under the hair gone; without the limit, dark bands down the cheek | gone in five reviewed views, a small wedge left in two; a drawn hairline in the close-up |

The two columns disagree on the face, and the reviewer's is the one to trust: the builder looked at fewer views.

**Leak** (`score.json` `leak`, `mvb2r/leak.py`): an observed texel is leaked when its exact source pixel lies in
pixels the generated image assigns to another object (masked objects: outside the own undilated mask or inside
another object's mask; others: inside any other object's mask or within 2 px of a nearer object's clay). It is
independent of the gate used for fusion, and blind to generated content displaced beyond those margins: the
wall and floor leaks of the first run scored 0 and were judged on contact sheets.

| object (the experiment's scripts) | its original gate (masks dilated 2 px) | its strict gate |
|---|---|---|
| skin | 231,691 / 5.44 M (4.3 %) | 0 |
| knit top | 22,328 / 0.95 M (2.3 %) | 0 |
| trousers | 7,272 / 1.22 M (0.6 %) | 0 |
| shoes | 3,294 / 0.81 M (0.4 %) | 0 |
| hair, every background object | 0 | 0 |

**Scores** (C+S8, mean of floor, back wall, sofa, knit top, skin; the leave-one-view-out evaluation mask uses the
legacy gate for every policy, so the sets are judged on the same pixels; strict observes fewer texels, so its
scored set is a subset):

| set | LOO PSNR px / view-mean | LOO LF / HF | sharpness | novel seam step / ratio | texture coarse seam |
|---|---|---|---|---|---|
| best (legacy) | 28.17 / 28.24 | 30.45 / 34.31 | 0.768 | 4.29 / 1.80 | 4.22 |
| best2 (strict) | **28.29 / 28.37** | 30.49 / 34.65 | 0.743 | **4.12** / 1.65 | 3.93 |
| best3 (strict + 12 mm hard face cut + hairline zone + sofa 4 px) | 28.27 / 28.36 | 30.46 / 34.64 | 0.742 | 4.14 / 1.64 | 3.77 |

What the skin paid: sharpness 0.94 -> 0.85 (best3 0.851), observed area 0.237 -> 0.197, texture coarse seam
7.04 -> 8.30 (best3 7.35); its LOO PSNR rose 23.12 -> 23.38 (best3 23.27). Back wall seam step 1.74 -> 1.08.
Unchanged by gating: a sweater wedge on the hip (clay geometry where the photo shows trousers) and a white spot on
the hair shell (geometry).
