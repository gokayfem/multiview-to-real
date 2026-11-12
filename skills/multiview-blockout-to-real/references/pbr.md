# PBR

The generated views contain one lighting setup baked into every pixel. The diagnostic `.blend` keeps it and says
so. The PBR `.blend` tries to remove it conservatively, so the scene can be relit, and reports how much is left.
Separating material from light in generated single-illumination views is approximate; do not claim more.

## The rule: keep the projected photo

The base colour is **the projected photo times a smooth correction**. Never replace projected texture with
material priors: an earlier version that did so looked like plastic, because the generated weave, grain, skin
and creases are what makes the result read as real. The priors in `mvb2r/material_priors.json` only bound each
object's mean albedo and set roughness, specular and bump strength.

## Stages of `pbr CASE [--method M] [--scale S] [--samples N] [--fit-iters K] [--synth-direct]`

1. **Shading basis.** For every view, the clay proxy is rendered under the scene's own lights, sun and sky
   separately (`pbr/shading/`), plus the sun's direct light alone (`<view>_sundirect.exr`, Cycles Diffuse
   Direct: no bounce), used only by the direct-light synthesis below.
2. **De-light (`pbr.delight: "conservative"`).** Per view and object, the relative proxy shading (low-passed,
   divided by its median over the object) gives one smooth scalar gain per pixel, `clip(1 / shading, 0.7,
   1.6)`; colour and detail are untouched (`pbr/delit/`). A view whose luminance does not follow the proxy
   shading (correlation below 0.3) is left unchanged and reported: de-lighting a target that lacks that shading
   would paint inverse shading into it.
3. **Project.** The de-lit views go through the same fusion, masks and regions as the diagnostic
   (`pbr/albedo/<method>/`).
4. **Materials.** One Principled BSDF per object: base colour = projected texture × one scalar per object that
   bounds its mean albedo to the class range from `material_priors.json`; roughness, specular and bump strength
   from the class prior; bump from the high-passed albedo luminance (`pbr/textures/<object>/`). Unobserved texels
   keep their plain same-object fill.
   With `pbr.synth_direct`, the direct-light patch synthesis (next section) runs here, on these textures.
5. **Rig fit.** The scene's own lights are kept and one global light scale is solved (exposure stays 0) by
   rendering `final.blend` at low resolution and comparing the median and 90th-percentile luminance ratio to the
   targets on well-observed object pixels; the fit is accepted when both are within 10%
   (`pbr/rig_fit.json`). There is no sun-direction search and no white-balance fit; a failed fit is reported,
   not hidden.
6. **Build.** `out/final.blend` with packed textures, the source and novel cameras and the rig, plus a README
   text block with measured versus inferred per object and the residual shading numbers (`pbr/eval.json`).

The first run went further by hand: it flattened the large-scale brightness of plain materials toward the
object median (the generated sun patch did not sit where the rig's sun put it), rejected near-clipped pixels,
and fitted the sun and sky separately with an exposure of 1.6 EV and a white-balance tint. The numbers below
come from that run.

## Direct-light patch synthesis (opt-in)

The generated photos carry the image model's own direct light, for example a sun patch from a window on a floor
and a wall. The de-light gain is smooth and clamped, so it cannot remove a hard-edged patch: after the light is
moved, a ghost of the old patch stays in the base colour. In the first run the independent reviewer ranked that
ghost among the top defects of the relightable version.

`pbr.synth_direct: true` in `case.json` (or `pbr --synth-direct`; `--no-synth-direct` overrides the case) replaces
the texels under the patch with texture synthesised from the shaded part of the same object
(`mvb2r/sunpatch.py`, `mvb2r/sunpatch_stage.py`). Default off.

- **Objects: named only.** List them in `pbr.synth_objects`, after looking at the relight sheet
  (`out/relight.jpg`) for the surfaces that carry the generated photo's own direct light. `pbr.synth_classes`
  (default `[]`) is an explicit opt-in by class (detail-critical objects skipped). Turning the step on with
  neither set is an error. Why: in the first real run of the port with a class default, a side table (wood) got a
  mask of 64% of the object with 255 components skipped, the window frame (paint) 54%, the skirting was touched,
  and a back wall with only soft window light got one 70,000-texel fill, while the floor, the one surface the step
  exists for, was skipped. The experiment itself went floor-only after its first attempt on walls looked wrong.
- **Mask** (texture space, per object) = the union of (A) the photo's own bright patch: raw projected (not de-lit)
  luminance, low-passed, above `k_bright` × the object's observed median; and (B) the texels the proxy's key
  light reaches directly: each view's `<view>_sundirect.exr`, thresholded to a lit image and projected with the
  textures' own visibility. It re-uses the projection inputs, so its gates follow `fusion.gate_policy`
  (strict by default), like the textures. Dilated by `dilate_m` metres (penumbra, mullion shadows), converted to texels from the
  object's texel size.
- **Fill**, per connected mask component: block quilting (`mvb2r/sunpatch_quilt.py`) from the shaded, observed
  texels of the same UV chart (and the same facing). "Shaded" is judged against the local level of the observed
  surface (two blocks wide), not one object median, so a smoothly lit floor still counts as source while contact
  shadows and bright residue do not. Every block, with its overlap, is copied whole from a window inside the
  source eroded by the overlap, so no copied texel is a source edge, plain inferred fill or a chart border.
  Per block, the candidates closest to the target level are ranked by how little they shift across the
  texture's dominant lines (structure tensor), and one that matches what is already known in the block's window
  (blocks placed before it and the surface around) is drawn, seeded; neighbouring blocks may not reuse an offset
  or a source window. Along a clear structure, blocks are three times longer along the lines (planks are not cut
  short), used only where they continue the known texels well. Texels left over are retried with square, half
  and quarter blocks. In the first real run, the earlier per-texel fallback copied the strip just past the
  source's edge into every row: stacked strips with the source's scalloped edge and dark lines between them.
  After quilting, a local level match (ratio form, per channel): the fill's low-passed level (sigma =
  half a block) is scaled to a pull-push interpolation of the surrounding shaded sources' level, so neither single
  blocks nor the whole patch differ in level and the fill keeps only its own detail above it. One mean gain from
  the shaded ring is only the fallback when no shaded source surrounds the component (in the first run's full-
  resolution build, a single gain left the fill as visibly darker blocks). Feathered over `feather` texels
  (capped at the dilation), only against shaded texels of the same chart: where the component meets the chart
  boundary (a 3D seam) the fill stays at full strength, so no line of the old patch survives there.
  Deterministic (fixed seed).
- **Source size:** a component needs at least 4 blocks' area of shaded observed texels on its chart
  (`MIN_SOURCE_BLOCKS`) and at least 0.1 × its own area (`MIN_SOURCE_RATIO`); otherwise it is left unchanged and
  listed under `skipped`, with no texture invented. A smaller source is tiled over a larger patch, as the
  experiment did (its floor mask was 23.7% of the object; the port's earlier 0.5 ratio skipped a real floor at
  0.32): each filled component records `source_ratio`, and below 1 `repetition_expected: true` (and a printed
  line), because the fill then repeats its source visibly.
- **Specks:** mask components smaller than a quarter block (`MIN_COMPONENT_BLOCKS`) are dropped before filling
  and only counted (`components_dropped_small`).
- **Parameters** (`pbr.synth_params`): `k_bright` 1.6, `dilate_m` 0.2, `feather` 24 (first-run values).

Synthesised texels are **inferred**, not measured: `pbr/textures/<object>/synth_mask.png`, and in
`pbr/eval.json` → `synth_direct.objects.<object>` the mask area, components filled and skipped, the ring colour
difference before and after, the level residual (RMS of the fill's low-passed luminance minus the interpolated
shaded level) before and after the level match, and `inferred_before` / `inferred_after`; the README in `final.blend` says so too.
`synth_direct.ran` says whether anything was synthesised. `pbr/textures/` holds what `final.blend` uses;
`pbr/textures_nosynth/` keeps the same textures without synthesis so a case can compare both (the relight
test's old-patch residue, measured on `pbr/textures/`, can be read for a run with and a run without).

Evidence (first run, full resolution, sun moved): brightness inside the old patch against the floor outside went
from 1.48 to 1.02 in the wide front view, 2.11 to 1.38 in the elevated view and 3.29 to 2.18 in a view where most
of the rest is light falling off from the window (1.0 = no ghost); albedo correlation with the proxy shading fell
from 0.30 to 0.16; 23.7% of the floor became inferred; PSNR against the targets fell from 24.9 to 23.5 dB, because
the targets contain the patch. (An earlier reduced-resolution proof: 1.53 to 1.07 and 2.12 to 1.43.)

Those numbers are from the experiment's own script. This code, run on the same room at 512 px with the floor
named in `pbr.synth_objects`: base colour inside the old patch against the same floor outside it (the relight
test's per-object residue) went from 4.4 to 1.04 in the frontal view, 6.2 to 1.31 and 10.0 to 1.29 in the two
three-quarter views and 5.3 to 1.00 in the elevated view; the shaded source was 0.47 of the patch
(`repetition_expected`); the inferred share of the floor went from 0.76 to 0.89. To the eye the fill is a plank
floor at the surrounding brightness, with short staggered planks, softer than the photograph, and a pale band
of original texture left along one edge of the mask.

Turn it on when the generated views show a hard direct-light patch (sun through a window, a spot) on walls,
floors or other plain, repetitive surfaces and the relight test shows its ghost. Leave it off for surfaces whose
detail is unique (a painting, a printed rug, a face) and when the patch covers most of a chart.

Known limits: an independent reviewer passed the synthesised floor at full frame and failed it at 100%: soft,
with no grain or knots; short planks repeating in a regular checker (clearest after relighting); redder than the
photographs. Hence opt-in. The proxy's key light is the scene's SUN; there is no sun-direction search, so (B) is wherever the
rig's sun lands, and (A) has to catch a generated patch that lies elsewhere. Plain-material flattening (pulling
the large-scale brightness of plain materials toward the object median, which the first run did by hand) is not
implemented. Synthesis copies texture; it cannot recover what lies under the patch.

## What to report

- **Residual baked shading:** the correlation of albedo luminance with the proxy shading, per view and mean,
  next to the same number for the raw targets. First run: 0.28 mean (raw 0.59); worst view 0.62 (raw 0.82).
- **Sun patch in the albedo:** albedo inside the target's floor patch over outside. First run: 1.4 to 2.0 (raw
  5.5 to 8.8): strongly reduced, not removed.
- **Luminance ratio render / target per view:** first run 0.98 to 1.04 in five views, 0.87 in the face close-up
  (one global rig under-predicts bounce light there; no per-view correction is applied).
- **Which views were left un-de-lit** and whether the rig fit was accepted (`pbr/eval.json`).
- **PSNR to the targets:** PBR 23.8 dB mean against 26.2 dB for the diagnostic. The diagnostic is the target
  colour by construction; the PBR render is lower because its light is physical.

## Relight honestly

`render CASE --relight` renders the PBR scene and the diagnostic scene with the key light moved (+70° azimuth,
25° elevation, warm) and writes `out/relight.jpg` and `out/relight.json`:

- the PBR scene must change and the diagnostic must not (first run: diagnostic mean change about 1e-6);
- the old sun patch's contrast before and after (first run: 7.9 → 1.2 in the wide view, where the new patch
  landed on the wall; 6.4 → 4.1 in another view, where residual baked brightness remained);
- a ghost of the old patch is normal without `pbr.synth_direct`: pale, stepped areas remain where the generated
  patch was flattened. Say so in the report and show it; with synthesis on, show the residue with and without.

## Not done

The view through an opening is the scene's sky light, not the generated exterior. Hair is a textured shell, not
a hair shader with strands. Specular highlights in the generated views stay in the base colour where the de-
light could not separate them. Metal and glass are out of scope.
