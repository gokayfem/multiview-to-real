# PBR

The generated views contain one lighting setup baked into every pixel. The diagnostic `.blend` keeps it and says
so. The PBR `.blend` tries to remove it conservatively, so the scene can be relit, and reports how much is left.
Separating material from light in generated single-illumination views is approximate; do not claim more.

## The rule: keep the projected photo

The base colour is **the projected photo times a smooth correction**. Never replace projected texture with
material priors: an earlier version that did so looked like plastic, because the generated weave, grain, skin
and creases are what makes the result read as real. The priors in `mvb2r/material_priors.json` only bound each
object's mean albedo and set roughness, specular and bump strength.

## Stages of `pbr CASE [--method M] [--scale S] [--samples N] [--fit-iters K]`

1. **Shading basis.** For every view, the clay proxy is rendered under the scene's own lights, sun and sky
   separately (`pbr/shading/`).
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
- a ghost of the old patch is normal: pale, stepped areas remain where the generated patch was flattened. Say so
  in the report and show it.

## Not done

The view through an opening is the scene's sky light, not the generated exterior. Hair is a textured shell, not
a hair shader with strands. Specular highlights in the generated views stay in the base colour where the de-
light could not separate them. Metal and glass are out of scope.
