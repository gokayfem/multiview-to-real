# Lessons

What went wrong or surprised in the first full run of this method, and the rule each finding became. The run
was one room (4.5 x 4 x 2.7 m) with a sofa, a side table, a window and one seated person, six cameras at 1664 x
1664, Seedream 5.5 edit on fal for the views, Blender 5.0.1 and Cycles. Every number here is from that run's
reports. None of the findings is specific to rooms or people. The findings marked *(fix rounds)* come from
the three rounds that followed an independent review of that run.

## Generation

- **Over-cap requests are silently downsized.** Asked for 4992 x 3328 (16.6 MP), the edit endpoint returned
  2504 x 1672 with no error, and the response's width and height were null. The highest size it returned
  natively was 2628 x 1752 (99.6% of the documented 4,624,220-pixel cap); requests within the cap came back
  exactly. Rules: `probe` records the cap and checks the returned size; every size is read from the file; a
  request never exceeds the verified cap; a resize is recorded as a resize and never called generated detail.
- **A panel can come back as a copy of another.** On the second input (four automatic views of a primitives
  scene), the image model returned the elevated view as a near-copy of its neighbour. Fused anyway, it smeared a
  second ball across the floor and put wall colour on object tops; consistency with that view included was 12.1
  ΔE in the low band and 0.04 in the high band, against 3 to 5 ΔE on the room scene. The gate now rejects it:
  its image matched the neighbour's clay better than its own (0.686 against 0.365) and the neighbour's image at
  0.992 edge correlation while the clays correlated at 0.505. Without it the three remaining views re-render
  cleanly, but still disagree in exposure (9.2 ΔE low band, 0.06 high band). Rule: every frozen view is gated
  and a failing one is marked `usable: false` and skipped downstream; a low band far above the usual range is a
  reason to look at the sheet before fusing.
- **A different call is a different interpretation.** A size probe at a slightly larger size gave a changed
  expression in the close-up and a differently coloured table. Rule: keep the accepted sheet unless a new one
  is better on what you can see; more pixels are not a reason to switch.
- **Acceptance must be fixed first.** Thresholds (IoU margin, contour distance, landmark offset, colour drift)
  were written before any refinement was inspected; one refinement failed by 0.006 IoU and a 1.5 px landmark
  excess and was retried, not waved through. Rule: `CONTRACT.md` and the acceptance file are filled before the
  first paid call and never lowered after a failure.
- **Masks need tight bands.** A 40 px GrabCut band let under-sofa shadow join the person (IoU 0.81, visibly
  wrong); a 15 px band fixed it. Rule: look at the mask overlays before trusting any IoU.
- **No global warp for local errors.** Two independent registration estimators never agreed within 2 px; the
  residuals were pose and silhouette differences. Rule: no registration unless both agree, small and global.
- **The price was a placeholder.** The endpoint documentation listed `$0`. Rule: costs are recorded as unknown,
  never as free; the call count and the budget are reported instead.

## Consistency

- **Independently generated views barely agree on fine detail.** High-band correlation between views was about
  0.16 (0.161 over 30 view pairs), against about 0.95 (0.947) for re-renders of one shared texture. Seams are
  therefore not only a fusion problem. Rule: `consistency` is measured and reported before projection, per
  object, and the README of a case never presents fused texture as one observed surface.
- **Render-then-refine fixes light more than detail.** One round improved low-band (lighting and colour)
  agreement by 35% (5.19 to 3.38 ΔE) and more than doubled floor-board agreement (0.207 to 0.489), but linen
  weave (0.145 to 0.173), knit (0.058 to 0.059), table grain (0.224 to 0.220) and hair stayed view-specific, and
  shoes got slightly worse (4.79 to 6.09 ΔE). Rule: `resync` is kept only if the score improves and alignment
  holds; one round by default; detail-critical regions stay single-source regardless.

## Geometry

- **Geometry decides realism.** A primitive mannequin and box sofa gave CG-looking views in every cell; real
  CC0 assets (MakeHuman through MPFB, Poly Haven furniture) gave photographic ones. Rule: start soft and human
  shapes from real assets; check the clay before any paid call.
- **The model follows the clay pose faithfully.** Posture and contact come back exactly as modelled. Rule:
  contact is measured (soles, seat, hands, penetration), not eyeballed.
- **An oversized shell renders as a helmet.** The hair volume was larger than the generated bob; every view
  showed a helmet outline with skin-coloured patches. Carving it to the generated hair masks raised hair IoU
  from 0.55 to 0.75 and from 0.63 to 0.84 in the two wide views (0.85 to 0.82 in one side view) and the close-up
  hair PSNR from 18.1 to 24.2 dB. Rule: `refit` objects are carved to several views' masks, and passes are
  regenerated after.
- **Downloaded assets had overlapping and stacked UV islands.** The sofa (85 polygons), the hair volume (39) and
  the shoes (left and right stacked) would have written two surfaces into one texel. Rule: overlap detection and
  repair is mandatory before projection, and the audit is reported.

## Projection

- **Depth must be camera-axis Z.** Blender 5.0's Cycles Z pass was verified to be camera-axis Z (median
  difference 1.5e-6 m); ray length differed by up to 0.91 m in the room. Rule: `views` proves the depth
  convention numerically per case; it is never assumed.
- **Generated objects are smaller than their proxies.** Generated limbs were slimmer than the proxy and the hair
  smaller than its shell, so pixels just outside the generated silhouette (background) projected onto proxy
  geometry. Rule: per-object generated masks are hard constraints; other objects reject the subject's dilated
  mask; alignment confidence is capped near the proxy boundary.
- **Source-view metrics barely separate fusion methods.** Observed-pixel PSNR on the six source views was 27.6,
  27.6 and 27.0 dB for best view, blend and labelling, while the crops showed pasted sun patches and blotchy
  linen. Rule: fusion is chosen by leave-one-view-out and seam visibility, confirmed on crops.
- **Margins must hold between every pair of objects** *(fix rounds)*. The first gating dilated each generated
  mask by 2 px and protected background objects only from the subject. Measured afterwards, 4.3% of the observed
  skin texels (231,691 of 5.44 M) and 2.3% of the knit top had taken a neighbouring object's pixels: trouser
  colour on every fingertip, sweater on the hip, shoe fragments on the floor, and the sofa's top rail and the
  table repeated on the wall. Eroding the object's own region and mask, rejecting every other object's mask and
  keeping a margin around every nearer silhouette took the measured leak to 0, raised held-out PSNR from 28.17
  to 28.29 dB and lowered the novel-view seam step from 4.29 to 4.12 (back wall 1.74 to 1.08). The skin paid for
  it: sharpness 0.94 to 0.85 and observed area 0.237 to 0.197, because more of it is now fill instead of
  wrong-object pixels. Rule: strict gating is the default and the leak count is reported per object. The count
  cannot see generated content displaced beyond the margins, so the crops are still looked at.
- **One camera per feature, with a gap before the next, is necessary and not sufficient** *(fix rounds)*. The
  face came from the close-up wherever the close-up was valid and from other cameras elsewhere; the pieces
  abutted, and the reviewer saw a doubled jawline and a second eye. One dominant camera per face, hand and
  shoe, other cameras allowed only 5 mm or more from its coverage (12 mm and a hard cut in both frequency bands
  for the face) and the gap filled removed the wrong-object smudges and scored slightly better. The builder then
  reported the doubled eye gone. The third review, at 100%, still found a doubled eye and jaw in one source view
  and one novel view of the baked scene, and in the relightable scene the cheek seam had become a mosaic of
  square skin blocks. A photograph of a face laid on a proxy head of a different shape cannot look right from
  another angle, whichever camera supplies it. Rule: a detail-critical region has one source and a gap before a
  second; expect it to hold only near the source camera unless the geometry matches the photographed shape.
- **A rejected pixel is not always the wrong colour** *(fix rounds)*. Under strict gating, skin texels at the
  hairline that only ever saw hair pixels were rejected and filled with skin colour. The hair shell is smaller
  than the photographed hair there, so every frontal view showed a pink band under the hair: a worse defect
  than the smudges the round had removed. The fix lets surface within 20 mm of the hair mesh take the
  photographed hair. Applied without a spatial limit it painted dark bands down the cheek, because the hair
  hangs within 20 mm of it; it had to be limited to above eye level and the back of the head. Rule: where one
  surface passes under another object, state which colour belongs there; contact zones are opt-in and bounded.
- **Fill must not draw attention** *(fix rounds)*. Once stricter gating enlarged the holes, the push-pull fill
  showed blocks, and never-seen walls took the median of all observed plaster, shadowed corners included, so
  they rendered dark with halos around the observed parts. Rule: the fill is a smooth pull-push; plain wall-like
  classes fade to the median of their well-lit observed texels. Filled texels stay flagged as inferred.

## PBR

- **Material priors instead of texture look like plastic.** Rule: the base colour stays "projected photo ×
  smooth correction"; priors only bound the mean albedo and set roughness and specular.
- **Residual baked shading remains and must be reported.** Correlation of albedo with proxy shading 0.28 (raw
  0.59); after relighting, a ghost of the old sun patch remains on the floor. Rule: the residual numbers and
  the relight sheet are part of the report; the diagnostic `.blend` is labelled as containing lighting.
- **De-lighting does not remove the image model's own direct light** *(fix rounds)*. The sun patch the image
  model painted on the floor survived a smooth, clamped de-lighting gain; with the sun moved, the old patch
  stayed 1.48 to 3.29 times brighter than the floor around it. Replacing the texels under it with texture
  synthesised from the shaded part of the same floor brought that to 1.02 (wide front view), 1.38 (elevated
  view) and 2.18 (a view where the rest is light falling off from the window), and the correlation of albedo
  with proxy shading from 0.30 to 0.16. Costs: 23.7% of the floor became inferred, and PSNR against the
  targets fell from 24.9 to 23.5 dB because the targets contain the patch. The reviewer passed the synthesised
  floor at full frame and failed it at 100%: soft, planks repeating in a regular checker, redder than the
  photographs. The first fill left darker blocks until its low-frequency level was matched to a smooth
  interpolation of the surrounding floor. Rule: the step is opt-in, its texels are inferred, the ghost is
  reported with and without it, and PSNR against the targets is not the measure of a relightable scene.
- **The subject keeps its photographed shading** *(fix rounds)*. After the sun moved, the person's brightness
  changed by a factor of 0.83 to 1.00 per view: the moved sun does not reach her and her fine shading is in the
  texture, where a smooth correction cannot remove it without destroying detail. Rule: say which objects respond
  to new light and which do not.
- **Real light shows the proxy** *(fix rounds)*. The baked version hides the proxy under the photograph; the
  relightable one does not. The proxy face's mouth and jaw and the proxy window's bar layout (different from the
  generated sash) became visible. Subsurface scattering on skin removed the plastic look (red to green ratio
  2.08 to 2.21, target 2.25) but not the shape. Rule: defects that appear only in the relightable version are
  usually geometry, not texture.

## Coverage

- **Coverage is the hard limit.** Area-weighted, the cameras saw 41% of the person's surface (24% of the skin,
  52% of the trousers), 23 to 31% of the floor and sofa in the PBR build, 1% of the ceiling and 0% of the wall
  behind the cameras. A 360° tour was 57% measured and 43% inferred (34% measured on the pan from the room
  centre). Rule: inferred surfaces are plain same-object fills, flagged per texel, counted per object, and
  tinted in a second copy of every tour; no 360° or whole-room claim is made beyond coverage.

## Process

- **Freeze before comparing.** Every method comparison used the same frozen targets, cameras, geometry and
  masks; a change to any of them means re-running everything after it. Rule: a fix is a re-run, never a patched
  output.
- **One number never stands for "real".** PSNR, the consistency bands, leave-one-view-out, seam ratios,
  coverage and a fresh reviewer's list answer different questions; report them side by side.
- **The builder's own look is too generous** *(fix rounds)*. Both scenes were judged acceptable from 384 px
  sheets. A fresh reviewer at 100% on fixed crop boxes found the head and hands failing in the source cameras
  themselves: hair that was a different object from the photographed hair, a face split into mismatched pieces,
  doubled fingers, trouser colour on the fingertips. Rule: acceptance is judged at native resolution on fixed
  crop boxes by someone who did not build the scene, and the same boxes are used in every later round.
- **A fix round can regress, and the numbers may not show it** *(fix rounds)*. The round that removed the leaks
  improved every mean score and introduced the pink hairline band, found only by the second review pass. The
  round that removed the band scored level with the one before it (28.29 against 28.27 dB, seam step 4.12
  against 4.14). Rule: a round is kept only if it is no worse on the numbers and better on the reviewer's own
  crops; the review is repeated after every round.
- **Name the cause before choosing the fix** *(fix rounds)*. Three rounds of gating and fusion changes left the
  hair shape, the hands, a sweater wedge on the hip and the mouth unchanged, because they are geometry: a
  carved shell is still a shell (a finer re-carve was tried and not adopted), and the proxy's fingers and hip
  are not where the photographed ones are. The reviewer's summary after three passes: the main problems are
  geometry, and every face fix traded one visible artefact for another of similar weight. Rule: sort each
  defect into gating, fusion, geometry, coverage or generation first; texture work does not repair geometry.
- **Join tour segments through black** *(fix rounds)*. A hard cut between the pan and the arc read as a glitch
  and a 0.5 s cross-fade showed two transparent copies of the subject. Rule: 0.25 s out, 0.25 s in.
- **A port is not validated by the original's results** *(0.2.0)*. The fix rounds were ported into this
  repository by three agents with more than 100 new unit tests, all green. The first run on a real case then found that
  the floor step skipped the floor (its source rule would have refused the experiment's own floor), selected
  furniture by class, looked for a render channel under a name Blender 5 does not use, and painted stacked
  strips where the experiment had painted planks; and that a sky image missing from an old case turned every
  relightable render magenta. Rule: every ported stage runs once end to end on a real case before release, and
  someone looks at its output.
- **Pixel margins belong to a resolution** *(0.2.0)*. Margins tuned at 1664 px and applied unchanged at 320 px
  removed a fifth of a thin object's observed texels and cost it 1.7 dB against the truth. Rule: margins scale
  with the camera size in both directions, with a floor of one pixel.
- **Know which baseline a number belongs to** *(0.2.0)*. The 4.3% leak was measured on the experiment's first
  gate, which widened masks. This repository's first release already shrank them and leaks 0.01% on the same
  scene, so quoting 4.3% for it would have credited the new policy with a drop that never happened here. Rule:
  re-measure the baseline in the code that ships before stating an improvement.

