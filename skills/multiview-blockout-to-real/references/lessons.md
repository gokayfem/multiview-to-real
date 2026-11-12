# Lessons

What went wrong or surprised in the first full run of this method, and the rule each finding became. The run
was one room (4.5 x 4 x 2.7 m) with a sofa, a side table, a window and one seated person, six cameras at 1664 x
1664, Seedream 5.5 edit on fal for the views, Blender 5.0.1 and Cycles. Every number here is from that run's
reports. None of the findings is specific to rooms or people.

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

## PBR

- **Material priors instead of texture look like plastic.** Rule: the base colour stays "projected photo ×
  smooth correction"; priors only bound the mean albedo and set roughness and specular.
- **Residual baked shading remains and must be reported.** Correlation of albedo with proxy shading 0.28 (raw
  0.59); after relighting, a ghost of the old sun patch remains on the floor. Rule: the residual numbers and
  the relight sheet are part of the report; the diagnostic `.blend` is labelled as containing lighting.

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
