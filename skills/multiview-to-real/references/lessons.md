# Lessons

What went wrong or surprised in the first full run of this method, and the rule each finding became. The run
was one room (4.5 x 4 x 2.7 m) with a sofa, a side table, a window and one seated person, six cameras at 1664 x
1664, Seedream 5.5 edit on fal for the views, Blender 5.0.1 and Cycles. Every number here is from that run's
reports. None of the findings is specific to rooms or people. The findings marked *(fix rounds)* come from
the three rounds that followed an independent review of that run.

The findings marked *(third input)* come from a later run on a rigid scene without people: two plaster walls, a
limestone floor, an oak cabinet of five parts and a terracotta block, six cameras on a 90° arc, the same
endpoint, Blender 5.2.2. It was made with a separate set of experiment scripts, not with this repository's CLI,
so its numbers describe the method and the image model, not this code. Where a rule below needs a stage this
repository does not have, the entry says so.

## Generation

- **Over-cap requests are silently downsized.** Asked for 4992 x 3328 (16.6 MP), the edit endpoint returned
  2504 x 1672 with no error, and the response's width and height were null. The highest size it returned
  natively was 2628 x 1752 (99.6% of the documented 4,624,220-pixel cap); requests within the cap came back
  exactly. Rules: `probe` records the cap and checks the returned size; every size is read from the file; a
  request never exceeds the verified cap; a resize is recorded as a resize and never called generated detail.
  Seen again on the third input: the same request returned the same 2504 x 1672, and 2592 x 1728 and 2144 x
  2144 came back exactly.
- **An exact-size sheet can still have seams off the lattice** *(third input)*. The accepted 2592 x 1728 sheet
  had its column seams 1 and 2 px and its row seam 8 px from the 864 px lattice. The first run's accepted sheet
  was within 2 px. Rule: cells are cut at the measured seams of each sheet, never at the nominal ones, and the
  offsets are reported.
- **The reference image is copied, with what you did not ask for** *(third input)*. The appearance reference
  showed a darker band across the cabinet doors and faint joints in the floor, neither of them in the scene
  design. One edit of the reference asking to remove both changed nothing (mean difference 1.8 of 255). The
  sheet dropped the floor joints and kept the band in all six views; it reached every fused texture and was
  taken out only in the relightable base colour, where a faint stripe still marks it. Rule: before the sheet
  call, look at each reference for features the geometry does not have. Replace the reference or accept them
  in writing; do not count on the sheet prompt or a later stage to remove them.
- **Say which way a directional texture runs, part by part** *(third input)*. The first sheet drew the door
  grain horizontally in two panels and vertically in four. One added paragraph that named the grain direction
  of each part (along the length on the top, horizontal on the doors, vertical on the side panels) made all six
  panels agree, and three reviews later found it consistent. Rule: the description of a wood, brushed, woven
  or planked object states its direction relative to the part.
- **Refinement buys resolution and can cost alignment** *(third input)*. Every refinement was further from the
  calibrated geometry than the cell it was made from: global shift 2.6 to 9.9 px for the cells (in 2144 px
  units) against 5.7 to 11.3 px, and the block's IoU with its clay fell in five of six views. Refining each view
  on its own did not bring the views closer in colour either (4.4 to 8.2 ΔE2000 between the cells, 4.8 to 9.2
  between the refinements, measured with that run's tool). Its drift check rejected all six. The builder
  accepted them anyway, four of them against the check even after registration, which is what "never lowered
  after a failure" exists to prevent; whether cells or refinements give the better texture was not measured.
  Rule: compare a refinement's alignment with its cell's in the same units, and when the rule says cell, use
  the cell.
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
- **Rigid objects move as a whole, each by its own amount** *(third input)*. With boxes instead of a person the
  residuals were not pose. The cabinet was drawn up to about 2% narrower, the block drifted on its own, and in
  one view the line between floor and wall sat about 25 px low at 2144 px. A global warp would again have moved
  correct pixels. One bounded affine map per rigid group, refined per visible flat face and estimated from
  edges against the calibrated passes, took the median edge residual from 2.6 to 22.0 px down to 0.9 to 2.0 px
  (displacements up to 31 px, up to 30 mm on the surface). A separate tool then measured a global shift of 0.3
  to 1.5 px (5.7 to 11.3 before) and a block IoU of 0.94 to 0.99 (0.78 to 0.95 before). The map was used only
  when sampling colour; cameras and geometry were never re-estimated. One view's room had no usable edge and
  kept the identity map at lowered confidence. Rule: the rule above stands for global warps and for soft
  shapes. For a rigid object with straight edges, measure its displacement against the calibrated passes
  before projecting, and report it per object. This repository has no stage that corrects it (`register`
  moves views towards the consensus, which is not the same), so on such a scene look at every object's edges
  over its clay first.
- **An edge-matching estimator fails quietly** *(third input)*. The first version of that estimator had three
  faults that its first run on real data did not reveal: a line locked onto a parallel neighbour; long edges
  outvoted short ones, so the cabinet's far end slid 5 to 12 px along its long axis while the edge residual
  read about 0.5 px; and room creases were matched to cabinet lines. A synthetic case with known, different
  displacements per object found them, and the first real-data result was thrown away. The corrected version
  recovers the known maps to within 1.7 px and flags a block moved 110 px instead of registering it. Rule: a
  registration estimator is validated on known displacements, with the error measured at points, before it
  touches generated views. An edge residual cannot see sliding along the edge.
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
- **Brightness between views can differ by a factor** *(third input)*. The same door surface differed between
  two views by a luminance factor of 0.3 to 3.4; the walls and the floor by 0.7 to 1.5. That run's gain solver
  wanted per-view gains of 0.47 to 1.72 on the cabinet and 0.81 to 1.24 on the room, and its ±25% bound left
  the seam step of its seam-aware method at 6.6 levels against 5.8 with a bound of 0.4 to 2.5. The image model
  shades each view as its own photograph, not as one light fixed in the world. Rule: read the gains a
  levelling step asks for before trusting its clamp, and report a clamp that binds. The levelling here works
  between regions, not per view, with a clamp of the same size (`level_max` 0.22 in log units, about ±25%) and
  no levelling at all for pairs beyond `level_reject` (0.35, about a factor of 1.4); how it behaves on such a
  scene has not been measured.
- **Displacement hides part of the detail agreement** *(third input)*. High-pass detail correlation between
  views was about 0 as generated (-0.02 to 0.03 per view) and 0.05 to 0.13 once each view was sampled through
  its registration: 0.22 to 0.27 on the door fronts, 0.05 to 0.09 on the cabinet top, about 0.01 on the floor
  and the back wall. So on the doors part of the same grain was there and displaced; on the floor and the walls
  every view has its own texture. Those numbers come from that run's tool and are not the bands of
  `consistency`. Rule: on a rigid object, look at its displacement before reading a low high band as invented
  detail.

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
  Seen again on the third input: against the registered, exposure-compensated targets the three methods were
  8.9, 8.3 and 9.0 of 255 (SSIM 0.879, 0.917, 0.893), while the mean colour step across seams was 17.0, 3.0 and
  5.8 levels, and the blend, best on both, was visibly softer at 100%. That run made no leave-one-view-out
  score, so its choice rests on seams and crops alone.
- **The ceiling is the views, not the projector** *(third input)*. The same projection code re-rendered known
  textures to 1.7 of 255 and the generated views to 8 to 9, whichever method fused them. Registration barely
  moved that error away from face boundaries (12.7 to 12.5 before exposure compensation) and raised SSIM by
  about 0.06. Rule: run the projector on known textures first and report both numbers, so that the gap is read
  as disagreement between views and not chased with fusion settings.
- **An ownership mask must follow the occlusion order** *(third input)*. Masks that removed every other
  object's footprint from an object's region carved strips off it: a band across the block's top, a notch on a
  door edge, a smeared cabinet end, each refilled from another view. Removing only what lies in front, with a
  colour check where two objects contend for a pixel, removed the band and the notch; the cabinet end, which
  also had only oblique views, was not judged again by a reviewer. Rule: where two objects' regions overlap in
  the image, the nearer one owns the pixel, and an object is never cut back for one behind it. Check this
  whenever masks are widened or moved.
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
  Seen again on the third input, on walls and a floor: to remove a faint paler patch on a wall, the room
  surfaces were put on a fitted tone and a procedural prior alone. A fresh reviewer rated the result flat CG
  paint, severity 4, against 3 for the patch, and the change was reverted.
- **One material fitted per object becomes several materials** *(third input)*. Five parts of one oak cabinet,
  each with its own fitted tone, came out as five oaks, from (0.22, 0.12, 0.06) to (0.59, 0.36, 0.20) in linear
  RGB, because the generated views shade the cabinet's left darker and one part had seven usable texels. One
  tone per material class, weighted by usable texels, gave one oak. Rule: objects stated to be the same
  material share one tone; the untied values are kept and reported. The scalar that bounds mean albedo here is
  per object, so check that parts of one piece of furniture do not end up visibly different.
- **A light fit can switch a light off** *(third input)*. Fitted jointly over all objects, the area light kept
  its weight and the uniform fill went to zero (its colour came back undefined and had to be set by hand). The
  fit explained about 92% of the luminance variation, and the renders had soft, flat light with almost no
  darkening where the plinth meets the floor, which two reviewers listed. Rule: read the fitted weights; a
  light at zero or an undefined value is reported as the fit's result, and contrast is not then tuned by hand.
- **A low-pass across a source boundary leaves a halo** *(third input)*. The paler wall patch was the halo of a
  brightness step between two source views, spread by the large-scale split of the material recovery. Computing
  the split inside each source-view region reduced it to a faint outline. Rule: any smoothing of a fused
  texture stops at the boundaries between source views.
- **Photo divided by shading is unreliable where the shading is small or the edge is off** *(third input)*. A
  pale line appeared where the plinth meets the floor (deep contact occlusion), and a pale strip beside the
  door joint, because the generated joint line sat a few millimetres from the modelled one. Fading the observed
  detail near face borders, in deep occlusion and near hidden regions (8 to 25 mm and 1 to 4 cm in the last
  build) took both out of the relightable renders; the next two reviewers saw them only in the unlit
  textures. Rule: recovered detail is trusted least at edges, contacts and the rim of what was observed, and
  is faded there.
- **Set a threshold from the values it will meet** *(third input)*. A confidence gate of 0.2 to 0.45 on the
  detail layer wiped the door grain, because confidence on the doors was typically 0.2 to 0.3; 0.06 to 0.13
  kept it. Rule: look at the distribution of a gating quantity on the surfaces that matter before choosing its
  limits.
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
  Seen again on the third input: a modelled 4 mm door joint with 5 mm bevels catches the light and reads
  lighter than the thin dark line in the generated views.

## Coverage

- **Coverage is the hard limit.** Area-weighted, the cameras saw 41% of the person's surface (24% of the skin,
  52% of the trousers), 23 to 31% of the floor and sofa in the PBR build, 1% of the ceiling and 0% of the wall
  behind the cameras. A 360° tour was 57% measured and 43% inferred (34% measured on the pan from the room
  centre). Rule: inferred surfaces are plain same-object fills, flagged per texel, counted per object, and
  tinted in a second copy of every tour; no 360° or whole-room claim is made beyond coverage.
- **Share of the whole mesh and share of the visible face are different numbers** *(third input)*. On a 90°
  arc the cameras saw 100% of each door front and 48% of each door, 88% of the cabinet top's upper face and
  44% of the top, 28% of the floor's upper face and 13% of the floor, because backs, undersides and faces
  pressed against other objects count in the whole. Rule: report both, and say which one a sentence uses.

## Video capture

From the exploration of 2026-11-12 on the first run's room (`references/video-capture.md`); not a stage of
this toolkit.

- **Frames of one video agree; separately generated stills do not.** Structure from motion on 109 video frames
  alone registered every frame (1.09 px), and matches on the person survived its geometric check as often as
  matches on the room (377 against 338 per 1,000 keypoints three frames apart). The six stills had agreed at
  0.16 to 0.24 in fine detail. Rule: when views must agree in detail, take them from one generation.
- **A zooming clip needs each frame's own intrinsics.** With one focal length for a clip that zooms from 1050
  to 1825 px, a free solve put known cameras 39 to 44 cm off; with per-frame focal lengths, 1 to 4 cm. The
  wrong result read like a property of the video. Rule: a camera matrix belongs to a frame, and a free solve is
  checked against every camera that is known before anything is concluded from it.
- **Short pinned segments follow the path; long shots drift.** A 12 s shot was off the clay's timing by 4 to
  67 frames depending on the seed; 4 s segments pinned at both ends to stills, by 1 to 3.
- **A driven video keeps the clay's outlines.** Part outlines differed from the video's by 0.3 to 3.8 mm and
  sparse points lay 4.5 to 7.5 mm from the proxy. A reshape driven by a depth reading made the outlines worse
  (overlap 0.86 to 0.65). Rule: reshape only when outlines and depth both ask for it.
- **Along the viewing ray, a small misfit looks large.** The same hair that is 4.5 mm from the proxy,
  perpendicular to it, read as 2.4 cm along the ray; the sweater as up to 9 cm. Rule: measure misfit
  perpendicular to the surface and against outlines.
- **A quantity fitted to data looks good on that data.** Cameras re-solved from feature matches landed those
  matches tighter (4.2 against 5.1 mm) and were level on the independent test (4.13 against 4.06). Rule: name
  the independent check, and without one do not claim an improvement.
- **One measure is not a diagnosis.** Three causes were reported and then withdrawn: cameras 40 cm off (one
  focal length for a zooming clip), new cameras better (judged on their own fitting data), shape off by
  centimetres (measured along the ray). Each fell to a second, different measurement. Rule: before acting on a
  cause, confirm it another way.
- **Gate by what the frame shows.** Sampling by clay ids alone put trouser colour on fingertips and
  neighbouring colour along the hair. Requiring the frame's own mask to agree, and keeping 5 px from edges of
  both, removed the smudges (skin error near boundaries 20.3 to 18.2).
- **Blend inside one UV island.** Switching source frames per texel left blocky patches on the floor; a
  Gaussian-weighted low band removed them. Feathering that crossed UV islands put blotches on a face mapped as
  many small islands. This toolkit's `uv_blur` (weights, gains, the scoring low band) is not limited to one
  island either; see the known limits.
- **A speckled control inflates sharpness.** Against a texture made of one nearest sample per texel, the second
  pass scored 0.77 where the first scored 0.86; at a 1.5 px scale the two were level (0.966 against 0.955).
  Rule: state the scale of a sharpness figure and look at the control.
- **Monocular depth is right inside an object and wrong between objects.** After an affine fit, a monocular
  estimate left 3 to 13 cm between the person and the room; its normals agreed with the scene at 0.96. Dense
  stereo between two frames with known cameras agreed to 4 to 11 mm up close, where the camera moves sideways.

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
- **Rigid geometry removes the geometry defects and leaves the rest** *(third input)*. With boxes only, three
  fresh reviewers found the final renders matching the grey geometry, with no doubled edges and no colour of
  one object on another. Every defect they listed was generation or material: the reference's band on the
  doors, a patch where two source views met, flat light, weak contact shadow, colours drifting from the
  references. Rule: this supports sorting defects by cause. When a scene has no soft shapes and still looks
  wrong, look at the views' agreement and at the material fit, not at the gates.
- **A reviewer can only judge what was in the packet** *(third input)*. Each of the three reviewers saw the
  first 37 to 43 frames of a 192-frame camera move, its slow start with almost no parallax, because the move
  was still rendering when the packets were made. The rest was checked only by the builder's frame statistics.
  Rule: a packet is made from finished outputs, and the report says which images and frames each reviewer
  opened.
- **A change after the last review is unreviewed** *(third input)*. After the third review one setting was
  reverted and the scene delivered. That exact state was seen only by the builder. Rule: the last step before
  delivery is a review of the delivered state, or the report says that it was not reviewed.

