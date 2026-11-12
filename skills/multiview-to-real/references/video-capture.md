# Video capture (explored, not a stage)

An alternative to the multi-view sheet, tried on the first run's room on 2026-11-12: render the grey scene as a
"driving video" along a known camera path, have a reference-to-video model re-render it photoreal, and project
its frames instead of six separately generated views. **Nothing on this page is implemented in `mvb2r.py`.**
The scripts live with that experiment and are tied to its scene. This page records what was measured, so the
next attempt starts from the findings and not from the guesses.

All numbers are from one scene (a 4.5 x 4 x 2.7 m room, a sofa, one seated person), one camera path in three
segments, 319 frames at 1080 x 1080 of which every third (109) was used.

## What was run

- **Driving video:** the clay scene rendered along the path with Blender's Workbench engine, about 0.3 s per
  frame (Cycles took about 6 s). Every clay frame's camera is exact.
- **Model:** `minimax/h3-max/reference-to-video` on fal, 1080P, the driving video as the reference video,
  stills of the person and the room as reference images, prompt expansion disabled. The prompt says that the
  video is the camera choreography frame by frame, that the images supply appearance only and that nothing in
  the room moves. Read the endpoint's schema before a call (`references/generation.md`); it changes.
- **Segments:** a full 12 s shot drifted from the clay's timing by 4 to 67 frames depending on the seed. Short
  segments (about 4 s) whose first and last frames are pinned to existing stills held it to 1 to 3 frames;
  without pins, 2 to 8.
- **Not usable:** `minimax/h3-max/keyframes` with the six stills as keys produced hard cuts between keys.
- **Cost:** about $47 at the documented prices for 245 calls on this route, drafts and map estimates included.

## What the checks showed

**Frames of one video agree with each other; separately generated stills do not.** COLMAP on the 109 frames
alone registered all of them in one model (8,492 points, 1.09 px). Counted per 1,000 keypoints, the matches
that survive its geometric check three frames later were 377 on the person, 338 on the room and 247 on the
furniture, and person and room stay level up to 60 frames apart. Skin was the most stable surface in the scene
(475); hair the weakest at 15 frames (93 against 140 for the sofa). The six stills of the first run agreed at 0.16 to
0.24 in fine detail, which is why no structure-from-motion and no fusion could reconcile them.

**A zooming clip needs each frame's own intrinsics.** The path interpolates focal length between the key
cameras (1050 to 1825 px here). Solved with one focal length for all frames, COLMAP put three of four exactly
known cameras 39 to 44 cm off and read 5 to 10 degrees of extra turn in two of the three segments. With each
frame's focal length it was within 1 to 4 cm and 0.5 to 3 degrees. The wrong run looked like a finding about the video and
was reported as one before it was caught.

**A camera set fitted to matches cannot be judged on those matches.** Cameras re-solved from COLMAP's matches
with the known cameras held fixed made the same floor detail land 4.2 mm apart across frames, against 5.1 mm
for the cameras fitted along the path. They had been adjusted on exactly those matches. The independent test,
the floor averaged over the whole clip, was level (4.13 against 4.06), and leaving one known camera out put it
26 to 104 mm from the truth (38 to 86 mm with flat surfaces held as well), more than the 17 mm between the two
camera sets. Nothing showed the new set to be
better, and the old one stayed.

**The video follows the clay's outlines, so the proxy already fits.** Over 109 frames each part's outer outline
differed from the video's mask of that part by 0.3 to 3.8 mm on average (the masks are seeded from the clay,
so they lean towards it; overlays agree by eye, strand tips aside). COLMAP's points lay 4.5 mm (hair), 7.5 mm
(sweater) and 6.8 mm (shoes) from the proxy surface, measured perpendicular to it. An earlier reading of 2.4 cm
for the hair and up to 9 cm for the sweater was the distance along the viewing ray, which grows without bound
at grazing angles. A reshape of the trousers driven by dense stereo made the outline overlap worse in eight of
eight frames (0.86 to 0.65) and was dropped.

**Depth from the video is available, within limits.** Rectifying two frames a few steps apart with their known
cameras and matching them with OpenCV's SGBM gave dense depth that agreed with COLMAP's points to 4 to 11 mm on
close frames and 1.5 to 5 cm on far ones. It needs sideways camera motion (42 of 109 frames qualified) and fails
on dark cloth without texture. A monocular estimate (MoGe-3) gave normals that agreed with the scene at 0.96,
but after an affine fit its depth between the person and the room was still off by 3 to 9 cm on most frames
and up to 13 cm.

## Projecting the frames

The first projection sampled every frame by the clay's object ids alone and switched source frames per texel.
It showed dark smudges on the fingertips, patches of neighbouring colour along the hair, dotted rims around
shoes and furniture legs, and blocky patches on the floor. The second pass changed three things:

1. A sample is kept only where the frame's own mask shows the same object as the clay, and never within 5 px
   (at 1080) of an edge of either.
2. The low band is a Gaussian-weighted mean around a per-texel centre frame, with no hard window, and its sums
   are feathered inside one UV island only. Feathering across islands put blotches on the face, whose UV map
   is many small islands.
3. Texels seen only near edges take the low band and no detail; isolated specks are dropped.

| | first look | second pass |
|---|---|---|
| Error near object boundaries, skin (mean absolute to the photograph, x255) | 20.3 | 18.2 |
| Same, hair / sweater / floor / sofa | 24.5 / 15.2 / 12.7 / 14.9 | 24.2 / 15.1 / 11.9 / 14.4 |
| Sofa low band, share of abrupt steps | 0.257 | 0.179 |
| Sharpness against a single-best-frame control | 0.86 | 0.77 |
| The same, recomputed at a 1.5 px scale | 0.955 | 0.966 |
| Observed area, skin / hair / floor | 0.241 / 0.316 / 0.244 | 0.227 / 0.306 / 0.243 |

The blocky floor and the fingertip smudges are gone. Still there: a light line at the collar, a light patch at
the hair edge, a pale blob at a sofa-leg tip, flat patches on fingers that no frame saw, and the hair is still
a shell. Both video textures hold about half the fine detail of the 1664 px photographs. The two sharpness rows
disagree because a control made of one nearest sample per texel is speckled, and a pixel-scale measure counts
the speckle as detail.

## Rules

- Pin both ends of short segments to stills; do not trust a long shot's timing.
- Give every frame its own camera matrix, and check a free solve against the cameras that are known.
- Judge a fitted quantity on data it was not fitted to. Say which check is independent.
- Measure misfit perpendicular to the surface and against outlines, never along the viewing ray.
- Before acting on a cause, confirm it with a second, different measurement. Three diagnoses in this
  exploration were overstated by a single measure: cameras 40 cm off, new cameras better, shape off by
  centimetres.
- Gate by what the frame shows, not only by the clay. Keep away from edges of both.
- Blend the low band without hard switches, inside one UV island.

## What a stage would need

A path with pinned key cameras and per-frame focal lengths; segment planning and the paid calls with their
logging; a per-frame camera fit against the clay with a check on the known frames; per-frame masks; a sample
store; the window fusion above; the comparisons on this page as its acceptance checks. None of it exists here.
