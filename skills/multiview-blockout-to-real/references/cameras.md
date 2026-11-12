# Cameras

Coverage is the hard limit of the method. A surface no camera sees is inferred, and nothing later can change
that. Place cameras for the surfaces the result must show, then measure.

## Placement

`new` keeps the input's cameras when it has at least two and `--views` is not given; otherwise it places an
automatic rig (6 views unless `--views N`; `--no-cameras` skips this). `cameras CASE --replace [--count N]
[--closeup OBJ] [--arc DEG] [--no-elevated]` re-places them later. Rooms are detected and automatic cameras are
kept inside them. The automatic set is:

- **an arc of views around the subject** (`views.auto.arc_degrees`, centred on the side the scene faces), at
  eye height, with a natural focal length (roughly 35 to 60 mm full-frame equivalent);
- **an optional close-up** on a named object (`--closeup OBJ`, `views.auto.closeup`), for faces, products and
  anything `detail_critical`: generated detail per surface area is highest there, and fusion prefers it;
- **an optional elevated wide** (`views.auto.elevated`) that shows the layout and the floor.

Cameras from the input are kept and marked `"source": "existing"` in `case.json`. Every camera renders at the
same square or rectangular resolution (`views.resolution`), pixel aspect 1, no lens shift unless the input had
it, no depth of field.

The first run used six cameras at 1664 x 1664: frontal wide (35 mm), front-left and front-right three-quarter
(40 mm), a face and upper-torso close-up (60 mm), a side view of the seated body (40 mm) and an elevated three-
quarter wide (35 mm).

## What to check on the preview

- Every camera is inside the room (or outside the object) with no wall, ceiling or furniture between it and the
  subject; a near clip plane that cuts geometry is a failure.
- No two views are near-duplicates: the image model treats them as separate panels anyway, and they add little
  coverage.
- The subject is not tiny in the wide views; in the first run's elevated wide the person was small, and that
  view had the lowest silhouette IoU (0.837).

## Overlap and coverage

`cameras` writes a per-object coverage table: the fraction of each object's surface seen by at least one view
and by at least two. Read it before the first paid call.

- **At least two views on everything that matters.** One view gives a texture; two views give a check
  (`consistency`) and a choice (fusion). A surface seen once cannot be scored by leave-one-view-out.
- **One close view on every `detail_critical` object.** It becomes the single source for that region.
- **Expect large unseen areas.** In the first run, area-weighted over each mesh, the cameras saw 24% of the
  floor, 41% of the sofa, 28% of the back wall, 1% of the ceiling, 0% of the wall behind the cameras and 41% of
  the person; backs and undersides were not seen. A 360° pan from the room centre was 34% measured.
- **Grazing views add little.** Texels seen at grazing angles are rejected (`projection.md`), so a camera that
  sees a wall edge-on does not cover it.

## How many views

Two to nine. More views give more coverage and more chances for disagreement: every view is a separate
interpretation by the image model, and in one multi-view sheet the cell size falls as views are added (the
sheet's pixel cap is fixed; `generation.md`). Six views at 876 x 876 cells were refined individually to 2144 x
2144 in the first run. Choose the smallest set that covers what the result must show.

## Changing cameras later

Changing a camera changes its passes, its cell, its target and everything fused from it. After any change,
re-run `views` and every stage after it; old targets are not reused for a moved camera.
