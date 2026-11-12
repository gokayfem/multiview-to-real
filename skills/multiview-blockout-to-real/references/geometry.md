# Geometry

The image model follows the clay. Whatever shape, pose and contact the blockout has, every generated view
reproduces it, and the projection can only paint what the geometry offers. Geometry decides realism more than
any later stage.

## Primitives against real assets

The first run started with a smooth primitive mannequin and a box sofa. The generated views looked like CG
renders in every cell, and the hair came back as a stiff, wig-like shell that followed the proxy. The scene was
rebuilt from CC0 assets: a MakeHuman body, clothes, shoes and hair envelope posed with MPFB, and Poly Haven's
sofa and side table. The same kind of prompt then gave photographic views, with the camel-back sofa, oak
frame, linen and the person's clothes all following the new shapes.

Rules:

- For people, upholstered furniture and anything soft, start from real assets with clear licences, and record
  each one (source, licence, file hash) in the case. Avoid packs whose per-asset licence disagrees with the pack
  page; the first run excluded such poses.
- Split a character into parts that differ in material (skin, hair, top, trousers, shoes): each gets its own
  generated mask and atlas.
- Keep the scene's own lights: the PBR stage fits their strengths, it does not invent new sources.

## Pose and contact

The image model keeps the clay pose faithfully, including mistakes: a lowered head, a forward lean or a foot
off the floor in the clay come back in every view. Set pose and contact in the blockout and measure them
instead of eyeballing:

- feet: minimum sole height above the floor and heel-to-toe height (first run: 1.1 to 1.7 mm, flat within 2
  mm);
- seat: the vertical gap between the body and the seat surface, and a cushion compressed where the body sits
  (first run: up to 10.5 cm);
- hands: distance to the surface they rest on (2.2 to 3.4 mm);
- penetration into furniture with a strict normal-aligned test (first run: calves into the sofa's front rail by
  up to about 5 cm, accepted and reported).

## UVs

Every object needs one UV map with no overlapping texels inside [0, 1]². Downloaded assets often have stacked
islands (the left and right shoes shared one island) or folded strips (the sofa's piping), and automatic
unwraps of closed volumes overlap too. `new` audits every object by rasterising its UV triangles and repairs any
overlap: the offending polygons become their own islands and the atlas is re-packed (first run: 85 such polygons
on the sofa and 39 on the hair, 0 overlapping texels afterwards). Projection into an overlapping atlas writes two
surfaces into the same texel, and nothing downstream can separate them.

## Refit by silhouette carving

When a `refit` object is larger than what was generated, the projection paints background or skin inside its
outline, and from every camera it reads as a shell. In the first run the hair volume was larger than the
generated bob: it rendered as a helmet with skin-coloured patches in every view.

`refit CASE [--object NAME] [--mode solid|shell] [--dilate PX] [--max-faces N] [--apply]` carves every object marked
`refit` (or the one named) to its generated masks, a visual hull restricted around the proxy:

1. A voxel grid over the object's bounding box grown by an outer offset, with the signed distance to the
   current mesh. Candidates are voxels inside the mesh or up to the outer offset (`solid`), or only a layer from
   a little inside to the outer offset (`shell`, for hair over a head). Voxel size and offsets scale with the
   object from the first run's hair values (2.5 mm voxels, 12 mm in, 35 mm out on a 0.36 m object).
2. In every view, a candidate is removed when it is in frame, the camera's ray reaches it without hitting other
   geometry first (first-surface test on the view's depth and id passes), and it projects outside the object's
   generated mask (dilated by `--dilate` px, 4 by default) or onto another object's generated mask.
3. The largest connected piece is kept, smoothed, meshed by marching cubes into a closed surface and smoothed
   again; Blender decimates it (`--max-faces`, 30000 by default) and unwraps a fresh UV map with island margins.
4. The result goes into a new `blockout_refit.blend` (`blockout.blend` is never overwritten), with exact passes
   of the new blend and `refit/refit.json`: per view, mask IoU, object pixels outside its mask and pixels
   covering other objects' masks, before and after; sheets in `refit/`.

It refuses when masks are missing or the object is seen by fewer than two views. Without `--apply` it only
writes the new blend and the report, so you can look first. `refit CASE --apply` keeps the current blockout as
`blockout_pre_refit.blend` (and refuses if that file already exists) and switches the case to the carved one;
then re-run `views` and every stage after it. Occlusion is known only from the first visible surface of each view, so geometry hidden behind
the oversized object itself is not used.

First run, hair IoU against the generated hair masks, before → after: 0.55 → 0.75, 0.66 → 0.76, 0.77 → 0.80,
0.66 → 0.84, 0.85 → 0.82, 0.63 → 0.84. The close-up's hair PSNR in the re-render went from 18.1 to 24.2 dB.
Limits: six views carve flat facets and a small dent; the back and underside are bounded only by the candidate
shell; waves and strands are not modelled.

Never carve to one view, and refit only objects you marked `refit` in `case.json`: geometry is edited to agree with several generated views, not to hide a single bad one.
