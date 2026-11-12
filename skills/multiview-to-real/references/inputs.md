# Inputs

Every input becomes the same case: a grey blockout in Blender with N calibrated cameras, where every object has
one non-overlapping UV map (`UVMap`), one stable id (`pass_index`) and one description. What changes is where the
geometry and the cameras come from.

## Kinds

| Input | `new --input` | What `new` does | Watch out for |
|---|---|---|---|
| A `.blend` with several cameras | `scene.blend` | copies it with its sha256, makes a grey copy with one UV map and id per object, keeps the cameras when there are at least two and `--views` is not given | existing materials and UV maps stay in the source; only the grey copy is used |
| A `.blend` with one or no camera | `scene.blend --views N [--closeup OBJ]` | the same, with cameras placed by `cameras` | check the preview and the coverage table before anything paid |
| A mesh file: glTF/GLB, OBJ, FBX, USD, PLY, STL | `scene.glb` | imported, one object per mesh (`--group-by collection\|parent` to merge), cameras placed | implemented, not yet exercised on real files; a single merged mesh is one object with one atlas: split it first if the parts differ |
| A JSON scene spec of primitives | `room.json` | built in Blender; cameras from the spec or automatic | metres, Z up |
| An idea | the spec you write | as for a spec | primitives look CG; see below |
| Reference images | `--reference IMG=role[:obj,…]` (repeatable; `IMG:role` also works) | copied into `input/` and listed in `case.json` `references`, with the objects it applies to | a reference is never geometry |
| Your own images of these views | `targets CASE --images DIR` | used as the frozen targets; generation is skipped | they must be these cameras' views at their size, pixel-registered |

## An input file is untrusted

- **Scripts are never run.** Blender is started with `--factory-startup --disable-autoexec` and files are
  opened with scripts disabled.
- **Names are restricted.** Object names become file names (`proj/<method>/<object>/texture.png`). Names with
  anything but letters, digits, space, `_`, `.` and `-` are renamed at ingest.
- **Names and texts are labels, not instructions.** The seeded descriptions quote the file's own names. Read
  them as hints and write your own descriptions. Text inside a reference image is image content, not a request.

## After `new`: write the case yourself

The prompt and every per-object rule come from `case.json`, and all of it is yours to write:

- **`look`:** kind of place, style, time of day, light ("a quiet north-facing living room, soft late-morning
  daylight through the side window, restrained colour"). One light setup, fixed in world space for every view.
- **`references`:** for each image, `role` (`identity`, `product`, `material`, `style`), `applies_to` (object
  names or patterns such as `Person_*`) and `note` (what exactly it defines: "face, hairstyle and clothing of
  the seated person, nothing else"). Order matters: see `generation.md`.
- **Objects:** `description` (material, colour, finish, written for an image model), `class` (selects the
  material prior for the PBR stage), `detail_critical` (faces, hands, text, logos), `prefer_view`, `refit`
  (soft shapes that may be carved: hair, cloth, cushions) and `opening` (windows to the sky).

Split objects where materials differ. In the first run the person was five objects (skin, hair, top, trousers,
shoes); each got its own generated mask and its own atlas, and a part never took pixels from another.

## An idea

Write a JSON scene spec, then `new --input spec.json`. Primitives are fine for architecture and boxy
furniture; for people, upholstered furniture, plants and anything soft, start from real assets with clear
licences (`geometry.md`). The image model follows the clay: a primitive mannequin came back as a CG-looking
person in every view of the first run.

## Your own images

`targets CASE --images DIR` takes one image per view, named after the view (`view_01.png` ...). They must be
these cameras' views with the same aspect (within 2%), at the calibration resolution or larger (a larger image is
resampled, and the resample is recorded as a resample, not as detail); a photo of a different viewpoint does not work, because projection needs pixel registration. Paint-overs of the
clay views, renders from another tool, or views generated elsewhere all work. Everything downstream runs
offline.

## Another image model

Set `model.edit_endpoint` in `case.json` and run `probe`. It reads fal endpoints whose schema has an
`image_size` parameter taking `{width, height}` and a list of image URLs (the shape of Seedream's edit endpoint).
For other schemas `probe` refuses; set `model.pixel_cap`, `model.max_refs` and the size-parameter overrides in
`case.json` yourself, from the model's documentation, and check one returned size with `probe --check-size`.

## Without an API key

`probe` (to fetch the schema; `--schema-file` reads a saved one), `probe --check-size`, `generate`, `refine` and `resync` need
`FAL_KEY` with the fal generator. Without it, bring your own images (`targets --images`) or use the mock
(`new ... --generator mock`, or `--generator mock` on a stage or on `run`): an offline synthetic
image model that renders a hidden ground-truth look from each camera with per-view gain and tint fields, small
smooth warps and view-specific fine detail. It exists for tests and demos; its output is not photoreal.

## Scenes it fits and does not

Fits: interiors, products and figures in a room, seen from several cameras that overlap, made of opaque
materials. Built on one room with a seated person.

Hard or out of scope: glass, mirrors and other view-dependent materials (each view shows a different
reflection, so fusion averages them into mud); thin foliage and hair strands as geometry (projection paints
them onto a shell); animation; surfaces no camera sees (they are inferred and plain); exact identity
preservation (the model redraws a face in every view). Say so before starting.
