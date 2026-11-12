"""Shared Blender-side library: clay material, the `UVMap` audit and repair, stable ids, texel density,
script stripping, lights, Cycles setup and the common finish of `new` (blockout.py / ingest.py).

Imported by the Blender scripts only (needs bpy). UV rules:
  - every object ends with exactly ONE UV map named `UVMap`, inside [0,1]^2, with zero overlapping texels at
    1024^2 (overlap = a texel covered by two different triangles, which also catches stacked and mirrored
    islands of downloaded assets);
  - an existing map that passes is kept (so authored UVs survive); otherwise Smart UV Project + pack, then the
    polygons that still overlap become their own islands (lightmap pack of the selection + re-pack), up to
    REPAIR_ROUNDS times; the result is checked again and the object fails loudly if it still overlaps.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from mvb2r import common  # noqa: E402

UV_NAME = "UVMap"
CLAY_NAME = "mvb2r_clay"
CLAY = (0.5, 0.5, 0.5, 1.0)
UV_RES = 1024
REPAIR_ROUNDS = 6
STRUCT_SPAN, SLAB_RATIO = 0.6, 0.10
CLASS_WORDS = ("plaster", "paint", "wood", "fabric", "knit", "leather", "skin", "hair", "metal", "plastic", "stone")
NOTES: list[str] = []


def log(tag: str, *a) -> None:
    print(f"[{tag}]", *a, flush=True)


def note(tag: str, msg: str) -> None:
    NOTES.append(msg)
    log(tag, "NOTE " + msg)


# ---------------------------------------------------------------- material
def clay_material():
    mat = bpy.data.materials.get(CLAY_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(CLAY_NAME)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = CLAY
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Metallic"].default_value = 0.0
    mat.diffuse_color = CLAY
    return mat


def set_clay(obj, mat) -> None:
    me = obj.data
    me.materials.clear()
    me.materials.append(mat)
    if len(me.polygons):
        me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), dtype=np.int32))
    me.update()


# ---------------------------------------------------------------- spec primitives (ported from Blockout 2 Real)
def euler_deg(r) -> Euler:
    return Euler([math.radians(a) for a in (r or (0, 0, 0))], "XYZ")


def add_part(bm, part: dict) -> None:
    t = part["type"]
    before = set(bm.verts)
    if t == "box":
        sx, sy, sz = part["size"]
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((sx, sy, sz, 1.0)))
    elif t in ("cylinder", "cone"):
        r1, r2 = (part["radius"], part["radius"]) if t == "cylinder" else (part["radius_bottom"], part["radius_top"])
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=int(part.get("segments", 32)),
                              radius1=r1, radius2=r2, depth=part["depth"])
    elif t == "sphere":
        dx, dy, dz = part["size"] if "size" in part else [2.0 * part["radius"]] * 3
        bmesh.ops.create_uvsphere(bm, u_segments=int(part.get("segments", 32)), v_segments=int(part.get("rings", 16)),
                                  radius=0.5, matrix=Matrix.Diagonal((dx, dy, dz, 1.0)))
    elif t == "plane":
        sx, sy = part["size"]
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5, matrix=Matrix.Diagonal((sx, sy, 1.0, 1.0)))
    verts = [v for v in bm.verts if v not in before]
    if t != "plane":
        bmesh.ops.recalc_face_normals(bm, faces=list({f for v in verts for f in v.link_faces}))
    m = Matrix.Translation(Vector(part.get("location", (0, 0, 0)))) @ euler_deg(part.get("rotation")).to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=m, verts=verts)


def mesh_from_parts(name: str, parts: list, rotation=None):
    bm = bmesh.new()
    for part in parts:
        add_part(bm, part)
    bmesh.ops.transform(bm, matrix=euler_deg(rotation).to_matrix().to_4x4(), verts=bm.verts[:])
    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def aim(obj, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler("XYZ")


def add_camera(scene, name: str, location, look_at=None, rotation=None, lens: float = 35.0, sensor: float = 36.0):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit, cd.sensor_width, cd.lens = "AUTO", float(sensor), float(lens)
    cd.clip_start, cd.clip_end = 0.02, 1000.0
    cam = bpy.data.objects.new(name, cd)
    scene.collection.objects.link(cam)
    cam.location = Vector(location)
    if look_at is not None:
        aim(cam, look_at)
    else:
        cam.rotation_euler = euler_deg(rotation)
    return cam


def add_light(scene, d: dict):
    lt = d.get("type", "POINT").upper()
    ld = bpy.data.lights.new(d.get("name", lt.lower()), lt)
    ld.energy = float(d.get("energy", 100.0))
    ld.color = tuple(d.get("color", (1.0, 1.0, 1.0)))[:3]
    if lt == "AREA":
        ld.size = float(d.get("size", 1.0))
    elif lt == "SUN":
        ld.angle = math.radians(float(d.get("angle_deg", 2.0)))
    ob = bpy.data.objects.new(d.get("name", lt.lower()), ld)
    scene.collection.objects.link(ob)
    ob.location = Vector(d.get("location", (0, 0, 0)))
    if "look_at" in d:
        aim(ob, d["look_at"])
    elif "direction" in d:
        ob.rotation_euler = Vector(d["direction"]).normalized().to_track_quat("-Z", "Y").to_euler()
    return ob


def ensure_lighting(scene, tag: str) -> None:
    """Neutral preview lights (named mvb2r_preview_*) only when the scene has no light and no world."""
    if any(o.type == "LIGHT" for o in scene.objects) or scene.world is not None:
        return
    add_light(scene, {"name": "mvb2r_preview_sun", "type": "SUN", "energy": 3.5, "direction": (-0.35, -1.0, -0.55)})
    world = bpy.data.worlds.new("mvb2r_preview_world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.65, 0.8, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    scene.world = world
    note(tag, "the scene had no lights and no world: neutral preview lights were added (mvb2r_preview_*)")


# ---------------------------------------------------------------- UV audit and repair
def _select_only(obj) -> None:
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def uv_triangles(obj):
    """(uv (T,3,2), polygon index (T,)) of the UVMap loop triangles."""
    me = obj.data
    me.calc_loop_triangles()
    lay = me.uv_layers[UV_NAME].data
    uv = np.empty(len(lay) * 2)
    lay.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    n = len(me.loop_triangles)
    loops = np.empty(n * 3, np.int64)
    me.loop_triangles.foreach_get("loops", loops)
    poly = np.empty(n, np.int64)
    me.loop_triangles.foreach_get("polygon_index", poly)
    return uv[loops].reshape(n, 3, 2), poly


def uv_audit(obj, res: int = UV_RES) -> dict:
    """Overlapping texels (two different triangles), the polygons involved, range and area of UVMap."""
    tris, poly = uv_triangles(obj)
    owner = -np.ones((res, res), np.int64)
    bad, clash_px, area = set(), 0, 0.0
    P = tris * res
    for t in range(len(P)):
        a, b, c = P[t]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        area += abs(d) / 2 / res / res
        if abs(d) < 1e-12:
            continue
        x0, y0 = np.maximum(np.floor(np.minimum(np.minimum(a, b), c)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(np.maximum(np.maximum(a, b), c)).astype(int), res)
        if x1 <= x0 or y1 <= y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
        l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
        m = (l1 > 1e-9) & (l2 > 1e-9) & (1 - l1 - l2 > 1e-9)
        if not m.any():
            continue
        blk = owner[y0:y1, x0:x1]
        prev = blk[m]
        clash = prev >= 0
        if clash.any():
            clash_px += int(clash.sum())
            bad.add(int(poly[t]))
            bad.update(int(poly[x]) for x in np.unique(prev[clash]))
        blk[m & (blk < 0)] = t
    flat = tris.reshape(-1, 2) if len(tris) else np.zeros((1, 2))
    return {"overlap_px": clash_px, "bad_polygons": bad, "uv_area_fraction": round(float(area), 4),
            "in_unit_square": bool((flat >= -1e-6).all() and (flat <= 1 + 1e-6).all()),
            "coverage_fraction": round(float((owner >= 0).mean()), 4)}


def _smart_unwrap(obj) -> None:
    _select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.01, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, shape_method="AABB", margin_method="FRACTION", margin=0.006)
    bpy.ops.object.mode_set(mode="OBJECT")


def _split_bad(obj, bad: set) -> None:
    """The overlapping polygons (and their 1-ring) become their own islands; all islands re-packed."""
    _select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    ring = set(bad)
    for i in bad:
        for e in bm.faces[i].edges:
            ring.update(g.index for g in e.link_faces)
    for f in bm.faces:
        f.select = f.index in ring
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT="SEL_FACES", PREF_PACK_IN_ONE=True, PREF_NEW_UVLAYER=False, PREF_MARGIN_DIV=0.2)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin_method="FRACTION", margin=0.004, shape_method="AABB")
    bpy.ops.object.mode_set(mode="OBJECT")


def _single_uvmap(obj) -> bool:
    """Make UVMap the only UV map, keeping the active-render map's data. True if there was data to keep."""
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new(name=UV_NAME)
        return False
    keep = next((u for u in me.uv_layers if u.active_render), me.uv_layers[0])
    keep.name = "__mvb2r_keep"
    for u in [u for u in me.uv_layers if u.name != "__mvb2r_keep"]:
        me.uv_layers.remove(u)
    me.uv_layers["__mvb2r_keep"].name = UV_NAME
    me.uv_layers[UV_NAME].active = True
    me.uv_layers[UV_NAME].active_render = True
    return True


def ensure_uv(obj, tag: str) -> dict:
    """Exactly one non-overlapping UVMap in [0,1]^2 (kept when it already passes, else rebuilt and repaired)."""
    had = _single_uvmap(obj)
    st = uv_audit(obj) if had else None
    if st and not st["overlap_px"] and st["in_unit_square"] and st["uv_area_fraction"] > 0.05:
        return {**_public(st), "method": "kept (existing map passes)", "repaired_polygons": 0}
    before = st["overlap_px"] if st else None
    _smart_unwrap(obj)
    st, repaired = uv_audit(obj), 0
    for _ in range(REPAIR_ROUNDS):
        if not st["overlap_px"]:
            break
        repaired += len(st["bad_polygons"])
        _split_bad(obj, st["bad_polygons"])
        st = uv_audit(obj)
    if st["overlap_px"] or not st["in_unit_square"]:
        raise RuntimeError(f"{obj.name}: UVMap still overlaps ({st['overlap_px']} px) or leaves [0,1] after repair")
    method = "smart project + pack" + (f", {repaired} overlapping polygons split into own islands" if repaired else "")
    if had:
        note(tag, f"{obj.name}: existing UV map rejected ({before} overlapping texels at {UV_RES}^2 or out of [0,1]); rebuilt")
    return {**_public(st), "method": method, "repaired_polygons": repaired, "overlap_before_px": before}


def _public(st: dict) -> dict:
    return {k: v for k, v in st.items() if k != "bad_polygons"}


# ---------------------------------------------------------------- geometry facts
def world_verts(obj) -> np.ndarray:
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    M = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def categorise(objs: list) -> dict:
    """structure: spans >= 60% of the scene box along two axes and is a slab (floors, walls, ceilings)."""
    boxes = {o.name: (world_verts(o).min(0), world_verts(o).max(0)) for o in objs}
    lo = np.min([b[0] for b in boxes.values()], axis=0)
    hi = np.max([b[1] for b in boxes.values()], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    out = {}
    for name, (a, b) in boxes.items():
        ext = b - a
        big = int(((ext / span) >= STRUCT_SPAN).sum()) >= 2
        slab = ext.min() <= SLAB_RATIO * max(ext.max(), 1e-6)
        out[name] = "structure" if big and slab and len(objs) > 2 else "object"
    return out


def guess_class(name: str, hints: str) -> str:
    text = f"{name} {hints}".lower()
    return next((c for c in CLASS_WORDS if c in text), "generic")


def class_of(name: str, hints: dict) -> tuple[str, str]:
    """(class, source): stated by a spec, else a GUESS from the name / material names (printed by `new`)."""
    if hints.get("class") in CLASS_WORDS + ("generic",):
        return hints["class"], "stated in the spec"
    g = guess_class(name, hints.get("materials", ""))
    return g, ("guessed from the name" if g != "generic" else "default")


def assign_ids(objs: list) -> dict:
    """Stable pass_index: an existing unique id in 1..65535 is kept; the others get the next free ids in name order."""
    ids, used = {}, set()
    for o in sorted(objs, key=lambda o: o.name):
        p = int(o.pass_index)
        if 1 <= p <= 65535 and p not in used:
            ids[o.name], used = p, used | {p}
    nxt = max(used, default=0) + 1
    for o in sorted(objs, key=lambda o: o.name):
        if o.name not in ids:
            ids[o.name], nxt = nxt, nxt + 1
    for o in objs:
        o.pass_index = ids[o.name]
    return ids


# ---------------------------------------------------------------- scripts in input files
def strip_scripts() -> dict:
    """Remove every text block and driver (nothing was run: --disable-autoexec, use_scripts=False)."""
    n_drv = 0
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.worlds, bpy.data.scenes, bpy.data.shape_keys, bpy.data.node_groups):
        for idb in coll:
            ad = getattr(idb, "animation_data", None)
            if ad:
                for d in list(ad.drivers):
                    ad.drivers.remove(d)
                    n_drv += 1
    texts = [common.clean_text(t.name, 60) for t in bpy.data.texts]
    for t in list(bpy.data.texts):
        bpy.data.texts.remove(t)
    return {"text_blocks": texts, "drivers": n_drv}


# ---------------------------------------------------------------- external images of an input file
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff", ".tga", ".bmp", ".webp")
MAX_EXTERNAL_BYTES = 256 * 1024 * 1024
MAX_EXTERNAL_PIXELS = common.MAX_IMAGE_PIXELS
EXTERNAL: dict = {"packed": [], "missing": [], "refused": [], "libraries": []}


def _resolve_path(fp: str, source_dir: Path | None) -> Path | None:
    """Absolute path (symlinks followed); relative `//` paths resolved against the ORIGINAL input's folder."""
    if not fp:
        return None
    if fp.startswith("//"):
        if source_dir is None:
            return None
        return (source_dir / fp[2:].replace("\\", "/")).resolve()
    return Path(bpy.path.abspath(fp)).resolve()


def external_verdict(p: Path | None, source_dir: Path | None, allow: bool) -> str | None:
    """None when the file may be read, else why not (missing / outside the input folder / type / size)."""
    if p is None or not p.is_file():
        return "missing"
    if not allow and (source_dir is None or not p.is_relative_to(source_dir.resolve())):
        return "outside the input folder"
    if p.suffix.lower() not in IMAGE_EXTS:
        return "not an image file type"
    if p.stat().st_size > MAX_EXTERNAL_BYTES:
        return "file too large"
    return None


def _pack_checked(img, p: Path) -> str | None:
    img.filepath = str(p)
    img.reload()
    w, h = img.size[0], img.size[1]
    if w <= 0 or h <= 0:
        return "did not load as an image"
    if w * h > MAX_EXTERNAL_PIXELS:
        return f"{w}x{h} exceeds {MAX_EXTERNAL_PIXELS} px"
    try:
        img.pack()
    except RuntimeError as e:
        return f"cannot be packed ({e})"
    return None


def resolve_external(source_dir: Path | None, tag: str, allow: bool = False) -> dict:
    """Pack the file images the scene references that lie inside the input's folder (or anywhere with `allow`),
    after checking they load and are within the pixel cap; list missing and refused ones."""
    for img in bpy.data.images:
        if img.source not in ("FILE", "SEQUENCE", "TILED") or img.packed_file is not None:
            continue
        name = common.clean_text(img.name, 60)
        raw = img.filepath_raw or img.filepath
        p = _resolve_path(raw, source_dir)
        why = external_verdict(p, source_dir, allow)
        if why is None:
            why = _pack_checked(img, p)
            if why is None:
                EXTERNAL["packed"].append({"image": name, "file": p.name, "bytes": p.stat().st_size})
                continue
        entry = {"image": name, "path": common.clean_text(Path(raw).name if raw else "", 120), "why": why}
        EXTERNAL["missing" if why == "missing" else "refused"].append(entry)
        if why != "missing":
            img.filepath = ""          # never rendered or packed later
        note(tag, f"WARNING external image {name!r} ({entry['path']}) not used: {why}")
    return EXTERNAL


def check_libraries(source_dir: Path | None, tag: str, allow: bool) -> None:
    """Linked .blend libraries: outside the input folder they are removed unless the user trusts the file."""
    for lib in list(bpy.data.libraries):
        p = _resolve_path(lib.filepath, source_dir)
        inside = p is not None and source_dir is not None and p.is_relative_to(source_dir.resolve())
        action = "kept (inside the input folder)" if inside else (
            "kept (--allow-external-images)" if allow else "REMOVED: outside the input folder (its data is absent)")
        EXTERNAL["libraries"].append({"path": common.clean_text(Path(lib.filepath).name, 120), "action": action})
        note(tag, f"WARNING linked library {Path(lib.filepath).name!r}: {action}")
        if not inside and not allow:
            bpy.data.libraries.remove(lib)


def _image_ok(img) -> bool:
    if img is None:
        return False
    if img.packed_file is not None:
        return True
    try:
        return Path(bpy.path.abspath(img.filepath)).is_file()
    except (RuntimeError, ReferenceError):
        return False


def neutral_world_if_missing(scene, tag: str) -> bool:
    """A world that uses an image that cannot load renders the missing-texture colour: replace it."""
    w = scene.world
    if w is None or not w.use_nodes or w.node_tree is None:
        return False
    bad = [n for n in w.node_tree.nodes if n.type in ("TEX_ENVIRONMENT", "TEX_IMAGE") and not _image_ok(n.image)]
    if not bad:
        return False
    nw = bpy.data.worlds.new("mvb2r_neutral_world")
    nw.use_nodes = True
    bg = nw.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.6, 0.62, 0.65, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = nw
    note(tag, f"WARNING world {common.clean_text(w.name, 60)!r} uses a missing image: replaced by a neutral grey world "
              "for the clay renders (the original lighting is NOT used)")
    return True


# ---------------------------------------------------------------- render setup
def setup_cycles(scene, samples: int) -> str:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 4
    scene.cycles.seed = 0
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.cycles.device = "CPU"
    return "CPU"


def mesh_objects(scene) -> list:
    return sorted((o for o in scene.objects if o.type == "MESH" and not o.hide_render), key=lambda o: o.name)


def camera_objects(scene) -> list:
    return sorted((o for o in scene.objects if o.type == "CAMERA" and o.data.type == "PERSP"), key=lambda o: o.name)


# ---------------------------------------------------------------- common finish of `new`
def object_row(o, uv: dict, category: str, hints: dict) -> dict:
    w = world_verts(o)
    area = sum(p.area for p in o.data.polygons)
    texel = math.sqrt(uv["uv_area_fraction"] * UV_RES * UV_RES / area) if area > 0 else None
    desc = hints.get("description") or common.clean_text(o.name.replace("_", " "), 60).lower()
    cls, src = class_of(o.name, hints)
    return {"name": o.name, "id": int(o.pass_index), "pass_index": int(o.pass_index), "category": category,
            "class": cls, "class_source": src, "description": desc,
            "bbox_min": [round(float(x), 5) for x in w.min(0)], "bbox_max": [round(float(x), 5) for x in w.max(0)],
            "n_vertices": len(o.data.vertices), "n_triangles": sum(len(p.vertices) - 2 for p in o.data.polygons),
            "surface_area_m2": round(float(area), 5), "uv_layer": UV_NAME, "uv": uv,
            "texel_px_per_m_at_1024": round(texel, 1) if texel else None}


def finish(case: Path, scene, tag: str, hints: dict, scene_cameras: list) -> None:
    """Clay, UVs, ids, blockout.blend, views/objects.json, views/scene.json."""
    objs = mesh_objects(scene)
    if not objs:
        raise SystemExit("the input has no renderable mesh geometry")
    mat = clay_material()
    ids = assign_ids(objs)
    cats = categorise(objs)
    rows = []
    for o in objs:
        set_clay(o, mat)
        uv = ensure_uv(o, tag)
        rows.append(object_row(o, uv, cats[o.name], hints.get(o.name, {})))
        log(tag, f"{o.name}: id {ids[o.name]}, {rows[-1]['n_triangles']} tris, UV {uv['method']}, "
                 f"texel {rows[-1]['texel_px_per_m_at_1024']} px/m @1024")
    neutral_world_if_missing(scene, tag)
    ensure_lighting(scene, tag)
    stripped = strip_scripts()
    if stripped["text_blocks"] or stripped["drivers"]:
        note(tag, f"removed {len(stripped['text_blocks'])} text block(s) and {stripped['drivers']} driver(s); none was run")
    vdir = case / "views"
    vdir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(case / "blockout.blend"), compress=True)
    common.write_json(vdir / "objects.json", rows)
    common.write_json(vdir / "scene.json", {"objects": rows, "cameras": scene_cameras, "notes": NOTES,
                                            "external_images": EXTERNAL})
    log(tag, f"summary: {len(rows)} objects, {len(scene_cameras)} usable cameras, {len(NOTES)} notes")
