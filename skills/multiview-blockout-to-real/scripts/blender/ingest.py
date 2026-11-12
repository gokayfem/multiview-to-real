"""ingest: any .blend or mesh file -> the grey case scene (CASE/blockout.blend, views/objects.json).

    blender -b --factory-startup --disable-autoexec --python scripts/blender/ingest.py -- --case CASE
        [--group-by object|collection|parent]

Ported from Blockout 2 Real (scripts/blender/ingest.py), reduced to what a multi-view case needs:
  1. the .blend is opened with use_scripts=False (its active scene only), or the mesh file (.glb .gltf .obj
     .fbx .usd* .ply .stl) is imported into an empty scene;
  2. the scene is evaluated with RENDER settings and every renderable instance becomes a plain mesh: modifiers
     applied, instances realised, curves/text/metaballs converted, world transform baked (origin kept at the
     object's world location; for a group at the bottom centre of its box). Volumes, grease pencil, hair
     curves and point clouds are skipped and listed. An existing pass_index is kept (stable ids);
  3. --group-by: one object per evaluated object, per collection, or per top parent;
  4. perspective cameras and lights are kept and frozen at their world matrix; names from the file are
     sanitised before they can become file names;
  5. clay.finish: clay material, one non-overlapping `UVMap` (existing one kept when it passes), ids,
     texel density, scripts stripped, blockout.blend.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clay as C  # noqa: E402
from mvb2r import common  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

TAG = "ingest"
GEOM_TYPES = ("MESH", "CURVE", "SURFACE", "FONT", "META")
SKIP_TYPES = ("VOLUME", "GREASEPENCIL", "GPENCIL", "CURVES", "POINTCLOUD")
IMPORTERS = {".glb": "gltf", ".gltf": "gltf", ".obj": "obj", ".fbx": "fbx", ".usd": "usd", ".usda": "usd",
             ".usdc": "usd", ".usdz": "usd", ".ply": "ply", ".stl": "stl"}


def load_input(path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kind, fp = IMPORTERS.get(ext), str(path)
    ops = {"gltf": lambda: bpy.ops.import_scene.gltf(filepath=fp), "obj": lambda: bpy.ops.wm.obj_import(filepath=fp),
           "fbx": lambda: bpy.ops.import_scene.fbx(filepath=fp), "usd": lambda: bpy.ops.wm.usd_import(filepath=fp),
           "ply": lambda: bpy.ops.wm.ply_import(filepath=fp), "stl": lambda: bpy.ops.wm.stl_import(filepath=fp)}
    if kind is None:
        raise SystemExit(f"unsupported input {ext}")
    if "FINISHED" not in ops[kind]():
        raise SystemExit(f"import of {path.name} failed")


def use_render_settings(scene) -> None:
    for o in scene.objects:
        o.hide_viewport = o.hide_render
        o.hide_set(False)
        for m in getattr(o, "modifiers", []):
            m.show_viewport = m.show_render
            if m.type in ("SUBSURF", "MULTIRES"):
                m.levels = m.render_levels
    for c in bpy.data.collections:
        c.hide_viewport = c.hide_render

    def walk(lc):
        lc.hide_viewport = False
        for ch in lc.children:
            walk(ch)
    walk(bpy.context.view_layer.layer_collection)


def root_of(o):
    while o.parent is not None:
        o = o.parent
    return o


def group_key(mode: str, orig, instancer, counter: dict) -> str:
    top = instancer if instancer is not None else orig
    if mode == "collection":
        cols = [c for c in top.users_collection if c != bpy.context.scene.collection]
        return cols[0].name if cols else root_of(top).name
    if mode == "parent":
        return root_of(top).name
    if instancer is None:
        return orig.name
    k = f"{instancer.name}_{orig.name}"
    counter[k] = counter.get(k, 0) + 1
    return f"{k}_{counter[k]:03d}"


def convert(scene, mode: str):
    dg = bpy.context.evaluated_depsgraph_get()
    groups, skipped, counter = {}, {}, {}
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type in SKIP_TYPES:
            skipped[ob.type] = skipped.get(ob.type, 0) + 1
            continue
        if ob.type not in GEOM_TYPES:
            continue
        orig = ob.original
        instancer = inst.parent.original if (inst.is_instance and inst.parent) else None
        try:
            me = bpy.data.meshes.new_from_object(ob, preserve_all_data_layers=True, depsgraph=dg)
        except RuntimeError as exc:
            C.note(TAG, f"could not convert {common.clean_text(orig.name, 60)}: {exc}")
            continue
        M = inst.matrix_world.copy()
        me.transform(M)
        if M.determinant() < 0:
            me.flip_normals()
        key = group_key(mode, orig, instancer, counter)
        single = M.translation.copy() if (mode == "object" and instancer is None) else None
        groups.setdefault(key, []).append((me, single, int(orig.pass_index), [m.name for m in me.materials if m]))
    return groups, skipped


def join(scene, name: str, members: list):
    objs = []
    for me, *_ in members:
        o = bpy.data.objects.new(name, me)
        scene.collection.objects.link(o)
        objs.append(o)
    obj = objs[0]
    if len(objs) > 1:
        with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=objs, selected_editable_objects=objs):
            bpy.ops.object.join()
    me = obj.data
    if len(members) == 1 and members[0][1] is not None:
        t = members[0][1]
    else:
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) if len(co) else np.zeros((1, 3))
        lo, hi = co.min(0), co.max(0)
        t = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    me.transform(Matrix.Translation(-t))
    obj.location = t
    ids = {m[2] for m in members if m[2] > 0}
    obj.pass_index = ids.pop() if len(ids) == 1 else 0
    return obj, sorted({n for m in members for n in m[3]})


def freeze(o) -> None:
    M = o.matrix_world.copy()
    o.animation_data_clear()
    for c in list(o.constraints):
        o.constraints.remove(c)
    o.parent = None
    o.matrix_world = M


def keep_cameras_and_lights(scene) -> list:
    keep = [o for o in scene.objects if (o.type == "LIGHT" and not o.hide_render) or o.type == "CAMERA"]
    for o in keep:
        freeze(o)
        for c in list(o.users_collection):
            c.objects.unlink(o)
        scene.collection.objects.link(o)
        o.name = common.safe_name(o.name, o.type.lower())
    for o in list(scene.objects):
        if o not in keep:
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    cams = [o for o in keep if o.type == "CAMERA"]
    for o in cams:
        if o.data.type != "PERSP":
            C.note(TAG, f"camera {o.name!r} is {o.data.type}; only perspective cameras are used")
    return [{"name": o.name, "role": "scene camera"} for o in C.camera_objects(scene)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--group-by", default="object", choices=("object", "collection", "parent"))
    ap.add_argument("--allow-external", action="store_true")
    args = ap.parse_args(common.script_args())
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    load_input(case / cfg["input"]["path"])
    src_dir = cfg["input"].get("source_dir")
    src_dir = Path(src_dir) if src_dir else (case / cfg["input"]["path"]).parent
    C.check_libraries(src_dir, TAG, args.allow_external)
    C.resolve_external(src_dir, TAG, args.allow_external)
    scene = bpy.context.scene
    for s in list(bpy.data.scenes):
        if s != scene:
            bpy.data.scenes.remove(s)
    use_render_settings(scene)
    bpy.context.view_layer.update()
    groups, skipped = convert(scene, args.group_by)
    for t, n in skipped.items():
        C.note(TAG, f"{n} {t} object(s) cannot become ordinary meshes and were skipped")
    cams = keep_cameras_and_lights(scene)
    hints, used = {}, set()
    for key in sorted(groups):
        name = base = common.safe_name(key)
        i = 1
        while name in used:
            i += 1
            name = f"{base[:common.NAME_MAX - 4]}_{i}"
        used.add(name)
        obj, mats = join(scene, name, groups[key])
        if len(obj.data.polygons) == 0 or sum(p.area for p in obj.data.polygons) < 1e-10:
            C.note(TAG, f"dropped {name!r}: no surface")
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        hints[obj.name] = {"materials": common.clean_text(" ".join(mats), 200),
                           "description": common.clean_text(name.replace("_", " ").lower() +
                                                            (f"; original materials {', '.join(mats[:3])}" if mats else ""), 300)}
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    C.log(TAG, f"converted {len(hints)} objects ({args.group_by}), {len(cams)} perspective cameras")
    C.finish(case, scene, TAG, hints, cams)


if __name__ == "__main__":
    main()
