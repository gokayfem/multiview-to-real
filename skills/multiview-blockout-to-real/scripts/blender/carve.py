"""Blender: replace one object's mesh with a carved mesh, decimate, re-unwrap UVMap, save a NEW blend.

    blender -b --factory-startup --disable-autoexec --python carve.py -- --blend blockout.blend --object NAME \
        --mesh carved.npz --out blockout_refit.blend [--max-faces 30000] [--margin 0.01]

carved.npz: verts (V,3) WORLD coordinates, faces (F,3). The object keeps its name, pass_index, materials and
transform; the input blend is never overwritten. Writes <out>.json with face count, closedness and UV stats.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_bpy import args_after_dashes, open_blend, parser  # noqa: E402


def replace_mesh(o, V, F):
    M = np.array(o.matrix_world, float)
    local = (np.c_[V, np.ones(len(V))] @ np.linalg.inv(M).T)[:, :3]
    me = bpy.data.meshes.new(o.data.name + "_refit")
    me.from_pydata(local.tolist(), [], F.tolist())
    me.update()
    for m in o.data.materials:
        me.materials.append(m)
    old = o.data
    o.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def select_only(o):
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o


def decimate(o, max_faces):
    if len(o.data.polygons) <= max_faces:
        return
    mod = o.modifiers.new("decimate", "DECIMATE")
    mod.ratio = max_faces / len(o.data.polygons)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def unwrap(o, margin):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")
    for uv in list(o.data.uv_layers)[1:]:
        o.data.uv_layers.remove(uv)
    o.data.uv_layers[0].name = "UVMap"


def stats(o):
    bm = bmesh.new()
    bm.from_mesh(o.data)
    closed = all(len(e.link_faces) == 2 for e in bm.edges)
    n = (len(bm.verts), len(bm.faces))
    bm.free()
    uv = np.empty(len(o.data.loops) * 2)
    o.data.uv_layers["UVMap"].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    return {"verts": n[0], "faces": n[1], "closed": closed,
            "uv_in_unit_square": bool(uv.min() >= 0 and uv.max() <= 1)}


def main():
    ap = parser(__doc__)
    ap.add_argument("--object", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=30000)
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args(args_after_dashes())
    if Path(a.out).resolve() == Path(a.blend).resolve():
        raise SystemExit("refusing to overwrite the input blend")
    open_blend(a.blend)
    o = bpy.data.objects.get(a.object)
    if o is None or o.type != "MESH":
        raise SystemExit(f"no mesh object {a.object!r}")
    z = np.load(a.mesh, allow_pickle=False)
    replace_mesh(o, z["verts"], z["faces"])
    select_only(o)
    decimate(o, a.max_faces)
    bpy.ops.object.shade_smooth()
    unwrap(o, a.margin)
    info = stats(o)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()), compress=True)
    Path(a.out + ".json").write_text(json.dumps(info, indent=1))
    print("[carve]", json.dumps(info))


if __name__ == "__main__":
    main()
