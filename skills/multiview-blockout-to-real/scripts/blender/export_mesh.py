"""Blender: export every renderable mesh (evaluated at RENDER subdivision levels) to npz for the rasteriser.

    blender -b --factory-startup --disable-autoexec --python export_mesh.py -- --blend blockout.blend --out DIR [--uv UVMap]

DIR/mesh_<id>.npz {tri_pos (T,3,3) world, tri_uv (T,3,2), tri_nrm (T,3,3) world corner normals} and
DIR/meshes.json [{name, id (pass_index), file, triangles}]. Ported from the experiment (bpy_export_mesh.py).
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_bpy import args_after_dashes, mesh_objects, open_blend, parser  # noqa: E402


def to_render_levels(scene):
    for o in scene.objects:
        for m in getattr(o, "modifiers", []):
            if m.type == "SUBSURF" and m.levels != m.render_levels:
                m.levels = m.render_levels


def export_one(o, dg, uvname, out):
    oe = o.evaluated_get(dg)
    me = oe.to_mesh()
    try:
        if uvname not in me.uv_layers:
            return None, f"no UV layer {uvname}"
        me.calc_loop_triangles()
        nl, nv, nt = len(me.loops), len(me.vertices), len(me.loop_triangles)
        if nt == 0:
            return None, "no triangles"
        uv = np.empty(nl * 2)
        me.uv_layers[uvname].data.foreach_get("uv", uv)
        co = np.empty(nv * 3)
        me.vertices.foreach_get("co", co)
        lv = np.empty(nl, np.int64)
        me.loops.foreach_get("vertex_index", lv)
        cn = np.empty(nl * 3)
        me.corner_normals.foreach_get("vector", cn)
        tl = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("loops", tl)
        tl = tl.reshape(nt, 3)
        M = np.array(oe.matrix_world, float)
        cow = co.reshape(nv, 3) @ M[:3, :3].T + M[:3, 3]
        nrm = cn.reshape(nl, 3)[tl] @ np.linalg.inv(M[:3, :3])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        pid = int(o.pass_index)
        f = out / f"mesh_{pid:03d}.npz"
        np.savez_compressed(f, tri_pos=cow[lv[tl]], tri_uv=uv.reshape(nl, 2)[tl], tri_nrm=nrm)
        return {"name": o.name, "id": pid, "file": f.name, "triangles": nt}, None
    finally:
        oe.to_mesh_clear()


def main():
    ap = parser(__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uv", default="UVMap")
    a = ap.parse_args(args_after_dashes())
    open_blend(a.blend)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    to_render_levels(scene)
    dg = bpy.context.evaluated_depsgraph_get()
    listing, skipped = [], []
    for o in mesh_objects(scene):
        rec, why = export_one(o, dg, a.uv, out)
        (listing.append(rec) if rec else skipped.append({"name": o.name, "reason": why}))
    ids = [d["id"] for d in listing]
    if len(ids) != len(set(ids)) or 0 in ids:
        raise SystemExit(f"pass_index must be unique and >= 1 per mesh: {ids}")
    (out / "meshes.json").write_text(json.dumps({"uv_layer": a.uv, "objects": listing, "skipped": skipped}, indent=1))
    print(f"[export_mesh] {len(listing)} meshes, skipped {len(skipped)}")


if __name__ == "__main__":
    main()
