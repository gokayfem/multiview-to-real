"""Blender: per-view shading basis of the grey blockout under the scene's OWN lights.

    blender -b --factory-startup --disable-autoexec --python shading_basis.py -- --blend blockout.blend --cameras cams.json \
        --out pbr/shading [--scale 0.5] [--samples 32] [--grey 0.5]

Every mesh gets one grey Diffuse BSDF. Two renders per view, linear EXR divided by the grey albedo:
<view>_sun.exr (SUN lights only, world off) and <view>_sky.exr (world + every other light); rig.json.
Ported from the experiment's bpy_shading_basis.py.
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_bpy import (args_after_dashes, load_views, make_camera, mesh_objects, open_blend, parser,  # noqa: E402
                         read_multilayer, set_format, setup_cycles, write_exr)


def background(scene):
    w = scene.world
    if w is None or w.node_tree is None:
        return None
    return next((n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)


def grey_material(g):
    mat = bpy.data.materials.new("shading_grey")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out, df = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeBsdfDiffuse")
    df.inputs["Color"].default_value = (g, g, g, 1)
    nt.links.new(df.outputs[0], out.inputs["Surface"])
    return mat


def main():
    ap = parser(__doc__)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--grey", type=float, default=0.5)
    a = ap.parse_args(args_after_dashes())
    open_blend(a.blend)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    mat = grey_material(a.grey)
    for o in mesh_objects(scene):
        o.data.materials.clear()
        o.data.materials.append(mat)
    setup_cycles(scene, a.samples)
    scene.cycles.use_denoising = True
    lights = [o for o in scene.objects if o.type == "LIGHT" and not o.hide_render]
    suns = [o for o in lights if o.data.type == "SUN"]
    bg = background(scene)
    wstr = bg.inputs["Strength"].default_value if bg else 0.0
    rig = {"suns": [{"name": o.name, "energy": o.data.energy, "color": list(o.data.color),
                     "direction": list(o.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 0, -1)))}
                    for o in suns], "other_lights": [o.name for o in lights if o not in suns], "world_strength": wstr}
    (out / "rig.json").write_text(json.dumps(rig, indent=1))
    for v in load_views(a.cameras):
        cam, W, H = make_camera(v, scale=a.scale)
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = W, H
        for comp in ("sun", "sky"):
            for o in lights:
                o.hide_render = (o in suns) != (comp == "sun")
            if bg is not None:
                bg.inputs["Strength"].default_value = 0.0 if comp == "sun" else wstr
            set_format(scene.render.image_settings, "OPEN_EXR_MULTILAYER", "32", "RGB")
            tmp = out / f"_{v['name']}_{comp}.exr"
            scene.render.filepath = str(tmp)
            bpy.ops.render.render(write_still=True)
            ch = read_multilayer(tmp)
            tmp.unlink()
            rgb = np.stack([ch[[k for k in ch if k.endswith(f"Combined.{c}")][0]] for c in "RGB"], -1) / a.grey
            write_exr(out / f"{v['name']}_{comp}.exr", rgb, ["R", "G", "B"])
        bpy.data.objects.remove(cam)
    for o in lights:
        o.hide_render = False
    print("[shading_basis] done")


if __name__ == "__main__":
    main()
