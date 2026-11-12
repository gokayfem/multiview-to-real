"""Blender: per-view shading basis of the grey blockout under the scene's OWN lights.

    blender -b --factory-startup --disable-autoexec --python shading_basis.py -- --blend blockout.blend --cameras cams.json \
        --out pbr/shading [--scale 0.5] [--samples 32] [--grey 0.5]

Every mesh gets one grey Diffuse BSDF. Two renders per view, linear EXR divided by the grey albedo:
<view>_sun.exr (SUN lights only, world off) and <view>_sky.exr (world + every other light); rig.json.
<view>_sundirect.exr: the Diffuse Direct pass (EXR channel "DiffDir", "Diffuse Direct" in Blender 5) of the sun render (light the SUN reaches directly, no bounce,
no albedo), used by the opt-in direct-light patch synthesis (mvb2r/sunpatch_stage.py).
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


def render_component(scene, out: Path, view: str, comp: str, grey: float) -> None:
    """One multilayer render of `comp` ("sun" or "sky"): <view>_<comp>.exr (combined / grey) and, for the sun,
    <view>_sundirect.exr (diffuse direct, which carries no albedo) when Blender wrote that pass. Both channel
    names are accepted: "Diffuse Direct", which Blender 5.0 writes, and "DiffDir"."""
    set_format(scene.render.image_settings, "OPEN_EXR_MULTILAYER", "32", "RGB")
    tmp = out / f"_{view}_{comp}.exr"
    scene.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    ch = read_multilayer(tmp)
    tmp.unlink()
    rgb = np.stack([ch[[k for k in ch if k.endswith(f"Combined.{c}")][0]] for c in "RGB"], -1) / grey
    write_exr(out / f"{view}_{comp}.exr", rgb, ["R", "G", "B"])
    direct = [[k for k in ch if k.endswith((f"DiffDir.{c}", f"Diffuse Direct.{c}"))] for c in "RGB"]
    if comp == "sun" and all(direct):
        write_exr(out / f"{view}_sundirect.exr", np.stack([ch[d[0]] for d in direct], -1), ["R", "G", "B"])


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
    for vl in scene.view_layers:
        vl.use_pass_diffuse_direct = True
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
            render_component(scene, out, v["name"], comp, a.grey)
        bpy.data.objects.remove(cam)
    for o in lights:
        o.hide_render = False
    print("[shading_basis] done")


if __name__ == "__main__":
    main()
