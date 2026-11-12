"""Blender: blockout + PBR textures -> out/final.blend (Principled, packed images, cameras, README).

    blender -b --factory-startup --disable-autoexec --python build_pbr.py -- --blend blockout.blend --materials mats.json \
        --out out/final.blend [--cameras cams.json ...] [--readme README.txt] [--exposure EV] [--light-scale S]
        [--openings A,B] [--sky r,g,b]

mats.json: {object name: {texture, height, roughness, specular, bump, metallic}}. The scene's own lights are kept
(scaled by --light-scale) and the view exposure set to --exposure (the rig fit writes both).
"""
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_bpy import args_after_dashes, load_views, make_camera, mesh_objects, open_blend, parser  # noqa: E402


def image(path, colorspace):
    img = bpy.data.images.load(str(Path(path).resolve()), check_existing=False)
    img.colorspace_settings.name = colorspace
    img.pack()
    return img


def principled(name, m):
    mat = bpy.data.materials.new(f"{name}__pbr")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image(m["texture"], "sRGB")
    tex.extension = "EXTEND"
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = float(m["roughness"])
    bsdf.inputs["Metallic"].default_value = float(m.get("metallic", 0.0))
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(m["specular"])
    if m.get("bump", 0) > 0 and m.get("height"):
        ht = nt.nodes.new("ShaderNodeTexImage")
        ht.image = image(m["height"], "Non-Color")
        nt.links.new(uv.outputs["UV"], ht.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = float(m["bump"])
        nt.links.new(ht.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    mat["base_color"] = "projected generated texture x smooth clamped de-light correction (residual baked shading)"
    return mat


def opening_material(name, sky):
    mat = bpy.data.materials.new(f"{name}__opening")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em, out = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = list(sky) + [1]
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def main():
    ap = parser(__doc__)
    ap.add_argument("--materials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", nargs="*", default=[])
    ap.add_argument("--readme", default=None)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--light-scale", type=float, default=1.0)
    ap.add_argument("--openings", default="")
    ap.add_argument("--sky", default="0.8,0.84,0.9")
    a = ap.parse_args(args_after_dashes())
    open_blend(a.blend)
    mats = json.loads(Path(a.materials).read_text())
    openings = set(filter(None, a.openings.split(",")))
    sky = [float(x) for x in a.sky.split(",")][:3]
    for o in mesh_objects(bpy.context.scene):
        if o.name in openings:
            mat = opening_material(o.name, sky)
        elif o.name in mats:
            mat = principled(o.name, mats[o.name])
        else:
            continue
        o.data.materials.clear()
        o.data.materials.append(mat)
    scene = bpy.context.scene
    for o in scene.objects:
        if o.type == "LIGHT":
            o.data.energy *= a.light_scale
    if scene.world is not None and scene.world.node_tree is not None:
        for n in scene.world.node_tree.nodes:
            if n.bl_idname == "ShaderNodeBackground":
                n.inputs["Strength"].default_value *= a.light_scale
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = a.exposure
    for path in a.cameras:
        for v in load_views(path):
            make_camera(v, name=v["name"])
    txt = bpy.data.texts.new("README_final_pbr")
    txt.write(Path(a.readme).read_text() if a.readme else "Approximate PBR from generated views.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()), compress=True)
    print(f"[build_pbr] {len(mats)} materials -> {a.out}")


if __name__ == "__main__":
    main()
