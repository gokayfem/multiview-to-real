"""Blender: grey blockout + projected texture sets -> out/diagnostic.blend (emission, baked lighting, switchable).

    blender -b --factory-startup --disable-autoexec --python build_diagnostic.py -- --blend blockout.blend \
        --sets multiband=proj/multiband best=proj/best --default multiband --out out/diagnostic.blend \
        --names names.json [--cameras cams.json ...] [--readme README.txt] [--openings A,B] [--sky 0.8,0.85,0.9]

Every mesh gets an Emission material driven by the shared node group "AppearanceSwitch" (Value node METHOD:
0 = first set, 1 = second, ...); images are packed. Opening objects (windows to the sky) are NOT projected
onto: they get a flat neutral sky emission labelled "not projected", and the world is the same flat colour.
names.json maps object name -> file-safe folder name. Lights are disabled (emission only).
"""
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_bpy import args_after_dashes, load_views, make_camera, mesh_objects, open_blend, parser  # noqa: E402


def make_group(n):
    g = bpy.data.node_groups.new("AppearanceSwitch", "ShaderNodeTree")
    for i in range(n):
        g.interface.new_socket(f"Set{i}", in_out="INPUT", socket_type="NodeSocketColor")
    g.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    gi, go = g.nodes.new("NodeGroupInput"), g.nodes.new("NodeGroupOutput")
    val = g.nodes.new("ShaderNodeValue")
    val.name = val.label = "METHOD"
    prev = gi.outputs[0]
    for i in range(1, n):
        sub = g.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.use_clamp = True
        sub.inputs[1].default_value = float(i - 1)
        g.links.new(val.outputs[0], sub.inputs[0])
        mix = g.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        g.links.new(sub.outputs[0], mix.inputs["Factor"])
        g.links.new(prev, mix.inputs[6])
        g.links.new(gi.outputs[i], mix.inputs[7])
        prev = mix.outputs[2]
    g.links.new(prev, go.inputs[0])
    return g, val


def emission_material(name, label):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    fr = nt.nodes.new("NodeFrame")
    fr.label = label
    return mat, nt, em


def textured(o, folder, sets, group, uv):
    mat, nt, em = emission_material(f"{o.name}__diagnostic",
                                    "DIAGNOSTIC: projected generated views, CONTAINS BAKED LIGHTING")
    mat["contains_baked_lighting"] = True
    gn = nt.nodes.new("ShaderNodeGroup")
    gn.node_tree = group
    nt.links.new(gn.outputs[0], em.inputs["Color"])
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = uv
    missing = []
    for i, (key, d) in enumerate(sets):
        f = Path(d) / folder / "texture.png"
        if not f.is_file():
            missing.append(f"{o.name}:{key}")
            gn.inputs[i].default_value = (0.18, 0.18, 0.18, 1)
            continue
        img = bpy.data.images.load(str(f.resolve()), check_existing=False)
        img.name = f"{folder}__{key}"
        img.colorspace_settings.name = "sRGB"
        img.pack()
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.label = f"method {key}"
        ti.image = img
        ti.extension = "EXTEND"
        nt.links.new(uvn.outputs["UV"], ti.inputs["Vector"])
        nt.links.new(ti.outputs["Color"], gn.inputs[i])
    return mat, missing


def flat_sky(world, sky):
    world.use_nodes = True
    nt = world.node_tree
    bg = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground"), None) or \
        nt.nodes.new("ShaderNodeBackground")
    for link in list(bg.inputs["Color"].links):
        nt.links.remove(link)
    bg.inputs["Color"].default_value = list(sky) + [1.0]
    bg.inputs["Strength"].default_value = 1.0
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None) or \
        nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])


def main():
    ap = parser(__doc__)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--default", default=None)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uv", default="UVMap")
    ap.add_argument("--cameras", nargs="*", default=[])
    ap.add_argument("--readme", default=None)
    ap.add_argument("--openings", default="")
    ap.add_argument("--sky", default="0.8,0.84,0.9")
    a = ap.parse_args(args_after_dashes())
    open_blend(a.blend)
    sets = [s.split("=", 1) for s in a.sets]
    names = json.loads(Path(a.names).read_text())
    openings = set(filter(None, a.openings.split(",")))
    sky = [float(x) for x in a.sky.split(",")][:3]
    group, val = make_group(len(sets))
    keys = [k for k, _ in sets]
    val.outputs[0].default_value = float(keys.index(a.default)) if a.default in keys else 0.0
    missing, done = [], 0
    for o in mesh_objects(bpy.context.scene):
        if o.name in openings:
            mat, nt, em = emission_material(f"{o.name}__opening_sky", "OPENING: neutral sky, NOT projected")
            em.inputs["Color"].default_value = sky + [1.0]
            mat["not_projected"] = True
        elif o.name in names:
            mat, miss = textured(o, names[o.name], sets, group, a.uv)
            missing += miss
        else:
            continue
        o.data.materials.clear()
        o.data.materials.append(mat)
        done += 1
    scene = bpy.context.scene
    scene.world = scene.world or bpy.data.worlds.new("World")
    flat_sky(scene.world, sky)
    for ob in scene.objects:
        if ob.type == "LIGHT":
            ob.hide_render = True
    for path in a.cameras:
        for v in load_views(path):
            make_camera(v, name=v["name"])
    txt = bpy.data.texts.new("README_diagnostic")
    txt.write(Path(a.readme).read_text() if a.readme else "DIAGNOSTIC ONLY: textures contain baked lighting.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()), compress=True)
    print(f"[build_diagnostic] {done} meshes, sets {keys}, missing {missing[:10]} -> {a.out}")


if __name__ == "__main__":
    main()
