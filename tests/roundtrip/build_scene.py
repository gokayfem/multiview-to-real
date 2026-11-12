"""Blender: tiny synthetic primitives scene for the fusion round trip (test fixture, not shipped as a stage).

    blender -b --factory-startup --disable-autoexec --python build_scene.py -- --out CASE_DIR --size 320

Writes CASE_DIR/blockout.blend (grey, one non-overlapping UVMap per object, pass_index ids, a sun),
CASE_DIR/views/cameras.json (4 calibrated cameras) and CASE_DIR/views/objects.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

OBJECTS = [("Floor", "plane", (0, 0, 0), 4.0), ("Box", "cube", (-0.8, 0.1, 0.5), 1.0),
           ("Ball", "sphere", (0.85, 0.2, 0.5), 0.5), ("Post", "cylinder", (0.0, 1.1, 0.45), 0.3),
           ("Gem", "icosphere", (0.45, -0.95, 0.3), 0.28)]  # dense mesh (20480 tris): sub-texel triangles


def add(kind, loc, size):
    if kind == "plane":
        bpy.ops.mesh.primitive_plane_add(size=size, location=loc)
    elif kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size, location=loc)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=loc, segments=32, ring_count=16)
    elif kind == "icosphere":
        bpy.ops.mesh.primitive_ico_sphere_add(radius=size, location=loc, subdivisions=6)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=size, depth=0.9, location=loc, vertices=24)
    return bpy.context.active_object


def unwrap(o):
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    o.data.uv_layers[0].name = "UVMap"


def camera_json(name, loc, target, W, f):
    loc, target = np.array(loc, float), np.array(target, float)
    fwd = target - loc
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, -fwd, loc
    w2c = np.diag([1.0, -1, -1, 1]) @ np.linalg.inv(c2w)
    K = [[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]]
    return {"name": name, "width": W, "height": W, "K": K, "w2c_opencv": w2c.tolist(), "c2w_blender": c2w.tolist(),
            "clip_start": 0.05, "clip_end": 100.0, "role": "synthetic"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    out = Path(a.out)
    (out / "views").mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    grey = bpy.data.materials.new("clay")
    rows = []
    for k, (name, kind, loc, size) in enumerate(OBJECTS, start=1):
        o = add(kind, loc, size)
        o.name = o.data.name = name
        o.pass_index = 10 * k
        bpy.ops.object.shade_smooth() if kind in ("sphere", "cylinder", "icosphere") else None
        unwrap(o)
        o.data.materials.append(grey)
        rows.append({"name": name, "id": 10 * k, "description": f"synthetic {kind}"})
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    bpy.context.scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4
    bpy.context.scene.world = world
    W, f = a.size, a.size * 1.15
    target = (0.0, 0.35, 0.45)
    cams = []
    for i, az in enumerate((-55, -20, 15, 50), start=1):
        th = math.radians(az)
        loc = (4.2 * math.sin(th), -4.2 * math.cos(th), 2.3 + 0.3 * (i % 2))
        cams.append(camera_json(f"view_{i:02d}", loc, target, W, f))
    (out / "views" / "cameras.json").write_text(json.dumps({"views": cams}, indent=1))
    (out / "views" / "objects.json").write_text(json.dumps(rows, indent=1))
    bpy.ops.wm.save_as_mainfile(filepath=str((out / "blockout.blend").resolve()))
    print("[build_scene] ok", out)


if __name__ == "__main__":
    main()
