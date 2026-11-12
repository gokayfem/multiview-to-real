"""blockout: build the grey case scene from a JSON scene spec (see mvb2r/inputs.py, templates/spec.example.json).

    blender -b --factory-startup --disable-autoexec --python scripts/blender/blockout.py -- --case CASE

Each spec object becomes ONE mesh (its primitives joined; rotation baked, origin = its location), pass_index =
its spec id. Spec cameras become camera objects (kept by `new` when there are at least two); spec lights and
world are built, otherwise neutral preview lights. Then clay.finish: clay material, `UVMap`, objects.json,
CASE/blockout.blend. Ported from Blockout 2 Real (scripts/blender/blockout.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clay as C  # noqa: E402
from mvb2r import common, inputs  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

TAG = "blockout"


def build(spec: dict, scene) -> tuple[dict, list]:
    hints = {}
    for o in spec["objects"]:
        mesh = C.mesh_from_parts(o["name"], o["parts"], o.get("rotation"))
        obj = bpy.data.objects.new(o["name"], mesh)
        if obj.name != o["name"]:
            raise SystemExit(f"object name {o['name']!r} is already taken")
        scene.collection.objects.link(obj)
        obj.location = Vector(o.get("location", (0, 0, 0)))
        obj.pass_index = int(o["id"])
        hints[o["name"]] = {"description": common.clean_text(o.get("description", ""), 300),
                            "class": common.clean_text(o.get("class", ""), 40)}
    cams = []
    for i, c in enumerate(spec.get("cameras", [spec["camera"]] if isinstance(spec.get("camera"), dict) else [])):
        lens = c.get("focal_length_mm", 35.0)
        if "fov_deg" in c and "focal_length_mm" not in c:
            import math
            lens = 36.0 / (2 * math.tan(math.radians(c["fov_deg"]) / 2))
        cam = C.add_camera(scene, c.get("name", f"spec_cam_{i + 1:02d}"), c["location"], c.get("look_at"),
                           c.get("rotation"), lens, c.get("sensor_width_mm", 36.0))
        cams.append({"name": cam.name, "role": common.clean_text(c.get("role", "spec camera"), 60)})
    for d in spec.get("lights", []):
        C.add_light(scene, d)
    if isinstance(spec.get("world"), dict):
        w = bpy.data.worlds.new("spec_world")
        w.use_nodes = True
        bg = w.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = (*spec["world"].get("color", (0.8, 0.8, 0.8))[:3], 1.0)
        bg.inputs["Strength"].default_value = float(spec["world"].get("strength", 1.0))
        scene.world = w
    return hints, cams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--group-by", default="object")
    ap.add_argument("--allow-external", action="store_true")
    args = ap.parse_args(common.script_args())
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    spec = inputs.load_spec(case / cfg["input"]["path"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    hints, cams = build(spec, scene)
    C.log(TAG, f"built {len(spec['objects'])} objects, {len(cams)} spec cameras")
    C.finish(case, scene, TAG, hints, cams)


if __name__ == "__main__":
    main()
