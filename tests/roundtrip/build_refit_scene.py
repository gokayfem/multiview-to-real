"""Blender (test fixture): floor + a sphere. Saves truth.blend (sphere r=0.4) and blockout.blend (the same
sphere scaled x1.5: an oversized shell to be carved), views/cameras.json (4 cameras), views/objects.json.

    blender -b --factory-startup --disable-autoexec --python build_refit_scene.py -- --out CASE --size 256
"""
import json
import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scene import camera_json, unwrap  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    out, size = Path(a[a.index("--out") + 1]), int(a[a.index("--size") + 1])
    (out / "views").mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name, floor.pass_index = "Floor", 10
    unwrap(floor)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, 0.6), segments=32, ring_count=16)
    blob = bpy.context.active_object
    blob.name, blob.pass_index = "Blob", 20
    unwrap(blob)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str((out / "truth.blend").resolve()))
    blob.scale = (1.5, 1.5, 1.5)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.wm.save_as_mainfile(filepath=str((out / "blockout.blend").resolve()))
    cams = []
    for i, az in enumerate((0, 90, 180, 270), start=1):
        th = math.radians(az + 20)
        cams.append(camera_json(f"view_{i:02d}", (3.5 * math.sin(th), -3.5 * math.cos(th), 1.8), (0, 0, 0.6),
                                size, size * 1.1))
    (out / "views" / "cameras.json").write_text(json.dumps({"views": cams}, indent=1))
    (out / "views" / "objects.json").write_text(json.dumps([{"name": "Floor", "id": 10}, {"name": "Blob", "id": 20}]))
    print("[build_refit_scene] ok")


if __name__ == "__main__":
    main()
