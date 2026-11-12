"""Blender: render a .blend from calibrated cameras (cameras.json format), beauty and/or data passes.

    blender -b --factory-startup --disable-autoexec --python render.py -- --blend F.blend --cameras cams.json --out DIR \
        [--scale 1.0] [--samples 32] [--only a,b] [--passes] [--no-beauty] [--switch K] [--max-bounces N]
        [--sun-rotate-z DEG] [--sun-elevation DEG] [--sun-color r,g,b] [--sun-scale S] [--denoise]

Beauty: DIR/<name>.png, 8-bit sRGB, Standard view transform, exposure from the file (0 if --zero-exposure).
--passes: DIR/<name>/{object_id.png, uv.exr (U,V of the active render UV map), depth_z.exr (camera-axis Z,
1e10 background)} at 1 sample with a 0.01 px box filter, i.e. exact per-pixel lookups for the texture renderer.
--switch K sets the diagnostic blend's AppearanceSwitch METHOD value. Sun options relight the scene.
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fusion_bpy import (args_after_dashes, load_views, make_camera, open_blend, parser, pick,  # noqa: E402
                         read_multilayer, set_format, setup_cycles, write_exr, write_png_ids)


def relight(scene, a):
    from mathutils import Vector
    for o in scene.objects:
        if o.type != "LIGHT" or o.data.type != "SUN":
            continue
        if a.sun_rotate_z or a.sun_elevation is not None:
            d = o.matrix_world.to_3x3() @ Vector((0, 0, -1))
            az = math.atan2(d.y, d.x) + math.radians(a.sun_rotate_z)
            el = math.asin(max(-1, min(1, -d.z))) if a.sun_elevation is None else math.radians(a.sun_elevation)
            nd = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)))
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = (-nd).to_track_quat("Z", "Y")
        if a.sun_color:
            o.data.color = [float(x) for x in a.sun_color.split(",")][:3]
        o.data.energy *= a.sun_scale


def pick_any(ch, passes, comps):
    for p in passes:
        try:
            return pick(ch, p, comps)
        except KeyError:
            continue
    raise KeyError(f"none of {passes}.{comps} in {sorted(ch)}")


def render_passes(scene, v, out, W, H):
    vl = bpy.context.view_layer
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    for at in ("use_pass_z", "use_pass_uv", "use_pass_object_index", "use_pass_position"):
        setattr(vl, at, True)
    set_format(scene.render.image_settings, "OPEN_EXR_MULTILAYER", "32", "RGBA")
    tmp = out / f"_{v['name']}_ml.exr"
    scene.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    ch = read_multilayer(tmp)
    tmp.unlink()
    ids = np.rint(pick_any(ch, ("Object Index", "IndexOB"), "X")).astype(np.int64)
    uv = pick_any(ch, ("UV",), "UV")
    pos = pick_any(ch, ("Position",), "XYZ")
    w2c = np.array(v["w2c_opencv"], float)
    zc = pos.astype(np.float64) @ w2c[2, :3] + w2c[2, 3]
    valid = ids > 0
    vd = out / v["name"]
    vd.mkdir(parents=True, exist_ok=True)
    write_png_ids(vd / "object_id.png", np.where(valid, ids, 0))
    uv3 = np.dstack([uv[..., 0], uv[..., 1], np.zeros_like(uv[..., 0])])
    write_exr(vd / "uv.exr", np.where(valid[..., None], uv3, -1.0), ["R", "G", "B"])
    write_exr(vd / "depth_z.exr", np.where(valid, zc, 1e10).astype(np.float32), ["Y"])
    for at in ("use_pass_z", "use_pass_uv", "use_pass_object_index", "use_pass_position"):
        setattr(vl, at, False)


def render_beauty(scene, v, out, a):
    scene.cycles.samples = a.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = bool(a.denoise)
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    scene.cycles.seed = 0
    set_format(scene.render.image_settings, "PNG", "8", "RGB")
    scene.render.filepath = str(out / f"{v['name']}.png")
    bpy.ops.render.render(write_still=True)


def main():
    ap = parser(__doc__)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--only", default="")
    ap.add_argument("--passes", action="store_true")
    ap.add_argument("--no-beauty", action="store_true")
    ap.add_argument("--switch", type=float, default=None)
    ap.add_argument("--zero-exposure", action="store_true")
    ap.add_argument("--max-bounces", type=int, default=None)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--sun-rotate-z", type=float, default=0.0)
    ap.add_argument("--sun-elevation", type=float, default=None)
    ap.add_argument("--sun-color", default=None)
    ap.add_argument("--sun-scale", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(args_after_dashes())
    open_blend(a.blend)
    views = load_views(a.cameras)
    if a.only:
        views = [v for v in views if v["name"] in set(a.only.split(","))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    setup_cycles(scene, a.samples, gpu=not a.cpu)
    scene.render.film_transparent = False
    if a.zero_exposure:
        scene.view_settings.exposure = 0.0
    if a.max_bounces is not None:
        scene.cycles.max_bounces = a.max_bounces
    relight(scene, a)
    if a.switch is not None:
        g = bpy.data.node_groups.get("AppearanceSwitch")
        if g is None:
            raise SystemExit("--switch needs a diagnostic blend built by build_diagnostic.py")
        g.nodes["METHOD"].outputs[0].default_value = a.switch
    log = []
    for v in views:
        t0 = time.time()
        cam, W, H = make_camera(v, scale=a.scale)
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = W, H
        if not a.no_beauty:
            render_beauty(scene, v, out, a)
        if a.passes:
            render_passes(scene, v, out, W, H)
        bpy.data.objects.remove(cam)
        log.append({"name": v["name"], "size": [W, H], "s": round(time.time() - t0, 2)})
    (out / "render_log.json").write_text(json.dumps({"blend": a.blend, "device": scene.cycles.device,
                                                     "samples": a.samples, "log": log}, indent=1))
    print(f"[render] {len(log)} cameras on {scene.cycles.device}")


if __name__ == "__main__":
    main()
