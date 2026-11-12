"""Shared helpers for the fusion half's Blender scripts (run inside Blender's Python)."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def args_after_dashes(argv=None):
    argv = list(sys.argv if argv is None else argv)
    return argv[argv.index("--") + 1:] if "--" in argv else []


def parser(doc):
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--blend", required=True, help=".blend to open (scripts in it are never run)")
    return ap


def open_blend(path):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"missing .blend {p}")
    bpy.ops.wm.open_mainfile(filepath=str(p.resolve()), load_ui=False, use_scripts=False)


def load_views(path):
    d = json.loads(Path(path).read_text())
    views = d.get("views", d.get("frames")) if isinstance(d, dict) else d
    if not isinstance(views, list):
        raise SystemExit(f"{path}: no 'views' list")
    return views


def make_camera(v, name=None, scale=1.0):
    """Blender camera reproducing K / c2w_blender exactly (horizontal sensor fit, shifts from cx, cy)."""
    K = np.array(v["K"], float)
    W, H = int(v["width"]), int(v["height"])
    if abs(K[0, 0] - K[1, 1]) > 1e-6 * K[0, 0]:
        raise SystemExit(f"{v['name']}: fx != fy (non-square pixels unsupported)")
    data = bpy.data.cameras.new(name or f"cam_{v['name']}")
    data.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    data.sensor_width = data.sensor_height = 36.0
    data.lens = K[0, 0] * 36.0 / max(W, H)
    data.shift_x = (W / 2 - K[0, 2]) / max(W, H)
    data.shift_y = (K[1, 2] - H / 2) / max(W, H)
    data.clip_start = float(v.get("clip_start", 0.01))
    data.clip_end = float(v.get("clip_end", 1000.0))
    cam = bpy.data.objects.new(name or f"cam_{v['name']}", data)
    bpy.context.scene.collection.objects.link(cam)
    cam.matrix_world = Matrix(np.array(v["c2w_blender"], float).tolist())
    return cam, int(round(W * scale)), int(round(H * scale))


def set_format(settings, fmt, depth, mode):
    if hasattr(settings, "media_type"):
        settings.media_type = "MULTI_LAYER_IMAGE" if fmt == "OPEN_EXR_MULTILAYER" else "IMAGE"
    settings.file_format = fmt
    settings.color_mode = mode
    settings.color_depth = depth
    if fmt.startswith("OPEN_EXR"):
        settings.exr_codec = "ZIP"


def setup_cycles(scene, samples, gpu=True):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    if gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
                try:
                    prefs.compute_device_type = backend
                except TypeError:
                    continue
                prefs.get_devices()
                if any(d.type == backend for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == backend
                    scene.cycles.device = "GPU"
                    break
        except Exception as e:  # noqa: BLE001 - CPU fallback is always valid
            print("[mvb2r] GPU setup failed, CPU:", e)
    scene.cycles.samples = samples
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_percentage = 100
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.gamma = 1.0


def read_multilayer(path):
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot open {path}")
    ch, k = {}, 0
    while inp.seek_subimage(k, 0):
        spec = inp.spec()
        px = np.asarray(inp.read_image(k, 0, 0, spec.nchannels, oiio.FLOAT)).reshape(
            spec.height, spec.width, spec.nchannels)
        for i, n in enumerate(spec.channelnames):
            ch[n] = px[:, :, i]
        k += 1
    inp.close()
    return ch


def pick(ch, pas, comps):
    out = []
    for c in comps:
        keys = [k for k in ch if k.endswith(f"{pas}.{c}")]
        if len(keys) != 1:
            raise KeyError(f"{pas}.{c} not unique in {sorted(ch)}")
        out.append(ch[keys[0]])
    return np.stack(out, -1) if len(out) > 1 else out[0]


def write_exr(path, arr, names):
    import OpenImageIO as oiio
    arr = np.ascontiguousarray(arr, np.float32)
    arr = arr[..., None] if arr.ndim == 2 else arr
    h, w, c = arr.shape
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.channelnames = tuple(names)
    spec.attribute("compression", "zip")
    o = oiio.ImageOutput.create(str(path))
    if o is None or not o.open(str(path), spec) or not o.write_image(arr):
        raise RuntimeError(f"write {path}: {oiio.geterror()}")
    o.close()


def write_png_ids(path, ids):
    import OpenImageIO as oiio
    big = ids.max() >= 256
    arr = ids.astype(np.uint16 if big else np.uint8)[..., None]
    spec = oiio.ImageSpec(ids.shape[1], ids.shape[0], 1, oiio.UINT16 if big else oiio.UINT8)
    o = oiio.ImageOutput.create(str(path))
    o.open(str(path), spec)
    o.write_image(np.ascontiguousarray(arr))
    o.close()


def safe(name):
    keep = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in name.replace(" ", "_")).lstrip(".")
    if not keep or ".." in keep:
        raise SystemExit(f"unsafe name {name!r}")
    return keep[:80]


def mesh_objects(scene):
    return [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
