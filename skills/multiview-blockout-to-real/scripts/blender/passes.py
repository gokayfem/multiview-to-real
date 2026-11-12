"""passes: clay beauty + exact ray-cast passes per camera, and cameras.json (ported from the experiment).

    blender -b --factory-startup --disable-autoexec --python scripts/blender/passes.py -- --case CASE [--samples N]
    blender ... passes.py -- --case CASE --project JOB.json OUT.json     (verify: world_to_camera_view of points)

Data passes come from one mathutils BVH over every pass-indexed mesh (world space) and one ray through every
pixel centre: object_id (un-antialiased), depth_z (camera-axis Z, NOT ray length; background 1e10),
position_world, normal_world (interpolated corner normals). The beauty is a Cycles render of the clay scene.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clay as C  # noqa: E402
from mvb2r import cameras as cam_mod  # noqa: E402
from mvb2r import common  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402
import OpenImageIO as oiio  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

TAG = "passes"
CONVENTION = ("World: Blender, Z up, metres. c2w_blender = camera matrix_world (looks along -Z, +Y up). "
              "w2c_opencv = diag(1,-1,-1,1) @ inv(c2w_blender): +X right, +Y down, +Z forward. u = fx*Xc/Zc + cx, "
              "v = fy*Yc/Zc + cy, origin at the top-left corner of the top-left pixel, pixel centres at +0.5, row 0 = "
              "top row of the file. P = K @ w2c_opencv[:3]. depth_z.exr = Zc in metres (camera-axis distance, NOT ray "
              "length), background 1e10, single channel 'Y'. position_world.exr / normal_world.exr = world XYZ / unit "
              "normal of the first hit (0 on background). object_id.png = pass_index (0 = nothing). Data maps are exact "
              "single rays through pixel centres (mathutils BVH), no anti-aliasing.")


def write_img(path: Path, arr: np.ndarray, fmt) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    spec = oiio.ImageSpec(w, h, c, fmt)
    if str(path).endswith(".exr"):
        spec.attribute("compression", "zip")
        spec.channelnames = ("Y",) if c == 1 else ("R", "G", "B")
    out = oiio.ImageOutput.create(str(path))
    if out is None or not out.open(str(path), spec) or not out.write_image(arr):
        raise RuntimeError(f"cannot write {path}: {oiio.geterror()}")
    out.close()


def camera_entry(scene, cam, view: dict) -> dict:
    W, H = scene.render.resolution_x, scene.render.resolution_y
    cd = cam.data
    K = cam_mod.intrinsics(W, H, cd.lens, cd.sensor_width, cd.sensor_height, cd.sensor_fit, cd.shift_x, cd.shift_y)
    c2w = np.array(cam.matrix_world, dtype=np.float64)
    w2c = cam_mod.w2c_opencv(c2w)
    return {"name": view["name"], "camera": cam.name, "role": view.get("role", ""), "source": view.get("source", ""),
            "width": W, "height": H, "pixel_aspect": 1.0, "lens_mm": cd.lens, "sensor_width_mm": cd.sensor_width,
            "sensor_height_mm": cd.sensor_height, "sensor_fit": cd.sensor_fit, "shift_x": cd.shift_x,
            "shift_y": cd.shift_y, "clip_start": cd.clip_start, "clip_end": cd.clip_end,
            "fov_deg": math.degrees(2 * math.atan(max(W, H) / 2 / K[0, 0])), "location": c2w[:3, 3].tolist(),
            "K": K.tolist(), "c2w_blender": c2w.tolist(), "w2c_opencv": w2c.tolist(),
            "P": cam_mod.projection(K, w2c).tolist()}


class SceneRays:
    """One BVH over all pass-indexed meshes (world-space triangles) with per-corner normals."""

    def __init__(self, scene) -> None:
        verts, tris, cns, owner, off = [], [], [], [], 0
        for ob in scene.objects:
            if ob.type != "MESH" or ob.pass_index < 1 or ob.hide_render:
                continue
            me = ob.data
            me.calc_loop_triangles()
            M = np.array(ob.matrix_world)
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            n = len(me.loop_triangles)
            tv, tl = np.empty(n * 3, np.int64), np.empty(n * 3, np.int64)
            me.loop_triangles.foreach_get("vertices", tv)
            me.loop_triangles.foreach_get("loops", tl)
            cn = np.empty(len(me.corner_normals) * 3)
            me.corner_normals.foreach_get("vector", cn)
            cn = cn.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])
            verts.append(co)
            tris.append(tv.reshape(-1, 3) + off)
            cns.append(cn[tl].reshape(n, 3, 3))
            owner.append(np.full(n, ob.pass_index, np.int32))
            off += len(co)
        self.V, self.T = np.concatenate(verts), np.concatenate(tris)
        self.CN, self.TO = np.concatenate(cns), np.concatenate(owner)
        self.bvh = BVHTree.FromPolygons(self.V.tolist(), self.T.tolist(), all_triangles=True)

    def cast(self, origin, dirs):
        o, cast = Vector(origin), self.bvh.ray_cast
        pos, tri = np.zeros((len(dirs), 3)), np.full(len(dirs), -1, np.int64)
        for i, d in enumerate(dirs.tolist()):
            loc, _, idx, _ = cast(o, Vector(d))
            if idx is not None:
                pos[i], tri[i] = loc, idx
        return pos, tri

    def normal(self, pos, tri):
        A, B, Cc = (self.V[self.T[tri, k]] for k in range(3))
        v0, v1, v2 = B - A, Cc - A, pos - A
        d00, d01, d11 = (np.einsum("ij,ij->i", x, y) for x, y in ((v0, v0), (v0, v1), (v1, v1)))
        d20, d21 = np.einsum("ij,ij->i", v2, v0), np.einsum("ij,ij->i", v2, v1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
        b1, b2 = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        cn = self.CN[tri]
        n = (1 - b1 - b2)[:, None] * cn[:, 0] + b1[:, None] * cn[:, 1] + b2[:, None] * cn[:, 2]
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def gbuffer(rays: SceneRays, cam: dict) -> dict:
    origin, dirs, (H, W) = cam_mod.pixel_rays(cam)
    pos, tri = rays.cast(origin, dirs)
    hit = tri >= 0
    nrm = np.zeros_like(pos)
    nrm[hit] = rays.normal(pos[hit], tri[hit])
    ids = np.zeros(len(tri), np.int32)
    ids[hit] = rays.TO[tri[hit]]
    w2c = np.array(cam["w2c_opencv"])
    pc = pos @ w2c[:3, :3].T + w2c[:3, 3]
    return {"position": np.where(hit[:, None], pos, 0).reshape(H, W, 3), "normal": nrm.reshape(H, W, 3),
            "id": ids.reshape(H, W), "depth": np.where(hit, pc[:, 2], common.BG_DEPTH).reshape(H, W)}


def write_view(vd: Path, g: dict) -> None:
    vd.mkdir(parents=True, exist_ok=True)
    write_img(vd / "depth_z.exr", g["depth"].astype(np.float32), oiio.FLOAT)
    write_img(vd / "position_world.exr", g["position"].astype(np.float32), oiio.FLOAT)
    write_img(vd / "normal_world.exr", g["normal"].astype(np.float32), oiio.FLOAT)
    big = g["id"].max() > 255
    write_img(vd / "object_id.png", g["id"].astype(np.uint16 if big else np.uint8), oiio.UINT16 if big else oiio.UINT8)


def render_clay(scene, path: Path) -> float:
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t0


def open_case_blend(case: Path, cfg: dict):
    bpy.ops.wm.open_mainfile(filepath=str(case / "blockout.blend"), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    res = cfg["views"]["resolution"]
    W, H = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    return scene


def project_points(case: Path, cfg: dict, job_path: str, out_path: str) -> None:
    from bpy_extras.object_utils import world_to_camera_view
    job = json.loads(Path(job_path).read_text())
    scene = open_case_blend(case, cfg)
    W, H = scene.render.resolution_x, scene.render.resolution_y
    res = {}
    for name, pts in job["points"].items():
        cam = bpy.data.objects[name]
        res[name] = [[n.x * W, (1.0 - n.y) * H, n.z] for n in (world_to_camera_view(scene, cam, Vector(p)) for p in pts)]
    Path(out_path).write_text(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--project", nargs=2, default=None)
    args = ap.parse_args(common.script_args())
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    if args.project:
        project_points(case, cfg, *args.project)
        return
    scene = open_case_blend(case, cfg)
    C.neutral_world_if_missing(scene, TAG)
    C.setup_cycles(scene, args.samples)
    for o in scene.objects:
        if o.type == "CAMERA":
            o.data.dof.use_dof = False
    views = cfg["views"]["cameras"]
    entries = []
    for v in views:
        cam = bpy.data.objects.get(v["camera"])
        if cam is None or cam.type != "CAMERA":
            raise SystemExit(f"case.json camera {v['camera']!r} is not a camera in blockout.blend")
        entries.append((cam, camera_entry(scene, cam, v)))
    vdir = case / "views"
    common.write_json(vdir / "cameras.json", {"convention": CONVENTION, "views": [e for _, e in entries]})
    rays = SceneRays(scene)
    for cam, e in entries:
        t0 = time.time()
        g = gbuffer(rays, e)
        write_view(vdir / e["name"], g)
        scene.camera = cam
        tb = render_clay(scene, vdir / e["name"] / "clay.png")
        C.log(TAG, f"view {e['name']} ({cam.name}): rays {time.time() - t0 - tb:.1f}s, clay {tb:.1f}s, "
                   f"{int((g['id'] > 0).sum())} object px")


if __name__ == "__main__":
    main()
