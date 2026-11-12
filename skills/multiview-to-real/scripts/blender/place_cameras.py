"""place_cameras: keep or place the case's cameras, per-object coverage from exact ray casts, preview sheet.

    blender -b --factory-startup --disable-autoexec --python scripts/blender/place_cameras.py -- --case CASE [--replace]

Cameras: case.json views.cameras when it lists >= 2 cameras that exist in blockout.blend (and --replace is not
given); otherwise mvb2r.cameras.plan_auto_cameras(views/objects.json, views.auto) -> camera objects
mvb2r_view_XX (previous automatic cameras removed), case.json updated, blockout.blend saved.
Coverage (views/coverage.json): per object, area-weighted surface samples; a sample is seen by a camera when it
projects inside the frame, faces it (cos > 0.05) and the ray from the camera reaches it unoccluded (BVH).
Fractions seen by >= 1 and >= 2 views, and per view. Preview: views/preview_sheet.png (Workbench, small).
"""
from __future__ import annotations

import argparse
import sys
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

TAG = "place_cameras"
SAMPLES_PER_OBJECT = 3000
PREVIEW_PX = 192


def existing(cfg: dict) -> list[dict]:
    cams = (cfg.get("views") or {}).get("cameras") or []
    ok = [c for c in cams if bpy.data.objects.get(c.get("camera", "")) is not None]
    return ok if len(ok) >= common.MIN_VIEWS and len(ok) == len(cams) else []


def place_auto(scene, cfg: dict, objects: list[dict]) -> list[dict]:
    for o in [o for o in scene.objects if o.type == "CAMERA" and o.name.startswith("mvb2r_view_")]:
        bpy.data.objects.remove(o, do_unlink=True)
    plan = cam_mod.plan_auto_cameras(objects, (cfg.get("views") or {}).get("auto") or {})
    out = []
    for p in plan:
        cam = C.add_camera(scene, f"mvb2r_{p['name']}", p["location"], p["look_at"], None, p["lens_mm"], p["sensor_width_mm"])
        if p["pulled_in_by_room"]:
            C.note(TAG, f"{p['name']} kept inside the room: lens widened to {p['lens_mm']:.1f} mm")
        out.append({"name": p["name"], "camera": cam.name, "role": p["role"], "source": "auto"})
    return out


def surface_samples(obj, rng) -> tuple[np.ndarray, np.ndarray]:
    me = obj.data
    me.calc_loop_triangles()
    M = np.array(obj.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    tv = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tv)
    A, B, Cc = (co[tv.reshape(-1, 3)[:, k]] for k in range(3))
    cr = np.cross(B - A, Cc - A)
    area = np.linalg.norm(cr, axis=1) / 2
    if area.sum() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    k = rng.choice(len(area), size=SAMPLES_PER_OBJECT, p=area / area.sum())
    r1, r2 = rng.random(len(k)), rng.random(len(k))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * A[k] + (s * (1 - r2))[:, None] * B[k] + (s * r2)[:, None] * Cc[k]
    n = cr[k] / np.maximum(np.linalg.norm(cr[k], axis=1, keepdims=True), 1e-12)
    return pts, n


def scene_bvh(scene):
    verts, tris, off = [], [], 0
    for o in C.mesh_objects(scene):
        me = o.data
        me.calc_loop_triangles()
        M = np.array(o.matrix_world)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        verts.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        tv = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tv)
        tris.append(tv.reshape(-1, 3) + off)
        off += len(verts[-1])
    return BVHTree.FromPolygons(np.concatenate(verts).tolist(), np.concatenate(tris).tolist(), all_triangles=True)


def seen_by(cam_obj, scene, pts, nrm, bvh, W, H) -> np.ndarray:
    cd = cam_obj.data
    K = cam_mod.intrinsics(W, H, cd.lens, cd.sensor_width, cd.sensor_height, cd.sensor_fit, cd.shift_x, cd.shift_y)
    c2w = np.array(cam_obj.matrix_world)
    uvz = cam_mod.project(cam_mod.projection(K, cam_mod.w2c_opencv(c2w)), pts)
    o = c2w[:3, 3]
    vd = o - pts
    dist = np.linalg.norm(vd, axis=1)
    ok = (uvz[:, 2] > 0) & (uvz[:, 0] >= 0) & (uvz[:, 0] < W) & (uvz[:, 1] >= 0) & (uvz[:, 1] < H)
    ok &= np.einsum("ij,ij->i", nrm, vd) / np.maximum(dist, 1e-9) > 0.05
    out = np.zeros(len(pts), bool)
    ov = Vector(o)
    for i in np.nonzero(ok)[0]:
        hit = bvh.ray_cast(ov, Vector((pts[i] - o) / dist[i]), dist[i] + 1e-3)
        out[i] = hit[0] is not None and abs(hit[3] - dist[i]) < max(2e-3, 1e-3 * dist[i])
    return out


def coverage(scene, views: list[dict], W: int, H: int) -> dict:
    rng = np.random.default_rng(0)
    bvh = scene_bvh(scene)
    res = {}
    for obj in C.mesh_objects(scene):
        pts, nrm = surface_samples(obj, rng)
        if not len(pts):
            continue
        seen = np.stack([seen_by(bpy.data.objects[v["camera"]], scene, pts, nrm, bvh, W, H) for v in views], 1)
        cnt = seen.sum(1)
        res[obj.name] = {"id": int(obj.pass_index), "seen_ge1": round(float((cnt >= 1).mean()), 4),
                         "seen_ge2": round(float((cnt >= 2).mean()), 4),
                         "per_view": {v["name"]: round(float(seen[:, i].mean()), 4) for i, v in enumerate(views)}}
    return {"method": "area-weighted surface samples, in frame, cos > 0.05, unoccluded ray cast",
            "samples_per_object": SAMPLES_PER_OBJECT, "resolution": [W, H], "objects": res}


def preview_sheet(scene, views: list[dict], path: Path, W: int, H: int) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = PREVIEW_PX, max(1, int(PREVIEW_PX * H / W))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    tiles = []
    tmp = path.parent / "_preview_tmp.png"
    for v in views:
        scene.camera = bpy.data.objects[v["camera"]]
        scene.render.filepath = str(tmp)
        bpy.ops.render.render(write_still=True)
        img = oiio.ImageInput.open(str(tmp))
        px = img.read_image(oiio.UINT8)
        img.close()
        tiles.append(px[..., :3])
    tmp.unlink(missing_ok=True)
    cols = min(len(tiles), 3)
    rows = -(-len(tiles) // cols)
    th, tw = tiles[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    out = oiio.ImageOutput.create(str(path))
    out.open(str(path), oiio.ImageSpec(sheet.shape[1], sheet.shape[0], 3, oiio.UINT8))
    out.write_image(sheet)
    out.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args(common.script_args())
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    bpy.ops.wm.open_mainfile(filepath=str(case / "blockout.blend"), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    objects = common.read_json(case / "views" / "objects.json")
    views = [] if args.replace else existing(cfg)
    if not views:
        views = place_auto(scene, cfg, objects)
        bpy.ops.wm.save_as_mainfile(filepath=str(case / "blockout.blend"), compress=True)
    res = cfg["views"]["resolution"]
    W, H = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))
    cfg = {**cfg, "views": {**cfg["views"], "cameras": views}}
    common.save_case(case, cfg)
    common.write_json(case / "views" / "coverage.json", coverage(scene, views, W, H))
    preview_sheet(scene, views, case / "views" / "preview_sheet.png", W, H)
    C.log(TAG, f"{len(views)} cameras: {', '.join(v['camera'] for v in views)}")


if __name__ == "__main__":
    main()
