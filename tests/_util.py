"""Shared bits of the test runner: Skip, optional imports, a synthetic case with exact passes (no Blender)."""
import importlib
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


class Skip(Exception):
    pass


def need(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise Skip(f"{module} not importable ({e})")


def need_blender() -> str:
    from mvb2r import common
    b = common.find_blender()
    if not b:
        raise Skip("Blender not found (set $BLENDER): Blender-dependent test skipped")
    return b


def tmpdir(prefix: str = "mvb2r_test_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


# ---------------------------------------------------------------- an analytic scene: ground plane + two boxes
BOXES = [(2, (-0.5, -0.3, 0.0), (0.0, 0.3, 0.6)), (3, (0.2, -0.2, 0.0), (0.7, 0.4, 0.4))]


def _hit_box(o, d, lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin, tmax = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
    ok = (tmax >= tmin) & (tmin > 1e-6)
    t = np.where(ok, tmin, np.inf)
    p = o + d * t[:, None]
    n = np.zeros_like(p)
    for a in range(3):
        n[np.isclose(p[:, a], lo[a], atol=1e-6), a] = -1
        n[np.isclose(p[:, a], hi[a], atol=1e-6), a] = 1
    return t, n


def render_passes(cam):
    """Exact passes of the analytic scene for a camera dict (K, c2w_blender, w2c_opencv, width, height)."""
    from mvb2r import cameras
    o, d, (H, W) = cameras.pixel_rays(cam)
    O = np.broadcast_to(o, d.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[:, 2] < -1e-9, -o[2] / d[:, 2], np.inf)
    gp = o + d * np.where(np.isfinite(tg), tg, 0)[:, None]
    tg = np.where((np.abs(gp[:, 0]) <= 3) & (np.abs(gp[:, 1]) <= 3), tg, np.inf)     # finite ground, like its bbox
    best_t, ids, nrm = tg.copy(), np.where(np.isfinite(tg), 1, 0), np.tile([0.0, 0.0, 1.0], (len(d), 1))
    for bid, lo, hi in BOXES:
        t, n = _hit_box(O, d, lo, hi)
        closer = t < best_t
        best_t, ids, nrm = np.where(closer, t, best_t), np.where(closer, bid, ids), np.where(closer[:, None], n, nrm)
    hit = np.isfinite(best_t)
    pos = np.where(hit[:, None], o + d * np.where(hit, best_t, 0)[:, None], 0)
    w2c = np.asarray(cam["w2c_opencv"])
    z = np.where(hit, (pos @ w2c[:3, :3].T + w2c[:3, 3])[:, 2], 1e10)
    return {"oid": ids.reshape(H, W).astype(np.uint8), "pos": pos.reshape(H, W, 3).astype(np.float32),
            "nrm": np.where(hit[:, None], nrm, 0).reshape(H, W, 3).astype(np.float32),
            "depth": z.reshape(H, W).astype(np.float32)}


def make_cam(name, eye, target, W=160, H=160, lens=35.0):
    from mvb2r import cameras
    c2w = cameras.look_at_c2w(eye, target)
    K = cameras.intrinsics(W, H, lens)
    w2c = cameras.w2c_opencv(c2w)
    return {"name": name, "camera": "cam_" + name, "width": W, "height": H, "K": K.tolist(),
            "c2w_blender": c2w.tolist(), "w2c_opencv": w2c.tolist(), "P": cameras.projection(K, w2c).tolist()}


def synthetic_case(n_views: int = 3, size: int = 160, generator: str = "mock") -> Path:
    """A case folder with exact views (no Blender): case.json, views/, cameras.json, objects.json, clay."""
    import cv2
    from mvb2r import common
    root = tmpdir()
    objs = [{"name": "ground", "id": 1, "class": "stone", "category": "structure", "description": "concrete",
             "bbox_min": [-3, -3, -0.01], "bbox_max": [3, 3, 0]},
            {"name": "block_a", "id": 2, "class": "wood", "category": "object", "description": "oak block",
             "bbox_min": list(BOXES[0][1]), "bbox_max": list(BOXES[0][2]), "group": "props"},
            {"name": "block_b", "id": 3, "class": "paint", "category": "object", "description": "teal block",
             "bbox_min": list(BOXES[1][1]), "bbox_max": list(BOXES[1][2]), "group": "props"}]
    cams = []
    for i in range(n_views):
        a = -0.6 + 1.2 * i / max(1, n_views - 1)
        eye = (2.4 * np.sin(a), -2.4 * np.cos(a), 1.3)
        cams.append(make_cam(common.view_name(i + 1), eye, (0.0, 0.0, 0.25), size, size))
    for cam in cams:
        vd = root / "views" / cam["name"]
        vd.mkdir(parents=True)
        p = render_passes(cam)
        cv2.imwrite(str(vd / "object_id.png"), p["oid"])
        cv2.imwrite(str(vd / "depth_z.exr"), p["depth"])
        cv2.imwrite(str(vd / "position_world.exr"), p["pos"][..., ::-1].copy())
        cv2.imwrite(str(vd / "normal_world.exr"), p["nrm"][..., ::-1].copy())
        shade = (0.3 + 0.7 * np.clip(p["nrm"] @ np.array([0.3, -0.5, 0.8]), 0, 1)) * (p["oid"] > 0)
        clay = (np.repeat(shade[..., None], 3, -1) * 200 + 30).astype(np.uint8)
        clay[..., 0] = clay[..., 0] + np.uint8(cams.index(cam))   # distinct files
        cv2.imwrite(str(vd / "clay.png"), clay)
    common.write_json(root / "views" / "cameras.json", {"views": cams})
    common.write_json(root / "views" / "objects.json", objs)
    cfg = {"name": root.name, "input": {"path": "input/x.json", "kind": "spec", "sha256": "0"}, "look": "test",
           "references": [], "objects": [{k: o[k] for k in ("name", "id", "class", "description")} | {"group": o.get("group")}
                                         for o in objs],
           "views": {"resolution": size, "cameras": [{"name": c["name"], "camera": c["camera"], "source": "auto"} for c in cams],
                     "auto": {"count": n_views}},
           "grid": {"rows": None, "cols": None}, "model": {"provider": "fal", "edit_endpoint": "owner/model/edit",
                                                          "generator": generator},
           "budget": {"max_calls": 20}, "fusion": {"method": "auto"}}
    common.write_json(root / "case.json", cfg)
    return root
