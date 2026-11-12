"""Numeric proof that views/ follows the conventions (run by `views`; `mvb2r.py views CASE --verify-only`).

Per view, on >= N_SAMPLES random object pixels:
  (a) reprojection: position_world projected with P lands on the pixel centre (px error, must be < 0.05)
  (b) back-projection: the pixel centre at depth_z lands on position_world (mm error)
  (c) depth_z == camera-axis Zc on every object pixel (must be < 1e-4 m), and the gap ray length - depth_z
      (> 0 off-axis: proves depth is not ray length)
  (d) the same world points through Blender's bpy_extras.object_utils.world_to_camera_view (Blender run via
      passes.py --project) agree with P (px, must be < 0.01)
  (f) clay.png quality: fails if > 5 % is the missing-texture magenta or its noise sigma (smooth areas) > 4 levels
  (e) landmarks (views/landmarks.json, optional) projected, occlusion-tested, drawn on clay -> verify/overlay_view_XX.png
Writes views/verify/report.json; exit code 1 when any check fails.
"""
from __future__ import annotations

import numpy as np

from . import cameras as cam_mod
from . import common

N_SAMPLES, SEED = 300, 12345
TOL_REPROJ_PX, TOL_DEPTH_M, TOL_BLENDER_PX = 0.05, 1e-4, 0.01
MAX_MISSING_COLOUR_FRACTION = 0.05      # Blender's missing-texture magenta
MAX_CLAY_NOISE = 4.0                    # robust noise sigma (8-bit levels) in smooth areas


def clay_quality(rgb: np.ndarray) -> dict:
    """Missing-texture colour fraction and robust noise level of a clay render (pure)."""
    import cv2
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    magenta = float(((r > 180) & (b > 180) & (g < 90)).mean())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    grad = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    flat = grad < np.percentile(grad, 60)
    sigma = float(1.4826 * np.median(np.abs(resid[flat]))) if flat.any() else 0.0
    std = float(resid[flat].std()) if flat.any() else 0.0
    return {"missing_colour_fraction": round(magenta, 4), "noise_sigma": round(max(sigma, 0.5 * std), 3)}


def clay_failures(name: str, q: dict) -> list[str]:
    out = []
    if q["missing_colour_fraction"] > MAX_MISSING_COLOUR_FRACTION:
        out.append(f"{name}: clay.png is {q['missing_colour_fraction']:.0%} missing-texture magenta (an image did not load)")
    if q["noise_sigma"] > MAX_CLAY_NOISE:
        out.append(f"{name}: clay.png noise {q['noise_sigma']:.1f} > {MAX_CLAY_NOISE} (raise --samples; it conditions the model)")
    return out


def view_checks(cam: dict, pos: np.ndarray, dep: np.ndarray, oid: np.ndarray, rng) -> tuple[dict, dict]:
    """Checks (a)-(c) of one view (pure); returns (report, samples for the Blender cross-check)."""
    P, K, w2c = (np.asarray(cam[k], float) for k in ("P", "K", "w2c_opencv"))
    cc = np.asarray(cam["c2w_blender"], float)[:3, 3]
    jj, ii = np.nonzero(oid > 0)
    if len(jj) == 0:
        raise common.StageError(f"{cam['name']}: no object is visible")
    pick = rng.choice(len(jj), size=min(N_SAMPLES, len(jj)), replace=False)
    j, i = jj[pick], ii[pick]
    X = pos[j, i].astype(np.float64)
    uvz = cam_mod.project(P, X)
    err = np.hypot(uvz[:, 0] - (i + 0.5), uvz[:, 1] - (j + 0.5))
    ray = np.c_[(i + 0.5 - K[0, 2]) / K[0, 0], (j + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(i))]
    Xb = (ray * dep[j, i][:, None].astype(np.float64) - w2c[:3, 3]) @ w2c[:3, :3]
    e3 = np.linalg.norm(Xb - X, axis=1) * 1000
    valid = oid > 0
    allZ = (np.c_[pos[valid].astype(np.float64), np.ones(int(valid.sum()))] @ w2c[:3].T)[:, 2]
    dz = np.abs(dep[valid].astype(np.float64) - allZ)
    gap = np.linalg.norm(X - cc, axis=1) - dep[j, i]
    rep = {"samples": int(len(i)),
           "reproj_err_px": {"median": float(np.median(err)), "max": float(err.max())},
           "backproject_err_mm": {"median": float(np.median(e3)), "max": float(e3.max())},
           "depth_z_minus_Zc_m": {"median": float(np.median(dz)), "max": float(dz.max())},
           "ray_length_minus_depth_z_m": {"median": float(np.median(gap)), "min": float(gap.min()), "max": float(gap.max())},
           "background_pixels": int((~valid).sum())}
    return rep, {"X": X.tolist(), "uv": uvz[:, :2].tolist()}


def failures_of(name: str, rep: dict) -> list[str]:
    out = []
    if rep["reproj_err_px"]["max"] > TOL_REPROJ_PX:
        out.append(f"{name}: reprojection max {rep['reproj_err_px']['max']:.4f} px > {TOL_REPROJ_PX}")
    if rep["depth_z_minus_Zc_m"]["max"] > TOL_DEPTH_M:
        out.append(f"{name}: depth_z != Zc (max {rep['depth_z_minus_Zc_m']['max']:.2e} m)")
    b = rep.get("blender_world_to_camera_view_vs_P_px")
    if b is None:
        out.append(f"{name}: Blender cross-check missing")
    elif b["max"] > TOL_BLENDER_PX:
        out.append(f"{name}: P disagrees with world_to_camera_view by {b['max']:.4f} px")
    return out


def blender_crosscheck(case, cams: list[dict], samples: dict) -> dict:
    """Run passes.py --project on the sampled world points; returns {camera object: [[u, v, z], ...]}."""
    vdir = common.views_dir(case) / "verify"
    job, out = vdir / "_project_in.json", vdir / "_project_out.json"
    common.write_json(job, {"blend": str(common.case_dir(case) / "blockout.blend"),
                            "points": {c["camera"]: samples[c["name"]]["X"] for c in cams},
                            "resolution": [int(cams[0]["width"]), int(cams[0]["height"])]})
    rc = common.run_blender("passes.py", ["--case", str(case), "--project", str(job), str(out)],
                            common.case_dir(case) / "logs" / "verify_blender.log")
    if rc != 0 or not out.is_file():
        raise common.StageError(f"Blender cross-check failed (logs/verify_blender.log):\n"
                                f"{common.log_tail(common.case_dir(case) / 'logs' / 'verify_blender.log')}")
    res = common.read_json(out)
    job.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return res


def overlay_landmarks(case, cam: dict, lms: dict, dep: np.ndarray) -> dict:
    from . import qa
    proj = qa.project_landmarks(lms, cam, dep)
    img = common.read_rgb(common.views_dir(case) / cam["name"] / "clay.png")
    import cv2
    for name, p in proj.items():
        if 0 <= p["u"] < img.shape[1] and 0 <= p["v"] < img.shape[0]:
            col = (0, 200, 0) if p["visible"] else (255, 0, 0)
            cv2.drawMarker(img, (int(p["u"]), int(p["v"])), col, cv2.MARKER_CROSS, 14, 2)
    common.write_rgb(common.views_dir(case) / "verify" / f"overlay_{cam['name']}.png", img)
    return proj


def run_verify(case) -> int:
    from . import qa
    case = common.case_dir(case)
    cams = common.load_cameras(case)
    rng = np.random.default_rng(SEED)
    report, samples = {"views": {}}, {}
    lms = qa.load_landmarks(case)
    for cam in cams:
        vp = common.view_passes(case, cam)
        rep, samples[cam["name"]] = view_checks(cam, vp["pos"], vp["depth"], vp["oid"], rng)
        rep["clay"] = clay_quality(common.read_rgb(common.views_dir(case) / cam["name"] / "clay.png"))
        if lms:
            rep["landmarks"] = overlay_landmarks(case, cam, lms, vp["depth"])
        report["views"][cam["name"]] = rep
    bl = blender_crosscheck(case, cams, samples)
    for cam in cams:
        b = np.asarray(bl[cam["camera"]], float)
        o = np.asarray(samples[cam["name"]]["uv"], float)
        d = np.hypot(b[:, 0] - o[:, 0], b[:, 1] - o[:, 1])
        report["views"][cam["name"]]["blender_world_to_camera_view_vs_P_px"] = {"median": float(np.median(d)), "max": float(d.max())}
    fails = [f for n, r in report["views"].items() for f in failures_of(n, r) + clay_failures(n, r["clay"])]
    vs = report["views"].values()
    report["summary"] = {"max_reproj_err_px": max(v["reproj_err_px"]["max"] for v in vs),
                         "max_backproject_err_mm": max(v["backproject_err_mm"]["max"] for v in vs),
                         "max_depth_z_minus_Zc_m": max(v["depth_z_minus_Zc_m"]["max"] for v in vs),
                         "max_ray_length_minus_depth_z_m": max(v["ray_length_minus_depth_z_m"]["max"] for v in vs),
                         "max_blender_vs_P_px": max(v["blender_world_to_camera_view_vs_P_px"]["max"] for v in vs),
                         "max_clay_noise": max(v["clay"]["noise_sigma"] for v in vs),
                         "max_clay_missing_colour": max(v["clay"]["missing_colour_fraction"] for v in vs)}
    report["failures"], report["ok"] = fails, not fails
    common.write_json(common.views_dir(case) / "verify" / "report.json", report)
    s = report["summary"]
    common.say(f"[verify] reprojection max {s['max_reproj_err_px']:.2e} px; back-projection max {s['max_backproject_err_mm']:.3f} mm; "
               f"depth_z-Zc max {s['max_depth_z_minus_Zc_m']:.2e} m; ray length - depth_z up to {s['max_ray_length_minus_depth_z_m']:.4f} m; "
               f"Blender world_to_camera_view vs P max {s['max_blender_vs_P_px']:.2e} px; clay noise max "
               f"{s['max_clay_noise']:.2f}, missing-texture colour max {s['max_clay_missing_colour']:.1%}")
    common.say("[verify] all checks passed" if not fails else "[verify] FAILURES:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1
