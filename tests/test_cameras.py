"""Camera maths, automatic placement inside rooms, and the pixel / depth conventions on an analytic scene."""
import math

import numpy as np

from _util import make_cam, render_passes
from mvb2r import cameras, verify


def test_target_projects_to_principal_point_and_depth_is_axis_z():
    cam = make_cam("view_01", (2, -3, 1.5), (0, 0, 0.5), 200, 150)
    uvz = cameras.project(cam["P"], [(0, 0, 0.5)])[0]
    assert abs(uvz[0] - 100) < 1e-9 and abs(uvz[1] - 75) < 1e-9
    assert abs(uvz[2] - np.linalg.norm(np.array([2, -3, 1.5]) - [0, 0, 0.5])) < 1e-9
    off = cameras.project(cam["P"], [(0.5, 0, 0.5)])[0]
    w2c = np.array(cam["w2c_opencv"])
    assert abs(off[2] - (w2c @ [0.5, 0, 0.5, 1])[2]) < 1e-12


def test_up_is_minus_v_and_right_is_plus_u():
    cam = make_cam("v", (0, -3, 0), (0, 0, 0))
    up = cameras.project(cam["P"], [(0, 0, 0.3)])[0]
    right = cameras.project(cam["P"], [(0.3, 0, 0)])[0]
    assert up[1] < cam["height"] / 2 and right[0] > cam["width"] / 2


def test_pixel_rays_go_through_pixel_centres():
    cam = make_cam("v", (1, -2, 1), (0, 0, 0), 64, 48)
    o, d, (H, W) = cameras.pixel_rays(cam)
    pts = o + d * 2.0
    uvz = cameras.project(cam["P"], pts)
    U, V = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    assert np.abs(uvz[:, 0] - U.ravel()).max() < 1e-9 and np.abs(uvz[:, 1] - V.ravel()).max() < 1e-9


def test_intrinsics_shift_and_vertical_fit():
    K = cameras.intrinsics(100, 200, 50, 36, 24, "AUTO", 0.1, -0.05)
    assert abs(K[0, 0] - 50 / 36 * 200) < 1e-9
    assert abs(K[0, 2] - (50 - 0.1 * 200)) < 1e-9 and abs(K[1, 2] - (100 - 0.05 * 200)) < 1e-9


def test_verify_checks_pass_on_exact_passes():
    cam = make_cam("view_01", (2.2, -2.0, 1.4), (0, 0, 0.3), 96, 96)
    p = render_passes(cam)
    rep, samples = verify.view_checks(cam, p["pos"], p["depth"], p["oid"], np.random.default_rng(0))
    rep["blender_world_to_camera_view_vs_P_px"] = {"median": 0.0, "max": 0.0}
    assert not verify.failures_of("view_01", rep), rep
    assert rep["ray_length_minus_depth_z_m"]["max"] > 0.01
    bad = dict(rep, depth_z_minus_Zc_m={"median": 0.1, "max": 0.2})
    assert any("depth_z != Zc" in f for f in verify.failures_of("v", bad))
    return f"reprojection max {rep['reproj_err_px']['max']:.1e} px"


def _room():
    walls = [("floor", [0, 0, -0.1], [4, 3, 0]), ("ceiling", [0, 0, 2.5], [4, 3, 2.6]),
             ("wall_w", [-0.1, 0, 0], [0, 3, 2.5]), ("wall_e", [4, 0, 0], [4.1, 3, 2.5]),
             ("wall_n", [0, 3, 0], [4, 3.1, 2.5]), ("wall_s", [0, -0.1, 0], [4, 0, 2.5])]
    objs = [{"name": n, "bbox_min": lo, "bbox_max": hi, "category": "structure"} for n, lo, hi in walls]
    return objs + [{"name": "chair", "bbox_min": [1.6, 2.0, 0], "bbox_max": [2.4, 2.8, 1.0], "category": "object"}]


def test_room_is_detected_and_cameras_stay_inside():
    objs = _room()
    room = cameras.detect_room(objs)
    assert room is not None and np.allclose(room[0], [0, 0, 0]) and np.allclose(room[1], [4, 3, 2.5])
    plan = cameras.plan_auto_cameras(objs, {"count": 6, "arc_degrees": 140, "closeup": "chair", "elevated": True})
    assert len(plan) == 6 and [p["name"] for p in plan] == [f"view_0{i}" for i in range(1, 7)]
    for p in plan:
        loc = np.array(p["location"])
        assert np.all(loc >= 0.2 - 1e-9) and np.all(loc <= np.array([4, 3, 2.5]) - 0.2 + 1e-9), p
        assert cameras.LENS_MIN <= p["lens_mm"] <= cameras.LENS_MAX
    assert any(p["pulled_in_by_room"] for p in plan)


def test_open_scene_has_no_room_and_arc_is_symmetric():
    objs = [{"name": "a", "bbox_min": [-0.5, -0.5, 0], "bbox_max": [0.5, 0.5, 1], "category": "object"}]
    assert cameras.detect_room(objs) is None
    plan = cameras.plan_auto_cameras(objs, {"count": 3, "arc_degrees": 90, "elevated": False})
    xs = [p["location"][0] for p in plan]
    assert abs(xs[0] + xs[2]) < 1e-6 and abs(xs[1]) < 1e-6
    d = np.linalg.norm(np.array(plan[1]["location"]) - [0, 0, 0.5])
    need = 1.15 * math.sqrt(3) / 2 / math.sin(cameras.fov_for_lens(35) / 2)
    assert abs(d - need) < 1e-3


def test_bad_rig_requests_fail_clearly():
    objs = [{"name": "a", "bbox_min": [0, 0, 0], "bbox_max": [1, 1, 1]}]
    for auto in ({"count": 1}, {"count": 10}, {"count": 4, "closeup": "nope"}):
        try:
            cameras.plan_auto_cameras(objs, auto)
        except ValueError:
            continue
        raise AssertionError(auto)


def test_clamp_inside_moves_along_the_line():
    room = (np.array([0, 0, 0.0]), np.array([4, 4, 3.0]))
    out = cameras.clamp_inside([10, 2, 1.5], [2, 2, 1.5], room, 0.2)
    assert np.allclose(out, [3.8, 2, 1.5])
    assert np.allclose(cameras.clamp_inside([1, 1, 1], [2, 2, 1.5], room), [1, 1, 1])


def test_coverage_table_rows():
    rows = cameras.coverage_table({"objects": {"a": {"seen_ge1": 0.5, "seen_ge2": 0.25, "per_view": {"view_01": 0.5}}}})
    assert "view_01" in rows[1] and "0.50" in rows[1]


def test_clay_quality_catches_missing_textures_and_noise():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    clean = np.dstack([(60 + xx + yy // 2).astype(np.uint8)] * 3)
    q = verify.clay_quality(clean)
    assert not verify.clay_failures("v", q), q
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    assert any("noise" in f for f in verify.clay_failures("v", verify.clay_quality(noisy)))
    mag = clean.copy()
    mag[:40] = (255, 0, 255)
    assert any("magenta" in f for f in verify.clay_failures("v", verify.clay_quality(mag)))
