"""Refit (silhouette carving): pure carve decisions on synthetic voxels / masks, and a Blender-gated carve of
an oversized sphere to the masks of a smaller one from 4 views."""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from _fusion_fixtures import Skip, quad, top_camera

HERE = Path(__file__).resolve().parent


def _prm(**kw):
    from mvb2r.voxcarve import CarveParams
    return CarveParams(**{"voxel": 0.05, "inner": 0.05, "outer": 0.2, "dilate": 0, **kw})


def _plane_points():
    xs = np.linspace(-0.9, 0.9, 19)
    X, Y = np.meshgrid(xs, xs)
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]


def test_carve_removes_outside_mask_only():
    from mvb2r.voxcarve import carve_view
    cam = top_camera(W=64, height=3.0, f=40.0)
    P = _plane_points()
    own = np.zeros((64, 64), bool)
    own[24:40, 24:40] = True                      # |x|,|y| < 0.6
    depth = np.full((64, 64), 1e10, np.float32)
    ids = np.zeros((64, 64), np.int32)
    rm = carve_view(P, np.ones(len(P), bool), cam, own, np.zeros_like(own), depth, ids, 1, _prm(), tol=0.05)
    inside = (np.abs(P[:, 0]) < 0.55) & (np.abs(P[:, 1]) < 0.55)
    assert not np.isin(np.nonzero(inside)[0], rm).any()
    outside = (np.abs(P[:, 0]) > 0.65) | (np.abs(P[:, 1]) > 0.65)
    assert np.isin(np.nonzero(outside)[0], rm).all()


def test_carve_respects_other_geometry_and_other_masks():
    from mvb2r.voxcarve import carve_view
    cam = top_camera(W=64, height=3.0, f=40.0)
    P = _plane_points()
    own = np.zeros((64, 64), bool)
    depth = np.full((64, 64), 1.0, np.float32)    # other object 1 m from the camera covers the whole view
    ids = np.full((64, 64), 7, np.int32)
    rm = carve_view(P, np.ones(len(P), bool), cam, own, np.zeros_like(own), depth, ids, 1, _prm(), tol=0.05)
    assert len(rm) == 0                           # occluded by OTHER geometry: no evidence, nothing carved
    own[:] = True
    other = np.zeros_like(own)
    other[:, :32] = True                          # another object's mask on the left half (own mask everywhere)
    rm = carve_view(P, np.ones(len(P), bool), cam, own, other, np.full((64, 64), 1e10, np.float32),
                    np.zeros((64, 64), np.int32), 1, _prm(), tol=0.05)
    assert len(rm) == 0                           # other mask counts only outside the object's own mask
    own[:, :32] = False
    rm = carve_view(P, np.ones(len(P), bool), cam, own, other, np.full((64, 64), 1e10, np.float32),
                    np.zeros((64, 64), np.int32), 1, _prm(), tol=0.05)
    assert np.all(P[rm, 0] < 0.05)


def test_signed_distance_candidates_and_mesh_closed():
    from mvb2r import voxcarve as vc
    P, _, _ = quad(size=1.0)
    box = []
    for z in (0.0, 1.0):  # crude closed cube from two quads + 4 sides
        box.append(P + [0, 0, z - 0.5])
    sd = vc.signed_distance(np.array([[0, 0, 0.9], [0, 0, 0.5]]), np.concatenate(box), 0.02)
    assert abs(abs(sd[0]) - 0.4) < 0.03 and abs(sd[1]) < 0.03
    prm = _prm(mode="shell")
    assert vc.candidates(np.array([-0.1, 0.0, 0.1, 0.3]), prm).tolist() == [False, True, True, False]
    occ = np.zeros((12, 12, 12), bool)
    occ[3:9, 3:9, 3:9] = True
    occ[0, 0, 0] = True  # floating speck removed
    occ = vc.largest_component(occ)
    assert occ.sum() == 216
    V, F = vc.mesh_from_occupancy(occ, np.zeros(3), _prm())
    assert vc.is_closed(F) and len(F) > 50


def test_scaled_params_and_metrics():
    from mvb2r import voxcarve as vc
    P, _, _ = quad(size=0.36 / np.sqrt(2))
    prm = vc.scaled_params(P)
    assert abs(prm.outer - 0.035) < 0.004 and prm.voxel >= 0.0025 * 0.9
    ids = np.zeros((10, 10), int)
    ids[2:8, 2:8] = 5
    own = np.zeros((10, 10), bool)
    own[2:6, 2:8] = True
    oth = np.zeros_like(own)
    oth[6:, :] = True
    m = vc.mask_metrics(ids, 5, own, oth)
    assert abs(m["iou"] - 24 / 36) < 1e-9 and m["outside_mask_px"] == 12 and m["covers_other_masks_px"] == 12


def test_refit_oversized_sphere_blender():
    sys.path.insert(0, str(HERE.parent / "skills" / "multiview-blockout-to-real" / "scripts"))
    from mvb2r.fusion_io import find_blender, read_ids, run_blender, write_json, write_png
    if find_blender() is None:
        raise Skip("Blender not found: refit carve test not run")
    tmp = Path(tempfile.mkdtemp(prefix="mvb2r_refit_"))
    case = tmp / "refit_case"
    try:
        run_blender("../../../../tests/roundtrip/build_refit_scene.py", ["--out", case, "--size", 256])
        cams = case / "views" / "cameras.json"
        run_blender("render.py", ["--cameras", cams, "--out", tmp / "truth", "--passes", "--no-beauty", "--cpu"],
                    case / "truth.blend")
        run_blender("render.py", ["--cameras", cams, "--out", case / "views", "--passes", "--no-beauty", "--cpu"],
                    case / "blockout.blend")
        for v in json.loads(cams.read_text())["views"]:
            ids = read_ids(tmp / "truth" / v["name"] / "object_id.png")
            n = v["name"].split("_")[-1]
            write_png(case / "targets" / f"mask_Blob_{n}.png", ((ids == 20) * 255).astype(np.uint8))
            write_png(case / "targets" / f"mask_Floor_{n}.png", ((ids == 10) * 255).astype(np.uint8))
        write_json(case / "case.json", {"name": "refit_case", "objects": [{"name": "Blob", "refit": True}]})
        import importlib
        ap = argparse.ArgumentParser()
        importlib.import_module("mvb2r.refit").register(ap.add_subparsers())
        a = ap.parse_args(["refit", str(case), "--dilate", "1"])  # exact synthetic masks
        assert a.func(a) == 0
        rep = json.loads((case / "refit" / "refit.json").read_text())["objects"]["Blob"]
        ious = [(m["before"]["iou"], m["after"]["iou"]) for m in rep["per_view"].values()]
        assert all(after > 0.9 and after > before + 0.1 for before, after in ious), ious
        assert rep["mesh_after"]["closed"] and rep["closed_before_decimate"], rep["mesh_after"]
        assert (case / "blockout.blend").is_file() and (case / "blockout_refit.blend").is_file()
        return "IoU before->after " + ", ".join(f"{b:.2f}->{a:.2f}" for b, a in ious) + f"; {rep['mesh_after']}"
    finally:
        if not os.environ.get("MVB2R_KEEP_ROUNDTRIP"):
            shutil.rmtree(tmp, ignore_errors=True)


test_refit_oversized_sphere_blender.slow = True
