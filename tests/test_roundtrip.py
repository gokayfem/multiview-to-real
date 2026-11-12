"""Synthetic end-to-end round trip of the fusion half (needs Blender; skipped with a message otherwise).

A tiny primitives scene with known ground-truth textures, 4 cameras at 320 px, perturbed "generated" views
(gain/tint fields, few-px smooth warps, view-specific detail), a generated mask for the detail-critical object.
Runs `project --method all` and `score`, then checks with pass/fail thresholds:
  texture PSNR vs ground truth on observed texels, object leakage, UV flip, single-source regions,
  coverage bookkeeping, re-renders / novel views / diagnostic.blend present, LOO and seam numbers in range.
Set MVB2R_KEEP_ROUNDTRIP=1 to keep the temp case folder.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _fusion_fixtures import Skip

HERE = Path(__file__).resolve().parent
# warps of up to ~4 px exceed the 1.5 px hard edge margin, so an ungated, half-occluded object may take a few
# per cent of its neighbours' colour at its silhouette (real generated views do the same); a masked object must not.
THRESH = {"psnr_min_db": 21.0, "leak_max": 0.03, "leak_max_masked": 0.01, "loo_full_max": 20.0, "seam_novel_max": 1.6}


def _blender_or_skip():
    sys.path.insert(0, str(HERE.parent / "skills" / "multiview-blockout-to-real" / "scripts"))
    from mvb2r.fusion_io import find_blender
    if find_blender() is None:
        raise Skip("Blender not found ($BLENDER / PATH / macOS app): synthetic round trip not run")


def _stage(name, argv):
    import importlib
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    importlib.import_module(f"mvb2r.{name}").register(sub)
    a = ap.parse_args([name] + argv)
    assert a.func(a) == 0, name


def _psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def _checks(case: Path) -> dict:
    from mvb2r.fusion_io import read_gray01, read_rgb_linear, linear_to_srgb
    cov = json.loads((case / "proj" / "coverage.json").read_text())
    hues = {int(k): np.array(v) for k, v in json.loads((case / "gt" / "truth.json").read_text())["hues"].items()}
    res = {}
    for m in cov["methods"]:
        for name, r in cov["objects"].items():
            c = r["coverage"]
            assert abs(c["observed"] + c["soft"] + c["inferred"] - 1) < 1e-6, (name, c)
            d = case / "proj" / m / name
            T = linear_to_srgb(read_rgb_linear(d / "texture.png"))
            G = linear_to_srgb(read_rgb_linear(case / "gt" / f"{r['id']}.png"))
            if G.shape != T.shape:  # automatic texture size: compare at the projected size
                import cv2
                G = cv2.resize(G, T.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            obs = read_gray01(d / "coverage.png") > 0.5
            p, pf = _psnr(T[obs], G[obs]), _psnr(T[obs], G[::-1][obs])
            hue = read_rgb_linear(d / "texture.png")[obs]
            hue = hue / np.maximum(hue.sum(1, keepdims=True), 1e-6)
            refs = {k: v / v.sum() for k, v in hues.items()}
            own = np.linalg.norm(hue - refs[r["id"]], axis=1)
            other = np.min([np.linalg.norm(hue - v, axis=1) for k, v in refs.items() if k != r["id"]], 0)
            leak = float(np.mean(other < own))
            res[f"{m}/{name}"] = {"psnr": round(p, 2), "psnr_flipped": round(pf, 2), "leak": round(leak, 4)}
            assert all((case / "proj" / m / sub / f"{v}.png").is_file()
                       for sub, v in [("rerender", "view_01"), ("novel", "novel_01")])
    bad = {k: v for k, v in res.items() if v["psnr"] < THRESH["psnr_min_db"] or v["psnr"] <= v["psnr_flipped"] + 1
           or v["leak"] > THRESH["leak_max_masked" if k.endswith("/Ball") else "leak_max"]}
    assert not bad, ("texture checks failed", bad)
    res["Gem_holes"] = dense_mesh_holes(case, cov)
    assert res["Gem_holes"]["after"] == 0, res["Gem_holes"]
    ball = cov["objects"]["Ball"]
    assert ball["detail_critical"] and ball["single_source_regions"], ball
    assert (case / "out" / "diagnostic.blend").is_file()
    return res


def dense_mesh_holes(case: Path, cov: dict, name: str = "Gem") -> dict:
    """Pixels of the dense-mesh object (all source views) whose nearest texel has no triangle: with the bleed
    padding (after) vs the plain centre-in-triangle raster at the same size (before)."""
    from mvb2r.fusion_io import read_ids
    from mvb2r.raster import texel_set
    from mvb2r.texlookup import load_passes
    r = cov["objects"][name]
    R = r["tex_size"]
    after = read_ids(case / "proj" / cov["methods"][0] / name / "source_raw.png") != 255
    from mvb2r.fusion_io import load_mesh
    z = load_mesh(case / "proj" / "_mesh", f"mesh_{r['id']:03d}.npz")
    before = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], R, gutter=0).mask()
    out = {"tex_size": R, "rule": r.get("tex_size_rule"), "pixels": 0, "before": 0, "after": 0}
    for v in sorted((case / "views").glob("view_*")):
        ps = load_passes(v)
        uv = ps.uv[ps.ids == r["id"]]
        x = np.clip((uv[:, 0] * R).astype(int), 0, R - 1)
        y = np.clip(((1 - uv[:, 1]) * R).astype(int), 0, R - 1)
        out["pixels"] += len(uv)
        out["before"] += int((~before[y, x]).sum())
        out["after"] += int((~after[y, x]).sum())
    return out


def test_roundtrip_project_and_score():
    _blender_or_skip()
    tmp = Path(tempfile.mkdtemp(prefix="mvb2r_rt_"))
    case = tmp / "rt_case"
    try:
        r = subprocess.run([sys.executable, str(HERE / "roundtrip" / "make_case.py"), str(case), "--size", "320",
                            "--tex", "256"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        _stage("project", [str(case), "--method", "all"])
        _stage("score", [str(case)])
        res = _checks(case)
        sc = json.loads((case / "proj" / "score.json").read_text())
        for m, s in sc["methods"].items():
            assert s["loo"]["mean"]["full"] <= THRESH["loo_full_max"], (m, s["loo"]["mean"])
            assert s["seams_novel"]["coarse_ratio_mean"] <= THRESH["seam_novel_max"], (m, s["seams_novel"])
        holes = res.pop("Gem_holes")
        worst = min(res.items(), key=lambda kv: kv[1]["psnr"])
        loo = {m: round(s["loo"]["mean"]["full"], 2) for m, s in sc["methods"].items()}
        return f"worst PSNR {worst[0]}={worst[1]['psnr']} dB; LOO full {loo}; dense mesh {holes}"
    finally:
        if not os.environ.get("MVB2R_KEEP_ROUNDTRIP"):
            shutil.rmtree(tmp, ignore_errors=True)


test_roundtrip_project_and_score.slow = True


def test_roundtrip_lit_delight_reduces_shading_correlation():
    """Generated views WITH baked shading from the scene's own sun: de-light must lower the log-luminance
    correlation with the proxy shading in every view (it does not remove it: the gain is clamped 0.7-1.6)."""
    _blender_or_skip()
    tmp = Path(tempfile.mkdtemp(prefix="mvb2r_lit_"))
    case = tmp / "lit_case"
    try:
        r = subprocess.run([sys.executable, str(HERE / "roundtrip" / "make_case.py"), str(case), "--size", "256",
                            "--tex", "256", "--lit"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        _stage("project", [str(case), "--method", "best", "--no-render"])
        _stage("pbr", [str(case), "--method", "best", "--fit-iters", "2"])
        ev = json.loads((case / "pbr" / "eval.json").read_text())["delight"]
        pairs = {v: (round(d["corr_raw"], 3), round(d["corr_delit"], 3)) for v, d in ev.items()}
        assert all(b < a for a, b in pairs.values()), pairs
        return f"corr raw -> delit {pairs}"
    finally:
        if not os.environ.get("MVB2R_KEEP_ROUNDTRIP"):
            shutil.rmtree(tmp, ignore_errors=True)


test_roundtrip_lit_delight_reduces_shading_correlation.slow = True


def background_leak(case: Path, method: str) -> dict:
    """Per object (floor excluded): fraction of observed texels whose colour is closer to the studio wall than
    to the object's own median colour."""
    from mvb2r.fusion_io import read_gray01, read_rgb_linear
    sys.path.insert(0, str(HERE / "roundtrip"))
    from make_case import WALL
    cov = json.loads((case / "proj" / "coverage.json").read_text())
    out = {}
    for n, r in cov["objects"].items():
        if n == "Floor":
            continue
        T = read_rgb_linear(case / "proj" / method / n / "texture.png")
        c = T[read_gray01(case / "proj" / method / n / "coverage.png") > 0.5]
        own = np.median(c, 0)
        out[n] = round(float(np.mean(np.linalg.norm(c - WALL, axis=1) < np.linalg.norm(c - own, axis=1))), 4)
    return out


def test_roundtrip_open_scene_generated_objects_smaller():
    """Open scene (background id 0 = studio wall in the generated views), objects 4 % smaller than the clay:
    texels must not take the wall colour (silhouette margins, mask gate, same-object sampling)."""
    _blender_or_skip()
    tmp = Path(tempfile.mkdtemp(prefix="mvb2r_open_"))
    case = tmp / "open_case"
    try:
        r = subprocess.run([sys.executable, str(HERE / "roundtrip" / "make_case.py"), str(case), "--size", "320",
                            "--shrink", "0.04"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        _stage("project", [str(case), "--method", "best,multiband", "--no-render"])
        leaks = {m: background_leak(case, m) for m in ("best", "multiband")}
        bad = {m: {n: v for n, v in d.items() if v > 0.02} for m, d in leaks.items()}
        assert not any(bad.values()), leaks
        return f"wall-coloured texel fraction {leaks}"
    finally:
        if not os.environ.get("MVB2R_KEEP_ROUNDTRIP"):
            shutil.rmtree(tmp, ignore_errors=True)


test_roundtrip_open_scene_generated_objects_smaller.slow = True
