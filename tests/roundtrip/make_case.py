"""Build the synthetic round-trip case: primitives scene, exact passes, ground-truth textures and perturbed
"generated" views (gain/tint fields, few-pixel smooth warps, view-specific fine detail), one object's masks.

    python3 tests/roundtrip/make_case.py CASE_DIR [--size 320] [--tex 256]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "skills" / "multiview-blockout-to-real" / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mvb2r.fusion_io import read_rgb_linear, run_blender, write_json, write_png, write_srgb_png  # noqa: E402
from mvb2r.texlookup import load_passes, render  # noqa: E402

HUES = {10: (0.55, 0.45, 0.30), 20: (0.70, 0.20, 0.15), 30: (0.15, 0.35, 0.65), 40: (0.25, 0.55, 0.20), 50: (0.60, 0.50, 0.08)}


def gt_texture(oid, R, seed):
    """Distinct hue per object, asymmetric pattern (stripes + a corner marker) so flips are detectable."""
    rng = np.random.default_rng(seed)
    j, i = np.mgrid[0:R, 0:R] / R
    base = np.array(HUES[oid], np.float32)
    stripes = 0.75 + 0.25 * np.sign(np.sin(2 * np.pi * (6 * i + 2 * j)))
    blobs = cv2.GaussianBlur(rng.random((R, R)).astype(np.float32), (0, 0), R / 64)
    blobs = 0.85 + 0.3 * (blobs - blobs.mean()) / (blobs.std() + 1e-6) * 0.3
    marker = 1.0 + 0.6 * ((i < 0.25) & (j < 0.25))
    return np.clip(base[None, None] * (stripes * blobs * marker)[..., None], 0, 1).astype(np.float32)


WALL = np.array([0.80, 0.80, 0.74], np.float32)   # "studio wall" the image model paints behind an open scene


def shrink_objects(img, ids, frac, floor_id=10):
    """Generated content `frac` smaller than the clay: every non-floor object is scaled toward its 2D centroid;
    the vacated ring shows the studio wall (as in real generated views of an open scene). Returns (img, ids)."""
    out, gid = img.copy(), ids.copy()
    H, W = ids.shape
    jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
    for oid in [o for o in np.unique(ids) if o not in (0, floor_id)]:
        m = ids == oid
        cy, cx = np.argwhere(m).mean(0)
        sx, sy = cx + (ii - cx) / (1 - frac), cy + (jj - cy) / (1 - frac)   # source of each shrunk pixel
        src_in = cv2.remap(m.astype(np.float32), sx, sy, cv2.INTER_NEAREST) > 0.5
        moved = cv2.remap(img, sx, sy, cv2.INTER_LINEAR)
        out[m & ~src_in] = WALL
        gid[m & ~src_in] = 0
        out[src_in] = moved[src_in]
        gid[src_in] = oid
    return out, gid


def perturb(img, ids, seed, size, strength=1.0, bg=(0.3, 0.32, 0.35)):
    """strength scales the per-view exposure / tint / low-frequency gain differences (1 = mild)."""
    rng = np.random.default_rng(seed)
    H, W = ids.shape
    low = cv2.resize(rng.normal(0, 1, (4, 4, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    expo = 0.0 if strength <= 1 else 0.12 * strength * rng.normal()
    gain = np.exp(strength * (0.06 * low[..., :1] + np.array([0.04, 0.0, -0.04]) * (seed % 3 - 1)) + expo)
    flow = cv2.resize(rng.normal(0, 1.5, (3, 3, 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
    warped = cv2.remap(img * gain, ii + flow[..., 0], jj + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    detail = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0)
    out = warped * (1 + 0.05 * detail[..., None])
    wids = cv2.remap(ids.astype(np.float32), ii + flow[..., 0], jj + flow[..., 1], cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REFLECT).astype(np.int32)   # where each object really is in the image
    return (np.where((wids > 0)[..., None], out, np.array(bg, np.float32)), float(np.abs(flow).max()), wids)


def render_lit(case: Path, size: int) -> Path:
    """Ground truth textures as diffuse albedo under the blockout's own sun + world, Cycles (baked shading)."""
    objs = json.loads((case / "views" / "objects.json").read_text())
    mats = {o["name"]: {"texture": str(case / "gt" / f"{o['id']}.png"), "roughness": 1.0, "specular": 0.0,
                        "bump": 0.0} for o in objs}
    write_json(case / "gt" / "lit_materials.json", mats)
    run_blender("build_pbr.py", ["--materials", case / "gt" / "lit_materials.json", "--out", case / "gt" / "lit.blend"],
                blend=case / "blockout.blend", log=case / "logs" / "lit.log")
    run_blender("render.py", ["--cameras", case / "views" / "cameras.json", "--out", case / "gt" / "lit",
                              "--samples", 64, "--denoise"], blend=case / "gt" / "lit.blend", log=case / "logs" / "lit.log")
    return case / "gt" / "lit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--tex", type=int, default=256)
    ap.add_argument("--lit", action="store_true", help="generated views carry baked shading from the scene's sun")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="open scene: generated objects this fraction smaller than the clay, studio wall behind")
    ap.add_argument("--strength", type=float, default=1.0, help="per-view colour / exposure disagreement (1 = mild)")
    a = ap.parse_args()
    case = Path(a.case).resolve()
    (case / "targets").mkdir(parents=True, exist_ok=True)
    run_blender("../../../../tests/roundtrip/build_scene.py", ["--out", case, "--size", a.size],
                log=case / "logs" / "build.log")
    run_blender("render.py", ["--cameras", case / "views" / "cameras.json", "--out", case / "views", "--passes",
                              "--no-beauty", "--cpu"], blend=case / "blockout.blend", log=case / "logs" / "passes.log")
    gt = {oid: gt_texture(oid, a.tex, oid) for oid in HUES}
    for oid, t in gt.items():
        write_srgb_png(case / "gt" / f"{oid}.png", t)
    cams = json.loads((case / "views" / "cameras.json").read_text())["views"]
    lit = render_lit(case, a.size) if a.lit else None
    warps = {}
    for k, c in enumerate(cams):
        ps = load_passes(case / "views" / c["name"])
        img, _ = render(ps, gt, (0.3, 0.32, 0.35))
        if lit is not None:
            img = read_rgb_linear(lit / f"{c['name']}.png")
        gen_ids = ps.ids
        if a.shrink > 0:
            img = np.where((ps.ids > 0)[..., None], img, WALL)
            img, gen_ids = shrink_objects(img, ps.ids, a.shrink)
        tgt, warps[c["name"]], wids = perturb(img, gen_ids, k + 1, a.size, a.strength,
                                              WALL if a.shrink > 0 else (0.3, 0.32, 0.35))
        write_srgb_png(case / "targets" / f"{c['name']}.png", tgt)
        num = c["name"].split("_")[-1]
        # generated masks follow the generated image (warped, shrunk), like the core half's targets stage;
        # the normal run masks only the detail-critical Ball, the open-scene run masks every object
        for oid, nm in ([(20, "Box"), (30, "Ball"), (40, "Post"), (50, "Gem")] if a.shrink > 0 else [(30, "Ball")]):
            write_png(case / "targets" / f"mask_{nm}_{num}.png", ((wids == oid) * 255).astype(np.uint8))
    cfg = {"name": case.name, "objects": [{"name": "Floor", "class": "stone"}, {"name": "Box", "class": "wood"},
                                          {"name": "Ball", "class": "plastic", "detail_critical": True},
                                          {"name": "Post", "class": "paint"},
                                          {"name": "Gem", "class": "stone", "detail_critical": True}],
           "fusion": {"method": "auto", "novel": 3}}  # texture size: automatic rule
    write_json(case / "case.json", cfg)
    write_json(case / "gt" / "truth.json", {"max_warp_px": warps, "hues": HUES})
    print("[make_case] ok", case)


if __name__ == "__main__":
    main()
