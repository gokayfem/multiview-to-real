"""Leak metric of `score`: observed texels whose source pixel the generated image assigns to another object.

Independent of the gate policy used for fusion: it reads only the delivered source labels
(proj/<method>/<object>/source_raw.png, soft_observed.png), the generated masks and the clay passes.
"""
from __future__ import annotations

import numpy as np

from mvb2r.fusion_io import read_ids, safe_name
from mvb2r.gating import occluder_bands, px

LEAK_MARGIN_PX = 2.0      # nearer-object clay margin at the 1664 px reference (scaled up like the gates)
LEAK_DZ = 0.03            # m: "nearer" = more than this closer than the pixel's own clay depth
UNOBSERVED_RAW = 0        # source_raw.png: 0 = unobserved, v + 1 = view v
SOFT_THRESHOLD = 127
LEAK_DEFINITION = (
    "An observed texel (not soft, not filled) is leaked when its exact source pixel, in the view it was taken "
    "from, lies in pixels the generated image assigns to another object. Objects with their own generated mask "
    "in that view: outside the own UNDILATED mask, or inside another object's mask. Objects without one: inside "
    "any other object's undilated mask, or within 2 px (1664 px reference) of a nearer object's clay pixels. "
    "Independent of the gate policy used for fusion. It cannot see generated content displaced beyond these "
    "margins (in the first run a table and a sofa rail repeated on the wall scored 0 here): that stays a visual "
    "check on the contact sheets.")


def bad_pixels(ci, obj: dict, vi: int) -> np.ndarray:
    """(H,W) bool: pixels of view vi that the generated image assigns to another object than `obj`."""
    own = ci.masks.get((obj["name"], vi))
    others = [m for (n, v), m in ci.masks.items() if v == vi and n != obj["name"]]
    other_any = np.any(others, 0) if others else np.zeros(ci.views[vi].ids.shape, bool)
    if own is not None:
        return ~own | other_any
    r = int(round(px({}, "leak_margin", LEAK_MARGIN_PX, ci.cams[vi])))
    hard, _ = occluder_bands(ci.views[vi], obj["id"], r, r, LEAK_DZ)
    return other_any | hard


def bad_pixel_cache(ci):
    """bad(obj, vi) -> bad_pixels, computed once per (object, view) (a full-image erosion each)."""
    memo: dict = {}

    def bad(obj: dict, vi: int) -> np.ndarray:
        key = (obj["name"], vi)
        if key not in memo:
            memo[key] = bad_pixels(ci, obj, vi)
        return memo[key]
    return bad


def leak_counts(ci, obj: dict, tex, source: np.ndarray, soft: np.ndarray, bad=None) -> dict:
    """{"observed", "leaked", "fraction"} of one object's texels given each texel's source view (-1 none).
    `bad`: a bad_pixel_cache(ci) shared across methods."""
    bad = bad or bad_pixel_cache(ci)
    obs = (source >= 0) & (source < len(ci.cams)) & ~soft
    leaked = np.zeros(tex.n, bool)
    for vi, cam in enumerate(ci.cams):
        k = np.flatnonzero(obs & (source == vi))
        if not len(k):
            continue
        u, v, _ = cam.project(tex.P[k])
        i = np.clip(np.floor(np.nan_to_num(u, nan=0.0)).astype(np.int64), 0, cam.width - 1)
        j = np.clip(np.floor(np.nan_to_num(v, nan=0.0)).astype(np.int64), 0, cam.height - 1)
        leaked[k] = bad(obj, vi)[j, i]
    n, b = int(obs.sum()), int(leaked.sum())
    return {"observed": n, "leaked": b, "fraction": float(b / n) if n else 0.0}


def method_summary(objects: dict) -> dict:
    """{"objects", "leaked_fraction"}; the fraction is None when nothing was measured (no source_raw.png of
    this method, or no observed texel)."""
    n = sum(r["observed"] for r in objects.values())
    b = sum(r["leaked"] for r in objects.values())
    return {"objects": objects, "leaked_fraction": float(b / n) if n else None}


def _labels(d, tex):
    raw = read_ids(d / "source_raw.png")[tex.rows, tex.cols].astype(np.int64)
    source = np.where(raw == UNOBSERVED_RAW, -1, raw - 1)
    sp = d / "soft_observed.png"
    soft = read_ids(sp)[tex.rows, tex.cols] > SOFT_THRESHOLD if sp.is_file() else np.zeros(tex.n, bool)
    return source, soft


def leak_report(case, ci, cov: dict, methods: list) -> dict:
    """score.json["leak"]: {"definition", "per_method": {method: {"objects": {name: counts}, "leaked_fraction"}}}."""
    from mvb2r.fusion_io import load_mesh
    from mvb2r.raster import texel_set
    per = {m: {} for m in methods}
    for obj in ci.objects:
        bad = bad_pixel_cache(ci)     # per object: its full-resolution masks are freed before the next one
        rec = cov["objects"].get(obj["name"], {})
        if obj["opening"] or not rec.get("projected"):
            continue
        z = load_mesh(ci.mesh_dir, obj["file"])
        tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], rec["tex_size"])
        for m in methods:
            d = case / "proj" / m / safe_name(obj["name"])
            if (d / "source_raw.png").is_file():
                per[m][obj["name"]] = leak_counts(ci, obj, tex, *_labels(d, tex), bad=bad)
    return {"definition": LEAK_DEFINITION, "per_method": {m: method_summary(o) for m, o in per.items()}}
