"""Texture size per object from the image evidence (not surface area), computed from the GEOMETRY.

Rule: for every triangle a view really sees (front-facing, centroid on this object in object_id and on the
depth_z surface), R_tri = sqrt(image area in source px^2 / UV area) is the texture size that gives one texel per
source pixel there (image resolution included). Per view take an image-area-weighted percentile of R_tri
(median; p85 for detail-critical objects so the sharpest region, e.g. a face in a close-up, is not starved),
take the best view, round up to a power of two. Independent of any raster, so sub-texel triangles count.
UV occupancy is included (UV area of the triangles). Clamped to
[min_tex, max_tex]; max_tex defaults to 4096 for detail-critical objects and 2048 otherwise, scaled with the
calibration resolution (x max(W,H)/1664, rounded to a power of two, within 1024..8192).
"""
from __future__ import annotations

import numpy as np

from mvb2r.fusion_io import load_mesh

REF_CAL = 1664


def pow2_ceil(x: float) -> int:
    return int(2 ** int(np.ceil(np.log2(max(x, 1.0)))))


def default_max(detail_critical: bool, cal_px: int) -> int:
    base = 4096 if detail_critical else 2048
    s = 2 ** int(np.round(np.log2(max(cal_px, 1) / REF_CAL)))
    return int(np.clip(base * s, 1024, 8192))


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(c, q / 100.0 * c[-1]), len(x) - 1)])


def split4(P: np.ndarray, UV: np.ndarray):
    """Midpoint subdivision of every triangle into four (positions and UVs)."""
    def mids(X):
        a, b, c = X[:, 0], X[:, 1], X[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        return np.concatenate([np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return mids(P), mids(UV)


def _px_area(cam, P, s):
    u, v, z = cam.project(np.asarray(P, np.float64).reshape(-1, 3))
    u, v, z = (a.reshape(-1, 3) for a in (u, v, z))
    px = 0.5 * ((u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])) * s * s
    return u, v, z, px


def refine_large(tri_pos, tri_uv, cam, s, max_px: float = 64.0, levels: int = 6):
    """Split triangles covering more than max_px source pixels (perspective varies across big triangles and one
    centroid cannot speak for them) until they are small enough or `levels` is reached."""
    P, UV = np.asarray(tri_pos, np.float64), np.asarray(tri_uv, np.float64)
    for _ in range(levels):
        _, _, _, px = _px_area(cam, P, s)
        big = np.nan_to_num(np.abs(px)) > max_px
        if not big.any() or len(P) > 2_000_000:
            break
        sp, su = split4(P[big], UV[big])
        P, UV = np.concatenate([P[~big], sp]), np.concatenate([UV[~big], su])
    return P, UV


def triangle_resolution(tri_pos, tri_uv, view, oid: int, tol_rel: float = 0.02):
    """(R_tri, image area px^2) of the triangles this view sees (centroid test against the id / depth passes;
    large triangles are subdivided first)."""
    cam = view.cam
    tri_pos, tri_uv = refine_large(tri_pos, tri_uv, cam, view.scale)
    V = np.asarray(tri_pos, np.float64).reshape(-1, 3)
    u, v, z = cam.project(V)
    u, v, z = (a.reshape(-1, 3) for a in (u, v, z))
    s = view.scale
    px = 0.5 * ((u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])) * s * s
    uv = np.asarray(tri_uv, np.float64)
    uva = 0.5 * np.abs((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
                       - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1]))
    cu, cv, cz = np.nanmean(u, 1), np.nanmean(v, 1), z.mean(1)
    H, W = view.ids.shape
    ok = np.all(z > cam.clip_start, 1) & np.isfinite(cu) & (cu >= 0) & (cu < W) & (cv >= 0) & (cv < H)
    i = np.clip(np.nan_to_num(cu).astype(int), 0, W - 1)
    j = np.clip(np.nan_to_num(cv).astype(int), 0, H - 1)
    ok &= (view.ids[j, i] == oid) & (np.abs(view.depth[j, i] - cz) < tol_rel * cz + 0.01) & (np.abs(px) > 0)
    ok &= uva > 1e-14
    return np.sqrt(np.abs(px[ok]) / uva[ok]), np.abs(px[ok])


def choose_size(ci, obj: dict, vis, opt: dict) -> dict:
    z = load_mesh(ci.mesh_dir, obj["file"])
    q = 85.0 if obj["detail_critical"] else 50.0
    best, view, npx = 0.0, None, 0
    for vd in ci.views:
        r, w = triangle_resolution(z["tri_pos"], z["tri_uv"], vd, obj["id"])
        if w.sum() < 50:  # fewer than ~50 source pixels: no evidence from this view
            continue
        val = weighted_percentile(r, w, q)
        if val > best:
            best, view, npx = val, vd.cam.name, int(w.sum())
    cal = max(max(c.width, c.height) for c in ci.cams)
    hi = int(opt.get("max_tex") or default_max(obj["detail_critical"], cal))
    lo = int(opt.get("min_tex", 256))
    R = int(np.clip(pow2_ceil(best), lo, hi)) if best > 0 else lo
    return {"tex_size": R, "best_view": view, "best_view_ppt_at_size": best / R if best else None,
            "percentile": q, "object_px": npx, "max_tex": hi, "clamped": bool(best > 0 and R != pow2_ceil(best))}
