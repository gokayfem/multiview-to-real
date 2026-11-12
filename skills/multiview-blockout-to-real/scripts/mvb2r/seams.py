"""Seam placement and seam measurement.

graphcut   labels on small UV cells with a DIFFERENCE-AWARE pairwise cost (Kwatra et al. / Lempitsky-Ivanov):
           cutting between views a and b costs |F_a - F_b| of the high band on both sides of the cut, cheaper at
           creases; then the multiband composite.
register   bounded smooth per-object flow of each view towards the consensus (render consensus into the view
           by splatting texels, DIS optical flow, normalised-convolution smoothing, clamp to reg_max_px), samples
           re-read at the shifted positions, then multiband. Light version of Zhou & Koltun colour-map optimisation.
seam_metrics  mean step across label boundaries vs inside labels (fine and coarse), ratio 1 = invisible.
Ported from the experiment's seam bake-off (scripts/seam/methods.py, reg.py, metrics.py).
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from mvb2r.fusion import FusionParams, ObjectInput, Result, consistency, forced_mask, tex_scale


# ----------------------------------------------------------------------------------------------- graph cut
def _pair_costs(st, cid, nc, feat, k_null):
    V, L = st.V, st.V + 1
    cnt = np.bincount(cid, minlength=nc).astype(np.float64)
    M = np.zeros((nc, L, L), np.float32)
    for a in range(V):
        for b in range(a + 1, V):
            both = st.valid[a] & st.valid[b]
            d = np.abs(feat[a] - feat[b]).mean(1) * both
            nb = np.bincount(cid, weights=both, minlength=nc)
            md = np.where(nb >= 0.25 * cnt, np.bincount(cid, weights=d, minlength=nc) / np.maximum(nb, 1), k_null)
            M[:, a, b] = M[:, b, a] = md
        M[:, a, V] = M[:, V, a] = k_null
    return M


def graphcut_labels(obj: ObjectInput, prm: FusionParams, crease: float = 0.35, k_null: float = 2.0) -> np.ndarray:
    from mvb2r.fusion import method_best
    from mvb2r.regions import alpha_expansion, cells_and_edges, data_costs, _texel_labels
    st, tex = obj.st, obj.tex
    cid, nc, edges = cells_and_edges(tex, max(2, int(round(prm.gc_cell * tex_scale(tex)))))
    D = data_costs(st, consistency(st, prm), cid, nc, prm, obj.regions)
    LP = obj.bands(prm.split_sigma, st) if obj.bands is not None else np.exp(st.loglp) - 0.02
    feat = 10.0 * (np.log(st.color + 0.02) - np.log(np.maximum(LP, 0) + 0.02))  # x10: ~10 % step = 1
    M = _pair_costs(st, cid, nc, feat, k_null)
    nrm = np.zeros((nc, 3))
    np.add.at(nrm, cid, tex.Ng.astype(np.float64))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    cosn = np.sum(nrm[edges[:, 0]] * nrm[edges[:, 1]], 1) if len(edges) else np.zeros(0)
    w = prm.gc_lam * np.where(cosn < np.cos(np.radians(30)), crease, 1.0)
    p, q = edges[:, 0], edges[:, 1]
    cell_lab, _ = alpha_expansion(D, edges, w, lambda lp, lq: M[p, lp, lq] + M[q, lp, lq], max_cycles=4)
    lab, _, _ = _texel_labels(obj, cell_lab, cid, method_best(obj, prm).source.astype(np.int64))
    return lab


def method_graphcut(obj: ObjectInput, prm: FusionParams) -> Result:
    from mvb2r.levels import method_multiband
    return method_multiband(obj, prm, labeller="graphcut")


# ----------------------------------------------------------------------------------------------- registration
def splat(values: np.ndarray, u: np.ndarray, v: np.ndarray, shape, sigma: float = 0.8):
    """Texel values -> image (nearest-pixel splat + normalised Gaussian fill of sub-texel gaps)."""
    H, W = shape
    i = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
    j = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
    num = np.zeros((H, W, values.shape[1]), np.float32)
    den = np.zeros((H, W), np.float32)
    np.add.at(num, (j, i), values)
    np.add.at(den, (j, i), 1.0)
    numb = cv2.GaussianBlur(num, (0, 0), sigma)
    denb = cv2.GaussianBlur(den, (0, 0), sigma)
    return numb / np.maximum(denb, 1e-6)[..., None], denb > 0.05


def _gray8(lin):
    from mvb2r.fusion_io import linear_to_srgb
    return np.ascontiguousarray((np.clip(linear_to_srgb(lin) @ np.array([.2126, .7152, .0722]), 0, 1) * 255)
                                .astype(np.uint8))


def view_flow(cons_img, target, mask, max_px: float, smooth: float):
    """Smooth bounded flow f with consensus(x) ~ target(x + f(x)) inside mask (calibration px)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    C8 = _gray8(np.where(mask[..., None], cons_img, target))
    T8 = _gray8(target)
    f = dis.calc(C8, T8, None)
    b = dis.calc(T8, C8, None)
    hh, ww = mask.shape
    jj, ii = np.mgrid[0:hh, 0:ww].astype(np.float32)
    fb = np.hypot(*(f + cv2.remap(b, ii + f[..., 0], jj + f[..., 1], cv2.INTER_LINEAR)).transpose(2, 0, 1))
    w = (mask * np.exp(-fb / 1.5) * (np.hypot(f[..., 0], f[..., 1]) < 2 * max_px)).astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), smooth)
    fs = np.stack([cv2.GaussianBlur(f[..., c] * w, (0, 0), smooth) for c in range(2)], -1) / \
        np.maximum(den, 1e-4)[..., None]
    if mask.any():
        fs = fs * np.clip(den / (np.percentile(den[mask], 50) + 1e-6), 0, 1)[..., None]
    mag = np.hypot(fs[..., 0], fs[..., 1])
    return fs * np.minimum(1.0, max_px / np.maximum(mag, 1e-6))[..., None]


def method_register(obj: ObjectInput, prm: FusionParams) -> Result:
    from mvb2r.levels import method_multiband
    from mvb2r.visibility import bilerp
    if obj.resample is None or obj.targets is None:
        return replace(method_multiband(obj, prm), extra={"register": "skipped: no image access"})
    cur, stats = obj, {}
    for _ in range(max(1, prm.reg_iters)):
        res = method_multiband(cur, prm)
        offsets = {}
        for vi, (img, shape) in enumerate(obj.targets):
            k = np.nonzero(res.observed & obj.st.valid[vi])[0]
            if img is None or len(k) < 200 or not obj.st.valid[vi].any():
                continue
            cons, m = splat(res.color[k], obj.st.u[vi, k], obj.st.v[vi, k], shape)
            F = view_flow(cons, img, m & cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool),
                          prm.reg_max_px, prm.reg_smooth)
            fv = bilerp(F, obj.st.u[vi], obj.st.v[vi])
            offsets[vi] = (fv[:, 0].astype(np.float32), fv[:, 1].astype(np.float32))
            stats[obj.st.view_names[vi]] = float(np.median(np.hypot(fv[k, 0], fv[k, 1])))
        cur = obj.resample(offsets, obj.st)  # keep the caller's stack (LOO: held-out view stays out)
    out = method_multiband(cur, prm)
    return replace(out, extra={**out.extra, "register_median_px": stats})


# ----------------------------------------------------------------------------------------------- metrics
def _pairs(img, lab, ok, d):
    s, i = [], []
    for A, B in (((slice(None), slice(None, -d)), (slice(None), slice(d, None))),
                 ((slice(None, -d), slice(None)), (slice(d, None), slice(None)))):
        k = ok[A] & ok[B]
        dd = np.abs(img[A] - img[B]).mean(-1) * 255
        diff = lab[A] != lab[B]
        s.append(dd[k & diff])
        i.append(dd[k & ~diff])
    return np.concatenate(s), np.concatenate(i)


def _same_surface(surf, d):
    m = np.ones(surf.shape, bool)
    m[:, :-d] &= surf[:, :-d] == surf[:, d:]
    m[:-d, :] &= surf[:-d, :] == surf[d:, :]
    return m


def seam_metrics(img_srgb: np.ndarray, lab: np.ndarray, ok: np.ndarray, surf: np.ndarray | None = None,
                 d_coarse: int = 6, blur_sigma: float = 3.0) -> dict:
    """Seam visibility: mean |step| (x255) across label boundaries / inside labels, fine (adjacent pixels)
    and coarse (blurred, pairs d_coarse apart). Ratio 1 = invisible; None when there is no boundary."""
    from mvb2r.levels import normalised_lowpass
    surf = np.zeros(lab.shape, np.int32) if surf is None else surf
    out = {}
    s, i = _pairs(img_srgb, lab, ok & _same_surface(surf, 1), 1)
    out["fine_ratio"] = float(s.mean() / max(i.mean(), 1e-6)) if len(s) and len(i) else None
    out["boundary_pairs"] = int(len(s))
    bl = normalised_lowpass(img_srgb, ok, blur_sigma)
    s, i = _pairs(bl, lab, ok & _same_surface(surf, d_coarse), d_coarse)
    out["coarse_ratio"] = float(s.mean() / max(i.mean(), 1e-6)) if len(s) and len(i) else None
    return out


def texture_seam_metrics(tex, color_srgb: np.ndarray, source: np.ndarray, observed: np.ndarray) -> dict:
    from mvb2r.regions import charts_of
    img = tex.scatter(color_srgb, 0, np.float32)
    lab = tex.scatter(source.astype(np.int32), -1, np.int32)
    ok = tex.scatter(observed, False, bool)
    surf = tex.scatter(charts_of(tex).astype(np.int32), -1, np.int32)
    return seam_metrics(img, lab, ok, surf)


def single_source_ok(res: Result, obj: ObjectInput) -> bool:
    """True when every texel of every single-source region uses a view of its region's list (no blending
    is checked by the methods themselves)."""
    m = forced_mask(obj.st, obj.regions)
    return bool(np.all(res.source[m] >= 0))
