"""Band-split fusion: frequency split and multi-band (Laplacian levels) blending, UV-space helpers.

Low bands are computed in IMAGE space per view (normalised convolution inside the object's admissible pixels,
linear light) and sampled at the texels; only the finest detail comes from one labelled source per texel.
Ported from the experiment's seam bake-off (scripts/seam/methods.py: SPLIT, MB).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from mvb2r.fusion import REF_CAL, ObjectInput, FusionParams, Result, forced_mask, logdist, tex_scale


def uv_blur(tex, vals: np.ndarray, sigma: float, weight: np.ndarray | None = None) -> np.ndarray:
    """Normalised Gaussian blur of per-texel values (N,) or (N,C) inside the atlas."""
    R = tex.R
    w = np.ones(tex.n, np.float32) if weight is None else np.asarray(weight, np.float32)
    v2 = vals[:, None] if vals.ndim == 1 else vals
    W = np.zeros((R, R), np.float32)
    W[tex.rows, tex.cols] = w
    den = cv2.GaussianBlur(W, (0, 0), sigma)[tex.rows, tex.cols]
    out = np.zeros(v2.shape, np.float32)
    for c in range(v2.shape[1]):
        img = np.zeros((R, R), np.float32)
        img[tex.rows, tex.cols] = v2[:, c] * w
        out[:, c] = cv2.GaussianBlur(img, (0, 0), sigma)[tex.rows, tex.cols] / np.maximum(den, 1e-6)
    return out[:, 0] if vals.ndim == 1 else out


def valid_ramp(tex, valid: np.ndarray, width: float) -> np.ndarray:
    """Smoothstep 0 -> 1 over `width` texels away from covered-but-invalid texels (chart borders don't count)."""
    img = np.ones((tex.R, tex.R), np.uint8)
    img[tex.rows, tex.cols] = valid.astype(np.uint8)
    d = cv2.distanceTransform(img, cv2.DIST_L2, 5)[tex.rows, tex.cols]
    x = np.clip(d / max(width, 1e-3), 0, 1)
    return (x * x * (3 - 2 * x)).astype(np.float32)


def normalised_lowpass(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass of img inside mask only (normalised convolution); zero outside support."""
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return (num / np.maximum(den, 1e-4)[..., None]).astype(np.float32)


def labels_for(obj: ObjectInput, prm: FusionParams, labeller: str) -> np.ndarray:
    """(N,) source label (-1 unobserved) from best / regions / graphcut labelling."""
    from mvb2r.fusion import method_best
    st = obj.st
    if labeller == "regions":
        from mvb2r.regions import method_regions
        lab = method_regions(obj, prm).source
    elif labeller == "graphcut":
        from mvb2r.seams import graphcut_labels
        lab = graphcut_labels(obj, prm)
    else:
        lab = method_best(obj, prm).source
    return np.where(st.valid.any(0), lab, -1).astype(np.int64)


def low_weights(obj: ObjectInput, prm: FusionParams, lab: np.ndarray) -> np.ndarray:
    """(V,N) low-band weights: score^p x ramp from invalid; single-source regions keep their source only."""
    st, tex = obj.st, obj.tex
    W = np.zeros((st.V, st.N), np.float32)
    for i in range(st.V):
        W[i] = np.where(st.valid[i], np.maximum(st.score[i], 1e-4) ** prm.blend_power, 0) * \
            valid_ramp(tex, st.valid[i], prm.split_ramp * tex_scale(tex))
    single = forced_mask(st, obj.regions)
    if single.any():
        feather = np.clip(np.maximum(uv_blur(tex, single.astype(np.float32), 6.0), single), 0, 1)
        own = (lab[None] == np.arange(st.V)[:, None])
        W = np.where(own, W, W * (1.0 - feather[None]))
    return W


def method_freqsplit(obj: ObjectInput, prm: FusionParams, labeller: str = "best", sigma: float | None = None) -> Result:
    """Low band: confidence-weighted blend of every valid view's image-space low-pass (normalised convolution
    inside the object's own admissible pixels, linear light); high band: the labelled source, as a ratio."""
    st = obj.st
    idx = np.arange(st.N)
    lab = labels_for(obj, prm, labeller)
    obs = lab >= 0
    labc = np.maximum(lab, 0)
    sigma = prm.split_sigma if sigma is None else sigma
    LP = obj.bands(sigma, st) if obj.bands is not None else np.exp(st.loglp) - 0.02
    W = low_weights(obj, prm, lab)
    ws = W.sum(0)
    Ls = LP[labc, idx]
    low = np.einsum("vn,vnc->nc", W, LP) / np.maximum(ws, 1e-12)[:, None]
    low = np.where((ws > 1e-8)[:, None], low, Ls)
    col = st.color[labc, idx] * np.clip(low / np.maximum(Ls, 2e-3), 0.25, 4.0)
    col = np.where(obs[:, None], np.clip(col, 0, 1.5), 0).astype(np.float32)
    conf = np.where(obs, st.score[labc, idx], 0).astype(np.float32)
    return Result(color=col, source=lab.astype(np.int16), confidence=conf, observed=obs,
                  extra={"labeller": labeller, "sigma_ref_px": sigma})


def method_regions_freqsplit(obj: ObjectInput, prm: FusionParams) -> Result:
    """The bake-off winner "C+S8": regions (C) labels supply the high band, low band at sigma 8 px (1664 ref)."""
    return method_freqsplit(obj, prm, labeller="regions", sigma=prm.auto_split_sigma)


def method_multiband(obj: ObjectInput, prm: FusionParams, labeller: str = "regions") -> Result:
    """Bands in log space: B0 = log I - log LP_s0, Bj = log LP_s(j-1) - log LP_sj, last = log LP_sk;
    band j is blended with the label one-hot feathered by mb_widths[j] texels."""
    st, tex = obj.st, obj.tex
    lab = labels_for(obj, prm, labeller)
    obs = lab >= 0
    logs = [np.log(st.color + 0.02)]
    for s in prm.mb_sigmas:
        lp = obj.bands(s, st) if obj.bands is not None else np.exp(st.loglp) - 0.02
        logs.append(np.log(np.maximum(lp, 0) + 0.02))
    bands = [logs[j] - logs[j + 1] for j in range(len(prm.mb_sigmas))] + [logs[-1]]
    k = tex_scale(tex)
    widths = [w * k for w in prm.mb_widths] + [prm.mb_widths[-1] * k] * max(0, len(bands) - len(prm.mb_widths))
    single = forced_mask(st, obj.regions)
    out = np.zeros((st.N, 3), np.float32)
    for j, B in enumerate(bands):
        out += _blend_band(tex, st, lab, obs, B, widths[j], single, prm.split_ramp * k)
    col = np.clip(np.exp(out) - 0.02, 0, 1.5)
    labc = np.maximum(lab, 0)
    conf = np.where(obs, st.score[labc, np.arange(st.N)], 0).astype(np.float32)
    return Result(color=np.where(obs[:, None], col, 0).astype(np.float32), source=lab.astype(np.int16),
                  confidence=conf, observed=obs, extra={"labeller": labeller})


def _blend_band(tex, st, lab, obs, B, width, single, ramp_max) -> np.ndarray:
    N = st.N
    acc = np.zeros((N, 3), np.float32)
    ws = np.zeros(N, np.float32)
    for i in range(st.V):
        oh = ((lab == i) & obs).astype(np.float32)
        if not st.valid[i].any():
            continue
        wv = uv_blur(tex, oh, width, weight=obs.astype(np.float32)) if width > 0.6 else oh
        wv = wv * st.valid[i] * valid_ramp(tex, st.valid[i], min(width, ramp_max))
        wv = np.where(single & (lab != i), 0, wv)
        acc += wv[:, None] * B[i]
        ws += wv
    own = B[np.maximum(lab, 0), np.arange(N)]
    return np.where((ws > 1e-6)[:, None], acc / np.maximum(ws, 1e-6)[:, None], own)


# ----------------------------------------------------------------------------------------------- image bands
def image_band_sampler(views: list, oid: int, admissible: list, u: np.ndarray, v: np.ndarray,
                       offsets: dict | None = None):
    """bands(sigma, stack) -> (V,N,3) linear: each view's normalised-convolution low-pass inside the object's
    admissible pixels, sampled bilinearly at the texels' projections (+ optional registration offsets)."""
    from mvb2r.visibility import bilerp
    cache: dict = {}

    def bands(sigma: float, st) -> np.ndarray:
        if sigma in cache:
            return cache[sigma]
        V, N = u.shape
        out = np.zeros((V, N, 3), np.float32)
        for vi, vd in enumerate(views):  # every view, independent of `st`: the cache is shared by LOO stacks
            if vd.image is None:
                continue
            s = vd.image.shape[1] / vd.cam.width
            m = admissible[vi]
            mm = m if m.shape == vd.image.shape[:2] else cv2.resize(
                m.astype(np.uint8), vd.image.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
            lp = normalised_lowpass(vd.image, mm, sigma * s * max(vd.cam.width, vd.cam.height) / REF_CAL)
            du, dv = offsets.get(vi, (0.0, 0.0)) if offsets else (0.0, 0.0)
            k = np.nonzero((u[vi] > -1) & (u[vi] < vd.cam.width + 1) & (v[vi] > -1) & (v[vi] < vd.cam.height + 1))[0]
            out[vi, k] = bilerp(lp, (u[vi, k] + (du[k] if np.ndim(du) else du)) * s,
                                (v[vi, k] + (dv[k] if np.ndim(dv) else dv)) * s)
        cache[sigma] = out
        return out
    return bands


# ----------------------------------------------------------------------------------------------- levelling
def _seam_log_ratios(st, cid, lab_cell, edges, region):
    """{(region_a, region_b): [log(lp_b) - log(lp_a) on texels valid in both views]} across label seams."""
    V, nc = st.V, len(lab_cell)
    order = np.argsort(cid, kind="stable")
    starts = np.searchsorted(cid[order], np.arange(nc + 1))
    la, lb = lab_cell[edges[:, 0]], lab_cell[edges[:, 1]]
    seam = edges[(la != lb) & (la < V) & (lb < V)]
    out: dict = {}
    for ca, cb in seam:
        va, vb = lab_cell[ca], lab_cell[cb]
        tx = np.r_[order[starts[ca]:starts[ca + 1]], order[starts[cb]:starts[cb + 1]]]
        tx = tx[st.valid[va, tx] & st.valid[vb, tx]]
        if len(tx) >= 8:
            out.setdefault((region[ca], region[cb]), []).append(st.loglp[vb, tx] - st.loglp[va, tx])
    return out


def level_gains(st, cid, lab_cell, edges, prm: FusionParams):
    """Per-region log-gain (RGB) so both sides of every seam agree (robust median, ridge, clamped).
    Pairs disagreeing by more than level_reject are content mismatch, not exposure, and are ignored."""
    V, nc = st.V, len(lab_cell)
    same = (lab_cell[edges[:, 0]] == lab_cell[edges[:, 1]]) & (lab_cell[edges[:, 0]] < V)
    es = edges[same]
    _, region = connected_components(coo_matrix((np.ones(len(es)), (es[:, 0], es[:, 1])), shape=(nc, nc)),
                                     directed=False)
    nreg = int(region.max()) + 1
    rows, cols, vals, rhs = [], [], [], []
    rejected = 0
    for (ra, rb), ds in _seam_log_ratios(st, cid, lab_cell, edges, region).items():
        d = np.concatenate(ds)
        med = np.median(d, 0)
        if len(d) < 20 or ra == rb:
            continue
        if np.mean(np.abs(med)) > prm.level_reject:
            rejected += 1
            continue
        wgt = np.sqrt(min(len(d), 400) / 400.0)
        eq = len(rhs)
        rows += [eq, eq]
        cols += [ra, rb]
        vals += [wgt, -wgt]
        rhs.append(wgt * med)
    gains = np.zeros((nreg, 3), np.float32)
    if rhs:
        eq = len(rhs)
        rows += list(range(eq, eq + nreg))
        cols += list(range(nreg))
        vals += [np.sqrt(prm.level_ridge)] * nreg
        A = coo_matrix((vals, (rows, cols)), shape=(eq + nreg, nreg)).tocsr()
        B = np.vstack([np.array(rhs), np.zeros((nreg, 3))])
        gains = np.stack([lsqr(A, B[:, c], atol=1e-8, btol=1e-8)[0] for c in range(3)], 1)
        gains = np.clip(gains, -prm.level_max, prm.level_max).astype(np.float32)
    info = {"level_regions": nreg, "level_pairs_used": len(rhs), "level_pairs_rejected_mismatch": rejected,
            "gain_abs_max": float(np.abs(gains).max()) if nreg else 0.0}
    return region, gains, info


def feather_labels(tex, st, lab, col, gains, forced, prm: FusionParams):
    """Narrow-band (feather_sigma texels) blend across label seams, only between views that agree in the low
    band after levelling and never into single-source regions. Returns (colour, band mask)."""
    if prm.feather_sigma <= 0:
        return col, np.zeros(st.N, bool)
    fs = max(prm.feather_sigma * tex_scale(tex), 0.75)
    obs = lab >= 0
    idx = np.arange(st.N)
    labc = np.maximum(lab, 0)
    ref = st.loglp[labc, idx] + gains
    acc = np.zeros((st.N, 3), np.float32)
    wsum = np.zeros(st.N, np.float32)
    for v in range(st.V):
        oh = ((lab == v) & obs).astype(np.float32)
        if not oh.any():
            continue
        wv = uv_blur(tex, oh, fs)
        gv = uv_blur(tex, gains * oh[:, None], fs, weight=np.maximum(oh, 1e-6))
        agree = st.valid[v] & (logdist(st.loglp[v] + gv, ref) <= prm.tau_gate)
        wv = np.where(agree | (lab == v), wv, 0)
        wv = np.where(forced & (lab != v), 0, wv)
        acc += wv[:, None] * st.color[v] * np.exp(gv)
        wsum += wv
    ok = obs & (wsum > 1e-6)
    fe = np.where(ok[:, None], acc / np.maximum(wsum, 1e-6)[:, None], col)
    band = ok & (np.abs(fe - col).max(1) > 1e-6)
    return np.where(ok[:, None], fe, col), band
