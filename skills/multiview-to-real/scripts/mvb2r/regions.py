"""Seam-aware regional labelling (method `regions`, C) and single-source regions for detail-critical objects.

Labels live on UV cells (cell x cell texels per connected chart); cells are linked by UV adjacency inside a chart
and by 3D proximity across charts. Energy = sum data + sum w * M(l_p, l_q) minimised by alpha-expansion with
exact binary graph cuts (Kolmogorov-Zabih construction, scipy max-flow). Potts M for `regions`; the metric
difference-aware M of `seams.graphcut_labels` uses the same solver. Adapted from Waechter et al. 2014
("Let There Be Color!") for generated views — see references/seams.md and references/projection.md. Ported from scripts/proj/{method_c,mrf}.py.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.spatial import cKDTree

from mvb2r.fusion import FusionParams, ObjectInput, Result, consistency, logdist, tex_scale

INF = 1e4


# ----------------------------------------------------------------------------------------------- solver
def energy(D, edges, w, Mfun, lab) -> float:
    e = float(D[np.arange(len(lab)), lab].sum())
    if len(edges):
        e += float(np.sum(w * Mfun(lab[edges[:, 0]], lab[edges[:, 1]])))
    return e


def potts(lp, lq):
    return (lp != lq).astype(np.float64)


def _expand(D, edges, w, Mfun, lab, alpha, scale=1000.0):
    n = len(lab)
    s, t = n, n + 1
    cs = D[:, alpha].astype(np.float64).copy()
    ct = D[np.arange(n), lab].astype(np.float64).copy()
    rows, cols, caps = [], [], []
    if len(edges):
        p, q = edges[:, 0], edges[:, 1]
        lp, lq = lab[p], lab[q]
        A = w * Mfun(lp, lq)
        B = w * Mfun(lp, np.full_like(lp, alpha))
        C = w * Mfun(np.full_like(lq, alpha), lq)
        np.add.at(cs, p, np.maximum(C - A, 0))
        np.add.at(ct, p, np.maximum(A - C, 0))
        np.add.at(ct, q, C)
        pq = B + C - A
        k = pq > 0
        rows.append(p[k])
        cols.append(q[k])
        caps.append(pq[k])
    m = np.minimum(cs, ct)
    nodes = np.arange(n)
    rows += [np.full(n, s), nodes]
    cols += [nodes, np.full(n, t)]
    caps += [cs - m, ct - m]
    cap = np.rint(np.concatenate(caps) * scale).astype(np.int64)
    keep = cap > 0
    r, c = np.concatenate(rows)[keep], np.concatenate(cols)[keep]
    G = coo_matrix((np.minimum(cap[keep], 2 ** 30).astype(np.int32), (r, c)), shape=(n + 2, n + 2)).tocsr()
    res = maximum_flow(G, s, t, method="dinic")
    Rg = (G - res.flow).tocsr()
    Rg.data = (Rg.data > 0).astype(np.int8)
    Rg.eliminate_zeros()
    in_s = np.zeros(n + 2, bool)
    in_s[breadth_first_order(Rg, s, directed=True, return_predecessors=False)] = True
    return np.where(in_s[:n], lab, alpha)


def alpha_expansion(D, edges, w, Mfun=potts, init=None, max_cycles: int = 5):
    """Minimise sum D[p, l_p] + sum_e w_e M(l_p, l_q) for a metric M. Returns (labels, energy history)."""
    D = np.minimum(np.asarray(D, np.float64), INF)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    w = np.asarray(w, np.float64)
    lab = np.argmin(D, 1) if init is None else np.asarray(init).copy()
    hist = [energy(D, edges, w, Mfun, lab)]
    for _ in range(max_cycles):
        improved = False
        for alpha in range(D.shape[1]):
            if not np.any(D[:, alpha] < 1e3):
                continue
            cand = _expand(D, edges, w, Mfun, lab, alpha)
            e = energy(D, edges, w, Mfun, cand)
            if e < hist[-1] - 1e-9:
                lab, improved = cand, True
                hist.append(e)
        if not improved:
            break
    return lab, hist


# ----------------------------------------------------------------------------------------------- graph
def charts_of(tex) -> np.ndarray:
    """UV chart per texel from the CORE texels (gutter texels take their nearest core texel's chart, so padding
    never bridges two islands)."""
    core = tex.core if tex.core is not None else np.ones(tex.n, bool)
    img = np.zeros((tex.R, tex.R), np.uint8)
    img[tex.rows[core], tex.cols[core]] = 1
    _, charts = cv2.connectedComponents(img, connectivity=8)
    lab = charts[tex.rows, tex.cols].astype(np.int64)
    return lab[tex.near] if tex.near is not None else lab


def _uv_pairs(tid):
    for a, b in ((tid[:, :-1], tid[:, 1:]), (tid[:-1, :], tid[1:, :])):
        yield a, b


def cells_and_edges(tex, cell: int):
    """Cell id per texel (UV block x chart) and undirected cell adjacency (UV inside charts + 3D across)."""
    chart = charts_of(tex)
    nb = (tex.R + cell - 1) // cell
    key = chart * nb * nb + (tex.rows // cell).astype(np.int64) * nb + (tex.cols // cell)
    uk, cid = np.unique(key, return_inverse=True)
    tid = np.full((tex.R, tex.R), -1, np.int64)
    tid[tex.rows, tex.cols] = np.arange(tex.n)
    pairs, border = [], np.zeros(tex.n, bool)
    for a, b in _uv_pairs(tid):
        k = (a >= 0) & (b >= 0)
        ca, cb = cid[a[k]], cid[b[k]]
        same = chart[a[k]] == chart[b[k]]
        pairs.append(np.c_[ca[(ca != cb) & same], cb[(ca != cb) & same]])
        border[a[(a >= 0) & (b < 0)]] = True
        border[b[(b >= 0) & (a < 0)]] = True
    pairs.append(_cross_chart_pairs(tex, chart, cid, np.nonzero(border)[0]))
    e = np.sort(np.concatenate(pairs).astype(np.int64), 1)
    e = np.unique(e[e[:, 0] != e[:, 1]], axis=0) if len(e) else e.reshape(0, 2)
    return cid, len(uk), e


def _cross_chart_pairs(tex, chart, cid, bi):
    if len(bi) < 2:
        return np.zeros((0, 2), np.int64)
    r = 2.5 * float(np.median(np.sqrt(tex.area[bi])))
    pp = cKDTree(tex.P[bi]).query_pairs(r, output_type="ndarray")
    if not len(pp):
        return np.zeros((0, 2), np.int64)
    ia, ib = bi[pp[:, 0]], bi[pp[:, 1]]
    k = (chart[ia] != chart[ib]) & (np.sum(tex.Ng[ia] * tex.Ng[ib], 1) > -0.3)
    return np.c_[cid[ia[k]], cid[ib[k]]]


def data_costs(st, inc, cid, nc, prm: FusionParams, regions) -> np.ndarray:
    V = st.V
    cnt = np.bincount(cid, minlength=nc).astype(np.float64)
    D = np.full((nc, V + 1), INF)
    for v in range(V):
        ms = np.bincount(cid, weights=st.score[v], minlength=nc) / cnt
        nv = np.bincount(cid, weights=st.valid[v], minlength=nc)
        ic = np.bincount(cid, weights=inc[v], minlength=nc) / np.maximum(nv, 1)
        D[:, v] = np.where(nv / cnt >= prm.min_valid_frac, 1.0 - ms + prm.k_consist * ic, INF)
    for rg in regions:
        inreg = np.bincount(cid, weights=rg["mask"], minlength=nc) / cnt > 0.5
        for v in range(V):
            ok = inreg & (D[:, v] < INF)
            D[ok, v] += -0.5 + 0.1 * rg["views"].index(v) if v in rg["views"] else prm.region_penalty
    D[:, V] = prm.null_cost
    D[:, :V] = np.where(D[:, :V] < INF, np.maximum(D[:, :V], 0), INF)
    return D


# ----------------------------------------------------------------------------------------------- method C
def _texel_labels(obj, cell_lab, cid, fallback_lab):
    st, idx = obj.st, np.arange(obj.st.N)
    lab = cell_lab[cid].astype(np.int64)
    bad = (lab >= st.V) | ~st.valid[np.minimum(lab, st.V - 1), idx]
    obs = st.valid.any(0)
    fallback = bad & obs
    lab = np.where(fallback, fallback_lab, lab)
    forced = np.zeros(st.N, bool)
    for rg in obj.regions:  # coherent single source at texel level
        done = np.zeros(st.N, bool)
        for v in rg["views"]:
            take = rg["mask"] & ~done & st.valid[v]
            lab[take] = v
            done |= take
        forced |= done
    return np.where(obs, lab, -1), fallback, forced


def method_regions(obj: ObjectInput, prm: FusionParams) -> Result:
    from mvb2r.fusion import method_best
    from mvb2r.levels import feather_labels, level_gains
    st, tex = obj.st, obj.tex
    inc = consistency(st, prm)
    cid, nc, edges = cells_and_edges(tex, max(2, int(round(prm.cell * tex_scale(tex)))))
    D = data_costs(st, inc, cid, nc, prm, obj.regions)
    cell_lab, hist = alpha_expansion(D, edges, np.ones(len(edges)) * prm.lam)
    best = method_best(obj, prm)
    lab, fallback, forced = _texel_labels(obj, cell_lab, cid, best.source.astype(np.int64))
    gains = np.zeros((st.N, 3), np.float32)
    info = {"cells": int(nc), "edges": int(len(edges)), "energy_initial": hist[0], "energy_final": hist[-1]}
    if prm.level:
        region, g, linfo = level_gains(st, cid, cell_lab, edges, prm)
        use = (cell_lab[cid] < st.V) & ~fallback & ~forced
        gains = np.where(use[:, None], g[region[cid]], 0).astype(np.float32)
        info.update(linfo)
    labc = np.maximum(lab, 0)
    col = st.color[labc, np.arange(st.N)] * np.exp(gains)
    col, band = feather_labels(tex, st, lab, col, gains, forced, prm)
    conf = np.where(lab >= 0, st.score[labc, np.arange(st.N)], 0).astype(np.float32)
    return Result(color=col.astype(np.float32), source=lab.astype(np.int16), confidence=conf, observed=lab >= 0,
                  extra={"seam_band": band, "fallback": fallback, "gains": gains, "info": info})


# ----------------------------------------------------------------------------------------------- single source
def single_source_regions(tex, st, prefer: int | None = None, parts: list | None = None) -> list[dict]:
    """One region per 3D-connected mesh part (or per (name, mask) of `parts`); views ranked by the prefer_view,
    then by projected resolution (sum over the part's valid texels of area x ppt^2 = source pixels covering it).
    The legacy policy's regions; the strict policy uses handover.coherent_regions."""
    out = []
    if parts is None:
        parts = [(f"part_{int(p)}", tex.part == p) for p in np.unique(tex.part)]
    for name, m in parts:
        px = np.array([np.sum(tex.area[m] * st.ppt[v, m] ** 2 * st.valid[v, m]) / max(tex.area[m].sum(), 1e-12)
                       for v in range(st.V)])
        order = [int(v) for v in np.argsort(-px) if px[v] > 0]
        if prefer is not None and prefer in order:
            order = [prefer] + [v for v in order if v != prefer]
        out.append({"name": name, "mask": m, "views": order})
    return out


def agree_with(st, v, ref_log, gains, tau) -> np.ndarray:
    return st.valid[v] & (logdist(st.loglp[v] + gains, ref_log) <= tau)
