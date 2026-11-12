"""Vectorised UV-atlas rasteriser: texel -> (triangle, barycentrics, world position, normals, area).

Texel (row j, col i) of an R x R texture has its centre at uv = ((i+0.5)/R, 1-(j+0.5)/R): PNG row 0 is the top
of the image, which Blender maps to v = 1. A texel belongs to a triangle when its centre lies inside the
triangle's UV footprint (closed edges; on a shared edge the later triangle wins). Triangles are processed in
size buckets so every numpy batch stays within a fixed memory budget. Gutter texels: every uncovered texel
within `gutter` texels (default 2) of a triangle gets the nearest triangle with clamped barycentrics
(gutter_assign), so sub-texel triangles / islands and island borders carry real surface samples (bleed padding)
instead of fill; they are flagged core = False and weigh zero in area statistics. Ported from the experiment
(scripts/proj/raster.py); the mesh comes from blender/export_mesh.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_BUDGET = 4_000_000  # candidate texels evaluated per numpy batch


def uv_to_texel_xy(tri_uv: np.ndarray, R: int) -> np.ndarray:
    """(T,3,2) uv in [0,1] -> (T,3,2) continuous texel coords (x right, y down)."""
    uv = np.asarray(tri_uv, np.float64)
    return np.stack([uv[..., 0] * R, (1.0 - uv[..., 1]) * R], -1)


def barycentric_2d(px, py, a, b, c):
    """Barycentrics of points (px, py) with respect to triangles a, b, c (broadcast, last axis = xy)."""
    v0x, v0y = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    v1x, v1y = c[..., 0] - a[..., 0], c[..., 1] - a[..., 1]
    v2x, v2y = px - a[..., 0], py - a[..., 1]
    d = v0x * v1y - v1x * v0y
    ds = np.where(np.abs(d) > 1e-14, d, 1e-14)
    l1 = (v2x * v1y - v1x * v2y) / ds
    l2 = (v0x * v2y - v2x * v0y) / ds
    return 1.0 - l1 - l2, l1, l2


def _bounds(xy: np.ndarray, R: int):
    x0 = np.maximum(np.ceil(xy[..., 0].min(1) - 0.5), 0).astype(np.int64)
    x1 = np.minimum(np.floor(xy[..., 0].max(1) - 0.5), R - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(xy[..., 1].min(1) - 0.5), 0).astype(np.int64)
    y1 = np.minimum(np.floor(xy[..., 1].max(1) - 0.5), R - 1).astype(np.int64)
    return x0, x1, y0, y1


def rasterise(tri_xy: np.ndarray, R: int, eps: float = 1e-9):
    """tri_xy (T,3,2) texel coords -> (tmap (R,R) int32, -1 = empty; bary (R,R,3) float32)."""
    if R < 1:
        raise ValueError("texture size must be >= 1")
    tmap = np.full((R, R), -1, np.int32)
    bary = np.zeros((R, R, 3), np.float32)
    xy = np.asarray(tri_xy, np.float64)
    if len(xy) == 0:
        return tmap, bary
    d = (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - \
        (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    x0, x1, y0, y1 = _bounds(xy, R)
    live = (x1 >= x0) & (y1 >= y0) & (np.abs(d) > 1e-14)
    S = np.maximum(x1 - x0, y1 - y0) + 1
    bucket = np.where(live, np.ceil(np.log2(np.maximum(S, 1))).astype(np.int64), -1)
    for bk in np.unique(bucket[bucket >= 0]):
        ids = np.nonzero(bucket == bk)[0]
        _raster_bucket(xy, ids, int(2 ** bk), (x0, x1, y0, y1), tmap, bary, eps)
    return tmap, bary


def _raster_bucket(xy, ids, s, box, tmap, bary, eps):
    x0, x1, y0, y1 = box
    rows = max(1, min(s, _BUDGET // s))  # sub-tile rows so huge triangles stay within the budget
    for oy0 in range(0, s, rows):
        oy, ox = np.mgrid[oy0:min(s, oy0 + rows), 0:s]
        ox, oy = ox.ravel(), oy.ravel()
        step = max(1, _BUDGET // len(ox))
        for k in range(0, len(ids), step):
            tt = ids[k:k + step]
            X = x0[tt, None] + ox[None]
            Y = y0[tt, None] + oy[None]
            inb = (X <= x1[tt, None]) & (Y <= y1[tt, None])
            a, b, c = xy[tt, 0][:, None], xy[tt, 1][:, None], xy[tt, 2][:, None]
            l0, l1, l2 = barycentric_2d(X + 0.5, Y + 0.5, a, b, c)
            r, q = np.nonzero(inb & (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps))
            tmap[Y[r, q], X[r, q]] = tt[r]
            bary[Y[r, q], X[r, q]] = np.stack([l0[r, q], l1[r, q], l2[r, q]], -1)


@dataclass(frozen=True)
class TexelSet:
    """Covered texels of one object's R x R atlas (N texels)."""
    R: int
    rows: np.ndarray     # (N,) int32
    cols: np.ndarray     # (N,) int32
    tri: np.ndarray      # (N,) int32 triangle index
    P: np.ndarray        # (N,3) float64 world position
    Ns: np.ndarray       # (N,3) float32 interpolated world normal
    Ng: np.ndarray       # (N,3) float32 geometric world normal
    area: np.ndarray     # (N,) float32 world area represented by the texel (m^2)
    part: np.ndarray     # (N,) int32 3D-connected mesh component of the texel's triangle
    core: np.ndarray | None = None  # (N,) bool: centre inside its triangle (False = gutter / sub-texel bleed)
    near: np.ndarray | None = None  # (N,) int32 index of the nearest core texel (itself when core)

    @property
    def weight(self) -> np.ndarray:
        """Area weight for statistics: gutter texels (padding) count zero."""
        return self.area * (self.core if self.core is not None else 1.0)

    @property
    def n(self) -> int:
        return len(self.rows)

    def scatter(self, values: np.ndarray, fill=0, dtype=None) -> np.ndarray:
        """Per-texel values (N,...) -> (R,R,...) image; uncovered texels get `fill`."""
        values = np.asarray(values)
        out = np.full((self.R, self.R) + values.shape[1:], fill, dtype or values.dtype)
        out[self.rows, self.cols] = values
        return out

    def gather(self, img: np.ndarray) -> np.ndarray:
        return np.asarray(img)[self.rows, self.cols]

    def mask(self) -> np.ndarray:
        m = np.zeros((self.R, self.R), bool)
        m[self.rows, self.cols] = True
        return m


def triangle_geometry(tri_pos: np.ndarray, tri_uv: np.ndarray, tri_nrm: np.ndarray | None = None):
    """Geometric normals (oriented like the shading normals), world area and uv area per triangle."""
    tri_pos = np.asarray(tri_pos, np.float64)
    cr = np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0])
    area_w = 0.5 * np.linalg.norm(cr, axis=1)
    ng = cr / np.maximum(2 * area_w, 1e-20)[:, None]
    if tri_nrm is not None:
        flip = np.sum(ng * np.asarray(tri_nrm).mean(1), 1) < 0
        ng = np.where(flip[:, None], -ng, ng)
    uv = np.asarray(tri_uv, np.float64)
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    area_uv = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return ng, area_w, area_uv


def mesh_components(tri_pos: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Connected components of triangles that share a vertex position (welded at 10^-decimals m)."""
    T = len(tri_pos)
    if T == 0:
        return np.zeros(0, np.int32)
    key = np.round(np.asarray(tri_pos, np.float64).reshape(-1, 3), decimals)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(T, 3)
    nv = int(vid.max()) + 1
    rows = np.repeat(np.arange(T), 3)
    G = coo_matrix((np.ones(3 * T), (rows, vid.ravel() + T)), shape=(T + nv, T + nv))
    _, lab = connected_components(G, directed=False)
    _, comp = np.unique(lab[:T], return_inverse=True)
    return comp.astype(np.int32)


def _point_tri_dist(px, py, xy, t):
    """Clamped barycentrics of points in triangles t and the distance to the clamped point (texel units)."""
    a, b, c = xy[t, 0], xy[t, 1], xy[t, 2]
    l = np.stack(barycentric_2d(px, py, a, b, c), -1)
    l = np.clip(l, 0, None)
    l /= np.maximum(l.sum(-1, keepdims=True), 1e-12)
    q = np.einsum("...k,...kj->...j", l, xy[t])
    return l, np.hypot(q[..., 0] - px, q[..., 1] - py)


def gutter_assign(tmap: np.ndarray, bary: np.ndarray, xy: np.ndarray, gutter: float = 2.0, k: int = 8):
    """Give every uncovered texel within `gutter` texels of a triangle that triangle (clamped barycentrics):
    sub-texel triangles / islands and island borders get real surface samples instead of fill (bleed padding).
    Returns (tmap, bary, core) as new arrays."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    R = tmap.shape[0]
    core = tmap >= 0
    if not core.any() or len(xy) == 0:
        return tmap.copy(), bary.copy(), core
    seed = np.zeros((R, R), bool)
    pts = np.concatenate([xy.reshape(-1, 2), xy.mean(1)])
    ij = np.clip(np.floor(pts).astype(int), 0, R - 1)
    seed[ij[:, 1], ij[:, 0]] = True
    reach = ndimage.binary_dilation(seed | core, iterations=int(np.ceil(gutter)) + 1)
    rows, cols = np.nonzero(reach & ~core)
    if not len(rows):
        return tmap.copy(), bary.copy(), core
    _, (nr, nc) = ndimage.distance_transform_edt(~core, return_indices=True)
    px, py = cols + 0.5, rows + 0.5
    _, cand = cKDTree(xy.mean(1)).query(np.c_[px, py], k=min(k, len(xy)))
    cand = np.c_[cand.reshape(len(px), -1), tmap[nr[rows, cols], nc[rows, cols]]]
    lam, dist = _point_tri_dist(px[:, None], py[:, None], xy, cand)
    best = np.argmin(dist, 1)
    ok = dist[np.arange(len(px)), best] <= gutter
    tm, ba = tmap.copy(), bary.copy()
    tm[rows[ok], cols[ok]] = cand[np.arange(len(px)), best][ok]
    ba[rows[ok], cols[ok]] = lam[np.arange(len(px)), best][ok]
    return tm, ba, core


def texel_set(tri_pos: np.ndarray, tri_uv: np.ndarray, tri_nrm: np.ndarray, R: int, gutter: float = 2.0) -> TexelSet:
    tri_pos = np.asarray(tri_pos, np.float64)
    if tri_pos.ndim != 3 or tri_pos.shape[1:] != (3, 3):
        raise ValueError(f"tri_pos must be (T,3,3), got {tri_pos.shape}")
    xy = uv_to_texel_xy(tri_uv, R)
    tmap, bary = rasterise(xy, R)
    tmap, bary, core_img = gutter_assign(tmap, bary, xy, gutter) if gutter > 0 else (tmap, bary, tmap >= 0)
    rows, cols = np.nonzero(tmap >= 0)
    t = tmap[rows, cols]
    b = bary[rows, cols].astype(np.float64)
    P = np.einsum("nk,nkj->nj", b, tri_pos[t])
    Ns = np.einsum("nk,nkj->nj", b, np.asarray(tri_nrm, np.float64)[t])
    Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
    ng, area_w, area_uv = triangle_geometry(tri_pos, tri_uv, tri_nrm)
    texel_area = area_w / np.maximum(area_uv * R * R, 1e-20)
    comp = mesh_components(tri_pos)
    return TexelSet(R=R, rows=rows.astype(np.int32), cols=cols.astype(np.int32), tri=t.astype(np.int32), P=P,
                    Ns=Ns.astype(np.float32), Ng=ng[t].astype(np.float32), area=texel_area[t].astype(np.float32),
                    part=comp[t] if len(comp) else np.zeros(len(t), np.int32), core=core_img[rows, cols],
                    near=_nearest_core(core_img, rows, cols))


def _nearest_core(core_img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    idx = np.full(core_img.shape, -1, np.int64)
    idx[rows, cols] = np.arange(len(rows))
    if not core_img.any():
        return np.arange(len(rows), dtype=np.int32)
    _, (nr, nc) = ndimage.distance_transform_edt(~core_img, return_indices=True)
    return idx[nr[rows, cols], nc[rows, cols]].astype(np.int32)

