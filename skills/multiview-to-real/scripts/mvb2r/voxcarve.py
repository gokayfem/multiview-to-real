"""Silhouette carving of one mesh to its generated masks (pure numpy / scipy / scikit-image).

Generic port of the experiment's hair carve (scripts/scene_v2/hair_carve_v2b.py):
  1. voxel grid over the object's bounding box grown by the outer offset; signed distance to the existing mesh
     from dense surface samples (KD-tree, sign from the nearest sample's normal);
  2. candidates: sd in (-inner, +outer) ("shell": a layer like hair over a head) or sd < +outer ("solid");
  3. per view, a candidate is REMOVED when it is in frame, the camera ray reaches it without hitting OTHER
     geometry first (first-surface test on the view's depth_z / object_id passes) and it projects outside the
     object's mask dilated by `dilate` px, or onto another object's generated mask (outside its own);
  4. largest 6-connected piece, Gaussian-smoothed occupancy, marching cubes (padded grid -> closed surface),
     Laplacian smoothing. Decimation and UV unwrap happen in Blender (blender/carve.py).
Occlusion uses only the first visible surface of each view: geometry hidden behind the oversized object
itself is not known (documented limit).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

REF_SIZE = 0.36          # experiment: hair box ~0.36 m, voxel 2.5 mm, -12 mm ... +35 mm
REF_VOXEL, REF_IN, REF_OUT = 0.0025, 0.012, 0.035


@dataclass(frozen=True)
class CarveParams:
    voxel: float
    inner: float
    outer: float
    dilate: int = 4
    mode: str = "solid"          # or "shell"
    max_axis: int = 160
    iso: float = 0.5
    smooth_iters: int = 10


def scaled_params(tri_pos: np.ndarray, mode: str = "solid", dilate: int = 4, max_axis: int = 160) -> CarveParams:
    """Experiment defaults scaled by the object's size (bounding-box diagonal / 0.36 m), voxel count capped."""
    P = tri_pos.reshape(-1, 3)
    size = float(np.linalg.norm(P.max(0) - P.min(0)))
    s = max(size / REF_SIZE, 1e-3)
    inner, outer = REF_IN * s, REF_OUT * s
    ext = (P.max(0) - P.min(0)) + 2 * outer
    voxel = max(REF_VOXEL * s, float(ext.max()) / max_axis)
    return CarveParams(voxel=voxel, inner=inner, outer=outer, dilate=dilate, mode=mode, max_axis=max_axis)


def surface_samples(tri_pos: np.ndarray, spacing: float, cap: int = 2_000_000, seed: int = 0):
    """Area-proportional random surface points + geometric normals (outward for consistently wound meshes)."""
    t = np.asarray(tri_pos, np.float64)
    cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = int(np.clip(area.sum() / max(spacing, 1e-9) ** 2, len(t), cap))
    rng = np.random.default_rng(seed)
    k = rng.choice(len(t), n, p=area / max(area.sum(), 1e-20))
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    pts = np.einsum("nk,nkj->nj", b, t[k])
    nrm = cr[k] / np.maximum(2 * area[k], 1e-20)[:, None]
    return np.vstack([pts, t.reshape(-1, 3)]), np.vstack([nrm, np.repeat(cr / np.maximum(2 * area, 1e-20)[:, None], 3, 0)])


def grid(tri_pos: np.ndarray, prm: CarveParams):
    P = tri_pos.reshape(-1, 3)
    lo = P.min(0) - prm.outer - 2 * prm.voxel
    n = np.ceil((P.max(0) + prm.outer + 2 * prm.voxel - lo) / prm.voxel).astype(int) + 1
    axes = [lo[a] + prm.voxel * np.arange(n[a]) for a in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return G, lo, tuple(int(x) for x in n)


def signed_distance(points: np.ndarray, tri_pos: np.ndarray, spacing: float) -> np.ndarray:
    S, N = surface_samples(tri_pos, spacing)
    d, i = cKDTree(S).query(points, k=1)
    sign = np.sign(np.sum((points - S[i]) * N[i], 1))
    return d * np.where(sign == 0, 1, sign)


def candidates(sd: np.ndarray, prm: CarveParams) -> np.ndarray:
    if prm.mode == "shell":
        return (sd > -prm.inner) & (sd < prm.outer)
    if prm.mode != "solid":
        raise ValueError(f"carve mode must be 'solid' or 'shell', got {prm.mode!r}")
    return sd < prm.outer


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0 if r > 0 else m.astype(bool)


def carve_view(points, keep, cam, own_mask, other_mask, depth, ids, oid, prm: CarveParams, tol: float):
    """Indices of kept voxels removed by one view (pure decision; see module doc)."""
    idx = np.nonzero(keep)[0]
    u, v, z = cam.project(points[idx])
    H, W = own_mask.shape
    inf = np.isfinite(u) & (z > cam.clip_start) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx, u, v, z = idx[inf], u[inf], v[inf], z[inf]
    i, j = u.astype(int), v.astype(int)
    pid = ids[j, i]
    blocked = (pid != oid) & (pid != 0) & (depth[j, i] < z - tol)   # other geometry first on the ray
    own = dilate(own_mask, prm.dilate)
    bad = ~own | (other_mask & ~own_mask)
    return idx[~blocked & bad[j, i]]


def largest_component(occ: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(occ)
    if n <= 1:
        return occ
    sizes = ndimage.sum(occ, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def mesh_from_occupancy(occ: np.ndarray, lo: np.ndarray, prm: CarveParams):
    """Closed triangle mesh (world coords) from the occupancy grid; None when nothing survives."""
    from skimage.measure import marching_cubes
    if occ.sum() < 8:
        return None
    f = ndimage.gaussian_filter(np.pad(occ.astype(np.float32), 2), 1.0)
    verts, faces, _, _ = marching_cubes(f, prm.iso)
    verts = laplacian_smooth((verts - 2) * prm.voxel + lo, faces, prm.smooth_iters)
    return verts, faces.astype(np.int64)


def laplacian_smooth(V: np.ndarray, F: np.ndarray, iters: int, lam: float = 0.5) -> np.ndarray:
    from scipy.sparse import coo_matrix
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    out = V.copy()
    for _ in range(iters):
        out = out + lam * (A @ out / deg[:, None] - out)
    return out


def is_closed(F: np.ndarray) -> bool:
    """Every edge shared by exactly two faces (watertight, manifold edges)."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(len(cnt) and np.all(cnt == 2))


def mask_metrics(ids: np.ndarray, oid: int, own_mask: np.ndarray, other_mask: np.ndarray) -> dict:
    obj = ids == oid
    inter, union = (obj & own_mask).sum(), (obj | own_mask).sum()
    return {"iou": float(inter / union) if union else None, "outside_mask_px": int((obj & ~own_mask).sum()),
            "covers_other_masks_px": int((obj & other_mask & ~own_mask).sum()), "object_px": int(obj.sum())}
