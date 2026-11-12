"""Shared synthetic fixtures for the fusion tests (pure numpy, no Blender)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "skills", "multiview-blockout-to-real", "scripts"))

import numpy as np  # noqa: E402

try:
    from _util import Skip  # noqa: F401  (core test helper)
except ImportError:  # pragma: no cover
    class Skip(Exception):
        pass


def quad(size=2.0, z=0.0):
    """Two triangles covering the square [-s/2, s/2]^2 at height z, UV = the full unit square, normal +Z."""
    h = size / 2
    P = np.array([[[-h, -h, z], [h, -h, z], [h, h, z]], [[-h, -h, z], [h, h, z], [-h, h, z]]], float)
    UV = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], float)
    N = np.tile([0, 0, 1.0], (2, 3, 1))
    return P, UV, N


def top_camera(name="view_01", W=128, height=3.0, x=0.0, y=0.0, f=None):
    """Camera at (x, y, height) looking straight down (-Z), image +Y = world -Y."""
    from mvb2r.pinhole import FLIP, Camera
    f = f or W * 0.9
    c2w = np.eye(4)
    c2w[:3, 3] = [x, y, height]  # Blender camera looks along its -Z: identity rotation looks down
    K = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1.0]])
    return Camera(name=name, width=W, height=W, K=K, w2c=FLIP @ np.linalg.inv(c2w))


def render_quad(cam, oid=1, z=0.0, size=2.0, colour_fn=None):
    """Exact depth / id / colour of the quad seen from a top camera (analytic ray-plane intersection)."""
    W, H = cam.width, cam.height
    j, i = np.mgrid[0:H, 0:W]
    x = (i + 0.5 - cam.K[0, 2]) / cam.K[0, 0]
    y = (j + 0.5 - cam.K[1, 2]) / cam.K[1, 1]
    C = cam.centre
    Zc = C[2] - z
    Xw = C[0] + x * Zc
    Yw = C[1] - y * Zc
    inside = (np.abs(Xw) <= size / 2) & (np.abs(Yw) <= size / 2)
    depth = np.where(inside, Zc, 1e10).astype(np.float32)
    ids = np.where(inside, oid, 0).astype(np.int32)
    img = None
    if colour_fn is not None:
        img = np.where(inside[..., None], colour_fn(Xw, Yw), 0).astype(np.float32)
    return depth, ids, img


def checker(X, Y, n=4):
    a = ((np.floor((X + 1) * n) + np.floor((Y + 1) * n)) % 2).astype(np.float32)
    return np.stack([0.2 + 0.5 * a, 0.3 + 0.2 * a, 0.4 - 0.2 * a], -1)


def dense_quad(n=60, size=2.0, uv_lo=0.0, uv_hi=1.0):
    """n x n grid quad (2 n^2 triangles), UV mapped to [uv_lo, uv_hi]^2 (sub-texel triangles at low R)."""
    h = size / 2
    g = np.linspace(-h, h, n + 1)
    t = np.linspace(uv_lo, uv_hi, n + 1)
    P, UV = [], []
    for i in range(n):
        for j in range(n):
            a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            for tri in ((a, b, c), (a, c, d)):
                P.append([[g[x], g[y], 0.0] for x, y in tri])
                UV.append([[t[x], t[y]] for x, y in tri])
    P, UV = np.array(P), np.array(UV)
    return P, UV, np.tile([0, 0, 1.0], (len(P), 3, 1))


def composite(layers):
    """Nearest-wins composite of render_quad outputs [(depth, ids, img), ...]."""
    depth = np.min([d for d, _, _ in layers], 0)
    pick = np.argmin([d for d, _, _ in layers], 0)
    ids = np.choose(pick, [i for _, i, _ in layers]).astype(np.int32)
    ids = np.where(depth >= 1e8, 0, ids)
    img = None
    if all(im is not None for _, _, im in layers):
        img = np.choose(pick[..., None], [im for _, _, im in layers]).astype(np.float32)
    return depth.astype(np.float32), ids, img


def shift_mask(m, dx=0, dy=0):
    """Mask shifted by whole pixels (misregistered generated mask)."""
    return np.roll(np.roll(m, dy, 0), dx, 1)


def two_object_case(W=128, n_views=2, shift=2, masks=("Card",)):
    """CaseInputs of a floor quad (id 1, z=0, 2 m) and a card quad (id 2, z=0.4, 0.6 m) above it, top cameras.
    Generated masks: the clay silhouette of each object named in `masks`, shifted by `shift` px."""
    from pathlib import Path
    from mvb2r.projinputs import CaseInputs
    from mvb2r.visibility import make_view
    cams = tuple(top_camera(f"view_0{k + 1}", W=W, x=x, height=2.6)
                 for k, x in enumerate((-0.3, 0.3, 0.0)[:n_views]))
    views, mk = [], {}
    for vi, cam in enumerate(cams):
        floor = render_quad(cam, oid=1, z=0.0, size=2.0, colour_fn=checker)
        card = render_quad(cam, oid=2, z=0.4, size=0.6, colour_fn=lambda X, Y: np.full(X.shape + (3,), 0.7, np.float32))
        depth, ids, img = composite([floor, card])
        views.append(make_view(cam, depth, ids, img))
        for name, oid in (("Floor", 1), ("Card", 2)):
            if name in masks:
                mk[(name, vi)] = shift_mask(ids == oid, shift, 0)
    objects = tuple({"name": n, "id": i, "file": f"{n}.npz", "class": "generic", "detail_critical": False,
                     "prefer_view": None, "opening": False} for n, i in (("Floor", 1), ("Card", 2)))
    return CaseInputs(case=Path("."), cfg={}, cams=cams, views=tuple(views), objects=objects, masks=mk,
                      mesh_dir=Path("."))


def with_meshes(ci, folder, card_regions=None, detail_critical=False):
    """`ci` with real mesh files (mesh_001.npz floor, mesh_002.npz card) in `folder` (a new temp dir)."""
    from dataclasses import replace
    from pathlib import Path
    folder = Path(folder)
    for k, (size, z) in enumerate(((2.0, 0.0), (0.6, 0.4)), start=1):
        P, UV, N = dense_quad(8, size)
        P = P + np.array([0, 0, z])
        np.savez(folder / f"mesh_{k:03d}.npz", tri_pos=P, tri_uv=UV, tri_nrm=N)
    objs = tuple({**o, "file": f"mesh_{o['id']:03d}.npz"} for o in ci.objects)
    objs = tuple({**o, "detail_critical": detail_critical, "regions": card_regions or ()} if o["name"] == "Card"
                 else o for o in objs)
    return replace(ci, objects=objs, mesh_dir=folder)
