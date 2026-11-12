"""Contact zones (EXPERIMENTAL, opt-in): where one object's surface really shows another object in the photos.

case.json `fusion.contacts`: [{"surface": "<object>", "occluder": "<object>", "within_mm": 20,
optional "box": [[x0,y0,z0],[x1,y1,z1]]}] (metres for the box).

Surface pixels whose 3D point lies within `within_mm` of the occluder's mesh (and inside `box` when given) ACCEPT
the occluder's generated-mask pixels (that mask eroded 1 px) instead of rejecting them (gate_policy "strict").
After the fill, unobserved surface texels in the zone are coloured with the occluder's median observed colour,
weight 1 within 0.75 x within_mm falling to 0 at 1.25 x within_mm. Those texels stay INFERRED: they are never
counted as observed in any coverage map or number.

Why it exists: under strict gating, skin under a hairline that only ever saw hair pixels was rejected and filled
with skin colour, which rendered as a pink band under the hair (first run, references/lessons.md).
Why `box` exists: without a spatial limit the first attempt painted dark bands down the cheek, because the hair
hangs within 20 mm of it; the zone had to be limited to the scalp edge.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mvb2r.fusion_io import StageError
from mvb2r.visibility import BACKGROUND_DEPTH
from mvb2r.shapes import box, capped_list, finite, inside_box, label, show

DEFAULT_WITHIN_MM = 20.0
MM = 1e-3
RAMP_IN, RAMP_OUT = 0.75, 1.25      # colouring weight 1 inside RAMP_IN x within, 0 beyond RAMP_OUT x within
OCCLUDER_POINTS_R = 1024            # occluder mesh sampled as the texel centres of a 1024^2 atlas
KEYS = {"surface", "occluder", "within_mm", "box"}
MAX_WITHIN_MM = 1000.0              # contact distance limit


def _parse_one(c, i: int, names) -> dict:
    where = f"fusion.contacts[{i}]"
    if not isinstance(c, dict):
        raise StageError(f"case.json {where} must be an object, got {show(c)}")
    if set(c) - KEYS:
        raise StageError(f"case.json {where}: unknown key(s) {show(sorted(map(str, set(c) - KEYS)))}")
    for k in ("surface", "occluder"):
        if c.get(k) not in names:
            raise StageError(f"case.json {where}.{k} {show(c.get(k))} is not an object "
                             f"({show(sorted(label(n) for n in names))})")
    if c["surface"] == c["occluder"]:
        raise StageError(f"case.json {where}: surface and occluder must differ")
    return {"surface": c["surface"], "occluder": c["occluder"],
            "within_mm": finite(f"{where}.within_mm", c.get("within_mm", DEFAULT_WITHIN_MM), positive=True,
                                maximum=MAX_WITHIN_MM),
            "box": box(f"{where}.box", c["box"]) if c.get("box") is not None else None}


def parse_contacts(contacts, names) -> tuple:
    """Validated fusion.contacts (empty tuple when absent)."""
    if contacts is None:
        return ()
    contacts = capped_list("fusion.contacts", contacts)
    return tuple(_parse_one(c, i, set(names)) for i, c in enumerate(contacts))


def surface_points(view) -> np.ndarray:
    """(H,W,3) world point of every pixel's clay surface (back-projected camera-axis depth; nan on background)."""
    cam = view.cam
    H, W = view.depth.shape
    j, i = np.mgrid[0:H, 0:W]
    Z = np.where(view.depth < BACKGROUND_DEPTH, view.depth, np.nan).astype(np.float64)
    Y = (j + 0.5 - cam.K[1, 2]) * Z / cam.fy
    X = (i + 0.5 - cam.K[0, 2] - cam.K[0, 1] * Y / Z) * Z / cam.fx
    Xc = np.stack([X, Y, Z], -1)
    Rm, t = cam.w2c[:3, :3], cam.w2c[:3, 3]
    return (Xc - t) @ Rm


def occluder_points(ci, name: str) -> np.ndarray:
    """(M,3) dense surface samples of an object's exported mesh."""
    from mvb2r.fusion_io import load_mesh
    from mvb2r.raster import texel_set
    row = next(o for o in ci.objects if o["name"] == name)
    z = load_mesh(ci.mesh_dir, row["file"])
    return texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], OCCLUDER_POINTS_R).P


def _near(P: np.ndarray, tree: cKDTree, dist_m: float) -> np.ndarray:
    d, _ = tree.query(P, k=1, distance_upper_bound=dist_m)
    return np.isfinite(d)


def contact_pixels(ci, obj: dict, contacts: tuple, points_of=None) -> list:
    """Per view {occluder name: (H,W) bool zone pixels} for `obj` as a contact surface ([{}...] when it is none).
    `points_of(name)` -> (M,3) occluder points (default: occluder_points from the exported mesh)."""
    mine = [c for c in contacts if c["surface"] == obj["name"]]
    out = [dict() for _ in ci.views]
    if not mine:
        return out
    points_of = points_of or (lambda n: occluder_points(ci, n))
    for c in mine:
        tree = cKDTree(points_of(c["occluder"]))
        for vi, view in enumerate(ci.views):
            own = view.ids == obj["id"]
            idx = np.flatnonzero(own)
            P = surface_points(view).reshape(-1, 3)[idx]
            hit = _near(P, tree, c["within_mm"] * MM) & inside_box(P, c["box"])
            zone = np.zeros(own.shape, bool)
            zone.ravel()[idx[hit]] = True
            prev = out[vi].get(c["occluder"])
            out[vi] = {**out[vi], c["occluder"]: zone if prev is None else (prev | zone)}
    return out


def contact_weights(P: np.ndarray, occ_points: np.ndarray, within_m: float, limit: tuple | None) -> np.ndarray:
    """(N,) colouring weight: 1 within RAMP_IN x within_m of the occluder, 0 from RAMP_OUT x, 0 outside `limit`."""
    lo, hi = RAMP_IN * within_m, RAMP_OUT * within_m
    d, _ = cKDTree(occ_points).query(P, k=1, distance_upper_bound=hi)
    d = np.where(np.isfinite(d), d, hi)
    w = np.clip((hi - d) / (hi - lo), 0, 1) * inside_box(P, limit)
    return w.astype(np.float32)


def contact_fill(img: np.ndarray, tex, observed: np.ndarray, weights: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """New (R,R,3) texture: unobserved texels blended towards `colour` by `weights`; observed texels unchanged."""
    w = np.where(observed, 0, weights).astype(np.float32)
    out = img.astype(np.float32).copy()
    cur = out[tex.rows, tex.cols]
    out[tex.rows, tex.cols] = cur * (1 - w[:, None]) + np.asarray(colour, np.float32)[None] * w[:, None]
    return out
