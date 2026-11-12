"""Unlit texture renderer from exact per-pixel UV passes (blender/render.py --passes).

A pixel of object k shows texture_k bilinearly sampled at the pixel's UV (Blender 'Linear' interpolation,
texel-centre convention of raster.py): an emission render at 1 sample per pixel, computed in numpy.
Used for consensus re-renders, novel views, leave-one-out images, seam visibility and the coverage-tinted tour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvb2r.fusion_io import StageError, read_exr, read_ids


@dataclass(frozen=True)
class Passes:
    ids: np.ndarray    # (H,W) int32, 0 = background
    uv: np.ndarray     # (H,W,2) float32 (-1 where background)
    depth: np.ndarray  # (H,W) float32 camera-axis Z (1e10 background)


def load_passes(folder: Path) -> Passes:
    folder = Path(folder)
    for f in ("object_id.png", "uv.exr", "depth_z.exr"):
        if not (folder / f).is_file():
            raise StageError(f"missing {folder / f} (rendered by `mvb2r.py project`)")
    ids = read_ids(folder / "object_id.png")
    uv = read_exr(folder / "uv.exr")[..., :2]
    depth = read_exr(folder / "depth_z.exr")
    depth = depth[..., 0] if depth.ndim == 3 else depth
    if uv.shape[:2] != ids.shape or depth.shape != ids.shape:
        raise StageError(f"{folder}: pass sizes disagree {ids.shape} {uv.shape} {depth.shape}")
    return Passes(ids=ids, uv=uv.astype(np.float32), depth=depth.astype(np.float32))


def tex_sample(img: np.ndarray, uv: np.ndarray, nearest: bool = False, valid: np.ndarray | None = None) -> np.ndarray:
    """Bilinear (or nearest) texture lookup. With `valid` (R,R bool, the atlas texels), taps outside it get zero
    weight (renormalised) so a lookup at an island border never reads fill; all-invalid -> nearest tap."""
    if valid is not None and not nearest:
        return _masked_bilinear(img, uv, valid)
    R = img.shape[0]
    x = uv[:, 0] * R - 0.5
    y = (1 - uv[:, 1]) * R - 0.5
    if nearest:
        return img[np.clip(np.rint(y).astype(int), 0, R - 1), np.clip(np.rint(x).astype(int), 0, R - 1)]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    im = img if img.ndim == 3 else img[..., None]
    ax, ay = ax[:, None], ay[:, None]
    x0c, x1c = np.clip(x0, 0, R - 1), np.clip(x0 + 1, 0, R - 1)
    y0c, y1c = np.clip(y0, 0, R - 1), np.clip(y0 + 1, 0, R - 1)
    o = (im[y0c, x0c] * (1 - ax) + im[y0c, x1c] * ax) * (1 - ay) + (im[y1c, x0c] * (1 - ax) + im[y1c, x1c] * ax) * ay
    return o if img.ndim == 3 else o[:, 0]


def _masked_bilinear(img, uv, valid):
    R = img.shape[0]
    im = img if img.ndim == 3 else img[..., None]
    x = uv[:, 0] * R - 0.5
    y = (1 - uv[:, 1]) * R - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    acc = np.zeros((len(x), im.shape[2]), np.float32)
    ws = np.zeros(len(x), np.float32)
    for dx, dy, w in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        xi, yi = np.clip(x0 + dx, 0, R - 1), np.clip(y0 + dy, 0, R - 1)
        w = w * valid[yi, xi]
        acc += w[:, None] * im[yi, xi]
        ws += w
    nn = im[np.clip(np.rint(y).astype(int), 0, R - 1), np.clip(np.rint(x).astype(int), 0, R - 1)]
    out = np.where((ws > 1e-6)[:, None], acc / np.maximum(ws, 1e-6)[:, None], nn)
    return out if img.ndim == 3 else out[:, 0]


def render(passes: Passes, textures: dict, background=(0.0, 0.0, 0.0), flat: dict | None = None,
           nearest: bool = False, valid: dict | None = None):
    """textures {object id: (R,R,C)}; flat {object id: colour} (e.g. openings). Returns (image, drawn mask)."""
    H, W = passes.ids.shape
    C = next(iter(textures.values())).shape[2] if textures and next(iter(textures.values())).ndim == 3 else \
        len(background)
    out = np.empty((H * W, C), np.float32)
    out[:] = np.asarray(background, np.float32)[:C]
    drawn = np.zeros(H * W, bool)
    ids = passes.ids.ravel()
    uv = passes.uv.reshape(-1, 2)
    for oid, img in textures.items():
        pix = np.nonzero(ids == oid)[0]
        if len(pix):
            v = tex_sample(img, uv[pix], nearest, (valid or {}).get(oid))
            out[pix] = v if v.ndim == 2 else v[:, None]
            drawn[pix] = True
    for oid, col in (flat or {}).items():
        pix = ids == oid
        out[pix] = np.asarray(col, np.float32)[:C]
    return out.reshape(H, W, C), drawn.reshape(H, W)
