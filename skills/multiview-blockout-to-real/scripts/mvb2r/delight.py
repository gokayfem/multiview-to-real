"""Conservative de-lighting of the targets against the scene's own shading basis (keeps the projection).

Per view and object: relative proxy shading rel = LP(S_sun + S_sky) / median over the object; the target is
multiplied by a SMOOTH, CLAMPED scalar gain clip(1/rel, gain_min, gain_max) (heavily low-passed, colour
untouched). Detail and colour of the generated image survive; some soft baked shading remains and is reported
(pbr/eval.json). A view whose log-luminance correlation with the proxy shading is below
min_corr (0.3) is left unchanged: de-lighting a target without that shading would bake in inverse shading
(observed on unlit synthetic and mock targets: correlation 0.0-0.2 became -0.2 to -0.36). Priors never replace measured texture here (see materials.py).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mvb2r.common import owned_path
from mvb2r.fusion_io import StageError, lum, read_exr, read_ids, read_rgb_linear, write_srgb_png
from mvb2r.visibility import masked_lowpass


def shading(shading_dir: Path, view: str, shape) -> np.ndarray:
    sun, sky = shading_dir / f"{view}_sun.exr", shading_dir / f"{view}_sky.exr"
    if not sun.is_file() or not sky.is_file():
        raise StageError(f"missing {sun} / {sky} (written by `mvb2r.py pbr`, shading step)")
    S = lum(read_exr(sun)) + lum(read_exr(sky))
    return cv2.resize(S.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def gain_map(S: np.ndarray, ids: np.ndarray, sigma: float, gmin: float, gmax: float) -> np.ndarray:
    """Smooth clamped de-light gain per pixel (1 on background)."""
    lp = masked_lowpass(np.repeat(S[..., None], 3, -1), ids, sigma)[..., 0]
    gain = np.ones_like(S)
    for oid in np.unique(ids[ids > 0]):
        m = ids == oid
        med = float(np.median(lp[m]))
        if m.sum() < 20 or med <= 1e-6:
            continue
        gain[m] = np.clip(med / np.maximum(lp[m], 1e-6), gmin, gmax)
    return masked_lowpass(np.repeat(gain[..., None], 3, -1), ids, sigma / 2)[..., 0]


def delight_view(case: Path, view: str, shading_dir: Path, out: Path, gmin=0.7, gmax=1.6,
                 min_corr: float = 0.3) -> dict:
    T = read_rgb_linear(case / "targets" / f"{view}.png")
    ids = read_ids(case / "views" / view / "object_id.png")
    if ids.shape != T.shape[:2]:
        ids = cv2.resize(ids.astype(np.float32), T.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.int32)
    S = shading(shading_dir, view, T.shape[:2])
    m = ids > 0
    raw = _corr(lum(T)[m], S[m])
    applied = raw >= min_corr  # a target that does not show the proxy shading is left alone (no inverse shading)
    g = gain_map(S, ids, sigma=max(T.shape[:2]) / 40.0, gmin=gmin, gmax=gmax) if applied else np.ones_like(S)
    A = np.clip(T * g[..., None], 0, 1)
    write_srgb_png(owned_output(case, out, view), A, bits=16)
    return {"applied": bool(applied), "gain_p05": float(np.percentile(g[m], 5)),
            "gain_p95": float(np.percentile(g[m], 95)), "corr_raw": raw, "corr_delit": _corr(lum(A)[m], S[m])}


def owned_output(case: Path, out: Path, view: str) -> Path:
    """The de-lit image path, refused when it or a folder above it inside the case is a symbolic link."""
    return owned_path(case, Path(out) / f"{view}.png")


def _corr(a, b) -> float:
    a, b = np.log(np.maximum(a, 1e-4)), np.log(np.maximum(b, 1e-4))
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / max(np.sqrt(np.sum(a * a) * np.sum(b * b)), 1e-12))
