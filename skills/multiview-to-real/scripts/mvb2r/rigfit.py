"""Rig fit: scale the scene's own lights so low-res renders of final.blend match target luminance.

Compared on well-observed object pixels (eroded, objects with observed area >= 0.3): median and p90 of the
render / target luminance ratio. One global light scale (exposure kept at 0); accepted when both ratios are
within +-10 %. A sun-direction search is NOT implemented (reported as such).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion_io import lum, read_ids, read_rgb_linear

TOL = 0.10


def ratios(render_dir: Path, case: Path, views: list, good_ids: list) -> dict:
    meds, p90s = [], []
    for v in views:
        R = read_rgb_linear(render_dir / f"{v}.png")
        T = cv2.resize(read_rgb_linear(case / "targets" / f"{v}.png"), R.shape[1::-1], interpolation=cv2.INTER_AREA)
        ids = cv2.resize(read_ids(case / "views" / v / "object_id.png").astype(np.float32), R.shape[1::-1],
                         interpolation=cv2.INTER_NEAREST).astype(np.int32)
        m = cv2.erode(np.isin(ids, good_ids).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if m.sum() < 50:
            continue
        lr, lt = lum(R)[m], lum(T)[m]
        meds.append(np.median(lr) / max(np.median(lt), 1e-6))
        p90s.append(np.percentile(lr, 90) / max(np.percentile(lt, 90), 1e-6))
    if not meds:
        return {"median_ratio": None, "p90_ratio": None, "views": 0}
    return {"median_ratio": float(np.median(meds)), "p90_ratio": float(np.median(p90s)), "views": len(meds)}


def within(r: dict, tol: float = TOL) -> bool:
    return r["median_ratio"] is not None and abs(r["median_ratio"] - 1) <= tol and abs(r["p90_ratio"] - 1) <= tol


def next_scale(scale: float, r: dict) -> float:
    """Linear light: render luminance is proportional to the light scale (sRGB clipping aside)."""
    if r["median_ratio"] is None:
        return scale
    return float(np.clip(scale / max(np.sqrt(r["median_ratio"] * r["p90_ratio"]), 1e-3), scale / 8, scale * 8))
