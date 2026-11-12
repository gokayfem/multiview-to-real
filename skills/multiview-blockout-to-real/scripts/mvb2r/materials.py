"""PBR material textures from the de-lit projection: base colour = projected texture x smooth clamped correction.

Priors (material_priors.json) only (1) bound each object's MEAN albedo luminance to its class range by one
scalar per object and (2) give roughness / specular / bump; unobserved texels keep the same-object fill
(plain colour). The measured texture is never replaced by a prior (the experiment's first version did and
looked like plastic).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion_io import StageError, lum, read_gray01, read_rgb_linear, safe_name, write_png, write_srgb_png

PRIORS = Path(__file__).resolve().parent / "material_priors.json"


def load_priors(path: Path = PRIORS) -> dict:
    data = json.loads(Path(path).read_text())
    if "generic" not in data.get("classes", {}):
        raise StageError(f"{path}: needs a 'generic' class")
    return data["classes"]


def class_prior(cls: str, priors: dict) -> dict:
    return priors.get(cls, priors["generic"])


def bound_mean(A: np.ndarray, obs: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, float]:
    """Scale the texture so the observed mean luminance lies in [lo, hi] (scale 1 when already inside)."""
    m = float(np.median(lum(A[obs]))) if obs.any() else float(np.median(lum(A)))
    s = float(np.clip(m, lo, hi) / max(m, 1e-6))
    return np.clip(A * s, 0, 1), s


def height_from(A: np.ndarray, R: int) -> np.ndarray:
    L = np.log(np.maximum(lum(A), 1e-4))
    hp = L - cv2.GaussianBlur(L, (0, 0), max(R / 256.0, 1.5))
    return np.clip(0.5 + hp * 0.5, 0, 1)


def make_material(albedo_dir: Path, out: Path, cls: str, priors: dict) -> dict:
    tp = albedo_dir / "texture.png"
    if not tp.is_file():
        raise StageError(f"missing {tp} (pbr projects the de-lit views first)")
    A = read_rgb_linear(tp)
    obs = read_gray01(albedo_dir / "coverage.png") > 0.5
    p = class_prior(cls, priors)
    base, s = bound_mean(A, obs, *p["albedo_lum"])
    out.mkdir(parents=True, exist_ok=True)
    write_srgb_png(out / "base_color.png", base)
    write_png(out / "height.png", np.rint(height_from(base, A.shape[0]) * 255).astype(np.uint8))
    return {"class": cls, "albedo_scale": s, "roughness": p["roughness"], "specular": p["specular"],
            "bump": p["bump"], "metallic": p.get("metallic", 0.0), "texture": str(out / "base_color.png"),
            "height": str(out / "height.png"), "folder": safe_name(albedo_dir.name)}
