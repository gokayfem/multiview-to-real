"""Conservative fill of unobserved texels, per-object maps and area-weighted coverage statistics.

Fill rule (INFERRED texels, flagged in unobserved.png and coverage.json): same object only. Near observed texels
a push-pull pyramid fill continues the observed colours; farther than `far` texels it fades to the object's
robust median colour, so large never-seen surfaces get a plain colour and no invented detail. An object with
almost nothing observed takes the median of observed objects of the same class (donor), else its class prior.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion import Result
from mvb2r.fusion_io import linear_to_srgb, u8, write_png, write_srgb_png
from mvb2r.visibility import REASON_ORDER, REASONS, primary_reason

OUTSIDE = 255
VIEW_COLOURS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                         [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]], np.uint8)
REASON_COLOURS = np.array([[0, 0, 0], [120, 120, 120], [60, 60, 200], [128, 0, 128], [200, 120, 255],
                           [255, 0, 0], [255, 140, 0], [255, 255, 0], [0, 200, 0], [0, 200, 200], [255, 0, 255],
                           [255, 128, 128], [128, 64, 0], [255, 255, 255]], np.uint8)


def push_pull(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fill w == 0 pixels of an (R,R,C) image from weighted neighbours at all scales."""
    R0, C = img.shape[0], img.shape[2]
    R = 1 << int(np.ceil(np.log2(max(R0, 2))))
    c = np.zeros((R, R, C), np.float32)
    ww = np.zeros((R, R), np.float32)
    c[:R0, :R0] = img
    ww[:R0, :R0] = w
    levels = []
    while c.shape[0] > 1:
        levels.append((c, ww))
        h = c.shape[0] // 2
        cw = (c * ww[..., None]).reshape(h, 2, h, 2, C).sum((1, 3))
        ws = ww.reshape(h, 2, h, 2).sum((1, 3))
        c = cw / np.maximum(ws, 1e-12)[..., None]
        ww = np.minimum(ws, 1.0)
    for cf, wf in reversed(levels):
        n2 = cf.shape[0]
        up = cv2.resize(c, (n2, n2), interpolation=cv2.INTER_LINEAR) if c.shape[0] > 1 else \
            np.repeat(np.repeat(c, 2, 0), 2, 1)
        up = up[..., None] if up.ndim == 2 else up
        a = np.clip(wf, 0, 1)[..., None]
        c = cf * a + up * (1 - a)
    return c[:R0, :R0]


def fill_texture(tex, color: np.ndarray, observed: np.ndarray, base: np.ndarray | None = None,
                 far: float | None = None) -> np.ndarray:
    """(R,R,3) linear texture: observed texels as given, the rest inferred from THIS object only."""
    R = tex.R
    med = np.median(color[observed], 0).astype(np.float32) if observed.any() else \
        np.asarray(base if base is not None else (0.18, 0.18, 0.18), np.float32)
    if not observed.any():
        return np.broadcast_to(med, (R, R, 3)).astype(np.float32).copy()
    img = np.zeros((R, R, 3), np.float32)
    w = np.zeros((R, R), np.float32)
    img[tex.rows[observed], tex.cols[observed]] = color[observed]
    w[tex.rows[observed], tex.cols[observed]] = 1
    pp = push_pull(img, w)
    far = far if far is not None else max(R / 32.0, 8.0)
    dist = cv2.distanceTransform((w == 0).astype(np.uint8), cv2.DIST_L2, 5)
    a = np.clip((dist - far) / far, 0, 1)[..., None]
    out = pp * (1 - a) + med[None, None] * a
    out[w > 0] = img[w > 0]
    return out.astype(np.float32)


def area_fractions(tex, observed: np.ndarray, soft: np.ndarray) -> dict:
    a = tex.weight.astype(np.float64)
    A = max(a.sum(), 1e-12)
    hard = observed & ~soft
    return {"observed": float(a[hard].sum() / A), "soft": float(a[soft].sum() / A),
            "inferred": float(a[~observed].sum() / A), "area_m2": float(A)}


def contributions(tex, res: Result, view_names) -> dict:
    a = tex.weight.astype(np.float64)
    ao = max(a[res.observed].sum(), 1e-12)
    return {n: float(a[res.observed & (res.source == v)].sum() / ao) for v, n in enumerate(view_names)}


def write_method_maps(out: Path, tex, res: Result, texture: np.ndarray) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_srgb_png(out / "texture.png", texture)
    raw = np.full((tex.R, tex.R), OUTSIDE, np.uint8)
    raw[tex.rows, tex.cols] = (res.source.astype(np.int32) + 1).astype(np.uint8)  # 0 = unobserved
    write_png(out / "source_raw.png", raw)
    fc = np.full((tex.R, tex.R, 3), 40, np.uint8)
    fc[tex.rows, tex.cols] = VIEW_COLOURS[np.clip(res.source.astype(np.int32) + 1, 0, len(VIEW_COLOURS) - 1)]
    write_png(out / "source.png", fc)
    write_png(out / "confidence.png", tex.scatter(u8(res.confidence), 0))
    write_png(out / "coverage.png", tex.scatter(np.where(res.observed, 255, 0).astype(np.uint8), 0))
    write_png(out / "unobserved.png", tex.scatter(np.where(res.observed, 0, 255).astype(np.uint8), 0))
    soft = res.extra.get("soft")
    if soft is not None:
        write_png(out / "soft_observed.png", tex.scatter(np.where(soft, 255, 0).astype(np.uint8), 0))
    if "seam_band" in res.extra:
        write_png(out / "seam_band.png", tex.scatter(np.where(res.extra["seam_band"], 255, 0).astype(np.uint8), 0))


def write_rejections(out: Path, tex, st) -> dict:
    """Per-reason count maps, per-view primary-reason maps, flags.npz; area fraction per reason and view."""
    out.mkdir(parents=True, exist_ok=True)
    step = 255 // max(st.V, 1)
    a = tex.weight.astype(np.float64)
    A = max(a.sum(), 1e-12)
    stats = {}
    for name in REASON_ORDER:
        hit = (st.flags & REASONS[name]) != 0
        write_png(out / f"reason_{name}.png", tex.scatter((hit.sum(0) * step).astype(np.uint8), 0))
        stats[name] = {st.view_names[v]: round(float(a[hit[v]].sum() / A), 5) for v in range(st.V)}
    write_png(out / "valid_view_count.png", tex.scatter((st.valid.sum(0) * step).astype(np.uint8), 0))
    for v in range(st.V):
        img = np.full((tex.R, tex.R, 3), 40, np.uint8)
        img[tex.rows, tex.cols] = REASON_COLOURS[primary_reason(st.flags[v])]
        write_png(out / f"primary_reason_{st.view_names[v]}.png", img)
    np.savez_compressed(out / "flags.npz", flags=st.flags, rows=tex.rows, cols=tex.cols, R=tex.R,
                        views=np.array(st.view_names))
    return {"reason_area_frac": stats,
            "valid_area_frac": {st.view_names[v]: float(a[st.valid[v]].sum() / A) for v in range(st.V)}}


def donor_colours(records: dict, textures: dict, min_obs: float = 0.05) -> dict:
    """{class: median linear colour} over objects of that class with >= min_obs observed area."""
    per: dict = {}
    for name, rec in records.items():
        if rec["coverage"]["observed"] + rec["coverage"]["soft"] >= min_obs and name in textures:
            per.setdefault(rec["class"], []).append(textures[name])
    return {c: np.median(np.array(v), 0).tolist() for c, v in per.items()}


def srgb_u8(lin: np.ndarray) -> np.ndarray:
    return u8(linear_to_srgb(lin))
