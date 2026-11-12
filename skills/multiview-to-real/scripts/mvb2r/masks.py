"""Generic per-object segmentation of a generated view, seeded by each object's own object_id region.

For every object visible in a view (>= MIN_FRACTION of the pixels):
  - sure foreground = its clay region eroded by ERODE_FRAC of the image size, probable foreground = the clay
    region, probable background = a band of BAND_FRAC around it, sure background = everything farther;
    OpenCV GrabCut (colour GMMs) decides the band; only components touching the sure foreground are kept.
  - objects that fill most of the view (>= LARGE_FRACTION) or are too thin to have a sure foreground keep
    their clay region ("prior only"): a colour model cannot place their outline, and the report says so.
  - reliability per object: Fisher separability of the object's interior colours from its surrounding band
    (Lab), with the mask/clay IoU: "high" (sep >= 2 and IoU >= 0.75), "moderate" (sep >= 1), "low", or
    "prior only". Errors larger than the band cannot be measured by construction.
Pixels claimed by several objects go to the object whose clay region contains them, else to the smaller one.
"""
from __future__ import annotations

import numpy as np

MIN_FRACTION = 0.0004
LARGE_FRACTION = 0.35
BAND_FRAC, ERODE_FRAC = 0.024, 0.006
WORK_MAX = 1024
GC_ITERS = 4


def _cv2():
    import cv2
    return cv2


def boundary(mask: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def chamfer(a: np.ndarray, b: np.ndarray) -> dict:
    """Symmetric boundary distance (px): mean and p95."""
    cv2 = _cv2()
    ba, bb = boundary(a), boundary(b)
    if not ba.any() or not bb.any():
        return {"mean": None, "p95": None}
    da = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 5)
    d = np.concatenate([db[ba], da[bb]])
    return {"mean": round(float(d.mean()), 2), "p95": round(float(np.percentile(d, 95)), 2)}


def separability(lab: np.ndarray, inside: np.ndarray, ring: np.ndarray) -> float:
    """Fisher distance between the Lab colours of two pixel sets (0 = indistinguishable)."""
    if inside.sum() < 20 or ring.sum() < 20:
        return 0.0
    a, b = lab[inside].astype(np.float64), lab[ring].astype(np.float64)
    d = a.mean(0) - b.mean(0)
    cov = (np.cov(a.T) + np.cov(b.T)) / 2 + np.eye(3) * 1.0
    return float(np.sqrt(max(d @ np.linalg.solve(cov, d), 0.0)))


def _grabcut(rgb: np.ndarray, clay: np.ndarray, erode: int, band: int) -> np.ndarray | None:
    cv2 = _cv2()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = clay.astype(np.uint8)
    fg = cv2.erode(m, k, iterations=erode)
    if fg.sum() < 10:
        return None
    near = cv2.dilate(m, k, iterations=band)
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[near > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[fg > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(np.ascontiguousarray(rgb[..., ::-1]), gc, None, bgd, fgd, GC_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab = cv2.connectedComponents(out, connectivity=8)
    if n > 1:
        keep = np.unique(lab[(fg > 0) & (out > 0)])
        out = np.isin(lab, keep[keep > 0]).astype(np.uint8)
    return out > 0


def _crop_box(mask: np.ndarray, pad: int):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return slice(max(0, ys.min() - pad), min(h, ys.max() + pad + 1)), slice(max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def segment_object(rgb: np.ndarray, lab: np.ndarray, clay: np.ndarray) -> dict:
    """Generated mask of one object (full image size) and its reliability."""
    cv2 = _cv2()
    h, w = clay.shape
    frac = float(clay.mean())
    size = max(h, w)
    band, erode = max(3, int(round(BAND_FRAC * size))), max(1, int(round(ERODE_FRAC * size)))
    if frac >= LARGE_FRACTION:
        return {"mask": clay.copy(), "reliability": "prior only", "separability": None,
                "method": "clay region (object fills the view)"}
    ys, xs = _crop_box(clay, band + 2)
    gen = _grabcut(rgb[ys, xs], clay[ys, xs], erode, band)
    if gen is None:
        return {"mask": clay.copy(), "reliability": "prior only", "separability": None,
                "method": "clay region (too thin for a sure foreground)"}
    full = np.zeros_like(clay)
    full[ys, xs] = gen
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inside = cv2.erode(clay.astype(np.uint8), k, iterations=erode) > 0
    ring = (cv2.dilate(clay.astype(np.uint8), k, iterations=band) > 0) & ~(cv2.dilate(clay.astype(np.uint8), k, iterations=2) > 0)
    sep = separability(lab, inside, ring)
    j = iou(full, clay)
    rel = "high" if sep >= 2.0 and j >= 0.75 else "moderate" if sep >= 1.0 else "low"
    return {"mask": full, "reliability": rel, "separability": round(sep, 3), "method": "GrabCut in a band around the clay outline"}


def _resolve_overlaps(results: dict, oid: np.ndarray) -> dict:
    order = sorted(results, key=lambda i: int(results[i]["mask"].sum()))
    taken = np.zeros(oid.shape, bool)
    out = {}
    own ={i: results[i]["mask"] & (oid == i) for i in results}           # pixels inside its own clay region
    for i in results:
        taken |= own[i]
    for i in order:
        extra = results[i]["mask"] & ~taken
        taken |= extra
        out[i] = {**results[i], "mask": own[i] | extra}
    return out


def segment_view(rgb: np.ndarray, oid: np.ndarray, ids=None) -> dict:
    """{object id: {mask, reliability, separability, method}} for every visible object of one view."""
    cv2 = _cv2()
    if rgb.shape[:2] != oid.shape:
        raise ValueError(f"image {rgb.shape[:2]} and object_id {oid.shape} differ")
    h, w = oid.shape
    scale = min(1.0, WORK_MAX / max(h, w))
    small_rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    small_oid = cv2.resize(oid.astype(np.int32), small_rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST) if scale < 1 else oid
    lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB)
    present = [int(i) for i in np.unique(small_oid) if i > 0 and (ids is None or int(i) in ids)]
    res = {}
    for i in present:
        clay = small_oid == i
        if clay.mean() < MIN_FRACTION:
            continue
        res[i] = segment_object(small_rgb, lab, clay)
    res = _resolve_overlaps(res, small_oid)
    if scale < 1:
        res = {i: {**r, "mask": cv2.resize(r["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0}
               for i, r in res.items()}
    return res
