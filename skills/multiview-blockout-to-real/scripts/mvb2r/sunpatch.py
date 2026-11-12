"""Direct-light patch synthesis, the pure texture-space parts (no file IO; used by sunpatch_stage.py).

Generated photos carry the image model's own direct light (a sun patch on a floor or wall). The de-light gain is
smooth and clamped, so the patch survives into the base colour and ghosts after relighting. Here, per object:

  mask  = dilate(A | B) & atlas, A = the photo's own bright patch (raw projected luminance, low-passed, above
          k_bright x the object's observed median), B = texels the proxy's key light reaches directly;
  fill  = per connected mask component, block quilting from the SHADED, OBSERVED part of the same UV chart: each
          block is copied from an offset along the texture's dominant structure lines (structure tensor of the
          observed texture: the direction of least change, so lines stay continuous), several offsets in turn
          until every texel has a source; then a LOCAL level match (ratio form, per channel): the fill's
          low-passed level is scaled to a pull-push interpolation of the surrounding shaded sources' low-passed
          level, so neither single blocks nor the whole patch differ in level and the fill keeps only its own
          detail above that level (one mean gain from the ring is only the fallback when no shaded source
          surrounds the component); feathered in. Deterministic (fixed seed).
A component whose chart has too little shaded observed area (below MIN_SOURCE_BLOCKS blocks or MIN_SOURCE_RATIO x
the component) is left unchanged and reported; a source smaller than its component is tiled over it (flagged
`repetition_expected`). Components below MIN_COMPONENT_BLOCKS blocks are dropped as speckle and only counted.
Every synthesised texel is INFERRED.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mvb2r.fusion_io import lum
from mvb2r.sunpatch_quilt import quilt

K_BRIGHT = 1.6             # bright patch: low-passed raw luminance above this x the observed median (first run)
DILATE_M = 0.2             # metres added around the mask (penumbra, mullion shadows; first run)
FEATHER = 24               # texels of blend at the mask border (first run; capped at the dilation radius)
SEED = 0
LOWPASS_FRAC = 1 / 128     # bright-patch low-pass sigma as a fraction of the atlas size
MAX_DILATE_FRAC = 1 / 16   # dilation radius never above this fraction of the atlas size
DIRECT_LIT_FRAC = 0.35     # a view pixel is directly lit above this x the 99th percentile of the direct light
DIRECT_REF_PCT = 99
DIRECT_EPS = 1e-4
DIRECT_VOTE = 0.5          # texture space: score-weighted share of views that see the texel lit
SHADED_BAND = (0.65, 1.35)  # source texels: low-passed luminance within this band x the LOCAL shaded level
SHADED_SIGMA_FRAC = 1 / 256
SHADED_SIGMA_MIN = 3.0
SHADED_REF_BLOCKS = 2.0    # the local shaded level: candidates low-passed at this many blocks (smooth light kept,
                           # contact shadows and bright residue rejected; the level match handles the rest)
SAME_FACE_COS = 0.9        # with normals: sources face the same way as the component
MIN_SOURCE_BLOCKS = 4      # a component needs at least this many blocks' area of shaded observed source
MAX_UNSOURCED_FRACTION = 0.5   # more of a component than this without a copied block: not filled (reported)
MIN_SOURCE_RATIO = 0.1     # ... and source >= this x the component (a small source may tile a larger patch)
MIN_COMPONENT_BLOCKS = 0.25  # mask components below this many blocks' area are dropped (speckle), counted only
BLOCK_FRAC, BLOCK_MIN, BLOCK_MAX = 1 / 24, 8, 96
GAIN_LIMITS = (0.5, 2.0)   # per-channel level-match gain (local ratio, or the single fallback gain)
LEVEL_SIGMA_BLOCKS = 0.5   # local level low-pass sigma, in blocks: wide enough to hide block-to-block steps
ELONGATE = 3.0             # blocks this many times longer along a clear structure (planks not cut short)
ELONGATE_MIN_COH = 0.2     # ... when the structure-tensor coherence reaches this
LEVEL_MIN_WEIGHT = 0.3     # low-passed source weight above which the shaded level is known (else pull-push)


@dataclass(frozen=True)
class SynthParams:
    k_bright: float = K_BRIGHT
    dilate_m: float = DILATE_M
    feather: int = FEATHER
    seed: int = SEED


def _blur_ratio(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur((img * w).astype(np.float32), (0, 0), sigma)
    return num / np.maximum(cv2.GaussianBlur(w.astype(np.float32), (0, 0), sigma), 1e-6)


def bright_mask(raw_lum: np.ndarray, observed: np.ndarray, inside: np.ndarray, k_bright: float,
                sigma: float) -> tuple[np.ndarray, float | None]:
    """(A, observed median): observed texels whose low-passed raw luminance exceeds k_bright x the median."""
    w = observed & inside
    if not w.any():
        return np.zeros(raw_lum.shape, bool), None
    lp = _blur_ratio(raw_lum, w.astype(np.float32), sigma)
    med = float(np.median(lp[w]))
    return w & (lp > k_bright * med), med


def combine_masks(bright: np.ndarray, direct: np.ndarray, inside: np.ndarray, radius: int) -> np.ndarray:
    u = (bright | direct).astype(np.uint8)
    if radius > 0:
        u = cv2.dilate(u, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    return (u > 0) & inside


def dilate_texels(dilate_m: float, texel_area_m2: np.ndarray, R: int) -> int:
    """Metres -> texels from the object's median texel size (world metres per texel)."""
    if dilate_m <= 0 or not len(texel_area_m2):
        return 0
    mpt = max(float(np.sqrt(np.median(texel_area_m2))), 1e-9)
    return int(min(round(dilate_m / mpt), R * MAX_DILATE_FRAC))


def structure_direction(L: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit texel (x, y) direction of least luminance change (along the structure lines) and its coherence."""
    gx = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    m = weight.astype(np.float32)
    J = np.array([[np.sum(gx * gx * m), np.sum(gx * gy * m)], [np.sum(gx * gy * m), np.sum(gy * gy * m)]])
    if not np.isfinite(J).all() or np.trace(J) <= 1e-12:
        return np.array([1.0, 0.0]), 0.0
    w, v = np.linalg.eigh(J)
    return v[:, 0] / np.linalg.norm(v[:, 0]), float((w[1] - w[0]) / max(w[1] + w[0], 1e-12))


def direct_lit_view(direct: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Image space: object pixels the key light reaches directly (relative to its 99th percentile there)."""
    obj = ids > 0
    vals = direct[obj & (direct > DIRECT_EPS)]
    if not len(vals):
        return np.zeros(direct.shape, bool)
    return obj & (direct > DIRECT_LIT_FRAC * float(np.percentile(vals, DIRECT_REF_PCT)))


def direct_from_stack(valid: np.ndarray, score: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Texture space (N,): score-weighted vote of the views that see the texel (projected 0/1 lit images)."""
    w = np.where(valid, np.maximum(score, 1e-6), 0.0)
    num = (w * color.mean(-1)).sum(0)
    den = w.sum(0)
    return (den > 0) & (num > DIRECT_VOTE * np.maximum(den, 1e-12))


def block_size(R: int) -> int:
    return int(np.clip(round(R * BLOCK_FRAC), BLOCK_MIN, BLOCK_MAX))


def feather_alpha(comp: np.ndarray, feather: float, chart: np.ndarray | None = None) -> np.ndarray:
    """Blend weight of the fill: ramps over `feather` texels from the component's border with texels of its own
    chart (`chart`, bool), and stays 1 where the component meets the chart boundary (empty atlas or another
    chart), so the original patch never survives along a 3D seam."""
    if feather <= 0:
        return comp.astype(np.float32)
    field = comp if chart is None else comp | ~chart
    d = cv2.distanceTransform(field.astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(comp, np.clip(d / feather, 0, 1), 0).astype(np.float32)


def shaded_source(base: np.ndarray, observed: np.ndarray, inside: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observed atlas texels outside the mask at the shaded level (no contact shadow, no other bright residue)."""
    R = base.shape[0]
    cand = observed & inside & ~mask
    if not cand.any():
        return cand
    lbl = _blur_ratio(lum(base), (observed & inside).astype(np.float32), max(R * SHADED_SIGMA_FRAC, SHADED_SIGMA_MIN))
    ref = _blur_ratio(lbl, cand.astype(np.float32), SHADED_REF_BLOCKS * block_size(R))
    return cand & (lbl > SHADED_BAND[0] * ref) & (lbl < SHADED_BAND[1] * ref)


def _component_source(src, charts, comp, normals):
    chart = charts == int(np.bincount(charts[comp]).argmax())
    out = src & chart
    if normals is not None:
        nm = normals[comp].mean(0)
        out &= (normals @ (nm / max(np.linalg.norm(nm), 1e-9))) > SAME_FACE_COS
    return out, chart


def _lowpass(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted low-pass of an (R,R,3) image (weights w)."""
    wf = w.astype(np.float32)
    num = cv2.GaussianBlur((img * wf[..., None]).astype(np.float32), (0, 0), sigma)
    return num / np.maximum(cv2.GaussianBlur(wf, (0, 0), sigma), 1e-6)[..., None]


def level_target(base, src, sigma):
    """(R,R,3) shaded level: the shaded sources low-passed at `sigma`, pull-pushed into the mask; None if unknown."""
    from mvb2r.coverage import push_pull
    known = cv2.GaussianBlur(src.astype(np.float32), (0, 0), sigma) > LEVEL_MIN_WEIGHT
    if not src.any() or not known.any():
        return None
    return push_pull(np.where(known[..., None], _lowpass(base, src, sigma), 0).astype(np.float32),
                     known.astype(np.float32))


def level_match(fill, base, comp, src, sigma, target=None):
    """Ratio form: fill x (interpolated shaded level / fill level), both low-passed at `sigma`, per channel.
    Returns (matched copy of fill or None when no shaded source is known, {residual before/after} or None)."""
    target = level_target(base, src, sigma) if target is None else target
    if target is None:
        return None, None
    ratio = np.clip(target / np.maximum(_lowpass(fill, comp, sigma), 1e-4), *GAIN_LIMITS)
    out = fill.copy()
    out[comp] = np.clip(fill[comp] * ratio[comp], 0, 1)
    resid = lambda f: float(np.sqrt(np.mean((lum(_lowpass(f, comp, sigma))[comp] - lum(target)[comp]) ** 2)))  # noqa: E731
    return out, {"level_residual_before": resid(fill), "level_residual_after": resid(out)}


def _fill_component(base, comp, src, chart, along, coh, block, feather, rng):
    """New values of `comp` (feathered over base) and its ring / level statistics."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * block + 1, 2 * block + 1))
    ring = (cv2.dilate(comp.astype(np.uint8), k) > 0) & src & ~comp
    target = level_target(base, src, LEVEL_SIGMA_BLOCKS * block) if ring.any() else None
    fill, left, _ = quilt(base, comp, src, along, block, rng, None if target is None else lum(target),
                          ELONGATE if coh >= ELONGATE_MIN_COH else 1.0)
    ring_mean = base[ring].mean(0) if ring.any() else base[src].mean(0)
    fill[left] = ring_mean
    matched, lvl = level_match(fill, base, comp, src, LEVEL_SIGMA_BLOCKS * block, target) if target is not None \
        else (None, None)
    if matched is None:  # fallback: no shaded source around the component, one mean gain
        gain = np.clip(ring_mean / np.maximum(fill[comp].mean(0), 1e-6), *GAIN_LIMITS)
        matched = fill.copy()
        matched[comp] = np.clip(fill[comp] * gain, 0, 1)
    a = feather_alpha(comp, feather, chart)[comp][:, None]
    vals = matched[comp] * a + base[comp] * (1 - a)
    before = float(np.linalg.norm(base[comp].mean(0) - ring_mean))
    after = float(np.linalg.norm(vals.mean(0) - ring_mean))
    return vals, {"texels": int(comp.sum()), "ring_diff_before": before, "ring_diff_after": after,
                  "unsourced_texels": int(left.sum()), "ring_texels": int(ring.sum()),
                  "level_match": "local" if lvl else "single gain (fallback)",
                  **(lvl or {"level_residual_before": None, "level_residual_after": None})}


def synthesise(base: np.ndarray, mask: np.ndarray, observed: np.ndarray, inside: np.ndarray, prm: SynthParams,
               normals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(new base colour, synthesised-texel mask, report). Inputs are not modified."""
    out, synth = base.copy(), np.zeros(mask.shape, bool)
    src = shaded_source(base, observed, inside, mask)
    along, coh = structure_direction(lum(base), src)
    _, charts = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)
    n, comps = cv2.connectedComponents((mask & inside).astype(np.uint8), connectivity=8)
    block, rng = block_size(base.shape[0]), np.random.default_rng(prm.seed)
    filled, skipped, dropped = [], [], 0
    for k in range(1, n):
        comp = comps == k
        if comp.sum() < MIN_COMPONENT_BLOCKS * block * block:
            dropped += 1
            continue
        csrc, chart = _component_source(src, charts, comp, normals)
        why = _source_problem(int(csrc.sum()), int(comp.sum()), block)
        if why:
            skipped.append({"texels": int(comp.sum()), "source_texels": int(csrc.sum()), "reason": why})
            continue
        vals, st = _fill_component(base, comp, csrc, chart, along, coh, block, prm.feather, rng)
        if st["unsourced_texels"] > MAX_UNSOURCED_FRACTION * comp.sum():
            skipped.append({"texels": int(comp.sum()), "source_texels": int(csrc.sum()),
                            "reason": f"no source window fits a block for {st['unsourced_texels']} of its texels "
                                      "(source too narrow): left unchanged"})
            continue
        out[comp] = vals
        synth |= comp
        ratio = float(csrc.sum() / comp.sum())
        filled.append({**st, "source_texels": int(csrc.sum()), "source_ratio": ratio,
                       "repetition_expected": ratio < 1})
    rep = _report(mask & inside, synth, filled, skipped, along, coh, block)
    return out, synth, {**rep, "components_dropped_small": dropped,
                        "min_component_texels": int(MIN_COMPONENT_BLOCKS * block * block)}


def _source_problem(src_texels: int, comp_texels: int, block: int) -> str | None:
    """Why a component cannot be filled from its source, or None."""
    need = MIN_SOURCE_BLOCKS * block * block
    if src_texels < need:
        return (f"too little shaded observed source on its UV chart ({src_texels} texels, minimum "
                f"{MIN_SOURCE_BLOCKS} blocks = {need}): left unchanged")
    if src_texels < MIN_SOURCE_RATIO * comp_texels:
        return (f"too little shaded observed source on its UV chart ({src_texels / comp_texels:.2f} x the "
                f"component, minimum ratio {MIN_SOURCE_RATIO}): left unchanged")
    return None


def _report(mask, synth, filled, skipped, along, coh, block) -> dict:
    tex = max(sum(f["texels"] for f in filled), 1)
    wavg = lambda key: float(sum(f[key] * f["texels"] for f in filled) / tex) if filled else None  # noqa: E731
    return {"mask_texels": int(mask.sum()), "synth_texels": int(synth.sum()), "components_filled": len(filled),
            "components_skipped": len(skipped), "skipped": skipped, "ring_diff_before": wavg("ring_diff_before"),
            "ring_diff_after": wavg("ring_diff_after"), "level_residual_before": _wavg_opt(filled, "level_residual_before"),
            "level_residual_after": _wavg_opt(filled, "level_residual_after"),
            "level_match": sorted({f["level_match"] for f in filled}), "unsourced_texels": sum(f["unsourced_texels"] for f in filled),
            "structure_direction_texel_xy": [float(along[0]), float(along[1])], "structure_coherence": coh,
            "block_texels": block, "components": filled,
            "source_ratio_min": min((f["source_ratio"] for f in filled), default=None),
            "repetition_expected": any(f["repetition_expected"] for f in filled), "note": "INFERRED: synthesised from the shaded observed part of the same chart"}


def _wavg_opt(filled: list, key: str) -> float | None:
    """Texel-weighted mean of a per-component value over the components that have it."""
    have = [f for f in filled if f[key] is not None]
    n = sum(f["texels"] for f in have)
    return float(sum(f[key] * f["texels"] for f in have) / n) if n else None
