"""Fill of unobserved texels from the SAME object's observed texels (the result is INFERRED, never observed).

    fill_texture(tex, color, observed, base=None, far=None, *, mode="smooth", reference="median", lift=False)

mode "smooth" (default): Gaussian pull-push. The observed texels are pulled down a Gaussian pyramid (colour
premultiplied by support); on the way up each finer level is blended in with alpha = clip(support / 0.3, 0, 1),
so the fill varies continuously: no blocks where the scale switches. mode "blocks": the 0.1.0 box push-pull
(kept to reproduce 0.1.0; it shows blocks once holes are large).

Far from any observation the fill fades to a plain reference colour, so a large never-seen surface gets no
invented detail. reference "median": median of the observed texels (0.1.0). reference "well_lit": median of the
observed texels whose luminance lies between the 50th and 90th percentile: on a plain wall the median of ALL
texels includes shadowed corners and makes never-seen walls too dark. lift=True: filled texels are lifted to
be no darker than that colour (lift fades in over 96 texels from the nearest observed texel, at most x1.6) and
the fade to it takes 192 texels. Texel distances are given at a 2048 texture and scale with its size.
fill_reference_for(class) picks ("well_lit", True) for plain wall-like classes, ("median", False) otherwise.
"""
from __future__ import annotations

import cv2
import numpy as np

from mvb2r.common import StageError

MODES = ("smooth", "blocks")
REFERENCES = ("median", "well_lit")
WALL_LIKE_CLASSES = ("plaster", "paint")      # plain wall-like materials of case.CLASSES: lifted well-lit fill
SUPPORT_FULL = 0.3                            # pull-push: support at which a finer level fully replaces coarser
WELL_LIT_PERCENTILES = (50.0, 90.0)           # luminance band of the well-lit reference colour
REFERENCE_TEX = 2048                          # texel distances below are given at this texture size
LIFT_TEXELS, LIFT_FADE_TEXELS = 96.0, 192.0   # lift fade-in; fade to the reference colour (lift=True)
MIN_LIFT_TEXELS, MIN_FADE_TEXELS = 4.0, 8.0   # floors for small textures
LIFT_MAX = 1.6                                # largest brightness factor of the lift
FAR_FRACTION, FAR_MIN = 1 / 32.0, 8.0         # 0.1.0 fade start (lift=False): max(R / 32, 8) texels
DEFAULT_GREY = (0.18, 0.18, 0.18)             # linear colour of an object never observed and without a donor
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
EPS = 1e-12


def fill_reference_for(cls: str) -> tuple[str, bool]:
    """(reference, lift) for a material class: plain wall-like classes get the lifted well-lit fill."""
    return ("well_lit", True) if cls in WALL_LIKE_CLASSES else ("median", False)


def fill_distances(R: int) -> tuple[float, float]:
    """(lift fade-in, fade to the reference colour) in texels for an R x R texture."""
    s = R / float(REFERENCE_TEX)
    return max(LIFT_TEXELS * s, MIN_LIFT_TEXELS), max(LIFT_FADE_TEXELS * s, MIN_FADE_TEXELS)


def well_lit_colour(obs_col: np.ndarray) -> np.ndarray:
    """Median of the colours whose luminance lies in the WELL_LIT_PERCENTILES band (median of all as fallback)."""
    col = np.asarray(obs_col, np.float32)
    lum = col @ LUM
    lo, hi = np.percentile(lum, WELL_LIT_PERCENTILES)
    band = (lum >= lo) & (lum <= hi)
    return np.median(col[band] if band.any() else col, 0).astype(np.float32)


def reference_colour(obs_col: np.ndarray, reference: str) -> np.ndarray:
    return well_lit_colour(obs_col) if reference == "well_lit" else np.median(obs_col, 0).astype(np.float32)


def push_pull(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    """0.1.0 fill: w == 0 pixels of an (R,R,C) image from box-averaged neighbours at all scales (blocky)."""
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


def _normalised(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    return c / np.maximum(s, EPS)[..., None]


def smooth_pull_push(img: np.ndarray, w: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Gaussian pull-push of an (H,W,3) image with support w in [0, 1]; `fallback` where nothing is supported."""
    c, s = (img * w[..., None]).astype(np.float32), w.astype(np.float32)
    levels = []
    while min(s.shape) > 1:
        levels.append((c, s))
        c, s = cv2.pyrDown(c), cv2.pyrDown(s)
    F = np.where((s > EPS)[..., None], _normalised(c, s), np.asarray(fallback, np.float32)[None, None])
    for cl, sl in reversed(levels):
        up = cv2.pyrUp(F, dstsize=(sl.shape[1], sl.shape[0]))
        up = up.reshape(sl.shape + (3,))
        a = np.clip(sl / SUPPORT_FULL, 0, 1)[..., None]
        F = a * _normalised(cl, sl) + (1 - a) * up
    return F.astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _fade(filled, dist, ref, R, far):
    """0.1.0 rule: the fill is kept up to `far` texels, then fades to `ref` over another `far` texels."""
    far = far if far is not None else max(R * FAR_FRACTION, FAR_MIN)
    a = np.clip((dist - far) / far, 0, 1)[..., None]
    return filled * (1 - a) + ref[None, None] * a


def _lift_and_fade(filled, dist, ref, R, far):
    """Lift the fill to be no darker than `ref` (fading in over the lift distance), then fade to `ref`."""
    lift_d, fade_d = fill_distances(R)
    fade_d = far if far is not None else fade_d
    lum = filled @ LUM
    lift = np.clip(float(ref @ LUM) / np.maximum(lum, 1e-4), 1.0, LIFT_MAX)[..., None]
    lifted = filled * (1 + (lift - 1) * _smoothstep(dist / lift_d)[..., None])
    a = _smoothstep(dist / fade_d)[..., None]
    return lifted * (1 - a) + ref[None, None] * a


def _check(mode: str, reference: str) -> None:
    if mode not in MODES:
        raise StageError(f"fill mode {str(mode)[:40]!r} is not one of {MODES}")
    if reference not in REFERENCES:
        raise StageError(f"fill reference {str(reference)[:40]!r} is not one of {REFERENCES}")


def fill_texture(tex, color: np.ndarray, observed: np.ndarray, base: np.ndarray | None = None,
                 far: float | None = None, *, mode: str = "smooth", reference: str = "median",
                 lift: bool = False) -> np.ndarray:
    """(R,R,3) linear texture: observed texels as given, the rest inferred from THIS object only."""
    _check(mode, reference)
    R = tex.R
    if not observed.any():
        col = np.asarray(base if base is not None else DEFAULT_GREY, np.float32)
        return np.broadcast_to(col, (R, R, 3)).astype(np.float32).copy()
    ref = reference_colour(color[observed], reference)
    img = np.zeros((R, R, 3), np.float32)
    w = np.zeros((R, R), np.float32)
    img[tex.rows[observed], tex.cols[observed]] = color[observed]
    w[tex.rows[observed], tex.cols[observed]] = 1
    filled = smooth_pull_push(img, w, ref) if mode == "smooth" else push_pull(img, w)
    dist = cv2.distanceTransform((w == 0).astype(np.uint8), cv2.DIST_L2, 5)
    out = (_lift_and_fade if lift else _fade)(filled, dist, ref, R, far)
    out[w > 0] = img[w > 0]
    return out.astype(np.float32)
