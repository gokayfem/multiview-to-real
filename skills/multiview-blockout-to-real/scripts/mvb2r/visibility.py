"""Per (view, texel) visibility tests with reason codes, the view score and same-object colour sampling.

Score of a valid sample (every factor in [0,1]):

    s = cos_theta                                   viewing angle (geometric normal vs direction to the camera)
      * ppt / (ppt + ppt_half)                      resolution: linear source pixels per texel
      * E(d_edge;  m_edge,  m_edge_soft)            distance to an object-id / depth discontinuity
      * E(d_border; m_border, m_border_soft)        distance to the image border
      * align                                       alignment confidence (targets/align_conf_XX.png, 1 if absent)
      * halo                                        < 1 in a band around other objects' clay silhouettes (legacy)
                                                    or nearer objects' silhouettes (strict; mvb2r/gating.py)

    E(d; a, b) = e_min + (1 - e_min) * smoothstep((d - a) / (b - a)),
    ppt = sqrt(texel_area * fx * fy * cos_theta / (Zc^2 * cos_alpha)), alpha = off-axis angle of the ray.

Hard rejections set reason bits (REASONS); `inconsistent` is soft (fusion decides). Depth test: the texel's Zc
against depth_z interpolated as 1/Z at the sub-pixel position, tolerance tol_abs + tol_rel*Zc +
tol_slope*(Zc/f)*tan(theta) (slope-scaled). Ported from the experiment's scripts/proj/visibility.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from mvb2r.pinhole import Camera
from mvb2r.raster import TexelSet

REASONS = {
    "behind": 1, "off_image": 2, "backface": 4, "grazing": 8, "occluded": 16, "depth_ahead": 32,
    "wrong_object": 64, "edge_margin": 128, "border_margin": 256, "low_align": 512, "inconsistent": 1024,
    "mask_gate": 2048, "self_edge": 4096,
}
HARD_REASONS = sum(v for k, v in REASONS.items() if k != "inconsistent")
REASON_ORDER = ["behind", "off_image", "backface", "grazing", "wrong_object", "occluded", "depth_ahead",
                "edge_margin", "self_edge", "border_margin", "low_align", "mask_gate", "inconsistent"]
BACKGROUND_DEPTH = 1e8  # depth_z at or beyond this is background (no surface)
REF_CAL_PX = 1664  # pixel margins below are tuned at this calibration size and scaled per camera
SOFT_ONLY = REASONS["grazing"] | REASONS["self_edge"]  # rejected only for these -> low-confidence "soft" sample
SCORE_FORMULA = ("s = cos * ppt/(ppt+ppt_half) * E(d_edge) * E(d_border) * align * halo; "
                 "E(d)=e_min+(1-e_min)*smoothstep((d-m)/(m_soft-m)); ppt=sqrt(texel_area*fx*fy*cos/(Zc^2*cos_alpha))")


@dataclass(frozen=True)
class VisParams:
    cos_min: float = 0.15
    tol_abs: float = 0.004
    tol_rel: float = 0.002
    tol_slope: float = 0.35
    m_edge: float = 1.5
    m_edge_soft: float = 6.0
    m_border: float = 2.0
    m_border_soft: float = 16.0
    e_min: float = 0.1
    align_min: float = 0.3
    ppt_half: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    def for_resolution(self, width: int, height: int) -> "VisParams":
        """Pixel margins are given at the 1664 px reference calibration and scale UP with larger cameras. They do
        not shrink below the reference: generated views misregister by a few pixels at any resolution (round trip
        at 320 px: scaled-down margins raised silhouette leakage from <1 % to 1.25 % on a masked object and
        3.8 % on an unmasked one)."""
        k = max(max(width, height) / REF_CAL_PX, 1.0)
        return replace(self, m_edge=self.m_edge * k, m_edge_soft=self.m_edge_soft * k,
                       m_border=self.m_border * k, m_border_soft=self.m_border_soft * k)


@dataclass(frozen=True)
class ViewData:
    cam: Camera
    depth: np.ndarray              # (H,W) float32 camera-axis Zc, background >= 1e8
    ids: np.ndarray                # (H,W) int32 object id, 0 = background
    edge_dist: np.ndarray          # (H,W) px to the nearest id / depth discontinuity
    id_edge_dist: np.ndarray       # (H,W) px to the nearest id change only
    image: np.ndarray | None       # (h,w,3) linear RGB (any size with the camera aspect)
    lowpass: np.ndarray | None     # (h,w,3) id-masked Gaussian low-pass of image
    align: np.ndarray | None = None

    @property
    def scale(self) -> float:
        return 1.0 if self.image is None else self.image.shape[1] / self.cam.width


@dataclass(frozen=True)
class Gates:
    """Per-object image-space gates in calibration pixels (all optional); built by mvb2r/gating.py."""
    accept: np.ndarray | None = None   # bool: texel valid only where True (generated mask / strict accept)
    reject: np.ndarray | None = None   # bool: texel rejected where True (legacy: dilated other objects' masks)
    score_mult: np.ndarray | None = None  # float: legacy halo / strict occluder soft-band factor


@dataclass(frozen=True)
class ViewSamples:
    flags: np.ndarray     # (N,) uint16
    score: np.ndarray     # (N,) float32, 0 where any hard flag
    soft: np.ndarray      # (N,) float32 score of samples rejected only as grazing / self-edge
    color: np.ndarray     # (N,3) float32 linear
    lowpass: np.ndarray   # (N,3) float32 linear
    u: np.ndarray         # (N,) float32 calibration px (-1e9 where behind)
    v: np.ndarray
    zc: np.ndarray
    ppt: np.ndarray
    cos: np.ndarray


# ----------------------------------------------------------------------------------------------- view data
def discontinuity_distance(ids: np.ndarray, depth: np.ndarray, rel: float = 0.04, abs_m: float = 0.03) -> np.ndarray:
    """Distance (px) to the nearest object-id change or depth jump between 4-neighbours."""
    e = np.zeros(ids.shape, bool)
    z = np.where(depth < BACKGROUND_DEPTH, depth, 1e3).astype(np.float64)
    for a, b in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                 ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))):
        jump = (ids[a] != ids[b]) | (np.abs(z[a] - z[b]) > abs_m + rel * np.minimum(z[a], z[b]))
        e[a] |= jump
        e[b] |= jump
    src = np.where(e, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_5).astype(np.float32)


def masked_lowpass(img: np.ndarray, labels: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass computed separately inside each label (never mixes objects)."""
    out = np.zeros_like(img)
    for lab in np.unique(labels):
        m = (labels == lab).astype(np.float32)
        num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)
        sel = m > 0
        out[sel] = num[sel] / np.maximum(den[sel], 1e-6)[:, None]
    return out


def make_view(cam: Camera, depth: np.ndarray, ids: np.ndarray, image: np.ndarray | None = None,
              align: np.ndarray | None = None, lowpass_sigma: float = 2.0) -> ViewData:
    if depth.shape != (cam.height, cam.width) or ids.shape != (cam.height, cam.width):
        raise ValueError(f"{cam.name}: depth {depth.shape} / ids {ids.shape} != camera {cam.height}x{cam.width}")
    ids = np.where(depth >= BACKGROUND_DEPTH, 0, ids).astype(np.int32)
    lp = None
    if image is not None:
        h, w = image.shape[:2]
        if abs(w / cam.width - h / cam.height) > 2.0 / min(cam.width, cam.height):
            raise ValueError(f"{cam.name}: image {w}x{h} does not have the camera aspect {cam.width}x{cam.height}")
        ids_img = ids if (h, w) == ids.shape else cv2.resize(ids.astype(np.float32), (w, h),
                                                             interpolation=cv2.INTER_NEAREST).astype(np.int32)
        lp = masked_lowpass(image, ids_img, lowpass_sigma * w / cam.width)
    return ViewData(cam=cam, depth=depth.astype(np.float32), ids=ids, edge_dist=discontinuity_distance(ids, depth),
                    id_edge_dist=discontinuity_distance(ids, np.zeros_like(depth)), image=image, lowpass=lp,
                    align=align)


# ----------------------------------------------------------------------------------------------- sampling
def bilerp(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample at continuous pixel coords (pixel-centre convention +0.5), edge-clamped."""
    h, w = img.shape[:2]
    x, y = np.asarray(x, np.float64) - 0.5, np.asarray(y, np.float64) - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        ax, ay = ax[:, None], ay[:, None]
    return ((img[y0c, x0c] * (1 - ax) + img[y0c, x1c] * ax) * (1 - ay)
            + (img[y1c, x0c] * (1 - ax) + img[y1c, x1c] * ax) * ay)


def interp_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Depth at sub-pixel (u, v): bilinear in 1/Z over taps within 5 % of the nearest pixel's depth."""
    H, W = depth.shape
    dn = depth[np.clip(np.floor(v).astype(np.int64), 0, H - 1), np.clip(np.floor(u).astype(np.int64), 0, W - 1)]
    dn = dn.astype(np.float64)
    x, y = u - 0.5, v - 0.5
    i0, j0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = x - i0, y - j0
    acc = np.zeros(len(u))
    ws = np.zeros(len(u))
    for di, dj, wt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        d = depth[np.clip(j0 + dj, 0, H - 1), np.clip(i0 + di, 0, W - 1)].astype(np.float64)
        ok = np.abs(d - dn) < 0.05 * dn
        acc += wt * ok / np.maximum(d, 1e-9)
        ws += wt * ok
    return np.where(ws > 1e-9, ws / np.maximum(acc, 1e-30), dn)


def sample_same_object(view: ViewData, img: np.ndarray, oid: int, u, v, zc, tol) -> np.ndarray:
    """Bilinear sample whose taps must be the same object and depth-consistent (no colour bleed across
    silhouettes). u, v in calibration px; the image may be larger (scale s). Falls back to the nearest pixel."""
    s = img.shape[1] / view.cam.width
    h, w = img.shape[:2]
    H, W = view.ids.shape
    x, y = u * s - 0.5, v * s - 0.5
    i0, j0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = x - i0, y - j0
    acc = np.zeros((len(u), img.shape[2]), np.float64)
    wsum = np.zeros(len(u), np.float64)
    for di, dj, wt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        ii, jj = np.clip(i0 + di, 0, w - 1), np.clip(j0 + dj, 0, h - 1)
        ci = np.clip(((ii + 0.5) / s).astype(np.int64), 0, W - 1)
        cj = np.clip(((jj + 0.5) / s).astype(np.int64), 0, H - 1)
        ok = (view.ids[cj, ci] == oid) & (np.abs(view.depth[cj, ci] - zc) < 3 * tol + 0.01 * zc)
        acc += (wt * ok)[:, None] * img[jj, ii]
        wsum += wt * ok
    nn = img[np.clip(np.floor(v * s).astype(np.int64), 0, h - 1), np.clip(np.floor(u * s).astype(np.int64), 0, w - 1)]
    good = wsum > 1e-6
    return np.where(good[:, None], acc / np.maximum(wsum, 1e-12)[:, None], nn).astype(np.float32)


# ----------------------------------------------------------------------------------------------- tests
def _geometry_flags(tex: TexelSet, view: ViewData, prm: VisParams):
    cam = view.cam
    H, W = view.ids.shape
    u, v, zc = cam.project(tex.P)
    f = np.zeros(tex.n, np.uint16)
    behind = ~(zc > cam.clip_start)
    f[behind] |= REASONS["behind"]
    u = np.where(behind, -1e9, u)
    v = np.where(behind, -1e9, v)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    f[~inside & ~behind] |= REASONS["off_image"]
    d = cam.centre[None] - tex.P
    rl = np.linalg.norm(d, axis=1)
    cosv = np.sum(d * tex.Ng, 1) / np.maximum(rl, 1e-12)
    f[cosv <= 0] |= REASONS["backface"]
    f[(cosv > 0) & (cosv < prm.cos_min)] |= REASONS["grazing"]
    zs = np.where(behind, 1.0, zc)
    return f, u, v, zs, rl, cosv, inside, behind


def depth_tolerance(prm: VisParams, zs: np.ndarray, cosv: np.ndarray, fx: float) -> np.ndarray:
    cpos = np.clip(cosv, 1e-3, 1.0)
    tan = np.minimum(np.sqrt(1 - cpos ** 2) / cpos, 10.0)
    return prm.tol_abs + prm.tol_rel * zs + prm.tol_slope * (zs / fx) * tan


def _image_flags(f, tex_oid, view, prm, u, v, zs, cosv, inside, gates: Gates):
    H, W = view.ids.shape
    i = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
    j = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
    tol = depth_tolerance(prm, zs, cosv, view.cam.fx)
    D = interp_depth(view.depth, np.clip(u, 0, W - 1e-6), np.clip(v, 0, H - 1e-6))
    f[inside & (zs > D + tol)] |= REASONS["occluded"]
    f[inside & (zs < D - tol)] |= REASONS["depth_ahead"]
    f[inside & (view.ids[j, i] != tex_oid)] |= REASONS["wrong_object"]
    de = view.edge_dist[j, i]
    near = inside & (de < prm.m_edge)
    idn = view.id_edge_dist[j, i] < prm.m_edge
    f[near & idn] |= REASONS["edge_margin"]
    f[near & ~idn] |= REASONS["self_edge"]
    db = np.minimum(np.minimum(u, W - u), np.minimum(v, H - v))
    f[inside & (db < prm.m_border)] |= REASONS["border_margin"]
    if gates.accept is not None:
        f[inside & ~gates.accept[j, i]] |= REASONS["mask_gate"]
    if gates.reject is not None:
        f[inside & gates.reject[j, i]] |= REASONS["mask_gate"]
    al = np.ones(len(u), np.float32)
    if gates.score_mult is not None:
        al = al * gates.score_mult[j, i]
    if view.align is not None:
        ha, wa = view.align.shape
        a = np.clip(bilerp(view.align, u * wa / W, v * ha / H), 0, 1).astype(np.float32)
        f[inside & (a < prm.align_min)] |= REASONS["low_align"]
        al = al * a
    return tol, de, db, al


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def edge_factor(d: np.ndarray, m: float, m_soft: float, e_min: float) -> np.ndarray:
    return e_min + (1 - e_min) * smoothstep((d - m) / max(m_soft - m, 1e-6))


def view_samples(tex: TexelSet, oid: int, view: ViewData, prm: VisParams, gates: Gates = Gates()) -> ViewSamples:
    """Visibility, reasons, score and colour of every texel of one object in one view."""
    prm = prm.for_resolution(view.cam.width, view.cam.height)
    f, u, v, zs, rl, cosv, inside, behind = _geometry_flags(tex, view, prm)
    tol, de, db, al = _image_flags(f, oid, view, prm, u, v, zs, cosv, inside, gates)
    cam = view.cam
    cpos = np.clip(cosv, 1e-3, 1.0)
    cos_alpha = np.clip(zs / np.maximum(rl, 1e-12), 1e-3, 1)
    ppt = np.sqrt(tex.area * cam.fx * cam.fy * cpos / (zs ** 2 * cos_alpha))
    sc = cpos * (ppt / (ppt + prm.ppt_half))
    sc = sc * edge_factor(de, prm.m_edge, prm.m_edge_soft, prm.e_min)
    sc = sc * edge_factor(db, prm.m_border, prm.m_border_soft, prm.e_min) * al
    ok = f == 0
    soft_ok = ((f & ~np.uint16(SOFT_ONLY)) == 0) & (cosv > 0.04)
    color = np.zeros((tex.n, 3), np.float32)
    lowp = np.zeros((tex.n, 3), np.float32)
    k = np.nonzero(soft_ok)[0]
    if view.image is not None and len(k):
        color[k] = sample_same_object(view, view.image, oid, u[k], v[k], zs[k], tol[k])
        lowp[k] = sample_same_object(view, view.lowpass, oid, u[k], v[k], zs[k], tol[k])
    return ViewSamples(flags=f, score=np.where(ok, sc, 0).astype(np.float32),
                       soft=np.where(soft_ok & ~ok, sc + 1e-4, 0).astype(np.float32), color=color, lowpass=lowp,
                       u=u.astype(np.float32), v=v.astype(np.float32), zc=np.where(behind, 0, zs).astype(np.float32),
                       ppt=np.where(behind, 0, ppt).astype(np.float32), cos=cosv.astype(np.float32))


def primary_reason(flags: np.ndarray) -> np.ndarray:
    """(…) uint16 flags -> index into REASON_ORDER + 1 of the highest-priority reason (0 = valid)."""
    out = np.zeros(flags.shape, np.uint8)
    for k, name in reversed(list(enumerate(REASON_ORDER, start=1))):
        out[(flags & REASONS[name]) != 0] = k
    return out
