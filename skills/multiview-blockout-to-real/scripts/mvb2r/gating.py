"""Per-object image gates for one view: the `strict` policy (default) and the `legacy` one it replaced.

case.json `fusion.gate_policy`:
  "legacy"  today's gates before the fix rounds: an object with its own generated mask samples only inside it
            (eroded 1 px, `gate_dilate` px when set); an object without one rejects the union of the other objects'
            masks dilated `reject_dilate` = 4 px and gets a score halo (x`halo_factor` 0.3 within `halo_px` 6 px)
            around those objects' clay silhouettes.
  "strict"  (default) every object: own clay region eroded `erode_own` = 2 px (1 or 0 px for thin objects, see
            thin_safe_erosion);
            WITH an own generated mask: AND that mask eroded `erode_gen` = 2 px (explicit gate_dilate wins, signed),
                AND NOT every other object's generated mask dilated `other_mask_dilate` = 2 px;
            WITHOUT one: NOT the union of the other objects' masks dilated `reject_dilate` = 8 px;
            AND NOT within `occ_margin` px of a NEARER other object's clay pixels (nearer = more than `occ_dz` =
            0.03 m closer than this pixel), score x`occ_soft_factor` 0.3 out to `occ_soft` px. Defaults 8 / 20 px
            without an own mask, 2 / 6 px (`occ_margin_masked` / `occ_soft_masked`) with one.
Pixel values are stated at the 1664 px reference. Strict defaults scale with the camera size in BOTH directions,
never below 1 px (`strict_px`): targets are resampled to the calibration size, so their misregistration in pixels
shrinks and grows with it, and unscaled margins would swallow thin objects of a small camera. Legacy defaults
scale up only (`px`). Explicit case.json values are used as given. Per object: `objects[].gate_dilate`,
`objects[].reject_dilate` override the fusion ones.
First run (references/seams.md, projection.md): the experiment's original gate, which dilated masks by 2 px,
let 4.3 % of observed skin texels take sweater / hair / trouser pixels; its strict gate took that to 0. The
`legacy` policy here already eroded masks by 1 px and leaks 0.01 % on the same scene; what `strict` adds in this
code base is the margin between background objects, the own-region erosion and the regions with a gap.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from mvb2r.fusion_io import StageError
from mvb2r.shapes import label, show
from mvb2r.visibility import REF_CAL_PX, Gates

POLICIES = ("strict", "legacy")
DEFAULT_POLICY = "strict"
STRICT_PX = {"erode_own": 2.0, "erode_gen": 2.0, "other_mask_dilate": 2.0, "reject_dilate": 8.0,
             "occ_margin": 8.0, "occ_soft": 20.0, "occ_margin_masked": 2.0, "occ_soft_masked": 6.0}
STRICT_OTHER = {"occ_soft_factor": 0.3, "occ_dz": 0.03}
STRICT_MIN_PX = 1.0                                    # a silhouette pixel is a mixture at any resolution
LEGACY_PX = {"gate_dilate": -1.0, "reject_dilate": 4.0, "halo_px": 6.0}
LEGACY_HALO_FACTOR = 0.3
SIGNED_KEYS = ("gate_dilate",)                         # may be negative (erode)
GATE_KEYS = tuple(sorted(set(STRICT_PX) | set(STRICT_OTHER) | set(LEGACY_PX) | {"halo_factor"}))
CONTACT_MASK_ERODE = 1                                 # occluder mask erosion inside a contact zone (px)
HALF = 0.5                                             # thin-object rule: erosion may remove at most half
NO_OCCLUDER_DEPTH = 1e9                                # m: 'no nearer object here' in the min-filter
MAX_GATE_PX = 256                                      # |pixel value| limit (a (2r+1)^2 kernel is built from it)
MAX_OCC_DZ = 10.0                                      # m: occ_dz limit
LIMITS = {**{k: float(MAX_GATE_PX) for k in set(STRICT_PX) | set(LEGACY_PX)}, "occ_dz": MAX_OCC_DZ,
          "occ_soft_factor": 1.0, "halo_factor": 1.0}


OTHER_FUSION_KEYS = ("method", "params", "by_class", "tex_size", "min_tex", "max_tex", "novel", "sky", "fill",
                     "gate_policy", "contacts")
FUSION_KEYS = tuple(sorted(set(OTHER_FUSION_KEYS) | set(GATE_KEYS)))   # every key the code reads from "fusion"


# ----------------------------------------------------------------------------------------------- options
def check_fusion_keys(f: dict) -> None:
    """Unknown keys of case.json "fusion" are refused (a misspelt margin would otherwise be ignored)."""
    unknown = sorted(str(k) for k in f if k not in FUSION_KEYS)
    if unknown:
        raise StageError(f"case.json fusion: unknown key(s) {show(unknown)}; allowed: {', '.join(FUSION_KEYS)}")


def _number(where: str, v, signed: bool = False, limit: float = float(MAX_GATE_PX)) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise StageError(f"case.json {where} must be a finite number, got {show(v)}")
    if not signed and v < 0:
        raise StageError(f"case.json {where} must be >= 0, got {show(v)}")
    if abs(v) > limit:
        raise StageError(f"case.json {where} must be at most {limit:g}{' in magnitude' if signed else ''}, "
                         f"got {show(v)}")
    return float(v)


def gate_options(f: dict) -> dict:
    """Validated gate keys of case.json "fusion" (policy always present)."""
    pol = f.get("gate_policy", DEFAULT_POLICY)
    if pol not in POLICIES:
        raise StageError(f"case.json fusion.gate_policy must be one of {POLICIES}, got {show(pol)}")
    out = {"gate_policy": pol}
    for k in GATE_KEYS:
        if k in f:
            out[k] = _number(f"fusion.{k}", f[k], k in SIGNED_KEYS, LIMITS[k])
    return out


def object_gate_overrides(o: dict) -> dict:
    """Validated per-object gate overrides (`gate_dilate` signed, `reject_dilate` >= 0)."""
    name = label(o.get("name"))
    return {k: _number(f"objects[{name}].{k}", o[k], k in SIGNED_KEYS) for k in ("gate_dilate", "reject_dilate")
            if k in o}


def px(opt: dict, key: str, default_at_ref: float, cam, obj: dict | None = None) -> float:
    """Per-object value, else the case.json value (both used as given), else the default (tuned at 1664 px)
    scaled UP for larger cameras (never below: misregistration does not shrink with resolution)."""
    if obj is not None and key in obj:
        return float(obj[key])
    if key in opt:
        return float(opt[key])
    return default_at_ref * max(max(cam.width, cam.height) / float(REF_CAL_PX), 1.0)


def strict_px(opt: dict, key: str, cam, obj: dict | None = None) -> float:
    """Per-object value, else the case.json value (both used as given), else the strict default (tuned at
    1664 px) scaled with the camera size in both directions, never below STRICT_MIN_PX."""
    if obj is not None and key in obj:
        return float(obj[key])
    if key in opt:
        return float(opt[key])
    scaled = STRICT_PX[key] * max(cam.width, cam.height) / float(REF_CAL_PX)
    return max(float(math.floor(scaled + 0.5)), STRICT_MIN_PX)   # whole pixels, halves up (round() goes to even)


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Dilate by r px (r > 0), erode by -r px (r < 0); a new array."""
    if r == 0:
        return m.copy()
    k = np.ones((2 * abs(r) + 1,) * 2, np.uint8)
    return (cv2.dilate if r > 0 else cv2.erode)(m.astype(np.uint8), k) > 0


# ----------------------------------------------------------------------------------------------- legacy
def legacy_gates(ci, obj: dict, vi: int, opt: dict) -> Gates:
    """The gates before the fix rounds, bit-for-bit (pinned by tests/test_gating.py)."""
    own = ci.masks.get((obj["name"], vi))
    cam = ci.cams[vi]
    if own is not None:
        return Gates(accept=dilate(own, int(round(px(opt, "gate_dilate", LEGACY_PX["gate_dilate"], cam, obj)))))
    others = [(o, ci.masks[(o["name"], vi)]) for o in ci.objects if (o["name"], vi) in ci.masks]
    if not others:
        return Gates()
    union = np.any([m for _, m in others], 0)
    clay = np.isin(ci.views[vi].ids, [o["id"] for o, _ in others]).astype(np.uint8)
    dist = cv2.distanceTransform(1 - clay, cv2.DIST_L2, 5)
    halo = np.where((clay == 0) & (dist <= px(opt, "halo_px", LEGACY_PX["halo_px"], cam)),
                    opt.get("halo_factor", LEGACY_HALO_FACTOR), 1.0)
    r = max(1, int(round(px(opt, "reject_dilate", LEGACY_PX["reject_dilate"], cam, obj))))
    return Gates(reject=dilate(union, r), score_mult=halo.astype(np.float32))


# ----------------------------------------------------------------------------------------------- strict
def thin_safe_erosion(own: np.ndarray, e: int) -> int:
    """Largest of (e, 1, 0) px whose erosion keeps at least half of the object's pixels in this view (thin
    objects such as a window frame or a skirting board would otherwise vanish from the view)."""
    n = int(own.sum())
    if n == 0:
        return e
    for r in (e, 1):
        if r <= 0 or dilate(own, -r).sum() >= HALF * n:
            return r
    return 0


def occluder_bands(view, oid: int, margin: int, soft: int, dz: float):
    """(hard, soft) bool: pixels within margin / soft px of a NEARER other object's clay pixels (min-filter of
    the other objects' depth compared with this pixel's depth)."""
    other = (view.ids != oid) & (view.ids > 0)
    dother = np.where(other, view.depth, np.float32(NO_OCCLUDER_DEPTH)).astype(np.float32)
    z = view.depth.astype(np.float32) - dz

    def band(r: int) -> np.ndarray:
        if r <= 0:
            return np.zeros(view.ids.shape, bool)
        return cv2.erode(dother, np.ones((2 * r + 1,) * 2, np.uint8)) < z
    return band(margin), band(max(soft, margin))


def _masked_accept(ci, obj, vi, opt, own_gen, zones) -> np.ndarray:
    cam = ci.cams[vi]
    explicit = "gate_dilate" in obj or "gate_dilate" in opt
    r = int(round(px(opt, "gate_dilate", 0, cam, obj))) if explicit else \
        -int(round(strict_px(opt, "erode_gen", cam)))
    acc = dilate(own_gen, r)
    rd = int(round(strict_px(opt, "other_mask_dilate", cam)))
    for o in ci.objects:
        m = ci.masks.get((o["name"], vi))
        if o["name"] == obj["name"] or m is None:
            continue
        zone = zones.get(o["name"])
        rej = dilate(m, rd)
        if zone is not None:      # contact zone: the photograph shows the occluder there -> take it
            rej &= ~zone
            acc |= zone & dilate(m, -CONTACT_MASK_ERODE)
        acc &= ~rej
    return acc


def _unmasked_accept(ci, obj, vi, opt, zones) -> np.ndarray:
    cam = ci.cams[vi]
    rd = max(1, int(round(strict_px(opt, "reject_dilate", cam, obj))))
    acc = np.ones(ci.views[vi].ids.shape, bool)
    for o in ci.objects:
        m = ci.masks.get((o["name"], vi))
        if o["name"] == obj["name"] or m is None:
            continue
        rej = dilate(m, rd)
        zone = zones.get(o["name"])
        if zone is not None:
            rej &= ~zone
        acc &= ~rej
    return acc


def strict_gates(ci, obj: dict, vi: int, opt: dict, zones: dict | None = None) -> Gates:
    """Strict cross-object gate (module docstring). `zones`: {occluder name: (H,W) bool contact-zone pixels of
    this object} from zones.contact_pixels; inside them the occluder's mask is accepted instead of rejected."""
    zones = zones or {}
    view, cam = ci.views[vi], ci.cams[vi]
    own = view.ids == obj["id"]
    e = thin_safe_erosion(own, int(round(strict_px(opt, "erode_own", cam))))
    acc = dilate(own, -e)
    own_gen = ci.masks.get((obj["name"], vi))
    if own_gen is not None:
        acc &= _masked_accept(ci, obj, vi, opt, own_gen, zones)
    else:
        acc &= _unmasked_accept(ci, obj, vi, opt, zones)
    sfx = "_masked" if own_gen is not None else ""
    hard, soft = occluder_bands(view, obj["id"],
                                int(round(strict_px(opt, "occ_margin" + sfx, cam))),
                                int(round(strict_px(opt, "occ_soft" + sfx, cam))),
                                float(opt.get("occ_dz", STRICT_OTHER["occ_dz"])))
    factor = float(opt.get("occ_soft_factor", STRICT_OTHER["occ_soft_factor"]))
    return Gates(accept=acc & ~hard, score_mult=np.where(soft & ~hard, factor, 1.0).astype(np.float32))


def gates_for_policy(ci, obj: dict, vi: int, opt: dict, zones: dict | None = None) -> Gates:
    if opt.get("gate_policy", DEFAULT_POLICY) == "legacy":
        return legacy_gates(ci, obj, vi, opt)
    return strict_gates(ci, obj, vi, opt, zones)
