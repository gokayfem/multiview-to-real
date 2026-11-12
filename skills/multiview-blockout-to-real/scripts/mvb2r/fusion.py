"""Per-texel fusion of V views: the sample stack, photo-consistency, method REGISTRY and the `auto` choice.

Methods (all in linear light; detail-critical regions always take ONE source per region, never blended):
  best       A  highest score per texel
  blend      B  agreement-gated blend: views within tau_gate (log low-pass) of the reference view, score^p weights
  regions    C  seam-aware labels (Potts MRF over UV cells) + per-region gain levelling + narrow agreement feather
  freqsplit     low band (sigma split_sigma = 16 px at 1664) = normalised-convolution blend of every valid view's
                image-space low-pass, high band = ratio detail of the BEST-view source
  regions-freqsplit  (alias regions+freqsplit) the bake-off winner "C+S8": regions labels give the high band,
                low band sigma auto_split_sigma = 8 px at 1664; `auto` for every class except detail-critical
  (all texel-unit widths scale with R/2048 and image sigmas with calibration/1664, as tuned in the experiment)
  multiband     Laplacian levels in log space, level j blended with the label one-hot feathered by widths[j]
  graphcut      multiband with difference-aware graph-cut seam placement for the labels
  register      bounded per-object flow of each view to the consensus, then multiband
  auto          AUTO_DEFAULT with per-class overrides in AUTO_BY_CLASS; detail-critical objects: best (single source)

`auto` evidence: AUTO_SOURCE below (first-run seam bake-off of the experiment, two real target sets).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np

from mvb2r.visibility import HARD_REASONS

EPS_LOG = 0.02

# `auto` -- update these three lines together when new bake-off evidence lands
AUTO_DEFAULT = "regions-freqsplit"
AUTO_STATUS = "final (first-run bake-off, two target sets)"
AUTO_SOURCE = ("winner C+S8 = regions labels for the high band + frequency-split low band sigma 8 px @1664 "
               "(floor, back wall, sofa, knit, skin at 2048^2; LOO PSNR pixel/view-mean; novel-view coarse seam "
               "step, 8-bit): best 25.02/25.09, 6.54; regions 25.57/25.52, 5.31 (27.92/28.19, 4.76 on the more "
               "consistent set); regions+freqsplit s8 25.64/25.72, 4.49 (28.17/28.24, 4.29); best+freqsplit s16 "
               "25.58/25.63, 4.47 (28.11/28.17, 4.04); multiband 25.52/25.59, 4.43 (28.05/28.14, 4.26); graph cut "
               "adds nothing; gated blend blurs (sharpness 0.61); registration no real LOO gain at 3-10x time. "
               "Sharpness: best 0.83, regions 0.76, regions+split 0.75, best+split 0.79, multiband 0.74. Split "
               "~12.5 s vs multiband ~28 s. Not fixable by fusion: LOO high-band correlation 0.02-0.39.")
AUTO_BY_CLASS: dict[str, str] = {}         # e.g. {"skin": "best"}; detail_critical objects are always single-source
AUTO_EVIDENCE = f"{AUTO_STATUS}: auto = {AUTO_DEFAULT}; {AUTO_SOURCE}"


@dataclass(frozen=True)
class FusionParams:
    tau_consist: float = 0.25    # mean |log lp_a - log lp_b| above which a view is an outlier vs the medoid
    tau_gate: float = 0.20       # blend / feather: views farther than this from the reference are not blended
    blend_power: float = 2.0
    cell: int = 16               # regions: label cell size in texels
    lam: float = 0.25            # regions: Potts weight
    k_consist: float = 0.4
    min_valid_frac: float = 0.3
    null_cost: float = 1.0
    region_penalty: float = 5.0
    level: bool = True
    level_max: float = 0.22
    level_reject: float = 0.35
    level_ridge: float = 0.05
    feather_sigma: float = 2.5
    gate_smooth: float = 2.0
    split_sigma: float = 16.0    # freqsplit: low-band sigma (px at the 1664 reference)
    auto_split_sigma: float = 8.0  # regions-freqsplit (auto): low-band sigma (px at the 1664 reference)
    split_ramp: float = 24.0     # freqsplit/multiband: texels over which a view's weight ramps up from invalid
    mb_sigmas: tuple = (4.0, 16.0, 48.0)
    mb_widths: tuple = (1.5, 6.0, 24.0, 64.0)
    gc_cell: int = 4
    gc_lam: float = 1.0
    reg_max_px: float = 6.0
    reg_smooth: float = 20.0
    reg_iters: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, over: dict | None) -> "FusionParams":
        if not over:
            return self
        known = set(self.to_dict())
        bad = [k for k in over if k not in known]
        if bad:
            raise ValueError(f"unknown fusion parameter(s) {bad}; known: {sorted(known)}")
        cast = {k: (tuple(v) if isinstance(getattr(self, k), tuple) else type(getattr(self, k))(v))
                for k, v in over.items()}
        return replace(self, **cast)


@dataclass(frozen=True)
class Stack:
    """All views' samples for one object's N texels."""
    valid: np.ndarray      # (V,N) bool
    score: np.ndarray      # (V,N) float32
    soft: np.ndarray       # (V,N) float32
    color: np.ndarray      # (V,N,3) float32 linear
    loglp: np.ndarray      # (V,N,3) float32 log(lowpass + EPS_LOG)
    flags: np.ndarray      # (V,N) uint16
    u: np.ndarray          # (V,N) float32 calibration px
    v: np.ndarray
    ppt: np.ndarray        # (V,N) float32
    view_names: tuple = ()

    @property
    def V(self) -> int:
        return self.valid.shape[0]

    @property
    def N(self) -> int:
        return self.valid.shape[1]

    def without(self, k: int) -> "Stack":
        """Same stack with view k removed from consideration (leave-one-view-out)."""
        keep = np.ones(self.V, bool)
        keep[k] = False
        return replace(self, valid=self.valid & keep[:, None], score=self.score * keep[:, None],
                       soft=self.soft * keep[:, None])


def build_stack(samples: list, view_names: list) -> Stack:
    flags = np.stack([s.flags for s in samples])
    valid = (flags & HARD_REASONS) == 0
    return Stack(valid=valid, score=np.where(valid, np.stack([s.score for s in samples]), 0).astype(np.float32),
                 soft=np.stack([s.soft for s in samples]).astype(np.float32),
                 color=np.stack([s.color for s in samples]).astype(np.float32),
                 loglp=np.log(np.stack([s.lowpass for s in samples]) + EPS_LOG).astype(np.float32),
                 flags=flags, u=np.stack([s.u for s in samples]), v=np.stack([s.v for s in samples]),
                 ppt=np.stack([s.ppt for s in samples]), view_names=tuple(view_names))


@dataclass(frozen=True)
class ObjectInput:
    """Everything a fusion method may use for one object."""
    name: str
    cls: str
    tex: object                              # raster.TexelSet
    st: Stack
    regions: tuple = ()                      # ({"name", "mask" (N,) bool, "views" [indices]}, ...)
    bands: Callable | None = None            # bands(sigma, stack) -> (V,N,3) image-space low-pass at the texels
    resample: Callable | None = None         # resample(offsets) -> ObjectInput with re-sampled colours / bands
    targets: tuple | None = None             # per view (linear image at calibration size or None, (H, W))

    def with_stack(self, st: Stack) -> "ObjectInput":
        return replace(self, st=st)


@dataclass
class Result:
    color: np.ndarray          # (N,3) float32 linear
    source: np.ndarray         # (N,) int16 view index of the (high-band) source, -1 = unobserved
    confidence: np.ndarray     # (N,) float32
    observed: np.ndarray       # (N,) bool
    extra: dict = field(default_factory=dict)


REF_TEX = 2048   # texel-unit parameters (cells, feather, ramps, band widths) are tuned at this texture size
REF_CAL = 1664   # image-space sigmas are in pixels of a 1664 px calibration image (the experiment's)


def tex_scale(tex) -> float:
    """Texel-unit parameters are multiplied by R / REF_TEX so the blend covers the same fraction of the surface."""
    return tex.R / REF_TEX


def logdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)), -1)


def consistency(st: Stack, prm: FusionParams, chunk: int = 500_000) -> np.ndarray:
    """(V,N) bool: valid samples that disagree with the per-texel medoid view (needs >= 3 valid views)."""
    V, N = st.V, st.N
    inc = np.zeros((V, N), bool)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        L = st.loglp[:, a:b]
        val = st.valid[:, a:b]
        d = np.mean(np.abs(L[:, None] - L[None, :]), -1)
        cost = np.where(val, np.sum(np.minimum(d, 0.5) * val[None, :, :], 1), np.inf)
        med = np.argmin(cost, 0)
        dmed = d[np.arange(V)[:, None], med[None, :], np.arange(b - a)[None, :]]
        inc[:, a:b] = val & (val.sum(0)[None] >= 3) & (dmed > prm.tau_consist)
    return inc


def force_regions(pref: np.ndarray, st: Stack, regions) -> np.ndarray:
    """Inside single-source regions the first valid view of the region's priority list wins."""
    s = pref.copy()
    for rg in regions:
        done = np.zeros(st.N, bool)
        for v in rg["views"]:
            take = rg["mask"] & ~done & st.valid[v]
            s[:, take] = -1.0
            s[v, take] = 2.0
            done |= take
    return s


def forced_mask(st: Stack, regions) -> np.ndarray:
    m = np.zeros(st.N, bool)
    for rg in regions:
        if rg["views"]:
            m |= rg["mask"] & st.valid[rg["views"]].any(0)
    return m


def method_best(obj: ObjectInput, prm: FusionParams) -> Result:
    st = obj.st
    s = force_regions(np.where(st.valid, st.score, -1.0), st, obj.regions)
    lab = np.argmax(s, 0)
    obs = st.valid.any(0)
    idx = np.arange(st.N)
    return Result(color=st.color[lab, idx], source=np.where(obs, lab, -1).astype(np.int16),
                  confidence=np.where(obs, st.score[lab, idx], 0).astype(np.float32), observed=obs)


def method_blend(obj: ObjectInput, prm: FusionParams) -> Result:
    from mvb2r.levels import uv_blur
    st, idx = obj.st, np.arange(obj.st.N)
    inc = consistency(st, prm)
    pref = np.where(st.valid & ~inc, st.score, -1.0)
    pref = np.where((pref.max(0) < 0)[None], np.where(st.valid, st.score, -1.0), pref)
    ref = np.argmax(force_regions(pref, st, obj.regions), 0)
    obs = st.valid.any(0)
    Lref = st.loglp[ref, idx]
    w = np.zeros((st.V, st.N), np.float32)
    for v in range(st.V):
        agree = st.valid[v] & (logdist(st.loglp[v], Lref) <= prm.tau_gate)
        if prm.gate_smooth > 0:  # the UV neighbourhood's majority decides (no speckle)
            agree = st.valid[v] & (uv_blur(obj.tex, agree.astype(np.float32), prm.gate_smooth) > 0.5)
        w[v] = np.where(agree, np.maximum(st.score[v], 1e-4) ** prm.blend_power, 0)
    single = forced_mask(st, obj.regions)
    w[:, single] = 0
    w[ref, idx] = np.where(obs, np.maximum(w[ref, idx], 1e-6), 0)
    ws = w.sum(0)
    col = np.einsum("vn,vnc->nc", w, st.color) / np.maximum(ws, 1e-12)[:, None]
    conf = np.sum(w * st.score, 0) / np.maximum(ws, 1e-12)
    return Result(color=col.astype(np.float32), source=np.where(obs, ref, -1).astype(np.int16),
                  confidence=np.where(obs, conf, 0).astype(np.float32), observed=obs,
                  extra={"n_blended": np.where(obs, (w > 0).sum(0), 0).astype(np.uint8)})


def apply_soft(res: Result, st: Stack) -> Result:
    """Texels with no valid view but a grazing / self-edge sample take the best such sample, low confidence."""
    cand = ~res.observed & (st.soft.max(0) > 0)
    extra = dict(res.extra, soft=cand)
    if not cand.any():
        return replace(res, extra=extra)
    k = np.nonzero(cand)[0]
    best = np.argmax(st.soft[:, k], 0)
    color, source, conf = res.color.copy(), res.source.copy(), res.confidence.copy()
    color[k] = st.color[best, k]
    source[k] = best
    conf[k] = 0.02
    return Result(color=color, source=source, confidence=conf, observed=res.observed | cand, extra=extra)


def _registry() -> dict:
    from mvb2r import levels, regions, seams
    return {"best": method_best, "blend": method_blend, "regions": regions.method_regions,
            "freqsplit": levels.method_freqsplit, "regions-freqsplit": levels.method_regions_freqsplit,
            "multiband": levels.method_multiband,
            "graphcut": seams.method_graphcut, "register": seams.method_register}


METHODS = ("best", "blend", "regions", "freqsplit", "regions-freqsplit", "multiband", "graphcut", "register")
LETTERS = {"A": "best", "B": "blend", "C": "regions", "regions+freqsplit": "regions-freqsplit",
           "C+S8": "regions-freqsplit"}


def resolve_method(name: str, cls: str = "generic", detail_critical: bool = False,
                   class_overrides: dict | None = None) -> str:
    """'auto' (or a letter) -> a concrete registry name. Detail-critical objects are always single-source."""
    name = LETTERS.get(name, name)
    if name != "auto":
        if name not in METHODS:
            raise ValueError(f"unknown fusion method {name!r}; known: {', '.join(METHODS + ('auto',))}")
        return name
    if detail_critical:
        return "best"
    table = {**AUTO_BY_CLASS, **(class_overrides or {})}
    return resolve_method(table.get(cls, AUTO_DEFAULT), cls, False, None)


def fuse(obj: ObjectInput, method: str, prm: FusionParams) -> Result:
    fn = _registry()[method]
    res = fn(obj, prm)
    return apply_soft(res, obj.st)
