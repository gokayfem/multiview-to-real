"""Coherent single-source regions with a hand-over band (detail-critical objects under gate_policy "strict").

Each region gets ONE dominant camera: its `prefer_view` when that camera sees the region, else the camera with
the largest total score over the region. Inside the region the dominant camera's samples rejected only as
grazing / self-edge (visibility.SOFT_ONLY) count as valid. Other cameras may supply only region texels at least
`handover_mm` (3D) from what the dominant camera covers; the band in between stays unobserved and is filled from
the same object. The low band across the hand-over is feathered (`cut` "feather", FEATHER_TEXELS at the 2048
reference) or cut hard (`cut` "hard": single source in both bands).

case.json `objects[].regions`: [{"name", "sphere": [x, y, z, r] | "box": [[x0,y0,z0],[x1,y1,z1]],
optional "prefer_view", "handover_mm" (5), "cut" ("feather")}], world metres; texels inside the shape form the
region. Without `regions` every 3D-connected part is one region with these defaults.
First run (references/seams.md): one camera per face / neck / hand / shoe removed a second camera's displaced
copies; the face used "hard" + 12 mm. It is necessary, not sufficient: a doubled eye and jaw stayed in a
three-quarter view, where the photograph lies on a proxy head of a different shape (references/lessons.md).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from mvb2r.fusion_io import StageError
from mvb2r.shapes import box, capped_list, finite, inside_box, inside_sphere, label, show, sphere

CUTS = ("feather", "hard")
DEFAULT_HANDOVER_MM = 5.0
DEFAULT_CUT = "feather"
FEATHER_TEXELS = 16.0       # low-band feather across the hand-over at the 2048 texture reference
MM = 1e-3
MAX_DOMINANT_POINTS = 200_000   # KD-tree subsample of the dominant camera's coverage
MAX_HANDOVER_MM = 1000.0        # hand-over band limit


def _parse_one(r, where: str, view_names) -> dict:
    if not isinstance(r, dict) or not isinstance(r.get("name"), str) or not r["name"]:
        raise StageError(f"case.json {where} must be an object with a non-empty 'name'")
    unknown = set(r) - {"name", "sphere", "box", "prefer_view", "handover_mm", "cut"}
    if unknown:
        raise StageError(f"case.json {where}: unknown key(s) {show(sorted(map(str, unknown)))}")
    if ("sphere" in r) == ("box" in r):
        raise StageError(f"case.json {where} needs exactly one of 'sphere' or 'box'")
    out = {"name": r["name"], "handover_mm": DEFAULT_HANDOVER_MM, "cut": DEFAULT_CUT, "prefer_view": None}
    out.update({"sphere": sphere(f"{where}.sphere", r["sphere"])} if "sphere" in r else
               {"box": box(f"{where}.box", r["box"])})
    if r.get("prefer_view") is not None:
        if view_names is not None and r["prefer_view"] not in view_names:
            raise StageError(f"case.json {where}.prefer_view {show(r['prefer_view'])} is not a camera "
                             f"({show([label(v) for v in view_names])})")
        out["prefer_view"] = r["prefer_view"]
    if "handover_mm" in r:
        out["handover_mm"] = finite(f"{where}.handover_mm", r["handover_mm"], minimum=0.0,
                                   maximum=MAX_HANDOVER_MM)
    if "cut" in r:
        if r["cut"] not in CUTS:
            raise StageError(f"case.json {where}.cut must be one of {CUTS}, got {show(r['cut'])}")
        out["cut"] = r["cut"]
    return out


def parse_regions(specs, object_name: str, view_names=None) -> tuple:
    """Validated region specs of one object (view names checked when given)."""
    if specs is None:
        return ()
    name = label(object_name)
    specs = capped_list(f"objects[{name}].regions", specs)
    return tuple(_parse_one(r, f"objects[{name}].regions[{i}]", view_names) for i, r in enumerate(specs))


def region_mask(tex, spec: dict) -> np.ndarray:
    return inside_sphere(tex.P, spec["sphere"]) if "sphere" in spec else inside_box(tex.P, spec["box"])


def _part_specs(tex) -> list:
    return [({"name": f"part_{int(p)}", "handover_mm": DEFAULT_HANDOVER_MM, "cut": DEFAULT_CUT,
              "prefer_view": None}, tex.part == p) for p in np.unique(tex.part)]


def _dominant(st, mask: np.ndarray, prefer: int | None) -> tuple:
    """(dominant view or None, other views by total score over the region)."""
    tot = (st.score * mask[None]).sum(1)
    seen = (st.valid[:, mask] | (st.soft[:, mask] > 0)).any(1) if mask.any() else np.zeros(st.V, bool)
    order = [int(v) for v in np.argsort(-tot, kind="stable") if tot[v] > 0]
    dom = prefer if prefer is not None and seen[prefer] else (order[0] if order else None)
    return dom, [v for v in order if v != dom]


def _far_from(P: np.ndarray, dom: np.ndarray, rest: np.ndarray, dist_m: float) -> np.ndarray:
    """(N,) bool: texels of `rest` farther than dist_m (3D) from every texel of `dom`."""
    far = np.zeros(len(P), bool)
    ri = np.nonzero(rest)[0]
    if not len(ri):
        return far
    di = np.nonzero(dom)[0]
    if not len(di) or dist_m <= 0:
        far[ri] = True
        return far
    di = di[:: max(1, len(di) // MAX_DOMINANT_POINTS)]
    d, _ = cKDTree(P[di]).query(P[ri], k=1, distance_upper_bound=dist_m)
    far[ri[~np.isfinite(d)]] = True
    return far


def _apply_region(tex, arrays: dict, spec: dict, mask: np.ndarray, prefer: int | None, scale: float, names):
    """Modify the NEW arrays of coherent_regions in place for one region; returns the region record."""
    st = arrays["st"]
    probe = replace(st, valid=arrays["valid"], score=arrays["score"], soft=arrays["soft"])
    dom, rest = _dominant(probe, mask, prefer)
    feather = 0.0 if spec["cut"] == "hard" else FEATHER_TEXELS * scale
    rec = {"name": spec["name"], "mask": mask, "views": [], "cut": spec["cut"], "feather": feather,
           "handover_mm": spec["handover_mm"], "dominant": None, "soft_extended": 0, "handover_texels": 0}
    if dom is None:
        return rec
    valid, score, soft = arrays["valid"], arrays["score"], arrays["soft"]
    ext = mask & ~valid[dom] & (soft[dom] > 0)
    valid[dom, ext] = True
    score[dom, ext] = soft[dom, ext]
    cover = mask & valid[dom]
    far = _far_from(tex.P, cover, mask & ~cover, spec["handover_mm"] * MM)
    kill = mask & ~far
    others = [i for i in range(st.V) if i != dom]
    cells = np.ix_(others, np.nonzero(kill)[0])
    valid[cells] = False
    score[cells] = 0
    soft[cells] = 0          # else fusion.apply_soft would refill the band from a second camera
    return {**rec, "views": [dom] + rest, "dominant": names[dom] if names else dom,
            "soft_extended": int(ext.sum()), "handover_texels": int(far.sum())}


def coherent_regions(tex, st, specs: tuple, prefer: int | None, scale: float) -> tuple:
    """(new Stack, regions) for a detail-critical object. `specs` from parse_regions (empty: one region per
    3D-connected part); `prefer`: the object's prefer_view index; `scale`: tex_scale (feather texels)."""
    names = list(st.view_names)
    pairs = [(s, region_mask(tex, s)) for s in specs] if specs else _part_specs(tex)
    arrays = {"st": st, "valid": st.valid.copy(), "score": st.score.copy(), "soft": st.soft.copy()}
    regions = []
    for spec, mask in pairs:
        pv = names.index(spec["prefer_view"]) if spec.get("prefer_view") in names else prefer
        regions.append(_apply_region(tex, arrays, spec, mask, pv, scale, names))
    return replace(st, valid=arrays["valid"], score=arrays["score"], soft=arrays["soft"]), tuple(regions)
