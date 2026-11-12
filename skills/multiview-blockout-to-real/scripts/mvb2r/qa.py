"""Alignment QA of a generated view against its clay view, per object and per group.

All measurements at the calibration resolution (the generated image is resampled to it first; a resize is
recorded, never counted as detail):
  per object  IoU and symmetric boundary chamfer (px) of the generated mask vs the clay object_id region,
              mask reliability (masks.py), pixel counts
  per group   the same on the union of the objects sharing case.json "group"
  landmarks   (views/landmarks.json, optional) for each visible landmark: how far the generated outline of
              its object moved at that point = distance(landmark, generated outline) - distance(landmark, clay outline)
  colour      dE76 per object between two images of the same view (refined vs cell)
"""
from __future__ import annotations

import numpy as np

from . import cameras as cam_mod
from . import common
from . import masks as masks_mod


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    import cv2
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return outside - inside


def project_landmarks(landmarks: dict, cam: dict, depth: np.ndarray, tol: float = 0.02) -> dict:
    """{name: {u, v, visible, object}} of landmarks in one view (visible = not occluded per depth_z)."""
    out = {}
    H, W = depth.shape
    for name, lm in landmarks.items():
        uvz = cam_mod.project(cam["P"], [lm["world"]])[0]
        u, v, z = float(uvz[0]), float(uvz[1]), float(uvz[2])
        inside = z > 0 and 0 <= u < W and 0 <= v < H
        vis = bool(inside and depth[int(v), int(u)] >= z - tol)
        out[name] = {"u": round(u, 2), "v": round(v, 2), "visible": vis, "object": lm.get("object")}
    return out


def landmark_offsets(projected: dict, gen: dict, clay_ids: np.ndarray, name_to_id: dict) -> dict:
    out = {}
    for name, p in projected.items():
        oid = name_to_id.get(p.get("object"))
        if not p["visible"] or oid is None or oid not in gen:
            continue
        iu, iv = int(p["u"]), int(p["v"])
        dg = float(_signed_distance(gen[oid]["mask"])[iv, iu])
        dc = float(_signed_distance(clay_ids == oid)[iv, iu])
        out[name] = {"object": p["object"], "u": p["u"], "v": p["v"], "offset_px": round(dg - dc, 2)}
    return out


def view_qa(rgb: np.ndarray, oid: np.ndarray, objects: list[dict], landmarks: dict | None = None,
            cam: dict | None = None, depth: np.ndarray | None = None):
    """(metrics dict, masks dict) of one generated view at calibration resolution."""
    by_id = {int(o["id"]): o for o in objects}
    seg = masks_mod.segment_view(rgb, oid, ids=set(by_id))
    per_obj = {}
    for i, r in seg.items():
        clay = oid == i
        o = by_id[i]
        per_obj[o["name"]] = {"id": i, "clay_px": int(clay.sum()), "gen_px": int(r["mask"].sum()),
                              "iou": round(masks_mod.iou(r["mask"], clay), 4), "chamfer_px": masks_mod.chamfer(r["mask"], clay),
                              "reliability": r["reliability"], "separability": r["separability"], "method": r["method"]}
    groups = {}
    for g in sorted({o.get("group") for o in objects if o.get("group")}):
        ids = [int(o["id"]) for o in objects if o.get("group") == g]
        clay = np.isin(oid, ids)
        gen = np.zeros_like(clay)
        for i in ids:
            if i in seg:
                gen |= seg[i]["mask"]
        if clay.sum() >= 50:
            groups[g] = {"iou": round(masks_mod.iou(gen, clay), 4), "chamfer_px": masks_mod.chamfer(gen, clay)}
    lms = {}
    if landmarks and cam is not None and depth is not None:
        proj = project_landmarks(landmarks, cam, depth)
        lms = landmark_offsets(proj, seg, oid, {o["name"]: int(o["id"]) for o in objects})
    return {"objects": per_obj, "groups": groups, "landmarks": lms}, seg


def colour_delta(a_rgb: np.ndarray, b_rgb: np.ndarray, oid: np.ndarray, objects: list[dict], min_px: int = 50) -> dict:
    """Mean CIE76 dE per object between two images of one view (same size)."""
    from skimage.color import rgb2lab
    if a_rgb.shape != b_rgb.shape:
        raise ValueError("colour_delta needs two images of the same size")
    de = np.linalg.norm(rgb2lab(a_rgb) - rgb2lab(b_rgb), axis=2)
    out = {}
    for o in objects:
        m = oid == int(o["id"])
        if m.sum() >= min_px:
            out[o["name"]] = round(float(de[m].mean()), 2)
    return out


def load_landmarks(case) -> dict:
    p = common.views_dir(case) / "landmarks.json"
    if not p.is_file():
        return {}
    data = common.read_json(p)
    return {k: v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("world"), list) and len(v["world"]) == 3}


def scaled(value_at_1664: float, width: int) -> float:
    """A pixel threshold set at the experiment's 1664 px, scaled to this calibration width."""
    return value_at_1664 * width / 1664.0


# ---------------------------------------------------------------- per-view usability gate
def view_summary(metrics: dict, rule: dict, n_px: int) -> dict:
    """Mean / min IoU over objects that are big enough AND reliably segmented, or the reason there is none."""
    checked = [o for o in metrics["objects"].values()
               if o["clay_px"] >= rule["min_object_fraction"] * n_px and o["reliability"] in rule["checked_reliabilities"]]
    if len(checked) < rule.get("min_reliable_objects", 1):
        rel = sorted({o["reliability"] for o in metrics["objects"].values()})
        return {"mean_iou": None, "min_iou": None, "reliable_objects": len(checked),
                "iou_reason": f"IoU not measurable: no object's generated mask is reliable (reliabilities: "
                              f"{', '.join(rel) or 'none'}); the image may not show this view"}
    ious = [o["iou"] for o in checked]
    worst = min(checked, key=lambda o: o["iou"])
    name = next(n for n, o in metrics["objects"].items() if o is worst)
    return {"mean_iou": round(float(np.mean(ious)), 4), "min_iou": round(float(min(ious)), 4), "min_iou_object": name,
            "reliable_objects": len(checked), "iou_reason": None}


def _edge_feature(rgb: np.ndarray, size: int = 96) -> np.ndarray:
    import cv2
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    f = cv2.GaussianBlur(mag, (0, 0), 2.0).ravel()
    f = f - f.mean()
    return f / max(float(np.linalg.norm(f)), 1e-9)


def edge_offset(img: np.ndarray, clay: np.ndarray, size: int = 256) -> dict:
    """Global shift (fraction of the width / height) of an image's edges relative to its clay: phase correlation."""
    import cv2

    def feat(rgb):
        g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1.5)
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(feat(clay), feat(img), win)
    return {"dx": round(float(dx) / size, 4), "dy": round(float(dy) / size, 4), "response": round(float(resp), 4)}


def panel_match(images: dict, clays: dict) -> dict:
    """For each view: edge correlation of its image with its OWN clay and with the best OTHER clay (pure)."""
    fi = {v: _edge_feature(im) for v, im in images.items()}
    fc = {v: _edge_feature(im) for v, im in clays.items()}
    out = {}
    for v, f in fi.items():
        corr = {u: float(f @ c) for u, c in fc.items()}
        others = {u: c for u, c in corr.items() if u != v}
        best = max(others, key=others.get) if others else None
        twins = {u: {"images": round(float(f @ fi[u]), 4), "clays": round(float(fc[v] @ fc[u]), 4)} for u in fi if u != v}
        out[v] = {"own": round(corr[v], 4), "best_other": best, "best_other_corr": round(others[best], 4) if best else None,
                  "pairs": twins, "offset": edge_offset(images[v], clays[v])}
    for v, m in out.items():                  # a near-duplicate pair: the one that matches its own clay worse is the copy
        cands = {u: p["images"] - p["clays"] for u, p in m["pairs"].items() if out[u]["own"] > m["own"]}
        m["duplicate_of"] = max(cands, key=cands.get) if cands else None
    return out


def view_gate(summary: dict, match: dict | None, rule: dict) -> dict:
    """{"usable": bool, "reasons": [...]} of one generated view (pure)."""
    reasons = []
    if summary["iou_reason"]:
        reasons.append(summary["iou_reason"])
    else:
        if summary["mean_iou"] < rule["min_mean_iou"]:
            reasons.append(f"mean object IoU {summary['mean_iou']:.3f} < {rule['min_mean_iou']}")
        if summary["min_iou"] < rule["min_object_iou"]:
            reasons.append(f"{summary['min_iou_object']} IoU {summary['min_iou']:.3f} < {rule['min_object_iou']}")
    if match and match.get("best_other") and match["best_other_corr"] > match["own"] + rule["panel_match_margin"]:
        reasons.append(f"the image matches the clay of {match['best_other']} ({match['best_other_corr']:.3f}) better than its "
                       f"own ({match['own']:.3f}): a swapped or wrong panel")
    for u, p in ((match or {}).get("pairs") or {}).items():
        if (u == match.get("duplicate_of") and p["images"] >= rule["duplicate_min_corr"]
                and p["images"] - p["clays"] >= rule["duplicate_excess_over_clays"]):
            reasons.append(f"near-duplicate of {u}'s image (edge correlation {p['images']:.3f}) although their clays differ "
                           f"({p['clays']:.3f})")
    off = (match or {}).get("offset")
    lim = rule.get("max_offset_frac")
    if off and lim and off["response"] >= rule.get("offset_min_response", 0.1) and max(abs(off["dx"]), abs(off["dy"])) > lim:
        reasons.append(f"content offset vs clay ({off['dx']:+.3f}, {off['dy']:+.3f}) of the frame > {lim}: shifted panel")
    return {"usable": not reasons, "reasons": reasons}


NEXT_STEP = "retry `mvb2r.py generate CASE --attempt 2`, or drop / move this camera"


def cell_rule(case) -> dict:
    """acceptance.json "cell" of the case, else the template's (older cases have none)."""
    p = common.case_dir(case) / "acceptance.json"
    rule = common.read_json(p).get("cell") if p.is_file() else None
    return rule or common.read_json(common.TEMPLATES_DIR / "acceptance.json")["cell"]


def gate_views(case, cams: list[dict], images: dict, metrics: dict) -> dict:
    """{view: {"summary", "match", "usable", "reasons"}} for a set of calibration-size images."""
    rule = cell_rule(case)
    clays = {c["name"]: common.read_rgb(common.views_dir(case) / c["name"] / "clay.png") for c in cams}
    match = panel_match(images, {v: clays[v] for v in images})
    out = {}
    for c in cams:
        v = c["name"]
        if v not in images:
            continue
        s = view_summary(metrics[v], rule, int(c["width"]) * int(c["height"]))
        out[v] = {"summary": s, "match": match.get(v), **view_gate(s, match.get(v), rule)}
    return out


def say_rejected(stage: str, gates: dict) -> None:
    for v, g in gates.items():
        if not g["usable"]:
            common.say(f"[{stage}] WARNING {v} rejected: {'; '.join(g['reasons'])} -> {NEXT_STEP}")
