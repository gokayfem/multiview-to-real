"""`targets`: freeze one image per view at calibration resolution, with alignment confidence and object masks.

    mvb2r.py targets CASE                 accepted refined view, else the generated cell
    mvb2r.py targets CASE --images DIR    your own photoreal images of exactly these views (view_XX.png),
                                          generation skipped; everything downstream runs offline

Writes targets/view_XX.png (calibration resolution), targets/align_conf_XX.png (0..255), targets/
mask_<object>_XX.png (generated masks), targets/targets.json. The native file stays where it is and its path,
size and sha256 are recorded. Shrinking a larger native is an area average of real detail; enlarging a
smaller one is a Lanczos resize and is labelled "resized, NOT generated detail".

Every frozen view is gated (acceptance.json "cell": object IoU, a reliably segmented object, and the image must
match its own clay better than any other view's). A failing view gets "usable": false and "status": "rejected" in
targets.json, its files go to targets/rejected/, and fewer than 2 usable views stops the stage.

align_conf: 255 where the generated outlines agree with the clay object_id, falling where a generated mask
disagrees (weighted by its reliability), blurred over a few pixels; the outline band of objects whose mask is
"prior only" or "low" is capped at 128 because their real outline is unknown.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from . import common, qa

REL_WEIGHT = {"high": 1.0, "moderate": 0.6, "low": 0.3, "prior only": 0.0}


def register(sub) -> None:
    p = sub.add_parser("targets", help="freeze targets: calibration-size images, align_conf, masks, targets.json")
    p.add_argument("case")
    p.add_argument("--images", default=None, help="folder with your own view_XX.png images")
    p.set_defaults(func=run)


def align_conf(oid: np.ndarray, seg: dict) -> np.ndarray:
    import cv2
    h, w = oid.shape
    size = max(h, w)
    dis = np.zeros(oid.shape, np.float32)
    cap = np.zeros(oid.shape, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band = max(1, int(round(0.0036 * size)))
    for i, r in seg.items():
        clay = oid == i
        wgt = REL_WEIGHT.get(r["reliability"], 0.0)
        dis = np.maximum(dis, wgt * (r["mask"] ^ clay).astype(np.float32))
        if wgt < 0.5:
            m = clay.astype(np.uint8)
            cap |= (cv2.dilate(m, k, iterations=band) - cv2.erode(m, k, iterations=band)) > 0
    r = max(1, int(round(0.0027 * size)))
    dis = cv2.dilate(dis, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    dis = cv2.GaussianBlur(dis, (0, 0), max(0.5, 0.0036 * size))
    conf = np.clip(255 * (1 - np.clip(dis * 1.5, 0, 1)), 0, 255)
    conf = np.where(cap, np.minimum(conf, 128), conf)
    return conf.astype(np.uint8)


def freeze_view(case, cam: dict, objects: list[dict], native: Path, out_dir: Path, info: dict,
                record_dir: Path | None = None) -> dict:
    """Resample one native image to the calibration size, segment it, write the target files into out_dir;
    paths are recorded as if under record_dir (the folder they will be swapped into)."""
    rec = record_dir or out_dir
    W, H = int(cam["width"]), int(cam["height"])
    idx = common.view_index(cam["name"])
    rgb = common.read_rgb(native)
    calib, method = common.resize_rgb(rgb, (W, H))
    if info.get("status") != "user_image" and method == "area downsample":
        method = "area downsample of the generated native (resampled; not new detail)"
    vp = common.view_passes(case, cam)
    metrics, seg = qa.view_qa(calib, vp["oid"], objects, qa.load_landmarks(case), cam, vp["depth"])
    common.write_rgb(out_dir / f"{cam['name']}.png", calib)
    common.write_gray(out_dir / f"align_conf_{idx:02d}.png", align_conf(vp["oid"], seg))
    by_id = {int(o["id"]): o for o in objects}
    masks = {}
    for i, r in seg.items():
        name = by_id[i]["name"]
        p = out_dir / f"mask_{common.file_token(name)}_{idx:02d}.png"
        common.write_gray(p, r["mask"].astype(np.uint8) * 255)
        masks[name] = {"path": common.rel(case, rec / p.name), "pixels": int(r["mask"].sum()), "reliability": r["reliability"]}
    h, w = rgb.shape[:2]
    return {**info, "target_png": common.rel(case, rec / f"{cam['name']}.png"),
            "align_conf_png": common.rel(case, rec / f"align_conf_{idx:02d}.png"),
            "native_path": common.rel(case, native), "native_size": [w, h], "native_sha256": common.sha256_file(native),
            "calibration_size": [W, H], "resize": method, "masks": masks, "qa": metrics}


def _generated_sources(case, cams: list[dict]) -> dict:
    cells = {c["view"]: c for c in common.read_json(common.require(case / "gen" / "cells" / "cells.json",
                                                                   f"mvb2r.py generate {case}"))["cells"]}
    dpath = case / "gen" / "refined" / "decisions.json"
    dec = common.read_json(dpath) if dpath.is_file() else {}
    out = {}
    for cam in cams:
        d = dec.get(cam["name"], {})
        if d.get("accepted"):
            att = next(a for a in d["attempts"] if a["path"] == d["accepted"])
            out[cam["name"]] = (case / d["accepted"], {"status": "accepted_refined", "request_id": att.get("request_id"),
                                                       "checks": att["checks"]})
        else:
            why = "no refine attempt" if not d else "refine attempts failed the acceptance rule"
            out[cam["name"]] = (case / cells[cam["name"]]["native"], {"status": "fallback_grid_cell", "why": why})
    return out


def _user_sources(case, cams: list[dict], folder: Path) -> dict:
    folder = Path(folder).expanduser().resolve()
    out = {}
    for cam in cams:
        cands = [folder / f"{cam['name']}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")]
        src = next((c for c in cands if c.is_file()), None)
        if src is None:
            raise common.StageError(f"--images: {folder} has no {cam['name']}.png (one image per view, named like the views)")
        w, h = common.image_size(src)
        ar, want = w / h, int(cam["width"]) / int(cam["height"])
        if abs(ar / want - 1) > 0.02:
            raise common.StageError(f"{src.name} is {w}x{h} (aspect {ar:.3f}); the view's aspect is {want:.3f}")
        dst = case / "input" / "user_targets" / f"{cam['name']}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        out[cam["name"]] = (dst, {"status": "user_image", "source": src.name})
    return out


def swap_in(case, tmp: Path, final: Path) -> None:
    """Replace `final` by the complete folder `tmp` (never a mix of old and new files)."""
    import shutil
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


def generation_of(final: Path) -> int:
    p = final / "targets.json"
    return int(common.read_json(p).get("generation", 0)) if p.is_file() else 0


def gate_frozen(case, cams, views: dict, tmp: Path, out_dir: Path) -> dict:
    """Gate every frozen view (qa.gate_views); a rejected one is moved to targets/rejected/ with usable: false."""
    imgs = {v: common.read_rgb(tmp / Path(t["target_png"]).name) for v, t in views.items()}
    gates = qa.gate_views(case, cams, imgs, {v: t["qa"] for v, t in views.items()})
    out = {}
    for v, t in views.items():
        g = gates[v]
        if g["usable"]:
            out[v] = {**t, "usable": True, "gate": g}
            continue
        (tmp / "rejected").mkdir(exist_ok=True)
        moved = {}
        for key in ("target_png", "align_conf_png"):
            name = Path(t[key]).name
            (tmp / name).replace(tmp / "rejected" / name)
            moved[key] = common.rel(case, out_dir / "rejected" / name)
        masks = {}
        for k, m in t["masks"].items():
            name = Path(m["path"]).name
            (tmp / name).replace(tmp / "rejected" / name)
            masks[k] = {**m, "path": common.rel(case, out_dir / "rejected" / name)}
        out[v] = {**t, **moved, "masks": masks, "usable": False, "status": "rejected",
                  "generation_status": t.get("status"), "gate": g, "next_step": qa.NEXT_STEP}
    qa.say_rejected("targets", gates)
    return out


def freeze_all(case, cams, objects, sources: dict, out_dir: Path, label: str) -> dict:
    """Freeze every view into a temporary folder, then swap it in as out_dir."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    views = {}
    for cam in cams:
        native, info = sources[cam["name"]]
        views[cam["name"]] = freeze_view(case, cam, objects, native, tmp, info, out_dir)
    views = gate_frozen(case, cams, views, tmp, out_dir)
    usable = [v for v, t in views.items() if t["usable"]]
    if len(usable) < 2:
        shutil.rmtree(tmp)
        raise common.StageError(f"only {len(usable)} usable view(s) ({', '.join(usable) or 'none'}); at least 2 are needed. "
                                f"{qa.NEXT_STEP}; previous targets/ left unchanged")
    doc = {"frozen": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": label, "generation": generation_of(out_dir) + 1,
           "views": views, "note": "targets are at calibration resolution; enlargement is a resize, never generated detail"}
    common.write_json(tmp / "targets.json", doc)
    swap_in(case, tmp, out_dir)
    return doc


def run(args) -> int:
    case = common.case_dir(args.case)
    cams = common.load_cameras(case)
    objects = common.case_objects(case)
    if args.images:
        sources, label = _user_sources(case, cams, args.images), "user images"
    else:
        sources, label = _generated_sources(case, cams), "generated"
    doc = freeze_all(case, cams, objects, sources, case / "targets", label)
    for v, t in doc["views"].items():
        common.say(f"[targets] {v}: {t['status']}{'' if t['usable'] else ' (NOT usable, in targets/rejected/)'} native {t['native_size'][0]}x{t['native_size'][1]} -> "
                   f"{t['calibration_size'][0]}x{t['calibration_size'][1]} ({t['resize']})")
    return 0
