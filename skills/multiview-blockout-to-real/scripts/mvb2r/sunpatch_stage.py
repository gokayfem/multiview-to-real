"""`pbr` step `synth_direct` (opt-in): replace the generated direct-light patch in plain, repetitive materials.

    case.json  "pbr": {"synth_direct": false, "synth_objects": ["Floor"], "synth_classes": [],
                       "synth_params": {"k_bright": 1.6, "dilate_m": 0.2, "feather": 24}}
    mvb2r.py pbr CASE --synth-direct | --no-synth-direct      (CLI wins over case.json)

Objects: NAMED only by default (`synth_objects`); `synth_classes` is an explicit opt-in (objects of those classes,
not detail-critical). The step is on with neither set: StageError. A class default touched furniture, frames
and walls with only soft window light in the first real run; the surfaces carrying the photo's own direct light
are a judgement call made on the relight sheet.
B (texels the proxy's key light reaches) comes from the shading basis: <view>_sundirect.exr (direct SUN light
only), per view thresholded to a 0/1 lit image and projected through the same visibility as the textures.
Writes pbr/synth/direct_views/, pbr/textures/<object>/{base_color,height,synth_mask}.png (synthesised; what
final.blend and the relight test use) and pbr/textures_nosynth/<object>/ (the same textures without synthesis).
Pure parts: sunpatch.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from mvb2r.case import CLASSES
from mvb2r.common import inside_case, owned_path
from mvb2r.fusion_io import (StageError, lum, read_exr, read_gray01, read_ids, read_rgb_linear, safe_name,
                             write_png, write_srgb_png)
from mvb2r.materials import height_from
from mvb2r.sunpatch import (LOWPASS_FRAC, SynthParams, bright_mask, combine_masks, dilate_texels,
                            direct_from_stack, direct_lit_view, synthesise)

PARAM_RANGES = {"k_bright": (float, 1.0, 10.0), "dilate_m": (float, 0.0, 5.0), "feather": (int, 0, 1024)}
NOSYNTH_DIR = "textures_nosynth"
DIRECT_SUFFIX, FALLBACK_SUFFIX = "_sundirect.exr", "_sun.exr"


def _list_of_str(v, key: str) -> list:
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise StageError(f"case.json pbr.{key} must be a list of names")
    return list(v)


def _params(p) -> dict:
    if not isinstance(p, dict):
        raise StageError("case.json pbr.synth_params must be an object")
    unknown = set(p) - set(PARAM_RANGES)
    if unknown:
        raise StageError(f"case.json pbr.synth_params: unknown {sorted(unknown)}; known {sorted(PARAM_RANGES)}")
    out = {"k_bright": SynthParams.k_bright, "dilate_m": SynthParams.dilate_m, "feather": SynthParams.feather}
    for k, v in p.items():
        typ, lo, hi = PARAM_RANGES[k]
        ok_type = isinstance(v, int) and not isinstance(v, bool) if typ is int else \
            isinstance(v, (int, float)) and not isinstance(v, bool)
        if not ok_type or not (lo < v <= hi if k == "k_bright" else lo <= v <= hi):
            raise StageError(f"case.json pbr.synth_params.{k} = {v!r}: needs {typ.__name__} in "
                             f"{'(' if k == 'k_bright' else '['}{lo}, {hi}]")
        out[k] = typ(v)
    return out


def synth_config(cfg: dict, cli: bool | None) -> dict | None:
    """Validated settings, or None when the step is off. Always validates the pbr.synth_* keys."""
    p = cfg.get("pbr", {}) or {}
    if not isinstance(p, dict):
        raise StageError("case.json 'pbr' must be an object")
    on = p.get("synth_direct", False)
    if not isinstance(on, bool):
        raise StageError("case.json pbr.synth_direct must be true/false")
    objects = _list_of_str(p.get("synth_objects", []), "synth_objects")
    classes = _list_of_str(p.get("synth_classes", []), "synth_classes")
    bad = [c for c in classes if c not in CLASSES]
    if bad:
        raise StageError(f"case.json pbr.synth_classes: {bad} not in {CLASSES}")
    conf = {**_params(p.get("synth_params", {})), "objects": objects, "classes": classes}
    if not (on if cli is None else cli):
        return None
    if not objects and not classes:
        raise StageError("pbr synth_direct is on but names no surface: look at the relight sheet (out/relight.jpg) "
                         "and list the objects that carry the generated photo's own direct light (a sun patch) in "
                         "case.json pbr.synth_objects (or opt in by class with pbr.synth_classes)")
    return conf


def select_objects(objects: dict, conf: dict) -> list[str]:
    """Projected, non-opening objects of the synth classes (not detail-critical) plus the named ones."""
    unknown = [n for n in conf["objects"] if n not in objects or not objects[n].get("projected")]
    if unknown:
        raise StageError(f"case.json pbr.synth_objects: {unknown} are not projected objects of this case")
    return [n for n, r in objects.items() if r.get("projected") and not r.get("opening") and
            (n in conf["objects"] or (r.get("class") in conf["classes"] and not r.get("detail_critical")))]


def write_direct_views(case: Path, views: list, shading_dir: Path, out: Path) -> dict:
    """Per view a 0/1 image of the object pixels the key light reaches directly (projected next)."""
    used, out = {}, owned_path(case, out)  # a planted link is refused before anything is read or written
    for v in views:
        p = shading_dir / f"{v}{DIRECT_SUFFIX}"
        if not p.is_file():
            p = shading_dir / f"{v}{FALLBACK_SUFFIX}"
        if not p.is_file():
            raise StageError(f"missing {shading_dir / (v + DIRECT_SUFFIX)} (written by the pbr shading step)")
        ids = read_ids(case / "views" / v / "object_id.png")
        D = cv2.resize(lum(read_exr(p)).astype(np.float32), ids.shape[::-1], interpolation=cv2.INTER_LINEAR)
        lit = direct_lit_view(D, ids)
        write_png(owned_path(case, out / f"{v}.png"), np.repeat(np.where(lit, 255, 0).astype(np.uint8)[..., None], 3, -1))
        used[v] = "direct" if p.name.endswith(DIRECT_SUFFIX) else "sun total (no direct-only pass: older basis)"
    return used


def clear_previous(case: Path) -> None:
    """Outputs of an earlier synth run (this stage owns them): never mix them with a new run. Every path is
    checked with owned_path first (no deleting through a symbolic link planted in the case)."""
    dirs = [owned_path(case, f"pbr/{NOSYNTH_DIR}"), owned_path(case, "pbr/synth")]
    textures = owned_path(case, "pbr/textures")
    masks = [owned_path(case, d / "synth_mask.png") for d in (textures.iterdir() if textures.is_dir() else ())]
    for d in dirs:
        if d.is_dir():
            shutil.rmtree(d)
    for f in masks:
        if f.is_file():
            f.unlink()


def run_synth(case: Path, cfg: dict, conf: dict, alb: dict, method: str, views: list, shading_dir: Path) -> dict:
    from mvb2r.projinputs import load_case_inputs
    clear_previous(case)
    names = select_objects(alb["objects"], conf)
    rep = {"enabled": True, "ran": False, "params": conf, "textures_nosynth": f"pbr/{NOSYNTH_DIR}", "objects": {}}
    if not names:
        return {**rep, "note": "no object selected (pbr.synth_classes / pbr.synth_objects)"}
    rep["direct_basis"] = write_direct_views(case, views, shading_dir, case / "pbr" / "synth" / "direct_views")
    ci = load_case_inputs(case, cfg, case / "pbr" / "synth" / "direct_views")
    rows = {o["name"]: o for o in ci.objects}
    for n in names:
        rep["objects"][n] = synth_object(case, ci, rows[n], alb["objects"][n], method, conf)
    rep["ran"] = any(o["synth_texels"] > 0 for o in rep["objects"].values())
    return rep


def _texture_inputs(case: Path, name: str, method: str):
    sn = safe_name(name)
    base = read_rgb_linear(case / "pbr" / "textures" / sn / "base_color.png")
    R = base.shape[0]
    observed = read_gray01(case / "pbr" / "albedo" / method / sn / "coverage.png") > 0.5
    rawdir = case / "proj" / method / sn
    if not (rawdir / "texture.png").is_file():
        raise StageError(f"missing {rawdir / 'texture.png'} (the raw projection of `mvb2r.py project`)")
    raw, raw_obs = read_rgb_linear(rawdir / "texture.png"), read_gray01(rawdir / "coverage.png")
    if raw.shape[0] != R:  # same object, other atlas size: compare at the PBR size
        raw = cv2.resize(raw, (R, R), interpolation=cv2.INTER_AREA)
        raw_obs = cv2.resize(raw_obs, (R, R), interpolation=cv2.INTER_NEAREST)
    return base, observed, raw, raw_obs > 0.5, R


def direct_texture(ci, row: dict, R: int):
    """(TexelSet, (R,R) bool B): the per-view lit images of `ci` projected with the textures' visibility."""
    from mvb2r.projinputs import fusion_options, object_input
    from mvb2r.visibility import VisParams
    oi = object_input(ci, row, R, VisParams(), fusion_options(ci.cfg)[2])
    return oi.tex, oi.tex.scatter(direct_from_stack(oi.st.valid, oi.st.score, oi.st.color), False)


def synth_object(case: Path, ci, row: dict, rec: dict, method: str, conf: dict) -> dict:
    base, observed, raw, raw_obs, R = _texture_inputs(case, row["name"], method)
    tex, B = direct_texture(ci, row, R)
    inside = tex.mask()
    A, med = bright_mask(lum(raw), raw_obs, inside, conf["k_bright"], max(R * LOWPASS_FRAC, 1.0))
    rad = dilate_texels(conf["dilate_m"], tex.area, R)
    feather = min(conf["feather"], rad) if rad > 0 else conf["feather"]
    prm = SynthParams(k_bright=conf["k_bright"], dilate_m=conf["dilate_m"], feather=feather)
    new, synth, r = synthesise(base, combine_masks(A, B, inside, rad), observed, inside, prm,
                               tex.scatter(tex.Ng, 0.0))
    write_outputs(case, row["name"], base, new, synth)
    w = tex.weight
    newly = float(w[tex.gather(synth & observed)].sum() / max(w.sum(), 1e-12))
    before = float(rec.get("coverage", {}).get("inferred", 0.0))
    return {**r, "class": rec.get("class"), "from_bright_patch": int(A.sum()), "from_direct_light": int(B.sum()),
            "observed_median_lum": med, "dilate_texels": rad, "feather_texels": feather,
            "mask_area_frac": float(r["mask_texels"] / max(inside.sum(), 1)),
            "inferred_before": before, "inferred_after": before + newly}


def write_outputs(case: Path, name: str, base: np.ndarray, new: np.ndarray, synth: np.ndarray) -> None:
    sn = safe_name(name)
    tdir, keep = owned_path(case, f"pbr/textures/{sn}"), owned_path(case, f"pbr/{NOSYNTH_DIR}/{sn}")
    files = ("base_color.png", "height.png", "synth_mask.png")
    dst = {f: owned_path(case, tdir / f) for f in files}  # refuse before writing anything
    pairs = [(inside_case(case, tdir / f), owned_path(case, keep / f)) for f in files[:2]]
    keep.mkdir(parents=True, exist_ok=True)
    for src, to in pairs:
        shutil.copy2(src, to)
    write_png(dst["synth_mask.png"], np.where(synth, 255, 0).astype(np.uint8))
    if synth.any():
        write_srgb_png(dst["base_color.png"], new)
        write_png(dst["height.png"], np.rint(height_from(new, new.shape[0]) * 255).astype(np.uint8))
