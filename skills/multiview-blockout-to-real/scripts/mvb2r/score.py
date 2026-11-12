"""Stage `score`: leave-one-view-out, seam visibility, sharpness retention, coverage, alignment; contact sheets.

    mvb2r.py score CASE [--methods a,b] [--no-loo]

Leave-one-view-out (LOO): fuse without view k, compare with view k's own samples on texels view k sees under the
LEGACY gates (whatever fusion.gate_policy is, so two policies are judged on the same pixels) AND the fusion
without k observed; reported as full, low-band (UV blur) and high-band (residual) mean |error| x255 in sRGB.
Seam visibility (ratio of the step across source-label boundaries to the step inside labels; 1 = invisible) in
texture space and in novel views (exact UV lookup). Sharpness retention = fine-detail energy of the re-render /
of the target on interior object pixels (< 1 = blurrier). Leak (mvb2r/leak.py): fraction of observed texels whose
source pixel the generated image assigns to another object. These are separate numbers: none of them alone proves
3D consistency — look at the contact sheets (out/sheets/: clay | target | re-render).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion import fuse, resolve_method
from mvb2r.common import clean_text
from mvb2r.fusion_io import (StageError, case_json, linear_to_srgb, read_gray01, read_ids, read_rgb_linear, resolve_case,
                             safe_name, view_num, write_json)
from mvb2r.levels import uv_blur
from mvb2r.pinhole import load_cameras, usable_cameras
from mvb2r.project import require_projection
from mvb2r.leak import leak_report
from mvb2r.projinputs import evaluation_stack, fusion_options, load_case_inputs, object_input, object_params
from mvb2r.seams import seam_metrics
from mvb2r.texlookup import load_passes, render as lookup_render
from mvb2r.visibility import VisParams

NOTE = ("No single number proves 3D consistency: LOO measures agreement with a held-out generated view, seams "
        "measure visible cuts, sharpness measures lost detail. Look at out/sheets/.")


def register(subparsers) -> None:
    p = subparsers.add_parser("score", help="leave-one-view-out, seams, sharpness, coverage; contact sheets")
    p.add_argument("case")
    p.add_argument("--methods", default=None, help="subset of the projected methods (default: all)")
    p.add_argument("--no-loo", action="store_true")
    p.set_defaults(func=run)


def run(args) -> int:
    case = resolve_case(args.case)
    cfg = case_json(case)
    cov = require_projection(case)
    methods = args.methods.split(",") if args.methods else list(cov["methods"])
    out = {"note": NOTE, "methods": {}, "views_used": cov.get("views"), "skipped_views": cov.get("skipped_views", [])}
    ci = load_case_inputs(case, cfg)
    check_projection(cov, fusion_options(cfg)[2], [c.name for c in ci.cams])
    loo = {} if args.no_loo else leave_one_out(case, cfg, cov, methods, ci)
    out["leak"] = leak_report(case, ci, cov, methods)
    for m in methods:
        out["methods"][m] = {"loo": loo.get(m), "seams_novel": novel_seams(case, cov, m),
                             "seams_texture": texture_seams(cov, m), "sharpness": sharpness(case, cov, m)}
    out["coverage"] = {n: r.get("coverage") for n, r in cov["objects"].items()}
    out["alignment"] = alignment(case)
    write_json(case / "proj" / "score.json", out)
    contact_sheets(case, methods)
    print_table(out)
    return 0


PREVIOUS_RELEASE = {"gate_policy": "legacy", "fill": "blocks"}   # coverage.json written before they were recorded


def check_projection(cov: dict, opt: dict, view_names: list) -> None:
    """Refuse to score a projection made with another gate policy, fill or set of usable views than case.json /
    targets.json now give: LOO would re-fuse differently from the delivered textures and source_raw labels
    would map to the wrong cameras."""
    diffs = []
    for k in ("gate_policy", "fill"):
        then, now = cov.get(k, PREVIOUS_RELEASE[k]), opt.get(k, PREVIOUS_RELEASE[k])
        if then != now:
            diffs.append(f"{k} {clean_text(then, 20)} at projection, {clean_text(now, 20)} now")
    if list(cov.get("views") or []) != list(view_names):
        diffs.append(f"views {clean_text(cov.get('views'), 120)} at projection, {clean_text(view_names, 120)} now")
    if diffs:
        raise StageError("proj/ was made with other settings (" + "; ".join(diffs) + "); re-run `mvb2r.py project` "
                         "before `score` (a projection from before 0.2.0 recorded neither and counts as "
                         "gate_policy legacy, fill blocks: set those in case.json to score it as it is)")


# ----------------------------------------------------------------------------------------------- LOO
def _band_errors(tex, a_srgb, b_srgb, m) -> dict:
    if m.sum() < 20:
        return None
    w = m.astype(np.float32)
    la, lb = uv_blur(tex, a_srgb, 4.0, w), uv_blur(tex, b_srgb, 4.0, w)
    e = lambda x, y: float(np.mean(np.abs(x[m] - y[m])) * 255)  # noqa: E731
    return {"full": e(a_srgb, b_srgb), "low": e(la, lb), "high": e(a_srgb - la, b_srgb - lb), "texels": int(m.sum())}


def loo_mask(eval_valid_k: np.ndarray, res) -> np.ndarray:
    """Texels scored for held-out view k: valid in k under the LEGACY gates (projinputs.evaluation_stack, the
    same for every gate policy) and observed (not soft) by the fusion without k."""
    return eval_valid_k & res.observed & ~res.extra.get("soft", np.zeros(len(eval_valid_k), bool))


def _loo_object(oi, ev, concrete: str, prm, acc: dict) -> None:
    for k in range(oi.st.V):
        res = fuse(oi.held_out(k), concrete, prm)
        r = _band_errors(oi.tex, linear_to_srgb(res.color), linear_to_srgb(ev.color[k]), loo_mask(ev.valid[k], res))
        if r:
            acc.setdefault(oi.st.view_names[k], []).append(r)


def leave_one_out(case: Path, cfg: dict, cov: dict, methods: list, ci=None) -> dict:
    prm, by_class, opt = fusion_options(cfg)
    ci = ci or load_case_inputs(case, cfg)
    acc: dict = {m: {} for m in methods}
    for obj in ci.objects:
        rec = cov["objects"].get(obj["name"], {})
        if obj["opening"] or not rec.get("projected"):
            continue
        oi = object_input(ci, obj, rec["tex_size"], VisParams(), opt)
        _, ev = evaluation_stack(ci, obj, rec["tex_size"], VisParams(), opt)
        for m in methods:
            concrete = resolve_method(m, obj["class"], obj["detail_critical"], by_class)
            _loo_object(oi, ev, concrete, object_params(prm, obj, opt), acc[m])
    return {m: _aggregate(v) for m, v in acc.items()}


def _aggregate(per_view: dict) -> dict:
    out = {}
    for v, rows in per_view.items():
        n = sum(r["texels"] for r in rows)
        out[v] = {k: sum(r[k] * r["texels"] for r in rows) / max(n, 1) for k in ("full", "low", "high")}
    if out:
        out["mean"] = {k: float(np.mean([out[v][k] for v in per_view])) for k in ("full", "low", "high")}
    return out


# ----------------------------------------------------------------------------------------------- seams
def _label_maps(case: Path, cov: dict, method: str):
    tex, lab, valid = {}, {}, {}
    for n, r in cov["objects"].items():
        if r.get("projected"):
            d = case / "proj" / method / safe_name(n)
            tex[r["id"]] = read_rgb_linear(d / "texture.png")
            lab[r["id"]] = read_ids(d / "source_raw.png")[..., None].astype(np.float32)
            valid[r["id"]] = lab[r["id"]][..., 0] != 255
    return tex, lab, valid


def novel_seams(case: Path, cov: dict, method: str) -> dict | None:
    js = case / "proj" / "novel" / "cameras.json"
    if not js.is_file():
        return None
    tex, lab, valid = _label_maps(case, cov, method)
    ratios = []
    for c in load_cameras(js):
        ps = load_passes(case / "proj" / "passes" / c.name)
        img, drawn = lookup_render(ps, tex, valid=valid)
        L, _ = lookup_render(ps, lab, (255.0,), nearest=True)
        L = L[..., 0].astype(np.int32)
        ok = drawn & (L > 0) & (L < 255)
        ok = cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        r = seam_metrics(linear_to_srgb(img), L + 1000 * ps.ids, ok, ps.ids)
        ratios.append(r)
    vals = [r["coarse_ratio"] for r in ratios if r["coarse_ratio"] is not None]
    fine = [r["fine_ratio"] for r in ratios if r["fine_ratio"] is not None]
    return {"coarse_ratio_mean": float(np.mean(vals)) if vals else None,
            "fine_ratio_mean": float(np.mean(fine)) if fine else None, "cameras": len(ratios)}


def texture_seams(cov: dict, method: str) -> dict:
    vals = [r["methods"][method]["texture_seams"].get("coarse_ratio") for r in cov["objects"].values()
            if r.get("projected") and method in r.get("methods", {})]
    vals = [v for v in vals if v is not None]
    return {"coarse_ratio_mean": float(np.mean(vals)) if vals else None}


# ----------------------------------------------------------------------------------------------- sharpness
def _fine_energy(gray, m):
    f = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    return float(np.sqrt(np.mean(f[m] ** 2)))


def sharpness(case: Path, cov: dict, method: str) -> dict:
    vals = {}
    for c in usable_cameras(case, "score"):
        rr = case / "proj" / method / "rerender" / f"{c.name}.png"
        tg = case / "targets" / f"{c.name}.png"
        if not rr.is_file() or not tg.is_file():
            continue
        R = linear_to_srgb(read_rgb_linear(rr)) @ np.array([.2126, .7152, .0722], np.float32)
        T = linear_to_srgb(read_rgb_linear(tg))
        T = cv2.resize(T, R.shape[::-1], interpolation=cv2.INTER_AREA) @ np.array([.2126, .7152, .0722], np.float32)
        ids = load_passes(case / "proj" / "passes" / c.name).ids
        m = cv2.erode((ids > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if m.sum() > 200:
            vals[c.name] = _fine_energy(R, m) / max(_fine_energy(T, m), 1e-9)
    return {"per_view": vals, "mean": float(np.mean(list(vals.values()))) if vals else None}


def alignment(case: Path) -> dict:
    out = {}
    for c in usable_cameras(case, "score"):
        p = case / "targets" / f"align_conf_{view_num(c.name)}.png"
        if p.is_file():
            out[c.name] = float(read_gray01(p).mean())
    return out or {"note": "no targets/align_conf_XX.png"}


# ----------------------------------------------------------------------------------------------- sheets
def contact_sheets(case: Path, methods: list, size: int = 384) -> list:
    """Per view and method: clay | target | re-render; per view: target + every method side by side."""
    from mvb2r.sheets import triptych, write_grid
    out = []
    for c in usable_cameras(case, "score"):
        for m in methods:
            out.append(triptych(case, c.name, m, case / "out" / "sheets" / f"{c.name}_{m}.jpg", size))
        paths = [case / "targets" / f"{c.name}.png"] + [case / "proj" / m / "rerender" / f"{c.name}.png"
                                                       for m in methods]
        ids = case / "proj" / "passes" / c.name / "object_id.png"
        out.append(write_grid(paths, ["target (generated)"] + [f"re-render {m}" for m in methods],
                              case / "out" / "sheets" / f"{c.name}_methods.jpg", cols=len(paths), size=size,
                              ids_paths=[None] + [ids] * len(methods)))
    return out


def print_table(out: dict) -> None:
    print(f"{'method':10s} {'LOO full':>9s} {'low':>6s} {'high':>6s} {'seam novel':>10s} {'seam tex':>8s} "
          f"{'sharp':>6s} {'leak %':>7s}")
    leak = (out.get("leak") or {}).get("per_method", {})
    for m, r in out["methods"].items():
        lo = (r["loo"] or {}).get("mean", {})
        f = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
        lk = leak.get(m, {}).get("leaked_fraction")
        print(f"{m:10s} {f(lo.get('full')):>9s} {f(lo.get('low')):>6s} {f(lo.get('high')):>6s} "
              f"{f((r['seams_novel'] or {}).get('coarse_ratio_mean')):>10s} "
              f"{f(r['seams_texture'].get('coarse_ratio_mean')):>8s} {f(r['sharpness'].get('mean')):>6s} "
              f"{f(None if lk is None else 100 * lk):>7s}")
    print("leak % = observed texels whose source pixel the generated image assigns to another object "
          "(score.json leak.definition; blind to content displaced beyond the margins)")
    print(NOTE)
