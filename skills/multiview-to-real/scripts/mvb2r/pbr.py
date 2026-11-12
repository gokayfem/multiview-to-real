"""Stage `pbr`: shading basis -> conservative de-light -> re-projection -> materials -> rig fit -> final.blend.

    mvb2r.py pbr CASE [--method M] [--scale 0.5] [--samples 32] [--fit-iters 3] [--synth-direct]

Writes pbr/shading/, pbr/delit/, pbr/albedo/<method>/<object>/, pbr/textures/<object>/{base_color,height}.png,
pbr/rig_fit.json, pbr/eval.json and out/final.blend (Principled, packed, source + novel cameras, README:
measured vs inferred per object, residual baked-shading correlation). Base colour = projected generated texture
x a smooth clamped correction; it still contains some baked shading (measured, reported).
Opt-in (`pbr.synth_direct` / --synth-direct, mvb2r/sunpatch_stage.py): the generated direct-light patch on
plain materials is replaced by texture synthesised from the shaded part of the same object (INFERRED);
pbr/textures_nosynth/ keeps the textures without it and pbr/eval.json "synth_direct" says whether it ran.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mvb2r.common import owned_path
from mvb2r.delight import delight_view
from mvb2r.fusion_io import case_json, resolve_case, run_blender, safe_name, write_json
from mvb2r.materials import load_priors, make_material
from mvb2r.pinhole import load_cameras, usable_cameras
from mvb2r.project import project_case, require_projection
from mvb2r.rigfit import TOL, next_scale, ratios, within
from mvb2r.sunpatch_stage import clear_previous, run_synth, synth_config


def register(subparsers) -> None:
    p = subparsers.add_parser("pbr", help="conservative de-light, Principled materials, rig fit, final.blend")
    p.add_argument("case")
    p.add_argument("--method", default=None, help="fusion method for the albedo projection (default: first)")
    p.add_argument("--scale", type=float, default=0.5, help="shading-basis render scale")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--fit-iters", type=int, default=3)
    p.add_argument("--synth-direct", action=argparse.BooleanOptionalAction, default=None,
                   help="replace the generated direct-light patch on plain materials by synthesised texture "
                        "(INFERRED; default: case.json pbr.synth_direct, false)")
    p.set_defaults(func=run)


def run(args) -> int:
    case = resolve_case(args.case)
    cfg = case_json(case)
    synth_conf = synth_config(cfg, args.synth_direct)  # validated before any render
    cov = require_projection(case)
    method = args.method or cov["methods"][0]
    log = case / "pbr" / "logs" / "pbr.log"
    cams = case / "views" / "cameras.json"
    preflight_outputs(case, [c.name for c in load_cameras(cams)])
    run_blender("shading_basis.py", ["--cameras", cams, "--out", case / "pbr" / "shading", "--scale", args.scale,
                                     "--samples", args.samples], case / "blockout.blend", log)
    views = [c.name for c in usable_cameras(case, "pbr")]
    dl = {v: delight_view(case, v, case / "pbr" / "shading", case / "pbr" / "delit") for v in views}
    alb = project_case(case, cfg, [method], case / "pbr" / "albedo", images=case / "pbr" / "delit")
    mats, readme = materials(case, alb, method)
    synth = synth_step(case, cfg, synth_conf, alb, method, views, readme)
    fit = fit_rig(case, mats, readme, args, views, alb, log)
    warnings = pbr_warnings(fit, dl)
    if warnings:  # say it in the blend too, then rebuild once with the same light scale
        with open(readme, "a") as fh:
            fh.write("WARNINGS:\n" + "".join(f"  - {w}\n" for w in warnings))
        build(case, mats, readme, fit["final"]["light_scale"], log,
              ",".join(n for n, r in alb["objects"].items() if r.get("opening")))
    write_json(case / "pbr" / "eval.json", {"status": "warning" if warnings else "ok", "warnings": warnings,
                                            "delight": dl, "rig_fit": fit, "method": method,
                                            "synth_direct": synth,
                                            "note": "corr = log-luminance correlation of image vs proxy shading on "
                                                    "object pixels; lower after de-light = less baked shading"})
    print(f"[pbr] final.blend; rig median/p90 ratio {fit['final']}; within ±{int(TOL * 100)}%: {fit['ok']}")
    for w in warnings:
        print(f"[pbr] WARNING: {w}")
    return 0


OUT_DIRS = ("pbr", "pbr/shading", "pbr/delit", "pbr/albedo", "pbr/textures", "pbr/rig_renders", "pbr/logs", "out")
OUT_FILES = ("pbr/eval.json", "pbr/materials.json", "pbr/rig_fit.json", "pbr/README_final.txt", "pbr/logs/pbr.log",
             "out/final.blend")
VIEW_FILES = ("pbr/shading/{v}_sun.exr", "pbr/shading/{v}_sky.exr", "pbr/shading/{v}_sundirect.exr",
              "pbr/delit/{v}.png", "pbr/rig_renders/{v}.png")


def preflight_outputs(case: Path, views: list) -> None:
    """Refuse (StageError) before any render when an output of this stage, or a folder above it, is a symbolic
    link planted in the case: writing or clearing through it would touch files outside the case."""
    for rel in OUT_DIRS + OUT_FILES + tuple(f.format(v=v) for v in views for f in VIEW_FILES):
        owned_path(case, rel)


def synth_step(case, cfg, conf, alb, method, views, readme) -> dict:
    """Opt-in direct-light patch synthesis on the material textures (before the rig fit builds the blend)."""
    if conf is None:
        clear_previous(case)
        return {"enabled": False, "ran": False}
    rep = run_synth(case, cfg, conf, alb, method, views, case / "pbr" / "shading")
    lines = synth_readme_lines(rep)
    if lines:
        with open(readme, "a") as fh:
            fh.write("".join(f"{s}\n" for s in lines))
    for n, o in rep["objects"].items():
        print(f"[pbr] synth_direct {n}: {o['synth_texels']} texels synthesised (INFERRED), "
              f"{o['components_filled']} filled, {o['components_skipped']} skipped, "
              f"{o['components_dropped_small']} specks dropped")
        if o["repetition_expected"]:
            print(f"[pbr] synth_direct {n}: source smaller than the patch (ratio {o['source_ratio_min']:.2f}): "
                  "the fill repeats it visibly")
    fallback = sorted(v for v, b in rep.get("direct_basis", {}).items() if b != "direct")
    if fallback:
        print(f"[pbr] WARNING: synth_direct: no direct-only shading in {fallback}; used the sun total")
    return rep


def synth_readme_lines(rep: dict) -> list[str]:
    if not rep.get("ran"):
        return []
    out = ["SYNTHESISED (INFERRED) where the generated direct-light patch was (pbr.synth_direct; "
           "pbr/textures_nosynth keeps the textures without it):"]
    for n, o in rep["objects"].items():
        out.append(f"  {n}: inferred {o['inferred_before']:.2f} -> {o['inferred_after']:.2f}; "
                   f"{o['components_filled']} patch(es) filled, {o['components_skipped']} left unchanged")
    return out


def pbr_warnings(fit: dict, delight: dict) -> list[str]:
    out = []
    f = fit["final"]
    if not fit["ok"]:
        out.append(f"rig check failed: render/target luminance median ratio {f['median_ratio']}, p90 ratio "
                   f"{f['p90_ratio']} (tolerance ±{int(TOL * 100)}%) after {len(fit['history'])} iterations")
    skipped = [v for v, d in delight.items() if not d.get("applied", True)]
    if skipped:
        out.append(f"de-light not applied in {skipped}: targets do not show the proxy shading (corr < 0.3)")
    return out


def materials(case: Path, alb: dict, method: str):
    pri = load_priors()
    mats, lines = {}, ["APPROXIMATE PBR from AI-generated views (not measured photographs).",
                       "Base colour = projected texture x smooth clamped de-light correction; SOME BAKED SHADING "
                       "REMAINS (see pbr/eval.json corr_delit). Priors only bound each object's mean albedo.",
                       "MEASURED vs INFERRED (area fractions; inferred = same-object plain fill):"]
    for n, r in alb["objects"].items():
        if not r.get("projected"):
            lines.append(f"  {n}: opening, not projected")
            continue
        out = owned_path(case, f"pbr/textures/{safe_name(n)}")
        for f in ("base_color.png", "height.png"):
            owned_path(case, out / f)
        m = make_material(case / "pbr" / "albedo" / method / safe_name(n), out, r["class"], pri)
        mats[n] = m
        c = r["coverage"]
        lines.append(f"  {n} ({r['class']}): measured {c['observed'] + c['soft']:.2f}, inferred {c['inferred']:.2f}")
    lines.append("Not a 360-degree reconstruction.")
    write_json(case / "pbr" / "materials.json", mats)
    (case / "pbr" / "README_final.txt").write_text("\n".join(lines) + "\n")
    return case / "pbr" / "materials.json", case / "pbr" / "README_final.txt"


def build(case, mats, readme, scale, log, openings):
    nov = case / "proj" / "novel" / "cameras.json"
    cams = [case / "views" / "cameras.json"] + ([nov] if nov.is_file() else [])
    run_blender("build_pbr.py", ["--materials", mats, "--out", case / "out" / "final.blend", "--readme", readme,
                                 "--light-scale", scale, "--openings", openings, "--cameras", *cams],
                case / "blockout.blend", log)


def fit_rig(case, mats, readme, args, views, alb, log) -> dict:
    good = [r["id"] for r in alb["objects"].values() if r.get("projected") and r["coverage"]["observed"] >= 0.3]
    openings = ",".join(n for n, r in alb["objects"].items() if r.get("opening"))
    scale, hist = 1.0, []
    for _ in range(max(1, args.fit_iters)):
        build(case, mats, readme, scale, log, openings)
        out = case / "pbr" / "rig_renders"
        run_blender("render.py", ["--cameras", case / "views" / "cameras.json", "--out", out, "--scale", 0.25,
                                  "--samples", 16, "--zero-exposure"], case / "out" / "final.blend", log)
        r = ratios(out, case, views, good)
        hist.append({"light_scale": scale, **r})
        if within(r):
            break
        scale = next_scale(scale, r)
    res = {"history": hist, "final": hist[-1], "ok": within(hist[-1]), "tolerance": TOL,
           "sun_direction_search": "not implemented", "check_render_scale": 0.25}
    write_json(case / "pbr" / "rig_fit.json", res)
    return res
