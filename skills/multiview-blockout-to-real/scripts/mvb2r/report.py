"""`report` (out/report.md from the files that exist) and `critique` (pack sheets for a FRESH reviewer).

    mvb2r.py report CASE
    mvb2r.py critique CASE                 pack out/sheets/critique_in/ and print the reviewer prompt
    mvb2r.py critique CASE --save FILE     validate the reviewer's JSON -> out/sheets/critique.json

The report states only what files show: sizes requested vs returned (from the downloaded files), the native
vs resized size of every target, the per-view acceptance table, consistency, coverage, calls and cost notes,
and the limits. Sections whose inputs are missing say which command produces them. The fusion half's
proj/*/score.json and pbr/eval.json are summarised when present.
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from . import common, falio

LIMITS = [
    "Generated views are not photo-consistent: the consistency table measures how far from it they are.",
    "A resize is never generated detail: enlarged cells and targets are labelled as resized.",
    "Coverage and inferred fill are per object; nothing outside the cameras' coverage is claimed.",
    "Out of scope: glass, mirrors and view-dependent materials; thin strands as geometry; animation; exact identity.",
    "The diagnostic .blend contains BAKED lighting (the generated photos' light is in its textures); only the PBR "
    ".blend is meant to be relit, and its base colour still carries the residual shading reported above.",
    "Nothing beyond the measured coverage is claimed: no 360 deg or whole-scene result outside what the cameras saw.",
]


def _j(case, rel):
    p = case / rel
    return common.read_json(p) if p.is_file() else None


def _missing(what: str, cmd: str) -> list[str]:
    return [f"_{what} not available - run `{cmd}`._", ""]


def section_views(case) -> list[str]:
    v = _j(case, "views/verify/report.json")
    if not v:
        return _missing("view verification", "mvb2r.py views CASE")
    s = v["summary"]
    return ["## Views and calibration", "",
            f"- reprojection max {s['max_reproj_err_px']:.2e} px, back-projection max {s['max_backproject_err_mm']:.3f} mm",
            f"- depth_z - camera-axis Z max {s['max_depth_z_minus_Zc_m']:.2e} m; ray length exceeds depth_z by up to "
            f"{s['max_ray_length_minus_depth_z_m']:.4f} m (depth is camera-axis Z, not ray length)",
            f"- Blender world_to_camera_view vs P max {s['max_blender_vs_P_px']:.2e} px; checks "
            f"{'passed' if v['ok'] else 'FAILED: ' + '; '.join(v['failures'])}", ""]


def section_coverage(case) -> list[str]:
    c = _j(case, "views/coverage.json")
    if not c:
        return _missing("coverage", "mvb2r.py cameras CASE")
    rows = ["## Coverage (fraction of each object's surface seen)", "", "| object | seen by >= 1 view | seen by >= 2 views |",
            "|---|---|---|"]
    rows += [f"| {n} | {o['seen_ge1']:.2f} | {o['seen_ge2']:.2f} |" for n, o in sorted(c["objects"].items())]
    return rows + [""]


def section_generation(case) -> list[str]:
    g = _j(case, "gen/grid_edit/result.json")
    if not g:
        return _missing("generation", "mvb2r.py generate CASE")
    rows = ["## Generation", "", f"- sheet requested {g['requested'][0]}x{g['requested'][1]}, returned "
            f"{g['returned'][0]}x{g['returned'][1]} ({'as requested' if g['honoured'] else 'NOT as requested'}; read from the file)"]
    d = _j(case, "gen/refined/decisions.json") or {}
    for v, dec in sorted(d.items()):
        for a in dec["attempts"]:
            failed = [k for k, ok in a["checks"].items() if not ok]
            rows.append(f"- {v} refine: requested {a['requested'][0]}x{a['requested'][1]}, returned "
                        f"{a['returned'][0]}x{a['returned'][1]}; {'accepted' if a['accepted'] else 'rejected'}"
                        + (f" ({'; '.join(failed[:3])})" if failed else ""))
    return rows + [""]


def section_targets(case) -> list[str]:
    t = _j(case, "targets/targets.json")
    if not t:
        return _missing("targets", "mvb2r.py targets CASE")
    rows = [f"## Targets ({t['source']})", ""]
    rej = {v: x for v, x in t["views"].items() if x.get("usable") is False}
    rows += [f"- WARNING {v} rejected (not used downstream): {'; '.join(x['gate']['reasons'])}" for v, x in sorted(rej.items())]
    rows += ["", "| view | status | native | calibration | resize | mean object IoU |", "|---|---|---|---|---|---|"]
    for v, x in sorted(t["views"].items()):
        ious = [o["iou"] for o in x["qa"]["objects"].values() if o["reliability"] in ("high", "moderate")]
        rows.append(f"| {v} | {x['status']} | {x['native_size'][0]}x{x['native_size'][1]} | "
                    f"{x['calibration_size'][0]}x{x['calibration_size'][1]} | {x['resize']} | "
                    f"{np.mean(ious):.3f} |" if ious else
                    f"| {v} | {x['status']} | {x['native_size'][0]}x{x['native_size'][1]} | "
                    f"{x['calibration_size'][0]}x{x['calibration_size'][1]} | {x['resize']} | - |")
    rel = {}
    for x in t["views"].values():
        for name, m in x["masks"].items():
            rel.setdefault(name, []).append(m["reliability"])
    rows += ["", "Generated-mask reliability per object: " +
             "; ".join(f"{n}: {', '.join(sorted(set(r)))}" for n, r in sorted(rel.items())), ""]
    return rows


def section_consistency(case) -> list[str]:
    c = _j(case, "gen/qa/consistency.json")
    if not c:
        return _missing("consistency", "mvb2r.py consistency CASE")
    rows = ["## Cross-view consistency", "", "| set | class | pixels | dE low band | dE full | NCC high band |", "|---|---|---|---|---|---|"]
    for name, s in c["sets"].items():
        for cls, r in sorted(s["summary"].items()):
            rows.append(f"| {name} | {cls} | {r['n']} | {r['dE_low']:.2f} | {r['dE_full']:.2f} | {r['ncc_high']:.3f} |")
    rounds = sorted((case / "gen").glob("round_*/decision.json"))
    for p in rounds:
        d = common.read_json(p)
        rows.append(f"\nResync round {d['round']}: kept {d.get('kept')}.")
    return rows + [""]


def section_calls(case) -> list[str]:
    s = falio.summary(case)
    if not s:
        return ["## Calls", "", "No model calls logged.", ""]
    rows = ["## Calls and cost", "", "| generator:endpoint | started | cache hits | failed | size not honoured | cost |",
            "|---|---|---|---|---|---|"]
    rows += [f"| {k} | {v['started']} | {v['cache_hits']} | {v['failed']} | {v['size_not_honoured']} | {v['cost_note']} |"
             for k, v in s.items()]
    return rows + [""]


def _methods(case) -> list[str]:
    proj = case / "proj"
    return sorted(p.name for p in proj.iterdir() if p.is_dir() and p.name != "passes"
                  and ((p / "rerender").is_dir() or any(p.glob("*/texture.png")))) if proj.is_dir() else []


def section_fusion(case) -> list[str]:
    """Every projected method, its re-renders and sheets; score, refit and PBR files with their warnings."""
    rows, methods = [], _methods(case)
    sheets = case / "out" / "sheets"
    for m in methods:
        n_sheet = len(list(sheets.glob(f"*_{m}.jpg"))) if sheets.is_dir() else 0
        n_rr = len(list((case / "proj" / m / "rerender").glob("view_*.png")))
        rows.append(f"- method `{m}`: {n_rr} re-renders, {n_sheet} contact sheets"
                    + ("" if n_sheet else " (no sheets written for this method)"))
    score = _j(case, "proj/score.json")
    if score is not None:
        keys = sorted(score) if isinstance(score, dict) else []
        rows.append(f"- score: proj/score.json ({', '.join(keys[:8])})")
    refit = _j(case, "refit/refit.json")
    if refit is not None:
        rows.append("- refit: refit/refit.json" + (" and blockout_refit.blend" if (case / "blockout_refit.blend").is_file() else ""))
    return (["## Fusion, refit and PBR", "", *rows, ""] if rows else [])


def section_texture_coverage(case) -> list[str]:
    cov = _j(case, "proj/coverage.json")
    if not isinstance(cov, dict) or not isinstance(cov.get("objects"), dict):
        return []
    rows = ["## Texture coverage per object (proj/coverage.json)", "",
            "| object | observed | soft | inferred (filled, not seen) |", "|---|---|---|---|"]
    for name, r in sorted(cov["objects"].items()):
        c = r.get("coverage") if isinstance(r, dict) else None
        if isinstance(c, dict):
            rows.append(f"| {name} | {c.get('observed', 0):.2f} | {c.get('soft', 0):.2f} | {c.get('inferred', 0):.2f} |")
        else:
            rows.append(f"| {name} | - | - | - ({common.clean_text((r or {}).get('note', 'not projected'), 80)}) |")
    tour = _j(case, "out/tour.json")
    if isinstance(tour, dict) and "measured_fraction_mean" in tour:
        rows += ["", f"360 deg tour: on average {tour['measured_fraction_mean']:.0%} of the visible surface is measured "
                     f"and {tour.get('inferred_fraction_mean', 1 - tour['measured_fraction_mean']):.0%} inferred."]
    return rows + [""]


def _delight_views(ev: dict) -> dict:
    dl = ev.get("delight")
    if isinstance(dl, dict) and isinstance(dl.get("views"), dict):
        dl = dl["views"]
    return {v: r for v, r in (dl or {}).items() if isinstance(r, dict) and "corr_raw" in r} if isinstance(dl, dict) else {}


def section_pbr(case) -> list[str]:
    ev = _j(case, "pbr/eval.json")
    if not isinstance(ev, dict):
        return []
    rows = ["## PBR: residual baked shading (pbr/eval.json)", "",
            "Base colour = projected photo x smooth correction. Correlation of image log-luminance with the proxy "
            "shading: raw target vs after de-light (lower = less baked shading; what remains is baked in).", "",
            "| view | de-light applied | corr raw | corr after |", "|---|---|---|---|"]
    for v, r in sorted(_delight_views(ev).items()):
        rows.append(f"| {v} | {'yes' if r.get('applied') else 'skipped (no measurable baked shading)'} | "
                    f"{r['corr_raw']:.3f} | {r.get('corr_delit', float('nan')):.3f} |")
    rows += [f"- WARNING {common.clean_text(w, 200)}" for w in ev.get("warnings", [])]
    rel = _j(case, "out/relight.json")
    if isinstance(rel, dict):
        rows += ["", "Relight (out/relight.json): " + "; ".join(f"{k}: {common.clean_text(v, 80)}" for k, v in
                                                            list(rel.items())[:8] if not isinstance(v, (dict, list)))]
    return rows + [""]


def section_stale(case) -> list[str]:
    t = case / "targets" / "targets.json"
    if not t.is_file():
        return []
    ref = t.stat().st_mtime
    stale = [p for p in ("proj/coverage.json", "proj/score.json", "out/diagnostic.blend", "pbr/eval.json",
                         "out/final.blend", "out/tour.json", "out/relight.json")
             if (case / p).is_file() and (case / p).stat().st_mtime < ref]
    if not stale:
        return []
    return ["## STALE outputs", "", f"These were made from older targets (targets.json is newer): {', '.join(stale)}. "
            "Refresh with `mvb2r.py run CASE --from project`.", ""]


def section_run(case) -> list[str]:
    r = _j(case, "logs/run_summary.json")
    if not r:
        return []
    rows = ["## Last `run`", "", "| stage | status | flags |", "|---|---|---|"]
    for s in r["stages"]:
        rows.append(f"| {s['stage']} | {s['status']} | {'<br>'.join(common.clean_text(f, 160) for f in s['flags'][:5]) or '-'} |")
    return rows + [""]


def build_report(case) -> str:
    cfg = common.load_case(case)
    lines = [f"# {cfg.get('name', case.name)} - Multiview Blockout 2 Real report", "",
             f"Input `{cfg['input']['path']}` ({cfg['input']['kind']}), {len(cfg.get('objects', []))} objects, "
             f"{len(cfg['views'].get('cameras', []))} views at {cfg['views']['resolution']} px, generator "
             f"`{cfg['model']['generator']}` ({cfg['model'].get('edit_endpoint')}).", ""]
    for sec in (section_views, section_coverage, section_generation, section_targets, section_consistency,
                section_calls, section_stale, section_fusion, section_texture_coverage, section_pbr, section_run):
        lines += sec(case)
    lines += ["## Limits", ""] + [f"- {x}" for x in LIMITS]
    return scrub_paths("\n".join(lines) + "\n", case)


def scrub_paths(text: str, case) -> str:
    """No absolute local path in a shareable file: the case folder -> CASE, the home folder -> ~ (pure)."""
    from pathlib import Path
    for p, rep in ((str(common.case_dir(case)), "CASE"), (str(Path(case)), "CASE"), (str(Path.home()), "~")):
        text = text.replace(p, rep)
    return text


# ---------------------------------------------------------------- critique
CRITIQUE_PROMPT = """You are shown {n} pairs of images of the same scene, one pair per camera:
  clay_<view>.png (a grey 3D render: the geometry authority) and photo_<view>.png (a generated photograph).
Files render_<method>_<view>.png, when present, are 3D re-renders of the textured scene by each fusion method:
judge them against photo_<view>.png under "methods" (seams, smears, misplaced detail).
Look only at these files. For each view answer:
  "geometry_ok": true/false (same objects, positions, silhouettes and framing as the clay),
  "problems": [{{"what": "...", "bbox": [x0, y0, x1, y1], "severity": 1-5}}],
and overall: "cross_view": do materials, colours and light agree across views? List disagreements.
Answer with JSON only: {{"views": {{"view_01": {{"geometry_ok": true, "problems": []}}}}, "cross_view": "..."}}"""


def pack_critique(case) -> int:
    t = _j(case, "targets/targets.json")
    if not t:
        raise common.StageError(f"missing targets/targets.json - run `mvb2r.py targets {case}` first")
    out = case / "out" / "sheets" / "critique_in"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for v, x in sorted(t["views"].items()):
        if x.get("usable") is False:
            continue
        shutil.copyfile(common.views_dir(case) / v / "clay.png", out / f"clay_{v}.png")
        shutil.copyfile(case / x["target_png"], out / f"photo_{v}.png")
        for m in _methods(case):
            rr = case / "proj" / m / "rerender" / f"{v}.png"
            if rr.is_file():
                shutil.copyfile(rr, out / f"render_{common.file_token(m)}_{v}.png")
    prompt = CRITIQUE_PROMPT.format(n=len(t["views"]))
    (out / "PROMPT.txt").write_text(scrub_paths(prompt, case) + "\n")
    common.say(f"[critique] packed {len(t['views'])} pairs in {common.rel(case, out)}. Give ONLY these files and this prompt "
               f"to a NEW reviewer that has seen nothing of the case, then save its answer with --save FILE:\n\n{prompt}")
    return 0


def save_critique(case, path) -> int:
    try:
        data = json.loads(open(path).read())
    except (OSError, json.JSONDecodeError) as e:
        raise common.StageError(f"critique answer {path} is not JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("views"), dict):
        raise common.StageError("critique answer must be an object with a 'views' object")
    for v, r in data["views"].items():
        if not isinstance(r, dict) or not isinstance(r.get("geometry_ok"), bool) or not isinstance(r.get("problems", []), list):
            raise common.StageError(f"critique answer for {common.clean_text(v, 20)!r} is malformed")
    common.write_json(case / "out" / "sheets" / "critique.json", data)
    common.say("[critique] saved out/sheets/critique.json")
    return 0


def register(sub) -> None:
    p = sub.add_parser("report", help="out/report.md from the files that exist")
    p.add_argument("case")
    p.set_defaults(func=run_report)
    c = sub.add_parser("critique", help="pack clay/photo pairs for a fresh reviewer; --save validates its answer")
    c.add_argument("case")
    c.add_argument("--save", default=None)
    c.add_argument("--judge", action="store_true", help="(reserved) blind judge of final renders")
    c.set_defaults(func=run_critique)


def run_report(args) -> int:
    case = common.case_dir(args.case)
    text = build_report(case)
    out = case / "out" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    common.say(f"[report] wrote {common.rel(case, out)} ({len(text.splitlines())} lines)")
    return 0


def run_critique(args) -> int:
    case = common.case_dir(args.case)
    return save_critique(case, args.save) if args.save else pack_critique(case)
