"""`resync`: one render-then-refine round, kept only if it is better.

    mvb2r.py resync CASE [--method NAME] [--generator ...] [--max-calls N] [--force]

Input (written by the fusion half's `project`): proj/<method>/rerender/view_XX.png, the fused texture
re-rendered from every source camera at calibration resolution (one shared texture = consistent placement
of details, with stitching artefacts).

  1. pilot: the two views with the largest mutual overlap (mapped valid pixels) are edited first with the
     consistency prompt (1 re-render = where details are, 2 current target native = photographic quality,
     3 clay = geometry, then the case's references).
  2. the pilot continues only if, on that pair, the high-band NCC improves by more than min_ncc_high_gain,
     the low-band dE does not grow by more than max_dE_low_increase, and alignment holds (every checked
     object's IoU drops by at most iou_drop_max) - acceptance.json "resync".
  3. the remaining views are edited; the full set is scored the same way. Only if the full set passes do the
     new targets replace the old ones (old ones kept in gen/round_K/previous_targets/). Otherwise the previous
     targets stay and gen/round_K/decision.json says why.
"""
from __future__ import annotations

import shutil

from . import common, consistency, falio, prompts
from . import generate as gen_mod
from . import model as model_mod
from . import refine as refine_mod
from . import targets as targets_mod


def register(sub) -> None:
    p = sub.add_parser("resync", help="render-then-refine round from the fusion re-renders; kept only if better")
    p.add_argument("case")
    p.add_argument("--method", default=None, help="proj/<method>/rerender to use (default: case fusion.method / the only one)")
    gen_mod.add_gen_args(p)
    p.set_defaults(func=run)


def find_method(case, cfg: dict, want: str | None) -> str:
    want = want or ((cfg.get("fusion") or {}).get("method") if (cfg.get("fusion") or {}).get("method") != "auto" else None)
    have = sorted(p.parent.name for p in (case / "proj").glob("*/rerender") if p.is_dir()) if (case / "proj").is_dir() else []
    if want:
        if want not in have:
            raise common.StageError(f"missing proj/{want}/rerender/ - run `mvb2r.py project {case} --method {want}` first")
        return common.check_name(want, "method")
    if len(have) != 1:
        raise common.StageError(f"need exactly one proj/<method>/rerender/ (found {have or 'none'}): run `mvb2r.py "
                                f"project {case}` first, or pass --method")
    return have[0]


def gate(rule: dict, before: dict, after: dict, qa_before: dict, qa_after: dict, checked: list[str]) -> dict:
    """Is `after` better than `before` and aligned? (pure)"""
    b, a = before["summary"].get("ALL"), after["summary"].get("ALL")
    if not b or not a:
        return {"pass": False, "reasons": ["no overlapping valid pixels to compare"]}
    reasons = []
    gain = a["ncc_high"] - b["ncc_high"]
    if gain <= rule["min_ncc_high_gain"]:
        reasons.append(f"high-band NCC did not improve ({b['ncc_high']:.4f} -> {a['ncc_high']:.4f})")
    if a["dE_low"] - b["dE_low"] > rule["max_dE_low_increase"]:
        reasons.append(f"low-band dE grew ({b['dE_low']:.2f} -> {a['dE_low']:.2f})")
    for view, objs in qa_after.items():
        for name, r in objs.items():
            prev = qa_before.get(view, {}).get(name)
            if prev and r["reliability"] in checked and r["iou"] < prev["iou"] - rule["iou_drop_max"]:
                reasons.append(f"{view}/{name}: IoU {prev['iou']:.3f} -> {r['iou']:.3f}")
    return {"pass": not reasons, "ncc_high_gain": round(gain, 4), "reasons": reasons}


def edit_view(case, cfg, cam, objects, current: dict, method: str, rnd: int, mdl: dict, args) -> tuple:
    users = [{**r, "kind": "reference"} for r in cfg.get("references", [])]
    fixed = [case / "proj" / method / "rerender" / f"{cam['name']}.png", case / current["native_path"],
             common.views_dir(case) / cam["name"] / "clay.png"]
    for p in fixed:
        common.require(p, f"mvb2r.py project {case}")
    kept = users[:max(0, int(mdl["max_images"]) - 3)]
    prompt = prompts.consistency(cfg, objects, prompts.renumber(kept, 4))
    size = model_mod.refine_size(mdl, int(cam["width"]), int(cam["height"]))
    out = case / "gen" / f"round_{rnd}" / "edits" / f"{cam['name']}.png"
    rec = falio.edit(case, {"tag": f"resync_r{rnd}_{cam['name']}", "family": "consistency", "prompt": prompt,
                            "size": size, "refs": fixed + [case / r["path"] for r in kept], "out": out,
                            "views": [cam["name"]], "attempt": rnd, "generator": args.generator,
                            "max_calls": args.max_calls, "force": args.force})
    return out, rec


def _score(geo, case, views: list[str], images: dict) -> dict:
    return consistency.compare(geo, {v: images[v] for v in views})


def run(args) -> int:
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    cams = common.load_cameras(case)
    objects = common.case_objects(case)
    mdl = model_mod.load_model(case, cfg, args.generator)
    model_mod.announce("resync", mdl)
    method = find_method(case, cfg, args.method)
    current = common.read_json(common.require(case / "targets" / "targets.json", f"mvb2r.py targets {case}"))["views"]
    rule = refine_mod.load_acceptance(case)["resync"]
    checked = refine_mod.load_acceptance(case)["refine"]["checked_reliabilities"]
    rnd = 1 + max([int(p.name.split("_")[1]) for p in (case / "gen").glob("round_*") if p.name.split("_")[1].isdigit()] or [0])
    rdir = case / "gen" / f"round_{rnd}"
    cams = consistency.usable_cams(case, cams)
    if len(cams) < 2:
        raise common.StageError("resync needs at least 2 usable views")
    geo = consistency.Geometry(case, cams, objects)
    pair = list(max(geo.overlap().items(), key=lambda kv: kv[1])[0])
    old_imgs = {c["name"]: common.read_rgb(case / current[c["name"]]["target_png"]) for c in cams}
    cand_dir, cand = rdir / "candidate", {}
    decision = {"round": rnd, "method": method, "pilot_pair": pair, "rule": rule}
    for stage, views in (("pilot", pair), ("rest", [c["name"] for c in cams if c["name"] not in pair])):
        for v in views:
            cam = next(c for c in cams if c["name"] == v)
            native, rec = edit_view(case, cfg, cam, objects, current[v], method, rnd, mdl, args)
            cand[v] = targets_mod.freeze_view(case, cam, objects, native, cand_dir,
                                              {"status": f"resync_round_{rnd}", "request_id": rec.get("request_id")})
        scope = pair if stage == "pilot" else [c["name"] for c in cams]
        new_imgs = {**old_imgs, **{v: common.read_rgb(case / cand[v]["target_png"]) for v in cand}}
        before, after = _score(geo, case, scope, old_imgs), _score(geo, case, scope, new_imgs)
        g = gate(rule, before, after, {v: current[v]["qa"]["objects"] for v in scope},
                 {v: cand[v]["qa"]["objects"] for v in scope if v in cand}, checked)
        decision[stage] = {"views": scope, "before": before["summary"].get("ALL"), "after": after["summary"].get("ALL"), **g}
        common.say(f"[resync] {stage} on {', '.join(scope)}: ncc_high {decision[stage]['before']} -> "
                   f"{decision[stage]['after']}; {'PASS' if g['pass'] else 'FAIL: ' + '; '.join(g['reasons'][:3])}")
        if not g["pass"]:
            decision["kept"] = "previous targets"
            common.write_json(rdir / "decision.json", decision)
            return 0
    _promote(case, rdir, cand, rnd)
    decision["kept"] = f"round {rnd} targets"
    common.write_json(rdir / "decision.json", decision)
    return 0


def _promote(case, rdir, cand: dict, rnd: int) -> None:
    """Copy the old targets aside, then swap in a COMPLETE new folder (no stale masks, `frozen` kept)."""
    prev = rdir / "previous_targets"
    if prev.exists():
        shutil.rmtree(prev)
    shutil.copytree(case / "targets", prev)
    tmp = case / "targets.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(rdir / "candidate", tmp)
    src = common.rel(case, rdir / "candidate")
    fix = lambda p: p.replace(src, "targets", 1)  # noqa: E731
    views = {v: {**t, "target_png": fix(t["target_png"]), "align_conf_png": fix(t["align_conf_png"]),
                 "masks": {k: {**m, "path": fix(m["path"])} for k, m in t["masks"].items()}} for v, t in cand.items()}
    import time
    common.write_json(tmp / "targets.json", {"frozen": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": f"resync round {rnd}",
                                             "generation": targets_mod.generation_of(case / "targets") + 1,
                                             "views": views, "previous": common.rel(case, prev),
                                             "note": "downstream outputs (proj/, pbr/, out/) older than this file are stale"})
    targets_mod.swap_in(case, tmp, case / "targets")
    common.say(f"[resync] round {rnd} targets promoted; proj/, pbr/ and out/ are now STALE: refresh with "
               f"`mvb2r.py run {case} --from project`")
