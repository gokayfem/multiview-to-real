"""`refine`: per-view high-resolution edits with a fixed acceptance rule; the cell stays when a refine fails it.

    mvb2r.py refine CASE [--views view_01,view_03] [--attempt K] [--generator ...] [--max-calls N]

References, in order: 1 clay view, 2 native generated cell, 3 the generated sheet, then the case's reference
images (dropped from the end when the model's limit is reached; said in the log). Size: the largest size with
the view's aspect within the verified pixel cap (model.refine_size).

Acceptance (CASE/acceptance.json "refine", fixed before results are seen; pixel values at 1664 px, scaled):
for every object covering >= min_object_fraction of the view whose generated mask reliability is in
checked_reliabilities:   IoU(refined) >= IoU(cell) - iou_vs_cell_margin,  chamfer mean <= max,
dE76(refined vs cell) <= dE_vs_cell_max;  every visible landmark |offset| <= max;  and, when
visual_review_required, gen/qa/visual.json says true for the view. Otherwise the cell is the target.
"""
from __future__ import annotations


from . import common, falio, prompts, qa
from . import generate as gen_mod
from . import model as model_mod


def register(sub) -> None:
    p = sub.add_parser("refine", help="per-view high-resolution edits with a fixed acceptance rule")
    p.add_argument("case")
    p.add_argument("--views", default=None, help="comma-separated view names (default: all)")
    gen_mod.add_gen_args(p)
    p.set_defaults(func=run)


def load_acceptance(case) -> dict:
    p = common.case_dir(case) / "acceptance.json"
    src = p if p.is_file() else common.TEMPLATES_DIR / "acceptance.json"
    return common.read_json(src)


def evaluate(rule: dict, cell: dict, ref: dict, de: dict, width: int, n_px: int, visual) -> dict:
    """Checks of one refined view against its cell (pure)."""
    checks = {}
    cmax = qa.scaled(rule["chamfer_mean_max_px_at_1664"], width)
    for name, r in ref["objects"].items():
        c = cell["objects"].get(name)
        if c is None or r["clay_px"] < rule["min_object_fraction"] * n_px or r["reliability"] not in rule["checked_reliabilities"]:
            continue
        checks[f"{name}: iou >= cell - margin"] = r["iou"] >= c["iou"] - rule["iou_vs_cell_margin"]
        if r["chamfer_px"]["mean"] is not None:
            checks[f"{name}: chamfer mean <= {cmax:.1f}px"] = r["chamfer_px"]["mean"] <= cmax
        if name in de:
            checks[f"{name}: dE vs cell <= {rule['dE_vs_cell_max']}"] = de[name] <= rule["dE_vs_cell_max"]
    lmax = qa.scaled(rule["landmark_offset_max_px_at_1664"], width)
    for name, lm in ref.get("landmarks", {}).items():
        checks[f"landmark {name}: |offset| <= {lmax:.1f}px"] = abs(lm["offset_px"]) <= lmax
    if rule.get("visual_review_required"):
        checks["visual review"] = visual is True
    return checks


def _refs(case, cam: dict, cells: dict, sheet, user_refs: list[dict], limit: int):
    fixed = [common.views_dir(case) / cam["name"] / "clay.png", case / cells[cam["name"]]["native"], sheet]
    room = max(0, limit - len(fixed))
    kept, dropped = user_refs[:room], user_refs[room:]
    return fixed + [case / r["path"] for r in kept], kept, dropped


def refine_view(case, cfg, cam, objects, cells, cell_qa, mdl, args, rule) -> dict:
    sheet = gen_mod.latest_sheet(case)
    users = [{**r, "kind": "reference"} for r in cfg.get("references", [])]
    refs, kept, dropped = _refs(case, cam, cells, sheet, users, int(mdl["max_images"]))
    prompt = prompts.refine(cfg, objects, prompts.renumber(kept, 4))
    W, H = int(cam["width"]), int(cam["height"])
    size = model_mod.refine_size(mdl, W, H)
    out = case / "gen" / "refined" / f"{cam['name']}_a{args.attempt}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{cam['name']}_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    rec = falio.edit(case, {"tag": f"refine_{cam['name']}_a{args.attempt}", "family": "refine", "prompt": prompt,
                            "size": size, "refs": refs, "out": out, "views": [cam["name"]], "attempt": args.attempt,
                            "generator": args.generator, "max_calls": args.max_calls, "force": args.force})
    native = common.read_rgb(out)
    calib, method = common.resize_rgb(native, (W, H))
    ref_qa = gen_mod.qa_one(case, cam, objects, calib, f"refined_a{args.attempt}")
    cell_rgb = common.read_rgb(case / cells[cam["name"]]["calib"])
    de = qa.colour_delta(calib, cell_rgb, common.view_passes(case, cam)["oid"], objects)
    visual = (common.read_json(case / "gen" / "qa" / "visual.json").get(cam["name"])
              if (case / "gen" / "qa" / "visual.json").is_file() else None)
    checks = evaluate(rule, cell_qa, ref_qa, de, W, W * H, visual)
    return {"path": common.rel(case, out), "requested": list(size), "returned": rec["returned_size"],
            "honoured": rec.get("size_honoured"), "request_id": rec.get("request_id"), "calib_resize": method,
            "qa": ref_qa, "dE_vs_cell": de, "checks": checks, "accepted": bool(checks) and all(checks.values()),
            "visual_review": visual if visual is not None else "not done",
            "dropped_references": [r["path"] for r in dropped]}


def run(args) -> int:
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    cams = common.load_cameras(case)
    objects = common.case_objects(case)
    mdl = model_mod.load_model(case, cfg, args.generator)
    model_mod.announce("refine", mdl)
    cells = {c["view"]: c for c in common.read_json(common.require(case / "gen" / "cells" / "cells.json",
                                                                   f"mvb2r.py generate {case}"))["cells"]}
    cell_qa = common.read_json(common.require(case / "gen" / "qa" / "cells.json", f"mvb2r.py generate {case}"))
    want = [v.strip() for v in args.views.split(",")] if args.views else [c["name"] for c in cams]
    unknown = set(want) - {c["name"] for c in cams}
    if unknown:
        raise common.StageError(f"unknown views {sorted(unknown)}")
    rule = load_acceptance(case)["refine"]
    dpath = case / "gen" / "refined" / "decisions.json"
    decisions = common.read_json(dpath) if dpath.is_file() else {}
    for cam in [c for c in cams if c["name"] in want]:
        gate = cell_qa[cam["name"]].get("gate", {"usable": True})
        if not gate["usable"]:
            common.say(f"[refine] WARNING {cam['name']} skipped: its cell was rejected ({'; '.join(gate['reasons'])}); "
                       "a refine would inherit the wrong content -> retry `generate --attempt 2`")
            continue
        try:
            att = refine_view(case, cfg, cam, objects, cells, cell_qa[cam["name"]], mdl, args, rule)
        except falio.BudgetError as e:
            common.say(f"[refine] WARNING budget exhausted at {cam['name']}: {e} - the remaining views keep their cells")
            break
        prev = decisions.get(cam["name"], {"attempts": []})
        attempts = [a for a in prev["attempts"] if a["path"] != att["path"]] + [att]
        accepted = next((a["path"] for a in reversed(attempts) if a["accepted"]), None)
        decisions = {**decisions, cam["name"]: {"attempts": attempts, "accepted": accepted,
                                                "status": "accepted_refined" if accepted else "fallback_cell"}}
        common.write_json(dpath, decisions)
        failed = [k for k, ok in att["checks"].items() if not ok]
        common.say(f"[refine] {cam['name']}: {att['returned'][0]}x{att['returned'][1]} "
                   f"{'ACCEPTED' if att['accepted'] else 'rejected'}" + (f" ({'; '.join(failed[:4])})" if failed else ""))
    return 0
