"""`generate`: ONE multi-reference edit of the clay sheet, seam detection, cell extraction, per-object QA.

    mvb2r.py generate CASE [--attempt K] [--generator fal|mock] [--max-calls N] [--force]

Writes gen/grid_edit/{prompt.txt, sheet_aK.png, result.json}, gen/cells/{cell_view_XX_native.png,
cell_view_XX_calib.png, cells.json}, gen/qa/cells.json (+ gen/qa/masks/). The sheet is requested at the
grid size planned from the verified pixel cap; the size actually returned is read from the file, and a
downsized answer is reported as such (the cells are then smaller; resizing them back is not detail).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cells as cells_mod
from . import common, falio, prompts, qa
from . import grid as grid_mod
from . import model as model_mod


def add_gen_args(p) -> None:
    p.add_argument("--generator", choices=("fal", "mock"), default=None, help="override case.json model.generator")
    p.add_argument("--max-calls", type=int, default=None, help="override case.json budget.max_calls")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--force", action="store_true", help="ignore the cache (a new paid call)")


def register(sub) -> None:
    p = sub.add_parser("generate", help="one multi-reference grid edit, seams, cells, per-object QA")
    p.add_argument("case")
    add_gen_args(p)
    p.set_defaults(func=run)


def _grid_inputs(case):
    producer = f"mvb2r.py grid {case}"
    layout = common.read_json(common.require(case / "grid" / "layout.json", producer))
    refs = common.read_json(common.require(case / "grid" / "references.json", producer))["references"]
    return layout, refs


def run(args) -> int:
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    cams = common.load_cameras(case)
    objects = common.case_objects(case)
    model_mod.announce("generate", model_mod.load_model(case, cfg, args.generator))
    layout, refs = _grid_inputs(case)
    sentence = grid_mod.layout_sentence(layout["rows"], layout["cols"], layout["n_views"], refs)
    prompt = prompts.grid_edit(cfg, objects, refs, layout, sentence)
    out_dir = case / "gen" / "grid_edit"
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    size = (int(layout["width"]), int(layout["height"]))
    sheet = out_dir / f"sheet_a{args.attempt}.png"
    job = {"tag": f"grid_edit_a{args.attempt}", "family": "grid", "prompt": prompt, "size": size,
           "refs": [case / r["path"] for r in refs], "out": sheet, "views": [c["name"] for c in cams],
           "attempt": args.attempt, "generator": args.generator, "max_calls": args.max_calls, "force": args.force}
    common.say(f"[generate] grid edit {size[0]}x{size[1]} with {len(refs)} references ...")
    rec = falio.edit(case, job)
    returned = tuple(rec["outputs"][0][k] for k in ("width", "height"))
    result = {"sheet": common.rel(case, sheet), "requested": list(size), "returned": list(returned),
              "honoured": returned == size, "request_id": rec.get("request_id"), "status": rec["status"],
              "attempt": args.attempt, "references": [r["path"] for r in refs],
              "note": None if returned == size else "the model returned a different size: cells are smaller than planned "
                                                    "and any enlargement of them is a resize, not generated detail"}
    common.write_json(out_dir / "result.json", result)
    doc = cells_mod.extract(case, sheet, layout, cams, case / "gen" / "cells")
    report = qa_cells(case, cams, objects, doc)
    common.say(f"[generate] sheet {returned[0]}x{returned[1]} ({'as requested' if result['honoured'] else 'NOT as requested'}), "
               f"{len(doc['cells'])} cells{'' if result['honoured'] else ' (WARNING: size not honoured)'}; seams: " + ", ".join(f"{s['offset_px']:+.1f}px" for s in
                                                               doc["seam_detection"]["col_seams"] + doc["seam_detection"]["row_seams"]))
    for v, r in report.items():
        iou = r["mean_iou"] if r["mean_iou"] is not None else r["iou_reason"]
        common.say(f"  {v}: mean object IoU {iou}; {'usable' if r['gate']['usable'] else 'REJECTED'}")
    qa.say_rejected("generate", {v: r["gate"] for v, r in report.items()})
    return 0


def qa_one(case, cam: dict, objects: list[dict], rgb: np.ndarray, tag: str) -> dict:
    """QA one calibration-size image of a view; masks written to gen/qa/masks/<view>_<tag>_<object>.png."""
    vp = common.view_passes(case, cam)
    metrics, seg = qa.view_qa(rgb, vp["oid"], objects, qa.load_landmarks(case), cam, vp["depth"])
    by_id = {int(o["id"]): o for o in objects}
    mdir = case / "gen" / "qa" / "masks"
    for i, r in seg.items():
        common.write_gray(mdir / f"{cam['name']}_{tag}_{common.file_token(by_id[i]['name'])}.png",
                          r["mask"].astype(np.uint8) * 255)
    s = qa.view_summary(metrics, qa.cell_rule(case), int(cam["width"]) * int(cam["height"]))
    return {**metrics, "mean_iou": s["mean_iou"], "min_iou": s["min_iou"], "iou_reason": s["iou_reason"]}


def qa_cells(case, cams: list[dict], objects: list[dict], doc: dict) -> dict:
    out, imgs = {}, {}
    for c in doc["cells"]:
        cam = next(x for x in cams if x["name"] == c["view"])
        imgs[c["view"]] = common.read_rgb(case / c["calib"])
        out[c["view"]] = qa_one(case, cam, objects, imgs[c["view"]], "cell")
    gates = qa.gate_views(case, cams, imgs, out)
    out = {v: {**r, "gate": {k: gates[v][k] for k in ("usable", "reasons", "match")}} for v, r in out.items()}
    common.write_json(case / "gen" / "qa" / "cells.json", out)
    return out


def probe_size(case, cfg: dict, info: dict) -> int:
    """`probe --check-size`: ONE call at the planned grid size; the returned size is read from the file."""
    layout, refs = _grid_inputs(case)
    size = (int(layout["width"]), int(layout["height"]))
    out = case / "gen" / "probe" / f"size_{size[0]}x{size[1]}.png"
    rec = falio.edit(case, {"tag": f"size_probe_{size[0]}x{size[1]}", "family": "probe", "size": size,
                            "prompt": "Reproduce Image 1 as a photograph, same layout.",
                            "refs": [case / refs[0]["path"]], "out": out,
                            "views": [layout["cells"][0]["view"]]})
    got = [rec["outputs"][0]["width"], rec["outputs"][0]["height"]]
    p = case / "gen" / "model.json"
    m = common.read_json(p)
    m["verified_sizes"] = [*m.get("verified_sizes", []), {"requested": list(size), "returned": got,
                                                         "request_id": rec.get("request_id")}]
    common.write_json(p, m)
    common.say(f"[probe] requested {size[0]}x{size[1]} -> returned {got[0]}x{got[1]} "
               f"({'honoured' if got == list(size) else 'NOT honoured'})")
    return 0 if got == list(size) else 3


def latest_sheet(case) -> Path:
    r = common.read_json(common.require(case / "gen" / "grid_edit" / "result.json", f"mvb2r.py generate {case}"))
    return case / r["sheet"]
