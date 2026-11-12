"""`grid`: exact rows x cols sheet of the clay views, layout.json, ordered references manifest.

    mvb2r.py grid CASE

Rules
  - rows x cols: the fewest panels that hold N views without a very elongated sheet (grid_shape); empty
    panels, when unavoidable, are flat grey and the prompt says so.
  - cell size: the largest integer cell with the views' aspect whose whole sheet fits the model's verified
    pixel cap (gen/model.json from `probe`), sides multiples of 8 (cell_size). Cell rectangles are exact
    integers, no gutters, row-major.
  - grid/clay_grid_master.png: cells at calibration resolution, checked bit for bit against each clay view.
    grid/clay_grid_conditioning.png: the same sheet at the generation size (what is sent; "resized" is recorded).
  - references (in this order): the conditioning sheet, the case's reference images, one clay copy per view.
    When the endpoint's reference limit is exceeded, per-view clay copies are dropped from the LAST view
    backwards (the first views keep theirs) and grid/references.json lists what was dropped. All references
    must be distinct files (sha256).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import common

MULTIPLE = 8
MAX_SHEET_ASPECT = 3.2
BLANK_GREY = 128


# ---------------------------------------------------------------- pure layout maths
def grid_shape(n: int, view_aspect: float = 1.0) -> tuple[int, int]:
    """(rows, cols) for n views: fewest empty panels, then the sheet aspect closest to 3:2 (sheet aspect <= 3.2)."""
    if not common.MIN_VIEWS <= n <= common.MAX_VIEWS:
        raise ValueError(f"number of views must be {common.MIN_VIEWS}..{common.MAX_VIEWS}, got {n}")
    best = None
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        aspect = cols * view_aspect / rows
        if not 1 / MAX_SHEET_ASPECT <= aspect <= MAX_SHEET_ASPECT:
            continue
        key = (rows * cols - n, abs(math.log(aspect / 1.5)))
        if best is None or key < best[0]:
            best = (key, (rows, cols))
    if best is None:
        raise ValueError(f"no grid for {n} views of aspect {view_aspect:.3f}")
    return best[1]


def choose_layout(g: dict, n: int, view_aspect: float) -> tuple[int, int, str]:
    """case.json grid only when a person set it (grid.source == "user"); otherwise automatic, per run (pure)."""
    if g.get("source") == "user" and g.get("rows") and g.get("cols"):
        rows, cols = int(g["rows"]), int(g["cols"])
        if rows * cols < n:
            raise common.StageError(f"case.json grid {rows}x{cols} (set by a person, grid.source = \"user\") cannot hold "
                                    f"{n} views: raise rows/cols or set grid.source to \"auto\"")
        return rows, cols, "user (case.json grid)"
    rows, cols = grid_shape(n, view_aspect)
    return rows, cols, "automatic (grid_shape); stored in grid/layout.json only"


def cell_size(view_w: int, view_h: int, rows: int, cols: int, cap: dict) -> tuple[int, int]:
    """Largest (cw, ch), multiples of 8, aspect of the view, rows*cols*cw*ch <= cap max_area, sides <= max_side."""
    max_area = int(cap["max_area"])
    max_side = int(cap.get("max_side") or 1 << 30)
    aspect = view_w / view_h
    per_cell = max_area / (rows * cols)
    ch = int(math.sqrt(per_cell / aspect))
    while ch > 0:
        chm = (ch // MULTIPLE) * MULTIPLE
        cw = int(round(chm * aspect / MULTIPLE)) * MULTIPLE
        if chm > 0 and cw > 0 and rows * cols * cw * chm <= max_area and cols * cw <= max_side and rows * chm <= max_side:
            ch = chm
            break
        ch -= 1
    if ch <= 0:
        raise ValueError(f"the pixel cap {cap} leaves no room for a {rows}x{cols} grid")
    min_area = int(cap.get("min_area") or 0)
    if rows * cols * cw * ch < min_area:
        raise ValueError(f"grid {cols * cw}x{rows * ch} is below the model's minimum area {min_area}")
    return cw, ch


def cell_rects(rows: int, cols: int, cw: int, ch: int) -> list[dict]:
    """Exact integer rectangles, row-major, 1-based index, x1/y1 exclusive."""
    out = []
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        out.append({"index": k + 1, "row": r, "col": c, "x0": c * cw, "y0": r * ch,
                    "x1": (c + 1) * cw, "y1": (r + 1) * ch, "w": cw, "h": ch})
    return out


def assemble(images: list[np.ndarray], rows: int, cols: int, cw: int, ch: int) -> np.ndarray:
    """Tile images (already cw x ch) row-major; missing panels are flat grey."""
    sheet = np.full((rows * ch, cols * cw, 3), BLANK_GREY, np.uint8)
    for rect, img in zip(cell_rects(rows, cols, cw, ch), images):
        if img.shape[:2] != (ch, cw):
            raise ValueError(f"panel {rect['index']} is {img.shape[1]}x{img.shape[0]}, expected {cw}x{ch}")
        sheet[rect["y0"]:rect["y1"], rect["x0"]:rect["x1"]] = img
    return sheet


def check_exact(sheet: np.ndarray, images: list[np.ndarray], rows: int, cols: int, cw: int, ch: int) -> None:
    for rect, img in zip(cell_rects(rows, cols, cw, ch), images):
        if not np.array_equal(sheet[rect["y0"]:rect["y1"], rect["x0"]:rect["x1"]], img):
            raise AssertionError(f"grid panel {rect['index']} differs from its view")


def order_references(sheet: dict, user_refs: list[dict], clays: list[dict], max_images: int):
    """Ordered list (sheet, user refs, per-view clays) within max_images; returns (refs, dropped)."""
    head = [sheet, *user_refs]
    if len(head) > max_images:
        raise ValueError(f"the sheet plus {len(user_refs)} reference image(s) already exceed the model's limit of "
                         f"{max_images} images; remove references from case.json")
    room = max_images - len(head)
    kept, dropped = clays[:room], clays[room:]
    refs = [{**r, "image": i + 1} for i, r in enumerate([*head, *kept])]
    return refs, dropped


def check_distinct(refs: list[dict]) -> None:
    seen = {}
    for r in refs:
        if r["sha256"] in seen:
            raise AssertionError(f"references {seen[r['sha256']]} and {r['path']} are the same file")
        seen[r["sha256"]] = r["path"]


def layout_sentence(rows: int, cols: int, n: int, refs: list[dict]) -> str:
    """The layout part of the grid prompt (also used by tests)."""
    clays = [r for r in refs if r.get("kind") == "clay"]
    s = (f"Image 1 is a sheet of exactly {rows} row{'s' if rows > 1 else ''} x {cols} column"
         f"{'s' if cols > 1 else ''} of panels, read in row-major order, with no gutters.")
    if rows * cols > n:
        s += f" Panels {n + 1} to {rows * cols} are flat grey and must stay flat grey."
    if clays:
        first, last = clays[0]["image"], clays[-1]["image"]
        cells = [r["view_index"] for r in clays]
        if cells == list(range(1, len(cells) + 1)):
            s += (f" Images {first} to {last} are higher-resolution copies of panels 1 to {len(cells)}, in "
                  f"row-major order; use them as geometry references for their matching panels, not as extra panels.")
        else:
            s += " " + " ".join(f"Image {r['image']} is a higher-resolution copy of panel {r['view_index']}."
                               for r in clays) + " Use them as geometry references, not as extra panels."
    s += f" Output exactly {rows * cols} panels in the same {rows} x {cols} layout, not more."
    return s


# ---------------------------------------------------------------- stage
def register(sub) -> None:
    p = sub.add_parser("grid", help="exact rows x cols clay sheet, layout.json, ordered references manifest")
    p.add_argument("case")
    p.add_argument("--generator", choices=("fal", "mock"), default=None, help="override case.json model.generator")
    p.set_defaults(func=run)


def _ref_entry(case, path: Path, kind: str, role: str, **extra) -> dict:
    w, h = common.image_size(path)
    return {"path": common.rel(case, path), "kind": kind, "role": role, "sha256": common.sha256_file(path),
            "width": w, "height": h, **extra}


def _clay_panels(case, cams: list[dict], cw: int, ch: int):
    natives, scaled, notes = [], [], []
    for cam in cams:
        img = common.read_rgb(common.require(common.views_dir(case) / cam["name"] / "clay.png", f"mvb2r.py views {case}"))
        natives.append(img)
        small, method = common.resize_rgb(img, (cw, ch))
        scaled.append(small)
        notes.append(method)
    return natives, scaled, notes


def run(args) -> int:
    from . import model as model_mod
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    cams = common.load_cameras(case)
    mdl = model_mod.load_model(case, cfg, getattr(args, "generator", None))
    model_mod.announce("grid", mdl)
    W, H = int(cams[0]["width"]), int(cams[0]["height"])
    if any((int(c["width"]), int(c["height"])) != (W, H) for c in cams):
        raise common.StageError("all views must share one calibration resolution")
    rows, cols, source = choose_layout(cfg.get("grid") or {}, len(cams), W / H)
    cw, ch = cell_size(W, H, rows, cols, mdl["pixel_cap"])
    out = case / "grid"
    natives, scaled, notes = _clay_panels(case, cams, cw, ch)
    master = assemble(natives, rows, cols, W, H)
    check_exact(master, natives, rows, cols, W, H)
    cond = assemble(scaled, rows, cols, cw, ch)
    common.write_rgb(out / "clay_grid_master.png", master)
    common.write_rgb(out / "clay_grid_conditioning.png", cond)
    layout = {"rows": rows, "cols": cols, "order": "row-major", "gutter_px": 0, "n_views": len(cams),
              "cell_w": cw, "cell_h": ch, "width": cols * cw, "height": rows * ch,
              "calibration": [W, H], "master": "grid/clay_grid_master.png",
              "conditioning": "grid/clay_grid_conditioning.png",
              "conditioning_resize": sorted(set(notes)), "layout_source": source, "pixel_cap": mdl["pixel_cap"],
              "cells": [{**r, "view": cams[r["index"] - 1]["name"] if r["index"] <= len(cams) else None}
                        for r in cell_rects(rows, cols, cw, ch)]}
    common.write_json(out / "layout.json", layout)
    refs, dropped = build_references(case, cfg, cams, mdl)
    common.write_json(out / "references.json", {"max_images": mdl["max_images"], "references": refs,
                                                "dropped": dropped,
                                                "drop_rule": "per-view clay copies dropped from the last view backwards"})
    common.say(f"[grid] {rows}x{cols} panels of {cw}x{ch} (sheet {cols * cw}x{rows * ch}, "
               f"{rows * cols * cw * ch} px of cap {mdl['pixel_cap']['max_area']}); master {cols * W}x{rows * H} exact")
    common.say(f"[grid] {len(refs)} references (limit {mdl['max_images']})"
               + (f"; dropped clay copies of {', '.join(d['view'] for d in dropped)}" if dropped else ""))
    return 0


def build_references(case, cfg: dict, cams: list[dict], mdl: dict):
    sheet = _ref_entry(case, case / "grid" / "clay_grid_conditioning.png", "grid", "layout and camera framing")
    user = []
    for r in cfg.get("references", []):
        p = common.inside_case(case, r["path"])
        user.append(_ref_entry(case, common.require(p, "case.json references"), "reference", r.get("role", "other"),
                               applies_to=r.get("applies_to", []), note=common.clean_text(r.get("note", ""))))
    clays = [_ref_entry(case, common.views_dir(case) / c["name"] / "clay.png", "clay", f"geometry of panel {i + 1}",
                        view=c["name"], view_index=i + 1) for i, c in enumerate(cams)]
    refs, dropped = order_references(sheet, user, clays, int(mdl["max_images"]))
    check_distinct(refs)
    return refs, [{"view": d["view"], "path": d["path"]} for d in dropped]
