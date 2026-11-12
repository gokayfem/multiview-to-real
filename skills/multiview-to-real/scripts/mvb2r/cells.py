"""Seam detection in a generated R x C sheet and cell extraction (ported from the experiment, generalised).

Generated sheets are not cut at exact positions: each internal column seam is measured from the column
gradient profile of every row band, each row seam from the row profile of every column band, searched
within +-4 % of the ideal position. A weak seam (peak / median < MIN_PEAK_RATIO) falls back to the ideal
position and says so. Cells are cut as pure crops (cell_XX_native.png, no resampling); a copy resampled
to the calibration size is written next to it with its method ("resized, NOT generated detail" when enlarged).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common

MIN_PEAK_RATIO = 1.5


def gradient_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """Mean |difference| across columns (axis=1) or rows (axis=0); index i = boundary between i and i+1."""
    d = np.abs(np.diff(gray.astype(np.float32), axis=axis))
    return d.mean(axis=0 if axis == 1 else 1)


def find_seam(profile: np.ndarray, expected: float, half: int) -> dict:
    lo, hi = max(0, int(expected - half)), min(len(profile), int(expected + half) + 1)
    win = profile[lo:hi]
    i = int(np.argmax(win))
    med = float(np.median(win))
    ratio = float(win[i]) / max(med, 1e-6)
    strong = ratio >= MIN_PEAK_RATIO
    pos = lo + i + 1 if strong else int(round(expected))
    return {"expected": round(float(expected), 2), "boundary_px": int(pos), "offset_px": round(pos - expected, 2),
            "peak_to_median": round(ratio, 2), "used": "measured" if strong else "ideal (weak seam)"}


def detect(gray: np.ndarray, rows: int, cols: int, search_frac: float = 0.04) -> dict:
    h, w = gray.shape
    out = {"width": w, "height": h, "rows": rows, "cols": cols, "col_seams": [], "row_seams": []}
    pad_y, pad_x = int(h / rows * 0.1), int(w / cols * 0.06)
    for r in range(rows):
        band = gray[int(h * r / rows) + pad_y:int(h * (r + 1) / rows) - pad_y]
        prof = gradient_profile(band, axis=1)
        for k in range(1, cols):
            out["col_seams"].append({"row": r, "k": k, **find_seam(prof, w * k / cols, max(2, int(w * search_frac)))})
    for c in range(cols):
        band = gray[:, int(w * c / cols) + pad_x:int(w * (c + 1) / cols) - pad_x]
        prof = gradient_profile(band, axis=0)
        for k in range(1, rows):
            out["row_seams"].append({"col": c, "k": k, **find_seam(prof, h * k / rows, max(2, int(h * search_frac)))})
    return out


def rects_from_seams(det: dict) -> list[tuple[int, int, int, int]]:
    """Cell rectangles (x0, y0, x1, y1), row-major: each boundary = mean of its measurements."""
    rows, cols, w, h = det["rows"], det["cols"], det["width"], det["height"]
    xs = [0] + [int(round(np.mean([s["boundary_px"] for s in det["col_seams"] if s["k"] == k]))) for k in range(1, cols)] + [w]
    ys = [0] + [int(round(np.mean([s["boundary_px"] for s in det["row_seams"] if s["k"] == k]))) for k in range(1, rows)] + [h]
    if any(b <= a for a, b in zip(xs, xs[1:])) or any(b <= a for a, b in zip(ys, ys[1:])):
        raise common.StageError(f"seam detection produced an invalid layout: xs={xs} ys={ys}")
    return [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(rows) for c in range(cols)]


def extract(case, sheet_path: Path, layout: dict, cams: list[dict], out_dir: Path) -> dict:
    """Cut the cells of a generated sheet; write natives, calibration-size copies and cells.json."""
    rgb = common.read_rgb(sheet_path)
    gray = rgb.astype(np.float32).mean(axis=2)
    det = detect(gray, int(layout["rows"]), int(layout["cols"]))
    W, H = int(cams[0]["width"]), int(cams[0]["height"])
    cells = []
    for i, (x0, y0, x1, y1) in enumerate(rects_from_seams(det)[:len(cams)]):
        view = cams[i]["name"]
        crop = rgb[y0:y1, x0:x1]
        native = out_dir / f"cell_{view}_native.png"
        common.write_rgb(native, crop)
        calib, method = common.resize_rgb(crop, (W, H))
        calib_path = out_dir / f"cell_{view}_calib.png"
        common.write_rgb(calib_path, calib)
        cells.append({"view": view, "index": i + 1, "rect_in_sheet": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                                                      "w": x1 - x0, "h": y1 - y0},
                      "native": common.rel(case, native), "native_sha256": common.sha256_file(native),
                      "native_size": [x1 - x0, y1 - y0], "calib": common.rel(case, calib_path),
                      "calib_resize": method, "anisotropy": round(((x1 - x0) / W) / ((y1 - y0) / H), 4)})
    doc = {"sheet": common.rel(case, sheet_path), "sheet_size": [det["width"], det["height"]],
           "seam_detection": det, "cells": cells}
    common.write_json(out_dir / "cells.json", doc)
    return doc
