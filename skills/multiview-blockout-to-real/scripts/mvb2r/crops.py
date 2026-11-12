"""Full-size review crops: fixed boxes judged at native resolution, the same boxes in every round.

A builder looking at reduced contact sheets calls results acceptable that a fresh reviewer at 100 % rejects
(faces, hands). So `critique` also writes, per box, a strip target | re-render <method> ... cut from the
images at their own pixels, never downscaled (a box under 256 px is enlarged x2 nearest-neighbour and says so).

Boxes, in pixels of targets/<view>.png, per usable view:
  detail_critical  bounding box of each detail_critical object in the clay id pass, padded 10 % (skipped < 32 px)
  error_tile       the three 256 px tiles with the largest mean |target - re-render| over object pixels
  user             case.json review.crops: [{"view": "view_01", "name": "head", "box": [x0, y0, x1, y1]}]
They are written to out/sheets/crop_boxes.json the first time and reused afterwards (`--new-boxes` recomputes;
user boxes added to case.json later are appended). Strips go to critique_in/crops/<round>/; other rounds stay.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from mvb2r import common
from mvb2r.common import StageError

BOXES_FILE = Path("out") / "sheets" / "crop_boxes.json"
DEFAULT_ROUND = "round_1"
TILE = 256                       # error tile side and the size under which a box is enlarged
ENLARGE = 2                      # nearest-neighbour factor for boxes under TILE
N_TILES = 3
MAX_TILE_OVERLAP = 0.25          # chosen error tiles overlap by at most this fraction of a tile
MIN_TILE_OBJECT = 0.25           # a tile counts only when this fraction of it is object pixels
PAD = 0.10                       # detail_critical boxes: padding per side, fraction of the box size
MIN_DETAIL_PX = 32               # detail_critical objects smaller than this (larger side) are skipped
MAX_USER_CROPS = 64
HEADER_PX, GAP_PX = 22, 4
LABEL_MAX = 40                   # names from case files drawn in headers / used in file names are cut to this
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bad(msg: str) -> StageError:
    return StageError(f"case.json review.crops: {msg}")


def _check_box(box, size: tuple[int, int], where: str) -> list[int]:
    W, H = size
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) and not isinstance(v, bool) for v in box)):
        raise _bad(f"{where}: box must be four integers [x0, y0, x1, y1]")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise _bad(f"{where}: box {box} must satisfy 0 <= x0 < x1 <= {W} and 0 <= y0 < y1 <= {H}")
    return list(box)


def user_boxes(cfg: dict, sizes: dict) -> list[dict]:
    """Validated case.json review.crops; `sizes` = {usable view: (W, H)} of its target."""
    review = cfg.get("review", {})
    if not isinstance(review, dict):
        raise _bad("review must be an object")
    raw = review.get("crops", [])
    if not isinstance(raw, list) or len(raw) > MAX_USER_CROPS:
        raise _bad(f"crops must be a list of at most {MAX_USER_CROPS} boxes")
    out, seen = [], set()
    for i, c in enumerate(raw):
        where = f"crop {i + 1}"
        if not isinstance(c, dict):
            raise _bad(f"{where} must be an object with view, name and box")
        view, name = c.get("view"), c.get("name")
        if view not in sizes:
            raise _bad(f"{where}: view {str(view)[:20]!r} is not a usable view ({sorted(sizes)})")
        if not common.is_safe_name(name):
            raise _bad(f"{where}: name {str(name)[:40]!r} must be letters, digits, space, _ . - only")
        if (view, name) in seen:
            raise _bad(f"{where}: {view} already has a box named {name!r}")
        seen.add((view, name))
        out.append({"view": view, "name": name, "box": _check_box(c.get("box"), sizes[view], where), "source": "user"})
    return out


def detail_boxes(ids: np.ndarray, objects: list, view: str) -> list[dict]:
    """Padded bounding box of each (name, id) object's pixels in an id map (skipped when under MIN_DETAIL_PX)."""
    H, W = ids.shape
    out = []
    for name, oid in objects:
        ys, xs = np.nonzero(ids == oid)
        if not len(xs):
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if max(x1 - x0, y1 - y0) < MIN_DETAIL_PX:
            continue
        px, py = int(round(PAD * (x1 - x0))), int(round(PAD * (y1 - y0)))
        box = [max(x0 - px, 0), max(y0 - py, 0), min(x1 + px, W), min(y1 + py, H)]
        out.append({"view": view, "name": common.safe_name(name), "box": box, "source": "detail_critical"})
    return out


def _positions(n: int, tile: int) -> list[int]:
    return sorted(set(list(range(0, n - tile + 1, max(tile // 2, 1))) + [n - tile]))


def _overlap(a: list, b: list) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _error_map(target: np.ndarray, rerenders: list) -> np.ndarray:
    H, W = target.shape[:2]
    errs = [np.abs(target.astype(np.float32) - (r if r.shape[:2] == (H, W) else
                   cv2.resize(r, (W, H), interpolation=cv2.INTER_AREA)).astype(np.float32)).mean(-1)
            for r in rerenders]
    return np.mean(errs, 0)


def error_tiles(target: np.ndarray, rerenders: list, obj: np.ndarray, view: str) -> list[dict]:
    """The N_TILES tiles with the largest mean |target - re-render| over object pixels (all methods averaged)."""
    if not rerenders or not obj.any():
        return []
    H, W = obj.shape
    t = min(TILE, W, H)
    err = _error_map(target, rerenders)
    cands = []
    for y in _positions(H, t):
        for x in _positions(W, t):
            m = obj[y:y + t, x:x + t]
            if m.mean() >= MIN_TILE_OBJECT:
                cands.append((float(err[y:y + t, x:x + t][m].mean()), [x, y, x + t, y + t]))
    chosen = []
    for score, box in sorted(cands, key=lambda c: -c[0]):
        if all(_overlap(box, c["box"]) <= MAX_TILE_OVERLAP * t * t for c in chosen):
            chosen.append({"view": view, "name": f"error_{len(chosen) + 1}", "box": box, "source": "error_tile",
                           "mean_abs_error_8bit": round(score, 2)})
        if len(chosen) == N_TILES:
            break
    return chosen


def _read(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR) if Path(path).is_file() else None


def rerender_path(case: Path, method: str, view: str) -> Path:
    """proj/<method>/rerender/<view>.png, refused when it resolves outside the case (a linked method folder or
    file would put an outside image into the reviewer pack)."""
    return common.inside_case(case, Path("proj") / common.file_token(method) / "rerender" / f"{common.file_token(view)}.png")


def _ids_at(case: Path, view: str, size: tuple[int, int]) -> np.ndarray | None:
    p = common.views_dir(case) / view / "object_id.png"
    if not p.is_file():
        return None
    ids = common.read_gray(p)
    return ids if ids.shape[::-1] == tuple(size) else cv2.resize(ids, size, interpolation=cv2.INTER_NEAREST)


def _detail_objects(case: Path) -> list:
    try:
        objs = common.case_objects(case)
    except StageError:
        return []
    return [(o["name"], int(o["id"])) for o in objs if o.get("detail_critical") and isinstance(o.get("id"), int)
            and isinstance(o.get("name"), str)]


def usable_targets(case: Path) -> dict:
    """{view: target path} for the usable views of targets/targets.json whose target file exists."""
    t = common.read_json(case / "targets" / "targets.json")
    out = {}
    for v, x in sorted(t.get("views", {}).items()):
        if x.get("usable") is False or not common.is_safe_name(v):
            continue
        p = common.inside_case(case, x.get("target_png", f"targets/{v}.png"))
        if p.is_file():
            out[v] = p
    return out


def compute_boxes(case: Path, targets: dict, methods: list) -> list[dict]:
    boxes, detail = [], _detail_objects(case)
    for v, tp in targets.items():
        target = _read(tp)
        if target is None:
            raise StageError(f"cannot read target {tp.name} - re-run `mvb2r.py targets {case.name}`")
        ids = _ids_at(case, v, target.shape[1::-1])
        rr = [x for x in (_read(rerender_path(case, m, v)) for m in methods) if x is not None]
        if ids is not None:
            boxes += detail_boxes(ids, detail, v) + error_tiles(target, rr, ids > 0, v)
    return boxes


# ---------------------------------------------------------------- boxes file
def _load_saved(path: Path, sizes: dict) -> list[dict]:
    data = common.read_json(path)
    boxes = data.get("boxes") if isinstance(data, dict) else None
    if not isinstance(boxes, list):
        raise StageError(f"{path.name} has no 'boxes' list: fix it or run `critique CASE --new-boxes`")
    out = []
    for i, b in enumerate(boxes):
        ok = isinstance(b, dict) and b.get("view") in sizes and common.is_safe_name(b.get("name"))
        try:
            box = _check_box(b.get("box"), sizes[b["view"]], f"box {i + 1}") if ok else None
        except StageError:
            box = None
        if box is None:
            raise StageError(f"{path.name} box {i + 1} ({common.clean_text(str(b), 80)}) does not fit the current "
                             "usable targets (resized or rejected view?): run `critique CASE --new-boxes`")
        out.append({**b, "box": box})
    return out


def _write_boxes(case: Path, boxes: list, sizes: dict) -> None:
    common.write_json(case / BOXES_FILE, {
        "frame": "pixels of targets/<view>.png; reused by every later `critique` round (--new-boxes recomputes)",
        "sizes": {v: list(s) for v, s in sizes.items()}, "boxes": boxes})


def resolve_boxes(case: Path, targets: dict, sizes: dict, methods: list, users: list, new_boxes: bool) -> list[dict]:
    """Saved boxes (plus user boxes added since), or freshly computed ones written for later rounds."""
    path = case / BOXES_FILE
    if path.is_file() and not new_boxes:
        saved = _load_saved(path, sizes)
        have = {(b["view"], b["name"]): b for b in saved}
        for u in users:
            if (u["view"], u["name"]) in have and have[(u["view"], u["name"])]["box"] != u["box"]:
                common.say(f"[critique] case.json box {u['view']}/{cap_label(u['name'])} ignored: the saved box of "
                           "that name is kept (fixed boxes); `critique CASE --new-boxes` applies it")
        added = [u for u in users if (u["view"], u["name"]) not in have]
        if added:
            _write_boxes(case, saved + added, sizes)
            common.say(f"[critique] added {len(added)} user box(es) from case.json to {BOXES_FILE}")
        return saved + added
    keys = {(u["view"], u["name"]) for u in users}
    boxes = [b for b in compute_boxes(case, targets, methods) if (b["view"], b["name"]) not in keys] + users
    _write_boxes(case, boxes, sizes)
    return boxes


# ---------------------------------------------------------------- strips
def cap_label(text) -> str:
    """A name from a case file shortened for a strip header or file name (labels, never instructions)."""
    t = common.clean_text(str(text), LABEL_MAX + 1)
    return t if len(t) <= LABEL_MAX else t[:LABEL_MAX - 1].rstrip(" .") + "_"


def _with_header(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((HEADER_PX, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (4, HEADER_PX - 7), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _hcat(cols: list) -> np.ndarray:
    h = max(c.shape[0] for c in cols)
    parts = []
    for c in cols:
        parts += [np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0))), np.full((h, GAP_PX, 3), 40, np.uint8)]
    return np.hstack(parts[:-1])


def strip(cols: list, box: list, size: tuple[int, int]) -> tuple[np.ndarray, str]:
    """(labelled strip, "native" | "x2 nearest"): the box cut from each (label, BGR image) at its own pixels."""
    W, H = size
    x0, y0, x1, y1 = box
    k = ENLARGE if max(x1 - x0, y1 - y0) < TILE else 1
    tiles = []
    for label, img in cols:
        h, w = img.shape[:2]
        sx, sy = w / W, h / H
        c = img[int(round(y0 * sy)):int(round(y1 * sy)), int(round(x0 * sx)):int(round(x1 * sx))]
        c = np.repeat(np.repeat(c, k, 0), k, 1) if k > 1 else c
        tiles.append(_with_header(c, label if (w, h) == (W, H) else f"{label} ({w}x{h} px image)"))
    return _hcat(tiles), ("native" if k == 1 else f"x{k} nearest")


def strip_labels(boxes: list) -> list[str]:
    """Capped box names, made unique within each view by a numeric suffix inside the cap."""
    used, out = set(), []
    for b in boxes:
        base, label, k = cap_label(b["name"]), cap_label(b["name"]), 1
        while (b["view"], common.file_token(label)) in used:
            k += 1
            label = cap_label(base)[:LABEL_MAX - len(f"_{k}")].rstrip(" .") + f"_{k}"
        used.add((b["view"], common.file_token(label)))
        out.append(label)
    return out


def write_strip(case: Path, b: dict, target: Path, size: tuple, methods: list, folder: Path,
                label: str | None = None) -> Path:
    cols = [("target", _read(target))]
    cols += [(f"re-render {cap_label(m)}", img) for m in methods
             if (img := _read(rerender_path(case, m, b['view']))) is not None]
    img, note = strip(cols, b["box"], size)
    label = label or cap_label(b["name"])
    out = folder / f"{b['view']}_{common.file_token(label)}.png"
    folder.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), _with_header(img, f"{b['view']} {label} {b['box']} ({note})")):
        raise StageError(f"cannot write {out}")
    return out


def pack(case: Path, out_root: Path, methods: list, round_name: str | None = None, new_boxes: bool = False) -> dict:
    """Crop strips of this round in out_root/crops/<round>/ (earlier rounds kept); returns boxes and strips."""
    rnd = round_name or DEFAULT_ROUND
    if not common.is_safe_name(rnd):
        raise StageError(f"--round {str(rnd)[:40]!r}: use letters, digits, space, _ . - only")
    folder = common.owned_path(case, Path(out_root) / "crops" / rnd)     # refuses planted links
    case = common.case_dir(case)
    targets = usable_targets(case)
    sizes = {v: common.image_size(p) for v, p in targets.items()}
    users = user_boxes(common.load_case(case), sizes)
    boxes = resolve_boxes(case, targets, sizes, methods, users, new_boxes)
    if folder.exists():
        shutil.rmtree(folder)
    strips = [write_strip(case, b, targets[b["view"]], sizes[b["view"]], methods, folder, label)
              for b, label in zip(boxes, strip_labels(boxes))]
    return {"round": rnd, "dir": folder, "boxes": boxes, "strips": strips}
