"""Contact sheets the agent must look at: clay | target | re-render (labelled), and simple image grids."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion_io import write_png


def _load(path: Path, size: int) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_COLOR) if Path(path).is_file() else None
    if a is None:
        a = np.full((size, size, 3), 60, np.uint8)
        cv2.putText(a, "missing", (10, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return a
    h, w = a.shape[:2]
    s = size / max(h, w)
    return cv2.resize(a, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def labelled(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_pad(imgs: list) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    return np.hstack([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs])


def grid(imgs: list, cols: int) -> np.ndarray:
    rows = [hstack_pad(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
    w = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows])


NO_GEOMETRY = "no geometry: not projected"


def mark_no_geometry(img: np.ndarray, ids_path: Path) -> np.ndarray:
    """Grey out (and label) the pixels of a re-render where the scene has no geometry (object id 0): an open
    scene's generated backdrop is not part of the scene and is never projected."""
    ids = cv2.imread(str(ids_path), cv2.IMREAD_UNCHANGED) if Path(ids_path).is_file() else None
    if ids is None:
        return img
    ids = ids[..., 0] if ids.ndim == 3 else ids
    empty = cv2.resize((ids == 0).astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    if not empty.any():
        return img
    out = img.copy()
    stripes = ((np.add.outer(np.arange(img.shape[0]), np.arange(img.shape[1])) // 6) % 2).astype(bool)
    out[empty] = np.where(stripes[empty][:, None], 70, 50)
    ys, xs = np.nonzero(empty)
    y = int(np.clip(np.median(ys), 40, img.shape[0] - 10))
    cv2.putText(out, NO_GEOMETRY, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def _rerender_tile(case: Path, view: str, method: str, size: int) -> np.ndarray:
    img = _load(case / "proj" / method / "rerender" / f"{view}.png", size)
    return mark_no_geometry(img, case / "proj" / "passes" / view / "object_id.png")


def triptych(case: Path, view: str, method: str, out: Path, size: int = 384) -> Path:
    clay = case / "views" / view / "clay.png"
    parts = [labelled(_load(clay, size), f"clay {view}"), labelled(_load(case / "targets" / f"{view}.png", size),
                                                                    "target (generated)"),
             labelled(_rerender_tile(case, view, method, size), f"re-render ({method})")]
    sheet = hstack_pad(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), sheet):
        raise IOError(f"cannot write {out}")
    return out


def write_grid(paths: list, labels: list, out: Path, cols: int, size: int = 320, ids_paths: list | None = None) -> Path:
    imgs = [labelled(mark_no_geometry(_load(p, size), q) if q else _load(p, size), t)
            for p, t, q in zip(paths, labels, ids_paths or [None] * len(paths))]
    write_png(out, grid(imgs, cols)[..., ::-1])
    return out
