"""Offline synthetic image model (`--generator mock`) for tests and demos. NOT a generator of real images.

It answers every edit job (grid sheet, per-view refine, consistency render-then-refine, size probe) by
rendering a HIDDEN ground-truth look of the scene from the job's camera(s), then adding the defects real
generated views have:
  - truth: per object a base colour (from its id) x world-space value noise at 8 cm and 1.2 cm, shaded by a
    fixed sun + ambient from the exact normal pass; background = a vertical sky gradient. Because the
    texture is a function of the world position it is identical in every view (a consistent "real" scene);
    gen/mock_truth/view_XX.png keeps it (calibration resolution) as ground truth for the tests.
  - per-view defects (seeded by view, family and attempt): a smooth gain field (+-GAIN) and colour tint
    (+-TINT), a smooth local warp of up to WARP_PX pixels, and view-specific fine detail (blurred noise)
    that does not line up between views.
  - defect amplitude per prompt family: grid 1.0, refine 0.8, consistency 0.45. These numbers are
    parameters of the mock, not measurements of any real model.
The mock renders at calibration resolution and resizes to the requested size (so its larger outputs carry
no extra detail - like any resize). Output files land exactly where the fal path writes them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import common

GAIN, TINT, WARP_PX, DETAIL = 0.12, 0.05, 2.5, 0.06
FAMILY_AMPLITUDE = {"grid": 1.0, "refine": 0.8, "consistency": 0.45, "probe": 1.0}
SUN = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])
TRUTH_SEED = 7


# ---------------------------------------------------------------- deterministic noise
def _hash3(ix, iy, iz, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791)
    h = (h ^ (seed * 2654435761)) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float64) / 65535.0


def value_noise3(p: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Smooth value noise in [0, 1] of world points p (..., 3) at a lattice spacing `scale` metres."""
    q = p / scale
    i0 = np.floor(q)
    f = q - i0
    s = f * f * (3 - 2 * f)
    i0 = i0.astype(np.int64)
    out = np.zeros(p.shape[:-1])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (s[..., 0] if dx else 1 - s[..., 0]) * (s[..., 1] if dy else 1 - s[..., 1]) * \
                    (s[..., 2] if dz else 1 - s[..., 2])
                out += w * _hash3(i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz, seed)
    return out


def smooth_field(shape: tuple[int, int], rng: np.random.Generator, cells: int = 4) -> np.ndarray:
    """Smooth random field in [-1, 1] of the given (H, W)."""
    import cv2
    small = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC).clip(-1, 1)


def base_colour(oid: int) -> np.ndarray:
    """Linear RGB base colour of object id `oid` (stable)."""
    h = hashlib.sha256(f"mvb2r-mock-{oid}".encode()).digest()
    rgb = np.array([0.15 + 0.6 * h[0] / 255, 0.15 + 0.6 * h[1] / 255, 0.15 + 0.6 * h[2] / 255])
    return rgb ** 1.6


def truth_albedo(oid: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Ground-truth linear albedo (..., 3) for object ids and world positions (0 = background -> 0)."""
    oid = np.asarray(oid)
    out = np.zeros(oid.shape + (3,))
    for i in np.unique(oid):
        if i == 0:
            continue
        m = oid == i
        p = pos[m].astype(np.float64)
        tex = 0.78 + 0.32 * value_noise3(p, 0.08, TRUTH_SEED + int(i)) + 0.22 * value_noise3(p, 0.012, TRUTH_SEED + 99 * int(i))
        out[m] = base_colour(int(i))[None, :] * tex[:, None]
    return np.clip(out, 0, 1)


def truth_render(oid: np.ndarray, pos: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Ground-truth linear image of one view from its exact passes."""
    shade = 0.35 + 0.65 * np.clip(nrm @ SUN, 0, 1)
    img = truth_albedo(oid, pos) * shade[..., None]
    h = oid.shape[0]
    sky = np.linspace(0.75, 0.45, h)[:, None, None] * np.array([0.55, 0.7, 1.0])
    return np.where((oid == 0)[..., None], np.broadcast_to(sky, img.shape), img)


# ---------------------------------------------------------------- defects
def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little"))


def defect(img_lin: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    """Per-view gain/tint field, smooth warp and view-specific fine detail (linear in, linear out)."""
    import cv2
    H, W = img_lin.shape[:2]
    gain = 1 + amp * GAIN * smooth_field((H, W), rng)
    tint = 1 + amp * TINT * np.stack([smooth_field((H, W), rng, 3) for _ in range(3)], -1)
    out = img_lin * gain[..., None] * tint
    dx = amp * WARP_PX * smooth_field((H, W), rng, 3)
    dy = amp * WARP_PX * smooth_field((H, W), rng, 3)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = cv2.remap(out.astype(np.float32), X + dx, Y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    detail = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8)
    out = out * (1 + amp * DETAIL * detail / max(float(detail.std()), 1e-6))[..., None]
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- jobs
def view_truth(case, view: str) -> np.ndarray:
    """Linear truth image of a view (cached as gen/mock_truth/view_XX.png, sRGB 8-bit)."""
    case = common.case_dir(case)
    cam = next((c for c in common.load_cameras(case) if c["name"] == view), None)
    if cam is None:
        raise common.StageError(f"mock: unknown view {view!r}")
    vp = common.view_passes(case, cam)
    nrm = common.read_exr(common.views_dir(case) / view / "normal_world.exr")
    lin = truth_render(vp["oid"], vp["pos"], nrm)
    path = case / "gen" / "mock_truth" / f"{view}.png"
    if not path.is_file():
        common.write_rgb(path, (common.linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8))
    return lin


def view_image(case, view: str, family: str, attempt: int, size: tuple[int, int]) -> np.ndarray:
    lin = view_truth(case, view)
    amp = FAMILY_AMPLITUDE.get(family, 1.0)
    out = defect(lin, _rng(view, family, attempt), amp)
    srgb = (common.linear_to_srgb(out) * 255 + 0.5).astype(np.uint8)
    return common.resize_rgb(srgb, size)[0]


def fault_panel(case, view: str, faults: dict, attempt: int, size: tuple[int, int]) -> np.ndarray:
    """A grid panel, with the test-only faults of case.json model.mock_faults: {"duplicate": {"view_04": "view_03"}}
    paints another view's panel in place of this one; {"shift": {"view_02": 0.15}} shifts its content sideways."""
    src = (faults.get("duplicate") or {}).get(view, view)
    img = view_image(case, common.check_name(src, "view"), "grid", attempt, size)
    frac = float((faults.get("shift") or {}).get(view, 0.0))
    return np.roll(img, int(round(frac * size[0])), axis=1) if frac else img


def grid_image(case, views: list[str], attempt: int, size: tuple[int, int]) -> np.ndarray:
    from . import grid as grid_mod
    layout = common.read_json(common.require(common.case_dir(case) / "grid" / "layout.json", f"mvb2r.py grid {case}"))
    rows, cols = int(layout["rows"]), int(layout["cols"])
    if size != (cols * layout["cell_w"], rows * layout["cell_h"]):
        cw, ch = size[0] // cols, size[1] // rows
    else:
        cw, ch = int(layout["cell_w"]), int(layout["cell_h"])
    faults = (common.load_case(case).get("model") or {}).get("mock_faults") or {}
    panels = [fault_panel(case, v, faults, attempt, (cw, ch)) for v in views]
    sheet = grid_mod.assemble(panels, rows, cols, cw, ch)
    return common.resize_rgb(sheet, size)[0]


def generate(case, job: dict, out: Path) -> str:
    """Write the mock answer of `job` to `out`; returns a mock request id."""
    size = (int(job["size"][0]), int(job["size"][1]))
    views = list(job.get("views") or [])
    if not views:
        raise common.StageError("mock: job names no view")
    attempt = int(job.get("attempt", 1))
    if job.get("family") == "grid":
        img = grid_image(case, views, attempt, size)
    else:
        img = view_image(case, views[0], job.get("family", "refine"), attempt, size)
    common.write_rgb(out, img)
    return "mock-" + hashlib.sha256(f"{job['tag']}|{attempt}|{size}".encode()).hexdigest()[:16]
