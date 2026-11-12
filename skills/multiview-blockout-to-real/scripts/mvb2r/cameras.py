"""`cameras`: keep the scene's cameras or place them automatically; per-object coverage from exact ray casts.

    mvb2r.py cameras CASE [--replace] [--count N] [--closeup OBJECT] [--no-elevated] [--arc DEG]

Camera maths (pure numpy, also imported by scripts/blender/place_cameras.py and passes.py):
  look_at_c2w, intrinsics (Blender lens/sensor/shift -> K), w2c_opencv, projection, project.
Automatic placement (plan_auto_cameras): an arc of views around the subject's bounding sphere, an optional
close-up on a named object, an optional elevated wide. When the scene is a room (structure slabs on at least
three sides) every camera is kept inside the room, never behind a wall; a camera pulled in by a wall gets a
wider lens so the subject still fits.
"""
from __future__ import annotations

import math

import numpy as np

from . import common

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
DEFAULT_LENS, DEFAULT_SENSOR = 35.0, 36.0
LENS_MIN, LENS_MAX = 12.0, 120.0
ROOM_MARGIN = 0.2
FRAME_MARGIN = 1.15


# ---------------------------------------------------------------- camera maths
def look_at_c2w(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Blender camera matrix_world looking from eye to target (camera -Z forward, +Y up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        raise ValueError("camera location equals its target")
    z = -fwd / n
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:                       # looking straight up/down
        x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def intrinsics(width: int, height: int, lens_mm: float, sensor_width_mm: float = DEFAULT_SENSOR,
               sensor_height_mm: float = 24.0, sensor_fit: str = "AUTO", shift_x: float = 0.0,
               shift_y: float = 0.0) -> np.ndarray:
    """K of a Blender perspective camera (square pixels), pixel-corner origin."""
    fit = sensor_fit if sensor_fit != "AUTO" else ("HORIZONTAL" if width >= height else "VERTICAL")
    sensor = sensor_width_mm if (fit == "HORIZONTAL" or sensor_fit == "AUTO") else sensor_height_mm
    maxdim = width if fit == "HORIZONTAL" else height
    f = lens_mm / sensor * maxdim
    cx = width / 2.0 - shift_x * maxdim
    cy = height / 2.0 + shift_y * maxdim
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def w2c_opencv(c2w) -> np.ndarray:
    return FLIP @ np.linalg.inv(np.asarray(c2w, float))


def projection(K, w2c) -> np.ndarray:
    return np.asarray(K, float) @ np.asarray(w2c, float)[:3]


def project(P, X) -> np.ndarray:
    """World points (N,3) -> (N,3) array of u, v, Zc."""
    X = np.atleast_2d(np.asarray(X, float))
    h = np.c_[X, np.ones(len(X))] @ np.asarray(P, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.c_[h[:, 0] / h[:, 2], h[:, 1] / h[:, 2], h[:, 2]]


def pixel_rays(cam: dict):
    """World-space unit ray directions through every pixel centre: (origin, dirs (H*W,3), (H, W))."""
    W, H = int(cam["width"]), int(cam["height"])
    K, c2w = np.asarray(cam["K"], float), np.asarray(cam["c2w_blender"], float)
    U, V = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cv = np.stack([(U - K[0, 2]) / K[0, 0], (V - K[1, 2]) / K[1, 1], np.ones_like(U)], -1).reshape(-1, 3)
    d_w = (d_cv * np.array([1.0, -1.0, -1.0])) @ c2w[:3, :3].T
    return c2w[:3, 3].copy(), d_w / np.linalg.norm(d_w, axis=1, keepdims=True), (H, W)


def fov_for_lens(lens_mm: float, sensor_mm: float = DEFAULT_SENSOR) -> float:
    return 2.0 * math.atan(sensor_mm / 2.0 / lens_mm)


def lens_for_fov(fov: float, sensor_mm: float = DEFAULT_SENSOR) -> float:
    return sensor_mm / 2.0 / math.tan(fov / 2.0)


# ---------------------------------------------------------------- scene analysis
def bbox_of(objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not objects:
        raise ValueError("no objects")
    lo = np.min([o["bbox_min"] for o in objects], axis=0)
    hi = np.max([o["bbox_max"] for o in objects], axis=0)
    return np.asarray(lo, float), np.asarray(hi, float)


def subject_objects(objects: list[dict]) -> list[dict]:
    """Everything that is not structure or an opening; all objects when that leaves none."""
    subj = [o for o in objects if o.get("category") != "structure" and not o.get("opening")]
    return subj or list(objects)


def _slab_at(struct: list[dict], axis: int, end: int, lo, hi, tol: float):
    """The thin, wide structure slab at one side of the structure box, or None."""
    span = hi - lo
    other = 1 - axis if axis < 2 else 0
    for o in struct:
        bmin, bmax = np.asarray(o["bbox_min"], float), np.asarray(o["bbox_max"], float)
        thick = bmax[axis] - bmin[axis]
        at_side = abs(bmin[axis] - lo[axis]) <= tol if end == 0 else abs(bmax[axis] - hi[axis]) <= tol
        if at_side and thick <= max(tol, 0.1 * span[axis]) and (bmax[other] - bmin[other]) >= 0.5 * span[other]:
            return bmin, bmax
    return None


def detect_room(objects: list[dict], tol: float = 0.15):
    """(lo, hi) interior bounds when structure slabs bound at least three horizontal sides, else None."""
    struct = [o for o in objects if o.get("category") == "structure"]
    if len(struct) < 3:
        return None
    lo, hi = bbox_of(struct)
    inner_lo, inner_hi, sides = lo.copy(), hi.copy(), 0
    for axis in (0, 1, 2):
        for end in (0, 1):
            slab = _slab_at(struct, axis, end, lo, hi, tol)
            if slab is None:
                continue
            sides += axis < 2
            if end == 0:
                inner_lo[axis] = slab[1][axis]
            else:
                inner_hi[axis] = slab[0][axis]
    if sides < 3 or np.any(inner_hi <= inner_lo):
        return None
    return inner_lo, inner_hi


def clamp_inside(eye, target, room, margin: float = ROOM_MARGIN) -> np.ndarray:
    """Move eye along the line to target until it is inside the room bounds shrunk by margin."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    if room is None:
        return eye
    lo, hi = np.asarray(room[0], float) + margin, np.asarray(room[1], float) - margin
    hi = np.maximum(hi, lo)
    tgt = np.clip(target, lo, hi)
    d = eye - tgt
    t_max = 1.0
    for a in range(3):
        if d[a] > 1e-12:
            t_max = min(t_max, (hi[a] - tgt[a]) / d[a])
        elif d[a] < -1e-12:
            t_max = min(t_max, (lo[a] - tgt[a]) / d[a])
    return np.clip(tgt + d * max(t_max, 0.0), lo, hi)


# ---------------------------------------------------------------- automatic placement
def _front_direction(centre: np.ndarray, room) -> np.ndarray:
    """Horizontal unit vector from the subject towards where the cameras stand."""
    if room is not None:
        mid = (np.asarray(room[0]) + np.asarray(room[1])) / 2.0
        d = mid - centre
        d[2] = 0.0
        if np.linalg.norm(d) > 0.25:
            return d / np.linalg.norm(d)
    return np.array([0.0, -1.0, 0.0])


def _place(name, role, centre, radius, azimuth, elevation, lens, room, front, fit_scale=1.0):
    fov = fov_for_lens(lens)
    dist = FRAME_MARGIN * fit_scale * radius / math.sin(fov / 2.0)
    base = math.atan2(front[1], front[0])
    a, e = base + azimuth, elevation
    eye = centre + dist * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
    eye2 = clamp_inside(eye, centre, room)
    d2 = float(np.linalg.norm(eye2 - centre))
    if d2 < dist - 1e-6:                          # pulled in by a wall: widen the lens to keep the framing
        need = 2.0 * math.asin(min(0.999, FRAME_MARGIN * fit_scale * radius / max(d2, 1e-6)))
        lens = lens_for_fov(need)
    lens = float(min(max(lens, LENS_MIN), LENS_MAX))
    return {"name": name, "role": role, "source": "auto", "location": [round(float(x), 4) for x in eye2],
            "look_at": [round(float(x), 4) for x in centre], "lens_mm": round(lens, 3),
            "sensor_width_mm": DEFAULT_SENSOR, "pulled_in_by_room": bool(d2 < dist - 1e-6)}


def plan_auto_cameras(objects: list[dict], auto: dict) -> list[dict]:
    """Camera dicts for an automatic rig (see module doc). `auto`: count, arc_degrees, closeup, elevated."""
    count = int(auto.get("count", 6))
    if not common.MIN_VIEWS <= count <= common.MAX_VIEWS:
        raise ValueError(f"views.auto.count must be {common.MIN_VIEWS}..{common.MAX_VIEWS}, got {count}")
    room = detect_room(objects)
    lo, hi = bbox_of(subject_objects(objects))
    centre, radius = (lo + hi) / 2.0, max(float(np.linalg.norm(hi - lo)) / 2.0, 0.05)
    front = _front_direction(centre, room)
    close = next((o for o in objects if o.get("name") == auto.get("closeup")), None)
    if auto.get("closeup") and close is None:
        raise ValueError(f"views.auto.closeup {auto['closeup']!r} is not an object of the case")
    n_arc = count - (1 if close else 0) - (1 if auto.get("elevated", True) and count >= 4 else 0)
    arc = math.radians(float(auto.get("arc_degrees", 120)))
    cams = []
    for k in range(n_arc):
        az = 0.0 if n_arc == 1 else -arc / 2 + arc * k / (n_arc - 1)
        cams.append(_place("", f"arc {math.degrees(az):+.0f} deg", centre, radius, az, math.radians(12),
                           DEFAULT_LENS, room, front))
    if auto.get("elevated", True) and count >= 4:
        cams.append(_place("", "elevated wide", centre, radius, 0.0, math.radians(40), 28.0, room, front, 1.05))
    if close:
        c_lo, c_hi = np.asarray(close["bbox_min"], float), np.asarray(close["bbox_max"], float)
        c_c, c_r = (c_lo + c_hi) / 2, max(float(np.linalg.norm(c_hi - c_lo)) / 2, 0.03)
        cams.append(_place("", f"close-up on {close['name']}", c_c, c_r, math.radians(10), math.radians(8),
                           50.0, room, front))
    return [{**c, "name": common.view_name(i + 1)} for i, c in enumerate(cams)]


def coverage_table(coverage: dict) -> list[str]:
    """Printable rows of views/coverage.json."""
    rows = [f"  {'object':<28} {'seen>=1':>8} {'seen>=2':>8}  best view"]
    for name, c in sorted(coverage.get("objects", {}).items()):
        per = c.get("per_view", {})
        best = max(per, key=per.get) if per else "-"
        rows.append(f"  {name[:28]:<28} {c['seen_ge1']:>8.2f} {c['seen_ge2']:>8.2f}  {best}")
    return rows


# ---------------------------------------------------------------- stage
def register(sub) -> None:
    p = sub.add_parser("cameras", help="(re)place automatic cameras, preview sheet, per-object coverage")
    p.add_argument("case")
    p.add_argument("--replace", action="store_true", help="replace existing cameras by an automatic rig")
    p.add_argument("--count", type=int, default=None, help=f"number of views ({common.MIN_VIEWS}-{common.MAX_VIEWS})")
    p.add_argument("--closeup", default=None, help="object to add a close-up view of")
    p.add_argument("--arc", type=float, default=None, help="arc of the automatic views in degrees")
    p.add_argument("--no-elevated", action="store_true", help="no elevated wide view")
    p.set_defaults(func=run)


def _update_auto(cfg: dict, args) -> dict:
    views = dict(cfg.get("views") or {})
    auto = dict(views.get("auto") or {})
    if args.count is not None:
        auto["count"] = args.count
    if args.closeup is not None:
        auto["closeup"] = args.closeup
    if args.arc is not None:
        auto["arc_degrees"] = args.arc
    if args.no_elevated:
        auto["elevated"] = False
    return {**cfg, "views": {**views, "auto": auto}}


def run(args) -> int:
    case = common.case_dir(args.case)
    cfg = _update_auto(common.load_case(case), args)
    common.save_case(case, cfg)
    extra = ["--case", str(case)] + (["--replace"] if args.replace else [])
    log = case / "logs" / "cameras.log"
    common.say(f"[cameras] placing / checking cameras in Blender (log: {common.rel(case, log)}) ...")
    rc = common.run_blender("place_cameras.py", extra, log)
    if rc != 0:
        common.say(f"[cameras] FAILED (exit {rc}):\n{common.log_tail(log)}")
        return rc
    for line in common.log_tail(log, 400).splitlines():
        if line.startswith("[place_cameras] NOTE"):
            common.say("  " + line)
    cov = common.read_json(common.views_dir(case) / "coverage.json")
    cams = common.load_case(case)["views"]["cameras"]
    common.say(f"[cameras] {len(cams)} cameras: " + ", ".join(f"{c['name']}={c['camera']} ({c['source']})" for c in cams))
    for row in coverage_table(cov):
        common.say(row)
    common.say("[cameras] preview sheet: views/preview_sheet.png; coverage: views/coverage.json")
    return 0
