"""Calibrated pinhole cameras in the repository convention, novel-camera placement, camera paths.

OpenCV camera: +X right, +Y down, +Z forward; w2c = diag(1,-1,-1,1)·inv(c2w_blender);
u = fx·X/Z + cx, v = fy·Y/Z + cy, origin at the top-left corner of the top-left pixel, centres at +0.5.
Ported from the experiment's scripts/proj/camera.py and tour.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mvb2r.fusion_io import StageError, read_json

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray
    w2c: np.ndarray
    clip_start: float = 0.01
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def c2w_blender(self) -> np.ndarray:
        return np.linalg.inv(FLIP @ self.w2c)

    @property
    def centre(self) -> np.ndarray:
        return self.c2w_blender[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        return self.w2c[2, :3]

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, np.float64) @ self.w2c[:3, :3].T + self.w2c[:3, 3]

    def project(self, X: np.ndarray):
        """World points (N,3) -> (u, v, Zc); u = v = nan where Zc <= 0."""
        Xc = self.to_camera(X)
        z = Xc[:, 2]
        zs = np.where(z > 1e-9, z, np.nan)
        u = self.K[0, 0] * Xc[:, 0] / zs + self.K[0, 1] * Xc[:, 1] / zs + self.K[0, 2]
        v = self.K[1, 1] * Xc[:, 1] / zs + self.K[1, 2]
        return u, v, z

    def to_json(self) -> dict:
        d = dict(self.meta)
        d.update({"name": self.name, "width": self.width, "height": self.height, "K": self.K.tolist(),
                  "w2c_opencv": self.w2c.tolist(), "c2w_blender": self.c2w_blender.tolist(),
                  "clip_start": self.clip_start})
        return d


def camera_from_json(v: dict) -> Camera:
    try:
        K = np.array(v["K"], np.float64).reshape(3, 3)
        if "w2c_opencv" in v:
            w2c = np.array(v["w2c_opencv"], np.float64).reshape(4, 4)
        else:
            w2c = FLIP @ np.linalg.inv(np.array(v["c2w_blender"], np.float64).reshape(4, 4))
        W, H = int(v["width"]), int(v["height"])
    except (KeyError, ValueError, TypeError) as e:
        raise StageError(f"camera entry {v.get('name', '?')!r} malformed: {e}") from e
    if W < 8 or H < 8 or not np.all(np.isfinite(K)) or not np.all(np.isfinite(w2c)):
        raise StageError(f"camera {v.get('name')!r}: bad size or non-finite matrices")
    meta = {k: v[k] for k in ("role", "camera", "clip_end") if k in v}
    return Camera(name=str(v["name"]), width=W, height=H, K=K, w2c=w2c,
                  clip_start=float(v.get("clip_start", 0.01)), meta=meta)


def load_cameras(path: Path, producer: str = "mvb2r.py views") -> list[Camera]:
    data = read_json(path, producer)
    views = data.get("views", data.get("cameras")) if isinstance(data, dict) else data
    if not isinstance(views, list) or not views:
        raise StageError(f"{path}: expected a non-empty 'views' list")
    return [camera_from_json(v) for v in views]


def usable_cameras(case: Path, stage: str = "this stage", min_views: int = 2) -> list[Camera]:
    """Source cameras minus the views targets/targets.json marks "usable": false (reasons printed)."""
    from mvb2r.fusion_io import split_usable
    return split_usable(case, load_cameras(Path(case) / "views" / "cameras.json"), min_views, stage)[0]


def cameras_json(cams: list[Camera], extra: dict | None = None) -> dict:
    d = {"views": [c.to_json() for c in cams]}
    d.update(extra or {})
    return d


# ----------------------------------------------------------------------------------------------- interpolation
def _quat_from_R(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    qb = -qb if np.dot(qa, qb) < 0 else qb
    d = float(np.clip(np.dot(qa, qb), -1, 1))
    if d > 0.9995:
        q = qa + t * (qb - qa)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * qa + np.sin(t * th) * qb) / np.sin(th)


def interpolate_camera(a: Camera, b: Camera, t: float, name: str, pivot: np.ndarray | None = None) -> Camera:
    """Camera between a and b (size of a): slerped rotation, intrinsics lerped; the centre moves on a
    straight line, or on an arc around `pivot` (radius and angle interpolated) when given."""
    ca, cb = a.c2w_blender, b.c2w_blender
    q = slerp(_quat_from_R(ca[:3, :3]), _quat_from_R(cb[:3, :3]), t)
    c2w = np.eye(4)
    c2w[:3, :3] = _R_from_quat(q)
    c2w[:3, 3] = _arc_point(ca[:3, 3], cb[:3, 3], t, pivot)
    Kb = b.K.copy()
    Kb[0] *= a.width / b.width
    Kb[1] *= a.height / b.height
    K = (1 - t) * a.K + t * Kb
    K[2] = [0, 0, 1]
    meta = {"role": f"novel t={t:.2f} between {a.name} and {b.name}"}
    return Camera(name=name, width=a.width, height=a.height, K=K, w2c=FLIP @ np.linalg.inv(c2w),
                  clip_start=a.clip_start, meta=meta)


def vslerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation of two unit VECTORS (no quaternion sign flip: the short arc from a to b).
    Tiny angle: lerp. Nearly opposite (> 179 deg): the arc goes over the world up axis (or +X if a is vertical)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    th = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    if th < 1e-6:
        out = (1 - t) * a + t * b
        return out / np.linalg.norm(out)
    if th > np.pi - np.radians(1.0):
        up = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
        mid = up - np.dot(up, a) * a
        mid /= np.linalg.norm(mid)
        return vslerp(a, mid, 2 * t) if t <= 0.5 else vslerp(mid, b, 2 * t - 1)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


def _arc_point(pa, pb, t, pivot):
    if pivot is None:
        return (1 - t) * pa + t * pb
    da, db = pa - pivot, pb - pivot
    ra, rb = np.linalg.norm(da), np.linalg.norm(db)
    if ra < 1e-9 or rb < 1e-9:
        return (1 - t) * pa + t * pb
    return pivot + vslerp(da, db, t) * ((1 - t) * ra + t * rb)


# ----------------------------------------------------------------------------------------------- placement
def neighbour_pairs(cams: list[Camera], k: int) -> list[tuple[int, int]]:
    """Up to k pairs of neighbouring source cameras (closest by centre distance + viewing-angle difference)."""
    n = len(cams)
    if n < 2 or k <= 0:
        return []
    C = np.array([c.centre for c in cams])
    F = np.array([c.forward for c in cams])
    scale = max(float(np.median(np.linalg.norm(C - C.mean(0), axis=1))), 1e-6)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ang = float(np.arccos(np.clip(np.dot(F[i], F[j]), -1, 1)))
            pairs.append((np.linalg.norm(C[i] - C[j]) / scale + ang, i, j))
    pairs.sort()
    first = [(i, j) for _, i, j in pairs]
    out, used = [], np.zeros(n, int)
    for i, j in first:  # first pass: each camera in at most two pairs (a chain of neighbours)
        if used[i] < 2 and used[j] < 2:
            out.append((i, j))
            used[i] += 1
            used[j] += 1
    out += [p for p in first if p not in out]
    return out[:k]


def novel_cameras(cams: list[Camera], k: int, pivot: np.ndarray | None = None) -> list[Camera]:
    return [interpolate_camera(cams[i], cams[j], 0.5, f"novel_{n + 1:02d}", pivot)
            for n, (i, j) in enumerate(neighbour_pairs(cams, k))]


def look_at(name: str, loc, target, fx_px: float, W: int, H: int | None = None, clip_start: float = 0.02) -> Camera:
    loc, target = np.asarray(loc, float), np.asarray(target, float)
    f = target - loc
    f = f / max(np.linalg.norm(f), 1e-12)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([0, 1.0, 0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, -f, loc
    H = H or W
    K = np.array([[fx_px, 0, W / 2], [0, fx_px, H / 2], [0, 0, 1.0]])
    return Camera(name=name, width=W, height=H, K=K, w2c=FLIP @ np.linalg.inv(c2w), clip_start=clip_start)


def path_through(cams: list[Camera], n: int, pivot: np.ndarray | None = None, prefix: str = "move") -> list[Camera]:
    """Eased path through the given cameras (n frames)."""
    nseg = len(cams) - 1
    if nseg < 1:
        raise ValueError("need at least two cameras for a camera move")
    out = []
    for f in range(n):
        u = f / max(n - 1, 1) * nseg
        i = min(int(u), nseg - 1)
        t = 0.5 - 0.5 * np.cos(np.pi * (u - i))
        out.append(interpolate_camera(cams[i], cams[i + 1], float(t), f"{prefix}_{f:04d}", pivot))
    return out


def pan_360(centre, n: int, fx_px: float, W: int, H: int, pitch_deg: float = -5.0) -> list[Camera]:
    out = []
    for k in range(n):
        th = 2 * np.pi * k / n
        d = np.array([np.cos(th), np.sin(th), np.tan(np.radians(pitch_deg))])
        out.append(look_at(f"pan_{k:04d}", centre, np.asarray(centre, float) + d, fx_px, W, H))
    return out
