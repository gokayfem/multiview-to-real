"""Small IO helpers shared by the fusion half (project, score, refit, pbr, render).

Self-contained on purpose: the fusion modules only rely on files in the case folder. When `mvb2r.common`
offers the same helper (case resolution, Blender discovery) it is preferred, so both halves agree.
Colour: images on disk are sRGB; everything in memory is LINEAR float32 (all blending happens in linear light).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

LUMW = np.array([0.2126, 0.7152, 0.0722], np.float32)
BLENDER_DIR = Path(__file__).resolve().parents[1] / "blender"


from mvb2r import common  # noqa: E402  (one source for shared helpers: no drifting copies)
from mvb2r.common import StageError, find_blender, linear_to_srgb, srgb_to_linear, write_json  # noqa: E402,F401


# ----------------------------------------------------------------------------------------------- colour
def lum(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, np.float32) @ LUMW


# ----------------------------------------------------------------------------------------------- images
def _imread(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing image {path}")
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"cannot decode image {path}")
    return a


def read_rgb_linear(path: Path) -> np.ndarray:
    """8/16-bit sRGB PNG/JPG -> float32 linear RGB (H,W,3), row 0 = top."""
    a = _imread(path)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return srgb_to_linear(a[..., :3][..., ::-1].astype(np.float32) / scale)


def read_gray01(path: Path) -> np.ndarray:
    a = _imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return a.astype(np.float32) / scale


def read_ids(path: Path) -> np.ndarray:
    a = _imread(path)
    return (a[..., 0] if a.ndim == 3 else a).astype(np.int32)


def read_exr(path: Path) -> np.ndarray:  # thin alias kept for the fusion modules
    return common.read_exr(path)


def write_srgb_png(path: Path, rgb_lin: np.ndarray, bits: int = 8) -> None:
    mx = 65535 if bits == 16 else 255
    s = np.rint(linear_to_srgb(rgb_lin) * mx).astype(np.uint16 if bits == 16 else np.uint8)
    write_png(path, s)


def write_png(path: Path, arr: np.ndarray) -> None:
    """uint8/uint16 gray or RGB (RGB order) -> PNG / JPG by extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., ::-1]
    if not cv2.imwrite(str(path), np.ascontiguousarray(a)):
        raise IOError(f"cannot write {path}")


def u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, np.float32) * 255), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- json
def read_json(path: Path, producer: str | None = None):
    path = Path(path)
    if not path.is_file():
        hint = f" (run `{producer}` first)" if producer else ""
        raise StageError(f"missing {path}{hint}")
    return common.read_json(path)


def safe_name(name: str) -> str:
    """Object / view name from a file -> a safe path component (mvb2r.common.file_token: raises on unsafe names)."""
    try:
        return common.file_token(name)
    except ValueError as e:
        raise StageError(str(e)) from e


def view_num(view: str) -> str:
    """'view_03' -> '03' (the XX of targets/view_XX.png)."""
    return safe_name(view).split("_")[-1]


# ----------------------------------------------------------------------------------------------- case
def resolve_case(case: str) -> Path:
    """CASE argument -> case folder (must contain case.json)."""
    p = Path(case).expanduser().resolve()
    if not (p / "case.json").is_file():
        raise StageError(f"not a case folder (no case.json): {p}. Create one with `mvb2r.py new`.")
    return p


def case_json(case_dir: Path) -> dict:
    data = read_json(case_dir / "case.json", "mvb2r.py new")
    if not isinstance(data, dict):
        raise StageError(f"{case_dir / 'case.json'} must be a JSON object")
    return data


def load_objects(case_dir: Path) -> list[dict]:
    """views/objects.json rows (name, id, ...) validated: unique names, unique positive integer ids."""
    rows = read_json(case_dir / "views" / "objects.json", "mvb2r.py new")
    rows = rows.get("objects", rows) if isinstance(rows, dict) else rows
    if not isinstance(rows, list) or not rows:
        raise StageError(f"{case_dir / 'views' / 'objects.json'} must be a non-empty list of objects")
    out, seen = [], set()
    for r in rows:
        name, oid = r.get("name"), r.get("id", r.get("pass_index"))
        if not isinstance(name, str) or not isinstance(oid, int) or oid < 1 or oid in seen:
            raise StageError(f"objects.json row needs a name and a unique integer id >= 1: {r!r}")
        seen.add(oid)
        out.append({**r, "name": name, "id": oid})
    return out


def object_settings(case: dict) -> dict:
    """case.json objects -> {name: settings} with defaults filled (never mutates `case`)."""
    out = {}
    for o in case.get("objects", []) or []:
        if not isinstance(o, dict) or "name" not in o:
            raise StageError(f"case.json objects entries need a 'name': {o!r}")
        out[o["name"]] = {"class": str(o.get("class", "generic") or "generic"),
                          "detail_critical": bool(o.get("detail_critical", False)),
                          "prefer_view": o.get("prefer_view"),
                          "refit": bool(o.get("refit", False)), "opening": bool(o.get("opening", False)),
                          "id": o.get("id")}
    return out


# ----------------------------------------------------------------------------------------------- blender
# ----------------------------------------------------------------------------------------------- targets.json
GATE_RELIABILITY = {"high", "medium", "unknown"}  # gate-quality masks; "low" / "prior only" never gate


def targets_meta(case_dir: Path) -> dict:
    """targets/targets.json "views" (optional; {} when absent or unreadable as an object)."""
    p = Path(case_dir) / "targets" / "targets.json"
    if not p.is_file():
        return {}
    data = common.read_json(p)
    views = data.get("views", {}) if isinstance(data, dict) else {}
    return views if isinstance(views, dict) else {}


def skip_reason(view_meta: dict) -> str:
    """Why `targets` marked a view unusable: the gate's own reasons first, then any plain reason field."""
    gate = view_meta.get("gate") if isinstance(view_meta.get("gate"), dict) else {}
    reasons = gate.get("reasons") if isinstance(gate.get("reasons"), list) else []
    text = "; ".join(str(r) for r in reasons) or view_meta.get("reason") or "marked unusable in targets.json"
    return str(text)[:300]


def split_usable(case_dir: Path, cams: list, min_views: int = 2, stage: str = "this stage") -> tuple[list, list]:
    """(usable cameras, [{"view", "reason"}]) from targets.json views.<v>.usable (missing key = usable).
    Prints the skipped views; refuses when fewer than `min_views` remain."""
    meta = targets_meta(case_dir)
    keep, skipped = [], []
    for c in cams:
        m = meta.get(c.name, {}) if isinstance(meta.get(c.name), dict) else {}
        if m.get("usable", True) is False:
            skipped.append({"view": c.name, "reason": skip_reason(m)})
        else:
            keep.append(c)
    for sk in skipped:
        print(f"[{stage}] skipping {sk['view']}: {sk['reason']}")
    if len(keep) < min_views:
        raise StageError(f"only {len(keep)} usable view(s) in targets/targets.json ({[s['view'] for s in skipped]} "
                         f"rejected); {stage} needs at least {min_views}: regenerate or refine the rejected views")
    return keep, skipped


def mask_reliability(meta: dict, view: str, obj: str) -> str:
    """Reliability of targets/mask_<obj>_<XX>.png as recorded by `targets` ("unknown" when not recorded)."""
    m = (meta.get(view) or {}).get("masks", {}) if isinstance(meta.get(view), dict) else {}
    r = m.get(obj, {}) if isinstance(m, dict) else {}
    return str(r.get("reliability", "unknown")) if isinstance(r, dict) else "unknown"


def run_blender(script: str, args: list, blend: Path | None = None, log: Path | None = None,
                timeout: int = 3 * 3600) -> str:
    """blender/<script> with the core half's command line (mvb2r.common.blender_command: background,
    --factory-startup, --disable-autoexec, argument list, no shell) plus a timeout. `blend` is passed as
    --blend (the script opens it with scripts disabled). Output is appended to `log`; failure -> StageError."""
    argv = (["--blend", str(blend)] if blend else []) + [str(x) for x in args]
    cmd = common.blender_command(script, argv)
    assert "--factory-startup" in cmd and "--disable-autoexec" in cmd  # never start Blender with auto-run
    env = dict(os.environ, MVB2R_PYTHON=common.python_exe(),
               PYTHONPATH=str(common.SCRIPTS_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise StageError(f"Blender script {script} did not finish within {timeout} s") from e
    text = (r.stdout or "")[-8000:] + "\n" + (r.stderr or "")[-3000:]
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(f"$ {script} {' '.join(argv)}\n{text}\n")
    if r.returncode != 0:
        raise StageError(f"Blender script {script} failed (exit {r.returncode}):\n{text[-2500:]}")
    return text


def mesh_path(mesh_dir: Path, name: str) -> Path:
    """A mesh file named in meshes.json -> its path, only if it is a plain mesh_NNN.npz inside mesh_dir."""
    import re
    if not isinstance(name, str) or not re.fullmatch(r"mesh_\d{1,6}\.npz", name):
        raise StageError(f"unexpected mesh file name {str(name)[:80]!r} in {mesh_dir / 'meshes.json'}")
    p = (Path(mesh_dir) / name).resolve()
    if p.parent != Path(mesh_dir).resolve() or not p.is_file():
        raise StageError(f"mesh file {p} missing or outside {mesh_dir} (re-run `mvb2r.py project`)")
    return p


def load_mesh(mesh_dir: Path, name: str) -> dict:
    with np.load(mesh_path(mesh_dir, name), allow_pickle=False) as z:
        return {k: z[k] for k in ("tri_pos", "tri_uv", "tri_nrm")}


def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def encode_mp4(frames_glob_dir: Path, pattern: str, out: Path, fps: int = 24) -> bool:
    """frames -> H.264 yuv420p mp4 when ffmpeg exists; returns False (frames kept) otherwise."""
    exe = ffmpeg()
    if exe is None:
        return False
    if "/" in pattern or pattern.count("%") != 1:
        raise ValueError(f"frame pattern must be a plain file name with one %d placeholder: {pattern!r}")
    # cwd = frames folder + relative pattern: a '%' in the case path is never read as a placeholder
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(Path(out).resolve())]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(frames_glob_dir), timeout=1800)
    except subprocess.TimeoutExpired:
        print("[mvb2r] ffmpeg timed out after 30 min, frames kept", file=sys.stderr)
        return False
    if r.returncode != 0:
        print(f"[mvb2r] ffmpeg failed, frames kept: {r.stderr[-500:]}", file=sys.stderr)
        return False
    return True
