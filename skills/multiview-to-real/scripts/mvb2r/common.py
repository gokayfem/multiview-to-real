"""Shared contract of every stage: case paths, JSON IO, names from untrusted files, Blender runner, image IO.

Standard library only at import time: Blender's bundled Python imports this module too (scripts/blender/*.py).
numpy / OpenCV are imported lazily inside the image helpers.

A case is a folder (see templates/CONTRACT.md):

    CASE/case.json                  the only scene-specific file
    CASE/input/                     the input copied with its sha256
    CASE/blockout.blend             grey copy, one UV map `UVMap` per object, stable pass_index ids
    CASE/views/                     cameras.json, objects.json, view_XX/{clay.png, depth_z.exr, ...}, verify/
    CASE/grid/  CASE/gen/  CASE/targets/  CASE/proj/  CASE/pbr/  CASE/out/

Conventions: world = Blender, Z up, metres. Camera = OpenCV (+X right, +Y down, +Z forward),
w2c_opencv = diag(1,-1,-1,1) @ inv(c2w_blender), u = fx*X/Z + cx, v = fy*Y/Z + cy, origin at the top-left
corner of the top-left pixel, pixel centres at +0.5. depth_z = camera-axis Z in metres (never ray length).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import unicodedata
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; numpy is imported lazily where it is used
    import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent          # .../skills/multiview-to-real/scripts
PACKAGE_DIR = Path(__file__).resolve().parent
BLENDER_DIR = SCRIPTS_DIR / "blender"
TEMPLATES_DIR = SCRIPTS_DIR.parent / "templates"
NAME_MAX = 60
MIN_VIEWS, MAX_VIEWS = 2, 9
BG_DEPTH = 1.0e10
MAX_IMAGE_PIXELS = 64_000_000          # any image read from a model, a user or an input file
BLENDER_TIMEOUT_S = int(os.environ.get("MVB2R_BLENDER_TIMEOUT", "7200"))


class StageError(RuntimeError):
    """A stage cannot run. The message names the missing input and the command that produces it."""


def say(*parts) -> None:
    print(*parts, flush=True)


# ---------------------------------------------------------------- tools on this machine
def find_blender() -> str | None:
    """$BLENDER, then `blender` on PATH, then the usual macOS / Linux locations."""
    cands = [os.environ.get("BLENDER"), shutil.which("blender"),
             "/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    return None


def python_exe() -> str:
    """The user's Python (numpy, scipy, opencv). Under Blender the CLI passes it in $MVB2R_PYTHON."""
    env = os.environ.get("MVB2R_PYTHON")
    if env:
        return env
    if "blender" in Path(sys.executable).name.lower():
        return shutil.which("python3") or "python3"
    return sys.executable


MIN_BLENDER = (4, 2)
_BLENDER_VERSION: dict = {}


def parse_blender_version(text: str) -> tuple[int, int] | None:
    m = re.search(r"Blender (\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def blender_version(blender: str) -> tuple[int, int] | None:
    if blender not in _BLENDER_VERSION:
        try:
            out = subprocess.run([blender, "-b", "--factory-startup", "--disable-autoexec", "--version"],
                                 capture_output=True, text=True, timeout=120).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        _BLENDER_VERSION[blender] = parse_blender_version(out)
    return _BLENDER_VERSION[blender]


def check_blender_version(blender: str) -> None:
    v = blender_version(blender)
    need = f"{MIN_BLENDER[0]}.{MIN_BLENDER[1]}+"
    if v is None:
        raise StageError(f"cannot read the version of {blender}; Blender {need} is required")
    if v < MIN_BLENDER:
        raise StageError(f"Blender {v[0]}.{v[1]} found; {need} is required (developed on 5.0)")


def blender_command(script: str, args: list[str]) -> list[str]:
    """Blender command line: background, factory startup, auto-run of scripts in files disabled."""
    blender = find_blender()
    if not blender:
        raise StageError("Blender not found: set $BLENDER or put `blender` on PATH (mvb2r.py doctor)")
    check_blender_version(blender)
    path = BLENDER_DIR / script
    if not path.is_file():
        raise StageError(f"missing Blender script {path}")
    return [blender, "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
            "--python", str(path), "--", *args]


def run_blender(script: str, args: list[str], log: Path) -> int:
    """Run a Blender script headless; stdout+stderr go to `log`. Returns the exit code."""
    env = dict(os.environ)
    env["MVB2R_PYTHON"] = python_exe()
    env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        try:
            return subprocess.run(blender_command(script, args), env=env, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=BLENDER_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[mvb2r] Blender stopped after {BLENDER_TIMEOUT_S} s ($MVB2R_BLENDER_TIMEOUT)\n")
            return 124


def log_tail(log: Path, n: int = 30) -> str:
    try:
        return "\n".join(Path(log).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def script_args(argv: list[str] | None = None) -> list[str]:
    """Arguments of a Blender script: everything after `--`."""
    argv = list(sys.argv if argv is None else argv)
    return argv[argv.index("--") + 1:] if "--" in argv else argv[1:]


# ---------------------------------------------------------------- case folders
def case_dir(case) -> Path:
    return Path(case).expanduser().resolve()


def case_json(case) -> Path:
    return case_dir(case) / "case.json"


def load_case(case) -> dict:
    p = case_json(case)
    if not p.is_file():
        raise StageError(f"not a case folder (no case.json): {case_dir(case)}. Create one with `mvb2r.py new`.")
    data = read_json(p)
    if not isinstance(data, dict):
        raise StageError(f"{p} is not a JSON object")
    return data


def save_case(case, cfg: dict) -> None:
    write_json(case_json(case), cfg)


def views_dir(case) -> Path:
    return case_dir(case) / "views"


def view_name(index: int) -> str:
    """1-based view index -> view_XX."""
    if not isinstance(index, int) or index < 1:
        raise ValueError(f"view index must be an int >= 1, got {index!r}")
    return f"view_{index:02d}"


def view_index(name: str) -> int:
    m = re.fullmatch(r"view_(\d{2})", str(name))
    if not m:
        raise StageError(f"{str(name)[:40]!r} is not a view name: views are named view_01 .. view_09 "
                         "(fix views/cameras.json / case.json views.cameras)")
    return int(m.group(1))


def inside_case(case, path) -> Path:
    """`path` resolved (symlinks followed); refused unless it lies inside the case folder."""
    root = case_dir(case)
    p = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.is_relative_to(root):
        raise StageError(f"{str(path)[:120]!r} is outside the case folder {root.name}/: only files inside the case are "
                         "read or uploaded (`new` copies references into input/)")
    return p


def owned_path(case, path) -> Path:
    """`path` (relative to the case, or absolute inside it) as something this run may create, empty, delete or
    write: refused when it leaves the case, or when it or any folder between it and the case root is a symbolic
    link. Links are NOT followed: a case folder from someone else can plant one (out/sheets/critique_in ->
    a folder elsewhere), and clearing or writing through it would destroy or overwrite files outside the case.
    Use it before every rmtree, unlink and write into a folder the stage empties; `inside_case` is for reads."""
    root = case_dir(case)
    raw = Path(path)
    outside = StageError(f"{str(path)[:120]!r} is outside the case folder {root.name}/: nothing outside the case "
                         "is deleted or written")
    if raw.is_absolute():
        absolute = Path(os.path.abspath(raw))
        bases = [b for b in (root, Path(os.path.abspath(Path(case).expanduser()))) if absolute.is_relative_to(b)]
        if not bases:
            raise outside
        raw = absolute.relative_to(bases[0])
    parts = Path(os.path.normpath(raw)).parts
    if ".." in parts:
        raise outside
    cur = root
    for part in parts:
        cur = cur / part
        if cur.is_symlink():
            raise StageError(f"{clean_text(cur.relative_to(root), 120)} in the case folder is a symbolic link; "
                             "refusing to delete or write through it (replace it with a real folder)")
    return cur


def require(path: Path, producer: str) -> Path:
    """Return path, or fail naming the file and the command that writes it."""
    path = Path(path)
    if not path.exists():
        raise StageError(f"missing {path} - run `{producer}` first")
    return path


def load_cameras(case) -> list[dict]:
    p = require(views_dir(case) / "cameras.json", f"mvb2r.py views {case}")
    views = read_json(p).get("views")
    if not isinstance(views, list) or not views:
        raise StageError(f"{p} has no views")
    return views


def load_objects(case) -> list[dict]:
    p = require(views_dir(case) / "objects.json", f"mvb2r.py new {case}")
    objs = read_json(p)
    if not isinstance(objs, list):
        raise StageError(f"{p} must be a list of objects")
    return objs


def case_objects(case) -> list[dict]:
    """Objects as the case describes them: views/objects.json rows overlaid with case.json "objects" (by name)."""
    cfg = load_case(case)
    over = {o.get("name"): o for o in cfg.get("objects", []) if isinstance(o, dict)}
    out = []
    for row in load_objects(case):
        merged = {**row, **over.get(row.get("name"), {})}
        out.append(merged)
    return out


# ---------------------------------------------------------------- JSON / hashes
def write_json(path, data) -> None:
    """Atomic JSON write. Neither the file nor its temporary sibling may be a symbolic link: a case folder from
    someone else can plant one, and the write would land outside the case."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for p in (path, tmp):
        if p.is_symlink():
            raise StageError(f"{clean_text(p.name, 80)} is a symbolic link; refusing to write through it "
                             "(replace it with a real file)")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(tmp, flags, 0o644), "w") as f:
        json.dump(data, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def write_text_in_case(case, path, text: str) -> Path:
    """Write a text file this stage owns (out/report.md); refused through a planted link (owned_path)."""
    out = owned_path(case, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise StageError(f"{path} is not valid JSON: {e}") from e


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(case, path) -> str:
    """Path relative to the case folder (as stored in JSON), POSIX style."""
    try:
        return Path(path).resolve().relative_to(case_dir(case)).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- names and text from input files
def safe_name(name, fallback: str = "object") -> str:
    """Letters, digits, space, _ . - only; no separators, no "..", no control characters, <= NAME_MAX."""
    s = unicodedata.normalize("NFC", str(name))
    s = re.sub(r"[^\w .\-]", "_", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    s = s[:NAME_MAX].rstrip(" .")
    return s or fallback


def is_safe_name(name) -> bool:
    return isinstance(name, str) and bool(name) and name == safe_name(name, fallback="")


def check_name(name, what: str = "object") -> str:
    """Return `name`, or raise: every place that builds a file path from a name calls this first."""
    if not is_safe_name(name):
        raise ValueError(f"{what} name {str(name)[:80]!r} cannot be used in file names "
                         f"(allowed: letters, digits, space, _ . -; suggested: {safe_name(name)!r})")
    return name


def file_token(name) -> str:
    """A safe name further reduced for file names (spaces -> _)."""
    return check_name(name).replace(" ", "_")


def clean_text(text, limit: int = 200) -> str:
    """Free text from an input file as one printable line, capped. A label, never an instruction."""
    s = "".join(ch if ch.isprintable() else " " for ch in str(text))
    return re.sub(r"\s+", " ", s).strip()[:limit]


# ---------------------------------------------------------------- images (lazy numpy / cv2)
def _cv2():
    """OpenCV for PNG / JPEG and image maths only; EXR goes through mvb2r.exrio on every platform."""
    import cv2
    return cv2


def check_pixels(path, cap: int | None = None) -> tuple[int, int]:
    """(w, h) from the file header; refuses images above the pixel cap before anything decodes them."""
    w, h = image_size(path)
    cap = cap or MAX_IMAGE_PIXELS
    if w * h > cap:
        raise StageError(f"{Path(path).name} is {w}x{h} = {w * h} px, above the {cap} px limit")
    return w, h


def read_rgb(path) -> "np.ndarray":
    """8-bit sRGB image -> uint8 (H, W, 3) RGB, row 0 = top (pixel count checked from the header first)."""
    cv2 = _cv2()
    if Path(path).suffix.lower() != ".exr":
        check_pixels(path)
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if a is None:
        raise StageError(f"cannot read image {path}")
    return a[..., ::-1].copy()


def write_rgb(path, rgb) -> None:
    cv2 = _cv2()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgb[..., ::-1]):
        raise StageError(f"cannot write {path}")


def write_gray(path, img) -> None:
    cv2 = _cv2()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise StageError(f"cannot write {path}")


def read_gray(path) -> "np.ndarray":
    cv2 = _cv2()
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise StageError(f"cannot read image {path}")
    return a[..., 0] if a.ndim == 3 else a


def read_exr(path) -> "np.ndarray":
    """EXR -> float32 via the built-in reader (mvb2r.exrio, no codec needed): a single channel (or "Y") -> (H, W);
    R, G, B -> (H, W, 3) in RGB order. Covers the dataWindow."""
    import numpy as np
    from . import exrio
    try:
        planes, _info = exrio.read_channels(path)
    except FileNotFoundError as e:
        raise StageError(f"missing EXR {path}") from e
    except (exrio.ExrError, struct.error, zlib.error, IndexError) as e:
        raise StageError(f"cannot read EXR {Path(path).name}: {e}") from e
    if {"R", "G", "B"} <= set(planes):
        return np.dstack([planes[c] for c in ("R", "G", "B")])
    if "Y" in planes:
        return planes["Y"]
    if len(planes) == 1:
        return next(iter(planes.values()))
    raise StageError(f"{Path(path).name}: channels {sorted(planes)} are neither Y nor R, G, B")


def write_exr(path, arr, compression: str = "ZIP") -> None:
    """float32 (H, W) -> channel Y, (H, W, 3) RGB -> channels R, G, B; ZIP (default) or NONE."""
    import numpy as np
    from . import exrio
    a = np.asarray(arr, np.float32)
    if a.ndim == 2:
        planes = {"Y": a}
    elif a.ndim == 3 and a.shape[2] == 3:
        planes = {"R": a[..., 0], "G": a[..., 1], "B": a[..., 2]}
    else:
        raise StageError(f"write_exr: expected (H, W) or (H, W, 3), got {a.shape}")
    exrio.write(path, planes, compression)


def image_size(path) -> tuple[int, int]:
    """(width, height) read from the file itself (never from an API response)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Image.DecompressionBombError as e:
        raise StageError(f"{Path(path).name}: {e}") from e


def resize_rgb(rgb, size: tuple[int, int]):
    """Resample to (w, h): area-average when shrinking, Lanczos when enlarging. Returns (image, method)."""
    cv2 = _cv2()
    h, w = rgb.shape[:2]
    if (w, h) == tuple(size):
        return rgb.copy(), "none (native)"
    shrink = size[0] <= w and size[1] <= h
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4
    out = cv2.resize(rgb, tuple(size), interpolation=interp)
    return out, ("area downsample" if shrink
                 else "Lanczos upsample - resized, NOT generated detail")


def srgb_to_linear(x):
    import numpy as np
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    import numpy as np
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


# ---------------------------------------------------------------- view passes
def view_passes(case, cam: dict) -> dict:
    """object_id, depth_z, position_world (and the camera) of one view, validated against its size."""
    vd = views_dir(case) / check_name(cam["name"], "view")
    producer = f"mvb2r.py views {case}"
    oid = read_gray(require(vd / "object_id.png", producer)).astype("int32")
    dz = read_exr(require(vd / "depth_z.exr", producer))
    pw = read_exr(require(vd / "position_world.exr", producer))
    shape = (int(cam["height"]), int(cam["width"]))
    for nm, a in (("object_id", oid), ("depth_z", dz), ("position_world", pw)):
        if a.shape[:2] != shape:
            raise StageError(f"{vd}/{nm} is {a.shape[:2]}, cameras.json says {shape}")
    return {"oid": oid, "depth": dz, "pos": pw, "cam": cam}
