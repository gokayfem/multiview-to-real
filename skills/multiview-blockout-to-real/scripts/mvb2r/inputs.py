"""What `new` accepts: input kinds, companion files, the JSON scene spec, reference images.

Ported from Blockout 2 Real (b2r/case.py, scripts/blender/blockout.py validate_spec), extended with a list of
cameras in the spec. Standard library only (Blender's Python imports it through blockout.py).

Scene spec (all lengths metres, angles degrees, Blender world axes, Z up) - templates/spec.example.json:
  {"cameras": [{"name", "location", "look_at" | "rotation", "focal_length_mm" | "fov_deg", "role"}],  optional
   "lights": [...], "world": {...}, "room": {...},                                                     optional
   "objects": [{"name", "id", "description", "class", "location", "rotation",
                "parts": [{"type": box|cylinder|cone|sphere|plane, ...}]}]}
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from urllib.parse import unquote

from . import common

MESH_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".usd", ".usda", ".usdc", ".usdz", ".ply", ".stl")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
PART_TYPES = ("box", "cylinder", "cone", "sphere", "plane")
REFERENCE_ROLES = ("identity", "product", "material", "style", "environment", "other")
MAX_REFERENCE_BYTES = 30 * 1024 * 1024
MAX_OBJECTS, MAX_PARTS, MAX_COORD = 2000, 500, 1.0e4      # spec limits (metres / degrees)


def input_kind(path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return "spec"
    if ext == ".blend":
        return "blend"
    if ext in MESH_EXTS:
        return "mesh"
    raise common.StageError(f"unsupported input type {ext!r}: expected a .json scene spec, a .blend, "
                            f"or one of {' '.join(MESH_EXTS)}")


def companions(src: Path) -> list[Path]:
    """Files a mesh format references by relative path (copied next to the input). Never outside its folder."""
    out: list[Path] = []
    ext = src.suffix.lower()
    try:
        if ext == ".gltf":
            g = json.loads(src.read_text())
            for key in ("buffers", "images"):
                for item in g.get(key, []):
                    uri = item.get("uri", "")
                    if uri and not uri.startswith("data:") and "://" not in uri:
                        out.append(src.parent / unquote(uri))
        elif ext == ".obj":
            out.extend(_obj_companions(src))
    except (OSError, ValueError, IndexError):
        return []
    root = src.parent.resolve()
    return [p for p in out if p.is_file() and p.resolve().is_relative_to(root)]


def _obj_companions(src: Path) -> list[Path]:
    out = []
    for line in src.read_text(errors="replace").splitlines():
        if not line.startswith("mtllib"):
            continue
        mtl = src.parent / line.split(None, 1)[1].strip()
        out.append(mtl)
        if mtl.is_file():
            for ml in mtl.read_text(errors="replace").splitlines():
                parts = ml.strip().split()
                if parts and parts[0].lower().startswith(("map_", "bump", "disp", "norm")):
                    out.append(mtl.parent / parts[-1])
    return out


def copy_input(src: Path, dest_dir: Path) -> dict:
    """Copy the input and its companions into dest_dir; return case.json "input"."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = common.safe_name(src.name, "input" + src.suffix.lower())
    dst = dest_dir / name
    shutil.copyfile(src, dst)
    comps = []
    for c in companions(src):
        relp = c.resolve().relative_to(src.parent.resolve())
        target = dest_dir / relp
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c, target)
        comps.append({"path": relp.as_posix(), "sha256": common.sha256_file(c)})
    info = {"path": f"input/{name}", "sha256": common.sha256_file(dst), "kind": input_kind(src),
            "source_dir": str(src.parent.resolve())}
    if comps:
        info["companions"] = comps
    return info


# ---------------------------------------------------------------- reference images
def parse_reference(arg: str) -> dict:
    """`IMG=role[:obj,...]` (preferred) or `IMG:role[:obj,...]` -> {"src", "role", "applies_to"}.
    The role is found from the right, so Windows paths (C:\\x.png:identity) parse correctly."""
    arg = str(arg)
    if "=" in arg:
        src, rest = arg.rsplit("=", 1)
        role, _, applies = rest.partition(":")
    else:
        parts = arg.rsplit(":", 2)
        if len(parts) == 3 and parts[1].strip().lower() in REFERENCE_ROLES:
            src, role, applies = parts
        elif len(parts) >= 2 and arg.rsplit(":", 1)[1].strip().lower() in REFERENCE_ROLES:
            (src, role), applies = arg.rsplit(":", 1), ""
        else:
            raise common.StageError(f"--reference {arg!r}: expected IMG=role[:object,...] "
                                    f"(roles: {', '.join(REFERENCE_ROLES)})")
    role = role.strip().lower()
    if role not in REFERENCE_ROLES or not src:
        raise common.StageError(f"--reference {arg!r}: role {role!r} is not one of {', '.join(REFERENCE_ROLES)}")
    return {"src": src, "role": role, "applies_to": [common.clean_text(a, 60) for a in applies.split(",") if a.strip()]}


def copy_reference(ref: dict, dest_dir: Path, index: int) -> dict:
    src = Path(ref["src"]).expanduser().resolve()
    if not src.is_file():
        raise common.StageError(f"reference image not found: {src}")
    if src.suffix.lower() not in IMAGE_EXTS:
        raise common.StageError(f"reference {src.name}: only {' '.join(IMAGE_EXTS)} images are accepted")
    if src.stat().st_size > MAX_REFERENCE_BYTES:
        raise common.StageError(f"reference {src.name} is larger than {MAX_REFERENCE_BYTES >> 20} MB")
    name = f"ref_{index:02d}_{common.safe_name(src.stem, 'ref')}{src.suffix.lower()}".replace(" ", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest_dir / name)
    common.image_size(dest_dir / name)  # decodes: refuses a file that is not an image
    return {"path": f"input/{name}", "role": ref["role"], "applies_to": ref["applies_to"],
            "note": "", "sha256": common.sha256_file(dest_dir / name)}


# ---------------------------------------------------------------- JSON scene spec
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and abs(x) <= MAX_COORD


def _vec(v, n: int, where: str) -> None:
    if not (isinstance(v, (list, tuple)) and len(v) == n and all(_finite(x) for x in v)):
        raise ValueError(f"{where}: expected {n} finite numbers within +-{MAX_COORD:g}, got {str(v)[:80]}")


def _pos(p: dict, key: str, where: str) -> None:
    v = p.get(key)
    if not (_finite(v) and v > 0):
        raise ValueError(f"{where}.{key}: expected a finite number > 0, got {str(v)[:40]}")


def validate_parts(parts, where: str) -> None:
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"{where}: 'parts' must be a non-empty list")
    if len(parts) > MAX_PARTS:
        raise ValueError(f"{where}: {len(parts)} parts, at most {MAX_PARTS}")
    for i, p in enumerate(parts):
        w = f"{where}.parts[{i}]"
        if not isinstance(p, dict) or p.get("type") not in PART_TYPES:
            raise ValueError(f"{w}: 'type' must be one of {PART_TYPES}")
        for key, n in (("location", 3), ("rotation", 3)):
            if key in p:
                _vec(p[key], n, f"{w}.{key}")
        if p["type"] == "box":
            _vec(p.get("size"), 3, f"{w}.size")
        elif p["type"] == "plane":
            _vec(p.get("size"), 2, f"{w}.size")
        elif p["type"] == "sphere" and "size" in p:
            _vec(p["size"], 3, f"{w}.size")
        elif p["type"] in ("cylinder", "sphere"):
            _pos(p, "radius", w)
        elif p["type"] == "cone":
            for k in ("radius_bottom", "depth"):
                _pos(p, k, w)
            if not (_finite(p.get("radius_top")) and p["radius_top"] >= 0):
                raise ValueError(f"{w}.radius_top: expected a finite number >= 0")
        if p["type"] == "cylinder":
            _pos(p, "depth", w)
        for k in ("segments", "rings"):
            if k in p and not (isinstance(p[k], int) and 3 <= p[k] <= 256):
                raise ValueError(f"{w}.{k}: expected an int in 3..256")


def validate_camera(c, where: str) -> None:
    if not isinstance(c, dict):
        raise ValueError(f"{where}: must be an object")
    _vec(c.get("location"), 3, f"{where}.location")
    if "look_at" in c:
        _vec(c["look_at"], 3, f"{where}.look_at")
    elif "rotation" in c:
        _vec(c["rotation"], 3, f"{where}.rotation")
    else:
        raise ValueError(f"{where}: needs 'look_at' or 'rotation'")
    if "name" in c and not common.is_safe_name(c["name"]):
        raise ValueError(f"{where}.name {c['name']!r} is not a safe name")


def validate_spec(spec) -> None:
    """Raise ValueError naming the first problem; a spec that passes can be built."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    objs = spec.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ValueError("spec needs a non-empty 'objects' list")
    if len(objs) > MAX_OBJECTS:
        raise ValueError(f"spec has {len(objs)} objects, at most {MAX_OBJECTS}")
    names, ids = set(), set()
    for i, o in enumerate(objs):
        w = f"objects[{i}]"
        if not isinstance(o, dict):
            raise ValueError(f"{w}: must be an object")
        name, oid = o.get("name"), o.get("id")
        if not common.is_safe_name(name):
            raise ValueError(f"{w}.name {name!r} is not a safe name (letters, digits, space, _ . -)")
        if not (isinstance(oid, int) and 1 <= oid <= 65535):
            raise ValueError(f"{w}.id must be an int in 1..65535")
        if name in names or oid in ids:
            raise ValueError(f"{w}: duplicate name or id ({name}, {oid})")
        names.add(name)
        ids.add(oid)
        for key in ("location", "rotation"):
            if key in o:
                _vec(o[key], 3, f"{w}.{key}")
        validate_parts(o.get("parts"), w)
    cams = spec.get("cameras", [spec["camera"]] if isinstance(spec.get("camera"), dict) else [])
    if not isinstance(cams, list):
        raise ValueError("'cameras' must be a list")
    for i, c in enumerate(cams):
        validate_camera(c, f"cameras[{i}]")


def load_spec(path: Path) -> dict:
    try:
        spec = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise common.StageError(f"cannot read scene spec {path}: {e}") from e
    try:
        validate_spec(spec)
    except ValueError as e:
        raise common.StageError(f"scene spec {path}: {e}") from e
    return spec


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
