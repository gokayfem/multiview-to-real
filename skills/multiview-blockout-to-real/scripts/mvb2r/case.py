"""`new` (scaffold a case from any input), `doctor`, and the case.json schema check.

    mvb2r.py new CASE --input X [--look TEXT] [--views N] [--closeup OBJECT] [--reference IMG:role[:obj,...] ...]
                      [--group-by object|collection|parent] [--resolution PX] [--generator fal|mock]
                      [--max-calls N] [--edit-endpoint ID] [--no-cameras]
    mvb2r.py doctor

`new`: copies the input (sha256) and reference images into CASE/input/, writes case.json and acceptance.json,
builds CASE/blockout.blend in Blender (spec -> blockout.py, .blend / mesh -> ingest.py: grey clay material,
one non-overlapping UV map `UVMap` per object, stable pass_index ids, texel density), merges the objects into
case.json with placeholder descriptions, keeps the scene's cameras when there are at least two (and --views
was not given), else places an automatic rig, and writes the coverage table (`cameras`).
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

from . import common, inputs

CLASSES = ("plaster", "paint", "wood", "fabric", "knit", "leather", "skin", "hair", "metal", "plastic", "stone", "generic")
DEFAULT_EDIT = "bytedance/seedream/v5.5/edit"
DEFAULT_RESOLUTION = 1024


def default_case(name: str, input_info: dict, args) -> dict:
    return {
        "name": name, "created": _dt.datetime.now().replace(microsecond=0).isoformat(), "input": input_info,
        "look": common.clean_text(args.look or "", 800), "references": [], "objects": [],
        "views": {"resolution": int(args.resolution), "cameras": [],
                  "auto": {"count": int(args.views or 6), "arc_degrees": 120, "closeup": args.closeup, "elevated": True}},
        "grid": {"rows": None, "cols": None, "source": "auto"},
        "model": {"provider": "fal", "edit_endpoint": args.edit_endpoint, "generator": args.generator},
        "budget": {"max_calls": int(args.max_calls)}, "fusion": {"method": "auto"}, "pbr": {"delight": "conservative"},
    }


def validate_case(cfg: dict) -> list[str]:
    """Problems of a case.json (empty list = valid)."""
    errs = []
    if not isinstance(cfg, dict):
        return ["case.json must be an object"]
    res = (cfg.get("views") or {}).get("resolution")
    sizes = res if isinstance(res, list) else [res]
    if not (len(sizes) in (1, 2) and all(isinstance(s, int) and 64 <= s <= 8192 for s in sizes)):
        errs.append("views.resolution must be an int (square) or [w, h], each 64..8192")
    ids, names = set(), set()
    for o in cfg.get("objects", []):
        if not common.is_safe_name(o.get("name")):
            errs.append(f"object name {str(o.get('name'))[:60]!r} is not safe")
        if o.get("id") in ids or o.get("name") in names:
            errs.append(f"duplicate object {o.get('name')!r} / id {o.get('id')}")
        ids.add(o.get("id"))
        names.add(o.get("name"))
        if o.get("class", "generic") not in CLASSES:
            errs.append(f"object {o.get('name')!r}: class {o.get('class')!r} not in {CLASSES}")
        for flag in ("detail_critical", "refit", "opening"):
            if not isinstance(o.get(flag, False), bool):
                errs.append(f"object {o.get('name')!r}: {flag} must be true/false")
    for r in cfg.get("references", []):
        if r.get("role") not in inputs.REFERENCE_ROLES:
            errs.append(f"reference {r.get('path')!r}: role must be one of {inputs.REFERENCE_ROLES}")
        if ".." in str(r.get("path", "")) or str(r.get("path", "")).startswith("/"):
            errs.append(f"reference path {r.get('path')!r} must be inside the case")
    cams = (cfg.get("views") or {}).get("cameras", [])
    if cams and not common.MIN_VIEWS <= len(cams) <= common.MAX_VIEWS:
        errs.append(f"{len(cams)} cameras; {common.MIN_VIEWS}..{common.MAX_VIEWS} are supported")
    if (cfg.get("model") or {}).get("generator") not in ("fal", "mock"):
        errs.append("model.generator must be 'fal' or 'mock'")
    mc = (cfg.get("budget") or {}).get("max_calls")
    if not (isinstance(mc, int) and mc >= 0):
        errs.append("budget.max_calls must be an int >= 0")
    return errs


# ---------------------------------------------------------------- new
def register(sub) -> None:
    p = sub.add_parser("new", help="scaffold a case from a .blend, mesh file or JSON scene spec")
    p.add_argument("case")
    p.add_argument("--input", required=True)
    p.add_argument("--look", default="")
    p.add_argument("--views", type=int, default=None, help="automatic cameras (2-9); default keeps >= 2 scene cameras, else 6")
    p.add_argument("--closeup", default=None)
    p.add_argument("--reference", action="append", default=[], help="IMG:role[:object,...]")
    p.add_argument("--group-by", default="object", choices=("object", "collection", "parent"))
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="calibration resolution (square px)")
    p.add_argument("--generator", default="fal", choices=("fal", "mock"))
    p.add_argument("--max-calls", type=int, default=30)
    p.add_argument("--edit-endpoint", default=DEFAULT_EDIT)
    p.add_argument("--no-cameras", action="store_true", help="skip camera placement / coverage")
    p.add_argument("--allow-external-images", action="store_true",
                   help="trust the input: pack images and keep linked libraries from OUTSIDE its folder")
    p.set_defaults(func=run_new)
    d = sub.add_parser("doctor", help="Python packages, Blender, FAL_KEY (never printed)")
    d.set_defaults(func=run_doctor)


def _scaffold(args) -> Path:
    src = Path(args.input).expanduser().resolve()
    if not src.is_file():
        raise common.StageError(f"input not found: {src}")
    if args.views is not None and not common.MIN_VIEWS <= args.views <= common.MAX_VIEWS:
        raise common.StageError(f"--views must be {common.MIN_VIEWS}..{common.MAX_VIEWS}")
    if inputs.input_kind(src) == "spec":
        inputs.load_spec(src)
    refs = [inputs.parse_reference(r) for r in args.reference]
    root = common.case_dir(args.case)
    if (root / "case.json").exists():
        raise common.StageError(f"{root} already has a case.json; pick a new folder (cases are never overwritten)")
    for stale in ("input", "views"):
        if (root / "case.failed.json").exists() and (root / stale).is_dir():
            shutil.rmtree(root / stale)          # leftovers of a failed `new` on this folder
    (root / "logs").mkdir(parents=True, exist_ok=True)
    info = inputs.copy_input(src, root / "input")
    cfg = default_case(root.name, info, args)
    cfg["references"] = [inputs.copy_reference(r, root / "input", i + 1) for i, r in enumerate(refs)]
    common.save_case(root, cfg)
    if (common.TEMPLATES_DIR / "acceptance.json").is_file():
        shutil.copyfile(common.TEMPLATES_DIR / "acceptance.json", root / "acceptance.json")
    return root


def merge_scene(cfg: dict, scene: dict, keep_cameras: bool) -> dict:
    """case.json after the Blender build: objects from objects.json, existing cameras when kept (pure)."""
    old = {o["name"]: o for o in cfg.get("objects", [])}
    objs = []
    for row in scene["objects"]:
        prev = old.get(row["name"], {})
        objs.append({"name": row["name"], "id": row["id"], "description": prev.get("description", row.get("description", "")),
                     "class": prev.get("class", row.get("class", "generic")), "group": prev.get("group"),
                     "class_source": prev.get("class_source", row.get("class_source", "default")),
                     "detail_critical": prev.get("detail_critical", False), "prefer_view": prev.get("prefer_view"),
                     "refit": prev.get("refit", False), "opening": prev.get("opening", False)})
    views = dict(cfg["views"])
    found = scene.get("cameras", [])
    if keep_cameras and len(found) >= common.MIN_VIEWS:
        found = found[:common.MAX_VIEWS]
        views["cameras"] = [{"name": common.view_name(i + 1), "camera": c["name"], "role": c.get("role", "scene camera"),
                             "source": "existing"} for i, c in enumerate(found)]
    return {**cfg, "objects": objs, "views": views}


def run_new(args) -> int:
    root = _scaffold(args)
    cfg = common.load_case(root)
    kind = cfg["input"]["kind"]
    script = "blockout.py" if kind == "spec" else "ingest.py"
    log = root / "logs" / "new.log"
    common.say(f"[new] {kind} input -> {cfg['input']['path']} (sha256 {cfg['input']['sha256'][:12]}...)")
    common.say("[new] note: case.json records the input's local folder (input.source_dir); remove it before sharing the case")
    common.say(f"[new] building blockout.blend with {script} (log: logs/new.log) ...")
    extra = ["--allow-external"] if args.allow_external_images else []
    rc = common.run_blender(script, ["--case", str(root), "--group-by", args.group_by, *extra], log)
    for line in common.log_tail(log, 2000).splitlines():
        if line.startswith(("[blockout] NOTE", "[ingest] NOTE", "[blockout] summary", "[ingest] summary")):
            common.say("  " + line)
    if rc != 0:
        common.say(f"[new] {script} FAILED (exit {rc}):\n{common.log_tail(log)}")
        failed = root / "case.failed.json"
        (root / "case.json").replace(failed)
        common.say(f"[new] case.json moved to {failed.name} so `new` can be run again on this folder (logs kept)")
        return rc
    scene = common.read_json(common.require(root / "views" / "scene.json", script))
    cfg = merge_scene(cfg, scene, keep_cameras=args.views is None)
    ext = scene.get("external_images") or {}
    cfg = {**cfg, "input": {**cfg["input"], "external_images": ext}}
    common.save_case(root, cfg)
    report_scene(scene)
    errs = validate_case(cfg)
    if errs:
        common.say("[new] case.json problems:\n  " + "\n  ".join(errs))
    common.say(f"[new] {len(cfg['objects'])} objects; cameras: "
               f"{len(cfg['views']['cameras'])} kept from the scene" if cfg["views"]["cameras"] else
               f"[new] {len(cfg['objects'])} objects; no usable scene cameras -> automatic rig of {cfg['views']['auto']['count']}")
    if not args.no_cameras:
        from . import cameras
        rc = cameras.run(_ns(case=str(root), replace=not cfg["views"]["cameras"], count=None, closeup=None,
                             arc=None, no_elevated=False))
        if rc:
            return rc
    _next_steps(root, cfg)
    return 0


def report_scene(scene: dict) -> None:
    """Print what the build guessed or could not resolve, so nothing changes behaviour silently."""
    ext = scene.get("external_images") or {}
    if ext.get("packed"):
        common.say(f"[new] packed {len(ext['packed'])} external image(s) into blockout.blend: "
                   + ", ".join(p["file"] for p in ext["packed"][:8]))
    for m in ext.get("missing", []):
        common.say(f"[new] WARNING external image {m['image']!r} ({m.get('path')}) is missing; recorded in "
                   f"case.json input.external_images")
    for m in ext.get("refused", []):
        common.say(f"[new] WARNING external image {m['image']!r} ({m.get('path')}) refused: {m.get('why')} "
                   f"(--allow-external-images trusts files outside the input folder)")
    for lib in ext.get("libraries", []):
        common.say(f"[new] WARNING linked library {lib['path']!r}: {lib['action']}")
    guesses = [o for o in scene["objects"] if o.get("class_source") == "guessed from the name"]
    if guesses:
        common.say("[new] class GUESSES from object names (confirm or change in case.json; they set material priors "
                   "and consistency classes): " + ", ".join(f"{o['name']} -> {o['class']}" for o in guesses))


def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def _next_steps(root: Path, cfg: dict) -> None:
    common.say("Next:")
    common.say("  1. look at views/preview_sheet.png and views/coverage.json; read views/objects.json")
    common.say(f"  2. in case.json write \"look\"{' (empty now)' if not cfg['look'] else ''}, each object's description,"
               f" class, and flags detail_critical / refit / opening; reference roles")
    common.say("  3. check acceptance.json before any generation")
    common.say(f"  4. mvb2r.py views {root}   then   mvb2r.py run {root} --to targets")


# ---------------------------------------------------------------- doctor
def run_doctor(_args) -> int:
    ok = True
    for mod, need in (("numpy", True), ("scipy", True), ("cv2", True), ("PIL", True), ("skimage", True),
                      ("fal_client", False)):
        try:
            __import__(mod)
            common.say(f"  ok       {mod}")
        except ImportError:
            common.say(f"  {'MISSING ' if need else 'optional'} {mod}")
            ok &= not need
    b = common.find_blender()
    if b:
        v = common.blender_version(b)
        good = v is not None and v >= common.MIN_BLENDER
        common.say(f"  {'ok      ' if good else 'TOO OLD '} blender {'.'.join(map(str, v)) if v else '?'} "
                   f"(needs {common.MIN_BLENDER[0]}.{common.MIN_BLENDER[1]}+)")
        ok &= good
    else:
        common.say("  MISSING  blender (set $BLENDER): new, cameras and views need it")
        ok = False
    from . import falio
    common.say(f"  {'ok      ' if falio.have_key() else 'missing '} FAL_KEY "
               f"{'(set; not shown)' if falio.have_key() else '(needed for probe --check-size, generate, refine, resync; `targets --images` and --generator mock work without it)'}")
    return 0 if ok else 1
