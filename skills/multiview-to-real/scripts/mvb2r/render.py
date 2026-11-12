"""Stage `render` and the render half of `project` (passes, consensus re-renders, novel views, diagnostic.blend).

    mvb2r.py render CASE [--blend final|diagnostic] [--move] [--tour] [--relight] [--scale 0.5] [--samples 32]

Default: source views -> out/renders/, novel views -> out/novel/ (from final.blend when it exists, else the
diagnostic). --move: eased arc between neighbouring source cameras -> out/camera_move.mp4. --tour: 360° pan
from the room's interior centre when the scene is a room, otherwise an orbit around the subject, + an arc
through the observed region (mode in out/tour.json) -> out/tour_360.mp4 and the coverage-tinted
out/tour_360_coverage.mp4 with the measured / inferred fraction of every frame (out/tour.json) — the inferred
share is shown, never hidden. --relight: PBR vs diagnostic under a moved, recoloured key light -> out/relight.jpg;
the diagnostic must not respond (emission); old-patch residue reported. MP4 = H.264 yuv420p when ffmpeg exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mvb2r.fusion_io import (StageError, case_json, read_json, read_rgb_linear, resolve_case, run_blender, safe_name,
                             write_json, write_srgb_png)
from mvb2r.pinhole import cameras_json, load_cameras, novel_cameras, usable_cameras
from mvb2r.texlookup import load_passes, render as lookup_render

DEFAULT_SKY = (0.8, 0.84, 0.9)


def register(subparsers) -> None:
    p = subparsers.add_parser("render", help="source / novel views, camera move, 360 tour, relight comparison")
    p.add_argument("case")
    p.add_argument("--blend", choices=["final", "diagnostic"], default=None)
    p.add_argument("--move", action="store_true")
    p.add_argument("--tour", action="store_true")
    p.add_argument("--relight", action="store_true")
    p.add_argument("--scale", type=float, default=0.5, help="render scale of the calibration size")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--frames", type=int, default=48, help="frames of the move / each tour part")
    p.add_argument("--method", default=None, help="projected method for coverage tint (default: first)")
    p.set_defaults(func=run)


def run(args) -> int:
    case = resolve_case(args.case)
    cfg = case_json(case)
    blend = pick_blend(case, args.blend)
    log = case / "out" / "logs" / "render.log"
    cams_js = case / "out" / "_paths" / "sources.json"
    write_json(cams_js, cameras_json(usable_cameras(case, "render")))
    nov_js = ensure_novel(case, cfg)
    common = ["--scale", args.scale, "--samples", args.samples]
    run_blender("render.py", ["--cameras", cams_js, "--out", case / "out" / "renders"] + common, blend, log)
    run_blender("render.py", ["--cameras", nov_js, "--out", case / "out" / "novel"] + common, blend, log)
    from mvb2r import tour
    if args.move:
        tour.camera_move(case, blend, args, log)
    if args.tour:
        tour.tour_360(case, cfg, blend, args, log)
    if args.relight:
        tour.relight(case, args, log)
    print(f"[render] {blend.name} -> {case / 'out'}")
    return 0


def pick_blend(case: Path, which: str | None) -> Path:
    final, diag = case / "out" / "final.blend", case / "out" / "diagnostic.blend"
    if which == "final" or (which is None and final.is_file()):
        if not final.is_file():
            raise StageError(f"missing {final} (run `mvb2r.py pbr {case.name}` first)")
        return final
    if not diag.is_file():
        raise StageError(f"missing {diag} (run `mvb2r.py project {case.name}` first)")
    return diag


def scene_pivot(case: Path, cams) -> np.ndarray:
    """Median of the surface points seen at the centre of each source camera."""
    from mvb2r.fusion_io import read_exr
    pts = []
    for c in cams:
        p = case / "views" / c.name / "depth_z.exr"
        if not p.is_file():
            continue
        d = read_exr(p)
        d = d[..., 0] if d.ndim == 3 else d
        z = float(d[c.height // 2, c.width // 2])
        if z < 1e8:
            pts.append(c.centre + c.forward * z)
    return np.median(np.array(pts), 0) if pts else np.mean([c.centre for c in cams], 0)


def room_bounds(case: Path):
    """(lo, hi) interior of an enclosing room (the core half's detect_room on views/objects.json), else None."""
    try:
        from mvb2r.cameras import detect_room
        rows = read_json(case / "views" / "objects.json")
        return detect_room(rows if isinstance(rows, list) else rows.get("objects", []))
    except (ImportError, StageError, KeyError, TypeError, ValueError):
        return None


def ensure_novel(case: Path, cfg: dict, regenerate: bool = False) -> Path:
    """Novel cameras on arcs between neighbouring sources; reused when `project` already wrote and validated
    them. In a room, a camera outside the room interior is skipped with a note."""
    js = case / "proj" / "novel" / "cameras.json"
    if js.is_file() and not regenerate:
        return js
    cams = usable_cameras(case, "project")
    k = int((cfg.get("fusion", {}) or {}).get("novel", 4))
    nov, skipped, room = novel_cameras(cams, k, pivot=scene_pivot(case, cams)), [], room_bounds(case)
    if room is not None:
        lo, hi = np.asarray(room[0]), np.asarray(room[1])
        keep = [c for c in nov if np.all(c.centre > lo) and np.all(c.centre < hi)]
        skipped = [{"name": c.name, "reason": "outside the room interior"} for c in nov if c not in keep]
        nov = keep
    write_json(js, cameras_json(nov, {"rule": "midpoints (arc around the scene centre) of neighbouring sources",
                                      "skipped": skipped}))
    return js


def validate_novel(case: Path, passes_dir: Path, near_frac: float = 0.1, max_bad: float = 0.25) -> list:
    """Drop novel cameras that sit inside / against geometry: more than `max_bad` of their pixels closer than
    `near_frac` x the median depth seen by the source cameras (exact passes). Rewrites cameras.json."""
    import shutil
    from mvb2r.fusion_io import read_exr
    js = case / "proj" / "novel" / "cameras.json"
    data = read_json(js)
    src = [read_exr(case / "views" / c.name / "depth_z.exr") for c in usable_cameras(case, "project")]
    med = float(np.median(np.concatenate([d[d < 1e8].ravel() for d in src])))
    keep, skipped = [], list(data.get("skipped", []))
    for v in data["views"]:
        d = load_passes(passes_dir / v["name"]).depth
        bad = float(np.mean(d < near_frac * med))
        if bad > max_bad:
            skipped.append({"name": v["name"], "reason": f"{bad:.0%} of pixels closer than {near_frac} x median "
                                                         "scene depth: inside or against geometry"})
            shutil.rmtree(passes_dir / v["name"], ignore_errors=True)
        else:
            keep.append(v)
    write_json(js, {**data, "views": keep, "skipped": skipped})
    for sk in skipped:
        print(f"[project] novel camera {sk['name']} skipped: {sk['reason']}")
    return skipped


# ----------------------------------------------------------------------------------------------- after project
def after_project(case: Path, cfg: dict, out: Path, methods: list[str]) -> None:
    log = case / "proj" / "logs" / "render.log"
    nov_js = ensure_novel(case, cfg, regenerate=True)
    blend = case / "blockout.blend"
    run_blender("render.py", ["--cameras", case / "views" / "cameras.json", "--out", out / "passes", "--passes",
                              "--no-beauty"], blend, log)
    run_blender("render.py", ["--cameras", nov_js, "--out", out / "passes", "--passes", "--no-beauty"], blend, log)
    validate_novel(case, out / "passes")
    cov = read_json(out / "coverage.json")
    sky = tuple((cfg.get("fusion", {}) or {}).get("sky", DEFAULT_SKY))
    for m in methods:
        rerender_method(case, out, m, cov, sky)
    build_diagnostic(case, cfg, out, methods, cov, nov_js, sky, log)


def method_textures(out: Path, method: str, cov: dict) -> tuple[dict, dict, dict]:
    """Textures, flat (opening) colours and atlas masks (texels with a triangle, incl. bleed) per object id."""
    from mvb2r.fusion_io import read_ids
    tex, flat, valid = {}, {}, {}
    for name, r in cov["objects"].items():
        if r.get("opening"):
            flat[r["id"]] = None
            continue
        d = out / method / safe_name(name)
        tex[r["id"]] = read_rgb_linear(d / "texture.png")
        valid[r["id"]] = read_ids(d / "source_raw.png") != 255
    return tex, flat, valid


def rerender_method(case: Path, out: Path, method: str, cov: dict, sky) -> None:
    tex, flat, valid = method_textures(out, method, cov)
    flat = {k: sky for k in flat}
    for cams, sub in ((usable_cameras(case, "project"), "rerender"),
                      (load_cameras(case / "proj" / "novel" / "cameras.json"), "novel")):
        for c in cams:
            img, _ = lookup_render(load_passes(out / "passes" / c.name), tex, (0, 0, 0), flat, valid=valid)
            write_srgb_png(out / method / sub / f"{c.name}.png", img)


def diagnostic_readme(cov: dict, methods: list[str]) -> str:
    lines = ["DIAGNOSTIC ONLY - APPEARANCE TRANSFER, NOT A MATERIAL MODEL",
             "Textures are the generated views projected into each object's UVMap. They CONTAIN BAKED LIGHTING and",
             "shadows from the generated images; materials are Emission (shadeless). Do not relight this file.",
             "Switch method: node group 'AppearanceSwitch' -> Value node 'METHOD': "
             + ", ".join(f"{i} = {m}" for i, m in enumerate(methods)),
             "Openings are NOT projected: they show a flat neutral sky (as does the world).",
             "Where the scene has no geometry (an open scene's background) nothing is projected: the backdrop the",
             "image model painted (studio wall, sky) is not part of the scene; renders show the flat world colour.",
             "Unobserved texels are INFERRED (same-object fill, plain colour). Per object (area fractions):"]
    for n, r in cov["objects"].items():
        if r.get("opening"):
            lines.append(f"  {n}: opening (not projected)")
        else:
            c = r["coverage"]
            lines.append(f"  {n}: observed {c['observed']:.2f}, soft {c['soft']:.2f}, inferred {c['inferred']:.2f}")
    lines.append("This is not a 360-degree reconstruction: only what the cameras saw is measured.")
    return "\n".join(lines) + "\n"


def build_diagnostic(case, cfg, out, methods, cov, nov_js, sky, log) -> Path:
    names = {n: safe_name(n) for n, r in cov["objects"].items() if not r.get("opening")}
    write_json(out / "_diag_names.json", names)
    readme = out / "_diag_readme.txt"
    readme.write_text(diagnostic_readme(cov, methods))
    openings = ",".join(n for n, r in cov["objects"].items() if r.get("opening"))
    target = case / "out" / "diagnostic.blend"
    run_blender("build_diagnostic.py", ["--sets"] + [f"{m}={out / m}" for m in methods] +
                ["--default", methods[0], "--names", out / "_diag_names.json", "--out", target,
                 "--cameras", case / "views" / "cameras.json", nov_js, "--readme", readme,
                 "--openings", openings, "--sky", ",".join(str(float(x)) for x in sky)],
                case / "blockout.blend", log)
    return target
