"""Camera move, 360° tour with coverage tint and measured / inferred fractions, relight comparison."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion_io import (encode_mp4, lum, read_gray01, read_json, read_rgb_linear, run_blender, safe_name,
                             write_json, write_png)
from mvb2r.pinhole import cameras_json, look_at, usable_cameras, neighbour_pairs, pan_360, path_through

TINT = np.array([255, 0, 200], np.float32)  # inferred texels: magenta


def _render_frames(case, blend, cams, sub, args, log, passes=False):
    js = case / "out" / "_paths" / f"{sub}.json"
    write_json(js, cameras_json(cams))
    out = case / "out" / sub
    extra = ["--passes"] if passes else []
    run_blender("render.py", ["--cameras", js, "--out", out, "--scale", args.scale, "--samples", args.samples,
                              "--max-bounces", 4] + extra, blend, log)
    return out


def camera_move(case: Path, blend: Path, args, log) -> Path:
    from mvb2r.render import scene_pivot
    cams = usable_cameras(case, "render")
    pairs = neighbour_pairs(cams, 1)
    if not pairs:
        raise ValueError("--move needs at least two source cameras")
    i, j = pairs[0]
    frames = path_through([cams[i], cams[j]], args.frames, scene_pivot(case, cams), "move")
    out = _render_frames(case, blend, frames, "move_frames", args, log)
    _rename_sequence(out, [c.name for c in frames])
    res = _video(out, case / "out" / "camera_move.mp4")
    write_json(case / "out" / "camera_move.json", {"frames": len(frames), "video": res})
    return case / "out" / ("camera_move.mp4" if res.endswith(".mp4") else "move_frames")


def _video(frames: Path, mp4: Path) -> str:
    """Encode or say clearly that the frames were kept; returns what exists."""
    if encode_mp4(frames, "frame_%04d.png", mp4):
        return str(mp4)
    print(f"[render] frames kept in {frames}, no mp4 (ffmpeg not found or failed)")
    return f"frames only: {frames}"


def _rename_sequence(folder: Path, names: list) -> None:
    for k, n in enumerate(names):
        p = folder / f"{n}.png"
        if p.is_file():
            p.replace(folder / f"frame_{k:04d}.png")


def orbit(piv, radius: float, height: float, n: int, fx: float, W: int, H: int):
    """Full circle around the subject at the source cameras' distance and height, looking at it."""
    out = []
    for k in range(n):
        th = 2 * np.pi * k / n
        loc = np.array([piv[0] + radius * np.cos(th), piv[1] + radius * np.sin(th), height])
        out.append(look_at(f"orbit_{k:04d}", loc, piv, fx, W, H))
    return out


def tour_cameras(case: Path, cfg: dict, n: int):
    """(cameras, mode). Room (core detect_room): 360° pan from the room's interior centre. Otherwise an orbit
    around the subject at the median source distance and height (a pan from the subject would sit inside it).
    Both are followed by an arc through the source cameras (the observed region)."""
    from mvb2r.render import room_bounds, scene_pivot
    cams = usable_cameras(case, "render")
    piv = scene_pivot(case, cams)
    c0 = cams[0]
    W, H = c0.width, c0.height
    room = room_bounds(case)
    heights = np.array([c.centre[2] for c in cams])
    if room is not None:
        lo, hi = np.asarray(room[0], float), np.asarray(room[1], float)
        centre = (lo + hi) / 2
        centre[2] = float(np.clip(np.median(heights), lo[2] + 0.3, hi[2] - 0.3))
        first, mode = pan_360(centre, n, c0.fx * 0.6, W, H), "room pan from the interior centre"
    else:
        radius = float(np.median([np.linalg.norm((c.centre - piv)[:2]) for c in cams]))
        first, mode = orbit(piv, radius, float(np.median(heights)), n, c0.fx, W, H), "orbit around the subject"
    order = sorted(range(len(cams)), key=lambda k: np.arctan2(*(cams[k].centre - piv)[:2][::-1]))
    arc = path_through([cams[k] for k in order], n, piv, "arc") if len(cams) > 1 else []
    return first + arc, mode


def coverage_lookup(case: Path, method: str) -> tuple[dict, dict]:
    cov = read_json(case / "proj" / "coverage.json")
    out, openings = {}, {}
    for name, r in cov["objects"].items():
        if r.get("opening"):
            openings[r["id"]] = True
            continue
        out[r["id"]] = read_gray01(case / "proj" / method / safe_name(name) / "coverage.png")[..., None]
    return out, openings


def tint_frame(frame: np.ndarray, passes, covtex: dict) -> tuple[np.ndarray, float]:
    """Magenta tint where the visible texel is inferred; returns (tinted image, measured fraction of object px)."""
    from mvb2r.texlookup import render as lookup_render
    meas, drawn = lookup_render(passes, covtex, (0.0,), nearest=True)
    meas = meas[..., 0] > 0.5
    obj = drawn
    inferred = obj & ~meas
    out = frame.astype(np.float32).copy()
    out[inferred] = 0.45 * out[inferred] + 0.55 * TINT
    frac = float(meas[obj].mean()) if obj.any() else 0.0
    return np.clip(out, 0, 255).astype(np.uint8), frac


def tour_360(case: Path, cfg: dict, blend: Path, args, log) -> dict:
    from mvb2r.texlookup import load_passes
    method = args.method or read_json(case / "proj" / "coverage.json")["methods"][0]
    cams, mode = tour_cameras(case, cfg, args.frames)
    out = _render_frames(case, blend, cams, "tour_frames", args, log, passes=True)
    covtex, _ = coverage_lookup(case, method)
    tinted_dir = case / "out" / "tour_coverage_frames"
    fracs = []
    for k, c in enumerate(cams):
        frame = cv2.imread(str(out / f"{c.name}.png"))[..., ::-1]
        ps = load_passes(out / c.name)
        tinted, f = tint_frame(frame, ps, covtex)
        cv2.putText(tinted, f"measured {f:.0%} / inferred {1 - f:.0%}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        write_png(tinted_dir / f"frame_{k:04d}.png", tinted)
        fracs.append(f)
    _rename_sequence(out, [c.name for c in cams])
    videos = {"tour_360": _video(out, case / "out" / "tour_360.mp4"),
              "tour_360_coverage": _video(tinted_dir, case / "out" / "tour_360_coverage.mp4")}
    rec = {"method": method, "mode": mode, "videos": videos, "frames": len(cams), "measured_fraction_mean": float(np.mean(fracs)),
           "inferred_fraction_mean": float(1 - np.mean(fracs)), "per_frame_measured": fracs,
           "note": "fraction of visible object pixels whose texel was observed by >= 1 camera; the rest is inferred"}
    write_json(case / "out" / "tour.json", rec)
    print(f"[tour] measured {rec['measured_fraction_mean']:.0%} / inferred {rec['inferred_fraction_mean']:.0%}")
    return rec


def relight(case: Path, args, log) -> dict:
    """PBR vs diagnostic under a moved (+70° azimuth, 25° elevation), warm key light, one source view."""
    view = usable_cameras(case, "render")[0].name
    res, imgs = {}, []
    for kind in ("final", "diagnostic"):
        blend = case / "out" / f"{kind}.blend"
        if not blend.is_file():
            continue
        base = case / "out" / "relight" / kind
        common = ["--cameras", case / "views" / "cameras.json", "--only", view, "--scale", args.scale,
                  "--samples", args.samples]
        run_blender("render.py", common + ["--out", base / "orig"], blend, log)
        run_blender("render.py", common + ["--out", base / "moved", "--sun-rotate-z", 70, "--sun-elevation", 25,
                                           "--sun-color", "1,0.7,0.45"], blend, log)
        a, b = read_rgb_linear(base / "orig" / f"{view}.png"), read_rgb_linear(base / "moved" / f"{view}.png")
        res[kind] = {"mean_abs_change_8bit": float(np.abs(a - b).mean() * 255)}
        imgs += [a, b]
    res.update(_old_patch_residue(case, view))
    res["diagnostic_zero_response"] = res.get("diagnostic", {}).get("mean_abs_change_8bit", 0) < 0.5
    if imgs:
        from mvb2r.fusion_io import linear_to_srgb
        row = np.hstack([np.rint(linear_to_srgb(x) * 255).astype(np.uint8) for x in imgs])
        write_png(case / "out" / "relight.jpg", row)
    write_json(case / "out" / "relight.json", res)
    return res


def _old_patch_residue(case: Path, view: str) -> dict:
    """Median PBR base colour (unlit lookup render) inside / outside the ORIGINAL sun patch: 1 = no residue."""
    from mvb2r.fusion_io import read_exr
    from mvb2r.texlookup import load_passes, render as lookup_render
    sun, cov = case / "pbr" / "shading" / f"{view}_sun.exr", case / "pbr" / "albedo" / "coverage.json"
    if not sun.is_file() or not cov.is_file():
        return {"old_patch_residue": None}
    tex = {r["id"]: read_rgb_linear(case / "pbr" / "textures" / safe_name(n) / "base_color.png")
           for n, r in read_json(cov)["objects"].items() if r.get("projected")}
    img, drawn = lookup_render(load_passes(case / "proj" / "passes" / view), tex)
    A = lum(img)
    S = cv2.resize(lum(read_exr(sun)), A.shape[::-1], interpolation=cv2.INTER_LINEAR)
    lit = S > 0.5 * np.percentile(S[drawn], 90)
    if (lit & drawn).sum() < 20 or (~lit & drawn).sum() < 20:
        return {"old_patch_residue": None}
    return {"old_patch_residue": float(np.median(A[lit & drawn]) / max(np.median(A[~lit & drawn]), 1e-6)),
            "old_patch_note": "base colour inside / outside the original sun patch (mixes objects; 1 = no residue)"}
