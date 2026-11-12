"""`views`: clay beauty + exact ray-cast passes per camera, cameras.json, numeric verification.

    mvb2r.py views CASE [--samples N] [--resolution PX] [--verify-only]

Blender (scripts/blender/passes.py) renders per view, at case.json views.resolution:
  view_XX/clay.png           8-bit sRGB Cycles beauty of the grey clay scene (--samples, default 32, denoised)
  view_XX/object_id.png      pass_index per pixel, un-antialiased (8-bit, 16-bit when an id > 255)
  view_XX/depth_z.exr        camera-axis Z in metres (NOT ray length), background 1e10, channel "Y"
  view_XX/position_world.exr world XYZ of the first hit (0 on background)
  view_XX/normal_world.exr   world unit shading normal (0 on background)
from one BVH ray cast through every pixel centre (exact, no anti-aliasing), plus cameras.json (K,
c2w_blender, w2c_opencv, P, lens/sensor/shift/clip, fov). Then verify.py proves the conventions numerically.
"""
from __future__ import annotations

from . import common, verify


def register(sub) -> None:
    p = sub.add_parser("views", help="clay beauty + exact ray-cast passes, cameras.json, numeric verification")
    p.add_argument("case")
    p.add_argument("--samples", type=int, default=32, help="Cycles samples of the clay beauty (denoised; it is the conditioning image)")
    p.add_argument("--resolution", type=int, default=None, help="override case.json views.resolution (square)")
    p.add_argument("--verify-only", action="store_true")
    p.set_defaults(func=run)


def run(args) -> int:
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    if args.resolution:
        cfg = {**cfg, "views": {**cfg["views"], "resolution": int(args.resolution)}}
        common.save_case(case, cfg)
    cams = (cfg.get("views") or {}).get("cameras") or []
    if not common.MIN_VIEWS <= len(cams) <= common.MAX_VIEWS:
        raise common.StageError(f"case.json has {len(cams)} cameras; run `mvb2r.py cameras {case}` first")
    common.require(case / "blockout.blend", f"mvb2r.py new {case}")
    if not args.verify_only:
        log = case / "logs" / "views.log"
        common.say(f"[views] {len(cams)} views at {cfg['views']['resolution']} px, clay {args.samples} samples "
                   f"(log: logs/views.log) ...")
        rc = common.run_blender("passes.py", ["--case", str(case), "--samples", str(args.samples)], log)
        if rc != 0:
            common.say(f"[views] passes.py FAILED (exit {rc}):\n{common.log_tail(log)}")
            return rc
        for line in common.log_tail(log, 400).splitlines():
            if line.startswith("[passes] view"):
                common.say("  " + line)
    return verify.run_verify(case)
