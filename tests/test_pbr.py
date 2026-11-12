"""PBR pure parts: de-light gain clamping, prior bounding of the mean albedo, rig-fit solve."""
import numpy as np


def test_gain_map_is_clamped_and_one_on_background():
    from mvb2r.delight import gain_map
    ids = np.zeros((64, 64), np.int32)
    ids[8:56, 8:56] = 3
    S = np.ones((64, 64), np.float32)
    S[8:56, 8:32] = 4.0                     # strongly lit half
    S[8:56, 32:56] = 0.05                   # deep shadow half
    g = gain_map(S, ids, sigma=2.0, gmin=0.7, gmax=1.6)
    inside = ids == 3
    assert g[inside].min() >= 0.7 - 1e-3 and g[inside].max() <= 1.6 + 1e-3
    assert g[20, 15] < 1 < g[20, 45]        # lit side darkened, shadow side brightened (bounded)
    assert np.allclose(g[0, 0], 1.0)


def test_bound_mean_keeps_detail_and_only_scales():
    from mvb2r.materials import bound_mean, class_prior, load_priors
    rng = np.random.default_rng(0)
    A = np.clip(0.02 + 0.01 * rng.random((32, 32, 3)), 0, 1).astype(np.float32)  # too dark for plaster
    obs = np.ones((32, 32), bool)
    lo, hi = class_prior("plaster", load_priors())["albedo_lum"]
    B, s = bound_mean(A, obs, lo, hi)
    assert s > 1 and np.allclose(B / A, s, rtol=1e-4)      # one scalar: measured detail is kept
    C, s2 = bound_mean(np.full((4, 4, 3), 0.6, np.float32), np.ones((4, 4), bool), lo, hi)
    assert s2 == 1.0                                        # inside the range: untouched
    assert class_prior("unknown-class", load_priors()) == load_priors()["generic"]


def test_rig_fit_converges_on_linear_model(tmp=None):
    from mvb2r.rigfit import next_scale, within
    true = 2.3                             # render luminance proportional to the light scale
    scale = 1.0
    for _ in range(3):
        r = {"median_ratio": scale / true, "p90_ratio": scale / true * 1.02}
        if within(r):
            break
        scale = next_scale(scale, r)
    assert within({"median_ratio": scale / true, "p90_ratio": scale / true * 1.02}), scale
    assert next_scale(1.0, {"median_ratio": None, "p90_ratio": None}) == 1.0


def test_rig_ratios_from_images():
    import tempfile
    from pathlib import Path
    from mvb2r.fusion_io import write_png, write_srgb_png
    from mvb2r.rigfit import ratios
    d = Path(tempfile.mkdtemp())
    ids = np.zeros((32, 32), np.uint8)
    ids[4:28, 4:28] = 5
    write_png(d / "views" / "view_01" / "object_id.png", ids)
    write_srgb_png(d / "targets" / "view_01.png", np.full((32, 32, 3), 0.2, np.float32))
    write_srgb_png(d / "r" / "view_01.png", np.full((32, 32, 3), 0.25, np.float32))
    r = ratios(d / "r", d, ["view_01"], [5])
    assert abs(r["median_ratio"] - 1.25) < 0.03 and r["views"] == 1


def test_pbr_warnings_for_failed_rig_and_skipped_delight():
    from mvb2r.pbr import pbr_warnings
    ok = {"ok": True, "final": {"median_ratio": 1.0, "p90_ratio": 1.0}, "history": [{}]}
    assert pbr_warnings(ok, {"view_01": {"applied": True}}) == []
    bad = {"ok": False, "final": {"median_ratio": 1.14, "p90_ratio": 0.87}, "history": [{}, {}]}
    w = pbr_warnings(bad, {"view_01": {"applied": False}})
    assert len(w) == 2 and "rig check failed" in w[0] and "view_01" in w[1]


def test_pbr_cli_synth_direct_flag():
    import argparse
    from mvb2r.pbr import register
    ap = argparse.ArgumentParser()
    register(ap.add_subparsers())
    assert ap.parse_args(["pbr", "c"]).synth_direct is None                 # case.json decides
    assert ap.parse_args(["pbr", "c", "--synth-direct"]).synth_direct is True
    assert ap.parse_args(["pbr", "c", "--no-synth-direct"]).synth_direct is False


def test_inferred_after_synthesis_counts_only_newly_inferred_area():
    from mvb2r.pbr import synth_readme_lines
    rep = {"ran": True, "objects": {"Wall": {"inferred_before": 0.1, "inferred_after": 0.25, "mask_texels": 10,
                                             "components_filled": 1, "components_skipped": 0}}}
    lines = synth_readme_lines(rep)
    assert any("Wall" in s and "0.25" in s for s in lines)
    assert synth_readme_lines({"ran": False, "objects": {}}) == []


def _planted_case(link_rel):
    import os
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp(prefix="mvb2r_pbr_sec_"))
    case, victim = root / "case", root / "victim"
    victim.mkdir()
    (victim / "base_color.png").write_bytes(b"victim")
    (case / link_rel).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(victim, case / link_rel)
    return case, victim


def _raises_stage_error(fn, why="symbolic link"):
    from mvb2r.common import StageError
    try:
        fn()
    except StageError as e:
        return why in str(e)
    return False


def test_pbr_preflight_refuses_planted_output_links():
    from mvb2r.pbr import preflight_outputs
    for rel in ("pbr/delit", "pbr/shading", "pbr", "out"):
        case, victim = _planted_case(rel)
        assert _raises_stage_error(lambda: preflight_outputs(case, ["view_01"])), rel
        assert (victim / "base_color.png").read_bytes() == b"victim"


def test_pbr_materials_refuse_a_planted_texture_link():
    from mvb2r.pbr import materials
    case, victim = _planted_case("pbr/textures/Wall")
    alb = {"objects": {"Wall": {"projected": True, "class": "plaster", "id": 1,
                                "coverage": {"observed": 1.0, "soft": 0.0, "inferred": 0.0}}}}
    assert _raises_stage_error(lambda: materials(case, alb, "best"))
    assert (victim / "base_color.png").read_bytes() == b"victim" and len(list(victim.iterdir())) == 1


def test_delight_refuses_a_planted_output_file_link():
    import os
    from mvb2r.delight import owned_output
    case, victim = _planted_case("elsewhere")
    (case / "pbr" / "delit").mkdir(parents=True)
    os.symlink(victim / "base_color.png", case / "pbr" / "delit" / "view_01.png")
    assert _raises_stage_error(lambda: owned_output(case, case / "pbr" / "delit", "view_01"))


BREAK_WORLD = """
import sys
import bpy
out = sys.argv[sys.argv.index("--") + 1]
w = bpy.data.worlds.new("hdri_world")
w.use_nodes = True
env = w.node_tree.nodes.new("ShaderNodeTexEnvironment")
im = bpy.data.images.new("sky_gone", 4, 4)
im.source = "FILE"
im.filepath = "//not_here/sky.hdr"
env.image = im
w.node_tree.links.new(env.outputs["Color"], w.node_tree.nodes["Background"].inputs["Color"])
bpy.context.scene.world = w
bpy.ops.wm.save_as_mainfile(filepath=out)
"""


def test_blender_fusion_side_renders_never_use_a_missing_world_image():
    """A world image that has gone missing since `new` (a moved case, an old case) rendered every PBR-side
    image magenta; every script that opens the scene through _fusion_bpy.open_blend replaces it and says so."""
    import subprocess
    from _util import need_blender, tmpdir
    from mvb2r import common, verify
    from mvb2r.fusion_io import run_blender
    blender = need_blender()
    case = tmpdir()
    run_blender("../../../../tests/roundtrip/build_scene.py", ["--out", case, "--size", 96], log=case / "logs" / "b.log")
    (case / "break.py").write_text(BREAK_WORLD)
    subprocess.run([blender, "-b", "--factory-startup", "--disable-autoexec", str(case / "blockout.blend"), "--python",
                    str(case / "break.py"), "--", str(case / "broken.blend")], capture_output=True, check=True, timeout=300)
    text = run_blender("render.py", ["--cameras", case / "views" / "cameras.json", "--out", case / "r", "--samples", 4,
                                     "--cpu"], blend=case / "broken.blend", log=case / "logs" / "r.log")
    assert "WARNING" in text and "missing image" in text, text[-600:]
    first = sorted((case / "r").glob("view_*.png"))[0]
    assert verify.clay_quality(common.read_rgb(first))["missing_colour_fraction"] < 0.01


CHECK_WORLD = """
import sys
import bpy
blender_dir, broken, unlinked, out = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, blender_dir)
import _fusion_bpy as fb
fb.open_blend(broken)
bpy.ops.wm.save_as_mainfile(filepath=out)
bpy.ops.wm.open_mainfile(filepath=out)
print("SAVED_WORLDS", sorted(w.name for w in bpy.data.worlds), "ACTIVE", bpy.context.scene.world.name)
bpy.ops.wm.open_mainfile(filepath=broken)
w = bpy.context.scene.world
for link in list(w.node_tree.links):
    w.node_tree.links.remove(link)          # the missing image is no longer connected to the output
bpy.ops.wm.save_as_mainfile(filepath=unlinked)
fb.open_blend(unlinked)
print("UNLINKED_ACTIVE", bpy.context.scene.world.name)
"""


def test_blender_missing_world_swap_keeps_the_original_world_and_ignores_unlinked_nodes():
    """The neutral stand-in must not cost the user their world: scripts that save (final.blend, the refit
    blockout) keep the original world in the file, and an image node that is not connected is not a reason."""
    import subprocess
    from _util import need_blender, tmpdir
    from mvb2r import common
    from mvb2r.fusion_io import run_blender
    blender = need_blender()
    case = tmpdir()
    run_blender("../../../../tests/roundtrip/build_scene.py", ["--out", case, "--size", 64], log=case / "logs" / "b.log")
    (case / "break.py").write_text(BREAK_WORLD)
    (case / "check.py").write_text(CHECK_WORLD)
    base = [blender, "-b", "--factory-startup", "--disable-autoexec"]
    subprocess.run(base + [str(case / "blockout.blend"), "--python", str(case / "break.py"), "--",
                           str(case / "broken.blend")], capture_output=True, check=True, timeout=300)
    r = subprocess.run(base + ["--python", str(case / "check.py"), "--", str(common.BLENDER_DIR), str(case / "broken.blend"),
                               str(case / "unlinked.blend"), str(case / "saved.blend")],
                       capture_output=True, text=True, check=True, timeout=300)
    saved = next(ln for ln in r.stdout.splitlines() if ln.startswith("SAVED_WORLDS"))
    assert "'hdri_world'" in saved and "ACTIVE mvb2r_neutral_world" in saved, saved
    assert "UNLINKED_ACTIVE hdri_world" in r.stdout, r.stdout[-400:]

