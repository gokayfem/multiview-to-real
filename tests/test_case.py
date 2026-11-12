"""case.json schema, inputs, scene merge; with Blender: the tiny end-to-end core path and UV repair."""
import json
import shutil

import numpy as np

from _util import need_blender, synthetic_case, tmpdir
from mvb2r import case as case_mod
from mvb2r import common, inputs

SPEC = common.TEMPLATES_DIR / "spec.example.json"


def test_template_spec_and_case_are_valid():
    inputs.validate_spec(json.loads(SPEC.read_text()))
    cfg = json.loads((common.TEMPLATES_DIR / "case.example.json").read_text())
    assert case_mod.validate_case(cfg) == []


def test_validate_case_reports_problems():
    cfg = common.load_case(synthetic_case(2))
    assert case_mod.validate_case(cfg) == []
    bad = {**cfg, "objects": cfg["objects"] + [{"name": "../x", "id": 1, "class": "lava"}],
           "references": [{"path": "../../etc/passwd", "role": "boss"}], "budget": {"max_calls": -1},
           "model": {"generator": "other"}, "views": {**cfg["views"], "resolution": 10}}
    errs = " | ".join(case_mod.validate_case(bad))
    for frag in ("not safe", "duplicate", "lava", "role", "inside the case", "max_calls", "generator", "resolution"):
        assert frag in errs, frag


def test_validate_case_checks_gate_regions_and_contacts():
    cfg = common.load_case(synthetic_case(2))
    names = [o["name"] for o in cfg["objects"]]
    good = {**cfg, "fusion": {"method": "auto", "gate_policy": "legacy", "occ_margin": 4,
                              "contacts": [{"surface": names[0], "occluder": names[1], "within_mm": 20}]},
            "objects": [{**cfg["objects"][0], "reject_dilate": 4,
                         "regions": [{"name": "r", "sphere": [0, 0, 0, 0.1], "cut": "hard"}]}] + cfg["objects"][1:]}
    assert case_mod.validate_case(good) == [], case_mod.validate_case(good)
    bad = {**cfg, "fusion": {"gate_policy": "loose", "contacts": [{"surface": names[0], "occluder": "ghost"}]},
           "objects": [{**cfg["objects"][0], "gate_dilate": "2",
                        "regions": [{"name": "r", "box": [[1, 1, 1], [0, 0, 0]]}]}] + cfg["objects"][1:]}
    errs = " | ".join(case_mod.validate_case(bad))
    for frag in ("gate_policy", "ghost", "gate_dilate", "minimum corner"):
        assert frag in errs, (frag, errs)


def test_validate_case_fusion_keys_fill_pbr_and_view_objects():
    import json as _json
    root = synthetic_case(2)
    cfg = common.load_case(root)
    errs = " | ".join(case_mod.validate_case({**cfg, "fusion": {"fill": "bogus", "occ_margn": 3}}))
    assert "fill" in errs and "occ_margn" in errs and "occ_margin" in errs, errs     # allowed keys listed
    errs = " | ".join(case_mod.validate_case({**cfg, "pbr": {"synth_direct": "yes"}}))
    assert "synth_direct" in errs, errs
    rows = _json.loads((root / "views" / "objects.json").read_text())
    rows.append({"name": "OnlyInViews", "id": 99})
    (root / "views" / "objects.json").write_text(_json.dumps(rows))
    first = cfg["objects"][0]["name"]
    c2 = {**cfg, "fusion": {"contacts": [{"surface": first, "occluder": "OnlyInViews"}]}}
    assert case_mod.validate_case(c2, root) == [], case_mod.validate_case(c2, root)
    assert any("OnlyInViews" in e for e in case_mod.validate_case(c2))


def test_spec_rejects_non_finite_and_huge_values():
    spec = json.loads(SPEC.read_text())
    o = spec["objects"][1]
    bad = [({**o, "location": [float("nan"), 0, 0]}, "finite"),
           ({**o, "location": [1e9, 0, 0]}, "finite"),
           ({**o, "parts": [{"type": "sphere", "radius": float("inf")}]}, "radius"),
           ({**o, "parts": [{"type": "cylinder", "radius": 0.1, "depth": -1}]}, "depth"),
           ({**o, "parts": [{"type": "box", "size": [1, 1, 1]}] * (inputs.MAX_PARTS + 1)}, "parts")]
    for obj, frag in bad:
        try:
            inputs.validate_spec({**spec, "objects": [obj]})
        except ValueError as e:
            assert frag in str(e), (frag, e)
        else:
            raise AssertionError(frag)
    many = [{**o, "name": f"o{i}", "id": i + 1} for i in range(inputs.MAX_OBJECTS + 1)]
    try:
        inputs.validate_spec({**spec, "objects": many})
    except ValueError as e:
        assert "at most" in str(e)
    else:
        raise AssertionError
    nan_text = SPEC.read_text().replace('"location": [-0.45, 0.1, 0]', '"location": [NaN, 0.1, 0]')
    d = tmpdir()
    (d / "s.json").write_text(nan_text)
    try:
        inputs.load_spec(d / "s.json")
    except common.StageError as e:
        assert "finite" in str(e)
    else:
        raise AssertionError("NaN accepted")


def test_spec_validation_names_the_problem():
    spec = json.loads(SPEC.read_text())
    cases = [({**spec, "objects": []}, "objects"),
             ({**spec, "objects": spec["objects"] + [dict(spec["objects"][0])]}, "duplicate"),
             ({**spec, "objects": [{**spec["objects"][0], "parts": [{"type": "torus"}]}]}, "type"),
             ({**spec, "cameras": [{"location": [0, 0, 1]}]}, "look_at"),
             ({**spec, "objects": [{**spec["objects"][0], "name": "a/b"}]}, "safe name")]
    for s, frag in cases:
        try:
            inputs.validate_spec(s)
        except ValueError as e:
            assert frag in str(e), (frag, e)
        else:
            raise AssertionError(frag)


def test_input_kinds_references_and_companions():
    assert inputs.input_kind("a.JSON") == "spec" and inputs.input_kind("x.blend") == "blend"
    assert inputs.input_kind("m.glb") == "mesh"
    try:
        inputs.input_kind("x.exe")
    except common.StageError:
        pass
    else:
        raise AssertionError
    assert inputs.parse_reference("/a/b.png:identity:Body,Hair") == {"src": "/a/b.png", "role": "identity",
                                                                     "applies_to": ["Body", "Hair"]}
    for bad in ("x.png", "x.png:boss"):
        try:
            inputs.parse_reference(bad)
        except common.StageError:
            continue
        raise AssertionError(bad)
    d = tmpdir()
    (d / "sub").mkdir()
    (d / "m.obj").write_text("mtllib ../../outside.mtl\nmtllib sub/ok.mtl\n")
    (d / "sub" / "ok.mtl").write_text("map_Kd tex.png\n")
    (d / "sub" / "tex.png").write_bytes(b"x")
    got = {p.name for p in inputs.companions(d / "m.obj")}
    assert got == {"ok.mtl", "tex.png"}, got
    (d / "notimg.png").write_text("hello")
    try:
        inputs.copy_reference({"src": str(d / "notimg.png"), "role": "style", "applies_to": []}, d / "in", 1)
    except Exception:
        pass
    else:
        raise AssertionError("a non-image was accepted as a reference")


def test_merge_scene_keeps_edits_and_cameras():
    cfg = common.load_case(synthetic_case(2))
    cfg["objects"][0]["description"] = "edited by the agent"
    scene = {"objects": [{"name": "ground", "id": 1, "description": "seed"}, {"name": "new", "id": 9}],
             "cameras": [{"name": f"Cam{i}"} for i in range(12)]}
    out = case_mod.merge_scene(cfg, scene, keep_cameras=True)
    assert out["objects"][0]["description"] == "edited by the agent" and out["objects"][1]["class"] == "generic"
    assert len(out["views"]["cameras"]) == common.MAX_VIEWS and out["views"]["cameras"][0]["camera"] == "Cam0"
    assert case_mod.merge_scene(cfg, {**scene, "cameras": scene["cameras"][:1]}, True)["views"]["cameras"] == cfg["views"]["cameras"]
    assert cfg["objects"][0]["description"] == "edited by the agent"   # input not mutated


# ---------------------------------------------------------------- Blender
def _cli(*argv) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("mvb2r_cli", common.SCRIPTS_DIR / "mvb2r.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli.main(list(argv))


def test_blender_core_round_trip_with_mock():
    need_blender()
    case = tmpdir() / "tiny"
    assert _cli("new", str(case), "--input", str(SPEC), "--views", "4", "--resolution", "96", "--generator", "mock") == 0
    assert _cli("views", str(case), "--samples", "1") == 0
    rep = common.read_json(case / "views" / "verify" / "report.json")
    assert rep["ok"] and rep["summary"]["max_reproj_err_px"] < 0.05
    for st in ("grid", "generate", "refine", "targets", "consistency"):
        assert _cli(st, str(case)) == 0, st
    t = common.read_json(case / "targets" / "targets.json")
    assert len(t["views"]) == 4
    # resync with a stand-in consensus re-render (the fusion half writes the real one)
    rr = case / "proj" / "stub" / "rerender"
    rr.mkdir(parents=True)
    for v in t["views"]:
        shutil.copyfile(case / "gen" / "mock_truth" / f"{v}.png", rr / f"{v}.png")
    assert _cli("resync", str(case)) == 0
    d = common.read_json(case / "gen" / "round_1" / "decision.json")
    assert d["pilot"]["views"] and "kept" in d
    assert _cli("report", str(case)) == 0 and "Cross-view consistency" in (case / "out" / "report.md").read_text()
    assert _cli("critique", str(case)) == 0
    return f"resync kept: {d['kept']}"


OBJ_STACKED = """o stacked
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 3 0 0
v 3 1 0
v 2 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3 4/4
f 5/1 6/2 7/3 8/4
"""


def test_blender_stacked_uvs_are_repaired():
    need_blender()
    d = tmpdir()
    (d / "stacked.obj").write_text(OBJ_STACKED)
    case = d / "c"
    assert _cli("new", str(case), "--input", str(d / "stacked.obj"), "--no-cameras", "--generator", "mock") == 0
    rows = common.read_json(case / "views" / "objects.json")
    uv = rows[0]["uv"]
    assert uv["overlap_px"] == 0 and uv["in_unit_square"] and "smart" in uv["method"], uv
    assert uv.get("overlap_before_px", 0) > 0, uv
    return f"overlap {uv['overlap_before_px']} px -> 0 ({uv['method']})"


MAKE_BLEND = r'''
import bpy, sys, mathutils
out, outside_abs = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
bpy.ops.mesh.primitive_plane_add(size=6)
for i, loc in enumerate(((3, -3, 2), (-3, -3, 2))):
    cam = bpy.data.objects.new(f"C{i}", bpy.data.cameras.new(f"C{i}"))
    sc.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector((0, 0, 0.5)) - cam.location).to_track_quat("-Z", "Y").to_euler()
def img(name, path):
    im = bpy.data.images.new(name, 4, 4); im.source = "FILE"; im.filepath = path; return im
w = bpy.data.worlds.new("hdri_world"); w.use_nodes = True
env = w.node_tree.nodes.new("ShaderNodeTexEnvironment")
env.image = img("sky_outside", "//../outside/sky.png")
w.node_tree.links.new(env.outputs["Color"], w.node_tree.nodes["Background"].inputs["Color"])
sc.world = w
mat = bpy.data.materials.new("m"); mat.use_nodes = True
for name, path in (("wood", "//textures/wood.png"), ("absolute", outside_abs), ("linked", "//textures/link.png"),
                   ("fake", "//textures/fake.png"), ("gone", "//textures/gone.png")):
    mat.node_tree.nodes.new("ShaderNodeTexImage").image = img(name, path)
bpy.data.objects["Cube"].data.materials.append(mat)
bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=False)
'''


def _external_case(blender):
    import subprocess
    d = tmpdir()
    for sub in ("scenes/textures", "outside"):
        (d / sub).mkdir(parents=True)
    common.write_rgb(d / "scenes" / "textures" / "wood.png", np.full((8, 8, 3), 120, np.uint8))
    common.write_rgb(d / "outside" / "sky.png", np.full((8, 8, 3), 200, np.uint8))
    common.write_rgb(d / "outside" / "secret.png", np.full((8, 8, 3), 50, np.uint8))
    (d / "scenes" / "textures" / "link.png").symlink_to(d / "outside" / "secret.png")
    (d / "scenes" / "textures" / "fake.png").write_text("this is not an image")
    (d / "make.py").write_text(MAKE_BLEND)
    subprocess.run([blender, "-b", "--factory-startup", "--disable-autoexec", "--python", str(d / "make.py"), "--",
                    str(d / "scenes" / "room.blend"), str(d / "outside" / "secret.png")],
                   capture_output=True, check=True, timeout=300)
    return d


def test_blender_external_images_stay_inside_the_input_folder():
    blender = need_blender()
    d = _external_case(blender)
    case = d / "case"
    assert _cli("new", str(case), "--input", str(d / "scenes" / "room.blend"), "--resolution", "96", "--generator", "mock") == 0
    ext = common.load_case(case)["input"]["external_images"]
    assert [p["file"] for p in ext["packed"]] == ["wood.png"], ext
    refused = {r["image"]: r["why"] for r in ext["refused"]}
    assert refused == {"sky_outside": "outside the input folder", "absolute": "outside the input folder",
                       "linked": "outside the input folder", "fake": "did not load as an image"}, refused
    assert [m["image"] for m in ext["missing"]] == ["gone"], ext
    assert "neutral grey world" in " ".join(common.read_json(case / "views" / "scene.json")["notes"])
    assert _cli("views", str(case), "--samples", "8") == 0
    rep = common.read_json(case / "views" / "verify" / "report.json")
    assert rep["summary"]["max_clay_missing_colour"] < 0.01 and rep["summary"]["max_clay_noise"] < 4, rep["summary"]
    trusted = d / "case_trusted"
    assert _cli("new", str(trusted), "--input", str(d / "scenes" / "room.blend"), "--allow-external-images",
                "--no-cameras", "--generator", "mock") == 0
    ext2 = common.load_case(trusted)["input"]["external_images"]
    assert {p["image"] for p in ext2["packed"]} == {"sky_outside", "absolute", "linked", "wood"}, ext2
    return f"packed {len(ext['packed'])}, refused {len(ext['refused'])}, missing {len(ext['missing'])}; trusted packed {len(ext2['packed'])}"


def test_generator_flag_works_on_a_case_made_for_fal():
    """new without --generator mock, then `run --generator mock` (the sequence the fusion agent hit)."""
    case = synthetic_case(3, 96, generator="fal")
    assert _cli("run", str(case), "--from", "probe", "--to", "targets", "--generator", "mock") == 0
    m = common.read_json(case / "gen" / "model.json")
    assert m["generator"] == "mock" and (case / "targets" / "targets.json").is_file()
    summ = common.read_json(case / "logs" / "run_summary.json")
    assert summ["generator"] == "mock" and summ["stages"][0]["stage"] == "probe" and summ["stages"][1]["stage"] == "grid"
    assert any(s["status"].startswith("skipped") for s in summ["stages"])
    assert _cli("refine", str(case)) == 2          # fal requested, mock model.json present: refused, not mixed
    assert _cli("report", str(case)) == 0 and "Last `run`" in (case / "out" / "report.md").read_text()


def test_fal_case_run_order_reaches_request_construction_without_a_key():
    """probe (saved schema, no network) runs BEFORE grid; generate builds the request and stops at the no-key refusal."""
    import os
    from test_model import schema
    case = synthetic_case(3, 96, generator="fal")
    sfile = tmpdir() / "schema.json"
    sfile.write_text(json.dumps(schema()))
    saved = {k: os.environ.pop(k) for k in ("FAL_KEY", "FAL_KEY_ID") if k in os.environ}
    try:
        assert _cli("run", str(case), "--from", "probe", "--to", "grid", "--schema-file", str(sfile)) == 0
        m = common.read_json(case / "gen" / "model.json")
        assert m["generator"] == "fal" and m["capability_sources"]["pixel_cap"] == "schema x-fal block"
        lay = common.read_json(case / "grid" / "layout.json")
        assert lay["layout_source"].startswith("automatic") and lay["width"] * lay["height"] <= 4624220
        assert common.load_case(case)["grid"].get("rows") is None          # the automatic layout is not frozen
        assert _cli("generate", str(case)) == 2
        assert (case / "gen" / "grid_edit" / "prompt.txt").is_file()
        from mvb2r import falio
        assert falio.read_log(case) == []
    finally:
        os.environ.update(saved)


def test_reference_arguments_parse_on_every_platform():
    assert inputs.parse_reference(r"C:\refs\a.png:identity") == {"src": r"C:\refs\a.png", "role": "identity", "applies_to": []}
    assert inputs.parse_reference(r"C:\refs\a.png=style") ["src"] == r"C:\refs\a.png"
    assert inputs.parse_reference("/x/a:b.png=material:Sofa,Chair")["applies_to"] == ["Sofa", "Chair"]
    assert inputs.parse_reference(r"C:\r\a.png:product:Box")["applies_to"] == ["Box"]


def test_run_flags_stale_outputs_and_unexpected_errors():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mvb2r_cli2", common.SCRIPTS_DIR / "mvb2r.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.flags_of(["[refine] WARNING budget exhausted", "fine", "missing-texture colour max 0.0%",
                         "[targets] view_01: fallback_grid_cell"]) == \
        ["[refine] WARNING budget exhausted", "[targets] view_01: fallback_grid_cell"]
    assert cli._status(0, ["x"]).startswith("WARNING") and cli._status(0, []) == "ok"
    assert cli.call_stage(lambda a: 1 / 0, None) == 3
    case = synthetic_case(2, 64)
    (case / "proj").mkdir()
    (case / "proj" / "score.json").write_text("{}")
    import time
    time.sleep(0.05)
    (case / "targets").mkdir()
    (case / "targets" / "targets.json").write_text("{}")
    assert cli.stale_outputs(case) == ["proj/score.json"]


def _faulty_case(faults: dict):
    case = synthetic_case(4, 128)
    cfg = common.load_case(case)
    cfg["model"]["mock_faults"] = faults
    common.save_case(case, cfg)
    return case


def _chain_with_fault(faults: dict, bad: str) -> dict:
    case = _faulty_case(faults)
    assert _cli("run", str(case), "--from", "probe", "--to", "consistency", "--generator", "mock") == 0
    cells = common.read_json(case / "gen" / "qa" / "cells.json")
    assert not cells[bad]["gate"]["usable"], cells[bad]["gate"]
    assert all(cells[v]["gate"]["usable"] for v in cells if v != bad), {v: cells[v]["gate"] for v in cells}
    t = common.read_json(case / "targets" / "targets.json")["views"]
    assert t[bad]["usable"] is False and t[bad]["status"] == "rejected" and "rejected/" in t[bad]["target_png"]
    assert not (case / "targets" / f"{bad}.png").exists() and (case / "targets" / "rejected" / f"{bad}.png").is_file()
    assert sum(x["usable"] for x in t.values()) == 3
    c = common.read_json(case / "gen" / "qa" / "consistency.json")["sets"]["targets"]
    assert bad not in c["per_view"] and len(c["per_view"]) == 3
    summ = common.read_json(case / "logs" / "run_summary.json")
    gen = next(s for s in summ["stages"] if s["stage"] == "generate")
    assert gen["status"].startswith("WARNING") and any(bad in f for f in gen["flags"])
    return t[bad]["gate"]


def test_mock_duplicated_panel_is_rejected_and_the_rest_completes():
    g = _chain_with_fault({"duplicate": {"view_04": "view_03"}}, "view_04")
    assert any("better than its own" in r or "not measurable" in r for r in g["reasons"]), g
    return "; ".join(g["reasons"])[:150]


def test_mock_shifted_panel_is_rejected_and_the_rest_completes():
    g = _chain_with_fault({"shift": {"view_02": 0.2}}, "view_02")
    return "; ".join(g["reasons"])[:150]


def test_targets_refuses_with_fewer_than_two_usable_views():
    case = _faulty_case({"duplicate": {"view_02": "view_01", "view_03": "view_01", "view_04": "view_01"}})
    assert _cli("run", str(case), "--from", "probe", "--to", "generate", "--generator", "mock") == 0
    assert _cli("targets", str(case)) == 2
    assert not (case / "targets").exists()
