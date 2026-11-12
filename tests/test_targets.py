"""Generated-object masks, QA, acceptance, align_conf and target freezing (synthetic case, mock images)."""
import numpy as np

from _util import synthetic_case, tmpdir
from mvb2r import common, masks, mockgen, qa, refine, targets


def _mock_view(case, view, amp=1.0, seed=1):
    lin = mockgen.view_truth(case, view)
    out = mockgen.defect(lin, np.random.default_rng(seed), amp)
    return (common.linear_to_srgb(out) * 255 + 0.5).astype(np.uint8)


def test_segmenter_follows_each_object_and_rates_itself():
    case = synthetic_case(2, 160)
    cam = common.load_cameras(case)[0]
    vp = common.view_passes(case, cam)
    rgb = _mock_view(case, cam["name"])
    seg = masks.segment_view(rgb, vp["oid"])
    assert set(seg) == {1, 2, 3}
    for i in (2, 3):
        assert masks.iou(seg[i]["mask"], vp["oid"] == i) > 0.8, (i, masks.iou(seg[i]["mask"], vp["oid"] == i))
        assert seg[i]["reliability"] in ("high", "moderate", "low")
    assert seg[1]["reliability"] == "prior only"
    total = sum(r["mask"].astype(int) for r in seg.values())
    assert total.max() <= 1, "masks overlap"


def test_shifted_image_lowers_iou_and_moves_landmarks():
    case = synthetic_case(2, 160)
    objs = common.case_objects(case)
    cam = common.load_cameras(case)[0]
    vp = common.view_passes(case, cam)
    rgb = _mock_view(case, cam["name"], amp=0.2)
    lms = {"corner": {"world": [0.0, 0.3, 0.6], "object": "block_a"}}
    good, _ = qa.view_qa(rgb, vp["oid"], objs, lms, cam, vp["depth"])
    shifted, _ = qa.view_qa(np.roll(rgb, 6, axis=1), vp["oid"], objs, lms, cam, vp["depth"])
    assert shifted["objects"]["block_a"]["iou"] < good["objects"]["block_a"]["iou"]
    assert shifted["objects"]["block_a"]["chamfer_px"]["mean"] > good["objects"]["block_a"]["chamfer_px"]["mean"]
    assert "props" in good["groups"]
    if "corner" in good["landmarks"]:
        assert abs(good["landmarks"]["corner"]["offset_px"]) < 3


def test_acceptance_rule_is_applied_per_object():
    rule = refine.load_acceptance(tmpdir())["refine"]
    obj = lambda iou, ch: {"clay_px": 50000, "iou": iou, "chamfer_px": {"mean": ch}, "reliability": "high"}
    cell = {"objects": {"a": obj(0.9, 2.0)}}
    ok = refine.evaluate(rule, cell, {"objects": {"a": obj(0.89, 2.0)}, "landmarks": {}}, {"a": 2.0}, 1664, 1664 ** 2, None)
    assert ok and all(ok.values())
    bad = refine.evaluate(rule, cell, {"objects": {"a": obj(0.80, 30.0)}, "landmarks": {"tip": {"offset_px": 25}}},
                          {"a": 12.0}, 1664, 1664 ** 2, None)
    assert sum(not v for v in bad.values()) == 4
    low = refine.evaluate(rule, cell, {"objects": {"a": {**obj(0.1, 90.0), "reliability": "low"}}, "landmarks": {}},
                          {}, 1664, 1664 ** 2, None)
    assert low == {}


def test_align_conf_drops_where_masks_disagree():
    oid = np.zeros((100, 100), np.int32)
    oid[30:70, 30:70] = 2
    gen = oid == 2
    gen_shift = np.roll(gen, 8, axis=1)
    c_ok = targets.align_conf(oid, {2: {"mask": gen, "reliability": "high"}})
    c_bad = targets.align_conf(oid, {2: {"mask": gen_shift, "reliability": "high"}})
    c_prior = targets.align_conf(oid, {2: {"mask": gen, "reliability": "prior only"}})
    assert c_ok.min() == 255 and c_bad[50, 72] < 100 and c_bad[50, 50] == 255
    assert c_prior[50, 30] <= 128 and c_prior[50, 50] == 255


def test_freeze_with_user_images_records_native_and_resize():
    case = synthetic_case(2, 96)
    d = tmpdir()
    for cam in common.load_cameras(case):
        img = _mock_view(case, cam["name"])
        common.write_rgb(d / f"{cam['name']}.png", common.resize_rgb(img, (48, 48))[0])
    from argparse import Namespace
    assert targets.run(Namespace(case=str(case), images=str(d))) == 0
    t = common.read_json(case / "targets" / "targets.json")
    v = t["views"]["view_01"]
    assert t["source"] == "user images" and v["status"] == "user_image"
    assert v["native_size"] == [48, 48] and v["calibration_size"] == [96, 96] and "NOT generated detail" in v["resize"]
    assert (case / v["align_conf_png"]).is_file() and v["masks"]["block_a"]["reliability"]
    common.write_rgb(d / "view_01.png", np.zeros((40, 80, 3), np.uint8))
    try:
        targets.run(Namespace(case=str(case), images=str(d)))
    except common.StageError as e:
        assert "aspect" in str(e)
    else:
        raise AssertionError("wrong aspect accepted")


def test_resize_wording_user_vs_generated():
    case = synthetic_case(2, 64)
    d = tmpdir()
    big = np.full((128, 128, 3), 90, np.uint8)
    common.write_rgb(d / "n.png", big)
    cam = common.load_cameras(case)[0]
    objs = common.case_objects(case)
    u = targets.freeze_view(case, cam, objs, d / "n.png", d / "o1", {"status": "user_image"})
    g = targets.freeze_view(case, cam, objs, d / "n.png", d / "o2", {"status": "fallback_grid_cell"})
    assert u["resize"] == "area downsample", u["resize"]
    assert "resampled; not new detail" in g["resize"]
