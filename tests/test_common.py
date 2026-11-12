"""mvb2r.common: names, case paths, JSON, image helpers."""
import numpy as np

from _util import tmpdir
from mvb2r import common


def test_view_names_round_trip():
    for i in (1, 2, 9):
        assert common.view_index(common.view_name(i)) == i
    for bad in (0, -1, "1"):
        try:
            common.view_name(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    try:
        common.view_index("view_1")
    except common.StageError as e:
        assert "view_01" in str(e)
        return
    raise AssertionError("accepted view_1")


def test_json_round_trip_and_atomic_write():
    d = tmpdir()
    common.write_json(d / "a" / "b.json", {"x": np.arange(3), "p": d})
    assert common.read_json(d / "a" / "b.json")["x"] == [0, 1, 2]
    assert not list((d / "a").glob("*.tmp"))
    (d / "bad.json").write_text("{nope")
    try:
        common.read_json(d / "bad.json")
    except common.StageError as e:
        assert "not valid JSON" in str(e)
    else:
        raise AssertionError("bad JSON accepted")


def test_require_names_the_producer():
    try:
        common.require(tmpdir() / "views" / "cameras.json", "mvb2r.py views CASE")
    except common.StageError as e:
        assert "cameras.json" in str(e) and "mvb2r.py views CASE" in str(e)
    else:
        raise AssertionError


def test_load_case_refuses_a_non_case():
    try:
        common.load_case(tmpdir())
    except common.StageError as e:
        assert "mvb2r.py new" in str(e)
    else:
        raise AssertionError


def test_resize_is_labelled_honestly():
    img = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
    same, m0 = common.resize_rgb(img, (60, 40))
    small, m1 = common.resize_rgb(img, (30, 20))
    big, m2 = common.resize_rgb(img, (120, 80))
    assert np.array_equal(same, img) and "native" in m0
    assert small.shape == (20, 30, 3) and "area" in m1
    assert big.shape == (80, 120, 3) and "NOT generated detail" in m2


def test_image_and_exr_io():
    d = tmpdir()
    rgb = np.zeros((8, 10, 3), np.uint8)
    rgb[..., 0] = 200
    common.write_rgb(d / "x.png", rgb)
    assert np.array_equal(common.read_rgb(d / "x.png"), rgb)
    assert common.image_size(d / "x.png") == (10, 8)
    import cv2
    a = np.dstack([np.full((4, 5), v, np.float32) for v in (1.0, 2.0, 3.0)])
    cv2.imwrite(str(d / "p.exr"), a[..., ::-1].copy())
    back = common.read_exr(d / "p.exr")
    assert np.allclose(back[0, 0], [1, 2, 3])


def test_srgb_linear_inverse():
    x = np.linspace(0, 1, 101)
    assert np.allclose(common.linear_to_srgb(common.srgb_to_linear(x)), x, atol=1e-5)


def test_case_objects_overlay_case_json():
    from _util import synthetic_case
    case = synthetic_case(2)
    cfg = common.load_case(case)
    cfg["objects"][1]["description"] = "changed"
    common.save_case(case, cfg)
    objs = {o["name"]: o for o in common.case_objects(case)}
    assert objs["block_a"]["description"] == "changed" and "bbox_min" in objs["block_a"]
