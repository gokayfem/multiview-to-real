"""Leak metric: observed texels whose source pixel the generated image assigns to another object (synthetic)."""
import numpy as np

from _fusion_fixtures import two_object_case


def _obj(ci, name):
    return next(o for o in ci.objects if o["name"] == name)


def test_bad_pixels_masked_object():
    from mvb2r.leak import bad_pixels
    ci = two_object_case(masks=("Card", "Floor"), shift=3)
    bad = bad_pixels(ci, _obj(ci, "Card"), 1)
    own, floor_m = ci.masks[("Card", 1)], ci.masks[("Floor", 1)]
    assert np.array_equal(bad, ~own | floor_m)                       # undilated own mask, any other mask


def test_bad_pixels_unmasked_object_uses_other_masks_and_nearer_clay():
    from mvb2r.leak import bad_pixels
    import cv2
    ci = two_object_case(masks=("Card",), shift=0)
    bad = bad_pixels(ci, _obj(ci, "Floor"), 1)
    card_mask = ci.masks[("Card", 1)]
    assert bad[card_mask].all()
    clay = (ci.views[1].ids == 2).astype(np.uint8)
    d = cv2.distanceTransform(1 - clay, cv2.DIST_L2, 5)
    floor = ci.views[1].ids == 1
    assert bad[floor & (d <= 2)].all() and not bad[floor & (d > 3)].any()
    assert not bad_pixels(ci, _obj(ci, "Card"), 1)[ci.views[1].ids == 2].any()   # floor is not nearer


def test_leaked_texels_and_report_shape():
    import tempfile
    from _fusion_fixtures import with_meshes
    from mvb2r.fusion_io import load_mesh
    from mvb2r.leak import LEAK_DEFINITION, leak_counts, method_summary
    from mvb2r.raster import texel_set
    ci = with_meshes(two_object_case(masks=("Card",), shift=6), tempfile.mkdtemp(prefix="mvb2r_leak_"))
    floor = _obj(ci, "Floor")
    z = load_mesh(ci.mesh_dir, floor["file"])
    tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], 64)
    source = np.full(tex.n, 1, np.int16)
    soft = np.zeros(tex.n, bool)
    c = leak_counts(ci, floor, tex, source, soft)
    u, v, _ = ci.cams[1].project(tex.P)
    inside = (u >= 0) & (u < ci.cams[1].width) & (v >= 0) & (v < ci.cams[1].height)
    assert c["observed"] == tex.n and 0 < c["leaked"] < inside.sum()
    c2 = leak_counts(ci, floor, tex, np.full(tex.n, -1, np.int16), soft)
    assert c2 == {"observed": 0, "leaked": 0, "fraction": 0.0}
    c3 = leak_counts(ci, floor, tex, source, np.ones(tex.n, bool))   # soft samples are not "observed"
    assert c3["observed"] == 0
    s = method_summary({"Floor": c, "Card": {"observed": 100, "leaked": 0, "fraction": 0.0}})
    assert set(s) == {"objects", "leaked_fraction"}
    assert abs(s["leaked_fraction"] - c["leaked"] / (c["observed"] + 100)) < 1e-12
    assert "displaced" in LEAK_DEFINITION


def test_missing_method_output_reports_none_and_bad_pixels_are_cached():
    from mvb2r.leak import bad_pixel_cache, method_summary
    s = method_summary({})
    assert s == {"objects": {}, "leaked_fraction": None}
    ci = two_object_case(masks=("Card",), shift=0)
    cache = bad_pixel_cache(ci)
    a = cache(_obj(ci, "Floor"), 1)
    assert cache(_obj(ci, "Floor"), 1) is a
