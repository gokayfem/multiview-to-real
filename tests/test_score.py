"""Score bookkeeping on synthetic arrays: band split, LOO aggregation, seam ratio with a known step."""
import numpy as np

from _fusion_fixtures import quad


def _tex(R=64):
    from mvb2r.raster import texel_set
    P, UV, N = quad()
    return texel_set(P, UV, N, R)


def test_band_errors_split_low_and_high():
    from mvb2r.score import _band_errors
    tex = _tex()
    rng = np.random.default_rng(0)
    a = np.full((tex.n, 3), 0.5, np.float32) + rng.normal(0, 0.02, (tex.n, 3)).astype(np.float32)
    m = np.ones(tex.n, bool)
    offset = _band_errors(tex, a + 0.1, a, m)          # pure low-band (exposure) difference
    assert abs(offset["full"] - 25.5) < 0.5 and offset["low"] > 20 and offset["high"] < 1.5, offset
    noise = _band_errors(tex, a + rng.normal(0, 0.05, a.shape).astype(np.float32), a, m)
    assert noise["high"] > 3 * noise["low"], noise    # fine detail difference lands in the high band
    assert _band_errors(tex, a, a, np.zeros(tex.n, bool)) is None


def test_aggregate_is_texel_weighted():
    from mvb2r.score import _aggregate
    out = _aggregate({"view_01": [{"full": 10, "low": 4, "high": 6, "texels": 100},
                                  {"full": 20, "low": 8, "high": 12, "texels": 300}],
                      "view_02": [{"full": 4, "low": 2, "high": 2, "texels": 50}]})
    assert out["view_01"]["full"] == 17.5 and out["view_02"]["full"] == 4
    assert out["mean"]["full"] == (17.5 + 4) / 2


def test_texture_seam_ratio_known_step():
    from mvb2r.seams import texture_seam_metrics
    tex = _tex(64)
    rng = np.random.default_rng(1)
    base = 0.5 + rng.normal(0, 0.01, (tex.n, 3)).astype(np.float32)
    src = (tex.cols >= 32).astype(np.int16)
    obs = np.ones(tex.n, bool)
    flat = texture_seam_metrics(tex, base, src, obs)
    step = texture_seam_metrics(tex, base + 0.08 * src[:, None], src, obs)
    assert flat["coarse_ratio"] < 1.5 and step["coarse_ratio"] > 4, (flat, step)
    assert step["boundary_pairs"] > 0


def test_loo_masks_only_held_out_view():
    from test_fusion import make_object
    from mvb2r.fusion import FusionParams, fuse
    obj, tex = make_object()
    k = 1
    res = fuse(obj.with_stack(obj.st.without(k)), "best", FusionParams())
    mask = obj.st.valid[k] & res.observed
    assert mask.sum() > 0.3 * tex.n and not (res.source[mask] == k).any()


def _case_with_targets(views_meta):
    import json
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp(prefix="mvb2r_usable_"))
    (d / "targets").mkdir()
    (d / "targets" / "targets.json").write_text(json.dumps({"views": views_meta}))
    return d


def test_unusable_views_are_skipped_and_refused_below_two():
    from _fusion_fixtures import top_camera
    from mvb2r.fusion_io import StageError, split_usable
    cams = [top_camera(f"view_0{k}") for k in (1, 2, 3, 4)]
    d = _case_with_targets({"view_04": {"usable": False, "reason": "near-duplicate of view_03"},
                            "view_02": {"usable": True}})
    keep, skipped = split_usable(d, cams)
    assert [c.name for c in keep] == ["view_01", "view_02", "view_03"]
    assert skipped == [{"view": "view_04", "reason": "near-duplicate of view_03"}]
    keep, skipped = split_usable(d.parent / "no_such_case", cams)      # no targets.json: all usable
    assert len(keep) == 4 and not skipped
    d2 = _case_with_targets({v: {"usable": False} for v in ("view_01", "view_02", "view_03")})
    try:
        split_usable(d2, cams)
        raise AssertionError("one usable view accepted")
    except StageError as e:
        assert "only 1 usable view" in str(e)


def test_mask_reliability_gate_filter():
    from mvb2r.fusion_io import GATE_RELIABILITY, mask_reliability
    meta = {"view_01": {"masks": {"ball": {"reliability": "high"}, "ground": {"reliability": "prior only"},
                                  "crate": {"reliability": "low"}}}}
    assert mask_reliability(meta, "view_01", "ball") in GATE_RELIABILITY
    assert mask_reliability(meta, "view_01", "ground") not in GATE_RELIABILITY
    assert mask_reliability(meta, "view_01", "crate") not in GATE_RELIABILITY
    assert mask_reliability(meta, "view_02", "ball") == "unknown" and "unknown" in GATE_RELIABILITY


def test_mask_gate_erodes_by_default():
    from mvb2r.projinputs import _dilate
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True
    assert _dilate(m, -1).sum() == 9 and _dilate(m, 1).sum() == 49 and _dilate(m, 0).sum() == 25


def test_loo_mask_uses_the_evaluation_validity_only():
    from types import SimpleNamespace
    from mvb2r.score import loo_mask
    ev = np.array([True, True, False, True])
    res = SimpleNamespace(observed=np.array([True, False, True, True]), extra={"soft": np.array([0, 0, 0, 1], bool)})
    assert loo_mask(ev, res).tolist() == [True, False, False, False]


def test_print_table_shows_leak():
    import contextlib
    import io
    from mvb2r.score import print_table
    out = {"methods": {"best": {"loo": None, "seams_novel": None, "seams_texture": {}, "sharpness": {}}},
           "leak": {"definition": "x", "per_method": {"best": {"objects": {}, "leaked_fraction": 0.043}}}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_table(out)
    assert "leak %" in buf.getvalue() and "4.30" in buf.getvalue()


def test_score_refuses_a_projection_made_with_other_settings():
    from mvb2r.fusion_io import StageError
    from mvb2r.score import check_projection
    cov = {"gate_policy": "strict", "fill": "smooth", "views": ["view_01", "view_02"]}
    check_projection(cov, {"gate_policy": "strict", "fill": "smooth"}, ["view_01", "view_02"])
    for opt, views, frag in (({"gate_policy": "legacy", "fill": "smooth"}, ["view_01", "view_02"], "gate_policy"),
                             ({"gate_policy": "strict", "fill": "blocks"}, ["view_01", "view_02"], "fill"),
                             ({"gate_policy": "strict", "fill": "smooth"}, ["view_01"], "views")):
        try:
            check_projection(cov, opt, views)
            raise AssertionError(frag)
        except StageError as e:
            assert frag in str(e) and "project" in str(e), str(e)
    # coverage.json of the previous release (nothing recorded) = legacy gates, blocks fill
    old = {"views": ["view_01", "view_02"]}
    check_projection(old, {"gate_policy": "legacy", "fill": "blocks"}, ["view_01", "view_02"])
