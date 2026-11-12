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
