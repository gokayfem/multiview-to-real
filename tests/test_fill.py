"""Fill of unobserved texels (mvb2r.fill): smooth Gaussian pull-push, well-lit reference, lift, class rule."""
import numpy as np

from _fusion_fixtures import quad

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _tex(R):
    from mvb2r.raster import texel_set
    P, UV, N = quad()
    return texel_set(P, UV, N, R)


def _roughness(img, observed_img):
    """99th percentile of |Laplacian| over texels > 2 texels from any observation (kinks and blocks)."""
    import cv2
    d = cv2.distanceTransform((~observed_img).astype(np.uint8), cv2.DIST_L2, 5)
    lap = np.abs(cv2.Laplacian(img, cv2.CV_32F)).max(-1)
    return float(np.percentile(lap[d > 2], 99))


def test_smooth_fill_keeps_observed_and_has_no_blocks():
    from mvb2r.fill import fill_texture
    R = 256
    tex = _tex(R)
    rng = np.random.default_rng(1)
    col = rng.uniform(0.1, 0.9, (tex.n, 3)).astype(np.float32)
    obs = (rng.random(tex.n) < 0.02) & (tex.cols < R // 2)       # sparse, jagged observations (strict gating)
    smooth = fill_texture(tex, col, obs, far=1e6)
    blocks = fill_texture(tex, col, obs, far=1e6, mode="blocks")
    assert np.allclose(smooth[tex.rows[obs], tex.cols[obs]], col[obs])
    seen = np.zeros((R, R), bool)
    seen[tex.rows[obs], tex.cols[obs]] = True
    s, b = _roughness(smooth, seen), _roughness(blocks, seen)
    assert s < 0.5 * b, (s, b)
    return f"fill roughness (99th pct |Laplacian|): smooth {s:.4f}, blocks {b:.4f}"


def test_blocks_mode_is_the_old_push_pull():
    from mvb2r.coverage import push_pull
    from mvb2r.fill import fill_texture
    tex = _tex(64)
    col = np.tile([0.2, 0.4, 0.6], (tex.n, 1)).astype(np.float32)
    col[tex.cols % 2 == 0] *= 1.2
    obs = tex.cols < 8
    out = fill_texture(tex, col, obs, mode="blocks", far=1e6)
    img = np.zeros((64, 64, 3), np.float32)
    w = np.zeros((64, 64), np.float32)
    img[tex.rows[obs], tex.cols[obs]] = col[obs]
    w[tex.rows[obs], tex.cols[obs]] = 1
    ref = push_pull(img, w)
    ref[w > 0] = img[w > 0]
    assert np.allclose(out, ref, atol=1e-6)


def test_well_lit_reference_ignores_shadowed_texels():
    from mvb2r.fill import fill_texture, well_lit_colour
    tex = _tex(64)
    lit, mid, dark = (np.array(c, np.float32) for c in ([0.6, 0.5, 0.4], [0.3, 0.25, 0.2], [0.05, 0.04, 0.03]))
    col = np.tile(lit, (tex.n, 1))
    col[tex.rows < 39] = mid                      # shadowed corners: 40 % dark, 20 % half-lit, 40 % lit
    col[tex.rows < 26] = dark
    obs = tex.cols < 8
    ref = well_lit_colour(col[obs])
    assert np.allclose(ref, lit, atol=1e-6), ref
    far_med = fill_texture(tex, col, obs)[:, 60]
    far_lit = fill_texture(tex, col, obs, reference="well_lit")[:, 60]
    assert np.allclose(far_lit, lit, atol=1e-4) and (far_med @ LUM).mean() < (far_lit @ LUM).mean()


def test_lift_keeps_the_fill_from_being_darker_than_the_reference():
    from mvb2r.fill import fill_distances, fill_texture, well_lit_colour
    R = 256
    tex = _tex(R)
    rng = np.random.default_rng(0)
    col = (np.array([0.5, 0.45, 0.4], np.float32) * rng.uniform(0.75, 1.0, (tex.n, 1))).astype(np.float32)
    col[tex.cols < 3] *= 0.7                      # a shadowed edge right at the observed border
    obs = tex.cols < 12
    ref_l = float(well_lit_colour(col[obs]) @ LUM)
    lift_d, _ = fill_distances(R)
    plain = fill_texture(tex, col, obs, reference="well_lit")
    lifted = fill_texture(tex, col, obs, reference="well_lit", lift=True)
    zone = (tex.cols >= 12 + lift_d)
    lum_l = lifted[tex.rows[zone], tex.cols[zone]] @ LUM
    assert lum_l.min() >= ref_l * 0.999, (lum_l.min(), ref_l)
    assert np.allclose(lifted[tex.rows[obs], tex.cols[obs]], col[obs])
    assert (plain[tex.rows[zone], tex.cols[zone]] @ LUM).min() < ref_l * 0.999


def test_fill_distances_scale_with_texture_size():
    from mvb2r.fill import fill_distances
    assert fill_distances(2048) == (96.0, 192.0)
    assert fill_distances(1024) == (48.0, 96.0)
    assert fill_distances(4096) == (192.0, 384.0)
    lo = fill_distances(64)
    assert lo[0] >= 4 and lo[1] >= 8


def test_fill_reference_for_wall_like_classes():
    from mvb2r.case import CLASSES
    from mvb2r.fill import WALL_LIKE_CLASSES, fill_reference_for
    assert set(WALL_LIKE_CLASSES) <= set(CLASSES)
    assert fill_reference_for("plaster") == ("well_lit", True) and fill_reference_for("paint") == ("well_lit", True)
    for c in ("skin", "hair", "fabric", "wood", "generic", "unknown-class"):
        assert fill_reference_for(c) == ("median", False), c


def test_bad_mode_or_reference_is_refused():
    from mvb2r.common import StageError
    from mvb2r.fill import fill_texture
    tex = _tex(32)
    col = np.full((tex.n, 3), 0.3, np.float32)
    obs = tex.cols < 4
    for kw in ({"mode": "fast"}, {"reference": "mean"}):
        try:
            fill_texture(tex, col, obs, **kw)
        except StageError:
            continue
        raise AssertionError(f"{kw} accepted")


def test_never_observed_object_takes_base_colour():
    from mvb2r.fill import fill_texture
    tex = _tex(32)
    out = fill_texture(tex, np.zeros((tex.n, 3), np.float32), np.zeros(tex.n, bool), base=(0.1, 0.2, 0.3),
                       reference="well_lit", lift=True)
    assert np.allclose(out, [0.1, 0.2, 0.3])
