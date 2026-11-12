"""Visibility tests and reason codes on an analytic quad (pure numpy)."""
import numpy as np

from _fusion_fixtures import checker, quad, render_quad, top_camera


def _setup(W=96, **kw):
    from mvb2r.raster import texel_set
    from mvb2r.visibility import make_view
    P, UV, N = quad()
    tex = texel_set(P, UV, N, 32)
    cam = top_camera(W=W, **kw)
    depth, ids, img = render_quad(cam, colour_fn=checker)
    return tex, cam, depth, ids, img, make_view(cam, depth, ids, img)


def test_valid_and_colour_matches_ground_truth():
    from mvb2r.visibility import VisParams, view_samples
    tex, cam, depth, ids, img, vd = _setup()
    s = view_samples(tex, 1, vd, VisParams())
    ok = s.flags == 0
    assert ok.mean() > 0.7, ok.mean()
    gt = checker(tex.P[:, 0], tex.P[:, 1])
    err = np.abs(s.color[ok] - gt[ok]).mean()
    assert err < 0.06, err


def test_reason_codes():
    from mvb2r.visibility import REASONS, Gates, VisParams, make_view, view_samples
    tex, cam, depth, ids, img, vd = _setup()
    prm = VisParams()
    # occluder: depth map 0.5 m closer in the left half
    d2 = depth.copy()
    d2[:, : cam.width // 2] -= 0.5
    s = view_samples(tex, 1, make_view(cam, d2, ids, img), prm)
    left = tex.P[:, 0] < -0.2
    assert ((s.flags[left] & REASONS["occluded"]) != 0).mean() > 0.9
    # wrong object id
    s = view_samples(tex, 7, vd, prm)
    assert ((s.flags & REASONS["wrong_object"]) != 0).mean() > 0.7
    # mask gate (accept nothing) and border margin exist
    gate = Gates(accept=np.zeros(ids.shape, bool))
    s = view_samples(tex, 1, vd, prm, gate)
    inside = (s.flags & REASONS["off_image"]) == 0
    assert ((s.flags[inside] & REASONS["mask_gate"]) != 0).all()
    # camera below the plane: backface; plane beyond the far side: behind
    s = view_samples(tex, 1, make_view(top_camera(W=96, height=-3.0), depth, ids, img), prm)
    assert ((s.flags & (REASONS["backface"] | REASONS["behind"])) != 0).all()


def test_edge_and_alignment_reduce_score():
    from mvb2r.visibility import REASONS, VisParams, make_view, view_samples
    tex, cam, depth, ids, img, vd = _setup(W=96, height=2.2)
    s = view_samples(tex, 1, vd, VisParams())
    edge = (s.flags & REASONS["edge_margin"]) != 0
    assert edge.any()
    al = np.full(depth.shape, 0.1, np.float32)
    s2 = view_samples(tex, 1, make_view(cam, depth, ids, img, align=al), VisParams())
    assert ((s2.flags & REASONS["low_align"]) != 0).sum() > 0.5 * (s.flags == 0).sum()


def test_interp_depth_plane_exact():
    from mvb2r.visibility import interp_depth
    j, i = np.mgrid[0:8, 0:8]
    z = 1.0 / (0.5 + 0.01 * (i + 0.5))  # 1/Z linear in x: a plane
    d = interp_depth(z.astype(np.float64), np.array([3.25]), np.array([4.0]))
    assert np.isclose(1 / d[0], 0.5 + 0.01 * 3.25, atol=1e-9)


def test_primary_reason_priority():
    from mvb2r.visibility import REASONS, REASON_ORDER, primary_reason
    f = np.array([0, REASONS["occluded"] | REASONS["behind"], REASONS["low_align"]], np.uint16)
    p = primary_reason(f)
    assert p[0] == 0 and REASON_ORDER[p[1] - 1] == "behind" and REASON_ORDER[p[2] - 1] == "low_align"
