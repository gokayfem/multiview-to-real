"""Fusion methods on a synthetic 3-view quad: every method, gating, single-source regions, LOO bookkeeping."""
import numpy as np

from _fusion_fixtures import checker, quad, render_quad, top_camera

GAINS = [(1.0, 1.0, 1.0), (1.08, 1.0, 0.94), (0.6, 1.4, 1.5)]  # view 3 strongly tinted (an outlier)


def make_object(regions=False, gains=GAINS, R=48):
    from mvb2r.fusion import ObjectInput, build_stack
    from mvb2r.levels import image_band_sampler
    from mvb2r.raster import texel_set
    from mvb2r.regions import single_source_regions
    from mvb2r.visibility import VisParams, make_view, view_samples
    P, UV, N = quad()
    tex = texel_set(P, UV, N, R)
    cams = [top_camera(f"view_0{k + 1}", W=96, x=x, height=2.6) for k, x in enumerate((-0.6, 0.6, 0.0))]
    views, samples = [], []
    for cam, g in zip(cams, gains):
        depth, ids, img = render_quad(cam, colour_fn=lambda X, Y: checker(X, Y) * np.array(g, np.float32))
        vd = make_view(cam, depth, ids, img)
        views.append(vd)
        samples.append(view_samples(tex, 1, vd, VisParams()))
    st = build_stack(samples, [c.name for c in cams])
    bands = image_band_sampler(views, 1, [v.ids == 1 for v in views], st.u, st.v)
    regs = tuple(single_source_regions(tex, st)) if regions else ()
    from dataclasses import replace
    from types import SimpleNamespace
    from mvb2r.projinputs import _resampler
    adm = [v.ids == 1 for v in views]
    base = ObjectInput(name="Quad", cls="generic", tex=tex, st=st, regions=regs, bands=bands,
                       targets=tuple((v.image, v.ids.shape) for v in views))
    return replace(base, resample=_resampler(base, SimpleNamespace(views=views), 1, samples, adm,
                                             VisParams())), tex


def test_every_method_runs_and_is_close_to_truth():
    from mvb2r.fusion import METHODS, FusionParams, fuse
    obj, tex = make_object(gains=GAINS[:2] + [(1.0, 1.0, 1.0)])
    gt = checker(tex.P[:, 0], tex.P[:, 1])
    errs = {}
    for m in METHODS:
        r = fuse(obj, m, FusionParams())
        assert r.color.shape == (tex.n, 3) and r.observed.mean() > 0.75, (m, r.observed.mean())
        errs[m] = float(np.abs(r.color[r.observed] - gt[r.observed]).mean())
        assert errs[m] < 0.08, (m, errs[m])
    return " ".join(f"{k}={v:.3f}" for k, v in errs.items())


def test_blend_rejects_disagreeing_view():
    from mvb2r.fusion import FusionParams, fuse
    obj, tex = make_object()
    gt = checker(tex.P[:, 0], tex.P[:, 1])
    r = fuse(obj, "blend", FusionParams())
    blended = r.extra["n_blended"]
    both = obj.st.valid.all(0)
    assert (blended[both] <= 2).mean() > 0.95  # the tinted view is gated out where the others agree
    assert np.abs(r.color[both] - gt[both]).mean() < 0.06


def test_single_source_regions_are_never_blended():
    from mvb2r.fusion import METHODS, FusionParams, fuse
    obj, tex = make_object(regions=True)
    rg = obj.regions[0]
    assert len(obj.regions) == 1 and rg["mask"].all()
    first = rg["views"][0]
    for m in METHODS:
        r = fuse(obj, m, FusionParams())
        ok = obj.st.valid[first]
        assert (r.source[ok] == first).all(), m
        if m != "register":  # register re-samples the same single view at registered positions
            assert np.allclose(r.color[ok], obj.st.color[first, ok], atol=2e-3), m


def test_prefer_view_wins():
    from mvb2r.regions import single_source_regions
    obj, tex = make_object()
    rg = single_source_regions(tex, obj.st, prefer=2)
    assert rg[0]["views"][0] == 2


def test_resolve_method_auto_rules():
    from mvb2r import fusion
    assert fusion.resolve_method("auto") == fusion.AUTO_DEFAULT
    assert fusion.resolve_method("auto", detail_critical=True) == "best"
    assert fusion.resolve_method("auto", "skin", class_overrides={"skin": "blend"}) == "blend"
    assert fusion.resolve_method("C") == "regions"
    try:
        fusion.resolve_method("nope")
        raise AssertionError("unknown method accepted")
    except ValueError:
        pass


def test_params_overrides_validated():
    from mvb2r.fusion import FusionParams
    p = FusionParams().with_overrides({"tau_gate": "0.3", "mb_sigmas": [2, 8]})
    assert p.tau_gate == 0.3 and p.mb_sigmas == (2, 8)
    try:
        FusionParams().with_overrides({"bogus": 1})
        raise AssertionError("unknown parameter accepted")
    except ValueError:
        pass


def test_loo_bookkeeping_excludes_view():
    from mvb2r.fusion import METHODS, FusionParams, fuse
    obj, tex = make_object()
    for k in range(obj.st.V):
        o2 = obj.with_stack(obj.st.without(k))
        assert not o2.st.valid[k].any() and obj.st.valid[k].any()  # input untouched
        for m in METHODS:  # register included: its re-sampling must keep view k out
            r = fuse(o2, m, FusionParams())
            assert not (r.source == k).any(), (m, k)


def test_consistency_flags_outlier():
    from mvb2r.fusion import FusionParams, consistency
    obj, _ = make_object()
    inc = consistency(obj.st, FusionParams())
    allv = obj.st.valid.all(0)
    assert inc[2, allv].mean() > 0.8 and inc[0, allv].mean() < 0.05


def test_seam_metrics_detects_step():
    from mvb2r.seams import seam_metrics
    img = np.full((64, 64, 3), 0.5, np.float32) + np.random.default_rng(0).normal(0, 0.01, (64, 64, 3))
    lab = np.zeros((64, 64), int)
    lab[:, 32:] = 1
    ok = np.ones((64, 64), bool)
    flat = seam_metrics(img.astype(np.float32), lab, ok)
    img2 = img.copy()
    img2[:, 32:] += 0.1
    step = seam_metrics(img2.astype(np.float32), lab, ok)
    assert flat["fine_ratio"] < 1.3 and step["fine_ratio"] > 3, (flat, step)


def test_fill_keeps_observed_and_flattens_far():
    from mvb2r.coverage import area_fractions, fill_texture
    from mvb2r.raster import texel_set
    P, UV, N = quad()
    tex = texel_set(P, UV, N, 64)
    col = np.tile([0.2, 0.4, 0.6], (tex.n, 1)).astype(np.float32)
    col[tex.cols % 2 == 0] *= 1.2
    obs = tex.cols < 8
    out = fill_texture(tex, col, obs)
    assert np.allclose(out[tex.rows[obs], tex.cols[obs]], col[obs])
    far = out[:, 60]
    assert far.std(0).max() < 1e-4  # plain colour, no invented detail
    fr = area_fractions(tex, obs, np.zeros(tex.n, bool))
    assert abs(fr["observed"] - 0.125) < 1e-6 and abs(fr["inferred"] - 0.875) < 1e-6


def test_register_resample_keeps_held_out_view_out():
    from mvb2r.fusion import FusionParams
    from mvb2r.seams import method_register
    obj, tex = make_object(gains=GAINS[:2] + [(1.0, 1.0, 1.0)])
    o2 = obj.with_stack(obj.st.without(0))
    re = o2.resample({1: (np.zeros(tex.n, np.float32), np.zeros(tex.n, np.float32))}, o2.st)
    assert not re.st.valid[0].any() and re.st.valid[1].any()
    held = obj.st.valid[0] & ~obj.st.valid[1:].any(0)          # texels only view 0 sees
    r = method_register(o2, FusionParams())
    assert not r.observed[held].any() and not (r.source == 0).any()
    assert "register_median_px" in r.extra
