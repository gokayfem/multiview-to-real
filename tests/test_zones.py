"""Contact zones (experimental): validation, pixel zones, gate exemption, the inferred colouring (synthetic)."""
import numpy as np

from _fusion_fixtures import quad, render_quad, top_camera, two_object_case

NAMES = ["Floor", "Card"]


def _card_points(n=40):
    g = np.linspace(-0.3, 0.3, n)
    X, Y = np.meshgrid(g, g)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, 0.4)]


def test_surface_points_back_project_depth():
    from mvb2r.visibility import make_view
    from mvb2r.zones import surface_points
    cam = top_camera(W=64, x=0.2, height=2.5)
    depth, ids, _ = render_quad(cam, z=0.0)
    P = surface_points(make_view(cam, depth, ids))
    m = ids > 0
    assert np.allclose(P[m][:, 2], 0.0, atol=1e-6)
    j, i = np.nonzero(m)
    u, v, _ = cam.project(P[m])
    assert np.allclose(u, i + 0.5, atol=1e-4) and np.allclose(v, j + 0.5, atol=1e-4)


def test_contacts_validated():
    from mvb2r.fusion_io import StageError
    from mvb2r.zones import parse_contacts
    bad = [{"surface": "Floor"}, {"surface": "Floor", "occluder": "Nope"}, {"surface": "Floor", "occluder": "Floor"},
           {"surface": "Floor", "occluder": "Card", "within_mm": -5},
           {"surface": "Floor", "occluder": "Card", "within_mm": float("nan")},
           {"surface": "Floor", "occluder": "Card", "box": [[0, 0], [1, 1]]},
           {"surface": "Floor", "occluder": "Card", "colour": 1}]
    for b in bad:
        try:
            parse_contacts([b], NAMES)
            raise AssertionError(f"accepted {b}")
        except StageError:
            pass
    try:
        parse_contacts({"surface": "Floor"}, NAMES)
        raise AssertionError("accepted a non-list")
    except StageError:
        pass
    c = parse_contacts([{"surface": "Floor", "occluder": "Card"}], NAMES)
    assert c[0]["within_mm"] == 20.0 and c[0]["box"] is None
    assert parse_contacts(None, NAMES) == ()


def test_contact_pixels_respect_distance_and_box():
    from mvb2r.zones import contact_pixels, parse_contacts
    ci = two_object_case(masks=("Card",), shift=0)
    floor = ci.objects[0]
    c = parse_contacts([{"surface": "Floor", "occluder": "Card", "within_mm": 450}], NAMES)
    z = contact_pixels(ci, floor, c, lambda name: _card_points())
    m = z[1]["Card"]
    ids = ci.views[1].ids
    assert m.any() and not (m & (ids != 1)).any()
    cb = parse_contacts([{"surface": "Floor", "occluder": "Card", "within_mm": 450,
                          "box": [[-5, -5, -1], [0, 5, 1]]}], NAMES)
    zb = contact_pixels(ci, floor, cb, lambda name: _card_points())[1]["Card"]
    assert zb.any() and zb.sum() < m.sum()
    from mvb2r.zones import surface_points
    P = surface_points(ci.views[1])
    assert (P[zb][:, 0] <= 1e-6).all()
    assert contact_pixels(ci, ci.objects[1], c, lambda name: _card_points()) == [{}, {}]   # not a surface


def test_zone_accepts_occluder_mask_pixels_under_strict():
    from mvb2r.projinputs import gates_for
    from mvb2r.zones import contact_pixels, parse_contacts
    ci = two_object_case(masks=("Card",), shift=6)
    floor = ci.objects[0]
    opt = {"gate_policy": "strict", "occ_margin": 0, "occ_soft": 0}
    c = parse_contacts([{"surface": "Floor", "occluder": "Card", "within_mm": 450}], NAMES)
    zones = contact_pixels(ci, floor, c, lambda name: _card_points())
    g0 = gates_for(ci, floor, 1, opt)
    g1 = gates_for(ci, floor, 1, opt, zones[1])
    card_mask_on_floor = ci.masks[("Card", 1)] & (ci.views[1].ids == 1)
    assert not (g0.accept & card_mask_on_floor).any()
    assert (g1.accept & card_mask_on_floor).any()


def test_contact_weights_ramp():
    from mvb2r.zones import contact_weights
    occ = np.array([[0.0, 0.0, 0.0]])
    P = np.array([[0.01, 0, 0], [0.015, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.01, 0, 0]])
    w = contact_weights(P, occ, 0.02, None)
    assert np.allclose(w[:4], [1.0, 1.0, 0.5, 0.0], atol=1e-6)
    wb = contact_weights(P, occ, 0.02, ((0.012, -1, -1), (1, 1, 1)))
    assert wb[0] == 0 and wb[1] == 1


def test_contact_fill_touches_only_unobserved():
    from mvb2r.raster import texel_set
    from mvb2r.zones import contact_fill
    P, UV, N = quad()
    tex = texel_set(P, UV, N, 32)
    img = np.full((32, 32, 3), 0.5, np.float32)
    obs = tex.cols < 16
    w = np.ones(tex.n, np.float32)
    out = contact_fill(img, tex, obs, w, np.array([0.1, 0.05, 0.02], np.float32))
    assert np.allclose(out[tex.rows[obs], tex.cols[obs]], 0.5)
    assert np.allclose(out[tex.rows[~obs], tex.cols[~obs]], [0.1, 0.05, 0.02])
    assert np.allclose(img, 0.5)                                      # input untouched


def test_project_contact_fill_keeps_coverage_and_observed_texels():
    import tempfile
    from dataclasses import replace
    from pathlib import Path
    from _fusion_fixtures import with_meshes
    from mvb2r.fusion_io import load_mesh, read_rgb_linear, write_png, write_srgb_png
    from mvb2r.project import apply_contact_fill
    from mvb2r.raster import texel_set
    from mvb2r.zones import parse_contacts
    tmp = Path(tempfile.mkdtemp(prefix="mvb2r_contact_"))
    (tmp / "mesh").mkdir()
    ci = with_meshes(two_object_case(masks=("Card",)), tmp / "mesh")
    ci = replace(ci, contacts=parse_contacts([{"surface": "Floor", "occluder": "Card", "within_mm": 600}], NAMES))
    R = 32
    z = load_mesh(ci.mesh_dir, ci.objects[0]["file"])
    tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], R)
    d = tmp / "out" / "best" / "Floor"
    d.mkdir(parents=True)
    obs = tex.cols < R // 2
    write_srgb_png(d / "texture.png", np.full((R, R, 3), 0.5, np.float32))
    cov_img = tex.scatter(np.where(obs, 255, 0).astype(np.uint8), 0)
    write_png(d / "coverage.png", cov_img)
    cov = {"objects": {"Floor": {"projected": True, "tex_size": R}, "Card": {"projected": True, "tex_size": R}}}
    apply_contact_fill(ci, cov, {"best": {"Card": [0.02, 0.02, 0.02], "Floor": [0.5, 0.5, 0.5]}}, tmp / "out")
    img = read_rgb_linear(d / "texture.png")[tex.rows, tex.cols]
    assert np.allclose(img[obs], 0.5, atol=3e-3)
    assert (img[~obs, 0] < 0.1).any()                                    # coloured towards the occluder
    assert np.array_equal(read_rgb_linear(d / "coverage.png")[..., 0] > 0.5, cov_img > 127)   # still unobserved
    assert cov["objects"]["Floor"]["contact_fill"]["best"][0]["texels"] > 0
