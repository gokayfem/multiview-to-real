"""UV rasteriser and pinhole camera maths (pure numpy)."""
import numpy as np

from _fusion_fixtures import quad, top_camera


def test_rasterise_covers_square_exactly():
    from mvb2r.raster import texel_set
    P, UV, N = quad()
    t = texel_set(P, UV, N, 32)
    assert t.n == 32 * 32, t.n
    # texel (row j, col i) centre uv = ((i+.5)/R, 1-(j+.5)/R) -> world x = -1 + 2u, y = -1 + 2v
    u = (t.cols + 0.5) / 32
    v = 1 - (t.rows + 0.5) / 32
    assert np.allclose(t.P[:, 0], -1 + 2 * u, atol=1e-9) and np.allclose(t.P[:, 1], -1 + 2 * v, atol=1e-9)
    assert np.allclose(t.area, (2.0 / 32) ** 2, rtol=1e-6)
    assert np.allclose(t.Ng, [0, 0, 1])


def test_rasterise_half_triangle_and_empty():
    from mvb2r.raster import rasterise, uv_to_texel_xy
    tm, _ = rasterise(uv_to_texel_xy(np.array([[[0, 0], [1, 0], [1, 1]]], float), 16), 16)
    n = int((tm >= 0).sum())
    assert 120 <= n <= 140, n  # half of 256 (+ the diagonal)
    tm, _ = rasterise(np.zeros((0, 3, 2)), 8)
    assert (tm < 0).all()


def test_mesh_components_two_parts():
    from mvb2r.raster import mesh_components
    P, _, _ = quad()
    P2 = P + np.array([5.0, 0, 0])
    comp = mesh_components(np.concatenate([P, P2]))
    assert comp[0] == comp[1] and comp[2] == comp[3] and comp[0] != comp[2]


def test_projection_convention():
    cam = top_camera(W=100, height=2.0, f=50.0)
    u, v, z = cam.project(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0, 0, 5.0]]))
    assert np.allclose([u[0], v[0], z[0]], [50, 50, 2.0])
    assert np.isclose(u[1], 75) and np.isclose(v[1], 25)  # world +Y is image up
    assert np.isnan(u[2]) and z[2] < 0


def test_camera_json_roundtrip_and_validation():
    from mvb2r.pinhole import camera_from_json
    from mvb2r.fusion_io import StageError
    cam = top_camera()
    c2 = camera_from_json(cam.to_json())
    assert np.allclose(c2.w2c, cam.w2c) and np.allclose(c2.K, cam.K)
    try:
        camera_from_json({"name": "x", "K": [[1]], "width": 4, "height": 4})
        raise AssertionError("bad camera accepted")
    except StageError:
        pass


def test_interpolate_and_novel_cameras():
    from mvb2r.pinhole import interpolate_camera, novel_cameras, path_through
    a, b = top_camera("view_01", x=-1.0), top_camera("view_02", x=1.0)
    m = interpolate_camera(a, b, 0.5, "mid")
    assert np.allclose(m.centre, [0, 0, 3.0])
    arc = interpolate_camera(a, b, 0.5, "arc", pivot=np.array([0, 0, 0.0]))
    assert np.isclose(np.linalg.norm(arc.centre), np.linalg.norm(a.centre))
    nov = novel_cameras([a, b, top_camera("view_03", x=3.0)], 2)
    assert len(nov) == 2 and nov[0].name == "novel_01"
    assert len(path_through([a, b], 5)) == 5




def test_texture_size_from_geometry():
    from mvb2r.texsize import default_max, pow2_ceil, triangle_resolution, weighted_percentile
    from mvb2r.visibility import make_view
    from _fusion_fixtures import dense_quad, render_quad
    cam = top_camera(W=256, height=2.0, f=200.0)   # plane parallel to the image: 100 px per metre everywhere
    depth, ids, _ = render_quad(cam)
    view = make_view(cam, depth, ids)
    P, UV, _ = dense_quad(80)                       # 12800 triangles, sub-texel at R = 64
    r, w = triangle_resolution(P, UV, view, 1)
    assert len(r) > 0.9 * len(P) and np.allclose(r, 200.0, rtol=1e-3), (len(r), r.min(), r.max())
    assert pow2_ceil(weighted_percentile(r, w, 50)) == 256
    P2, UV2, _ = dense_quad(80, uv_hi=0.5)          # a quarter of the UV square -> twice the size
    r2, _ = triangle_resolution(P2, UV2, view, 1)
    assert np.allclose(r2, 400.0, rtol=1e-3)
    Pq, UVq, _ = quad()                             # two huge triangles: subdivided before the centroid test
    rq, wq = triangle_resolution(Pq, UVq, view, 1)
    assert abs(weighted_percentile(rq, wq, 50) - 200.0) < 1.0 and wq.sum() > 0.9 * (ids == 1).sum()
    r3, _ = triangle_resolution(P, UV, view, 7)     # wrong object id: no evidence
    assert len(r3) == 0
    assert abs(weighted_percentile(np.array([1.0, 2, 3, 4]), np.array([1.0, 1, 1, 5]), 85) - 4) < 1e-9
    assert default_max(True, 1664) == 4096 and default_max(False, 1664) == 2048 and default_max(False, 512) == 1024


def test_subtexel_triangles_leave_no_holes():
    from mvb2r.raster import texel_set
    from mvb2r.texlookup import tex_sample
    from _fusion_fixtures import dense_quad
    R = 32
    P, UV, N = dense_quad(60, uv_lo=0.2, uv_hi=0.71)   # 7200 sub-texel triangles, island edge mid-texel
    t = texel_set(P, UV, N, R)
    j, i = np.mgrid[0:R, 0:R]
    u, v = (i + 0.5) / R, 1 - (j + 0.5) / R
    island = (u >= 0.2) & (u <= 0.71) & (v >= 0.2) & (v <= 0.71)
    m = t.mask()
    assert (island & ~m).sum() == 0                    # every in-island texel has a triangle
    near = (u > 0.2 - 1.5 / R) & (u < 0.71 + 1.5 / R) & (v > 0.2 - 1.5 / R) & (v < 0.71 + 1.5 / R)
    assert (near & ~m).sum() == 0                      # bleed padding around the island border
    assert t.core.sum() == island.sum() and (~t.core).sum() > 0
    assert np.isclose(t.weight.sum(), t.area[t.core].sum())
    img = np.zeros((R, R, 3), np.float32)
    img[m] = 1.0                                       # island + padding white, fill outside black
    q = np.random.default_rng(0).uniform(0.2, 0.71, (5000, 2))
    assert tex_sample(img, q, valid=m).min() > 0.999   # never reads fill
    assert tex_sample(img, q).min() > 0.999            # with the padding even plain bilinear stays inside


def test_arc_interpolation_wide_angles():
    from mvb2r.pinhole import interpolate_camera, vslerp
    for deg in (100.0, 120.0, 150.0, 170.0, 179.9):
        th = np.radians(deg)
        a = top_camera("a", x=3.0)
        b_pos = np.array([3 * np.cos(th), 3 * np.sin(th), 3.0])
        b = top_camera("b", x=b_pos[0], y=b_pos[1])
        for t, want in ((0.0, a.centre), (1.0, b.centre)):
            c = interpolate_camera(a, b, t, "m", pivot=np.array([0, 0, 3.0])).centre
            assert np.allclose(c, want, atol=1e-6), (deg, t, c, want)
        mid = interpolate_camera(a, b, 0.5, "m", pivot=np.array([0, 0, 3.0])).centre
        ang = np.degrees(np.arctan2(mid[1], mid[0]))
        if deg < 179:
            assert np.isclose(np.linalg.norm(mid[:2]), 3.0) and abs(ang - deg / 2) < 1e-6, (deg, ang)   # the short arc, never the far side
    v = vslerp(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), 0.5)
    assert np.allclose(v, [0, 0, 1])                       # opposite vectors: defined, over the up axis
