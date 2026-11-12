"""Direct-light patch synthesis (pbr.synth_direct): masks, structure direction, quilting fill, config validation."""
import numpy as np

R = 128
PERIOD = 8


def stripes(seed: int = 0) -> np.ndarray:
    """Clean 'plank' texture: horizontal stripes (vary along y only) with a little per-texel noise."""
    rng = np.random.default_rng(seed)
    y = np.arange(R)[:, None]
    band = np.where((y // (PERIOD // 2)) % 2 == 0, 0.30, 0.18) + 0.03 * np.sin(y * 0.7)
    tex = np.repeat(np.repeat(band, R, 1)[..., None], 3, -1) * np.array([1.0, 0.8, 0.6])
    return (tex + 0.005 * rng.standard_normal((R, R, 3))).astype(np.float32)


def disk(cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[:R, :R]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def lit(clean: np.ndarray, patch: np.ndarray, gain: float = 3.0) -> np.ndarray:
    return np.clip(np.where(patch[..., None], clean * gain, clean), 0, 1).astype(np.float32)


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def _lum(x):
    from mvb2r.fusion_io import lum
    return lum(x)


def test_bright_mask_finds_the_patch_from_the_photo_alone():
    from mvb2r.sunpatch import bright_mask
    patch = disk(64, 64, 14)
    raw = lit(stripes(), patch)
    ones = np.ones((R, R), bool)
    A, med = bright_mask(_lum(raw), ones, ones, k_bright=1.6, sigma=1.0)
    inter, union = (A & patch).sum(), (A | patch).sum()
    assert inter / union > 0.8, inter / union
    assert 0.15 < med < 0.3
    assert not bright_mask(_lum(stripes()), ones, ones, 1.6, 1.0)[0].any()   # no patch, nothing found


def test_mask_from_direct_light_alone_and_from_both():
    from mvb2r.sunpatch import combine_masks
    ones = np.ones((R, R), bool)
    a, b = disk(30, 30, 8), disk(95, 90, 10)
    none = np.zeros((R, R), bool)
    mb = combine_masks(none, b, ones, radius=3)
    assert (mb & b).sum() == b.sum() and mb.sum() > b.sum() and not mb[30, 30]
    assert mb[90, 95 + 10 + 3] and not mb[90, 95 + 10 + 6]                 # dilated by 3 texels, not more
    both = combine_masks(a, b, ones, radius=3)
    assert both[30, 30] and both[90, 95]
    inside = np.zeros((R, R), bool)
    inside[:, :96] = True
    assert not combine_masks(a, b, inside, radius=3)[:, 96:].any()          # never outside the atlas


def test_dilation_radius_from_texel_density():
    from mvb2r.sunpatch import dilate_texels
    area = np.full(100, (0.01) ** 2, np.float32)                             # 1 cm texels
    assert dilate_texels(0.2, area, 4096) == 20
    assert dilate_texels(0.2, area, 128) == 128 // 16                       # bounded by the atlas size
    assert dilate_texels(0.0, area, 4096) == 0


def test_structure_direction_follows_the_stripes():
    from mvb2r.sunpatch import structure_direction
    along, coh = structure_direction(_lum(stripes()), np.ones((R, R), bool))
    assert abs(along[0]) > 0.99 and coh > 0.8
    along_t, _ = structure_direction(_lum(stripes()).T.copy(), np.ones((R, R), bool))
    assert abs(along_t[1]) > 0.99


def test_fill_restores_stripes_matches_ring_and_is_deterministic():
    from mvb2r.sunpatch import SynthParams, synthesise
    clean, patch = stripes(), disk(64, 64, 14)
    base = lit(clean, patch)
    mask = dilate(patch, 4)
    ones = np.ones((R, R), bool)
    before = (base.copy(), mask.copy())
    prm = SynthParams(feather=3)
    out, synth, rep = synthesise(base, mask, ones, ones, prm)
    err = np.abs(out[patch] - clean[patch]).mean()
    unfilled = np.abs(base[patch] - clean[patch]).mean()
    assert err < 0.02 and err < 0.05 * unfilled, (err, unfilled)
    assert rep["components_filled"] == 1 and rep["components_skipped"] == 0
    assert rep["ring_diff_after"] < 0.1 * rep["ring_diff_before"] and rep["ring_diff_after"] < 0.01, rep
    assert (synth == mask).all() and rep["mask_texels"] == int(mask.sum())
    assert np.array_equal(out[~mask], base[~mask])                          # outside the mask untouched
    out2, _, _ = synthesise(base, mask, ones, ones, prm)
    assert np.array_equal(out, out2)                                        # fixed seed
    assert np.array_equal(base, before[0]) and np.array_equal(mask, before[1])   # inputs not mutated


def test_component_without_shaded_source_is_left_unchanged_and_reported():
    from mvb2r.sunpatch import SynthParams, synthesise
    clean = stripes()
    p_ok, p_bad = disk(32, 64, 10), disk(100, 64, 10)
    base = lit(clean, p_ok | p_bad)
    inside = np.ones((R, R), bool)
    inside[:, 63:66] = False                                                # two UV charts: x < 63 and x > 65
    observed = inside.copy()
    observed[:, 66:] = dilate(p_bad, 4)[:, 66:]                             # right chart: only the patch observed
    mask = dilate(p_ok | p_bad, 4) & inside
    out, synth, rep = synthesise(base, mask, observed, inside, SynthParams(feather=3))
    assert rep["components_filled"] == 1 and rep["components_skipped"] == 1, rep
    assert "source" in rep["skipped"][0]["reason"]
    assert np.array_equal(out[:, 66:], base[:, 66:]) and not synth[:, 66:].any()
    assert np.abs(out[p_ok] - clean[p_ok]).mean() < 0.02


def test_direct_lit_view_and_texture_space_vote():
    from mvb2r.sunpatch import direct_lit_view, direct_from_stack
    D = np.zeros((40, 60), np.float32)
    D[10:20, 10:30] = 2.0
    D[30:35, 40:50] = 0.05                                                  # bounce-level light: not direct
    ids = np.zeros((40, 60), np.int32)
    ids[5:38, 5:55] = 4
    m = direct_lit_view(D, ids)
    assert m[15, 20] and not m[32, 45] and m.sum() == 200
    valid = np.array([[True, True, False], [True, False, True]])
    score = np.ones((2, 3), np.float32)
    color = np.zeros((2, 3, 3), np.float32)
    color[0, 0] = color[1, 0] = 1.0                                         # both views say lit
    color[0, 1] = 1.0                                                       # only valid view says lit
    color[0, 2] = 1.0                                                       # invalid view says lit, valid says no
    assert direct_from_stack(valid, score, color).tolist() == [True, True, False]


def test_config_defaults_validation_and_selection():
    from mvb2r.common import StageError
    from mvb2r.sunpatch_stage import select_objects, synth_config
    assert synth_config({}, None) is None and synth_config({"pbr": {"synth_direct": True}}, False) is None
    for on in ({}, {"synth_direct": True}):                                 # on, but nothing named: refused
        try:
            synth_config({"pbr": on}, True)
        except StageError as e:
            assert "relight" in str(e) and "synth_objects" in str(e)
        else:
            raise AssertionError("synthesis without named objects or classes accepted")
    c = synth_config({"pbr": {"synth_objects": ["Floor"]}}, True)
    assert c["k_bright"] == 1.6 and c["dilate_m"] == 0.2 and c["feather"] == 24 and c["classes"] == []
    c = synth_config({"pbr": {"synth_direct": True, "synth_objects": ["Rug"],
                              "synth_params": {"k_bright": 2.0}}}, None)
    assert c["k_bright"] == 2.0 and c["objects"] == ["Rug"]
    bad = [{"synth_direct": "yes"}, {"synth_direct": True, "synth_objects": "Rug"},
           {"synth_direct": True, "synth_params": {"k_bright": 0.5}},
           {"synth_direct": True, "synth_params": {"dilate_m": -1}},
           {"synth_direct": True, "synth_params": {"feather": 1.5}},
           {"synth_direct": True, "synth_params": {"bogus": 1}},
           {"synth_direct": True, "synth_classes": ["unobtainium"]}, "nope"]
    for b in bad:
        try:
            synth_config({"pbr": b}, None)
        except StageError:
            continue
        raise AssertionError(f"accepted {b}")
    objs = {"A": {"class": "plaster", "projected": True}, "B": {"class": "fabric", "projected": True},
            "C": {"class": "fabric", "projected": True}, "D": {"class": "wood", "projected": False, "opening": True},
            "E": {"class": "wood", "projected": True, "detail_critical": True}}
    conf = synth_config({"pbr": {"synth_direct": True, "synth_objects": ["C"]}}, None)
    assert select_objects(objs, conf) == ["C"]                              # named only: no class default
    conf = synth_config({"pbr": {"synth_direct": True, "synth_classes": ["plaster"]}}, None)
    assert select_objects(objs, conf) == ["A"]                              # explicit class opt-in
    conf = synth_config({"pbr": {"synth_direct": True, "synth_objects": ["Nope"]}}, None)
    try:
        select_objects(objs, conf)
    except StageError as e:
        assert "Nope" in str(e)
    else:
        raise AssertionError("unknown synth object accepted")


def test_direct_light_reaches_texture_space_through_the_projection():
    """Lit images (floor X < 0 directly lit) -> B in the floor atlas, through the textures' own visibility."""
    import tempfile
    from dataclasses import replace
    from pathlib import Path
    import _fusion_fixtures as ff
    from mvb2r.sunpatch_stage import direct_texture
    from mvb2r.visibility import make_view
    ci = ff.two_object_case(W=96, n_views=2, masks=())
    d = Path(tempfile.mkdtemp(prefix="mvb2r_sunpatch_"))
    P, UV, N = ff.dense_quad(n=8)
    np.savez(d / "mesh_001.npz", tri_pos=P, tri_uv=UV, tri_nrm=N)
    views = []
    for cam in ci.cams:
        floor = ff.render_quad(cam, 1, 0.0, 2.0, lambda X, Y: np.repeat((X < 0)[..., None], 3, -1).astype(np.float32))
        card = ff.render_quad(cam, 2, 0.4, 0.6, lambda X, Y: np.zeros(X.shape + (3,), np.float32))
        depth, ids, img = ff.composite([floor, card])
        views.append(make_view(cam, depth, ids, img))
    row = {**ci.objects[0], "file": "mesh_001.npz"}
    ci = replace(ci, views=tuple(views), mesh_dir=d, objects=(row, ci.objects[1]))
    tex, B = direct_texture(ci, row, 64)
    cols = np.nonzero(B)[1]
    assert len(cols) and (cols < 32).mean() > 0.97, (cols < 32).mean()
    left = tex.mask()
    left[:, 32:] = False
    left[12:56] = False                                                     # the card hides this band in both views
    left[:2], left[-2:], left[:, :2] = False, False, False                  # silhouette border: rejected as unseen
    assert B[left].mean() > 0.97, B[left].mean()


def test_outputs_keep_the_unsynthesised_textures_and_clear_on_rerun():
    import tempfile
    from pathlib import Path
    from mvb2r.fusion_io import read_gray01, read_rgb_linear, write_png, write_srgb_png
    from mvb2r.sunpatch_stage import NOSYNTH_DIR, clear_previous, write_outputs
    case = Path(tempfile.mkdtemp(prefix="mvb2r_sunpatch_"))
    t = case / "pbr" / "textures" / "Wall"
    base = np.full((16, 16, 3), 0.5, np.float32)
    write_srgb_png(t / "base_color.png", base)
    write_png(t / "height.png", np.full((16, 16), 128, np.uint8))
    new, synth = base * 0.5, np.zeros((16, 16), bool)
    synth[4:8, 4:8] = True
    write_outputs(case, "Wall", base, new, synth)
    assert abs(read_rgb_linear(case / "pbr" / NOSYNTH_DIR / "Wall" / "base_color.png").mean() - 0.5) < 0.01
    assert abs(read_rgb_linear(t / "base_color.png").mean() - 0.25) < 0.01
    assert read_gray01(t / "synth_mask.png").sum() == 16
    clear_previous(case)
    assert not (case / "pbr" / NOSYNTH_DIR).exists() and not (t / "synth_mask.png").exists()
    assert (t / "base_color.png").is_file()


def _gradient_case():
    """Shaded level rises 0.6 -> 1.4 across x (along the stripes), so sources copied from other x differ in level."""
    clean = stripes() * np.linspace(0.6, 1.4, R, dtype=np.float32)[None, :, None]
    patch = disk(64, 64, 20)
    return clean, patch, lit(clean, patch), dilate(patch, 4)


def test_fill_level_follows_a_smooth_gradient_without_block_steps():
    """One gain per component cannot fit both ends of a gradient; the local level match must."""
    import cv2
    from mvb2r.sunpatch import SynthParams, synthesise
    clean, patch, base, mask = _gradient_case()
    ones = np.ones((R, R), bool)
    out, _, rep = synthesise(base, mask, ones, ones, SynthParams(feather=3))
    lp = lambda x: cv2.GaussianBlur(_lum(x), (0, 0), PERIOD)                 # noqa: E731  (above the stripes)
    core = cv2.erode(patch.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    rel = lp(out)[core] / lp(clean)[core]
    assert np.abs(rel - 1).max() < 0.08, np.abs(rel - 1).max()              # within 8 % (one gain: about 13 %)
    row = lp(out)[64, 50:79] / lp(clean)[64, 50:79]                         # along one row through the fill
    assert np.abs(np.diff(row)).max() < 0.01, np.abs(np.diff(row)).max()    # no block-to-block level step > 1 %
    assert rep["level_residual_after"] < 0.3 * rep["level_residual_before"], rep


def test_single_gain_is_the_fallback_without_surrounding_source():
    from mvb2r.sunpatch import level_match
    clean, patch, base, mask = _gradient_case()
    fill = base * 0.5
    out, res = level_match(fill, base, mask, np.zeros((R, R), bool), sigma=4.0)
    assert out is None and res is None                                     # caller falls back to one gain


def _planted(link_rel: str):
    """A case whose `link_rel` is a symbolic link to a victim folder outside it (with files that must survive)."""
    import os
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp(prefix="mvb2r_sunpatch_sec_"))
    case, victim = root / "case", root / "victim"
    victim.mkdir()
    for f in ("synth_mask.png", "base_color.png", "height.png", "keep.txt"):
        (victim / f).write_bytes(b"victim")
    (case / link_rel).parent.mkdir(parents=True)
    os.symlink(victim, case / link_rel)
    return case, victim


def _refused(fn) -> bool:
    from mvb2r.common import StageError
    try:
        fn()
    except StageError:
        return True
    return False


def test_planted_texture_link_is_refused_and_victim_untouched():
    from mvb2r.sunpatch_stage import clear_previous, write_outputs
    case, victim = _planted("pbr/textures/Wall")
    before = {f.name: f.read_bytes() for f in victim.iterdir()}
    img, synth = np.full((8, 8, 3), 0.5, np.float32), np.ones((8, 8), bool)
    assert _refused(lambda: clear_previous(case))
    assert _refused(lambda: write_outputs(case, "Wall", img, img * 0.5, synth))
    assert {f.name: f.read_bytes() for f in victim.iterdir()} == before


def test_planted_synth_link_is_refused_and_victim_untouched():
    from mvb2r.sunpatch_stage import clear_previous, write_direct_views
    case, victim = _planted("pbr/synth")
    before = {f.name: f.read_bytes() for f in victim.iterdir()}
    assert _refused(lambda: clear_previous(case))
    assert _refused(lambda: write_direct_views(case, ["view_01"], case / "pbr" / "shading",
                                               case / "pbr" / "synth" / "direct_views"))
    assert {f.name: f.read_bytes() for f in victim.iterdir()} == before


def test_linked_source_texture_outside_the_case_is_not_copied():
    import os
    from mvb2r.sunpatch_stage import write_outputs
    case, victim = _planted("elsewhere_link")
    t = case / "pbr" / "textures" / "Wall"
    t.mkdir(parents=True)
    os.symlink(victim / "base_color.png", t / "base_color.png")
    (t / "height.png").write_bytes(b"h")
    img = np.full((8, 8, 3), 0.5, np.float32)
    assert _refused(lambda: write_outputs(case, "Wall", img, img, np.zeros((8, 8), bool)))
    assert (victim / "base_color.png").read_bytes() == b"victim"
    assert not (case / "pbr" / "textures_nosynth" / "Wall" / "base_color.png").exists()


def test_patch_touching_the_chart_edge_is_filled_to_the_edge():
    """Feather only against shaded texels of the same chart: never keep the patch along the chart boundary."""
    from mvb2r.sunpatch import SynthParams, feather_alpha, synthesise
    inside = np.zeros((R, R), bool)
    inside[10:118, 10:118] = True                                           # one chart, empty atlas around it
    base = np.where(inside[..., None], 0.2, 0.0).astype(np.float32) * np.ones(3, np.float32)
    patch = np.zeros((R, R), bool)
    patch[10:30, 40:80] = True                                              # touches the chart's top edge
    base[patch] = 0.6
    mask = dilate(patch, 8) & inside                                        # feather <= dilation, as the stage caps it
    out, _, rep = synthesise(base, mask, inside, inside, SynthParams(feather=8))
    assert rep["components_filled"] == 1
    assert np.abs(out[10, 40:80] - 0.2).max() < 0.02, out[10, 40:80, 0]      # edge row at the filled level
    a = feather_alpha(mask, 8, inside)
    assert a[10, 60] == 1.0 and a[20, 32] <= 1 / 8 + 1e-6 and a[20, 38] > a[20, 33]   # still ramps vs shaded texels


def _strip_case(src_cols):
    """Stripes, a patch over rows 20..99 x cols 10..99 (the mask), shaded observed source only in `src_cols`."""
    clean = stripes()
    mask = np.zeros((R, R), bool)
    mask[20:100, 10:100] = True
    base = lit(clean, mask)
    observed = mask.copy()
    observed[20:100, src_cols] = True
    return clean, base, mask, observed


def test_small_source_tiles_a_larger_patch_and_is_flagged():
    from mvb2r.sunpatch import SynthParams, synthesise
    clean, base, mask, observed = _strip_case(slice(100, 128))             # source about 1/3 of the patch
    out, synth, rep = synthesise(base, mask, observed, np.ones((R, R), bool), SynthParams(feather=0))
    assert rep["components_filled"] == 1 and rep["components_skipped"] == 0, rep
    assert rep["repetition_expected"] is True and 0.25 < rep["source_ratio_min"] < 0.4, rep
    assert np.abs(out[mask] - clean[mask]).mean() < 0.03


def test_source_below_the_absolute_minimum_is_skipped_and_reported():
    from mvb2r.sunpatch import SynthParams, synthesise
    clean, base, mask, observed = _strip_case(slice(125, 128))             # 240 texels < 4 blocks of 8 x 8
    out, synth, rep = synthesise(base, mask, observed, np.ones((R, R), bool), SynthParams(feather=0))
    assert rep["components_filled"] == 0 and rep["components_skipped"] == 1, rep
    assert "minimum" in rep["skipped"][0]["reason"] and np.array_equal(out, base) and not synth.any()


def test_speckle_components_are_dropped_and_counted_not_listed():
    from mvb2r.sunpatch import SynthParams, synthesise
    clean, patch = stripes(), disk(64, 64, 14)
    mask = dilate(patch, 4)
    mask[5:7, 5:7] = True                                                  # a 4-texel speck
    base = lit(clean, patch)
    ones = np.ones((R, R), bool)
    out, synth, rep = synthesise(base, mask, ones, ones, SynthParams(feather=3))
    assert rep["components_dropped_small"] == 1 and rep["components_skipped"] == 0 and rep["skipped"] == []
    assert rep["components_filled"] == 1 and not synth[5:7, 5:7].any()


def _ragged_case():
    """Stripes; patch rows 30..99 x cols 20..89; shaded source = the rest minus random blobs (ragged edges)."""
    import cv2
    rng = np.random.default_rng(3)
    clean = stripes()
    mask = np.zeros((R, R), bool)
    mask[30:100, 20:90] = True
    holes = np.zeros((R, R), np.uint8)
    for y, x in rng.integers(0, R, (25, 2)):
        cv2.circle(holes, (int(x), int(y)), int(rng.integers(2, 6)), 1, -1)
    src = ~mask & (holes == 0)
    return clean, lit(clean, mask), mask, src


def test_quilt_copies_only_from_the_eroded_source_and_varies_neighbour_offsets():
    import cv2
    from mvb2r.sunpatch_quilt import overlap, quilt
    clean, base, mask, src = _ragged_case()
    fill, left, log = quilt(base, mask, src, np.array([1.0, 0.0]), 8, np.random.default_rng(0))
    assert log and not left.any()
    by_pos = {}
    for by, bx, T, dy, dx in log:
        ov = overlap(T)
        er = cv2.erode(src.astype(np.uint8), np.ones((2 * ov + 1,) * 2, np.uint8)) > 0
        ty, tx = np.nonzero(mask[max(by - ov, 0):by + T + ov, max(bx - ov, 0):bx + T + ov])
        ty, tx = ty + max(by - ov, 0), tx + max(bx - ov, 0)
        assert er[ty + dy, tx + dx].all(), (by, bx)                         # never outside the eroded source
        by_pos[(by, bx, T)] = (dy, dx)
    shared = 0
    for (by, bx, T), o in by_pos.items():
        for nb in ((by, bx - T, T), (by - T, bx, T)):
            if nb in by_pos and max(abs(o[0] - by_pos[nb][0]), abs(o[1] - by_pos[nb][1])) < T / 2:
                shared += 1
    assert shared == 0, shared                                             # alternatives exist: never shared
    reused = 0                                                             # nor the same SOURCE window nearby
    for (by, bx, T), o in by_pos.items():
        for dy_, dx_ in ((0, -T), (-T, 0), (-T, -T), (-T, T), (-2 * T, 0), (0, -2 * T)):
            nb = (by + dy_, bx + dx_, T)
            if nb in by_pos:
                s0, s1 = (by + o[0], bx + o[1]), (nb[0] + by_pos[nb][0], nb[1] + by_pos[nb][1])
                reused += max(abs(s0[0] - s1[0]), abs(s0[1] - s1[1])) < T / 2
    assert reused == 0, reused


def test_no_level_step_across_block_boundaries_with_a_ragged_source():
    from mvb2r.sunpatch import SynthParams, block_size, synthesise
    clean, base, mask, src = _ragged_case()
    observed = src | mask
    out, synth, rep = synthesise(base, mask, observed, np.ones((R, R), bool), SynthParams(feather=0))
    assert rep["components_filled"] == 1 and rep["unsourced_texels"] == 0, rep
    T = block_size(R)
    xs = np.arange(21, 90)                                                 # steps along the stripes (x)
    d = np.abs(np.diff(out[32:98, 20:90], axis=1)).mean((0, 2))
    at_boundary = (xs - 20) % T == 0
    assert d[at_boundary].mean() <= 1.2 * d[~at_boundary].mean(), (d[at_boundary].mean(), d[~at_boundary].mean())
    assert np.abs(out[mask] - clean[mask]).mean() < 0.02


def test_quilt_prefers_source_windows_at_the_target_level():
    """A dim source region and a bright one: with a target level, blocks come from the matching region only."""
    from mvb2r.sunpatch_quilt import quilt
    clean = stripes()
    base = clean.copy()
    base[:, 100:] *= 2.5                                                   # bright source region on the right
    mask = np.zeros((R, R), bool)
    mask[30:100, 30:90] = True
    src = ~mask
    src[:, 90:100] = False                                                 # dim region: x < 30, bright: x >= 100
    level = np.full((R, R), float(_lum(clean).mean()), np.float32)
    _, _, log = quilt(base, mask, src, np.array([1.0, 0.0]), 8, np.random.default_rng(0), level=level)
    sx = np.array([bx + dx for _, bx, _, _, dx in log])
    assert len(log) and (sx < 30).all(), sorted(set(sx.tolist()))[-5:]


def test_component_whose_source_fits_no_block_is_left_unchanged_and_reported():
    """Enough source area but no window a block can be copied from: the patch must not become one flat colour
    reported as filled."""
    import numpy as np
    from mvb2r import sunpatch
    R = 32
    rng = np.random.default_rng(0)
    base = (0.2 + 0.02 * rng.standard_normal((R, R, 3))).astype(np.float32)
    mask = np.zeros((R, R), bool)
    mask[10:22, 10:22] = True
    base[mask] = 0.6
    everything = np.ones((R, R), bool)
    out, synth, rep = sunpatch.synthesise(base, mask, everything, everything, sunpatch.SynthParams())
    assert 0 < sunpatch.MAX_UNSOURCED_FRACTION < 1
    assert rep["components_filled"] == 0 and not synth.any() and np.array_equal(out, base), rep
    assert rep["components_skipped"] == 1 and "no source window" in rep["skipped"][0]["reason"], rep

