"""Per-object image gates: the legacy policy pinned bit-for-bit, the strict policy rule by rule (synthetic)."""
import numpy as np

from _fusion_fixtures import two_object_case

# digests of today's gates (view_02 of two_object_case), pinned BEFORE the strict policy was added
LEGACY = {
    ("Card", 0, "Floor"): (None, (1560, 12757680), (15793.2, 129.38)),
    ("Card", 0, "Card"): ((870, 7114860), None, None),
    ("Card", 1, "Floor"): (None, (1892, 15472776), (15616.0, 127.93)),
    ("Card", 1, "Card"): ((1260, 10304280), None, None),
    ("Card+Floor", 0, "Floor"): ((6360, 52027044), None, None),
    ("Card+Floor", 0, "Card"): ((870, 7114860), None, None),
    ("Card+Floor", 1, "Floor"): ((7800, 63805512), None, None),
    ("Card+Floor", 1, "Card"): ((1260, 10304280), None, None),
}
LEGACY_OPTS = [{}, {"gate_dilate": 2, "reject_dilate": 6, "halo_px": 10, "halo_factor": 0.5}]
# the strict margins of a 1664 px camera, given explicitly (explicit values are used as given at any size);
# the fixture cameras are 128 px, where the defaults shrink to their 1 px floor
STRICT_REF = {"gate_policy": "strict", "erode_own": 2, "erode_gen": 2, "other_mask_dilate": 2, "reject_dilate": 8,
              "occ_margin": 8, "occ_soft": 20, "occ_margin_masked": 2, "occ_soft_masked": 6}


def _dig(a):
    if a is None:
        return None
    if a.dtype == bool:
        return (int(a.sum()), int(np.flatnonzero(a).sum()))
    return (round(float(a.sum()), 2), round(float((a.ravel() * np.arange(a.size)).sum() / 1e6), 2))


def _obj(ci, name):
    return next(o for o in ci.objects if o["name"] == name)


def test_legacy_policy_reproduces_pinned_gates():
    from mvb2r.projinputs import gates_for
    for masks in (("Card",), ("Card", "Floor")):
        ci = two_object_case(masks=masks)
        for k, opt in enumerate(LEGACY_OPTS):
            for o in ci.objects:
                g = gates_for(ci, o, 1, {**opt, "gate_policy": "legacy"})
                got = (_dig(g.accept), _dig(g.reject), _dig(g.score_mult))
                assert got == LEGACY[("+".join(masks), k, o["name"])], (masks, k, o["name"], got)


def test_strict_is_the_default_policy():
    from mvb2r.projinputs import fusion_options
    _, _, opt = fusion_options({})
    assert opt["gate_policy"] == "strict"


def test_strict_masked_object_erodes_own_and_rejects_others():
    from mvb2r.projinputs import gates_for
    ci = two_object_case(masks=("Card", "Floor"), shift=0)
    card, floor = _obj(ci, "Card"), _obj(ci, "Floor")
    g = gates_for(ci, card, 1, STRICT_REF)
    own = ci.views[1].ids == 2
    assert g.accept is not None and not (g.accept & ~own).any()
    assert g.accept.sum() < own.sum() and g.accept.sum() > 0.6 * own.sum()   # eroded 2 px, not emptied
    gf = gates_for(ci, floor, 1, STRICT_REF)
    # floor pixels within 2 px of the card's generated mask are rejected
    import cv2
    near_card = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert not (gf.accept & near_card).any()


def test_strict_unmasked_object_rejects_other_masks_dilated_8():
    from mvb2r.projinputs import gates_for
    import cv2
    ci = two_object_case(masks=("Card",), shift=0)
    g = gates_for(ci, _obj(ci, "Floor"), 1, STRICT_REF)
    card = ci.masks[("Card", 1)]
    ring8 = cv2.dilate(card.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    assert not (g.accept & ring8).any()
    ring12 = cv2.dilate(card.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    assert (g.accept & ring12 & ~ring8).any() or (g.score_mult[ring12 & ~ring8] < 1).any()
    g4 = gates_for(ci, {**_obj(ci, "Floor"), "reject_dilate": 4}, 1, {**STRICT_REF, "occ_margin": 0})
    ring4 = cv2.dilate(card.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert (g4.accept & ring8 & ~ring4).any()            # per-object override narrows the rejection


def test_strict_occluder_margin_and_soft_band():
    from mvb2r.projinputs import gates_for
    ci = two_object_case(masks=(), shift=0)                 # no generated masks: only the occluder rule acts
    g = gates_for(ci, _obj(ci, "Floor"), 1, STRICT_REF)
    ids = ci.views[1].ids
    import cv2
    card = (ids == 2).astype(np.uint8)
    d = cv2.distanceTransform(1 - card, cv2.DIST_L2, 5)
    floor = ids == 1
    assert not (g.accept & floor & (d <= 7)).any()          # hard margin 8 px (square kernel >= disc)
    assert (g.accept & floor & (d > 12) & (d < 19)).any()
    soft = g.score_mult[floor & (d > 12) & (d < 19)]
    assert np.allclose(soft, 0.3)
    assert np.allclose(g.score_mult[floor & (d > 40)], 1.0)
    # the card is NEARER than the floor; the floor is not nearer than the card -> no margin on the card
    gc = gates_for(ci, _obj(ci, "Card"), 1, STRICT_REF)
    assert np.allclose(gc.score_mult[ids == 2], 1.0)


def test_strict_defaults_scale_with_the_camera_both_ways_with_a_floor():
    from types import SimpleNamespace
    from mvb2r.gating import STRICT_MIN_PX, STRICT_PX, strict_px
    cam = lambda s: SimpleNamespace(width=s, height=s)  # noqa: E731
    for key, ref in STRICT_PX.items():
        assert strict_px({}, key, cam(1664)) == ref
        assert strict_px({}, key, cam(3328)) == 2 * ref                    # larger camera: larger margin
        assert strict_px({}, key, cam(832)) == max(ref / 2, STRICT_MIN_PX)   # smaller camera: smaller margin
        assert strict_px({}, key, cam(64)) == STRICT_MIN_PX                  # never below the floor
        assert strict_px({key: 5}, key, cam(64)) == 5                      # case value: used as given
        assert strict_px({key: 5}, key, cam(64), {key: 3}) == 3            # object value wins
    assert STRICT_MIN_PX == 1.0


def test_thin_object_erosion_falls_back():
    from mvb2r.gating import thin_safe_erosion
    m = np.zeros((40, 40), bool)
    m[10:13, 5:35] = True          # 3 px thick: 2 px erosion empties it, 1 px keeps a third
    assert thin_safe_erosion(m, 2) == 0
    m2 = np.zeros((40, 40), bool)
    m2[10:15, 5:35] = True         # 5 px: 2 px keeps 1/5 (<half) -> 1 px keeps 3/5
    assert thin_safe_erosion(m2, 2) == 1
    m3 = np.zeros((40, 40), bool)
    m3[5:35, 5:35] = True
    assert thin_safe_erosion(m3, 2) == 2


def test_gate_options_validated():
    from mvb2r.fusion_io import StageError
    from mvb2r.projinputs import fusion_options
    bad = [{"gate_policy": "loose"}, {"occ_margin": -1}, {"occ_dz": float("nan")}, {"erode_own": "2"},
           {"occ_soft_factor": 1.5}]
    for f in bad:
        try:
            fusion_options({"fusion": f})
            raise AssertionError(f"accepted {f}")
        except StageError:
            pass
    _, _, opt = fusion_options({"fusion": {"gate_policy": "legacy", "occ_margin": 4}})
    assert opt["gate_policy"] == "legacy" and opt["occ_margin"] == 4.0


def _meshed(**kw):
    import tempfile
    from _fusion_fixtures import with_meshes
    return with_meshes(two_object_case(n_views=3, masks=("Card",), shift=1), tempfile.mkdtemp(prefix="mvb2r_gate_"),
                       **kw)


def test_strict_detail_critical_gets_coherent_regions_and_loo_recomputes_them():
    from mvb2r.projinputs import object_input
    from mvb2r.visibility import VisParams
    ci = _meshed(detail_critical=True)
    card = _obj(ci, "Card")
    oi = object_input(ci, card, 32, VisParams(), {"gate_policy": "strict"})
    assert len(oi.regions) == 1 and oi.regions[0]["cut"] == "feather" and oi.regions[0]["dominant"]
    dom = oi.regions[0]["views"][0]
    held = oi.held_out(dom)
    assert not held.st.valid[dom].any() and held.regions[0]["views"][0] != dom
    assert held.st.valid.any()                       # another camera takes over the region
    leg = object_input(ci, card, 32, VisParams(), {"gate_policy": "legacy"})
    assert "cut" not in leg.regions[0] and leg.regionize is not None


def test_loo_evaluation_stack_ignores_gate_policy_and_gate_overrides():
    from mvb2r.projinputs import evaluation_stack, object_input
    from mvb2r.visibility import VisParams
    ci = _meshed()
    custom = {"gate_policy": "legacy", "reject_dilate": 12, "gate_dilate": 3, "halo_px": 15}
    for o in ci.objects:
        _, a = evaluation_stack(ci, o, 32, VisParams(), STRICT_REF)
        _, b = evaluation_stack(ci, {**o, "reject_dilate": 20, "gate_dilate": -4}, 32, VisParams(), custom)
        ref = object_input(ci, o, 32, VisParams(), {"gate_policy": "legacy"}).st     # legacy DEFAULTS
        strict = object_input(ci, o, 32, VisParams(), STRICT_REF).st
        assert np.array_equal(a.valid, b.valid) and np.array_equal(a.valid, ref.valid)
        assert np.array_equal(a.color, ref.color)
        assert strict.valid.sum() < ref.valid.sum()        # the strict gate itself does reject more
        moved = object_input(ci, {**o, "reject_dilate": 20, "gate_dilate": -4}, 32, VisParams(), custom).st
        assert not np.array_equal(moved.valid, ref.valid)  # the overrides do change FUSION gates


def test_region_prefer_view_on_unusable_view_falls_back():
    from mvb2r.fusion_io import StageError
    from mvb2r.handover import parse_regions
    from mvb2r.projinputs import usable_region_specs
    specs = parse_regions([{"name": "r", "sphere": [0, 0, 0, 1], "prefer_view": "view_03"},
                           {"name": "s", "sphere": [0, 0, 0, 1], "prefer_view": "view_01"}], "Card",
                          ["view_01", "view_02", "view_03"])
    out, notes = usable_region_specs("Card", specs, ["view_01", "view_02"])
    assert out[0]["prefer_view"] is None and out[1]["prefer_view"] == "view_01"
    assert specs[0]["prefer_view"] == "view_03"                 # input untouched
    assert len(notes) == 1 and "view_03" in notes[0] and "\n" not in notes[0]
    try:
        parse_regions([{"name": "r", "sphere": [0, 0, 0, 1], "prefer_view": "view_09"}], "Card",
                      ["view_01", "view_02", "view_03"])
        raise AssertionError("unknown camera accepted")
    except StageError:
        pass


def test_ignored_settings_are_reported():
    from mvb2r.projinputs import ignored_settings
    objs = [{"name": "A", "detail_critical": False, "regions": ({"name": "r"},)},
            {"name": "B", "detail_critical": True, "regions": ({"name": "f", "handover_mm": 12.0, "cut": "hard"},)}]
    notes = ignored_settings(objs, {"gate_policy": "legacy"}, contacts=({"surface": "A"},))
    text = " | ".join(notes)
    assert "A" in text and "detail_critical" in text and "contacts" in text and "handover" in text
    assert ignored_settings(objs[1:], {"gate_policy": "strict"}, contacts=()) == []


def test_every_method_runs_on_strict_coherent_regions():
    from mvb2r.fusion import METHODS, FusionParams, fuse
    from mvb2r.projinputs import object_input, object_params
    from mvb2r.visibility import VisParams
    ci = _meshed(detail_critical=True)
    card = _obj(ci, "Card")
    opt = {"gate_policy": "strict"}
    oi = object_input(ci, card, 32, VisParams(), opt)
    dom = oi.regions[0]["views"][0]
    for m in METHODS:
        r = fuse(oi, m, object_params(FusionParams(), card, opt))
        mine = oi.st.valid[dom]
        assert (r.source[mine] == dom).all(), m
        other = r.observed & ~mine & ~r.extra.get("soft", np.zeros(len(mine), bool))
        assert not (other & oi.regions[0]["mask"] & (r.source == dom)).any(), m
