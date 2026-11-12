"""Cross-view consistency on exact synthetic passes: identical content agrees, view-specific detail does not."""
import numpy as np

from _util import synthetic_case
from mvb2r import common, consistency, mockgen, resync


def _set(case, cams, amp, seed0=0):
    out = {}
    for i, c in enumerate(cams):
        lin = mockgen.view_truth(case, c["name"])
        if amp:
            lin = mockgen.defect(lin, np.random.default_rng(seed0 + i), amp)
        out[c["name"]] = (common.linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
    return out


def test_truth_is_consistent_and_defects_are_not():
    case = synthetic_case(3, 160)
    cams = common.load_cameras(case)
    geo = consistency.Geometry(case, cams, common.case_objects(case), use_conf=False)
    truth = consistency.compare(geo, _set(case, cams, 0))["summary"]["ALL"]
    noisy = consistency.compare(geo, _set(case, cams, 1.0))["summary"]["ALL"]
    milder = consistency.compare(geo, _set(case, cams, 0.4))["summary"]["ALL"]
    assert truth["n"] > 1000
    assert truth["ncc_high"] > 0.6 and truth["dE_low"] < 2.0, truth
    assert noisy["ncc_high"] < milder["ncc_high"] < truth["ncc_high"], (noisy, milder, truth)
    assert noisy["dE_low"] > truth["dE_low"]
    ov = geo.overlap()
    assert len(ov) == 3 and max(ov.values()) > 0
    return f"ncc_high truth {truth['ncc_high']:.3f} / mild {milder['ncc_high']:.3f} / full defects {noisy['ncc_high']:.3f}"


def test_pair_map_rejects_occluded_and_foreign_pixels():
    case = synthetic_case(2, 96)
    cams = common.load_cameras(case)
    vp = {c["name"]: common.view_passes(case, c) for c in cams}
    a, b = cams[0]["name"], cams[1]["name"]
    valid = {n: v["oid"] > 0 for n, v in vp.items()}
    mx, my, ok = consistency.pair_map(vp[b]["pos"], valid[b], cams[0]["P"], vp[a]["depth"], vp[a]["oid"], vp[b]["oid"], valid[a])
    iu, iv = (mx[ok] + 0.5).astype(int), (my[ok] + 0.5).astype(int)
    assert (vp[a]["oid"][np.clip(iv, 0, 95), np.clip(iu, 0, 95)] == vp[b]["oid"][ok]).mean() > 0.99
    assert ok.sum() < valid[b].sum()


def test_resync_gate():
    rule = {"min_ncc_high_gain": 0.0, "max_dE_low_increase": 1.0, "iou_drop_max": 0.03}
    b = {"summary": {"ALL": {"ncc_high": 0.3, "dE_low": 3.0}}}
    better = {"summary": {"ALL": {"ncc_high": 0.5, "dE_low": 3.2}}}
    q = {"view_01": {"a": {"iou": 0.9, "reliability": "high"}}}
    assert resync.gate(rule, b, better, q, q, ["high"])["pass"]
    assert not resync.gate(rule, b, b, q, q, ["high"])["pass"]
    worse_align = {"view_01": {"a": {"iou": 0.8, "reliability": "high"}}}
    g = resync.gate(rule, b, better, q, worse_align, ["high"])
    assert not g["pass"] and "IoU" in g["reasons"][0]
    assert not resync.gate(rule, b, {"summary": {"ALL": {"ncc_high": 0.5, "dE_low": 5.0}}}, q, q, ["high"])["pass"]
    assert not resync.gate(rule, {"summary": {}}, better, q, q, ["high"])["pass"]
