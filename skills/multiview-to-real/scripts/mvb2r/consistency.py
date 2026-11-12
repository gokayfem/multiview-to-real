"""`consistency`: projector-free cross-view score of a set of view images, per object class.

    mvb2r.py consistency CASE [--set NAME=PATTERN ...] [--views view_01,view_02] [--out FILE]

PATTERN contains {view} (e.g. targets/{view}.png); default: the frozen targets. For every ordered pair
(A, B) each valid pixel q of B is mapped into A through B's position_world and A's P, and kept only if it
lands inside A, is visible in A (|depth_A - Zc| < 1 cm + 1 %), has the same object id in both views, is not
within EDGE px of an object_id boundary, has align_conf >= CONF_MIN in both views (when targets exist), and
its object is not an `opening`. A is sampled bilinearly at the mapped point. Bands, per object class (case.json
"class"), normalised convolution inside the class's valid region:
  low  = Gaussian sigma SIG_LO          -> lighting / colour agreement: dE76 (Lab) of the low bands
  high = G(sigma 1) - G(sigma SIG_LO)   -> do invented fine details line up: NCC (1 = same detail, 0 = unrelated)
  full = dE76 of the unfiltered pair
Generated views are NOT photo-consistent; this number says how far from it they are.
Pixel scales are set at 1664 px (EDGE 4, SIG_LO 8) and scaled to the calibration width.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import common

CONF_MIN = 200
EDGE_AT_1664, SIG_LO_AT_1664, SIG_HI = 4, 8.0, 1.0


def scales(width: int) -> tuple[int, float, int]:
    edge = max(1, int(round(EDGE_AT_1664 * width / 1664)))
    sig_lo = max(2.0, SIG_LO_AT_1664 * width / 1664)
    return edge, sig_lo, max(30, int(400 * (width / 1664) ** 2))


def interior(oid: np.ndarray, edge: int) -> np.ndarray:
    import cv2
    k = np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)
    o = oid.astype(np.float32)
    return (cv2.erode(o, k) == o) & (cv2.dilate(o, k) == o)


def masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * (m[..., None] if img.ndim == 3 else m), (0, 0), sigma)
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-4)
    return num / (den[..., None] if img.ndim == 3 else den)


def pair_map(pw_b, valid_b, P_a, depth_a, oid_a, oid_b, valid_a):
    """Remap of A onto B's pixels: (mapx, mapy, ok) (pure)."""
    H, W = oid_b.shape
    X = pw_b.reshape(-1, 3).astype(np.float64)
    p = np.c_[X, np.ones(len(X))] @ np.asarray(P_a, float).T
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = p[:, 0] / z, p[:, 1] / z
    Ha, Wa = oid_a.shape
    ok = valid_b.reshape(-1) & (z > 0.05) & (u >= 0.5) & (u < Wa - 1.5) & (v >= 0.5) & (v < Ha - 1.5)
    iu = np.clip(np.nan_to_num(u), 0, Wa - 1).astype(int)
    iv = np.clip(np.nan_to_num(v), 0, Ha - 1).astype(int)
    ok &= np.abs(depth_a[iv, iu] - z) < 0.01 + 0.01 * np.abs(z)
    ok &= oid_a[iv, iu] == oid_b.reshape(-1)
    ok &= valid_a[iv, iu]
    mapx = np.nan_to_num(u - 0.5).astype(np.float32).reshape(H, W)
    mapy = np.nan_to_num(v - 0.5).astype(np.float32).reshape(H, W)
    return mapx, mapy, ok.reshape(H, W)


class Geometry:
    """Valid pixels and pair maps of a set of views (from the exact passes)."""

    def __init__(self, case, cams: list[dict], objects: list[dict], use_conf: bool = True):
        self.cams = {c["name"]: c for c in cams}
        W = int(cams[0]["width"])
        self.edge, self.sig_lo, self.min_px = scales(W)
        self.classes = {}
        for o in objects:
            if not o.get("opening"):
                self.classes.setdefault(o.get("class") or "generic", []).append(int(o["id"]))
        ids = np.array(sorted(i for v in self.classes.values() for i in v))
        self.vp = {n: common.view_passes(case, c) for n, c in self.cams.items()}
        self.valid = {}
        for n, vp in self.vp.items():
            ok = interior(vp["oid"], self.edge) & np.isin(vp["oid"], ids)
            conf = common.case_dir(case) / "targets" / f"align_conf_{common.view_index(n):02d}.png"
            if use_conf and conf.is_file():
                ok &= common.read_gray(conf) >= CONF_MIN
            self.valid[n] = ok
        self.pair_keys = list(itertools.permutations(self.cams, 2))

    def pair(self, a: str, b: str):
        """(mapx, mapy, ok) of A sampled onto B, computed on demand (one pair in memory at a time)."""
        A, B = self.vp[a], self.vp[b]
        return pair_map(B["pos"], self.valid[b], self.cams[a]["P"], A["depth"], A["oid"], B["oid"], self.valid[a])

    def overlap(self) -> dict:
        """Mapped pixel count per unordered pair."""
        out = {}
        for a, b in self.pair_keys:
            key = tuple(sorted((a, b)))
            out[key] = out.get(key, 0) + int(self.pair(a, b)[2].sum())
        return out


def _class_stats(lb, la, gb, ga, m, sig_lo) -> dict:
    lo_b, lo_a = masked_blur(lb, m, sig_lo), masked_blur(la, m, sig_lo)
    hb = masked_blur(gb, m, SIG_HI) - masked_blur(gb, m, sig_lo)
    ha = masked_blur(ga, m, SIG_HI) - masked_blur(ga, m, sig_lo)
    x, y = ha[m].astype(np.float64), hb[m].astype(np.float64)
    return {"n": int(m.sum()), "dE_full": float(np.linalg.norm(la - lb, axis=2)[m].mean()),
            "dE_low": float(np.linalg.norm(lo_a - lo_b, axis=2)[m].mean()),
            "xy": float((x * y).sum()), "xx": float((x * x).sum()), "yy": float((y * y).sum())}


def _crop(m: np.ndarray, pad: int):
    ys, xs = np.nonzero(m)
    return slice(max(0, ys.min() - pad), ys.max() + pad + 1), slice(max(0, xs.min() - pad), xs.max() + pad + 1)


def compare(geo: Geometry, imgs: dict) -> dict:
    """Per-class and overall low/high band agreement of a set of images {view: uint8 RGB}."""
    import cv2
    from skimage.color import rgb2lab
    lab = {n: rgb2lab(im).astype(np.float32) for n, im in imgs.items()}
    gray = {n: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) for n, im in imgs.items()}
    acc, per_pair, per_view = {}, {}, {}
    for a, b in geo.pair_keys:
        if a not in imgs or b not in imgs:
            continue
        mx, my, ok = geo.pair(a, b)
        wl = np.dstack([cv2.remap(lab[a][..., i], mx, my, cv2.INTER_LINEAR) for i in range(3)])
        wg = cv2.remap(gray[a], mx, my, cv2.INTER_LINEAR)
        pp = {}
        for cls, ids in geo.classes.items():
            m_full = ok & np.isin(geo.vp[b]["oid"], ids)
            if m_full.sum() < geo.min_px:
                continue
            sl = _crop(m_full, int(3 * geo.sig_lo) + 4)
            s = _class_stats(lab[b][sl], wl[sl], gray[b][sl], wg[sl], m_full[sl], geo.sig_lo)
            pp[cls] = {"n": s["n"], "dE_full": round(s["dE_full"], 3), "dE_low": round(s["dE_low"], 3),
                       "ncc_high": round(s["xy"] / np.sqrt(s["xx"] * s["yy"] + 1e-9), 4)}
            for key, store in ((cls, acc), (a, per_view), (b, per_view)):
                A = store.setdefault(key, {"n": 0, "dE_full": 0.0, "dE_low": 0.0, "xy": 0.0, "xx": 0.0, "yy": 0.0})
                A["n"] += s["n"]
                for k in ("dE_full", "dE_low"):
                    A[k] += s[k] * s["n"]
                for k in ("xy", "xx", "yy"):
                    A[k] += s[k]
        per_pair[f"{a}->{b}"] = pp
    views = {v: r for v, r in summarise(per_view).items() if v != "ALL"}
    return {"summary": summarise(acc), "per_view": views, "per_pair": per_pair}


def summarise(acc: dict) -> dict:
    out = {}
    for cls, A in acc.items():
        if A["n"]:
            out[cls] = {"n": A["n"], "dE_full": round(A["dE_full"] / A["n"], 3), "dE_low": round(A["dE_low"] / A["n"], 3),
                        "ncc_high": round(A["xy"] / np.sqrt(A["xx"] * A["yy"] + 1e-9), 4)}
    tot = sum(A["n"] for A in acc.values())
    if tot:
        out["ALL"] = {"n": tot, "dE_full": round(sum(A["dE_full"] for A in acc.values()) / tot, 3),
                      "dE_low": round(sum(A["dE_low"] for A in acc.values()) / tot, 3),
                      "ncc_high": round(sum(A["xy"] for A in acc.values()) /
                                        np.sqrt(sum(A["xx"] for A in acc.values()) * sum(A["yy"] for A in acc.values()) + 1e-9), 4)}
    return out


def load_set(case, pattern: str, views) -> dict:
    case = common.case_dir(case)
    out = {}
    for v in views:
        p = case / pattern.format(view=v)
        out[v] = common.read_rgb(common.require(p, "a stage that writes " + pattern))
    return out


def usable_cams(case, cams: list[dict]) -> list[dict]:
    """Views whose frozen target is usable (targets.json views.<view>.usable is not false)."""
    p = common.case_dir(case) / "targets" / "targets.json"
    if not p.is_file():
        return cams
    t = common.read_json(p).get("views", {})
    keep = [c for c in cams if t.get(c["name"], {}).get("usable", True) is not False]
    dropped = [c["name"] for c in cams if c not in keep]
    if dropped:
        common.say(f"[consistency] WARNING skipping unusable view(s) {', '.join(dropped)} (targets.json usable: false)")
    return keep


def register(sub) -> None:
    p = sub.add_parser("consistency", help="cross-view reprojection score: low band (light/colour), high band (detail)")
    p.add_argument("case")
    p.add_argument("--set", action="append", default=None, help="NAME=PATTERN with {view}; default targets/{view}.png")
    p.add_argument("--views", default=None)
    p.add_argument("--out", default="gen/qa/consistency.json")
    p.set_defaults(func=run)


def run(args) -> int:
    case = common.case_dir(args.case)
    cams = common.load_cameras(case)
    if args.views:
        want = set(v.strip() for v in args.views.split(","))
        cams = [c for c in cams if c["name"] in want]
    elif not args.set:
        cams = usable_cams(case, cams)
    geo = Geometry(case, cams, common.case_objects(case))
    sets = args.set or ["targets=targets/{view}.png"]
    res = {"method": __doc__, "views": [c["name"] for c in cams], "sets": {}}
    for spec in sets:
        if "=" not in spec or "{view}" not in spec:
            raise common.StageError(f"--set {spec!r}: expected NAME=PATTERN containing {{view}}")
        name, pat = spec.split("=", 1)
        r = compare(geo, load_set(case, pat, [c["name"] for c in cams]))
        res["sets"][name] = {"pattern": pat, **r}
        common.say(f"[consistency] {name}:")
        for cls, s in sorted(r["summary"].items()):
            common.say(f"  {cls:<12} n={s['n']:>8}  dE_low {s['dE_low']:6.2f}  dE_full {s['dE_full']:6.2f}  ncc_high {s['ncc_high']:.3f}")
        common.say("  per view (pairs it takes part in):")
        for v, s in sorted(r["per_view"].items()):
            common.say(f"  {v:<12} n={s['n']:>8}  dE_low {s['dE_low']:6.2f}  dE_full {s['dE_full']:6.2f}  ncc_high {s['ncc_high']:.3f}")
    out = case / args.out
    common.write_json(out, res)
    return 0
