"""Stage `project`: visibility-aware multiview fusion into each object's UV atlas.

    mvb2r.py project CASE [--method auto|best|blend|regions|freqsplit|multiband|graphcut|register|all|a,b]
                          [--no-render] [--images DIR] [--out-name proj]

Writes proj/<method>/<object>/{texture,source,source_raw,confidence,coverage,unobserved,soft_observed}.png,
proj/rejection/<object>/ (reason maps, flags.npz), proj/coverage.json (observed / soft / inferred area fractions
per object), proj/project.json; then (unless --no-render) proj/passes/ (exact UV / id / depth passes of the
source and novel cameras), proj/<method>/rerender/view_XX.png (consensus re-renders, consumed by `resync`),
proj/<method>/novel/novel_XX.png and out/diagnostic.blend (emission, baked lighting, method switch).
The projection is written to a temporary folder and swapped in only when it completed, so a crash never mixes
new textures with an old coverage.json. Texture size per object: mvb2r/texsize.py (printed).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from mvb2r.coverage import (area_fractions, contributions, donor_colours, fill_texture, write_method_maps,
                            write_rejections)
from mvb2r.fill import MODES as FILL_MODES, fill_reference_for, reference_colour
from mvb2r.fusion import AUTO_EVIDENCE, LETTERS, METHODS, consistency, fuse, resolve_method
from mvb2r.fusion_io import (StageError, case_json, linear_to_srgb, read_rgb_linear, resolve_case, safe_name,
                             write_json, write_srgb_png)
from mvb2r.fusion_io import load_mesh
from mvb2r.projinputs import fusion_options, load_case_inputs, object_input, object_params
from mvb2r.raster import texel_set
from mvb2r.zones import MM, RAMP_IN, RAMP_OUT, contact_fill, contact_weights, occluder_points
from mvb2r.texsize import choose_size
from mvb2r.seams import texture_seam_metrics
from mvb2r.visibility import SCORE_FORMULA, VisParams

DEFAULT_FILL = FILL_MODES[0]


def register(subparsers) -> None:
    p = subparsers.add_parser("project", help="multiview fusion into UV textures, diagnostic.blend, re-renders")
    p.add_argument("case")
    p.add_argument("--method", default=None, help="auto (default from case.json), a registry name, a,b or all")
    p.add_argument("--images", default=None, help="project these images (view_XX.png) instead of targets/")
    p.add_argument("--out-name", default="proj", help="output folder name inside the case (default proj; one plain name)")
    p.add_argument("--no-render", action="store_true", help="skip passes, re-renders and diagnostic.blend")
    p.set_defaults(func=run)


def method_list(arg: str | None, cfg: dict) -> list[str]:
    raw = arg or (cfg.get("fusion", {}) or {}).get("method", "auto")
    names = list(METHODS) if raw == "all" else [LETTERS.get(m.strip(), m.strip()) for m in str(raw).split(",")
                                                if m.strip()]
    for m in names:
        resolve_method(m)  # validates
    return names


def run(args) -> int:
    case = resolve_case(args.case)
    cfg = case_json(case)
    t0 = time.time()
    out = case / safe_name(args.out_name)
    images = Path(args.images).resolve() if args.images else None
    methods = method_list(args.method, cfg)
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        record = project_case(case, cfg, methods, tmp, images)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    swap_in(tmp, out)
    print(f"[project] {len(record['objects'])} objects, methods {record['methods']} in {time.time() - t0:.0f}s")
    if not args.no_render:
        from mvb2r.render import after_project
        after_project(case, cfg, out, record["methods"])
    return 0


def swap_in(tmp: Path, out: Path, carry=("_mesh", "logs")) -> None:
    """Replace `out` by the completed `tmp`, keeping caches that live in it (mesh export, logs)."""
    for name in carry:
        if (out / name).exists() and not (tmp / name).exists():
            shutil.move(str(out / name), str(tmp / name))
    old = out.with_name(out.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    shutil.rmtree(old, ignore_errors=True)


def project_case(case: Path, cfg: dict, methods: list[str], out: Path, images: Path | None = None) -> dict:
    prm, by_class, opt = fusion_options(cfg)
    vis = VisParams()
    ci = load_case_inputs(case, cfg, images)
    names = [c.name for c in ci.cams]
    cov = {"score_formula": SCORE_FORMULA, "gate_policy": opt["gate_policy"], "fill": opt.get("fill", DEFAULT_FILL), "views": names, "skipped_views": list(ci.skipped), "methods": methods, "auto_evidence": AUTO_EVIDENCE,
           "fill_rule": "same object only: pull-push near observed texels, fading to a plain reference colour "
                        "farther away (the median; for plain wall-like classes the well-lit median). "
                        "INFERRED, no invented detail", "objects": {}}
    textures: dict = {m: {} for m in methods}
    for obj in ci.objects:
        t1 = time.time()
        if obj["opening"]:
            cov["objects"][obj["name"]] = {"id": obj["id"], "class": obj["class"], "opening": True,
                                           "projected": False, "note": "opening: never projected onto (neutral sky)"}
            continue
        rec, tex_by_method = project_object(ci, obj, methods, prm, by_class, vis, opt, out)
        cov["objects"][obj["name"]] = rec
        for m in methods:
            textures[m][obj["name"]] = tex_by_method[m]
        print(f"[project] {obj['name']}: R={rec['tex_size']} observed={rec['coverage']['observed']:.3f} "
              f"soft={rec['coverage']['soft']:.3f} ({time.time() - t1:.1f}s)", flush=True)
    apply_donor_fill(cov, textures, out)
    if opt.get("gate_policy") == "strict":
        apply_contact_fill(ci, cov, textures, out)
    write_json(out / "coverage.json", cov)
    write_json(out / "project.json", {"methods": methods, "vis_params": vis.to_dict(), "fusion_params": prm.to_dict(),
                                      "by_class": by_class, "options": opt,
                                      "images": str(images) if images else "targets/"})
    return cov


def project_object(ci, obj, methods, prm, by_class, vis, opt, out: Path):
    size = {"tex_size": int(opt["tex_size"]), "rule": "case.json fusion.tex_size"} if opt.get("tex_size") else \
        choose_size(ci, obj, vis, opt)
    R = size["tex_size"]
    ppt = size.get("best_view_ppt_at_size")
    print(f"[project] {obj['name']}: texture {R}x{R}" + (f" ({ppt:.2f} source px per texel in {size['best_view']}"
          f"{', clamped' if size.get('clamped') else ''})" if ppt else ""), flush=True)
    oi = object_input(ci, obj, R, vis, opt)
    st, tex = oi.st, oi.tex
    sn = safe_name(obj["name"])
    rec = {"id": obj["id"], "class": obj["class"], "detail_critical": obj["detail_critical"], "tex_size": R, "tex_size_rule": size,
           "texels": tex.n, "opening": False, "projected": True,
           "single_source_regions": [region_record(r, st.view_names) for r in oi.regions],
           "inconsistent_sample_frac": float(consistency(st, prm).sum() / max(st.valid.sum(), 1)), "methods": {}}
    rec.update(write_rejections(out / "rejection" / sn, tex, st))
    tex_by_method = {}
    for m in methods:
        concrete = resolve_method(m, obj["class"], obj["detail_critical"], by_class)
        res = fuse(oi, concrete, object_params(prm, obj, opt))
        full, observed_med = object_fill(tex, res.color, res.observed, obj["class"], opt.get("fill", DEFAULT_FILL))
        write_method_maps(out / m / sn, tex, res, full)
        soft = res.extra.get("soft", np.zeros(tex.n, bool))
        rec["coverage"] = area_fractions(tex, res.observed, soft)
        rec["methods"][m] = {"resolved": concrete, "contribution_area_frac": contributions(tex, res, st.view_names),
                             "texture_seams": texture_seam_metrics(tex, linear_to_srgb(res.color), res.source,
                                                                   res.observed & ~soft),
                             "info": {k: v for k, v in res.extra.items() if k in ("info", "register_median_px")}}
        tex_by_method[m] = None if observed_med is None else observed_med.tolist()
    return rec, tex_by_method


def object_fill(tex, color: np.ndarray, observed: np.ndarray, cls: str, mode: str):
    """(filled (R,R,3) texture, colour the object donates to unobserved same-class objects or None).
    "blocks" reproduces the previous release exactly: plain median, no well-lit reference or lift, for every
    class; "smooth" uses the class-aware reference (fill.fill_reference_for)."""
    reference, lift = ("median", False) if mode == "blocks" else fill_reference_for(cls)
    full = fill_texture(tex, color, observed, mode=mode, reference=reference, lift=lift)
    med = reference_colour(color[observed], reference) if observed.any() else None
    return full, med


def apply_donor_fill(cov: dict, textures: dict, out: Path, min_obs: float = 0.05) -> None:
    """Objects (almost) never observed take the median colour of observed same-class objects (INFERRED)."""
    recs = {n: r for n, r in cov["objects"].items() if r.get("projected")}
    for m, meds in textures.items():
        donors = donor_colours(recs, {n: c for n, c in meds.items() if c is not None}, min_obs)
        for n, r in recs.items():
            seen = r["coverage"]["observed"] + r["coverage"]["soft"]
            if seen >= min_obs or r["class"] not in donors:
                continue
            d = out / m / safe_name(n)
            img = read_rgb_linear(d / "texture.png")
            obs = read_rgb_linear(d / "coverage.png")[..., 0] > 0.5
            col = np.asarray(donors[r["class"]], np.float32)
            write_srgb_png(d / "texture.png", np.where(obs[..., None], img, col[None, None]))
            r.setdefault("donor_fill", {})[m] = {"class": r["class"], "colour_linear": col.tolist(),
                                                 "rule": "median of observed same-class objects (INFERRED)"}


def region_record(r: dict, view_names) -> dict:
    keys = ("cut", "handover_mm", "dominant", "soft_extended", "handover_texels")
    return {"name": r["name"], "texels": int(r["mask"].sum()),
            "views_in_priority": [view_names[i] for i in r["views"]], **{k: r[k] for k in keys if k in r}}


def apply_contact_fill(ci, cov: dict, medians: dict, out: Path) -> None:
    """fusion.contacts (experimental): AFTER fill_texture, unobserved surface texels near the occluder take the
    occluder's median observed colour (zones.contact_weights). They stay unobserved / INFERRED in every map."""
    for c in ci.contacts:
        rec = cov["objects"].get(c["surface"], {})
        if not rec.get("projected"):
            continue
        row = next(o for o in ci.objects if o["name"] == c["surface"])
        z = load_mesh(ci.mesh_dir, row["file"])
        tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], rec["tex_size"])
        w = contact_weights(tex.P, occluder_points(ci, c["occluder"]), c["within_mm"] * MM, c["box"])
        for m, meds in medians.items():
            col = meds.get(c["occluder"])
            if col is None or not w.any():
                continue
            d = out / m / safe_name(c["surface"])
            obs = read_rgb_linear(d / "coverage.png")[tex.rows, tex.cols, 0] > 0.5
            img = contact_fill(read_rgb_linear(d / "texture.png"), tex, obs, w, np.asarray(col, np.float32))
            write_srgb_png(d / "texture.png", img)
            rec.setdefault("contact_fill", {}).setdefault(m, []).append(
                {"occluder": c["occluder"], "colour_linear": list(col), "within_mm": c["within_mm"],
                 "texels": int(((w > 0) & ~obs).sum()), "rule": f"occluder median observed colour, weight 1 within "
                 f"{RAMP_IN} x within_mm to 0 at {RAMP_OUT} x (INFERRED: counted as unobserved)"})


def require_projection(case: Path, method: str | None = None) -> dict:
    p = case / "proj" / "coverage.json"
    if not p.is_file():
        raise StageError(f"missing {p} (run `mvb2r.py project {case.name}` first)")
    from mvb2r.fusion_io import read_json
    cov = read_json(p)
    if method and method not in cov.get("methods", []):
        raise StageError(f"method {method!r} not projected (have {cov.get('methods')}); "
                         f"run `mvb2r.py project {case.name} --method {method}`")
    return cov
