"""Load everything `project` needs from the case folder and build each object's fusion input.

Reads (never writes) views/{cameras,objects}.json, views/view_XX/{depth_z.exr, object_id.png},
targets/view_XX.png (+ optional align_conf_XX.png, mask_<object>_XX.png), case.json, proj/_mesh (export cache).
Per-object gates follow case.json fusion.gate_policy ("strict" default, "legacy"): see mvb2r/gating.py.
Detail-critical objects get coherent single-source regions (mvb2r/handover.py under strict); fusion.contacts
(experimental) adds contact zones (mvb2r/zones.py). Openings are never projected onto.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from mvb2r.fill import MODES as FILL_MODES
from mvb2r.fusion import FusionParams, ObjectInput, build_stack, tex_scale
from mvb2r.common import clean_text
from mvb2r.gating import DEFAULT_POLICY, check_fusion_keys, dilate, gate_options, gates_for_policy
from mvb2r.fusion_io import (GATE_RELIABILITY, StageError, load_mesh, mask_reliability, split_usable, targets_meta, load_objects, object_settings, read_exr, read_gray01, read_ids, read_json,
                             read_rgb_linear, run_blender, safe_name, view_num)
from mvb2r.levels import image_band_sampler
from mvb2r.pinhole import Camera, load_cameras
from mvb2r.raster import texel_set
from mvb2r.handover import DEFAULT_CUT, DEFAULT_HANDOVER_MM, coherent_regions, parse_regions, region_mask
from mvb2r.regions import single_source_regions
from mvb2r.zones import contact_pixels, parse_contacts
from mvb2r.visibility import Gates, ViewData, VisParams, depth_tolerance, make_view, sample_same_object, view_samples


MIN_CELL = 2  # smallest label cell (texels)
EVALUATION_GATES = {"gate_policy": "legacy"}            # LOO evaluation: legacy gates, default values
OBJECT_GATE_KEYS = ("gate_dilate", "reject_dilate")      # per-object gate overrides


@dataclass(frozen=True)
class CaseInputs:
    case: Path
    cfg: dict
    cams: tuple
    views: tuple            # ViewData per camera
    objects: tuple          # rows: name, id, file, class, detail_critical, prefer_view, opening
    masks: dict             # {(object name, view index): bool mask at calibration size}
    mesh_dir: Path
    skipped: tuple = ()     # views rejected in targets.json ({"view", "reason"})
    contacts: tuple = ()    # validated fusion.contacts (zones.parse_contacts)


def export_mesh(case: Path, out: Path) -> Path:
    blend = case / "blockout.blend"
    if not blend.is_file():
        raise StageError(f"missing {blend} (run `mvb2r.py new {case}` first)")
    js = out / "meshes.json"
    if not js.is_file() or js.stat().st_mtime < blend.stat().st_mtime:
        run_blender("export_mesh.py", ["--out", out], blend=blend, log=case / "proj" / "logs" / "export_mesh.log")
    return out


def _target_image(case: Path, cam: Camera) -> np.ndarray:
    p = case / "targets" / f"{cam.name}.png"
    if not p.is_file():
        raise StageError(f"missing {p} (run `mvb2r.py targets {case.name}` first)")
    img = read_rgb_linear(p)
    if abs(img.shape[1] / cam.width - img.shape[0] / cam.height) > 0.01:
        raise StageError(f"{p} is {img.shape[1]}x{img.shape[0]}, camera {cam.name} is {cam.width}x{cam.height}")
    return img


def load_view_data(case: Path, cam: Camera, images: Path | None = None) -> ViewData:
    vd = case / "views" / cam.name
    for f in ("depth_z.exr", "object_id.png"):
        if not (vd / f).is_file():
            raise StageError(f"missing {vd / f} (run `mvb2r.py views {case.name}` first)")
    depth = read_exr(vd / "depth_z.exr")
    depth = depth[..., 0] if depth.ndim == 3 else depth
    img = read_rgb_linear(images / f"{cam.name}.png") if images else _target_image(case, cam)
    al = case / "targets" / f"align_conf_{view_num(cam.name)}.png"
    return make_view(cam, depth, read_ids(vd / "object_id.png"), img, read_gray01(al) if al.is_file() else None)


def load_masks(case: Path, cams, names) -> dict:
    """{(object, view index): mask} for masks reliable enough to gate (targets.json reliability high / medium /
    not recorded). "low" and "prior only" masks (e.g. the clay outline reused) are ignored: such an object
    then rejects the other objects' masks like any unmasked object."""
    meta, out = targets_meta(case), {}
    for name in names:
        for vi, cam in enumerate(cams):
            p = case / "targets" / f"mask_{safe_name(name)}_{view_num(cam.name)}.png"
            if p.is_file() and mask_reliability(meta, cam.name, name) in GATE_RELIABILITY:
                m = read_gray01(p)
                if m.shape != (cam.height, cam.width):
                    m = cv2.resize(m, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
                out[(name, vi)] = m > 0.5
    return out


def _object_row(r: dict, s: dict, mesh: dict, all_names: list, usable: list) -> dict:
    specs, notes = usable_region_specs(r["name"], parse_regions(s.get("regions"), r["name"], all_names), usable)
    for n in notes:
        print(f"[project] {n}", flush=True)
    return {"name": r["name"], "id": r["id"], "file": mesh["file"],
            "class": s.get("class", r.get("class", "generic")) or "generic",
            "detail_critical": bool(s.get("detail_critical", r.get("detail_critical", False))),
            "prefer_view": s.get("prefer_view", r.get("prefer_view")),
            "opening": bool(s.get("opening", r.get("opening", False))), "regions": specs, **s.get("gate", {})}


def load_case_inputs(case: Path, cfg: dict, images: Path | None = None) -> CaseInputs:
    all_cams = load_cameras(case / "views" / "cameras.json")
    cams, skipped = split_usable(case, all_cams, stage="project")
    cams = tuple(cams)
    rows = load_objects(case)
    sett = object_settings(cfg)
    f = cfg.get("fusion") or {}
    policy = gate_options(f)["gate_policy"] if isinstance(f, dict) else DEFAULT_POLICY
    mesh_dir = export_mesh(case, case / "proj" / "_mesh")
    meshes = {m["name"]: m for m in read_json(mesh_dir / "meshes.json")["objects"]}
    objs = []
    for r in rows:
        if r["name"] not in meshes:
            continue  # not a renderable mesh with a UV map (reported by export_mesh)
        if meshes[r["name"]]["id"] != r["id"]:
            raise StageError(f"{clean_text(r['name'], 40)}: objects.json id {r['id']} != pass_index "
                             f"{meshes[r['name']]['id']}")
        objs.append(_object_row(r, sett.get(r["name"], {}), meshes[r["name"]], [c.name for c in all_cams],
                                [c.name for c in cams]))
    if not objs:
        raise StageError("no object of views/objects.json has an exported mesh with a UVMap")
    contacts = parse_contacts(f.get("contacts") if isinstance(f, dict) else None, [o["name"] for o in objs])
    for n in ignored_settings(objs, {"gate_policy": policy}, contacts):
        print(f"[project] {n}", flush=True)
    views = tuple(load_view_data(case, c, images) for c in cams)
    return CaseInputs(case=case, cfg=cfg, cams=cams, views=views, objects=tuple(objs),
                      masks=load_masks(case, cams, [o["name"] for o in objs]), mesh_dir=mesh_dir,
                      skipped=tuple(skipped), contacts=contacts)


def usable_region_specs(obj_name: str, specs: tuple, usable: list) -> tuple:
    """(specs, notes): a region prefer_view on a camera marked unusable in targets.json falls back to the object's
    prefer_view / the score rule, like the object-level prefer_view (one note per region). New dicts."""
    out, notes = [], []
    for sp in specs:
        if sp.get("prefer_view") and sp["prefer_view"] not in usable:
            notes.append(f"{clean_text(obj_name, 40)}: region {clean_text(sp['name'], 40)} prefer_view "
                         f"{clean_text(sp['prefer_view'], 40)} is unusable (targets.json); using the object's "
                         f"prefer_view / largest total score instead")
            sp = {**sp, "prefer_view": None}
        out.append(sp)
    return tuple(out), notes


def ignored_settings(objs, opt: dict, contacts: tuple) -> list[str]:
    """One line per case.json setting that has no effect, and why."""
    notes = []
    legacy = opt.get("gate_policy", DEFAULT_POLICY) == "legacy"
    for o in objs:
        name = clean_text(o["name"], 40)
        if o.get("regions") and not o.get("detail_critical"):
            notes.append(f"objects[{name}].regions ignored: {name} is not detail_critical (regions are its "
                         f"single-source regions)")
        elif o.get("regions") and legacy and any(r.get("handover_mm") != DEFAULT_HANDOVER_MM or
                                                 r.get("cut") != DEFAULT_CUT for r in o["regions"]):
            notes.append(f"objects[{name}].regions handover_mm / cut ignored under gate_policy legacy (no "
                         f"hand-over band; the regions keep a priority list of views)")
    if contacts and legacy:
        notes.append("fusion.contacts ignored under gate_policy legacy (contact zones belong to the strict gate)")
    return notes


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Dilate by r px (r > 0), erode by -r px (r < 0)."""
    return dilate(m, r)


def gates_for(ci: CaseInputs, obj: dict, vi: int, opt: dict, zones: dict | None = None) -> Gates:
    """Gates of one object in view vi under case.json fusion.gate_policy (mvb2r/gating.py)."""
    return gates_for_policy(ci, obj, vi, opt, zones)


def object_input(ci: CaseInputs, obj: dict, R: int, vis: VisParams, opt: dict) -> ObjectInput:
    z = load_mesh(ci.mesh_dir, obj["file"])
    tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], R)
    strict = opt.get("gate_policy", DEFAULT_POLICY) == "strict"
    zones = contact_pixels(ci, obj, ci.contacts) if strict and ci.contacts else [{} for _ in ci.views]
    gates = [gates_for(ci, obj, vi, opt, zones[vi]) for vi in range(len(ci.views))]
    samples = [view_samples(tex, obj["id"], v, vis, g) for v, g in zip(ci.views, gates)]
    raw = build_stack(samples, [c.name for c in ci.cams])
    adm = [(v.ids == obj["id"]) & (g.accept if g.accept is not None else True) for v, g in zip(ci.views, gates)]
    regionize = _regionizer(tex, obj, strict, [c.name for c in ci.cams]) if obj["detail_critical"] else None
    st, regions = regionize(raw) if regionize else (raw, ())
    targets = tuple((_calib(v), (v.cam.height, v.cam.width)) for v in ci.views)
    base = ObjectInput(name=obj["name"], cls=obj["class"], tex=tex, st=st, regions=regions,
                       bands=image_band_sampler(list(ci.views), obj["id"], adm, st.u, st.v), targets=targets,
                       raw_st=raw if regionize else None, regionize=regionize)
    return replace(base, resample=_resampler(base, ci, obj["id"], samples, adm, vis))


def _regionizer(tex, obj: dict, strict: bool, names: list):
    """stack -> (stack, regions) of a detail-critical object: coherent regions with a hand-over (strict) or
    today's priority-list regions (legacy); `objects[].regions` shapes when given, else one per mesh part."""
    prefer = names.index(obj["prefer_view"]) if obj.get("prefer_view") in names else None
    specs = tuple(obj.get("regions") or ())

    def regionize(st):
        if strict:
            return coherent_regions(tex, st, specs, prefer, tex_scale(tex))
        if not specs:
            return st, tuple(single_source_regions(tex, st, prefer))
        regs = [single_source_regions(tex, st, names.index(s["prefer_view"]) if s["prefer_view"] else prefer,
                                      [(s["name"], region_mask(tex, s))])[0] for s in specs]
        return st, tuple(regs)
    return regionize


def evaluation_stack(ci: CaseInputs, obj: dict, R: int, vis: VisParams, opt: dict):
    """(texels, stack) under the LEGACY gates with their DEFAULT values, whatever gate_policy, fusion gate keys or
    per-object overrides say: the leave-one-view-out evaluation mask and reference colours, so two settings are
    judged on the same pixels. `opt` is accepted for symmetry with object_input and not used for gating."""
    z = load_mesh(ci.mesh_dir, obj["file"])
    tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], R)
    plain = {k: v for k, v in obj.items() if k not in OBJECT_GATE_KEYS}     # legacy DEFAULTS: no gate overrides
    gates = [gates_for(ci, plain, vi, EVALUATION_GATES) for vi in range(len(ci.views))]
    samples = [view_samples(tex, obj["id"], v, vis, g) for v, g in zip(ci.views, gates)]
    return tex, build_stack(samples, [c.name for c in ci.cams])


def _calib(v: ViewData) -> np.ndarray:
    if v.image.shape[:2] == (v.cam.height, v.cam.width):
        return v.image
    return cv2.resize(v.image, (v.cam.width, v.cam.height), interpolation=cv2.INTER_AREA)


def _resampler(base: ObjectInput, ci: CaseInputs, oid: int, samples, adm, vis: VisParams):
    def resample(offsets: dict, st=None) -> ObjectInput:
        """Re-sample colours at shifted positions. `st` is the CURRENT stack (e.g. leave-one-out): its validity
        is kept, so a held-out view never comes back."""
        st = base.st if st is None else st
        color = st.color.copy()
        for vi, (du, dv) in offsets.items():
            s, view = samples[vi], ci.views[vi]
            k = np.nonzero(st.valid[vi])[0]
            if not len(k):
                continue
            tol = depth_tolerance(vis, s.zc[k], s.cos[k], view.cam.fx)
            color[vi, k] = sample_same_object(view, view.image, oid, s.u[k] + du[k], s.v[k] + dv[k], s.zc[k], tol)
        bands = image_band_sampler(list(ci.views), oid, adm, st.u, st.v, offsets)
        return replace(base, st=replace(st, color=color), bands=bands)
    return resample


def object_params(prm: FusionParams, obj: dict, opt: dict) -> FusionParams:
    """Per-object fusion parameters: detail-critical objects under the strict policy use half-size label cells
    (first run: 8 instead of 16 texels on skin stopped rectangular patches on jaw and neck)."""
    if obj.get("detail_critical") and opt.get("gate_policy", DEFAULT_POLICY) == "strict":
        return replace(prm, cell=max(MIN_CELL, prm.cell // 2))
    return prm


def fusion_options(cfg: dict) -> tuple[FusionParams, dict, dict]:
    """case.json "fusion" -> (params, per-class method overrides, misc options)."""
    f = cfg.get("fusion", {}) or {}
    if not isinstance(f, dict):
        raise StageError("case.json 'fusion' must be an object")
    check_fusion_keys(f)
    prm = FusionParams().with_overrides(f.get("params"))
    by_class = f.get("by_class", {}) or {}
    opt = {k: f[k] for k in ("tex_size", "min_tex", "max_tex", "novel", "sky") if k in f}
    fill = f.get("fill", FILL_MODES[0])
    if fill not in FILL_MODES:
        raise StageError(f"case.json fusion.fill must be one of {FILL_MODES}, got {str(fill)[:40]!r}")
    return prm, by_class, {**opt, **gate_options(f), "fill": fill}
