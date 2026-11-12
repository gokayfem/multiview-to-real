"""Load everything `project` needs from the case folder and build each object's fusion input.

Reads (never writes) views/{cameras,objects}.json, views/view_XX/{depth_z.exr, object_id.png},
targets/view_XX.png (+ optional align_conf_XX.png, mask_<object>_XX.png), case.json, proj/_mesh (export cache).
Per-object gates: an object with its own generated mask samples only inside it (eroded 1 px by default, `gate_dilate` px when set; pixel defaults are tuned at 1664 px and scaled
per camera, explicit case.json values are used as given);
objects without a mask reject the dilated union of the other objects' masks and get a low-confidence halo
band (x`halo_factor`) around those objects' clay silhouettes. Openings are never projected onto.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from mvb2r.fusion import FusionParams, ObjectInput, build_stack
from mvb2r.fusion_io import (GATE_RELIABILITY, StageError, load_mesh, mask_reliability, split_usable, targets_meta, load_objects, object_settings, read_exr, read_gray01, read_ids, read_json,
                             read_rgb_linear, run_blender, safe_name, view_num)
from mvb2r.levels import image_band_sampler
from mvb2r.pinhole import Camera, load_cameras
from mvb2r.raster import texel_set
from mvb2r.regions import single_source_regions
from mvb2r.visibility import Gates, ViewData, VisParams, depth_tolerance, make_view, sample_same_object, view_samples


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


def load_case_inputs(case: Path, cfg: dict, images: Path | None = None) -> CaseInputs:
    cams, skipped = split_usable(case, load_cameras(case / "views" / "cameras.json"), stage="project")
    cams = tuple(cams)
    rows = load_objects(case)
    sett = object_settings(cfg)
    mesh_dir = export_mesh(case, case / "proj" / "_mesh")
    meshes = {m["name"]: m for m in read_json(mesh_dir / "meshes.json")["objects"]}
    objs = []
    for r in rows:
        s = sett.get(r["name"], {})
        if r["name"] not in meshes:
            continue  # not a renderable mesh with a UV map (reported by export_mesh)
        if meshes[r["name"]]["id"] != r["id"]:
            raise StageError(f"{r['name']}: objects.json id {r['id']} != pass_index {meshes[r['name']]['id']}")
        objs.append({"name": r["name"], "id": r["id"], "file": meshes[r["name"]]["file"],
                     "class": s.get("class", r.get("class", "generic")) or "generic",
                     "detail_critical": bool(s.get("detail_critical", r.get("detail_critical", False))),
                     "prefer_view": s.get("prefer_view", r.get("prefer_view")),
                     "opening": bool(s.get("opening", r.get("opening", False)))})
    if not objs:
        raise StageError("no object of views/objects.json has an exported mesh with a UVMap")
    views = tuple(load_view_data(case, c, images) for c in cams)
    return CaseInputs(case=case, cfg=cfg, cams=cams, views=views, objects=tuple(objs),
                      masks=load_masks(case, cams, [o["name"] for o in objs]), mesh_dir=mesh_dir,
                      skipped=tuple(skipped))


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Dilate by r px (r > 0), erode by -r px (r < 0)."""
    if r == 0:
        return m
    k = np.ones((2 * abs(r) + 1,) * 2, np.uint8)
    return (cv2.dilate if r > 0 else cv2.erode)(m.astype(np.uint8), k) > 0


def _px(opt: dict, key: str, default_at_ref: float, cam) -> float:
    """case.json value in pixels as given; otherwise the default (tuned at 1664 px) scaled UP for larger cameras
    (never below the default: misregistration does not shrink with resolution, see VisParams.for_resolution)."""
    if key in opt:
        return float(opt[key])
    return default_at_ref * max(max(cam.width, cam.height) / 1664.0, 1.0)


def gates_for(ci: CaseInputs, obj: dict, vi: int, opt: dict) -> Gates:
    own = ci.masks.get((obj["name"], vi))
    cam = ci.cams[vi]
    if own is not None:
        # default: ERODE the generated mask by 1 px (texels sample only well inside the generated object; the
        # silhouette ring of a generated object that is smaller / shifted vs the clay is filled from the same
        # object instead of taking the background). case.json fusion.gate_dilate > 0 dilates (e.g. hair wisps).
        return Gates(accept=_dilate(own, int(round(_px(opt, "gate_dilate", -1, cam)))))
    others = [(o, ci.masks[(o["name"], vi)]) for o in ci.objects if (o["name"], vi) in ci.masks]
    if not others:
        return Gates()
    union = np.any([m for _, m in others], 0)
    clay = np.isin(ci.views[vi].ids, [o["id"] for o, _ in others]).astype(np.uint8)
    dist = cv2.distanceTransform(1 - clay, cv2.DIST_L2, 5)
    halo = np.where((clay == 0) & (dist <= _px(opt, "halo_px", 6.0, cam)), opt.get("halo_factor", 0.3), 1.0)
    return Gates(reject=_dilate(union, max(1, int(round(_px(opt, "reject_dilate", 4, cam))))),
                 score_mult=halo.astype(np.float32))


def object_input(ci: CaseInputs, obj: dict, R: int, vis: VisParams, opt: dict) -> ObjectInput:
    z = load_mesh(ci.mesh_dir, obj["file"])
    tex = texel_set(z["tri_pos"], z["tri_uv"], z["tri_nrm"], R)
    gates = [gates_for(ci, obj, vi, opt) for vi in range(len(ci.views))]
    samples = [view_samples(tex, obj["id"], v, vis, g) for v, g in zip(ci.views, gates)]
    st = build_stack(samples, [c.name for c in ci.cams])
    adm = [(v.ids == obj["id"]) & (g.accept if g.accept is not None else True) for v, g in zip(ci.views, gates)]
    regions = ()
    if obj["detail_critical"]:
        names = [c.name for c in ci.cams]
        prefer = names.index(obj["prefer_view"]) if obj["prefer_view"] in names else None
        regions = tuple(single_source_regions(tex, st, prefer))
    targets = tuple((_calib(v), (v.cam.height, v.cam.width)) for v in ci.views)
    base = ObjectInput(name=obj["name"], cls=obj["class"], tex=tex, st=st, regions=regions,
                       bands=image_band_sampler(list(ci.views), obj["id"], adm, st.u, st.v), targets=targets)
    return replace(base, resample=_resampler(base, ci, obj["id"], samples, adm, vis))


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


def fusion_options(cfg: dict) -> tuple[FusionParams, dict, dict]:
    """case.json "fusion" -> (params, per-class method overrides, misc options)."""
    f = cfg.get("fusion", {}) or {}
    if not isinstance(f, dict):
        raise StageError("case.json 'fusion' must be an object")
    prm = FusionParams().with_overrides(f.get("params"))
    by_class = f.get("by_class", {}) or {}
    opt = {k: f[k] for k in ("tex_size", "min_tex", "max_tex", "gate_dilate", "reject_dilate", "halo_px",
                             "halo_factor", "novel", "sky") if k in f}
    return prm, by_class, opt
