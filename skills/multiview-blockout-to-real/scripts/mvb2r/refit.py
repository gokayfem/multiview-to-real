"""Stage `refit`: carve `refit` objects (hair, cloth, soft props) to their generated silhouettes.

    mvb2r.py refit CASE [--object NAME] [--mode solid|shell] [--dilate 4] [--max-faces 30000]

For each object flagged "refit": true in case.json (or named with --object): voxel carving to
targets/mask_<object>_XX.png across every view (mvb2r/voxcarve.py), closed smoothed surface, decimated,
UVMap re-unwrapped with island margins, written to a NEW file blockout_refit.blend (blockout.blend is never
overwritten). Report refit/refit.json: per view mask IoU, object pixels outside its mask and pixels covering
other objects' masks, before -> after (exact passes of the new blend); sheets refit/<object>_views.jpg.
Refuses when masks are missing or the object is seen by fewer than 2 views. Then the user must point the
case at the new blend and re-run `views` (the command prints the next commands; `--apply` repoints the case and keeps the old blend).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from mvb2r import voxcarve as vc
from mvb2r.fusion_io import (StageError, load_mesh, case_json, load_objects, object_settings, read_exr, read_ids, resolve_case,
                             run_blender, safe_name, write_json)
from mvb2r.pinhole import usable_cameras
from mvb2r.projinputs import export_mesh, load_masks

MIN_VIEWS = 2


def register(subparsers) -> None:
    p = subparsers.add_parser("refit", help="carve refit objects to their generated masks -> blockout_refit.blend")
    p.add_argument("case")
    p.add_argument("--object", default=None, help="carve this object (default: every object with refit: true)")
    p.add_argument("--mode", choices=["solid", "shell"], default="solid",
                   help="solid: whole interior carvable; shell: only a layer -inner..+outer around the mesh (hair)")
    p.add_argument("--dilate", type=int, default=4, help="mask dilation in px before carving")
    p.add_argument("--max-faces", type=int, default=30000)
    p.add_argument("--apply", action="store_true",
                   help="after carving, keep blockout.blend as blockout_pre_refit.blend and use the refit blend")
    p.set_defaults(func=run)


def apply_refit(case: Path) -> list[str]:
    """Point the case at blockout_refit.blend, keeping the old blend (refuses to overwrite a previous backup)."""
    new, cur, keep = case / "blockout_refit.blend", case / "blockout.blend", case / "blockout_pre_refit.blend"
    if not new.is_file():
        raise StageError(f"missing {new} (run `mvb2r.py refit {case}` first)")
    if keep.exists():
        raise StageError(f"{keep} already exists (a previous refit was applied): move it away first")
    cur.replace(keep)
    shutil.copyfile(new, cur)
    return [f"blockout.blend now holds the refit mesh; the previous one is {keep.name}"]


def targets_of(case: Path, cfg: dict, name: str | None) -> list[dict]:
    rows = {r["name"]: r for r in load_objects(case)}
    sett = object_settings(cfg)
    names = [name] if name else [n for n, s in sett.items() if s["refit"]]
    if not names:
        raise StageError("no object to refit: mark one \"refit\": true in case.json or pass --object NAME")
    for n in names:
        if n not in rows:
            raise StageError(f"{n!r} is not in views/objects.json")
    return [rows[n] for n in names]


def view_data(case: Path, cams) -> list:
    out = []
    for c in cams:
        d = case / "views" / c.name
        for f in ("depth_z.exr", "object_id.png"):
            if not (d / f).is_file():
                raise StageError(f"missing {d / f} (run `mvb2r.py views {case.name}` first)")
        dz = read_exr(d / "depth_z.exr")
        out.append((dz[..., 0] if dz.ndim == 3 else dz, read_ids(d / "object_id.png")))
    return out


def usable_views(name, oid, cams, vdata, masks) -> list[int]:
    return [i for i in range(len(cams)) if (name, i) in masks and
            (int((vdata[i][1] == oid).sum()) >= 50 or int(masks[(name, i)].sum()) >= 50)]


def other_masks(name, i, masks, shape):
    others = [m for (n, k), m in masks.items() if k == i and n != name]
    return np.any(others, 0) if others else np.zeros(shape, bool)


def carve_object(case, obj, cams, vdata, masks, mesh_dir, args) -> dict:
    name, oid = obj["name"], int(obj["id"])
    views = usable_views(name, oid, cams, vdata, masks)
    if len(views) < MIN_VIEWS:
        raise StageError(f"refit {name!r}: needs targets/mask_{safe_name(name)}_XX.png in >= {MIN_VIEWS} views where "
                         f"it is visible, found {len(views)} (run `mvb2r.py targets {case.name}` to write masks)")
    tri = load_mesh(mesh_dir, f"mesh_{oid:03d}.npz")["tri_pos"]
    prm = vc.scaled_params(tri, args.mode, args.dilate)
    G, lo, n = vc.grid(tri, prm)
    keep = vc.candidates(vc.signed_distance(G, tri, prm.voxel / 2), prm)
    n_cand, removed = int(keep.sum()), {}
    for i in views:
        depth, ids = vdata[i]
        rm = vc.carve_view(G, keep, cams[i], masks[(name, i)], other_masks(name, i, masks, ids.shape), depth, ids,
                           oid, prm, tol=2 * prm.voxel)
        keep[rm] = False
        removed[cams[i].name] = int(len(rm))
    occ = vc.largest_component(keep.reshape(n))
    mesh = vc.mesh_from_occupancy(occ, lo, prm)
    if mesh is None:
        raise StageError(f"refit {name!r}: carving removed everything; check the masks (dilate more with --dilate)")
    out = case / "refit" / safe_name(name)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "carved.npz", verts=mesh[0], faces=mesh[1])
    return {"params": prm.__dict__, "grid": list(n), "candidates": n_cand, "removed_by_view": removed,
            "kept_voxels": int(occ.sum()), "views_used": [cams[i].name for i in views],
            "closed_before_decimate": vc.is_closed(mesh[1]), "mesh_file": str(out / "carved.npz")}


def verify(case, new_blend, obj, cams, vdata, masks, log) -> dict:
    out = case / "refit" / safe_name(obj["name"]) / "after"
    run_blender("render.py", ["--cameras", case / "views" / "cameras.json", "--out", out, "--passes",
                              "--no-beauty"], new_blend, log)
    per = {}
    for i, c in enumerate(cams):
        if (obj["name"], i) not in masks:
            continue
        own, oth = masks[(obj["name"], i)], other_masks(obj["name"], i, masks, vdata[i][1].shape)
        after = read_ids(out / c.name / "object_id.png")
        per[c.name] = {"before": vc.mask_metrics(vdata[i][1], obj["id"], own, oth),
                       "after": vc.mask_metrics(after, obj["id"], own, oth)}
    sheet(case, obj, cams, vdata, masks, out)
    return per


def sheet(case, obj, cams, vdata, masks, after_dir) -> None:
    import cv2
    from mvb2r.sheets import grid as img_grid, labelled
    tiles = []
    for i, c in enumerate(cams):
        if (obj["name"], i) not in masks:
            continue
        m = masks[(obj["name"], i)]
        for tag, ids in (("before", vdata[i][1]), ("after", read_ids(after_dir / c.name / "object_id.png"))):
            img = np.full(m.shape + (3,), 40, np.uint8)
            img[ids > 0] = 90
            img[ids == obj["id"]] = (200, 160, 60)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnt, -1, (0, 255, 255), 1)
            tiles.append(labelled(cv2.resize(img, (320, 320 * m.shape[0] // m.shape[1])), f"{c.name} {tag} (yellow=mask)"))
    cv2.imwrite(str(case / "refit" / f"{safe_name(obj['name'])}_views.jpg"), img_grid(tiles, 4))


def run(args) -> int:
    case = resolve_case(args.case)
    cfg = case_json(case)
    objs = targets_of(case, cfg, args.object)
    cams = usable_cameras(case, "refit")
    vdata = view_data(case, cams)
    masks = load_masks(case, cams, [r["name"] for r in load_objects(case)])
    mesh_dir = export_mesh(case, case / "proj" / "_mesh")
    log = case / "refit" / "refit.log"
    src, report = case / "blockout.blend", {"objects": {}}
    for k, obj in enumerate(objs):
        rec = carve_object(case, obj, cams, vdata, masks, mesh_dir, args)
        dst = case / "refit" / f"_stage_{k}.blend"
        run_blender("carve.py", ["--object", obj["name"], "--mesh", rec["mesh_file"], "--out", dst,
                                 "--max-faces", args.max_faces], src, log)
        rec["mesh_after"] = __import__("json").loads(Path(str(dst) + ".json").read_text())
        rec["per_view"] = verify(case, dst, obj, cams, vdata, masks, log)
        report["objects"][obj["name"]] = rec
        src = dst
    final = case / "blockout_refit.blend"
    shutil.copyfile(src, final)
    report["new_blend"] = str(final)
    if args.apply:
        report["applied"] = apply_refit(case)
        report["next"] = [f"mvb2r.py views {case}  (cameras and passes must be regenerated for the new mesh)"]
    else:
        report["next"] = [f"mvb2r.py refit {case} --apply  (or point the case at {final.name} yourself; "
                          "the old blend is kept as blockout_pre_refit.blend)",
                          f"mvb2r.py views {case}"]
    write_json(case / "refit" / "refit.json", report)
    print_summary(report)
    return 0


def print_summary(report: dict) -> None:
    for name, r in report["objects"].items():
        for v, m in r["per_view"].items():
            b, a = m["before"], m["after"]
            print(f"[refit] {name} {v}: IoU {b['iou']:.3f} -> {a['iou']:.3f}, outside mask {b['outside_mask_px']} -> "
                  f"{a['outside_mask_px']} px, on other masks {b['covers_other_masks_px']} -> "
                  f"{a['covers_other_masks_px']} px")
        print(f"[refit] {name}: mesh {r['mesh_after']}")
    print(f"[refit] wrote {report['new_blend']}" + ("; applied: " + report["applied"][0] if report.get("applied")
                                                     else " (blockout.blend untouched)") + ". Next:")
    for line in report["next"]:
        print("    " + line)
