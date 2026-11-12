"""The three prompt families that worked in the experiment, as templates filled from the case.

  grid_edit    ONE multi-reference edit of the clay sheet -> a photoreal sheet with the same panels
  refine       one view at the largest verified size: clay view = geometry authority, generated panel =
               appearance authority, sheet = cross-view consistency, references = their stated roles
  consistency  render-then-refine: a consensus re-render (one shared texture seen from this camera) is the
               authority for WHERE details are; the accepted view supplies photographic quality

Nothing scene-specific lives here: the look, the object descriptions and the reference roles come from
case.json; the layout sentence from grid.layout_sentence. Free text from the case is cleaned to one line.
"""
from __future__ import annotations

from string import Template

from . import common

ROLE_TEXT = {
    "identity": "defines only the identity and appearance of {what}",
    "product": "defines only the exact design of {what}",
    "material": "defines only the material and finish of {what}",
    "style": "defines only the photographic style and colour palette",
    "environment": "defines only what is seen through openings and the surroundings",
    "other": "is a reference for {what}",
}

GRID_EDIT = Template("""Produce ONE image that preserves the exact panel layout of Image 1. It must look like $n real photographs taken with a full-frame camera, not a 3D render. The panels show $n camera views of ONE identical physical scene at ONE identical moment. Image 1 defines the layout and the framing of every camera. $layout
$references
The scene: $look
Realise every grey shape in its exact existing shape, position and scale as a real object: $objects
One coherent light setup, fixed in world space across all panels, with natural falloff, soft contact shadows and bounce light. Photographic qualities: natural dynamic range, mild lens softness toward the corners, subtle fine grain, true-to-life colour, no CGI smoothness, no stylised grading.
Keep every panel in its original position and exact boundaries. Do not merge panels, add borders or text, change crop or lens, move objects, invent objects, or add people. Output only the $rows x $cols sheet at the requested resolution.""")

REFINE = Template("""Produce ONE single full-frame real photograph taken with a full-frame camera, not a 3D render, not a collage, grid or split image, with no borders, panels or text.
Image 1 is the geometry and camera authority: reproduce exactly its camera viewpoint, framing, crop, lens perspective and the exact shape, position and scale of every object. Do not move, add, remove, resize or redesign anything.
Image 2 is a low-resolution photograph of this exact same view and is the appearance authority: keep every material, colour, light patch and shadow it already establishes. Only add fine photographic detail at higher resolution.
Image 3 shows the same scene from the other cameras at the same moment; use it only for consistency of appearance, not as layout.
$references
The scene: $look
Objects: $objects
Same light direction, intensity and colour as Image 2, true-to-life colour, no new objects, no CGI smoothness, no stylised grading, natural mild grain.""")

CONSISTENCY = Template("""Produce ONE single full-frame real photograph of this exact scene from this exact camera; not a collage, grid or split image, no borders, no text.
Image 1 is a 3D re-render of exactly this scene from exactly this camera, made from one shared texture. It is the authority for WHERE every surface detail is and for surface colours: keep the position, direction and pattern of every visible detail exactly where Image 1 has it, pixel-aligned, with the same colours and brightness.
Only repair Image 1: remove its stitching artefacts - visible seams, patch edges, brightness steps between patches, smeared or duplicated details, colour bleeding, and flat or blank filled areas. Wherever Image 1 is damaged, blank, flat or smeared, take the content from Image 2.
Image 2 is a clean photograph of the same view: use it for photographic quality and smooth continuous light and shadow. Image 3 is the geometry of this camera.
$references
Objects: $objects
Same framing, same light direction; do not move, add or remove anything; no stylised grading, natural mild grain.""")


def object_lines(objects: list[dict], limit: int = 40) -> str:
    rows = []
    for o in objects[:limit]:
        if o.get("opening"):
            rows.append(f"{common.clean_text(o['name'], 60)}: an opening; {common.clean_text(o.get('description', ''), 200)}")
        elif o.get("description"):
            rows.append(f"{common.clean_text(o['name'], 60)}: {common.clean_text(o['description'], 240)}")
    return "; ".join(rows) + "." if rows else "keep every object as it is."


def reference_lines(refs: list[dict]) -> str:
    out = []
    for r in refs:
        if r.get("kind") != "reference":
            continue
        what = ", ".join(common.clean_text(a, 60) for a in r.get("applies_to") or []) or "the matching objects"
        text = ROLE_TEXT.get(r.get("role"), ROLE_TEXT["other"]).format(what=what)
        note = f" ({common.clean_text(r['note'], 160)})" if r.get("note") else ""
        out.append(f"Image {r['image']} {text}{note}; it is not a panel and not geometry.")
    return " ".join(out)


def look_text(cfg: dict) -> str:
    look = common.clean_text(cfg.get("look") or "", 800)
    return look or "a real, lived-in scene photographed in natural light."


def grid_edit(cfg: dict, objects: list[dict], refs: list[dict], layout: dict, layout_sentence: str) -> str:
    return GRID_EDIT.substitute(n=layout["n_views"], rows=layout["rows"], cols=layout["cols"],
                                layout=layout_sentence, references=reference_lines(refs),
                                look=look_text(cfg), objects=object_lines(objects)).strip()


def refine(cfg: dict, objects: list[dict], refs: list[dict]) -> str:
    return REFINE.substitute(references=reference_lines(refs), look=look_text(cfg),
                             objects=object_lines(objects)).strip()


def consistency(cfg: dict, objects: list[dict], refs: list[dict]) -> str:
    return CONSISTENCY.substitute(references=reference_lines(refs), objects=object_lines(objects)).strip()


def renumber(refs: list[dict], first: int) -> list[dict]:
    """User references renumbered to follow the fixed images of a per-view family."""
    users = [r for r in refs if r.get("kind") == "reference"]
    return [{**r, "image": first + i} for i, r in enumerate(users)]
