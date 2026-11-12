"""`probe`: learn what the image model accepts from its own schema; gen/model.json is the single source.

    mvb2r.py probe CASE [--schema-file JSON] [--check-size]

  - fetches the endpoint's OpenAPI schema (free): https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=...
    (or reads --schema-file), finds the request schema and records: allowed parameters, the image-list
    parameter and its reference limit (maxItems or "Up to N images" in its description), the size parameter
    and the pixel cap (x-fal max_area/min_area/aspect, or the numbers stated in its description), price note.
  - --check-size spends ONE call (within the budget) to request the planned grid size and reads the size of
    the returned FILE (responses have been seen to report null dimensions and to downsize over-cap requests
    silently). The result is stored under "verified_sizes".
  - with case.json model.generator == "mock" the offline mock model is recorded instead (no network).

Every generation stage builds its request with build_arguments(): only schema-listed parameters, sizes
checked against the cap. Nothing is invented (no seed, strength, negative prompt or guidance unless the
schema lists it AND the case asks for it).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from . import common

SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={endpoint}"
ENDPOINT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)+$")
MAX_SCHEMA_BYTES = 5 * 1024 * 1024
MOCK_MODEL = {
    "endpoint": "mock", "generator": "mock", "allowed_params": ["prompt", "image_urls", "image_size"],
    "image_param": "image_urls", "size_param": "image_size", "size_format": "object", "max_images": 10,
    "pixel_cap": {"max_area": 1_048_576, "min_area": 0, "max_side": 4096, "min_aspect": 1 / 16, "max_aspect": 16,
                  "source": "mock generator constant"},
    "price": {"note": "0 (offline mock, no network)"},
    "verified_sizes": [], "fixed_arguments": {},
}
FIXED_ARGUMENT_CANDIDATES = {"num_images": 1, "output_format": "png"}


def check_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not ENDPOINT_RE.match(endpoint) or ".." in endpoint:
        raise common.StageError(f"model endpoint {str(endpoint)[:80]!r} is not an endpoint id like owner/model/edit")
    return endpoint


# ---------------------------------------------------------------- schema parsing (pure)
def _resolve(schema: dict, node: dict) -> dict:
    ref = node.get("$ref") if isinstance(node, dict) else None
    if not ref:
        return node
    name = ref.rsplit("/", 1)[-1]
    return schema.get("components", {}).get("schemas", {}).get(name, {})


def input_schema(schema: dict, endpoint: str) -> dict:
    """The request-body schema of POST /<endpoint>."""
    path = schema.get("paths", {}).get("/" + endpoint, {})
    body = path.get("post", {}).get("requestBody", {}).get("content", {}).get("application/json", {})
    node = _resolve(schema, body.get("schema", {}))
    if not node.get("properties"):
        raise common.StageError(f"schema has no request body for POST /{endpoint}")
    return node


CAP_SENTENCE = re.compile(r"total\s+pixels?\s+(?:must\s+be\s+|is\s+)?between\s+([\d,]{4,})\s+and\s+([\d,]{4,})", re.I)
OVERRIDE_HINT = ("set case.json model.pixel_cap {\"max_area\": N, \"min_area\": N, \"max_side\": N}, model.max_refs, "
                 "model.size_param and/or model.image_param after reading the endpoint's documentation")


def _side_max(schema: dict, prop: dict):
    for alt in [prop] + prop.get("anyOf", []) + prop.get("oneOf", []):
        obj = _resolve(schema, alt)
        w = obj.get("properties", {}).get("width") if isinstance(obj, dict) else None
        if isinstance(w, dict) and "height" in obj.get("properties", {}):
            return True, w.get("maximum")
    return False, None


def pixel_cap(schema: dict, prop: dict) -> dict | None:
    """Pixel cap of a size property: the structured x-fal block, else ONLY an explicit "total pixels ... between
    A and B" sentence. Anything less certain returns None (the caller refuses and asks for an override)."""
    xf = prop.get("x-fal") or {}
    _, side = _side_max(schema, prop)
    if isinstance(xf.get("max_area"), int) and xf["max_area"] > 0:
        return {"max_area": int(xf["max_area"]), "min_area": int(xf.get("min_area") or 0), "max_side": side,
                "min_aspect": xf.get("min_aspect_ratio"), "max_aspect": xf.get("max_aspect_ratio"),
                "source": "schema x-fal block"}
    m = CAP_SENTENCE.search(prop.get("description", ""))
    if m:
        lo, hi = (int(x.replace(",", "")) for x in m.groups())
        return {"max_area": max(lo, hi), "min_area": min(lo, hi), "max_side": side, "min_aspect": None,
                "max_aspect": None, "source": "schema description (explicit total-pixels sentence)"}
    return None


def _size_param(schema: dict, props: dict, override: str | None):
    if override:
        if override not in props:
            raise common.StageError(f"case.json model.size_param {override!r} is not a parameter of the schema")
        return override, "override (case.json model.size_param)"
    for name, prop in props.items():
        if _side_max(schema, prop)[0]:
            return name, f"schema (parameter {name!r} takes {{width, height}})"
    return None, "none found"


def _image_param(props: dict, override: str | None):
    if override:
        if override not in props:
            raise common.StageError(f"case.json model.image_param {override!r} is not a parameter of the schema")
        return override, "override (case.json model.image_param)"
    for name, p in props.items():
        if p.get("type") == "array" and "image" in name and "url" in name:
            return name, f"schema (array parameter {name!r})"
    return None, "none found"


def max_images(prop: dict | None):
    """(limit, source) from maxItems, else an explicit "Up to N images" sentence."""
    if prop is None:
        return None, "no image parameter"
    if isinstance(prop.get("maxItems"), int):
        return int(prop["maxItems"]), "schema maxItems"
    m = re.search(r"\b[Uu]p to (\d+) images\b", prop.get("description", ""))
    return (int(m.group(1)), "schema description (\"up to N images\")") if m else (None, "not stated")


def parse_schema(schema: dict, endpoint: str, overrides: dict | None = None) -> dict:
    """Everything the generation stages need, from the endpoint's own schema; refuses to guess."""
    ov = overrides or {}
    node = input_schema(schema, endpoint)
    props = node["properties"]
    img_name, img_src = _image_param(props, ov.get("image_param"))
    size_name, size_src = _size_param(schema, props, ov.get("size_param"))
    cap = pixel_cap(schema, props[size_name]) if size_name else None
    cap_src = cap["source"] if cap else "not determinable"
    if isinstance(ov.get("pixel_cap"), dict) and ov["pixel_cap"].get("max_area"):
        cap, cap_src = {**(cap or {}), **ov["pixel_cap"], "source": "override (case.json model.pixel_cap)"}, "override"
    if size_name and cap is None:
        raise common.StageError(f"{endpoint}: the schema states no reliable pixel cap for {size_name!r}; {OVERRIDE_HINT}")
    n_img, n_src = max_images(props.get(img_name))
    if isinstance(ov.get("max_refs"), int) and ov["max_refs"] > 0:
        n_img, n_src = ov["max_refs"], "override (case.json model.max_refs)"
    if img_name and n_img is None:
        raise common.StageError(f"{endpoint}: the schema does not state how many reference images it takes; {OVERRIDE_HINT}")
    fixed = {k: v for k, v in FIXED_ARGUMENT_CANDIDATES.items() if k in props}
    return {"endpoint": endpoint, "generator": "fal", "allowed_params": sorted(props),
            "required": sorted(node.get("required", [])), "image_param": img_name, "max_images": n_img or 0,
            "size_param": size_name, "size_format": "object" if size_name else None, "pixel_cap": cap,
            "capability_sources": {"image_param": img_src, "max_images": n_src, "size_param": size_src,
                                   "pixel_cap": cap_src},
            "price": {"note": "unknown (the schema states no price; never assumed free)"},
            "verified_sizes": [], "fixed_arguments": fixed}


def check_size(model: dict, width: int, height: int) -> None:
    cap = model.get("pixel_cap") or {}
    area = width * height
    if area > cap.get("max_area", area) or area < (cap.get("min_area") or 0):
        raise common.StageError(f"{width}x{height} ({area} px) is outside the model's pixel range "
                                f"[{cap.get('min_area')}, {cap.get('max_area')}]: over-cap requests are silently downsized")
    if cap.get("max_side") and max(width, height) > int(cap["max_side"]):
        raise common.StageError(f"{width}x{height}: a side exceeds the model's maximum {cap['max_side']}")
    ar = width / height
    if cap.get("min_aspect") and not float(cap["min_aspect"]) <= ar <= float(cap.get("max_aspect") or 1e9):
        raise common.StageError(f"aspect {ar:.3f} is outside the model's range")


def build_arguments(model: dict, prompt: str, size: tuple[int, int], image_urls: list[str] | None) -> dict:
    """Request arguments limited to schema-verified parameters."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise common.StageError("empty prompt")
    allowed = set(model["allowed_params"])
    if "prompt" not in allowed:
        raise common.StageError("the model's schema has no 'prompt' parameter")
    args = {"prompt": prompt}
    if size is not None:
        if not model.get("size_param"):
            raise common.StageError("the model's schema has no size parameter; it cannot be asked for a size")
        check_size(model, int(size[0]), int(size[1]))
        args[model["size_param"]] = {"width": int(size[0]), "height": int(size[1])}
    if image_urls is not None:
        if not model.get("image_param"):
            raise common.StageError("the model's schema takes no reference images")
        if not 1 <= len(image_urls) <= int(model["max_images"]):
            raise common.StageError(f"{len(image_urls)} references; the model takes 1..{model['max_images']}")
        args[model["image_param"]] = list(image_urls)
    args.update({k: v for k, v in model.get("fixed_arguments", {}).items() if k in allowed})
    unknown = set(args) - allowed
    if unknown:
        raise common.StageError(f"parameters not in the schema: {sorted(unknown)}")
    return args


def refine_size(model: dict, view_w: int, view_h: int, multiple: int = 16) -> tuple[int, int]:
    """Largest single-image size with the view's aspect within the cap (multiples of 16)."""
    cap = model["pixel_cap"]
    aspect = view_w / view_h
    h = int((cap["max_area"] / aspect) ** 0.5) // multiple * multiple
    while h > 0:
        w = int(round(h * aspect / multiple)) * multiple
        if w * h <= cap["max_area"] and max(w, h) <= int(cap.get("max_side") or 1 << 30):
            return w, h
        h -= multiple
    raise common.StageError("no refine size fits the model's cap")


# ---------------------------------------------------------------- files
def generator_of(cfg: dict, override: str | None = None) -> str:
    g = override or (cfg.get("model") or {}).get("generator") or "fal"
    if g not in ("fal", "mock"):
        raise common.StageError(f"model.generator must be 'fal' or 'mock', got {g!r}")
    return g


def load_model(case, cfg: dict | None = None, generator: str | None = None) -> dict:
    """gen/model.json; for the mock generator it is written on first use."""
    case = common.case_dir(case)
    cfg = cfg or common.load_case(case)
    p = case / "gen" / "model.json"
    if generator_of(cfg, generator) == "mock":
        if not p.is_file() or common.read_json(p).get("generator") != "mock":
            common.write_json(p, {**MOCK_MODEL, "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return common.read_json(p)
    m = common.read_json(common.require(p, f"mvb2r.py probe {case}"))
    want = (cfg.get("model") or {}).get("edit_endpoint")
    if m.get("generator") == "mock":
        raise common.StageError(f"{p} is the offline MOCK model but this stage runs the fal generator ({want}): pass "
                                f"--generator mock to keep using the mock, or run `mvb2r.py probe {case}` for the fal endpoint")
    if m.get("endpoint") != want:
        raise common.StageError(f"{p} describes {m.get('endpoint')!r}, case.json asks for {want!r}: run "
                                f"`mvb2r.py probe {case}` again")
    override = (cfg.get("model") or {}).get("pixel_cap")
    hosts = (cfg.get("model") or {}).get("download_hosts")
    m = {**m, "download_hosts": list(hosts)} if isinstance(hosts, list) and hosts else m
    return {**m, "pixel_cap": {**m["pixel_cap"], **override}} if override else m


def announce(stage: str, mdl: dict) -> None:
    """Say which generator and endpoint are in effect (every generation stage calls this)."""
    cap = (mdl.get("pixel_cap") or {}).get("max_area")
    common.say(f"[{stage}] generator {mdl['generator']}, endpoint {mdl['endpoint']}, pixel cap {cap}, "
               f"<= {mdl.get('max_images')} references")


def fetch_schema(endpoint: str, timeout: float = 30.0) -> dict:
    url = SCHEMA_URL.format(endpoint=urllib.parse.quote(check_endpoint(endpoint), safe="/"))
    req = urllib.request.Request(url, headers={"User-Agent": "multiview-blockout-to-real"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(MAX_SCHEMA_BYTES + 1)
    except (urllib.error.URLError, OSError) as e:
        raise common.StageError(f"cannot fetch the schema of {endpoint}: {type(e).__name__}: {str(e)[:200]} "
                                "(offline? use `probe --schema-file`)") from e
    if len(data) > MAX_SCHEMA_BYTES:
        raise common.StageError(f"schema for {endpoint} is larger than {MAX_SCHEMA_BYTES} bytes")
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise common.StageError(f"schema for {endpoint} is not JSON: {e}") from e


# ---------------------------------------------------------------- stage
def register(sub) -> None:
    p = sub.add_parser("probe", help="fetch the endpoint schema: parameters, reference limit, pixel cap -> gen/model.json")
    p.add_argument("case")
    p.add_argument("--schema-file", default=None, help="use a saved OpenAPI schema instead of fetching it")
    p.add_argument("--check-size", action="store_true", help="spend ONE call to verify the returned size")
    p.add_argument("--generator", choices=("fal", "mock"), default=None)
    p.set_defaults(func=run)


def run(args) -> int:
    case = common.case_dir(args.case)
    cfg = common.load_case(case)
    if generator_of(cfg, args.generator) == "mock":
        announce("probe", load_model(case, cfg, "mock"))
        return 0
    endpoint = check_endpoint((cfg.get("model") or {}).get("edit_endpoint"))
    schema = common.read_json(args.schema_file) if args.schema_file else fetch_schema(endpoint)
    info = parse_schema(schema, endpoint, cfg.get("model") or {})
    info = {**info, "schema_source": (f"file {Path(args.schema_file).name}" if args.schema_file
                                      else SCHEMA_URL.format(endpoint=endpoint)),
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "schema_sha256": __import__("hashlib").sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()}
    out = case / "gen" / "model.json"
    old = common.read_json(out) if out.is_file() else {}
    if old.get("endpoint") == endpoint:
        info["verified_sizes"] = old.get("verified_sizes", [])
    common.write_json(out, info)
    common.write_json(case / "gen" / "schema.json", schema)
    common.say(f"[probe] {endpoint}: params {', '.join(info['allowed_params'])}")
    src = info["capability_sources"]
    common.say(f"[probe] references <= {info['max_images']} ({info['image_param']}; {src['max_images']}); pixel cap "
               f"{(info['pixel_cap'] or {}).get('max_area')} ({src['pixel_cap']}); size parameter {info['size_param']} "
               f"({src['size_param']}); price {info['price']['note']}")
    if args.check_size:
        common.require(case / "grid" / "layout.json", f"mvb2r.py grid {case}")
        from . import generate
        return generate.probe_size(case, cfg, info)
    return 0
