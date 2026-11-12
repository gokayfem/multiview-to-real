"""Schema parsing, request construction, budget, cache and the mock path of falio."""
import os

import numpy as np

from _util import synthetic_case
from mvb2r import common, falio, model

EP = "owner/model/edit"


def schema(x_fal=True, desc_limit=True):
    size = {"description": "Total pixels must be between 921600 and 4624220, with aspect ratio between 1/16 and 16.",
            "anyOf": [{"$ref": "#/components/schemas/ImageSize"}, {"enum": ["auto_2K"], "type": "string"}]}
    if x_fal:
        size["x-fal"] = {"max_area": 4624220, "min_area": 921600, "min_aspect_ratio": 0.0625, "max_aspect_ratio": 16}
    urls = {"type": "array", "items": {"type": "string"},
            "description": "Up to 10 images are supported; if more are sent, only the last 10 are used." if desc_limit else ""}
    if not desc_limit:
        urls["maxItems"] = 4
    return {"paths": {"/" + EP: {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/In"}}}}}}},
            "components": {"schemas": {
                "In": {"properties": {"prompt": {"type": "string"}, "image_urls": urls, "image_size": size,
                                      "num_images": {"type": "integer"}, "output_format": {"type": "string"},
                                      "enable_safety_checker": {"type": "boolean"}}, "required": ["prompt", "image_urls"]},
                "ImageSize": {"properties": {"width": {"type": "integer", "maximum": 14142},
                                             "height": {"type": "integer", "maximum": 14142}}}}}}


def test_schema_gives_params_limit_and_cap():
    m = model.parse_schema(schema(), EP)
    assert m["max_images"] == 10 and m["image_param"] == "image_urls" and m["size_param"] == "image_size"
    assert m["pixel_cap"]["max_area"] == 4624220 and m["pixel_cap"]["max_side"] == 14142
    assert m["fixed_arguments"] == {"num_images": 1, "output_format": "png"}
    assert "unknown" in m["price"]["note"] and "free" not in m["price"]["note"].replace("never assumed free", "")
    m2 = model.parse_schema(schema(x_fal=False, desc_limit=False), EP)
    assert m2["pixel_cap"]["max_area"] == 4624220 and "description" in m2["pixel_cap"]["source"] and m2["max_images"] == 4


def test_arguments_only_from_the_schema():
    m = model.parse_schema(schema(), EP)
    a = model.build_arguments(m, "a prompt", (2144, 2144), ["u1", "u2"])
    assert set(a) == {"prompt", "image_size", "image_urls", "num_images", "output_format"}
    assert a["image_size"] == {"width": 2144, "height": 2144}
    for bad in (lambda: model.build_arguments(m, "p", (4992, 3328), ["u"]),       # over the cap: silently downsized
                lambda: model.build_arguments(m, "p", (100, 100), ["u"]),         # under the minimum
                lambda: model.build_arguments(m, " ", (2144, 2144), ["u"]),
                lambda: model.build_arguments(m, "p", (2144, 2144), ["u"] * 11)):
        try:
            bad()
        except common.StageError:
            continue
        raise AssertionError("accepted")
    no_size = {**m, "allowed_params": ["prompt", "image_urls"], "size_param": None}
    try:
        model.build_arguments(no_size, "p", (1024, 1024), ["u"])
    except common.StageError as e:
        assert "size" in str(e)


def test_refine_size_is_the_largest_under_the_cap():
    m = model.parse_schema(schema(), EP)
    assert model.refine_size(m, 1664, 1664) == (2144, 2144)
    w, h = model.refine_size(m, 1920, 1080)
    assert w * h <= 4624220 and abs(w / h - 16 / 9) < 0.02


def test_endpoint_ids_are_validated():
    for good in ("bytedance/seedream/v5.5/edit", "fal-ai/x/edit"):
        assert model.check_endpoint(good) == good
    for bad in ("../x", "http://x/y", "x", "a/../b", "a b/c", None):
        try:
            model.check_endpoint(bad)
        except common.StageError:
            continue
        raise AssertionError(bad)


def test_budget_refuses_before_starting():
    case = synthetic_case(2)
    cfg = common.load_case(case)
    for i in range(3):
        falio.append_log(case, {"generator": "mock", "status": "success" if i else "error"})
    falio.append_log(case, {"generator": "mock", "status": "cache_hit"})
    falio.append_log(case, {"generator": "fal", "status": "success"})
    assert falio.check_budget(case, cfg, "mock", 1, 4) == {"used": 3, "limit": 4}
    try:
        falio.check_budget(case, cfg, "mock", 2, 4)
    except falio.BudgetError as e:
        assert "3 of 4" in str(e)
    else:
        raise AssertionError
    assert falio.max_calls({"budget": {"max_calls": 7}}) == 7


def test_mock_edit_is_logged_cached_and_budgeted():
    case = synthetic_case(2, 96)
    common.write_json(case / "grid" / "layout.json", {"rows": 1, "cols": 2, "cell_w": 96, "cell_h": 96})
    ref = case / "views" / "view_01" / "clay.png"
    job = {"tag": "t", "family": "refine", "prompt": "p", "size": (128, 128), "refs": [ref],
           "out": case / "gen" / "refined" / "x.png", "views": ["view_01"], "attempt": 1, "max_calls": 1}
    r1 = falio.edit(case, job)
    assert r1["status"] == "success" and r1["returned_size"] == [128, 128] and r1["size_honoured"]
    assert (case / "gen" / "mock_truth" / "view_01.png").is_file()
    r2 = falio.edit(case, {**job, "out": case / "gen" / "refined" / "y.png"})
    assert r2["status"] == "cache_hit" and (case / "gen" / "refined" / "y.png").is_file()
    try:
        falio.edit(case, {**job, "attempt": 2})
    except falio.BudgetError:
        pass
    else:
        raise AssertionError("budget not enforced")
    s = falio.summary(case)["mock:mock"]
    assert s["started"] == 1 and s["cache_hits"] == 1
    sheet = falio.edit(case, {**job, "family": "grid", "views": ["view_01", "view_02"], "size": (192, 96),
                              "out": case / "gen" / "grid_edit" / "s.png", "max_calls": 5})
    assert sheet["returned_size"] == [192, 96]


def test_mock_truth_is_view_independent():
    from mvb2r import mockgen
    p = np.random.default_rng(0).uniform(-1, 1, (500, 3))
    ids = np.full(500, 2)
    a, b = mockgen.truth_albedo(ids, p), mockgen.truth_albedo(ids, p.copy())
    assert np.array_equal(a, b) and a.std() > 0.01
    assert (mockgen.truth_albedo(np.zeros(5, int), p[:5]) == 0).all()


def test_load_model_requires_probe_for_fal():
    case = synthetic_case(2, generator="fal")
    try:
        model.load_model(case)
    except common.StageError as e:
        assert "probe" in str(e)
    else:
        raise AssertionError
    common.write_json(case / "gen" / "model.json", {**model.parse_schema(schema(), EP)})
    assert model.load_model(case)["max_images"] == 10


def test_fal_path_builds_and_validates_without_spending():
    """The fal request is built from the probed schema; with no key nothing is started, logged or counted."""
    case = synthetic_case(2, 96, generator="fal")
    common.write_json(case / "gen" / "model.json", model.parse_schema(schema(), EP))
    saved = {k: os.environ.pop(k) for k in ("FAL_KEY", "FAL_KEY_ID") if k in os.environ}
    try:
        job = {"tag": "t", "family": "refine", "prompt": "p", "size": (2144, 2144),
               "refs": [case / "views" / "view_01" / "clay.png"], "out": case / "gen" / "x.png", "views": ["view_01"]}
        try:
            falio.edit(case, job)
        except common.StageError as e:
            assert "FAL_KEY is not set" in str(e)
        else:
            raise AssertionError("ran without a key")
        try:
            falio.edit(case, {**job, "size": (4992, 3328)})
        except common.StageError as e:
            assert "silently downsized" in str(e)
        else:
            raise AssertionError("over-cap size accepted")
        assert falio.read_log(case) == []
    finally:
        os.environ.update(saved)


def test_capabilities_are_never_guessed():
    vague = schema(x_fal=False)
    vague["components"]["schemas"]["In"]["properties"]["image_size"]["description"] = "Sides between 1024 and 4096 pixels."
    try:
        model.parse_schema(vague, EP)
    except common.StageError as e:
        assert "model.pixel_cap" in str(e)
    else:
        raise AssertionError("guessed a cap from loose numbers")
    m = model.parse_schema(vague, EP, {"pixel_cap": {"max_area": 2_000_000, "min_area": 0}})
    assert m["pixel_cap"]["max_area"] == 2_000_000 and m["capability_sources"]["pixel_cap"] == "override"
    nolimit = schema()
    nolimit["components"]["schemas"]["In"]["properties"]["image_urls"]["description"] = "reference images"
    try:
        model.parse_schema(nolimit, EP)
    except common.StageError as e:
        assert "model.max_refs" in str(e)
    else:
        raise AssertionError("guessed a reference limit")
    assert model.parse_schema(nolimit, EP, {"max_refs": 3})["max_images"] == 3
    renamed = schema()
    props = renamed["components"]["schemas"]["In"]["properties"]
    props["output_size"] = props.pop("image_size")
    m2 = model.parse_schema(renamed, EP)
    assert m2["size_param"] == "output_size" and "width, height" in m2["capability_sources"]["size_param"]
