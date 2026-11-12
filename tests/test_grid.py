"""Grid layout, cell sizes under the pixel cap, references manifest, layout sentence, seam detection."""
import numpy as np

from mvb2r import cells, grid

SEEDREAM_CAP = {"max_area": 4624220, "min_area": 921600, "max_side": 14142}


def test_grid_shape_for_every_supported_count():
    want = {2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 3)}
    for n, rc in want.items():
        assert grid.grid_shape(n) == rc, (n, grid.grid_shape(n))
    assert grid.grid_shape(4, 16 / 9) == (2, 2)
    for bad in (1, 10):
        try:
            grid.grid_shape(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_cell_size_fits_the_cap_exactly():
    for n in range(2, 10):
        for aspect in ((1664, 1664), (1920, 1080), (1080, 1350)):
            r, c = grid.grid_shape(n, aspect[0] / aspect[1])
            cw, ch = grid.cell_size(*aspect, r, c, SEEDREAM_CAP)
            assert cw % 8 == 0 and ch % 8 == 0
            assert r * c * cw * ch <= SEEDREAM_CAP["max_area"]
            assert abs(cw / ch - aspect[0] / aspect[1]) < 0.03, (n, aspect, cw, ch)
            assert r * c * (cw + 8) * (ch + 8) > SEEDREAM_CAP["max_area"] * 0.97   # nearly the largest
    assert grid.cell_size(1664, 1664, 2, 3, SEEDREAM_CAP) == (872, 872)
    try:
        grid.cell_size(100, 100, 3, 3, {"max_area": 50_000, "min_area": 921600})
    except ValueError as e:
        assert "minimum area" in str(e)
    else:
        raise AssertionError


def test_cells_tile_the_sheet_exactly():
    rects = grid.cell_rects(2, 3, 40, 30)
    cover = np.zeros((60, 120), int)
    for r in rects:
        cover[r["y0"]:r["y1"], r["x0"]:r["x1"]] += 1
    assert (cover == 1).all() and [r["index"] for r in rects] == list(range(1, 7))
    imgs = [np.full((30, 40, 3), 10 * i, np.uint8) for i in range(5)]
    sheet = grid.assemble(imgs, 2, 3, 40, 30)
    grid.check_exact(sheet, imgs, 2, 3, 40, 30)
    assert (sheet[30:, 80:] == grid.BLANK_GREY).all()


def _ref(path, kind, **kw):
    return {"path": path, "kind": kind, "sha256": path, **kw}


def test_references_drop_clays_from_the_last_view_and_stay_distinct():
    sheet = _ref("grid.png", "grid")
    users = [_ref("ref1.png", "reference", role="identity")]
    clays = [_ref(f"clay{i}.png", "clay", view=f"view_0{i}", view_index=i) for i in range(1, 10)]
    refs, dropped = grid.order_references(sheet, users, clays, 10)
    assert [r["image"] for r in refs] == list(range(1, 11)) and refs[0]["kind"] == "grid"
    assert [d["view"] for d in dropped] == ["view_09"]
    grid.check_distinct(refs)
    s = grid.layout_sentence(3, 3, 9, refs)
    assert "Images 3 to 10 are higher-resolution copies of panels 1 to 8" in s, s
    try:
        grid.check_distinct(refs + [refs[1]])
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate accepted")
    try:
        grid.order_references(sheet, users * 10, clays, 10)
    except ValueError:
        return
    raise AssertionError


def test_layout_sentence_counts():
    refs = [{"image": 1, "kind": "grid"}] + [{"image": i + 2, "kind": "clay", "view_index": i + 1} for i in range(5)]
    s = grid.layout_sentence(2, 3, 5, refs)
    assert "exactly 2 rows x 3 columns" in s and "Images 2 to 6" in s and "panels 1 to 5" in s
    assert "Panels 6 to 6 are flat grey" in s and "Output exactly 6 panels" in s


def test_seams_are_found_where_the_panels_change():
    rng = np.random.default_rng(1)
    h, w = 300, 450
    sheet = np.zeros((h, w), np.float32)
    xs, ys = [0, 143, 302, w], [0, 152, h]
    for r in range(2):
        for c in range(3):
            sheet[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = rng.uniform(40, 200) + rng.normal(0, 3, (ys[r + 1] - ys[r], xs[c + 1] - xs[c]))
    det = cells.detect(sheet, 2, 3)
    rects = cells.rects_from_seams(det)
    assert rects[0] == (0, 0, 143, 152) and rects[5] == (302, 152, 450, 300), rects
    flat = cells.detect(np.full((100, 300), 7.0, np.float32), 1, 3)
    assert all(s["used"].startswith("ideal") for s in flat["col_seams"])
    assert cells.rects_from_seams(flat) == [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)]


def test_layout_is_automatic_unless_a_person_set_it():
    assert grid.choose_layout({"rows": 2, "cols": 3}, 4, 1.0)[:2] == (2, 2)          # stale automatic values ignored
    assert grid.choose_layout({"rows": 1, "cols": 4, "source": "user"}, 4, 1.0)[:2] == (1, 4)
    try:
        grid.choose_layout({"rows": 2, "cols": 3, "source": "user"}, 7, 1.0)
    except Exception as e:
        assert "cannot hold" in str(e) and "grid.source" in str(e)
    else:
        raise AssertionError
