"""Tour transition: dip through black between the pan / orbit and the arc (pure fade arithmetic + faded copies)."""
import numpy as np

from _util import tmpdir


def test_dip_factors_go_to_black_at_the_cut():
    from mvb2r.tour import dip_factors
    f = dip_factors(48, 96, fps=24, fade_s=0.25)
    assert f.shape == (96,) and f.min() == 0.0 and f.max() == 1.0
    assert f[47] == 0.0 and f[48] == 0.0                         # black on both sides of the cut
    assert np.all(f[:42] == 1.0) and np.all(f[54:] == 1.0)       # 6 frames = 0.25 s at 24 fps each way
    assert np.all(np.diff(f[41:48]) < 0) and np.all(np.diff(f[48:55]) > 0)


def test_dip_factors_without_two_segments_change_nothing():
    from mvb2r.tour import dip_factors
    for n_first, n in ((0, 30), (30, 30), (0, 0)):
        assert np.all(dip_factors(n_first, n, fps=24, fade_s=0.25) == 1.0)


def test_dip_factors_clamp_a_fade_longer_than_a_segment():
    from mvb2r.tour import dip_factors
    f = dip_factors(3, 10, fps=24, fade_s=1.0)
    assert f[2] == 0.0 and f[3] == 0.0 and f[0] > 0 and np.all(np.diff(f[:3]) < 0) and np.all(np.diff(f[3:]) > 0)


def test_dip_factors_reject_bad_input():
    from mvb2r.common import StageError
    from mvb2r.tour import dip_factors
    for args in ((5, 3, 24, 0.25), (-1, 3, 24, 0.25), (1, 3, 0, 0.25), (1, 3, 24, -1.0)):
        try:
            dip_factors(*args[:2], fps=args[2], fade_s=args[3])
        except StageError:
            continue
        raise AssertionError(f"{args} accepted")


def test_faded_copy_leaves_rendered_frames_untouched():
    import cv2
    from mvb2r.tour import dip_factors, faded_copy
    src, dst = tmpdir() / "frames", tmpdir() / "faded"
    src.mkdir()
    for k in range(20):
        cv2.imwrite(str(src / f"frame_{k:04d}.png"), np.full((8, 10, 3), 200, np.uint8))
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    f = dip_factors(10, 20, fps=24, fade_s=0.125)
    faded_copy(src, dst, f)
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    out = [cv2.imread(str(dst / f"frame_{k:04d}.png")) for k in range(20)]
    assert out[9].max() == 0 and out[10].max() == 0 and out[0].min() == 200 and out[19].min() == 200
    assert 0 < out[8].mean() < 200


def test_dip_video_without_ffmpeg_keeps_todays_behaviour():
    from mvb2r import fusion_io, tour
    src = tmpdir() / "frames"
    src.mkdir()
    real = fusion_io.ffmpeg
    fusion_io.ffmpeg = lambda: None
    try:
        res, trans = tour.dip_video(src.parent, src, src.parent / "x.mp4", n_first=5)
    finally:
        fusion_io.ffmpeg = real
    assert res.startswith("frames only") and trans["kind"] == "none" and not (src.parent / "faded").exists()


def test_planted_dip_folder_link_is_refused_and_its_target_untouched():
    import cv2
    from mvb2r import fusion_io, tour
    from mvb2r.common import StageError
    case, victim = tmpdir(), tmpdir("mvb2r_victim_")
    (victim / "keep.txt").write_text("precious")
    frames = case / "out" / "tour_frames"
    frames.mkdir(parents=True)
    for k in range(10):
        cv2.imwrite(str(frames / f"frame_{k:04d}.png"), np.full((8, 8, 3), 200, np.uint8))
    (case / "out" / "_tour_frames_dip").symlink_to(victim, target_is_directory=True)
    real = fusion_io.ffmpeg
    fusion_io.ffmpeg = lambda: "/usr/bin/false"
    try:
        tour.dip_video(case, frames, case / "out" / "t.mp4", n_first=5)
    except StageError:
        pass
    else:
        raise AssertionError("planted link accepted")
    finally:
        fusion_io.ffmpeg = real
    assert sorted(p.name for p in victim.iterdir()) == ["keep.txt"] and (victim / "keep.txt").read_text() == "precious"


def test_patch_residue_compares_each_object_with_itself():
    """A dark floor with a bright wall behind it: the old inside / outside ratio mixed the two and read a clean
    floor as 0.2 and a floor with a leftover patch as about 1."""
    import numpy as np
    from mvb2r.tour import patch_residue
    ids = np.ones((40, 40), np.int32)
    ids[:20] = 2                                   # wall above, floor below
    lum = np.where(ids == 2, 0.5, 0.1).astype(np.float32)
    lit = np.zeros((40, 40), bool)
    lit[25:35, 10:30] = True                       # the original patch lies on the floor only
    drawn = np.ones((40, 40), bool)
    clean = patch_residue(lum, lit, ids, drawn)
    assert abs(clean["old_patch_residue"] - 1.0) < 1e-6 and list(clean["old_patch_per_object"]) == [1]
    left = lum.copy()
    left[lit] = 0.2                                # a leftover patch twice as bright as the floor around it
    ghost = patch_residue(left, lit, ids, drawn)
    assert abs(ghost["old_patch_residue"] - 2.0) < 1e-6
    assert patch_residue(lum, np.zeros_like(lit), ids, drawn)["old_patch_residue"] is None

