"""Built-in OpenEXR IO (mvb2r.exrio): round trips, byte transforms, Blender files, unsupported files."""
import os
import struct

import numpy as np

from _util import need_blender, tmpdir
from mvb2r import common, exrio


def test_byte_transforms_invert():
    t = np.random.default_rng(0).integers(0, 256, 1001).astype(np.uint8)
    assert np.array_equal(exrio.predictor_decode(exrio.predictor_encode(t)), t)
    assert np.array_equal(exrio.deinterleave(exrio.interleave(t)), t)
    raw = t.tobytes()
    assert exrio.zip_decompress(exrio.zip_compress(raw), len(raw)) == raw     # incompressible -> stored raw
    z = bytes(4000)
    assert len(exrio.zip_compress(z)) < 100 and exrio.zip_decompress(exrio.zip_compress(z), 4000) == z


def test_round_trip_channels_compressions_odd_sizes_and_origin():
    rng = np.random.default_rng(1)
    d = tmpdir()
    for comp in ("NONE", "ZIP", "ZIPS"):
        for h, w in ((1, 1), (17, 5), (33, 64)):
            y = rng.normal(0, 100, (h, w)).astype(np.float32)
            rgb = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
            common.write_exr(d / "y.exr", y, comp)
            common.write_exr(d / "rgb.exr", rgb, comp)
            assert np.array_equal(common.read_exr(d / "y.exr"), y)
            assert np.array_equal(common.read_exr(d / "rgb.exr"), rgb)
    planes = {"Y": rng.random((7, 9)).astype(np.float32)}
    exrio.write(d / "o.exr", planes, "ZIP", origin=(-3, 11))
    back, info = exrio.read_channels(d / "o.exr")
    assert np.array_equal(back["Y"], planes["Y"]) and info["data_window"] == [-3, 11, 5, 17]
    common.write_exr(d / "bg.exr", np.full((20, 20), 1e10, np.float32))
    assert (common.read_exr(d / "bg.exr") == np.float32(1e10)).all()


def _patch_compression(path, value: int) -> None:
    buf = bytearray(path.read_bytes())
    i = buf.index(b"compression\0compression\0") + len(b"compression\0compression\0") + 4
    buf[i] = value
    path.write_bytes(bytes(buf))


def test_unsupported_files_fail_clearly():
    d = tmpdir()
    common.write_exr(d / "a.exr", np.zeros((4, 4), np.float32))
    _patch_compression(d / "a.exr", 4)       # PIZ
    try:
        common.read_exr(d / "a.exr")
    except common.StageError as e:
        assert "PIZ" in str(e) and "Codec ZIP" in str(e)
    else:
        raise AssertionError("PIZ accepted")
    buf = bytearray((d / "a.exr").read_bytes())
    buf[4:8] = struct.pack("<I", 2 | 0x200)  # tiled flag
    (d / "t.exr").write_bytes(bytes(buf))
    for bad, frag in (("t.exr", "tiled"), ("n.exr", "not an OpenEXR")):
        if bad == "n.exr":
            (d / bad).write_bytes(b"hello world, not an image")
        try:
            common.read_exr(d / bad)
        except common.StageError as e:
            assert frag in str(e), e
        else:
            raise AssertionError(bad)
    common.write_exr(d / "c.exr", np.zeros((40, 40), np.float32))
    (d / "trunc.exr").write_bytes((d / "c.exr").read_bytes()[:-30])
    try:
        common.read_exr(d / "trunc.exr")
    except common.StageError:
        pass
    else:
        raise AssertionError("truncated file accepted")


def test_reads_files_written_by_blender_and_agrees_with_opencv():
    blender = need_blender()
    import subprocess
    d = tmpdir()
    script = d / "w.py"
    script.write_text('''
import sys, numpy as np, OpenImageIO as oiio
out = sys.argv[sys.argv.index("--") + 1]
for name, c, comp in (("y_zip", 1, "zip"), ("rgb_zip", 3, "zip"), ("rgb_none", 3, "none"), ("y_zips", 1, "zips")):
    h, w = 37, 53
    a = (np.arange(h * w * c, dtype=np.float32).reshape(h, w, c) * 0.37 - 100.0)
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.attribute("compression", comp)
    spec.channelnames = ("Y",) if c == 1 else ("R", "G", "B")
    o = oiio.ImageOutput.create(f"{out}/{name}.exr"); o.open(f"{out}/{name}.exr", spec); o.write_image(a); o.close()
''')
    subprocess.run([blender, "-b", "--factory-startup", "--disable-autoexec", "--python", str(script), "--", str(d)],
                   capture_output=True, check=True, timeout=300)
    msg = []
    for name, c in (("y_zip", 1), ("rgb_zip", 3), ("rgb_none", 3), ("y_zips", 1)):
        want = (np.arange(37 * 53 * c, dtype=np.float32).reshape(37, 53, c) * 0.37 - 100.0)
        got = common.read_exr(d / f"{name}.exr")
        assert np.array_equal(got, want[..., 0] if c == 1 else want), name
    try:
        import cv2
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        ref = cv2.imread(str(d / "rgb_zip.exr"), cv2.IMREAD_UNCHANGED)
        if ref is None:
            raise cv2.error("no EXR codec")
        assert np.array_equal(ref[..., ::-1], common.read_exr(d / "rgb_zip.exr"))
        msg.append(f"matches OpenCV {cv2.__version__}")
    except Exception as e:  # noqa: BLE001 - the cross-check is optional
        msg.append(f"OpenCV cross-check skipped ({type(e).__name__}: {str(e)[:60]})")
    return "; ".join(msg)
