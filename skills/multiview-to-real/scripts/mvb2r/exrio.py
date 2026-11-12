"""Pure-Python / numpy OpenEXR reader and writer (no OpenCV or OpenEXR codec needed).

Supported: single-part SCANLINE files, channels HALF / FLOAT / UINT with sampling 1, compression NONE, ZIPS
(1 scanline per block) and ZIP (16 scanlines per block), any line order (blocks are placed by their own y),
any dataWindow origin (the array covers the dataWindow). That is what the tool writes itself: Blender's
`passes.py` and the fusion half's EXR files are written with ZIP, and `write_exr` below writes ZIP or NONE.

Not supported (a clear StageError names the file): tiled, multi-part, deep, and RLE / PIZ / PXR24 / B44 / DWA
compression. Re-save such a file from Blender with Image > Save As, OpenEXR, Codec "ZIP" (or "None").

ZIP block layout: zlib-deflated bytes -> undo the predictor (running sum of byte - 128, mod 256) -> de-interleave
(the first half holds the even bytes, the second half the odd ones) -> for each scanline, each channel in
alphabetical order, `width` little-endian values.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = 20000630
PIXEL_TYPES = {0: ("<u4", 4), 1: ("<f2", 2), 2: ("<f4", 4)}
COMPRESSION_NAMES = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ", 5: "PXR24", 6: "B44", 7: "B44A",
                     8: "DWAA", 9: "DWAB"}
LINES_PER_BLOCK = {0: 1, 2: 1, 3: 16}
RESAVE = "re-save it from Blender as OpenEXR with Codec ZIP (or None), single layer, scanline"


class ExrError(ValueError):
    pass


# ---------------------------------------------------------------- byte transforms (pure)
def predictor_decode(t: np.ndarray) -> np.ndarray:
    """Running sum of (byte - 128) mod 256."""
    d = t.astype(np.int64) - 128
    d[0] = t[0]
    return (np.cumsum(d) & 0xFF).astype(np.uint8)


def predictor_encode(t: np.ndarray) -> np.ndarray:
    d = np.empty_like(t)
    d[0] = t[0]
    d[1:] = ((t[1:].astype(np.int64) - t[:-1].astype(np.int64) + 128 + 256) & 0xFF).astype(np.uint8)
    return d


def deinterleave(t: np.ndarray) -> np.ndarray:
    out = np.empty_like(t)
    half = (len(t) + 1) // 2
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out


def interleave(raw: np.ndarray) -> np.ndarray:
    return np.concatenate([raw[0::2], raw[1::2]])


def zip_decompress(data: bytes, raw_size: int) -> bytes:
    if len(data) >= raw_size:          # stored uncompressed: compression did not shrink this block
        return data[:raw_size]
    t = np.frombuffer(zlib.decompress(data), np.uint8)
    if len(t) != raw_size:
        raise ExrError(f"ZIP block inflates to {len(t)} bytes, expected {raw_size}")
    return deinterleave(predictor_decode(t)).tobytes()


def zip_compress(raw: bytes) -> bytes:
    t = np.frombuffer(raw, np.uint8)
    packed = zlib.compress(predictor_encode(interleave(t)).tobytes(), 6)
    return packed if len(packed) < len(raw) else raw


# ---------------------------------------------------------------- header
def _cstr(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def _channels(value: bytes) -> list[dict]:
    out, pos = [], 0
    while pos < len(value) and value[pos] != 0:
        name, pos = _cstr(value, pos)
        ptype, _plin, xs, ys = struct.unpack_from("<iB3xii", value, pos)
        pos += 16
        out.append({"name": name, "type": ptype, "xs": xs, "ys": ys})
    return out


def parse_header(buf: bytes, path="") -> tuple[dict, int]:
    """(header dict, offset of the offset table)."""
    if len(buf) < 8 or struct.unpack_from("<i", buf, 0)[0] != MAGIC:
        raise ExrError(f"{path}: not an OpenEXR file")
    flags = struct.unpack_from("<I", buf, 4)[0]
    for bit, what in ((0x200, "tiled"), (0x800, "deep"), (0x1000, "multi-part")):
        if flags & bit:
            raise ExrError(f"{path}: {what} EXR files are not supported; {RESAVE}")
    pos, hdr = 8, {}
    while buf[pos] != 0:
        name, pos = _cstr(buf, pos)
        typ, pos = _cstr(buf, pos)
        size = struct.unpack_from("<i", buf, pos)[0]
        value = buf[pos + 4:pos + 4 + size]
        pos += 4 + size
        if typ == "chlist":
            hdr[name] = _channels(value)
        elif typ == "box2i":
            hdr[name] = struct.unpack("<iiii", value)
        elif typ in ("compression", "lineOrder"):
            hdr[name] = value[0]
        else:
            hdr[name] = (typ, value)
    for key in ("channels", "compression", "dataWindow"):
        if key not in hdr:
            raise ExrError(f"{path}: header has no {key!r}")
    return hdr, pos + 1


# ---------------------------------------------------------------- read
def read_channels(path) -> tuple[dict, dict]:
    """({channel name: float32 (H, W)}, info) of a supported EXR file."""
    path = Path(path)
    buf = path.read_bytes()
    hdr, pos = parse_header(buf, path.name)
    comp = hdr["compression"]
    if comp not in LINES_PER_BLOCK:
        raise ExrError(f"{path.name}: compression {COMPRESSION_NAMES.get(comp, comp)} is not supported; {RESAVE}")
    chans = sorted(hdr["channels"], key=lambda c: c["name"])
    for c in chans:
        if c["type"] not in PIXEL_TYPES or c["xs"] != 1 or c["ys"] != 1:
            raise ExrError(f"{path.name}: channel {c['name']!r} has an unsupported type or sub-sampling; {RESAVE}")
    x0, y0, x1, y1 = hdr["dataWindow"]
    W, H = x1 - x0 + 1, y1 - y0 + 1
    if W <= 0 or H <= 0 or W * H > 1 << 28:
        raise ExrError(f"{path.name}: implausible dataWindow {hdr['dataWindow']}")
    lpb = LINES_PER_BLOCK[comp]
    n_blocks = -(-H // lpb)
    offsets = struct.unpack_from(f"<{n_blocks}Q", buf, pos)
    planes = {c["name"]: np.zeros((H, W), np.float32) for c in chans}
    row_bytes = sum(PIXEL_TYPES[c["type"]][1] for c in chans) * W
    for off in offsets:
        _read_block(buf, off, comp, chans, planes, (y0, H, W, lpb, row_bytes), path.name)
    info = {"width": W, "height": H, "data_window": list(hdr["dataWindow"]),
            "compression": COMPRESSION_NAMES[comp], "channels": [c["name"] for c in chans]}
    return planes, info


def _read_block(buf, off, comp, chans, planes, geom, name) -> None:
    y0, H, W, lpb, row_bytes = geom
    if off + 8 > len(buf):
        raise ExrError(f"{name}: offset table points past the end of the file (truncated?)")
    y, size = struct.unpack_from("<ii", buf, off)
    first = y - y0
    n = min(lpb, H - first)
    if first < 0 or n <= 0:
        raise ExrError(f"{name}: block for line {y} lies outside the dataWindow")
    data = buf[off + 8:off + 8 + size]
    raw = data if comp == 0 else zip_decompress(data, n * row_bytes)
    if len(raw) != n * row_bytes:
        raise ExrError(f"{name}: block at line {y} has {len(raw)} bytes, expected {n * row_bytes}")
    pos = 0
    for r in range(n):
        for c in chans:
            dt, nb = PIXEL_TYPES[c["type"]]
            planes[c["name"]][first + r] = np.frombuffer(raw, dt, W, pos).astype(np.float32)
            pos += nb * W


# ---------------------------------------------------------------- write
def _attr(name: str, typ: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(value)) + value


def write(path, planes: dict, compression: str = "ZIP", origin: tuple[int, int] = (0, 0)) -> None:
    """Write float32 channel planes {name: (H, W)} as a scanline EXR (ZIP or NONE)."""
    comp = {"NONE": 0, "ZIP": 3, "ZIPS": 2}.get(compression.upper())
    if comp is None:
        raise ExrError(f"cannot write compression {compression!r} (ZIP, ZIPS or NONE)")
    names = sorted(planes)
    arrs = [np.ascontiguousarray(planes[n], np.float32) for n in names]
    H, W = arrs[0].shape
    if any(a.shape != (H, W) for a in arrs):
        raise ExrError("all channels must have the same size")
    x0, y0 = origin
    chlist = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for n in names) + b"\0"
    box = struct.pack("<iiii", x0, y0, x0 + W - 1, y0 + H - 1)
    header = (struct.pack("<iI", MAGIC, 2) + _attr("channels", "chlist", chlist)
              + _attr("compression", "compression", bytes([comp])) + _attr("dataWindow", "box2i", box)
              + _attr("displayWindow", "box2i", box) + _attr("lineOrder", "lineOrder", b"\0")
              + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
              + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)) + b"\0")
    lpb = LINES_PER_BLOCK[comp]
    blocks = []
    for first in range(0, H, lpb):
        rows = range(first, min(H, first + lpb))
        raw = b"".join(a[r].astype("<f4").tobytes() for r in rows for a in arrs)
        data = raw if comp == 0 else zip_compress(raw)
        blocks.append(struct.pack("<ii", y0 + first, len(data)) + data)
    table_pos = len(header)
    offsets, pos = [], table_pos + 8 * len(blocks)
    for b in blocks:
        offsets.append(pos)
        pos += len(b)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(header + struct.pack(f"<{len(offsets)}Q", *offsets) + b"".join(blocks))
