"""Block quilting for sunpatch.py: fill a mask component with blocks copied from its shaded, observed source.

Every block (with its overlap) is copied whole from a window lying entirely inside the source ERODED by the
overlap, so no copied texel is a source-edge texel, plain inferred fill or a chart border. Candidate windows are
all such positions in the source (sampled every block/8 texels). Per block, offsets within half a block of the
left and upper neighbours' offsets are excluded, and so are source windows within half a block of those used by
the blocks in NEIGHBOURS (no visible repeat of one source spot in adjacent blocks or rows); the
CANDIDATE_EVAL candidates that shift least ACROSS the structure lines (offsets along the lines keep boards
continuous) are scored by the mean squared difference to what is already known in the block's window (blocks
placed before it, and the shaded surface around the component), as in image quilting, among the windows whose
mean luminance is within LEVEL_TOL of the target level at the block when one is given, and one candidate within
MATCH_TOL of the best score is drawn (seeded). Overlaps are blended with linear ramps. With `elongate` > 1, blocks are that much
longer ALONG the structure (boards, courses) when it is near an image axis, so copied planks are not cut short. Texels no block can
cover are retried with square blocks, then at half and quarter block size; what is left is returned as unsourced.
"""
from __future__ import annotations

import cv2
import numpy as np

OVERLAP_FRAC = 1 / 4       # block overlap (linear ramps) as a fraction of the block
CANDIDATE_EVAL = 64        # candidates (least across the structure) scored by overlap error per block
LEVEL_TOL = 1.5            # with a target level: source windows whose mean luminance is within this factor of
                           # the level at the block (if any), so the level match rescales, not repaints
MATCH_TOL = 0.1            # a block draws among candidates within (1 + this) x the best overlap error (seeded)
CANDIDATE_STEP_FRAC = 1 / 8  # candidate windows are sampled every block x this texels
NEIGHBOUR_MIN_FRAC = 0.5   # neighbouring blocks' offsets differ by at least this x the block (max-norm)
NEIGHBOURS = ((0, -1), (-1, 0), (-1, -1), (-1, 1), (-2, 0), (0, -2))  # (row, col) of blocks placed before
ELONGATED_MAX_REL_ERR = 0.5  # elongated block: best overlap error <= this x the variance of the known texels
AXIS_COS = 0.94            # elongated blocks only when the structure is within ~20 degrees of an image axis
SCALES = (1, 2, 4)         # block, then block/2, block/4 for texels the full block cannot cover
MIN_BLOCK = 8


def overlap(T: int) -> int:
    return max(int(T * OVERLAP_FRAC), 1)


def windows(src: np.ndarray, eh: int, ew: int, step: int) -> np.ndarray:
    """(N, 2) top-left (y, x) of eh x ew windows entirely inside `src`, on a grid of `step`."""
    ii = cv2.integral(src.astype(np.uint8))
    full = (ii[eh:, ew:] - ii[:-eh, ew:] - ii[eh:, :-ew] + ii[:-eh, :-ew]) == eh * ew
    ys, xs = np.nonzero(full[::step, ::step])
    return np.stack([ys * step, xs * step], -1) if len(ys) else np.zeros((0, 2), int)


def window_means(lum_img: np.ndarray, cands: np.ndarray, eh: int, ew: int) -> np.ndarray:
    ii = cv2.integral(lum_img.astype(np.float64))
    y, x = cands[:, 0], cands[:, 1]
    return (ii[y + eh, x + ew] - ii[y, x + ew] - ii[y + eh, x] + ii[y, x]) / (eh * ew)


def at_level(cands, cand_lvl, target: float):
    """Candidates within LEVEL_TOL of the target luminance (all of them when none is)."""
    if cand_lvl is None or not np.isfinite(target) or target <= 0:
        return cands
    ok = np.abs(np.log(np.maximum(cand_lvl, 1e-6) / target)) <= np.log(LEVEL_TOL)
    return cands[ok] if ok.any() else cands


def choose(cands, top_left, perp_yx, neighbours, min_dist, rng, match=None, max_rel_err=np.inf):
    """One offset (dy, dx) from source window to target, or None. `neighbours`: ((offset or None), source
    top-left) of the blocks around; `match` = (base, ky, kx, known values)."""
    off = cands - np.asarray(top_left)
    ok = np.ones(len(off), bool)
    for nb_off, nb_src in neighbours:
        if nb_off is not None:
            ok &= np.abs(off - nb_off).max(1) >= min_dist
        ok &= np.abs(cands - nb_src).max(1) >= min_dist
    off = off[ok]
    if not len(off):
        return None
    off = off[np.argsort(np.abs(off @ perp_yx), kind="stable")[:CANDIDATE_EVAL]]
    if match is not None and len(match[1]):
        base, ky, kx, val = match
        err = np.array([np.mean((base[ky + o[0], kx + o[1]] - val) ** 2) for o in off])
        if err.min() > max_rel_err * max(float(val.var(0).sum() / 3), 1e-12):
            return None                                   # no candidate continues what is known well enough
        off = off[err <= err.min() * (1 + MATCH_TOL) + 1e-12]
    return tuple(int(v) for v in off[int(rng.integers(len(off)))])


def _known(base, acc, wsum, outside, y0, y1, x0, x1):
    """Texels of the window already known: placed blocks (acc / wsum) and the shaded surface around."""
    w = wsum[y0:y1, x0:x1]
    placed, out = w > 0, outside[y0:y1, x0:x1]
    ky, kx = np.nonzero(placed | out)
    vals = np.where(placed[ky, kx, None], acc[y0:y1, x0:x1][ky, kx] / np.maximum(w[ky, kx], 1e-9)[:, None],
                    base[y0:y1, x0:x1][ky, kx])
    return ky + y0, kx + x0, vals


def _ramp(d: np.ndarray, T: int, ov: int) -> np.ndarray:
    return np.clip(np.minimum(d + ov + 1, T + ov - d) / (2 * ov + 1), 1e-3, 1.0)


def _blocks(todo: np.ndarray, th: int, tw: int):
    ys, xs = np.nonzero(todo)
    for i, by in enumerate(range(ys.min(), ys.max() + 1, th)):
        for j, bx in enumerate(range(xs.min(), xs.max() + 1, tw)):
            yield i, j, by, bx


def _candidates(base, src, th, tw, ov, level):
    er = cv2.erode(src.astype(np.uint8), np.ones((2 * ov + 1,) * 2, np.uint8)) > 0
    cands = windows(er, th + 2 * ov, tw + 2 * ov, max(int(min(th, tw) * CANDIDATE_STEP_FRAC), 1))
    lvl = None if level is None or not len(cands) else \
        window_means(base @ np.array([0.2126, 0.7152, 0.0722]), cands, th + 2 * ov, tw + 2 * ov)
    return cands, lvl


def quilt_pass(base, todo, src, along, size, rng, acc, wsum, log, level=None) -> None:
    """One block size (th, tw) over the `todo` texels (accumulates into acc / wsum, appends to log). An elongated
    block is placed only when it continues the known texels well (ELONGATED_MAX_REL_ERR)."""
    th, tw = size
    max_rel = ELONGATED_MAX_REL_ERR if th != tw else np.inf
    ov = overlap(min(th, tw))
    cands, cand_lvl = _candidates(base, src, th, tw, ov, level)
    if not len(cands) or not todo.any():
        return
    perp_yx, chosen, H, W = np.array([along[0], -along[1]]), {}, todo.shape[0], todo.shape[1]
    for i, j, by, bx in _blocks(todo, th, tw):
        win = (max(by - ov, 0), min(by + th + ov, H), max(bx - ov, 0), min(bx + tw + ov, W))
        ty, tx = np.nonzero(todo[win[0]:win[1], win[2]:win[3]])
        if not len(ty):
            continue
        nbs = [(chosen[(i + a, j + b)][0] if (a, b) in NEIGHBOURS[:2] else None, chosen[(i + a, j + b)][1])
               for a, b in NEIGHBOURS if (i + a, j + b) in chosen]
        tgt = np.nan if level is None else float(level[min(by + th // 2, H - 1), min(bx + tw // 2, W - 1)])
        o = choose(at_level(cands, cand_lvl, tgt), (by - ov, bx - ov), perp_yx, nbs, min(th, tw) * NEIGHBOUR_MIN_FRAC,
                   rng, (base, *_known(base, acc, wsum, src, *win)), max_rel)
        if o is None:
            continue
        chosen[(i, j)] = (o, (by - ov + o[0], bx - ov + o[1]))
        ty, tx = ty + win[0], tx + win[2]
        w = _ramp(ty - by, th, ov) * _ramp(tx - bx, tw, ov)
        acc[ty, tx] += base[ty + o[0], tx + o[1]] * w[:, None]
        wsum[ty, tx] += w
        log.append((int(by), int(bx), int(min(th, tw)), o[0], o[1]))


def block_shape(block: int, along, elongate: float) -> tuple[int, int]:
    """(th, tw): the block stretched `elongate` x along the structure when it is near an image axis."""
    if elongate <= 1 or max(abs(along[0]), abs(along[1])) < AXIS_COS:
        return block, block
    long = int(round(block * elongate))
    return (long, block) if abs(along[1]) > abs(along[0]) else (block, long)


def quilt(base, comp, src, along, block, rng, level=None, elongate: float = 1.0):
    """(filled copy of base, texels without a source, log of (by, bx, T, dy, dx) per copied block)."""
    acc, wsum = np.zeros(base.shape, np.float32), np.zeros(comp.shape, np.float32)
    log, todo = [], comp.copy()
    for s in SCALES:
        T = max(block // s, MIN_BLOCK)
        for shape in dict.fromkeys((block_shape(T, along, elongate), (T, T))):  # elongated first, then square
            quilt_pass(base, todo, src, along, shape, rng, acc, wsum, log, level)
            todo = comp & (wsum <= 0)
        if not todo.any() or T == MIN_BLOCK:
            break
    fill = base.copy()
    got = comp & (wsum > 0)
    fill[got] = acc[got] / wsum[got][:, None]
    return fill, comp & ~got, log
