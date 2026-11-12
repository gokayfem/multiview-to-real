"""World-space shapes of case.json (sub-object regions, contact-zone limits): validation and point tests.

    {"sphere": [x, y, z, r]}                 metres, r > 0
    {"box": [[x0, y0, z0], [x1, y1, z1]]}    metres, axis-aligned, x0 <= x1, y0 <= y1, z0 <= z1
"""
from __future__ import annotations

import math

import numpy as np

from mvb2r.common import clean_text
from mvb2r.fusion_io import StageError

MAX_ECHO = 80            # characters of an input value repeated in an error message
MAX_NAME = 40            # characters of an object / view name built into a message
MAX_ITEMS = 64           # entries of fusion.contacts and of one object's regions
MAX_COORD_M = 1000.0     # |coordinate| and sphere radius / box extent limit (metres)


def show(v) -> str:
    """An input value for an error message: one printable line, at most MAX_ECHO characters."""
    return clean_text(repr(v), MAX_ECHO)


def label(name) -> str:
    """A name from an input file for an error message: one printable line, at most MAX_NAME characters."""
    return clean_text(name, MAX_NAME)


def capped_list(where: str, v, limit: int = MAX_ITEMS) -> list:
    """`v` as a list of at most `limit` entries, else StageError."""
    if not isinstance(v, list):
        raise StageError(f"case.json {where} must be a list, got {show(v)}")
    if len(v) > limit:
        raise StageError(f"case.json {where} has {len(v)} entries; at most {limit} are supported")
    return v


def finite(where: str, v, minimum: float | None = None, positive: bool = False,
           maximum: float = MAX_COORD_M) -> float:
    """A finite number (not bool) with |v| <= maximum, optionally >= minimum or > 0, else StageError."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise StageError(f"case.json {where} must be a finite number, got {show(v)}")
    if positive and v <= 0:
        raise StageError(f"case.json {where} must be > 0, got {show(v)}")
    if minimum is not None and v < minimum:
        raise StageError(f"case.json {where} must be >= {minimum}, got {show(v)}")
    if abs(v) > maximum:
        raise StageError(f"case.json {where} must be at most {maximum:g} in magnitude, got {show(v)}")
    return float(v)


def box(where: str, v) -> tuple:
    """Validated ((x0, y0, z0), (x1, y1, z1))."""
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (list, tuple)) and len(c) == 3
                                                                  for c in v)):
        raise StageError(f"case.json {where} must be [[x0, y0, z0], [x1, y1, z1]] (metres), got {show(v)}")
    lo = tuple(finite(f"{where}[0][{i}]", x) for i, x in enumerate(v[0]))
    hi = tuple(finite(f"{where}[1][{i}]", x) for i, x in enumerate(v[1]))
    if any(a > b for a, b in zip(lo, hi)):
        raise StageError(f"case.json {where}: the first corner must be the minimum corner, got {show(v)}")
    return lo, hi


def sphere(where: str, v) -> tuple:
    """Validated (x, y, z, r)."""
    if not (isinstance(v, (list, tuple)) and len(v) == 4):
        raise StageError(f"case.json {where} must be [x, y, z, r] (metres), got {show(v)}")
    c = tuple(finite(f"{where}[{i}]", x) for i, x in enumerate(v[:3]))
    return c + (finite(f"{where}[3] (radius)", v[3], positive=True),)


def inside_box(P: np.ndarray, b: tuple | None) -> np.ndarray:
    if b is None:
        return np.ones(len(P), bool)
    lo, hi = np.asarray(b[0]), np.asarray(b[1])
    return np.all((P >= lo) & (P <= hi), 1)


def inside_sphere(P: np.ndarray, s: tuple) -> np.ndarray:
    return np.sum((P - np.asarray(s[:3])) ** 2, 1) <= s[3] ** 2
