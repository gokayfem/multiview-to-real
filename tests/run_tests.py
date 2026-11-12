#!/usr/bin/env python3
"""Run every test_* function in tests/test_*.py. No pytest needed.

  python3 tests/run_tests.py              # all tests (Blender ones skip with a message when Blender is absent)
  python3 tests/run_tests.py grid model   # only these modules

A test returns an optional message. A test raises Skip(reason) when what it needs is not there (Blender,
a module the other half has not shipped yet, an optional package); skips are printed, not failed.
"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "skills", "multiview-blockout-to-real", "scripts"))

from _util import Skip  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mods = sorted(f[5:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    if args:
        mods = [m for m in mods if m in args]
    failed, skipped, n = [], 0, 0
    for m in mods:
        try:
            mod = importlib.import_module(f"test_{m}")
        except Skip as e:
            skipped += 1
            print(f"  skip  {m}: {e}", flush=True)
            continue
        for name in sorted(k for k in dir(mod) if k.startswith("test_")):
            fn = getattr(mod, name)
            t0 = time.time()
            try:
                msg = fn()
                n += 1
                print(f"  ok    {m}.{name} ({time.time() - t0:.1f}s){'  ' + msg if msg else ''}", flush=True)
            except Skip as e:
                skipped += 1
                print(f"  skip  {m}.{name}: {e}", flush=True)
            except Exception:
                n += 1
                failed.append(f"{m}.{name}")
                print(f"  FAIL  {m}.{name} ({time.time() - t0:.1f}s)", flush=True)
                traceback.print_exc()
    print(f"{n - len(failed)}/{n} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
