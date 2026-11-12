#!/usr/bin/env python3
"""Multiview Blockout 2 Real CLI: any grey 3D scene -> the same scene textured from several generated views.

Every stage is a subcommand that runs alone; `run` chains them within the budget. Stage modules live in
scripts/mvb2r/ and expose register(subparsers) and run(args). Stages owned by the fusion half (project, score,
refit, pbr, render) are registered when their modules import; an import failure is printed.

  mvb2r.py doctor
  mvb2r.py new CASE --input X [--look ...] [--views N] [--closeup OBJ] [--reference IMG=role ...]
  mvb2r.py cameras | views | probe | grid | generate | refine | targets | consistency | resync CASE
  mvb2r.py report | critique CASE
  mvb2r.py run CASE [--from STAGE] [--to STAGE] [--generator mock] [--max-calls N] [--schema-file F] [--resync]

`run` order: views, probe, grid, generate, refine, targets, consistency, project, score, [resync with --resync],
pbr, render, report. `probe --check-size` (one paid call) is a separate, explicit step after `grid`.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvb2r import __version__, common  # noqa: E402

CORE = ("case", "cameras", "views", "grid", "model", "generate", "refine", "targets", "consistency", "resync", "report")
FUSION = ("project", "score", "refit", "pbr", "render")
RUN_ORDER = ("views", "probe", "grid", "generate", "refine", "targets", "consistency", "project", "score",
             "resync", "pbr", "render", "report")
OPT_IN = {"resync"}
OFFLINE_SKIP = {"probe"}
GEN_STAGES = ("probe", "grid", "generate", "refine", "resync")
FLAG_WORDS = ("WARNING", "FAIL", "rejected", "REJECTED", "fallback", "STALE", "budget", "error:")
DOWNSTREAM = ("proj/coverage.json", "proj/score.json", "out/diagnostic.blend", "pbr/eval.json", "out/final.blend",
              "out/tour.json")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mvb2r.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"multiview-blockout-to-real {__version__}")
    ap.add_argument("--debug", action="store_true", help="print tracebacks of unexpected errors")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in CORE:
        importlib.import_module(f"mvb2r.{name}").register(sub)
    for name in FUSION:
        try:
            mod = importlib.import_module(f"mvb2r.{name}")
        except ModuleNotFoundError as e:
            if e.name != f"mvb2r.{name}":
                print(f"warning: stage {name!r} unavailable: {e}", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001 - a broken optional module must not hide the core CLI
            print(f"warning: stage {name!r} unavailable: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if hasattr(mod, "register"):
            mod.register(sub)
    r = sub.add_parser("run", help="the stages in order within the budget")
    r.add_argument("case")
    r.add_argument("--from", dest="first", default=RUN_ORDER[0], choices=RUN_ORDER)
    r.add_argument("--to", dest="last", default="report", choices=RUN_ORDER)
    r.add_argument("--generator", choices=("fal", "mock"), default=None)
    r.add_argument("--max-calls", type=int, default=None)
    r.add_argument("--schema-file", default=None, help="passed to probe (a saved OpenAPI schema)")
    r.add_argument("--resync", action="store_true", help="include the paid render-then-refine round (opt-in)")
    r.set_defaults(func=None)
    return ap


class _Tee:
    """Copy of everything a stage prints, so `run` can summarise its warnings."""

    def __init__(self, stream):
        self.stream, self.lines = stream, []

    def write(self, text):
        self.stream.write(text)
        self.lines.extend(t for t in text.splitlines() if t.strip())
        return len(text)

    def flush(self):
        self.stream.flush()


def flags_of(lines: list[str]) -> list[str]:
    """Lines carrying an explicit marker the stages emit: WARNING, FAIL, rejected, fallback, STALE, budget, error: (pure)."""
    return [ln.strip()[:160] for ln in lines if any(w in ln for w in FLAG_WORDS)]


def stale_outputs(case: Path) -> list[str]:
    """Downstream files older than the current targets (e.g. after a resync promotion) (pure on the files)."""
    t = case / "targets" / "targets.json"
    if not t.is_file():
        return []
    ref = t.stat().st_mtime
    return [p for p in DOWNSTREAM if (case / p).is_file() and (case / p).stat().st_mtime < ref]


def _stage_argv(stage: str, case, args) -> list[str]:
    argv = [stage, str(case)]
    if stage in GEN_STAGES and args.generator:
        argv += ["--generator", args.generator]
    if stage in ("generate", "refine", "resync") and args.max_calls is not None:
        argv += ["--max-calls", str(args.max_calls)]
    if stage == "probe" and args.schema_file:
        argv += ["--schema-file", str(args.schema_file)]
    return argv


def call_stage(func, sub, debug: bool = False) -> int:
    """Run one stage; any error becomes a one-line message and a non-zero exit (traceback with --debug)."""
    try:
        return int(func(sub) or 0)
    except common.StageError as e:
        common.say(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001 - reported, never swallowed
        common.say(f"error: {type(e).__name__}: {str(e)[:300]}" + ("" if debug else "  (--debug for the traceback)"))
        if debug:
            traceback.print_exc()
        return 3


def _run_one(ap, stage, case, args) -> tuple[int, list[str]]:
    try:
        sub = ap.parse_args(_stage_argv(stage, case, args))
    except SystemExit:
        return 2, [f"stage {stage} is not available (see the warning printed above)"]
    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        common.say(f"[run] {stage} ...")
        rc = call_stage(sub.func, sub, getattr(args, "debug", False))
    return rc, flags_of(tee.lines)


def _status(rc: int, flags: list[str]) -> str:
    if rc:
        return f"FAILED (exit {rc})"
    return f"WARNING ({len(flags)})" if flags else "ok"


def run_chain(ap: argparse.ArgumentParser, args) -> int:
    case = common.case_dir(args.case)
    gen = args.generator or (common.load_case(case).get("model") or {}).get("generator", "fal")
    todo = list(RUN_ORDER[RUN_ORDER.index(args.first): RUN_ORDER.index(args.last) + 1])
    summary, rc = [], 0
    for stage in todo:
        if stage in OPT_IN and not args.resync:
            summary.append({"stage": stage, "status": "skipped (opt-in: run --resync, or run it by hand)", "flags": []})
            continue
        if gen == "mock" and stage in OFFLINE_SKIP:
            summary.append({"stage": stage, "status": "skipped (mock generator: no schema to probe)", "flags": []})
            continue
        rc, flags = _run_one(ap, stage, case, args)
        summary.append({"stage": stage, "status": _status(rc, flags), "flags": flags})
        if rc:
            break
    stale = stale_outputs(case)
    common.write_json(case / "logs" / "run_summary.json", {"generator": gen, "stages": summary, "stale": stale})
    common.say("\n[run] summary (also logs/run_summary.json):")
    for s in summary:
        common.say(f"  {s['stage']:<12} {s['status']}")
        for f in s["flags"][:6]:
            common.say(f"      ! {f}")
    if stale:
        common.say(f"  STALE (older than targets/): {', '.join(stale)} - refresh with `mvb2r.py run {case} --from project`")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "run":
        try:
            return run_chain(ap, args)
        except common.StageError as e:
            common.say(f"error: {e}")
            return 2
    return call_stage(args.func, args, args.debug)


if __name__ == "__main__":
    sys.exit(main())
