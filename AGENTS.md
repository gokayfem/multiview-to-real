# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini) working on this repository. To use the
skill rather than develop it, see the README.

## Layout

```text
skills/multiview-blockout-to-real/   the skill (this is what gets installed)
  SKILL.md                           entry point: input routing, workflow, decision rules, hard rules, reference routing
  references/                        conventions, inputs, cameras, generation, consistency, projection, seams,
                                     geometry, pbr, validation, lessons
  templates/                         CONTRACT.md (copied into every case), spec.example.json, acceptance.json,
                                     case.example.json
  scripts/mvb2r.py                   CLI: doctor | new | cameras | views | grid | probe | generate | refine | targets |
                                     consistency | project | score | resync | refit | pbr | render | critique |
                                     report | run
  scripts/mvb2r/                     importable library (numpy, scipy, scikit-image, Pillow, opencv; fal-client)
    common      the data contract in code: case paths, JSON IO, Blender runner, image IO, logging, names
    falio       every image-model call: cache, budget, one attempt, log in gen/calls.jsonl, downloads checked
    mockgen     the offline synthetic image model (--generator mock) used by the tests and demos
    case        `new` and `doctor`; inputs (ingest routing), cameras, views, verify, grid, model (`probe`),
                prompts, generate, cells, refine, qa, masks, targets, consistency, resync, report (`report`
                and `critique`)
    fusion side pinhole, raster, visibility, projinputs, texsize, fusion, fusion_io, regions, levels, seams,
                coverage, texlookup, project, score, sheets, refit, voxcarve, delight, materials (with
                material_priors.json), rigfit, pbr, render, tour
  scripts/blender/                   scripts run inside Blender 4.2+: ingest, blockout, place_cameras, passes, clay
                                     (core); export_mesh, build_diagnostic, build_pbr, render, shading_basis, carve,
                                     _fusion_bpy (fusion)
tests/                               python3 tests/run_tests.py [module ...]; Blender round trips skip without Blender;
                                     roundtrip/ holds their scene builders; CI in .github/workflows/checks.yml
adapters/                            copy-in prompts for tools without skill support
gallery/                             README images from the first run
.claude-plugin/                      Claude Code plugin + marketplace manifests
```

## Commands

```bash
python3 tests/run_tests.py                                         # offline, no key; Blender tests skip without Blender
python3 tests/run_tests.py grid model                              # only these test modules
python3 skills/multiview-blockout-to-real/scripts/mvb2r.py --help
python3 skills/multiview-blockout-to-real/scripts/mvb2r.py doctor
```

Every stage is a subcommand: `mvb2r.py <stage> CASE [options]`. Each module that owns a subcommand exposes
`register(subparsers)` and `run(args) -> int`. Blender stages run as `blender -b --factory-startup
--disable-autoexec --python scripts/blender/<script>.py -- ...` through `mvb2r.common`.

## Rules for changes

- Input-agnostic: no scene, subject, prompt, number of views, object name, path or image model hardcoded in
  `skills/`. Everything scene-specific lives in a case's `case.json`.
- Keep the data contracts in `references/conventions.md` stable (coordinates, depth, pixel convention, case
  layout, `case.json`); `scripts/mvb2r/common.py` is their code form (the fusion half uses the small
  `fusion_io` helpers, which defer to `common` where both offer the same thing). Modules talk only through files
  in the case folder and through these helpers.
- A stage that lacks an input fails with a message naming the missing file and the command that produces it.
- Honesty in code: a failed call or render stops the stage and writes nothing that could pass for a result; a
  resize is recorded as a resize; depth is camera-axis Z and checked; inferred texels are flagged; the
  diagnostic `.blend` says it contains lighting; a `$0` price is "unknown".
- fal: only parameters present in the endpoint's fetched schema; every call logged; cached; budgeted; no
  automatic retries.
- Untrusted input: `.blend` files opened with auto-execution disabled; names sanitised before they become
  paths; nothing from a file or a model response is executed; downloads size-limited and type-checked.
- Small files (200 to 400 lines typical, 800 at most), small functions, no mutation of inputs, explicit errors.
- No outputs in the repository (cases, renders, `.blend` files, reports): `cases/` is git-ignored. The README
  images in `gallery/` are the only images. No third-party assets are bundled.
- Every new module gets a test in `tests/` against a closed form, a synthetic scene or the mock generator.
  Tests run without fal and without a key.
- No secrets in files or logs. `FAL_KEY` is read from the environment and never printed.
