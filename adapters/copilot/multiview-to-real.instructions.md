---
applyTo: "**/*.py"
---

Copy this file into your repository's `.github/instructions/`. For agent mode, also install the skill with
`gh skill install gokayfem/multiview-to-real multiview-to-real`.

- Texture grey scenes from several views with the `multiview-to-real` skill: `mvb2r.py new` → write
  `case.json` (look, references, descriptions, classes, flags) → `cameras` → `views` → `probe` → `grid` →
  `generate` → `refine` → `targets` → `consistency` → `project` → `score` → (opt-in `resync` or `refit`, then re-run what follows) → `pbr` →
  `render` → `critique` (fresh reviewer) → `report`.
- Depth is camera-axis Z; sizes are read from the returned files; a resize is never generated detail.
- Nothing generated is composited into a render; coverage is reported per object and inferred surfaces are
  labelled.
- Paid calls go through the case budget and are logged; `FAL_KEY` is never printed.
