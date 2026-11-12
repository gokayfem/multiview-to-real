# ChatGPT (Custom GPT)

1. Create a GPT and paste `adapters/universal-system-prompt.md` (the part below the line) into **Instructions**.
2. Upload as **Knowledge**: `skills/multiview-blockout-to-real/SKILL.md` and the files in
   `skills/multiview-blockout-to-real/references/`.
3. Enable **Code Interpreter** so it can read your contact sheets, coverage and consistency tables and run the
   plain-Python parts of `scripts/mvb2r/` (scoring, the consistency measure) on files you upload. Blender, fal
   and the renders run on your machine.
