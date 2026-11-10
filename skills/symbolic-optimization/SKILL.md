---
name: symbolic-optimization
description: Run controlled symbolic-regression experiments using symbolic-wsp.
---

# symbolic-optimization

Read docs/methodology.md. Generate development tasks using `uv run symbolic-wsp dataset --split dev --output runs/dev.json`. Freeze expression inventory, prompt, K and seeds. Use `run --checkpoint ID --dataset FILE --output RESULT` and condition baseline, feedback, wsp, random or no_edit. Propose edits only from development diagnostics; use validation for acceptance. Run `paired BEFORE AFTER`; require five paired tasks, no failures and upper clustered bootstrap CI below zero. Record proposal cost separately and log all attempts. A dedicated evaluator account owns final data. Never claim reusable symbolic optimization from one successful edit. Run `demo --output runs/fresh-demo` for executable infrastructure acceptance.

All commands above use the `uv run symbolic-wsp` prefix. See `--help` for argument details.
