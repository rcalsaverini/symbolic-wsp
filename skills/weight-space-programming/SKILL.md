---
name: weight-space-programming
description: Inspect and reversibly edit checkpoint weights with the symbolic-wsp CLI.
---

# weight-space-programming

Run `uv run symbolic-wsp inspect ID` and `probe --checkpoint ID --observations FILE`. State a mechanistic hypothesis before editing. Write JSON with mode rank_one, project_out, ablate, random or dsl; use `edit ID --spec FILE --hypothesis TEXT`. Preserve parent ID. Inspect `compare BASE CHILD` and manifests/history. Run paired validation before retaining; record rejected edits too. Use `rollback CHILD` and explicitly supply returned ID on later calls. No backpropagation, optimizers or fine-tuning. No arbitrary-code sandbox is provided. Never access final evaluator data.

All commands above use the `uv run symbolic-wsp` prefix. See `--help` for argument details.
