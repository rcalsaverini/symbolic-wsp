# Executed verification

For the subsequent real pretrained-model baseline study, see the
[feasibility experiment](../experiments/pretrained-baseline-01/README.md).
The checks below describe the initial toolkit implementation.

Executed on CPU with Python 3.12.14:

- `pytest -q`: **24 passed** (including optional Transformers integration).
- Ruff lint and format checks passed for source, scripts and tests.
- End-to-end CLI demo: 10 validation functions, 3 repeats, 8 candidates;
  240 model generations for each of five conditions.
- Targeted edit versus baseline: mean held-out NMSE delta 0, clustered bootstrap
  CI [0, 0]. Edit rejected, parent rollback verified. This is a negative synthetic
  result, not evidence about pretrained models.
- Offline Transformers test used a locally initialized one-layer GPT-2 (8 hidden
  dimensions) and a tiny WordLevel tokenizer. It exercised actual forward passes,
  constrained generation, hidden/logit probes, full-state snapshot, ablation,
  edited-state loading and native checkpoint/tokenizer export. Zero model downloads.
- Source research copies were checked byte-for-byte against the uploads.
- Both skill symlinks resolve to shared SKILL.md sources; frontmatter checked.

Environment-only optional test dependencies: torch 2.15.0+cpu, transformers 4.57.6.
To reproduce optional tests, install the HF extra. For CPU-only machines one can
install torch from https://download.pytorch.org/whl/cpu before installing
Transformers in the same environment. The default locked install skips the single
HF integration test if torch/Transformers are absent. No model downloads are used
by that test even when present.

Not executed: real pretrained SmolLM2/Qwen inference, GPU execution, interactive
Claude Code/Codex sessions, or deployment of a separate-UID final evaluator. CI
workflow is supplied but has not run on GitHub. No capability improvements claimed.

Recorded demo files are under `examples/recorded-demo/`. These are reproducible
synthetic validation data, not final test data.

GitHub repository creation was attempted for `rcalsaverini/symbolic-wsp`, but the
integration returned `Resource not accessible by integration (createRepository)`.
No upstream writes were attempted. Publish this independent local Git repository
with an account token permitted to create repositories:

```bash
gh repo create rcalsaverini/symbolic-wsp --private --source . --remote origin --push
```

Alternatively create the empty repository in GitHub, then:

```bash
git remote add origin https://github.com/rcalsaverini/symbolic-wsp.git
git push -u origin main
```
