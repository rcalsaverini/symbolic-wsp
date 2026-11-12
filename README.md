# symbolic-wsp

Independent, harness-neutral research workbench for reversible weight interventions
and symbolic regression. Python engine + JSON CLI + shared Claude Code/Codex skills.
No Pi dependency, gradient descent, PPO, PointAdapter or vocabulary surgery.

This is a working research prototype. The default demonstration uses a synthetic
categorical neural head, **not a pretrained LLM**. WSP improvement is a hypothesis.

## Research experiment

The intended experiment asks whether direct, gradient-free edits to a pretrained
model improve its symbolic-expression proposals on unseen functions, under matched
generation budgets. See the detailed [experiment proposal](docs/wsp-symbolic-optimization-experiment.md)
for hypotheses, controls, data isolation, a proposed pilot, interpretation criteria,
and the distinction between today's finite-catalog experiment and future structural
symbolic search. The companion [interpretation and caveats document](docs/interpreting-wsp-improvements.md)
examines capability acquisition, execution repair, compiled programs, and controls
motivated by external investigations. These are research plans, not reports of
pretrained-model gains.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pytest -q
uv run symbolic-wsp demo --output runs/demo-01
cat runs/demo-01/summary.json
```

The demo runs baseline, feedback, targeted edit, matched-norm random and no-edit
conditions, records 240 generations each, compares validation results, records its
retention/rejection decision and rolls back. Use a fresh output directory each run.
Checkpoint manifests/history live in `runs/checkpoints/`.

## Pretrained baseline feasibility test

A reproducible ten-function test compares a pinned SmolLM2-135M-Instruct model
with observation ablation, uniform sampling, and exhaustive catalog selection.
See the [protocol and reproduction instructions](experiments/pretrained-baseline-01/README.md)
for the frozen configuration and execution details. This tests catalog selection,
not weight editing or open-ended symbolic synthesis.

## First-class baseline diagnostics

The [diagnostic suite](experiments/baseline-diagnostics-01/README.md) compares exact
catalog probabilities, four observation controls, three explicit prompt variants,
and pinned 135M/360M models. It also runs a development-gated rank-one intervention
with random, restored no-edit, and first-token logit-bias controls.

```bash
uv run symbolic-wsp diagnostics --small-model LOCAL_135M --large-model LOCAL_360M --output runs/diagnostics --phase all
```

Development selection is frozen before evaluation. Use the documented preparation
commands for local model weights; default tests never download them.

## CLI and Python

```bash
uv run symbolic-wsp --help
uv run symbolic-wsp dataset --split dev --seed 17 --count 10 --output runs/dev.json
uv run symbolic-wsp import model.safetensors
uv run symbolic-wsp inspect CHECKPOINT_ID
uv run symbolic-wsp edit CHECKPOINT_ID --spec examples/rank-one.json --hypothesis 'Test a small output-direction intervention'
uv run symbolic-wsp compare BASE_ID CHILD_ID
uv run symbolic-wsp rollback CHILD_ID
uv run symbolic-wsp export CHECKPOINT_ID exported.safetensors
uv run symbolic-wsp run --checkpoint CHECKPOINT_ID --dataset runs/dev.json --output runs/base.json
uv run symbolic-wsp paired runs/base.json runs/edited.json
```

Example edit uses the demo head schema; adapt vectors/tensor name for other models.
Place global `--store PATH` before the subcommand. Output JSON is stable schema 1
for datasets/runs; errors exit 2 with structured JSON on stderr. API example:

```python
from symbolic_wsp.wsp.checkpoints import Store
from symbolic_wsp.wsp.editing import edit, rollback
from symbolic_wsp.symbolic.evaluator import score
print(score('sin(x)', [0, 1], [0, 0.84]))
```

## Optional real Transformers backend

```bash
uv sync --extra hf --dev
uv run symbolic-wsp hf-snapshot --model HuggingFaceTB/SmolLM2-135M-Instruct
uv run symbolic-wsp run --backend hf --model LOCAL_MODEL_DIRECTORY --checkpoint BASE_ID --dataset runs/dev.json --output runs/hf-base.json -k 4
uv run symbolic-wsp probe --backend hf --model LOCAL_MODEL_DIRECTORY --checkpoint BASE_ID --observations examples/observations.json
uv run symbolic-wsp hf-export --backend hf --model LOCAL_MODEL_DIRECTORY --checkpoint CHILD_ID --output runs/hf-edited
```

Snapshot and reuse a locally pinned model/tokenizer directory. `--device cuda`
uses GPU in the CLI worker process. No model downloads occur in the default tests.
Generation grammar defaults to 14 expressions; `--expressions inventory.json`
replaces it with an explicit JSON string list. This finite grammar guarantees
valid output but strongly limits search: do not confuse inventory selection with
open-ended symbolic synthesis. HF integration checks and limitations are described
in `docs/verification.md`; unexecuted checks are explicitly identified there.

Research originals are byte-preserved under `research/`. See
[architecture](docs/architecture.md), [methodology](docs/methodology.md),
[upstream analysis](docs/upstream-analysis.md), and [agent setup](docs/agent-setup.md).
Apache-2.0 code; upstream notices retained in NOTICE. Research documents retain
their author's original rights; they are not relicensed by the code license.
