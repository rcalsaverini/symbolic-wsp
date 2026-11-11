# Pretrained baseline feasibility test

This is a development study of a frozen pretrained model, not a WSP intervention
or an independent final test. The ten functions and four comparisons follow the
baseline proposal discussed before execution. The committed `config.json` fixes the
model revision, inventory, generation seeds, dataset seed, K, and decoding settings.

## Questions

1. Can SmolLM2-135M-Instruct run reliably with the catalog-constrained decoder?
2. Does conditioning the generator on numerical observations improve selection?
3. How much does the finite inventory limit the attainable fit?

The five exactly representable functions are `sin(x)`, `cos(x)`, `x*x`, `exp(x)` and
`1/(x*x+1)`. Five outside-catalog functions are `sin(2*x)`, `cos(2*x)`, `2*x*x`,
`exp(x/2)` and `1/(x*x+2)`. Each has 24 observed points and 24 independent interpolation
points in [-1, 1], plus 24 extrapolation points in [1, 2]. There is no observation
noise. Targets are public development labels and are never passed to the generator.

## Conditions and selection

- **Conditioned:** pretrained model receives the observed x/y values as text.
- **No observations:** the same model receives the same instruction and chat
  formatting but a statement that numerical observations are unavailable.
- **Uniform:** sample with replacement from the same catalog.
- **Exhaustive:** evaluate all 14 entries and select by the observation objective.

Each sampling condition draws eight candidates for each of three seeds per task:
240 candidates per condition. Exhaustive search evaluates 140 candidates overall.
All selectors, including the no-observations control, receive observation scores.
Only proposal generation is ablated in that control; it is not a pipeline that
never sees data. Duplicate proposals are retained and scored again, without caching.

Use `NMSE + 0.001 * AST complexity` for observation selection, with lexical
expression order as a deterministic tie-break. Reserved points never participate
in selection. Report interpolation and extrapolation separately. Exhaustive search
minimizes the observation objective, not held-out error. Coverage also reports the
minimum observation NMSE separately from the complexity-penalized selector.

## Prompt and generation details

Use the model's native chat template with an assistant generation prompt. Serialize
x and y arrays using six significant decimal digits; stored datasets and evaluator
inputs retain float64 precision. Record the full rendered prompt for every model
call. No examples, family labels, target formula, held-out points, or feedback are
included. The exact instruction is stored in the source and trace.

Use CPU float32, three PyTorch threads, deterministic algorithms, sampling with
temperature 1, top-k disabled, top-p 1, one beam, repetition penalty 1, and KV caching.
GenerationConfig is constructed explicitly; every effective setting is recorded.
The token budget is the longest permitted catalog sequence including EOS. The
original tokenizer/head remain intact. Completed rows may absorb EOS padding while
longer batch mates finish, but no noncatalog output is accepted.

For task index i and base seed s, the effective sampling seed is s + 1009*i in all
three sampling conditions. NumPy and PyTorch use different RNG implementations;
this matches declared seeds, not individual random draws. No-observation draws are
fresh per task rather than reusing one sequence set for the entire suite.

## Reproduction

Requires optional HF dependencies. A CPU-only installation can use:

```bash
uv sync --dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install 'transformers>=4.45,<5'
uv run --no-sync pytest -q
uv run --no-sync python scripts/prepare_baseline_model.py --output runs/models/smollm2-135m
uv run --no-sync python -m symbolic_wsp.experiments.pretrained_baseline --model runs/models/smollm2-135m --output runs/baseline-smoke --smoke
uv run --no-sync python -m symbolic_wsp.experiments.pretrained_baseline --model runs/models/smollm2-135m --output runs/baseline-full
```

Use new output directories for each run. `uv sync --extra hf --dev` is an alternative
when the default PyTorch distribution suits the machine. The model download is
explicit and pinned; default unit tests never download weights. Local model files
are SHA256-checked before inference. Weights are not included in the Git repository.
The model's downloaded README retains its upstream license information.

## Artifacts and accounting

The output contains the exact configuration, effective decoding configuration,
tasks, per-candidate traces and scores (`rows.jsonl`), coverage diagnostics, summaries,
model-file hashes, and environment metadata. Rows are written incrementally so an
interrupted run retains evidence. A fresh output is required; automatic resume is
not implemented. Smoke mode uses tasks 0 and 5 with two candidates and the first seed.

Timing excludes model loading and file hashing from the trial loop. Loading time
is recorded separately for the full study; peak process RSS includes loading and
inference on Linux. Generation time, prompt-token count, padded prefill token slots,
actual output tokens including EOS, and generation calls are recorded per model row.
Generated tokens exclude trailing EOS padding. For uniform and exhaustive conditions,
timing includes candidate selection and numerical scoring, not neural inference.
Coverage enumeration is reused for exhaustive results, and its measured evaluation
time is charged to that condition. Evaluator counts include candidate observation
scores plus two evaluations of the selected candidate on reserved regimes.

No GPU FLOPs, energy, or monetary costs are measured. Equal K does not equal equal
compute. Exhaustive enumeration has a different candidate budget. All conditions
use the same observation selector, but output diversity affects their effective
number of unique evaluated expressions.

## Interpretation limits

This suite has only five functions per group and three draws per task. Paired
bootstrap intervals resample functions after averaging their repeats and are
strictly descriptive. They do not correct for multiple comparisons, establish
performance across independent task suites, or assess editing-session variance.
The smoke run is a disclosed feasibility check, not a tuning set; do not choose
sampling settings or targets in response to its performance.

Exact target-string recovery is meaningful only for the representable group and
is not a test of symbolic equivalence. Catalog constraints guarantee syntax, not
numerical validity in general. Report invalid selections and do not silently
replace failed outputs. NMSE can be large on low-variance targets, so include MSE
and per-function results. Inventory changes, new prompts, or larger models would
constitute follow-up experiments with fresh declared protocols.

The decision is whether to investigate prompting/model suitability, expand the
inventory, or proceed to edits. No result from this study demonstrates WSP success,
new structural synthesis, or an independently tested general search capability.
