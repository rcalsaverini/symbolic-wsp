# Pretrained baseline feasibility test

This is a development study of a frozen pretrained model, not a WSP intervention
or an independent final test. The ten functions and four comparisons follow the
baseline proposal discussed before execution. The committed `config.json` fixes the
model revision, inventory, generation seeds, dataset seed, K, and decoding settings.
The active schema-2 config also makes the full prompt explicit; the completed
schema-1 run is preserved in `results/config.json`. See [exact prompts and provenance](PROMPTS.md)
for all rendered inputs and the verified equivalence of the configured replay.

## Executed outcome

The full pinned-model study completed on CPU in **371.2 seconds** of trial-loop
wall time, with a peak process RSS of **1316.2 MiB**. All 480 model-generated
candidates were valid. See [the generated results report](RESULTS.md) and
[raw artifacts](results/). The complete suite had 860 candidate evaluations across
four methods, plus reserved-point evaluations of the selected candidates.

On these ten functions, conditioning on observations did not improve mean
interpolation NMSE over the no-observations generator (8.423 versus 8.209).
Conditioning changed proposal frequencies, but neither model condition proposed
any trigonometric or exponential catalog entries in 240 draws each. The conditioned
model repeated candidates heavily (54.6% duplicate rate) and selected the literal
target in only 20% of the 15 representable-task repeats. Uniform sampling recovered
40%; exhaustive observation selection recovered all five representable targets.

Mean interpolation NMSE favored conditioning over uniform sampling slightly, but
the descriptive paired interval crossed zero. Extrapolation conclusions also varied
by function and metric: mean extrapolation NMSE was worse than the no-observations
control while mean extrapolation MSE was better. Low target variance, especially for
sine on [1, 2], magnifies normalized errors. Do not select a favorable aggregate
while omitting this disagreement or the per-function table.

Coverage is a substantial limitation outside the catalog. For `1/(x*x+2)`, even the
lowest observation NMSE in the inventory was about 40.49. No weight edit can add its
missing expression while decoding is restricted to this fixed inventory. Exhaustive
selection also illustrates that minimizing the observation objective need not
minimize extrapolation error: it was worse than the conditioned model on some
outside-catalog functions. No held-out result was used to change the selector.

For a catalog of 14 cheap expressions, exhaustive selection took about 0.017 s,
compared with 306.1 s for conditioned generation. These timings have different
candidate budgets and exclude model loading. This run establishes no practical
advantage for neural search in this catalog, and supplies no evidence about WSP.

**Recommended next step:** investigate prompt/model suitability and catalog effects
before editing weights. A separately declared follow-up could compare richer
numeric conditioning and a stronger model, include observation-shuffling controls,
and expand constants/structures with an explicit coverage audit. A no-observations
ablation also changes prompt length and tokens, so differences alone would not
establish numerical understanding. Keep this run as the unchanged reference;
follow-ups must not be presented as additional draws from its frozen protocol.

## Smoke test and verification history

The first smoke attempt failed after one recorded row because the grammar mask
returned no allowed continuation for an already finished batch member. Transformers
still invokes its mask while longer batch mates run. The fix allows valid completed
rows to absorb EOS padding; it does not change sampling settings, inventory, or
tasks. A regression test covers completed, padded, and invalid prefixes, and an
offline Transformers test covers mixed-length batched completions.

The corrected smoke run completed both tasks with two candidates each in 6.61 s.
The [failed attempt](smoke/failed/) and [successful smoke artifacts](smoke/passed/)
are preserved. Failed-call timing is unavailable; do not count it as zero. Smoke
mode reindexes its two selected tasks for effective-seed derivation, so its second
seed differs from the corresponding full-run task. It is not a numerical replay
of the full study.

Verification: **29 tests passed** with optional HF dependencies installed; Ruff lint
and format checks passed. All 100 full-run records, all 860 candidate scores,
selected candidates, reserved scores, summaries, descriptive comparisons, configuration
and dataset digests, and recorded source hashes were independently recomputed and
checked against the saved artifacts. No real-model test or model download is added
to the default automated suite.

Regenerate the Markdown tables from the committed records with:

```bash
uv run --no-sync python scripts/report_pretrained_baseline.py experiments/pretrained-baseline-01/results --output experiments/pretrained-baseline-01/RESULTS.md
uv run --no-sync python scripts/verify_pretrained_baseline.py experiments/pretrained-baseline-01/results
```

The exact-score verification command assumes the recorded NumPy version and numerical environment.

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
included. The [prompt document](PROMPTS.md) shows the exact instruction, injected
system message, both rendered conditions, and all ten task-specific prompt files.
The active config pins all prompt text, numerical precision, and chat-template
formatting; new runs reject missing prompt specifications and record their digest.

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
