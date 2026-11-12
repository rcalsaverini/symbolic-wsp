# First-class baseline diagnostics

This follow-up makes the proposed baseline tests a reproducible CLI experiment.
It does not overwrite the initial feasibility study. The frozen configuration is
[`config.json`](config.json); model preparation uses the two pinned model configs.

## Tests and phases

| Test | Implementation | What it measures |
| --- | --- | --- |
| Exact catalog distribution | `CatalogOracle` | Every leaf probability under the actual autoregressive grammar mask |
| Observation controls | correct, shuffled y, another function's y, absent | Sensitivity to the numerical relationship and to superficial prompt content |
| Prompt elicitation | minimal, explicit catalog, two fixed demonstrations | Whether task framing reveals useful baseline behavior |
| Model scale | pinned SmolLM2 135M and 360M | Whether findings persist in one larger model from the same family |
| Targeted intervention | gated development-only rank-one proposal, random and logit controls | Whether an explicit weight intervention adds value after baseline diagnosis |

Run phases separately or use `--phase all`, which executes development,
intervention, then evaluation. Exact enumeration eliminates target sampling noise;
no extra generation seeds are needed to estimate the catalog distribution.

```bash
uv sync --extra hf --dev
uv run python scripts/prepare_baseline_model.py --config experiments/baseline-diagnostics-01/model-135m.json --output runs/models/135m
uv run python scripts/prepare_baseline_model.py --config experiments/baseline-diagnostics-01/model-360m.json --output runs/models/360m
uv run symbolic-wsp diagnostics --small-model runs/models/135m --large-model runs/models/360m --output runs/diagnostics --phase all
```

For an already prepared CPU-only environment, use `uv run --no-sync` to retain its
CPU PyTorch installation. Individual phases use `--phase development`,
`--phase intervention --resume` and `--phase evaluation --resume`. The same config
is mandatory on resume; completed matrix rows are reused. Use a new output for a
new protocol. Model downloads are explicit and excluded from ordinary unit tests.

## Data separation and prompt selection

Five development expressions, two demonstration expressions and five representable
evaluation expressions are disjoint. Development, demonstration and evaluation
points have separate configured seeds. The evaluation suite additionally includes
five outside-catalog functions, always reported separately. All functions and data
here are public development/validation material, not an independent final test.
The evaluation functions appeared in the earlier baseline; new points do not make
them novel structures or erase that research history.

Demonstrations are fixed before execution: eight pairs each for `x+1` and `sin(x)+x`.
No evaluation label or reserved point is supplied in a prompt. The same examples
and numerical precision are used for both models. All prompt text is explicit in
`prompts.json`. A model's prompt variant is selected using minimum development
expected observation-objective regret, with a lexical tie-break. It is frozen in
`selections.json` before evaluation. All three variants are still evaluated to
show the full preregistered comparison, not to choose an evaluation winner.

Observation controls are applied only to the proposal generator. Selection of the
best of K still uses the original, correct observed data. Shuffling permutes y
without changing x or the y multiset. The other-function condition evaluates the
next task's function at the original x positions. Absence also changes prompt
length; it alone cannot establish sensitivity to the numerical relationship.

## Exact probabilities and best-of-K

One full prompt prefill is shared by a batched, teacher-forced pass over every
catalog continuation. At every token prefix, probabilities are renormalized over
allowed next tokens, including EOS. This is not a softmax over whole-expression
likelihoods and not a length-normalized ranking. The probabilities over all leaves
must sum to one; the small roundoff mass is recorded before final normalization.
Raw per-prefix allowed-token log probabilities are retained for audit.

A unit test checks this calculation against separately recomputed, uncached full
forwards on an actual tiny Transformers model. Another verifies prefix masking
against a hand-computed tree with large forbidden-token logits.

Expressions are ranked by observation NMSE + complexity penalty, with lexical
tie-breaking. If an expression has proposal probability p_i and better expressions
have total probability b_i, its probability of being selected from K independent
proposals with replacement is `(1-b_i)^K - (1-b_i-p_i)^K`. The report uses these
probabilities to compute expected errors on reserved points. It does not choose
expressions using reserved error or equate best observation fit with best
extrapolation. A brute-force draw-enumeration test checks the formula.

Identical rendered prompts may reuse an exact distribution while weights are
unchanged. Cache keys include root-logit bias; applying or restoring an edit clears
the cache. Cached rows record `cache_hit=true` and zero additional inference seconds,
not zero original construction cost. No-edit intervention controls recompute the
baseline after an explicit restore.

## Predeclared intervention

Only the 135M model is edited. The gate requires at least two development targets
to have exact-expression probability below the configured threshold under every
prompt variant. Failure to meet the gate produces an explicit `not_attempted`
record rather than an arbitrary edit. This is a scientific gate, not a test failure.

When the gate passes, use the selected development prompt and the last module
matching the configured suffix (`mlp.down_proj` for these models). The read vector
is the normalized mean of that module's input at the assistant-start position on
development contexts. The write vector is the normalized difference between mean
output embeddings for the failing targets' first tokens and the baseline's expected
first-token embedding. This is a deliberately simple access/steering hypothesis,
not a discovered symbolic algorithm.

Construct two gradient-free rank-one updates with Frobenius norms 0.1% and 1% of
the original tensor norm. Select on development expected observation regret only;
require the configured minimum improvement to pass that screening rule. Freeze
`decision.json` before constructing evaluation tasks. A rejected candidate may be
evaluated as a labeled diagnostic, never retained because of evaluation results.

Evaluate the chosen candidate against baseline, restored no-edit, two random edits
with matched delta norm, and a first-token logit-bias control using the same target
token set. The latter tests simple runtime output steering and is not a complete
equivalent program. Random edits match tensor and norm, not rank. There is no claim
that these controls exhaust alternative explanations.

Use the existing immutable WSP tensor store and edit operations. These checkpoints
contain the affected tensor, not a full language model. The pinned baseline-model
manifest plus the named tensor override identifies the candidate. All edit vectors,
parameters, delta norms, hypotheses and decisions are recorded. The original tensor
is restored exactly, hash-checked, and logged through rollback. No generator or
probe is trained; activation averaging and embedding differences are explicit
algebraic operations.

## Scope and interpretation

The catalog remains finite; structural synthesis is not tested. The selected
expressions have no fitted constants. Report expected interpolation/extrapolation
MSE and NMSE separately, including per-function values: low variance can dominate
NMSE. One model pair does not establish a scaling law. The intervention is one
predeclared proposal family, not a benchmark of autonomous editing agents.

Uniform and exhaustive catalog methods remain comparison references. The direct
code baseline here is exhaustive numerical catalog selection; it is not an agent
asked to invent a new program. Broader code-versus-weight studies and isolated
final testing remain future work. No successful result alone establishes creation
of a new capability or excludes access repair/program insertion.
