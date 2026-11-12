# First-class baseline diagnostics

This follow-up makes the proposed baseline tests a reproducible CLI experiment.
It does not overwrite the initial feasibility study. The frozen configuration is
[`config.json`](config.json); model preparation uses the two pinned model configs.

## Executed findings

See [the generated report](RESULTS.md) and [raw records](results/). The complete run
contains **270 matrix records** (30 development and 240 evaluation distributions)
and **75 intervention records**. This includes both pretrained models, all three
prompts and all four evaluation controls. Exact probability checks, all expected
best-of-eight metrics, rendered prompts, frozen selections, no-edit equality,
rollback and phase source provenance passed the independent artifact audit.

Both models selected the explicit-catalog prompt on development. It raised some
useful expression probabilities substantially: for example, the 135M probability
of `sin(x)` rose from about 0.000477 under the minimal prompt to 0.0745 with the
catalog. The earlier absence of sine in sampled outputs did not mean that the
constrained decoder prohibited it.

However, correct observations did not consistently improve results over controls.
On five representable evaluation functions, selected-prompt expected interpolation
NMSE was 1.161 with correct data versus 1.153 with shuffled data for 135M; for 360M,
4.415 versus 4.411. Average distribution total variation after shuffling was about
0.029 and 0.037, respectively. These results do not establish reliable use of the
x/y pairing under this input representation.

The larger model does not give a uniform improvement: its selected-prompt
interpolation NMSE is worse than 135M, but raw interpolation MSE is better (0.1249
versus 0.1632). Its expected exact-target hit rate at K=8 is about 44.35%, near the
uniform-sampling reference of 44.73%; that alone does not make its error distribution
better. The full report includes both metrics, per-function target probabilities,
controls and outside-catalog results. Five functions per group do not support a
broad model-ranking or scaling claim.

The repeated-failure gate passed on development targets `x`, `2*x` and `sin(x*x)`.
Both rank-one magnitudes worsened development regret; the less harmful candidate
(0.1% of the chosen tensor's Frobenius norm) was explicitly rejected before evaluation.
It was evaluated only as a diagnostic. On representable evaluation functions it
raised mean target probability from 0.0901 to 0.1197, while worsening expected
interpolation NMSE from 1.161 to 13.44. Random controls stayed near baseline;
restored no-edit results matched baseline exactly. The original tensor was restored
and its hash checked. No edit was retained or deployed.

This is evidence for distinguishing output-distribution steering from useful search
improvement, not evidence of newly acquired symbolic capability. The immediate
bottleneck remains elicitation/numerical conditioning and the finite catalog; this
particular edit did not repair it. Stronger claims require broader tasks and
independent evaluation, not further selection on these results.

Recorded matrix inference time sums to about 22.2 minutes; intervention inference
is recorded separately. Timings are shared-CPU observations. Model weights and
materialized tensor-checkpoint binaries are omitted from Git; pinned download
identities, all edit vectors/seeds/parameters, manifests, history, and rollback
hashes are included. Rerunning the command materializes the tensor states locally.

Verification: **39 automated tests pass**, including uncached probability-reference
checks for GPT-2 and Llama, exact best-of-K checks, observation-control invariants,
a real tiny-Llama edit/rollback integration, and the complete offline phase/resume
workflow. Ordinary tests never download pretrained weights.

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
is mandatory on resume; completed matrix rows are reused. Source changes within
a phase are rejected rather than mixing computations from different revisions. An incomplete
intervention aborts rather than appending duplicate trials; start a fresh study
output to repeat that phase. A completed intervention is reused. Use a new output for a
new protocol. Use one writer per output directory; do not run phases concurrently. Model downloads are explicit and excluded from ordinary unit tests.

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


## Reporting and verification

After all phases complete, the report and audit commands are:

```bash
uv run --no-sync python scripts/report_diagnostics.py runs/diagnostics --output runs/diagnostics/RESULTS.md
uv run --no-sync python scripts/verify_diagnostics.py runs/diagnostics --small-tokenizer runs/models/135m --large-tokenizer runs/models/360m
```

The audit reconstructs every leaf probability from recorded per-prefix masked log
probabilities, recomputes expected selected-candidate metrics, rerenders prompts,
checks development selections, and checks no-edit/rollback and source provenance.
The first-token bias control operates on the measured root distribution; it is not
a general deployment hook or an exported weight change.


## Source provenance

The recorded study identifies the actual execution revision for each phase and
archives the corresponding source by hash. A small reporting edge-case fix for
empty subgroups was made between intervention and evaluation; it does not affect
the configured groups or model probabilities, and the phase sources are separately
preserved. No historical performance record was rewritten.

Future runs automatically save per-phase environments, source snapshots and
provenance. Their metadata now includes the Git revision and worktree state; source
snapshots remain authoritative when a checkout is dirty or Git is unavailable.
The source-change guard protects interrupted phase resumes. These provenance
improvements were verified with the offline workflow test; they do not constitute
a rerun of the pretrained-model measurements.
