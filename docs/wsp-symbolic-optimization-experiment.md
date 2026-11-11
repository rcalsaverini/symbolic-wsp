# Experiment: WSP as a mechanism for symbolic optimization

## Research question

Given a collection of numerical observations, can we directly modify a language
model's weights so that it proposes better mathematical expressions? If this works
on some examples, does the change transfer to functions that did not guide the
choice of intervention?

The immediate goal is to design an experiment that can answer these questions,
including when the answer is negative. We do not assume that an agent capable of
inspecting weights can create a symbolic optimization capability.

This document describes the intended experiment. The suggested pilot numbers are
protocol decisions to record before execution, not observed results or settings
already enforced by the code.

## Background

The general problem is to find a valid expression that minimizes an objective:

\[
e^\star=\arg\min_{e\in\mathrm{Exp}_G}f(e).
\]

The grammar \(G\) defines the expression space. The function \(f\) may measure
prediction error, complexity, or a program's performance against a specification.

The 2021 proposal was to learn an encoder, a decoder, and an objective surrogate.
Search would take place in a continuous representation, followed by decoding into
expressions. Building a reliable tree decoder was a major practical difficulty.

The later symbolic regression proposal used an LLM as a generator, an adapter for
continuous observations, and PPO to adjust its behavior. This reduces the need to
build a dedicated decoder but requires operating a training pipeline.

The current experiment investigates another possibility: an agent formulates and
tests direct interventions in a pretrained generator's weights. Instead of updating
parameters through gradients, we apply an explicit modification:

\[
\theta' = \theta + \Delta\theta_{\mathrm{agent}}.
\]

The agent can inspect tensors, observe activations, and measure behavior. It does
not perform backpropagation, PPO, or generator fine-tuning. Observations enter as
text; there is no PointAdapter or modification of the tokenizer or vocabulary head.

An improvement in this experiment would not validate the original latent-space
search hypothesis. It would provide evidence about a different mechanism addressing
the same broader problem. The original texts are preserved in [research/](../research/).

## What improvement would mean

Let \(p_\theta(e\mid\mathcal D,G)\) denote the model's distribution over expressions
under grammar constraints, given observations \(\mathcal D\). We are not merely
trying to increase the probability of a particular answer. We want to test whether
the modified distribution produces lower-error candidates on new tasks with the
same generation budget.

For each task, we generate \(K\) candidates and select using the available observations:

\[
\hat e_\theta = \arg\min_{e\in\{e_1,\ldots,e_K\}}
\left[\operatorname{NMSE}_{\mathrm{obs}}(e)+\lambda C(e)\right].
\]

We then measure the error of \(\hat e_\theta\) on numerical points not used in that
selection. There are two distinct meanings of generalization:

1. **New points from the same function:** numerical interpolation and extrapolation.
2. **New functions:** transfer of the intervention to tasks that did not guide the edit.

Predicting new points from a known function does not demonstrate transfer between
functions. Similarly, replacing a global preference for `x` with `sin(x)` may help
periodic tasks while hurting all others. The controls should expose this effect.

## Hypotheses and alternative explanations

| Hypothesis | Required evidence | Alternative explanation to rule out |
| --- | --- | --- |
| The intervention changes the proposal distribution | Repeated changes in candidates, logits, or frequencies | Sampling variation |
| The change improves search in the studied domain | Lower error at fixed K on functions not used in development | More generation, prompt changes, or memorization |
| The mechanistic choice of edit adds value | An advantage over comparable random controls | Generic weight perturbation |
| The improvement is reusable | Transfer between functions and families without substantial regressions | Preference for a particular family |

The operational null hypothesis is that the edit does not reduce expected error on
held-out tasks under comparable conditions. A negative result may indicate either
an ineffective intervention or an insufficiently sensitive pilot. It does not
prove that WSP is impossible for every model or domain.

## The key limitation of the first version

The current backend generates within a finite expression inventory. Each expression
is validated and compiled into sequences of the original tokenizer. The mask permits
only continuations that terminate in an expression from that inventory. This handles
functions and operators spanning multiple tokens, but does not provide open-ended
structural synthesis.

With the default inventory of 14 expressions, the executable question is narrower:
**does the edit improve candidate selection within a fixed catalog?**

Before attributing a result to the generator, we need to measure catalog coverage.
For each development or validation task, an evaluator can enumerate the inventory
and compute its best achievable error. This oracle diagnoses the search space; it
is not a neural condition and is not supplied as model feedback. If no candidate
explains a function, an edit cannot solve the task within that space.

The report should distinguish representable tasks, approximable tasks, and tasks
outside the inventory using numerical criteria fixed in advance. Constructing a
catalog after inspecting final answers would invalidate the experimental separation.
Inventory coverage, a uniform baseline, and per-family analyses still require
dedicated scripts.

A second stage could replace the catalog with a compositional grammar bounded by
size and depth. Only then would it be appropriate to investigate search for structures
not enumerated in advance. This extension is not implemented.

## Experimental conditions

| Condition | Weights | Generation and evaluation |
| --- | --- | --- |
| A: baseline | Original checkpoint | K proposals under the fixed grammar |
| B: inference-time feedback | Same original checkpoint | K sequential proposals; previous observation losses enter the context |
| C: WSP | Checkpoint with an intervention selected through development/validation | Same regime as A |
| D: random edit | Baseline perturbation with a norm comparable to C | Same regime as A; repeat with independent seeds |
| E: no-edit | Same state as A, reloaded | Repetition to check the execution path |

C versus A measures the effect of the edit. C versus D tests whether the chosen
direction is better than perturbing weights. B versus A measures the value of
feedback without editing. E versus A should reproduce the same results in the
deterministic backend; differences on real hardware should be investigated and
documented.

Matching K controls candidate count but does not equalize computational cost:
B uses progressively longer contexts. Recording tokens, latency, and the cost of
finding an edit is necessary for an economic comparison. The current runner counts
generations and total time; it does not automatically measure agent tokens, FLOPs,
or monetary cost.

The current random control matches the Frobenius norm in the modified tensor. It
does not match rank, sparsity, or geometry. A stronger mechanistic conclusion would
require controls with the same support and rank as the edit, across multiple seeds.
These additional controls are experimental proposals, not existing CLI features.

## Data organization and access

Separate development, validation, and final test functions, as well as observed and
held-out points within each function. The current generator uses distinct structural
templates across splits and random streams derived from the split name. This reduces
trivial duplicates but does not guarantee algebraic inequivalence or sufficient
diversity for a broad benchmark.

The agent uses development tasks to diagnose behavior and formulate interventions.
Validation selects or rejects candidates within a declared budget. Repeated validation
queries are also a form of selection: record how many proposals were tested and reserve
an independent final test. Do not interpret the winning edit's validation interval
as though no search had taken place.

The final test must be held under a separate OS account or in a container/machine
whose data are not mounted into the agent's tools. A CLI flag, a mode-0700 file under
the same user, or a prompt instruction does not provide that isolation. The final
evaluation script illustrates the workflow; provisioning the separation is external.

The current runner combines interpolation and extrapolation in the `heldout` field.
Reporting them separately requires extending the format or analyzing points under
rules declared before execution. The agent must not use final errors to select
another edit.

## Proposed pilot

Start with a small, open causal model, such as SmolLM2-135M-Instruct, provided the
baseline offers a usable signal. Pin the local model and tokenizer revision,
dependency versions, device, inventory, and prompt serialization.

A reasonable initial feasibility configuration would be 30 development functions,
30 validation functions, K = 16, and five generation seeds. These values do not
replace a later sample-size analysis. Distribute tasks across the five current
families and check coverage before editing. If the catalog or model produces a
floor effect, correct the design before interpreting negative results.

Limit the initial search to ten edit proposals, recording the agent's time and
token budget. Start each proposal from the baseline to avoid confusing isolated
edits with a cumulative sequence. If later experiments compose edits, record the
entire lineage and test ablations of individual components.

After freezing an intervention, evaluate it once on private final functions.
Determine the final function count and minimum meaningful effect from pilot
variability and available resources before opening the test.

## A concrete intervention cycle

Suppose the baseline almost never proposes trigonometric expressions, even when
the data are periodic and those expressions are present in the catalog.

1. Measure this failure in development and check whether it depends on inventory
   coverage, tokenization, prompt, or budget.
2. Compare periodic and nonperiodic contexts; inspect relevant logits and
   representations. Current probes expose last-token hidden states and logits.
   `probe` uses only observation JSON, whereas `generate` includes an instruction;
   aligning these contexts is pending work before attributing a trajectory mechanism.
3. Formulate a testable hypothesis: for example, an MLP output direction excessively
   suppresses probability mass on trigonometric continuations in this context.
   This is an investigative hypothesis, not a circuit identified by the toolkit.
4. Specify the tensor, axis orientation, direction, and magnitude. Apply a small
   projection or rank-one update without modifying the baseline.
5. Evaluate the candidate with the same inventory, prompt, data, K, and seeds.
   Include nonperiodic functions and the corresponding random control.
6. Record the decision, including cases where sine frequency increases but error
   worsens. Return to the parent checkpoint when the intervention is rejected.

This project does not contain an algorithm that automatically discovers the correct
direction. Claude Code or Codex supplies the investigation and proposal; the engine
supplies executable operations and auditable records.

## Metrics and decision rule

The proposed primary metric is the paired difference in held-out NMSE of the
expression selected using observations. Negative values favor the edit. The
implementation normalizes by `max(var(y), 1e-12)` and measures complexity by counting
mathematical AST nodes. Nearly constant functions can produce very large NMSE;
report MSE as a complementary metric.

Also retain numerical validity, the candidate loss distribution, complexity,
structural diversity, per-family results, edit magnitude, and search cost. Syntactic
validity guaranteed by the catalog does not guarantee a valid numerical domain.
AST diversity does not establish functional diversity.

The current comparator aggregates repetitions within each function and computes a
function-level paired bootstrap, avoiding treating seeds as independent tasks.
The screening rule requires at least five paired functions, no invalid pairs, and
an upper 95% bootstrap confidence bound below zero. This minimum is an operational
guard, not a sample-size justification. The `paired` CLI computes the rule; formal
decision recording uses the `record_decision` API, already used in the demonstration.

For the scientific experiment, also predeclare a minimum meaningful improvement,
per-family regression limits, and a limit on complexity increases. These additional
criteria are not enforced by the current comparator. On the final test, report the
effect, uncertainty, and failures without selecting new interventions from the
result. Do not claim acquisition of a general policy merely because a validation
interval falls below zero.

## Getting started with the toolkit

First, run the infrastructure demonstration in a new directory:

```bash
uv sync --dev
uv run pytest -q
uv run symbolic-wsp demo --output runs/wsp-symbolic-demo
```

It uses a synthetic generator and requires no pretrained weights. Its results check
execution paths; they do not test the LLM hypothesis.

For a real model, install the HF extra, prepare a locally pinned copy, and use the
same inventory files across conditions. Here, `BASE_ID` and `CHILD_ID` denote IDs
returned by earlier commands; `inventory.json` is a JSON list of validatable
expressions prepared before evaluation:

```bash
uv sync --extra hf --dev
uv run symbolic-wsp dataset --split dev --seed 17 --count 30 --output runs/dev.json
uv run symbolic-wsp dataset --split validation --seed 17 --count 30 --output runs/validation.json
uv run symbolic-wsp hf-snapshot --model /path/to/pinned-model
uv run symbolic-wsp run --backend hf --model /path/to/pinned-model --checkpoint BASE_ID --expressions inventory.json --dataset runs/validation.json -k 16 --seeds 11 22 33 44 55 --output runs/baseline.json
uv run symbolic-wsp edit BASE_ID --spec intervention.json --hypothesis 'Mechanistic hypothesis and observable prediction'
uv run symbolic-wsp run --backend hf --model /path/to/pinned-model --checkpoint CHILD_ID --expressions inventory.json --dataset runs/validation.json -k 16 --seeds 11 22 33 44 55 --condition wsp --output runs/edited.json
uv run symbolic-wsp paired runs/baseline.json runs/edited.json
uv run symbolic-wsp rollback CHILD_ID
```

`intervention.json` must use dimensions and names from the real checkpoint. The
rank-one example in `examples/` targets the demonstration's synthetic head and must
not be applied directly to an HF model. Diagnose behavior on development tasks before
editing, and run feedback, no-edit, and random controls as well. These commands
illustrate a workflow; they do not imply that a pretrained-model experiment has
already been executed.

## Required report contents

- Checkpoint, tokenizer, inventory, prompt, and execution environment identities.
- Function families and counts, catalog coverage, splits, and public seeds.
- All proposals, hypotheses, tensors, norms, costs, decisions, and lineages.
- Paired comparisons with baseline, feedback, and random controls; per-family regressions.
- Numerical and complexity metrics, intervals, invalid pairs, and coverage limitations.
- Independent final results and the distinction between output preference, an improved
  proposal distribution, and evidence of reusable search.

A rejected edit with an explicit hypothesis and reproducible evaluation is also a
useful result. The first experiment aims to determine whether there is a signal
that justifies expanding the grammar and the experimental program.

## Proposal versus implementation

| Component | Current status |
| --- | --- |
| Identified checkpoints, editing, comparison, and rollback | Implemented |
| Safe expression evaluation and complexity penalty | Implemented |
| Catalog-constrained HF generation; textual observations | Implemented; integration tested with a tiny local model |
| Conditions A–E, repetitions, and paired comparison | Implemented |
| Improvement in a pretrained LLM's weights | Not yet investigated experimentally |
| Coverage audit, catalog oracle, and uniform baseline | To implement for the scientific pilot |
| Probes aligned with the prompt and complete trajectories | To extend |
| Random controls matching rank/support | To implement |
| Separate interpolation/extrapolation and per-family reports | To extend |
| Truly isolated final testing and complete costs | Requires external infrastructure and instrumentation |
| Compositional grammar and open-ended structural search | Later stage |

For operational details, see [architecture](architecture.md),
[methodology](methodology.md), [executed verification](verification.md), and
[agent setup](agent-setup.md).
