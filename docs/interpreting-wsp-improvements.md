# Interpreting WSP improvements: capabilities, execution, and compiled programs

This document supplements the [experiment proposal](wsp-symbolic-optimization-experiment.md)
and [operational methodology](methodology.md). It addresses what a positive result
would mean, which alternative explanations remain, and which additional experiments
would distinguish them. These are proposed controls, not newly implemented features
or results from a pretrained model.

## Three explanations that can overlap

An improvement in symbolic regression is compatible with at least three accounts:

1. **Capability acquisition:** the edit enables a useful computation that we could
   not elicit from the baseline under the specified controls.
2. **Execution or access repair:** the model already has a useful strategy, but the
   edit improves its selection, routing, maintenance, or completion, including by
   removing loops.
3. **Program insertion:** the editing agent designs a procedure and implements it
   through weight changes. The target executes that procedure, potentially using
   existing representations or generation capabilities.

These are not mutually exclusive experimental labels. An inserted procedure can
repair access to an existing strategy and create a new externally observable
capability at the same time. The report should describe the evidence for each
mechanism instead of assigning every edit to exactly one category.

A finite set of failed baseline trials cannot establish that a computation was
absent from the original weights. Use the operational claim "not elicited under
these controls" rather than claiming creation from nothing. State whether the
claim concerns the target alone or the combined system of editing agent, tools,
development examples, and target model. Knowledge supplied by a stronger agent
can account for a target improvement without demonstrating autonomous discovery by
the target.

"Static program" also needs a useful definition. Every frozen network implements a
fixed computation; that broad definition distinguishes nothing. Here, program
insertion means an identifiable procedure deliberately compiled into the weights:
a lookup, a conditional rule, an operator-selection heuristic, or a search routine.
Such a procedure can be input-dependent, iterative, and generalizable. Generalization
therefore does not by itself rule out program insertion.

## What the two external investigations contribute

Lorenzo Ciarpaglini reports reconstructing rule-based programs from small networks
written by LLMs, and finding stronger policies when the same model wrote code
directly. This motivates separating algorithm design from its representation as
weights. His experiments concern networks written from scratch, not patches to a
pretrained symbolic generator; they do not establish the mechanism of our edits.
See [The program behind the weights](https://lorenzociarpa.org/blog/program-behind-the-weights).

Kevin Bondzio reports patches that detect an existing activation signal and steer
an initial response, alongside comparisons of collateral effects and later training.
This motivates testing routing and preservation before inferring new competence.
Those task-specific reports do not imply that symbolic strategies are already
encoded in our baseline or that every weight edit will behave similarly.
See [Weight-space programming, tested](https://kevinpl07.github.io/weight-patching/).

We have read these accounts, not independently reproduced their experiments. The
controls below are our proposed adaptations. Neither post's strongest conclusion
should be treated as a universal result about weight editing.

## Separate behavioral effect, mechanism, and representation value

There are three questions, each requiring its own evidence:

| Question | Required comparison | What remains unresolved |
| --- | --- | --- |
| Does the edit improve behavior? | Paired original versus edited checkpoint, fixed protocol, independent functions | Why the improvement occurs |
| What computation explains it? | Assistance controls, causal ablations, counterfactual tasks, candidate program reconstruction | Whether weights are the best place to execute it |
| Does storing it in weights offer an advantage? | Ordinary code and runtime steering using comparable information and budgets | Broader or autonomous capability acquisition |

A positive score difference answers only the first question. A successful compiled
heuristic can be a valuable engineering result, especially under deployment
constraints. It is a different result from evidence of a reusable symbolic-search
mechanism that the simpler alternatives fail to explain.

## Elicit baseline strategies before claiming acquisition

Evaluate the baseline and edited model under a predefined assistance ladder:

| Assistance | What it investigates |
| --- | --- |
| Original prompt and fixed generation budget | Total effect under the primary protocol |
| More samples or a larger token budget | Rare baseline successes and budget sensitivity |
| An explicit strategy instruction | Ability to execute a supplied strategy |
| Decomposition with externally maintained intermediate state | Difficulty maintaining or sequencing a procedure |
| Repetition handling and controlled restarts | Execution failures that can be repaired externally |
| A controlled first token or expression prefix | Whether initiating a behavior is sufficient |

Apply comparable assistance to both models where meaningful, and report additional
compute rather than mixing these runs into the primary matched-budget estimate.
Forcing a prefix changes the task distribution; it is a diagnostic intervention,
not an unassisted baseline. Strategy hints must not include private answers.

If simple loop recovery closes the gap, execution repair becomes a plausible
explanation. If the baseline needs a complete strategy description, it may possess
execution ability without selecting or inventing that strategy. If an advantage
persists, this is evidence against the tested access explanations, not proof that
all latent-capability explanations have been eliminated.

Separate self-contained expression generation from a multi-step search process.
The current target receives observations and produces expressions; it does not
maintain an explicit symbolic search state. Claims about maintaining a strategy
require a task and interface in which that strategy can actually be observed.

## Measure execution failures without selecting away the problem

Record the probability of completing an attempt, reasons for termination, repeated
tokens, repeated candidates, total generated tokens, and cost until a satisfactory
expression is found. Record expression quality as well as completion rate.

Completion rate times success conditional on completion gives total success when
an incomplete attempt counts as failure. This decomposition helps describe where
a gain appears, but comparing only completed attempts can select different tasks
or trajectories in the two models. It does not causally isolate execution repair.
The stronger test is to intervene on baseline execution and measure how much of
the original gap remains.

Token loops and repeated complete expressions are different failures. The finite
catalog mask permits only bounded paths ending at valid leaves, so it already
removes many forms of nontermination. Our current setup has limited sensitivity
to loop repair. A separate diagnostic condition with a sufficiently expressive
bounded grammar, or carefully controlled unconstrained generation, would be needed.
Evaluate syntax and numerical failures explicitly in that condition; do not combine
its scores with the catalog experiment as though the protocols were identical.

## Add an ordinary-program baseline from the same agent

Ask the agent to express its proposed improvement as ordinary code as well as a
weight edit. Code may rank catalog entries, select operator families, or allocate
a search budget. It should receive only the task observations and permitted feedback
at evaluation time, not ground-truth expressions, family labels unavailable to the
model, or reserved numerical points.

Use independent sessions or counterbalance order to avoid always allowing one arm
to benefit from the other arm's discoveries. Both arms should have the same task
specification, development examples, tool access relevant to the hypothesis, feedback
opportunities, and declared design budgets. Log unavoidable asymmetries, including
tensor access. Keep baseline construction separate from private evaluation.

Distinguish two experiments:

- **Method comparison:** independent agent sessions design code or weight edits.
  This compares the complete workflows, including the difficulty of each format.
- **Representation comparison:** implement the same identified algorithm in code
  and weights, verifying their agreement where possible. This isolates the cost
  and benefits of the representation more directly.

Measure search quality, latency, memory, candidate/evaluator calls, and the cost of
designing each artifact. If a small program reproduces the gain, the algorithm may
be useful while weight compilation offers no search-quality advantage. If weights
are cheaper or easier to deploy, that remains a legitimate benefit to report.
Failure of one attempted code reconstruction does not establish that no equivalent
program exists.

## Decompose a patch into detection, triggering, and action

For a proposed routing edit, distinguish three claims:

1. The baseline represents the relevant task distinction.
2. The baseline can execute the desired response when appropriately initiated.
3. The edit connects that representation to the response.

A probe can test accessibility of a distinction, but decodability does not establish
that the model uses it. A highly expressive probe can perform much of the task itself.
Report probe complexity, training examples, labels, fitting method, held-out accuracy,
and a simple observation-based comparator. Fit only on permitted development data.

Supervised probe fitting is auxiliary learning even if generator weights remain
frozen. Our gradient-free requirement concerns target updates; it must not be
reported as "no learning anywhere" when a detector was fitted. Match the examples
and optimization budget available to competing detectors.

Use causal tests when feasible: remove the suspected read direction, disable the
edited component, suppress its proposed write direction, or transplant the proposed
runtime intervention into the original model. Record whether the expected change
in behavior actually follows, including effects on unaffected cases. Necessity and
sufficiency are relative to these interventions and can be obscured by redundant
circuits or off-distribution activations.

Compare the permanent weight patch with a runtime hook that uses the same detector
and steers logits or activations without changing stored weights. Similar results
support a routing account and help measure what permanent storage contributes.
Count hook compute and storage rather than treating them as free.

Current `probe` and `generate` contexts differ: the former uses observation JSON,
while generation adds an instruction. Before trajectory-level claims, align prompt,
tokenizer, token positions, grammar masks, and checkpoint identity. Collect probes
from actual generation contexts, and distinguish raw logits from masked probabilities.
A last-token probe from a different context is not evidence for the proposed route.

## Try to recover the inserted procedure

Inspect the weight delta, not just the edited checkpoint: support, rank, repeated
values, scales, component interactions, and activation-dependent effects. Sparse or
simple-looking weights suggest hypotheses; they do not prove program compilation.

Where feasible, write an explicit candidate program and compare its intermediate
outputs and decisions with those of the edited model. For stochastic generation,
compare conditional distributions or suitably controlled trajectories, not only
whether one sampled expression agrees. Search for disagreements on counterfactual
inputs and inputs not used to derive the program.

Ablate individual edit components, vary their strength under a predefined sweep,
and reverse the edit. State where an explanation reproduces behavior and where it
fails. Agreement on visited cases is empirical equivalence on those cases, not a
proof over all inputs. Failure to reconstruct a program is also not evidence that
a capability emerged without one.

## Counterfactuals that distinguish adaptation from global bias

Use paired tasks selected before edit search:

| Contrast | What it tests |
| --- | --- |
| Periodic functions with new frequencies, phases, trends, and compositions | Adaptive choice versus a general preference for sine |
| Similar observed shapes requiring different catalog expressions | Discrimination versus a coarse family trigger |
| Reordered observations | Dependence on presentation order |
| Rescaled inputs/outputs or changed numeric serialization | Dependence on surface encoding |
| Equivalent variable or expression spellings, where supported | Dependence on token spelling |
| Removal or substitution of a favored catalog entry | Reliance on one output versus useful alternatives |
| New function structures and separately reserved families | Transfer beyond development templates |

Declare which transformations preserve the target relation and which deliberately
change the required answer. For inventory changes, measure the new coverage ceiling
and run every competing method under the same changed inventory. Otherwise a drop
may reflect lost expressibility rather than a brittle mechanism.

Finite observations may leave several functions indistinguishable. Failure to infer
an unobservable distinction is not evidence of a deficient strategy. Construct
informative contrasts and report ambiguity, interpolation, and extrapolation separately.
Success on these tests supports conditional adaptation; it does not exclude a
compiled program capable of that adaptation.

## Enumerate the small catalog before attributing value to neural search

The proposal already calls for a coverage oracle. Add a distinct **executable
exhaustive-search baseline**: evaluate every permitted expression on observations
and select by the same objective used by the model-based search. With the default
14 entries, this is an essential practical comparator.

This baseline minimizes the observation objective over the catalog, not held-out
error; it need not select the function with best generalization. Keep an evaluator-only
coverage diagnostic separate from any deployed selection procedure, and never use
private numerical points as search feedback.

Also compare uniform sampling and simple observation-derived heuristics at matched
candidate/evaluator budgets. Exhaustive search may use a different budget, so report
its cost explicitly. A neural method could offer value by reaching useful candidates
with fewer evaluations as inventories grow. A gain over a weak sampler alone does
not establish that weight editing is a useful optimization method for an enumerable
14-expression problem.

## Preservation, controls, and independent edit discovery

Measure target gains against regressions on functions outside the diagnosed failure
mode. Predeclare per-family limits and include control tasks with similar superficial
cues. For claims about general pretrained-model behavior, add a separate small
language-behavior control set, with no reuse of private final answers in edit search.

Exact output agreement and distributional drift are complementary diagnostics;
wording changes are not necessarily errors, and preserved wording is not sufficient
evidence of preserved competence. A task-accuracy versus collateral-change curve
across predefined edit strengths is more informative than one selected operating point.

Keep the existing norm-matched random control, and add random directions matching
rank/support and affected components where practical. Use several random seeds.
An arbitrary random edit is not automatically a fair control for a structured one.
For composite edits, compare components individually and together.

Repeat the editing process in independent agent sessions, not only target sampling.
Five generation seeds for one successful patch do not estimate the probability of
finding a useful patch. Record empty edits, failed hypotheses, human assistance,
validation queries, time, agent tokens, and all proposed artifacts. Do the same for
the code baseline. Analyze task repetitions within functions and editing sessions
as separate sources of variation; the current task-clustered bootstrap does not
estimate between-session discovery uncertainty.

Use nested selection or an untouched final evaluation to address repeated validation
search. Do not pool only successful sessions or interpret a selected validation
interval as confirmatory evidence. A novel task can reduce direct memorization
concerns without excluding use of familiar algorithms; novelty alone does not
establish a newly acquired reasoning mechanism.

## Persistence under later training is a separate question

Bondzio's reported loss of patches during subsequent training motivates an optional
durability study. It does not settle whether an original improvement was a new
capability, a routing change, or an inserted program. Both learned and inserted
computations can be disrupted by later updates.

Our immediate use case is a frozen model. Continued training is outside the initial
experiment and must not be introduced silently into the gradient-free workflow.
If studied later, use a separately declared protocol and track changes in upstream
representations as well as the edited weights. An unchanged detector can stop working
because its input distribution changed. Similarly, quantization or other deployment
transformations require separate robustness measurements, not assumptions of persistence.

## A practical sequence and claim limits

Begin with catalog coverage, exhaustive and uniform baselines, the existing A–E
conditions, and preservation tests. Then add assistance controls and the direct-code
comparison. Investigate detector/action mechanisms and matched runtime steering only
when a reproducible gain gives those probes something to explain. Expand the grammar
before making claims about structural synthesis or long search procedures.

| Observation | Defensible interpretation | Claim not yet established |
| --- | --- | --- |
| Lower error in one edited run | A candidate improvement worth repeating | Reliable editing method |
| Repeated gains across independent functions and sessions | Reproducible behavioral improvement under the tested protocol | Its mechanism |
| Baseline loop recovery reproduces the gain | Execution repair is a plausible explanation | New strategy acquisition |
| A detector plus forced initiation reproduces the gain | Access/routing is sufficient under that intervention | Improved internal search computation |
| A reconstructed program reproduces the gain | Evidence for that procedure on tested inputs | Equivalence on every input |
| Code achieves the same quality at lower cost | The algorithm is useful; weight storage has no demonstrated advantage here | Universal inferiority of weight editing |
| Advantage survives the tested assistance, program, and transfer controls | Stronger evidence for competence beyond those alternatives | Proof of absence from baseline weights or general symbolic intelligence |

For each retained or rejected intervention, append an explanation record: candidate
mechanisms, predicted effects, assistance results, recovered program if any, causal
ablations, counterfactuals, preservation, costs, independent-session outcomes, and
remaining alternatives. Report mixed explanations when appropriate.

## Implementation boundary

The toolkit already supports immutable edits, rollback, candidate records, paired
comparisons, random perturbations, and basic probes. Most controls proposed here
require additional instrumentation or experiment scripts: program baselines,
assistance ladders, aligned trajectory capture, runtime hooks, mechanism ablations,
exhaustive selection, preservation suites, and repeated-session analysis.

This document changes the interpretation and planned design, not the capabilities
of the existing CLI. It reports no new experimental findings.
