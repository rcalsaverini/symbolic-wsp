# Experimental protocol and limits

For the research motivation and proposed scientific pilot, see the
[detailed experiment proposal](wsp-symbolic-optimization-experiment.md).
This page describes the current implementation’s operational rules. Additional
proposed controls for capability, execution, and program-insertion explanations are
in [Interpreting WSP improvements](interpreting-wsp-improvements.md).

Original latent-space encoder/decoder/surrogate optimization and direct WSP editing
are different hypotheses. This toolkit tests infrastructure for the latter; it
cannot establish the former. Steering a single answer, changing a proposal
distribution and acquiring a reusable search policy require progressively stronger
evidence. The synthetic categorical-head demo establishes none of these for LLMs.

Conditions: baseline A, observation-feedback B, targeted WSP C, norm-matched random
D and explicit no-edit control. Each receives K candidates per function per seed;
feedback sees only observation losses, never held-out numerical points. Best
candidate selection uses observation NMSE + lambda * AST complexity. Evaluation
then measures that chosen candidate on held-out interpolation and extrapolation
points. Candidate records retain errors, complexity, validity and AST diversity;
exact AST diversity is not semantic diversity. Invalid outcomes are null/error
records, not silently converted to zero error. Paired comparisons refuse mismatched
data fingerprints, K, seeds or objective. Uncertainty uses a function-clustered
paired bootstrap, aggregating repeated seeds first. Failed pairs are counted and
prevent edit acceptance. Acceptance requires at least five functions and an upper
95% bootstrap confidence bound below zero. This is a screening rule, not a
multiple-comparison correction or proof of generalization. Record the full search
history to quantify selection bias.

Task splits use disjoint structural templates and split-derived RNG streams across
polynomial, trig, exponential, rational and composition families. IDs hash the
rendered generating expression, which is omitted from emitted tasks. Structural
deduplication does NOT prove algebraic nonequivalence, and this small template
suite is not a representative benchmark. Expand families before capability claims.
Observation serialization is textual; no adapters or neural training are used.

Final evaluation must be performed by a different evaluator OS account/container
with private data and secret seed never mounted into the agent workspace. The
regular runner refuses final data unless explicitly in evaluator mode. This flag
and the example script's environment assertion are workflow guards, not isolation.
`scripts/final_evaluate.py` creates mode-0700 private storage and exports only
aggregates, but same-UID agents can still bypass permissions: provision actual
separate principals externally. Do not generate final data in the editing workspace.
The supplied final-evaluator example uses the tiny backend; HF runs can use the
same runner from a separately controlled evaluator process.

Wall time and model-generation counts are measured. Agent tokens/cost default to
null (unavailable), never zero. GPU FLOPs, energy, monetary costs and edit-discovery
budgets are not automatically metered; attach external measurements to records.
Matched Frobenius norms do not match random and targeted edit rank/distribution.
Run several random seeds and report edit-discovery cost separately. Predeclare a
budget, hypothesis, validation criterion and regression controls before searching.
No extensive searches or real-model improvements are claimed in this release.
