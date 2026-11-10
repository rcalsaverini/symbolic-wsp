# Upstream review

Reviewed Felipe Penha's `pi-weight-space-programming` at
`13a486e19685701ab1916ca2a85fdc02c928b635`, including
`harness/wsp_engine.py`, `harness/prelude.py`, `extensions/index.ts`,
`harness/presets.json`, weight-space-surgery skill, and model-poisoning and
knowledge-injection experiment READMEs. Original research checkout:
`RikiyaT/lm_understand-logs` at `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

Reuse: `upstream_dsl.py` extracts the slice parser, tensor-name resolver and
`apply_edits_dsl` unchanged in substance from the Apache-2.0 engine. The editing
module adapts its rank-one, output-direction projection and ablation formulas.
LICENSE and complete upstream NOTICE are retained. New wrappers add immutable
checkpoint identities, dimension checks and all-or-nothing candidate publication.
DSL failures never publish the partially modified in-memory copy.

Not imported: the prelude patches builtins and directory listing to hide evaluation
paths. Such guards are not security boundaries. Its BF16 sparse manifest logic
also loses exact float32 deltas, so this project retains complete original states.
The extension is a Pi API/tool registration layer; it is replaced by a shell CLI.
Presets impose experiment-specific tool budgets; our runner records explicit K and
repeated seeds. Architecture-specific QK/OV circuit heuristics are not generalized
here: portable SVD geometry and real activation/logit probes provide the first
slice without assuming GPT/Qwen tensor naming.

Poisoning and knowledge injection illustrate highly targeted behavioral routing.
Their reported successes do not establish reusable symbolic optimization and are
not used as benchmarks or copied into this project's claims. No changes were made
in either upstream repository.
