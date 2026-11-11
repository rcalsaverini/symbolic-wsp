Read the [experiment proposal](../../docs/wsp-symbolic-optimization-experiment.md)
for the scientific question, pilot design and implementation gaps before starting.
Read [Interpreting WSP improvements](../../docs/interpreting-wsp-improvements.md)
for competing mechanisms and additional proposed controls. Distinguish implemented
tools from pending instrumentation when planning an investigation.

Investigate a systematic generation failure on development tasks. Read architecture
and methodology first. Freeze the observation prompt, expression inventory, K and
seeds. Inspect baseline weights and successful/failed contexts. State a mechanistic
hypothesis and regression criteria before writing edit JSON. Apply one small edit,
compare with no-edit and norm-matched random controls on validation, and retain or
reject using the predefined rule. Log every failed proposal. Never inspect final
data, train parameters, or interpret isolated steering as a learned search policy.
