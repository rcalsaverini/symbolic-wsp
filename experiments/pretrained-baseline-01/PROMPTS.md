# Exact prompts and task configuration

The prompt is part of the experiment configuration. The active
[`config.json`](config.json) contains a required `prompt` block with:

- `version`: prompt configuration schema version.
- `system_message`: explicit system text, including the model's original default.
- `user_template`: literal text with exactly one `{observations}` placeholder.
- `no_observations`: replacement text for the ablated condition.
- `significant_digits`: numerical serialization precision for x/y arrays.
- `chat_template`: the complete Jinja chat template; no tokenizer-default lookup.
- `add_generation_prompt`: whether to append the assistant-role prefix.

Changing any of these fields changes the configuration digest. New runs save
`prompt-config.json` and its digest, as well as each rendered prompt and its hash.
The generator must use the same prompt specification as the experiment runner;
a mismatch is rejected. There are no implicit fallback prompts for new runs.

## The prompt actually used in the completed study

The original implementation supplied this user template:

```text
Find an expression in x for these observations. Return only the expression.
{observations}
```

For the conditioned case, `{observations}` is compact JSON with `x` and `y` arrays
rounded to six significant digits. For the no-observations case it is exactly:

```text
No numerical observations are provided.
```

The pinned tokenizer's chat template added this system message automatically in
the original run:

```text
You are a helpful AI assistant named SmolLM, trained by Hugging Face
```

The active configuration supplies that message explicitly and pins the same chat
template. This preserves the actual rendered inputs while removing the hidden
reliance on tokenizer defaults.

## Verbatim rendered inputs

These examples are extracted from the first `task-00`, seed-11 rows of
[`results/rows.jsonl`](results/rows.jsonl), not reconstructed approximations.
The special-token spellings represent the exact chat delimiters. The final
assistant prefix is where expression generation starts.

### Conditioned: task-00

Prompt tokens: **483**. [Exact text file](prompts/task-00.txt).

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
Find an expression in x for these observations. Return only the expression.
{"x":[-0.0157018,0.974073,0.68532,0.38077,-0.874198,0.803298,-0.236768,-0.710856,-0.462461,-0.460773,-0.911323,0.490641,0.370467,0.782613,0.513199,-0.890547,0.639824,0.919957,-0.128855,0.130909,-0.124573,0.661652,-0.317636,-0.970031],"y":[-0.0157012,0.827181,0.63292,0.371635,-0.767029,0.71965,-0.234562,-0.652482,-0.446152,-0.444641,-0.790315,0.471191,0.362051,0.705135,0.490967,-0.777416,0.597054,0.795576,-0.128499,0.130536,-0.124251,0.614421,-0.312321,-0.824903]}<|im_end|>
<|im_start|>assistant
```

### No observations

Prompt tokens: **51**. [Exact text file](prompts/no-observations.txt).

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
Find an expression in x for these observations. Return only the expression.
No numerical observations are provided.<|im_end|>
<|im_start|>assistant
```

## All task prompts and validation

Each task's conditioned prompt is identical across its three sampling seeds. The
no-observations prompt is identical across all tasks and seeds. The 60 model calls
therefore use 11 distinct prompt texts:

- [Conditioned task-00](prompts/task-00.txt)
- [Conditioned task-01](prompts/task-01.txt)
- [Conditioned task-02](prompts/task-02.txt)
- [Conditioned task-03](prompts/task-03.txt)
- [Conditioned task-04](prompts/task-04.txt)
- [Conditioned task-05](prompts/task-05.txt)
- [Conditioned task-06](prompts/task-06.txt)
- [Conditioned task-07](prompts/task-07.txt)
- [Conditioned task-08](prompts/task-08.txt)
- [Conditioned task-09](prompts/task-09.txt)
- [No-observations control](prompts/no-observations.txt)

[`prompts/manifest.json`](prompts/manifest.json) maps all 60 calls to those files,
with their SHA256 hashes and token counts. The configured replay was checked
against every recorded prompt byte-for-byte, using the pinned local tokenizer;
all prompt token counts also matched. This validation does not rerun model inference
or create new performance measurements.

To export and verify again, after preparing the pinned model/tokenizer locally:

```bash
uv run --no-sync python scripts/export_baseline_prompts.py --tokenizer runs/models/smollm2-135m --output runs/exported-prompts
```

## Historical configuration versus the explicit replay

The completed study used a schema-1 config without a prompt block. Its immutable
configuration is [`results/config.json`](results/config.json), and its original
digest remains in [`results/summary.json`](results/summary.json). Do not replace
those files with the new configuration or claim that the original run logged
fields it did not log.

The active schema-2 config explicitly captures the same rendered prompts; its
configuration digest differs. The prompt manifest records both identities. Future
runs use the explicit config and write new metadata in a fresh output directory.
The actual execution source was preserved under [`results/source/`](results/source/)
and checked against the original source hashes. The exact-score verifier uses those
snapshots when validating historical provenance. The original code is also available
at commit `942e8d4`.

## Interpretation

This was a minimal zero-shot instruction, with no demonstrations, strategy hints,
function-family labels, or list of permitted expressions in the prompt. The catalog
was enforced by the external token mask only. It was not a prompt-engineering study
or a test of the best elicitable performance of the model.

Consequently, poor results under this prompt do not establish absent mathematical
capability. The two conditions also differ in length and token content, so any
conditioning effect alone would not establish numerical understanding. Alternative
prompts, examples, catalog descriptions, or number formats belong to separately
configured follow-ups; do not retroactively change this study's inputs or results.
