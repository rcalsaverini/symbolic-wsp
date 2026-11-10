# Architecture

The pure Python engine is independent of Pi and agent APIs. `symbolic-wsp` exposes
JSON on stdout and errors `{ok:false,error:{type,message}}` on stderr (exit 2).
Use `--help` for argument contracts. Imports never download models.

Checkpoint state consists of a complete finite float tensor dictionary. SHA256
covers tensor names, dtype, shape, bytes and creation metadata (including parent
lineage). Loads verify the digest before using a state. Immutable directories hold
`model.safetensors` and creation metadata. `history.jsonl` records every proposal,
including identical/no-op proposals, decisions and rollback. Rollback returns the
parent identity; callers must explicitly use it in subsequent inference. No global
mutable "current model" exists. Store is single-writer; concurrent writes are not
supported. A filesystem owner can alter files: use read-only mounts for baseline
archives. Full snapshots favor correctness over storage efficiency.

NumPy engine supports NPZ and safetensors; no pickle. The HF adapter snapshots a
complete state dict as float32, loads edits strictly into an explicitly selected
model architecture and can export native Transformers directories with tokenizer.
Nonfloating buffers and bfloat16 preservation are currently unsupported by the
portable store. Use a small dense causal model; quantized/sharded huge models are
outside the tested scope. Device is explicit (`cpu`, `cuda`, etc). A separate shell
process running `run --backend hf --device cuda` acts as the GPU worker; no network
service is required. Hooks are not installed globally. `probe` returns last-token
hidden states and top logits from a real forward pass; a run records candidate
trajectories but does not persist all per-token activations.

Grammar is deliberately a **finite bounded expression language**, supplied as a
JSON list (`--expressions`). The AST validator is configurable in the Python API.
Every permitted expression is parsed before generation. Complete pretrained-token
sequences form a trie; masking allows only paths to a valid EOS leaf. Operators
and functions may span several tokens. Tokenizer and LM head are unchanged.
This is exact within the configured language, not a general EBNF parser. Larger
expression inventories require more CPU time; arbitrary open-ended structural
search is future work. No fabricated fallback expression is emitted.

The evaluator recursively interprets whitelisted AST nodes with NumPy. Character,
node, depth, point-count, constant and exponent limits bound work. Domain and
nonfinite outputs are rejected. No `eval`, generated Python, SymPy parser or
lambdify is used. This is a bounded interpreter, not a security sandbox for arbitrary
programs. Constant fitting is not implemented; structure and constant optimization
can be separated later without updating generator weights.
