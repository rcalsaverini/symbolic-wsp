"""Run outside the editing agent account; never mount its private data directory there."""

import argparse
import json
import os
from pathlib import Path

from symbolic_wsp.experiments.runner import run
from symbolic_wsp.inference.backend import TinyBackend
from symbolic_wsp.symbolic.datasets import generate
from symbolic_wsp.wsp.checkpoints import Store

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--private-dir", required=True)
p.add_argument("--store", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--seed", required=True, type=int)
p.add_argument("--output", required=True)
a = p.parse_args()
private = Path(a.private_dir)
private.mkdir(mode=0o700, parents=True, exist_ok=True)
if private.stat().st_mode & 0o077:
    raise ValueError("private directory must be mode 0700")
if "WSP_EVALUATOR_ISOLATED" not in os.environ:
    raise ValueError(
        "run under separate evaluator account/container, then set WSP_EVALUATOR_ISOLATED=1"
    )
data = generate("final", a.seed, 20)
(private / "tasks.json").write_text(json.dumps(data))
result = run(
    TinyBackend(Store(a.store), a.checkpoint), data, 8, (101, 102, 103), final=True
)
# Only aggregate metrics leave evaluator filesystem.
values = [
    r["heldout"]["nmse"]
    for r in result["rows"]
    if r["heldout"] and r["heldout"]["valid"]
]
Path(a.output).write_text(
    json.dumps(
        dict(
            checkpoint=a.checkpoint,
            mean_nmse=sum(values) / len(values) if values else None,
            valid=len(values),
            total=len(result["rows"]),
        )
    )
)
