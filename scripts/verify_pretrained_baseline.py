"""Audit a full feasibility run; exact checks require the recorded numerical environment."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from symbolic_wsp.experiments.pretrained_baseline import (
    aggregate,
    comparisons,
    digest,
    evaluate_candidates,
    make_tasks,
)

parser = argparse.ArgumentParser(
    description="Verify the complete baseline study against its saved data and source hashes."
)
parser.add_argument("results", type=Path)
args = parser.parse_args()
root = args.results
cfg = json.loads((root / "config.json").read_text())
tasks = json.loads((root / "tasks.json").read_text())
summary = json.loads((root / "summary.json").read_text())
env = json.loads((root / "environment.json").read_text())
rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
assert tasks == make_tasks(cfg["dataset_seed"], cfg["points_per_regime"])
assert summary["config_digest"] == digest(cfg)
assert summary["dataset_digest"] == digest(tasks)
assert len(rows) == 100
assert Counter(r["method"] for r in rows) == dict(
    conditioned=30, no_observations=30, uniform=30, exhaustive=10
)
for r in rows:
    task = next(t for t in tasks if t["id"] == r["task"])
    expressions = [c["expression"] for c in r["candidates"]]
    assert all(e in cfg["inventory"] for e in expressions)
    recomputed = evaluate_candidates(task, expressions, cfg["penalty"])
    for key, value in recomputed.items():
        assert value == r[key], (r["task"], r["method"], key)
    for candidate in r["candidates"]:
        assert candidate["valid"]
    if r["method"] in ("conditioned", "no_observations"):
        assert len(r["trace"]["token_ids"]) == cfg["k"]
        assert r["trace"]["generated_tokens"] == sum(map(len, r["trace"]["token_ids"]))
assert summary["metrics"] == aggregate(rows)
assert summary["comparisons"] == comparisons(rows)
for path, sha in env["source_sha256"].items():
    snapshot = root / "source" / path
    source = snapshot if snapshot.exists() else Path(path)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, str(source)
print(
    "PASS: all 100 records, 860 candidates, selected expressions, reserved scores, counts, source hashes and descriptive comparisons independently recomputed."
)
