"""Audit the full diagnostic matrix and intervention from saved probabilities."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from symbolic_wsp.experiments.diagnostics import (
    choose_prompt,
    expected_selection,
    observation_control,
    prompt_variants,
    render,
    tasks,
)
from symbolic_wsp.symbolic.grammar import TokenGrammar


def close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            close(a[k], b[k])
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
    else:
        assert a == b, (a, b)


def verify_distribution(dist, grammar):
    branches = {
        tuple(r["prefix"]): r["token_log_probabilities"] for r in dist["branches"]
    }
    for branch in branches.values():
        assert np.isclose(sum(np.exp(list(branch.values()))), 1.0, atol=1e-7)
    probs = {
        e: float(
            np.exp(
                sum(branches[tuple(seq[:i])][str(token)] for i, token in enumerate(seq))
            )
        )
        for seq, e in grammar.sequences.items()
    }
    mass = sum(probs.values())
    assert np.isclose(mass, 1.0, atol=1e-7)
    for e, p in probs.items():
        assert np.isclose(dist["probabilities"][e], p / mass, rtol=1e-10, atol=1e-12)
    assert hashlib.sha256(dist["prompt"].encode()).hexdigest() == dist["prompt_sha256"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--small-tokenizer", required=True)
    p.add_argument("--large-tokenizer", required=True)
    args = p.parse_args()
    root = args.results
    cfg = json.loads((root / "config.json").read_text())
    choices = json.loads((root / "selections.json").read_text())
    variants, demos = prompt_variants(cfg)
    assert variants == json.loads((root / "prompts.json").read_text())
    rows = [json.loads(s) for s in (root / "rows.jsonl").read_text().splitlines()]
    expected = len(cfg["models"]) * (
        len(cfg["development_targets"])
        * len(cfg["prompt_variants"])
        * len(cfg["development_controls"])
        + len(cfg["evaluation_targets"])
        * len(cfg["prompt_variants"])
        * len(cfg["observation_controls"])
    )
    assert len(rows) == expected
    keys = {
        (r["model"], r["phase"], r["task"], r["variant"], r["control"]) for r in rows
    }
    assert len(keys) == expected
    tokenizers = {
        key: AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        for key, path in [
            ("135m", args.small_tokenizer),
            ("360m", args.large_tokenizer),
        ]
    }
    grammars = {
        key: TokenGrammar(tok, cfg["inventory"]) for key, tok in tokenizers.items()
    }
    for r in rows:
        examples = tasks(
            cfg["development_targets"]
            if r["phase"] == "development"
            else cfg["evaluation_targets"],
            cfg["development_seed"]
            if r["phase"] == "development"
            else cfg["evaluation_seed"],
            cfg["points"],
        )
        task = next(t for t in examples if t["id"] == r["task"])
        obs = observation_control(task, examples, r["control"], cfg["control_seed"])
        assert (
            render(tokenizers[r["model"]], obs, variants[r["variant"]])
            == r["distribution"]["prompt"]
        )
        verify_distribution(r["distribution"], grammars[r["model"]])
        close(
            expected_selection(
                r["distribution"]["probabilities"],
                task,
                cfg["inventory"],
                cfg["k"],
                cfg["penalty"],
            ),
            r["metrics"],
        )
    for model in choices:
        close(
            choose_prompt(
                [
                    r
                    for r in rows
                    if r["model"] == model and r["phase"] == "development"
                ],
                variants,
            ),
            choices[model],
        )
    inter = json.loads((root / "intervention/summary.json").read_text())
    if inter["status"] == "completed":
        edits = [
            json.loads(s)
            for s in (root / "intervention/rows.jsonl").read_text().splitlines()
        ]
        assert len(edits) == len(cfg["development_targets"]) * (
            1 + len(cfg["intervention"]["relative_norms"])
        ) + len(cfg["evaluation_targets"]) * (
            4 + len(cfg["intervention"]["random_seeds"])
        )
        for r in edits:
            phase = r["condition"].split("/")[0]
            examples = tasks(
                cfg["development_targets"]
                if phase == "development"
                else cfg["evaluation_targets"],
                cfg["development_seed"]
                if phase == "development"
                else cfg["evaluation_seed"],
                cfg["points"],
            )
            task = next(t for t in examples if t["id"] == r["task"])
            verify_distribution(
                r["distribution"], grammars[cfg["intervention"]["model"]]
            )
            close(
                expected_selection(
                    r["distribution"]["probabilities"],
                    task,
                    cfg["inventory"],
                    cfg["k"],
                    cfg["penalty"],
                ),
                r["metrics"],
            )
        for r in edits:
            if r["condition"] == "evaluation/no_edit":
                baseline = next(
                    x
                    for x in edits
                    if x["condition"] == "evaluation/baseline"
                    and x["task"] == r["task"]
                )
                close(r["metrics"], baseline["metrics"])
                close(
                    r["distribution"]["probabilities"],
                    baseline["distribution"]["probabilities"],
                )
        assert json.loads((root / "intervention/rollback.json").read_text())["restored"]
    provenance = json.loads((root / "provenance.json").read_text())
    for phase, commit in provenance.items():
        env = json.loads((root / f"environment-{phase}.json").read_text())
        for path, sha in env["source_sha256"].items():
            snapshot = root / "source-snapshots" / (sha + ".py")
            if snapshot.exists():
                raw = snapshot.read_bytes()
            elif commit is not None:
                raw = subprocess.check_output(["git", "show", commit + ":" + path])
            else:
                raw = Path(path).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == sha, (phase, path)
    print(
        f"PASS: {len(rows)} matrix rows, all prefix/leaf probabilities, prompts, expected metrics, selections, intervention records, no-edit equality, rollback and phase source provenance verified."
    )


if __name__ == "__main__":
    main()
