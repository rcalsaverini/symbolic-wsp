"""Protocol and selection invariants without network access or pretrained weights."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from symbolic_wsp.experiments.pretrained_baseline import (
    evaluate_candidates,
    make_tasks,
    prompt_messages,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "experiments/pretrained-baseline-01/config.json").read_text())


def test_tasks_and_coverage():
    tasks = make_tasks(17)
    assert tasks == make_tasks(17)
    assert tasks != make_tasks(18)
    assert [t["group"] for t in tasks].count("representable") == 5
    for t in tasks:
        for regime in ("observations", "interpolation", "extrapolation"):
            assert len(t[regime]["x"]) == 24
            assert len(set(t[regime]["x"])) == 24
        assert set(t["observations"]["x"]).isdisjoint(t["interpolation"]["x"])
        assert all(1 <= x <= 2 for x in t["extrapolation"]["x"])
        outcome = evaluate_candidates(t, CFG["inventory"], 0.001)
        subset = evaluate_candidates(t, ["x", "sin(x)"], 0.001)
        assert outcome["best"]["objective"] <= subset["best"]["objective"] + 1e-12
        floor = min(c["nmse"] for c in outcome["candidates"])
        assert (floor < 1e-20) == (t["group"] == "representable")


def test_no_reserved_selection_or_prompt_leakage():
    task = make_tasks(17)[0]
    before = evaluate_candidates(task, CFG["inventory"], 0.001)
    changed = copy.deepcopy(task)
    changed["interpolation"]["y"] = [100] * 24
    changed["extrapolation"]["y"] = [-100] * 24
    after = evaluate_candidates(changed, CFG["inventory"], 0.001)
    assert before["best"] == after["best"]
    assert before["interpolation"] != after["interpolation"]
    obs = dict(task["observations"], target="SECRET", heldout="SECRET")
    assert "SECRET" not in str(prompt_messages(obs))
    assert "No numerical observations" in prompt_messages(None)[0]["content"]


class FakeGenerator:
    parameter_count = 1
    generation = SimpleNamespace(to_dict=lambda: {"test": True})

    def __init__(self):
        self.inputs = []

    def generate(self, observations, k, seed):
        self.inputs.append(observations)
        return dict(
            expressions=["x"] * k,
            token_ids=[[1]] * k,
            seconds=0.0,
            generated_tokens=k,
            generation_calls=1,
        )


def test_complete_protocol_budget_and_failures(tmp_path):
    generator = FakeGenerator()
    summary = run(CFG, generator, tmp_path / "full")
    assert len(generator.inputs) == 60
    assert sum(o is None for o in generator.inputs) == 30
    for observations in generator.inputs:
        if observations is not None:
            assert set(observations) == {"x", "y"}
    rows = [
        json.loads(line)
        for line in (tmp_path / "full/rows.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 100
    for method in ("conditioned", "no_observations", "uniform"):
        assert sum(len(r["candidates"]) for r in rows if r["method"] == method) == 240
    assert sum(len(r["candidates"]) for r in rows if r["method"] == "exhaustive") == 140
    assert all(r["observation_regret"] >= -1e-12 for r in rows)
    assert summary["metrics"]["conditioned"]["all"]["mean_duplicate_rate"] == 0.875
    assert (
        summary["comparisons"]["conditioned_minus_uniform/all/interpolation"]["tasks"]
        == 10
    )
    with pytest.raises(ValueError, match="fresh"):
        run(CFG, generator, tmp_path / "full")
    bad = evaluate_candidates(make_tasks(17)[0], ["log(-1)"], 0.001)
    assert bad["best"] is None and bad["numerical_validity"] == 0


def test_finished_grammar_rows_absorb_only_eos():
    from symbolic_wsp.symbolic.grammar import TokenGrammar

    class CharacterTokenizer:
        eos_token_id = 999

        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))

    grammar = TokenGrammar(CharacterTokenizer(), ["x", "sin(x)"])
    assert grammar.allowed([ord("x"), 999]) == [999]
    assert grammar.allowed([ord("x"), 999, 999, 999]) == [999]
    assert grammar.allowed([ord("s"), 999]) == []
    assert grammar.allowed([ord("x"), 999, ord("x")]) == []
