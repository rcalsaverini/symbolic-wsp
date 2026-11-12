"""Scientific invariants for the first-class diagnostic experiments."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from symbolic_wsp.experiments.diagnostics import (
    choose_prompt,
    expected_selection,
    observation_control,
    prompt_variants,
    tasks,
)
from symbolic_wsp.inference.catalog import probabilities_from_logits
from symbolic_wsp.symbolic.evaluator import score

CFG = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "experiments/baseline-diagnostics-01/config.json"
    ).read_text()
)


def test_masked_probability_normalization_at_each_prefix():
    seqs = [(1, 0), (1, 3, 0), (2, 0)]
    logits = {
        (): np.array([999.0, np.log(3), 0.0, -5.0, 10000.0]),
        (1,): np.array([np.log(2), 999.0, 999.0, 0.0, 10000.0]),
        (1, 3): np.zeros(5),
        (2,): np.zeros(5),
    }
    probs, _ = probabilities_from_logits(seqs, logits)
    assert list(probs.values()) == pytest.approx([0.5, 0.25, 0.25])
    assert sum(probs.values()) == pytest.approx(1.0)


def test_best_k_expectation_matches_exhaustive_draw_enumeration():
    task = tasks(["x"], 7, 8)[0]
    inventory = ["x", "0", "1"]
    p = dict(x=0.2, **{"0": 0.3, "1": 0.5})
    result = expected_selection(p, task, inventory, 3, 0.001)
    empirical = {e: 0.0 for e in inventory}
    for draw in itertools.product(inventory, repeat=3):
        best = min(
            draw, key=lambda e: (score(e, **task["observations"])["objective"], e)
        )
        empirical[best] += np.prod([p[e] for e in draw])
    assert result["selection_probabilities"] == pytest.approx(empirical)
    assert result["exact_target_hit_probability"] == pytest.approx(1 - 0.8**3)
    with pytest.raises(ValueError):
        expected_selection(dict(x=0.4), task, inventory, 3, 0.001)


def test_observation_controls_and_task_separation():
    ts = tasks(["x", "x*x"], 17, 24)
    original = ts[0]["observations"]
    shuffled = observation_control(ts[0], ts, "shuffled", 1)
    assert shuffled["x"] == original["x"]
    assert (
        sorted(shuffled["y"]) == sorted(original["y"])
        and shuffled["y"] != original["y"]
    )
    other = observation_control(ts[0], ts, "other_function", 1)
    np.testing.assert_allclose(other["y"], np.array(original["x"]) ** 2)
    assert observation_control(ts[0], ts, "absent", 1) is None
    assert ts == tasks(["x", "x*x"], 17, 24)
    assert set(CFG["development_targets"]).isdisjoint(CFG["evaluation_targets"])
    assert set(CFG["demonstration_targets"]).isdisjoint(
        CFG["development_targets"] + CFG["evaluation_targets"]
    )
    variants, demos = prompt_variants(CFG)
    assert set(variants) == set(CFG["prompt_variants"])
    assert variants["few_shot"]["user_template"].count("{observations}") == 1
    assert len(demos) == 2


def test_prompt_selection_uses_development_correct_only():
    rows = [
        dict(variant=v, control=c, metrics=dict(expected_observation_regret=r))
        for v, c, r in [
            ("minimal", "correct", 2.0),
            ("catalog", "correct", 1.0),
            ("minimal", "shuffled", 0.0),
            ("catalog", "shuffled", 99.0),
        ]
    ]
    assert choose_prompt(rows, ["minimal", "catalog"])["variant"] == "catalog"


def test_intervention_gate_records_not_attempted(tmp_path, monkeypatch):
    import sys

    from symbolic_wsp.experiments.intervention import run_intervention

    monkeypatch.setitem(sys.modules, "torch", None)
    variants, _ = prompt_variants(CFG)
    result = run_intervention(
        None, None, None, CFG, variants, dict(variant="minimal"), [], tmp_path
    )
    assert result["status"] == "not_attempted"
    assert (
        json.loads((tmp_path / "intervention/gate.json").read_text())["failing_targets"]
        == []
    )
    assert not (tmp_path / "intervention/tensor-checkpoints").exists()


def test_first_class_phase_workflow_and_resume(tmp_path, monkeypatch):
    import copy
    import hashlib

    from symbolic_wsp.experiments import diagnostics, intervention

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, sort_keys=True)

    class Oracle:
        def __init__(self, model, tokenizer, inventory):
            self.inventory = inventory

        def distribution(self, prompt):
            return dict(
                probabilities={e: 1 / len(self.inventory) for e in self.inventory},
                seconds=0.0,
                cache_hit=False,
                prompt=prompt,
            )

    monkeypatch.setattr(diagnostics, "validate_model", lambda path, identity: identity)
    monkeypatch.setattr(
        diagnostics, "load_model", lambda path, cfg: (object(), Tokenizer())
    )
    monkeypatch.setattr(diagnostics, "CatalogOracle", Oracle)
    monkeypatch.setattr(
        diagnostics.importlib.metadata, "version", lambda name: "offline-test"
    )

    def skip_intervention(*args):
        target = args[-1] / "intervention"
        target.mkdir(exist_ok=True)
        (target / "summary.json").write_text(
            json.dumps(dict(status="not_attempted", reason="offline workflow fixture"))
        )

    monkeypatch.setattr(intervention, "run_intervention", skip_intervention)
    out = tmp_path / "study"
    paths = {"135m": "unused", "360m": "unused"}
    result = diagnostics.run_study(CFG, paths, out, phase="all")
    assert result["rows"] == 270
    assert len((out / "rows.jsonl").read_text().splitlines()) == 270
    assert set(json.loads((out / "provenance.json").read_text())) == {
        "development",
        "intervention",
        "evaluation",
    }
    for phase in ("development", "intervention", "evaluation"):
        env = json.loads((out / ("environment-" + phase + ".json")).read_text())
        for sha in env["source_sha256"].values():
            assert (
                hashlib.sha256(
                    (out / "source-snapshots" / (sha + ".py")).read_bytes()
                ).hexdigest()
                == sha
            )
    diagnostics.run_study(CFG, paths, out, phase="evaluation", resume=True)
    assert len((out / "rows.jsonl").read_text().splitlines()) == 270
    changed = copy.deepcopy(CFG)
    changed["k"] = 9
    with pytest.raises(ValueError, match="configuration mismatch"):
        diagnostics.run_study(changed, paths, out, phase="evaluation", resume=True)

    monkeypatch.setattr(
        diagnostics, "source_fingerprint", lambda: {"changed": "source"}
    )
    with pytest.raises(ValueError, match="source changed"):
        diagnostics.run_study(CFG, paths, out, phase="evaluation", resume=True)
