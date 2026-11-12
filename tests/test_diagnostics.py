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


def test_intervention_gate_records_not_attempted(tmp_path):
    from symbolic_wsp.experiments.intervention import run_intervention

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
