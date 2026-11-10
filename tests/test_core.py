import json
import subprocess
import sys

import numpy as np
import pytest

from symbolic_wsp.cli import demo
from symbolic_wsp.experiments.runner import paired, run
from symbolic_wsp.inference.backend import TinyBackend
from symbolic_wsp.symbolic.datasets import generate
from symbolic_wsp.symbolic.evaluator import Grammar, evaluate, score
from symbolic_wsp.symbolic.grammar import TokenGrammar
from symbolic_wsp.wsp.checkpoints import Store
from symbolic_wsp.wsp.editing import edit, rollback


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "x.__class__",
        "[x]",
        "x[0]",
        "sum(x)",
        "sin(x,y)",
        "True",
        "(lambda:1)()",
        "x//2",
        "1e999",
        "1/0",
        "log(-1)",
        "sqrt(-1)",
        "exp(1000)",
        "x**100000",
    ],
)
def test_reject(expr):
    assert not score(expr, [1, 2], [1, 2])["valid"]


def test_evaluation():
    np.testing.assert_allclose(evaluate("sin(x)^2+cos(x)^2", [0.1, 0.7]), 1)
    assert score("x*x", [1, 2], [1, 4])["nmse"] == 0
    assert not score("sin(x)", [1], [1], grammar=Grammar(functions=()))["valid"]
    assert not score("x", [1, 2], [1])["valid"]


def test_versions(tmp_path):
    s = Store(tmp_path)
    base = s.put({"w": np.eye(3, dtype="float32")})
    e = edit(
        s,
        base,
        dict(mode="rank_one", tensor="w", u=[1, 0, 0], v=[0, 1, 0], alpha=0.5),
        "test",
    )
    assert s.get(e["child"])["w"][0, 1] == 0.5
    assert s.get(base)["w"][0, 1] == 0
    assert rollback(s, e["child"]) == base
    for spec in [
        dict(mode="project_out", tensor="w", direction=[1, 0, 0]),
        dict(mode="ablate", tensor="w", axis=0, indices=[0]),
    ]:
        e = edit(s, base, spec, "remove first output")
        np.testing.assert_array_equal(s.get(e["child"])["w"][0], 0)
    n = len(list(tmp_path.iterdir()))
    with pytest.raises(ValueError):
        edit(s, base, dict(mode="dsl", dsl="w[0,0] = 4\nunknown[0] = 1"), "failure")
    assert len(list(tmp_path.iterdir())) == n
    e = edit(s, base, dict(mode="random", tensor="w", norm=0.25, seed=7), "control")
    assert e["delta"]["w"]["norm"] == pytest.approx(0.25, rel=1e-6)


class MultiToken:
    eos_token_id = 999

    def encode(self, s, **kwargs):
        return [ord(c) for c in s]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def test_grammar():
    g = TokenGrammar(MultiToken(), ["sin(x)", "x+1"])
    assert g.allowed([ord("s")]) == [ord("i")]
    assert g.allowed(list(map(ord, "sin(x)"))) == [999]
    assert g.allowed([ord("z")]) == []


def test_datasets():
    a = generate("dev", 17, 10)
    b = generate("validation", 17, 10)
    assert a == generate("dev", 17, 10)
    assert not {t["id"] for t in a["tasks"]} & {t["id"] for t in b["tasks"]}


def test_demo(tmp_path):
    s = Store(tmp_path / "store")
    r = demo(s, tmp_path / "demo")
    assert r["rollback"] == r["baseline"]
    a = json.loads((tmp_path / "demo/baseline.json").read_text())
    b = json.loads((tmp_path / "demo/no_edit.json").read_text())
    assert a["rows"] == b["rows"] and paired(a, b)["mean_delta"] == 0
    assert a["model_evaluations"] == 240
    b["k"] = 9
    with pytest.raises(ValueError):
        paired(a, b)
    with pytest.raises(ValueError):
        run(TinyBackend(s, r["baseline"]), generate("final", 123, 1))


def test_cli(tmp_path):
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "symbolic_wsp.cli",
            "--store",
            str(tmp_path),
            "nonsense",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and json.loads(p.stderr)["ok"] is False


def test_skill_installation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for client in (".agents", ".claude"):
        for name in ("weight-space-programming", "symbolic-optimization"):
            path = root / client / "skills" / name / "SKILL.md"
            assert path.read_text() == (root / "skills" / name / "SKILL.md").read_text()


def test_integrity_and_lineage(tmp_path):
    from safetensors.numpy import save_file

    s = Store(tmp_path)
    base = s.put({"w": np.eye(2)})
    no_op = edit(
        s,
        base,
        dict(mode="ablate", tensor="w", axis=0, indices=[0], scale=1),
        "no-op audit",
    )
    assert no_op["child"] != base
    assert rollback(s, no_op["child"]) == base
    save_file({"w": np.zeros((2, 2))}, str(tmp_path / base / "model.safetensors"))
    with pytest.raises(ValueError, match="integrity"):
        s.get(base)
