"""JSON CLI: results on stdout, structured failures on stderr."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .experiments.runner import paired, record_decision, run
from .inference.backend import TinyBackend
from .symbolic.datasets import generate, save_dataset
from .symbolic.evaluator import score
from .symbolic.grammar import DEFAULT_EXPRESSIONS
from .wsp.checkpoints import Store, compare, inspect
from .wsp.editing import edit, rollback


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def output(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))
    return value


def execute(args):
    store = Store(args.store)
    cmd = args.command
    if cmd == "diagnostics":
        from .experiments.diagnostics import run_study

        return run_study(
            read(args.config),
            {"135m": args.small_model, "360m": args.large_model},
            args.output,
            args.phase,
            args.resume,
        )
    if cmd == "import":
        return dict(checkpoint=store.import_file(args.path))
    if cmd == "inspect":
        return inspect(store.get(args.checkpoint))
    if cmd == "compare":
        return compare(store.get(args.before), store.get(args.after))
    if cmd == "edit":
        return edit(store, args.checkpoint, read(args.spec), args.hypothesis)
    if cmd == "rollback":
        return dict(checkpoint=rollback(store, args.checkpoint))
    if cmd == "export":
        store.export(args.checkpoint, args.path)
        return dict(path=args.path)
    if cmd == "dataset":
        data = generate(args.split, args.seed, args.count)
        save_dataset(data, args.output)
        return dict(path=args.output, tasks=len(data["tasks"]))
    if cmd == "evaluate":
        return score(args.expression, **read(args.observations), penalty=args.penalty)
    if cmd in ("run", "probe", "hf-snapshot", "hf-export"):
        if args.backend == "tiny":
            backend = TinyBackend(store, args.checkpoint)
        else:
            from .inference.hf import HFBackend

            expressions = (
                read(args.expressions) if args.expressions else DEFAULT_EXPRESSIONS
            )
            backend = HFBackend(
                args.model,
                args.device,
                expressions,
                store if args.checkpoint else None,
                args.checkpoint,
            )
        if cmd == "probe":
            return backend.probe(read(args.observations))
        if cmd == "hf-snapshot":
            return dict(checkpoint=backend.snapshot(store))
        if cmd == "hf-export":
            backend.export(args.output)
            return dict(path=args.output)
        return output(
            args.output,
            run(
                backend,
                read(args.dataset),
                args.k,
                tuple(args.seeds),
                args.condition,
                args.penalty,
                args.final_evaluator,
            ),
        )
    if cmd == "paired":
        return paired(read(args.before), read(args.after))
    if cmd == "demo":
        return demo(store, args.output)
    raise ValueError("unknown command")


def demo(store, destination):
    dest = Path(destination)
    if dest.exists():
        raise ValueError("demo output already exists; use a new directory")
    dest.mkdir(parents=True)
    rng = np.random.default_rng(42)
    base = store.put(
        {"head": rng.normal(0, 0.1, (len(DEFAULT_EXPRESSIONS), 2)).astype("float32")},
        dict(source="synthetic fixture"),
    )
    spec = dict(
        mode="rank_one",
        tensor="head",
        alpha=0.15,
        u=[float(i == 7) for i in range(len(DEFAULT_EXPRESSIONS))],
        v=[1.0, 0.0],
    )
    event = edit(
        store,
        base,
        spec,
        "Fixture: increasing sine expression logit may shift proposals; no learned capability claim.",
    )
    child = event["child"]
    magnitude = event["delta"]["head"]["norm"]
    random_edit = edit(
        store,
        base,
        dict(mode="random", tensor="head", norm=magnitude, seed=5),
        "Matched Frobenius-norm random control",
    )
    data = generate("validation", 17, 10)
    output(dest / "validation.json", data)
    results = {}
    for condition, identity in [
        ("baseline", base),
        ("feedback", base),
        ("wsp", child),
        ("random", random_edit["child"]),
        ("no_edit", base),
    ]:
        results[condition] = run(
            TinyBackend(store, identity), data, 8, (1, 2, 3), condition
        )
        output(dest / (condition + ".json"), results[condition])
    report = paired(results["baseline"], results["wsp"])
    record_decision(store, child, results["baseline"], results["wsp"], report)
    restored = rollback(store, child)
    assert restored == base
    return output(
        dest / "summary.json",
        dict(
            synthetic=True,
            baseline=base,
            edited=child,
            random=random_edit["child"],
            comparison=report,
            rollback=restored,
            matched_generation_budget=240,
            claim="Infrastructure demonstration only, no evidence of LLM improvement.",
        ),
    )


def main():
    try:
        parser = Parser(description=__doc__)
        parser.add_argument("--store", default="runs/checkpoints")
        sub = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
        for name in ("import", "export", "inspect", "edit", "rollback", "compare"):
            p = sub.add_parser(name)
            if name not in ("import", "compare"):
                p.add_argument("checkpoint")
            if name in ("import", "export"):
                p.add_argument("path")
            if name == "compare":
                p.add_argument("before")
                p.add_argument("after")
            if name == "edit":
                p.add_argument("--spec", required=True)
                p.add_argument("--hypothesis", required=True)
        p = sub.add_parser("dataset")
        p.add_argument("--split", choices=["dev", "validation", "final"], default="dev")
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--count", type=int, default=10)
        p.add_argument("--output", required=True)
        p = sub.add_parser("evaluate")
        p.add_argument("expression")
        p.add_argument("--observations", required=True)
        p.add_argument("--penalty", type=float, default=0.001)
        for name in ("run", "probe", "hf-snapshot", "hf-export"):
            p = sub.add_parser(name)
            p.add_argument(
                "--backend",
                choices=["tiny", "hf"],
                default="hf" if name.startswith("hf-") else "tiny",
            )
            p.add_argument("--checkpoint")
            p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
            p.add_argument("--device", default="cpu")
            p.add_argument("--expressions")
            if name == "probe":
                p.add_argument("--observations", required=True)
            elif name == "hf-export":
                p.add_argument("--output", required=True)
            elif name == "run":
                p.add_argument("--dataset", required=True)
                p.add_argument("--output", required=True)
                p.add_argument("-k", type=int, default=8)
                p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
                p.add_argument(
                    "--condition",
                    choices=["baseline", "feedback", "wsp", "random", "no_edit"],
                    default="baseline",
                )
                p.add_argument("--penalty", type=float, default=0.001)
                p.add_argument("--final-evaluator", action="store_true")
        p = sub.add_parser("diagnostics")
        p.add_argument(
            "--config", default="experiments/baseline-diagnostics-01/config.json"
        )
        p.add_argument("--small-model", required=True)
        p.add_argument("--large-model", required=True)
        p.add_argument("--output", required=True)
        p.add_argument(
            "--phase",
            choices=["all", "development", "evaluation", "intervention"],
            default="all",
        )
        p.add_argument("--resume", action="store_true")
        p = sub.add_parser("paired")
        p.add_argument("before")
        p.add_argument("after")
        p = sub.add_parser("demo")
        p.add_argument("--output", required=True)
        args = parser.parse_args()
        print(json.dumps(dict(ok=True, result=execute(args)), allow_nan=False))
    except Exception as exc:
        print(
            json.dumps(
                dict(ok=False, error=dict(type=type(exc).__name__, message=str(exc)))
            ),
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
