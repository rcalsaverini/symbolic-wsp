"""Small pretrained feasibility study; no weight updates or private final data."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..symbolic.evaluator import evaluate, score
from ..symbolic.grammar import TokenGrammar

TARGETS = (
    ("representable", "sin(x)"),
    ("representable", "cos(x)"),
    ("representable", "x*x"),
    ("representable", "exp(x)"),
    ("representable", "1/(x*x+1)"),
    ("outside_catalog", "sin(2*x)"),
    ("outside_catalog", "cos(2*x)"),
    ("outside_catalog", "2*x*x"),
    ("outside_catalog", "exp(x/2)"),
    ("outside_catalog", "1/(x*x+2)"),
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_tasks(seed: int, points: int = 24) -> list[dict]:
    tasks = []
    for index, (group, target) in enumerate(TARGETS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        task = dict(id=f"task-{index:02d}", group=group, target=target)
        for regime, low, high in [
            ("observations", -1, 1),
            ("interpolation", -1, 1),
            ("extrapolation", 1, 2),
        ]:
            x = rng.uniform(low, high, points)
            task[regime] = dict(x=x.tolist(), y=evaluate(target, x).tolist())
        tasks.append(task)
    return tasks


def prompt_messages(observations: dict | None) -> list[dict]:
    instruction = (
        "Find an expression in x for these observations. Return only the expression."
    )
    if observations is None:
        payload = "No numerical observations are provided."
    else:
        # Explicit serialization: precision differs from evaluator's float64 inputs.
        payload = json.dumps(
            {key: [float(f"{v:.6g}") for v in observations[key]] for key in ("x", "y")},
            separators=(",", ":"),
        )
    return [dict(role="user", content=instruction + "\n" + payload)]


class BaselineGenerator:
    """Explicit decoding config and chat serialization, local weights only."""

    def __init__(self, model_path: str, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.torch = torch
        torch.set_num_threads(cfg["torch_threads"])
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=False
            )
            .to("cpu")
            .eval()
        )
        self.grammar = TokenGrammar(self.tokenizer, cfg["inventory"])
        self.generation = GenerationConfig(
            **cfg["generation"],
            max_new_tokens=max(map(len, self.grammar.sequences)),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def generate(self, observations: dict | None, k: int, seed: int) -> dict:
        messages = prompt_messages(observations)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        width = ids.shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and width + self.generation.max_new_tokens > limit:
            raise ValueError("prompt exceeds context; truncation forbidden")

        def allowed(batch_id, tokens):
            return self.grammar.allowed(tokens[width:].tolist())

        self.torch.manual_seed(seed)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                ids,
                attention_mask=self.torch.ones_like(ids),
                generation_config=self.generation,
                num_return_sequences=k,
                prefix_allowed_tokens_fn=allowed,
            )
        elapsed = time.perf_counter() - started
        sequences = []
        for row in output[:, width:].tolist():
            if self.grammar.eos not in row:
                raise ValueError("generation failed to terminate in grammar")
            row = row[: row.index(self.grammar.eos) + 1]
            sequences.append(row)
        return dict(
            expressions=[self.grammar.expression(row) for row in sequences],
            token_ids=sequences,
            prompt=prompt,
            prompt_tokens=width,
            generated_tokens=sum(map(len, sequences)),
            prefill_token_slots=width * k,
            generation_calls=1,
            seconds=elapsed,
        )


def evaluate_candidates(task: dict, expressions: list[str], penalty: float) -> dict:
    candidates = [
        dict(expression=e, **score(e, **task["observations"], penalty=penalty))
        for e in expressions
    ]
    valid = [c for c in candidates if c["valid"]]
    # Canonical lexical tie-break is independent of sampling order.
    best = (
        min(valid, key=lambda c: (c["objective"], c["expression"])) if valid else None
    )
    return dict(
        candidates=candidates,
        best=best,
        interpolation=score(
            best["expression"], **task["interpolation"], penalty=penalty
        )
        if best
        else None,
        extrapolation=score(
            best["expression"], **task["extrapolation"], penalty=penalty
        )
        if best
        else None,
        numerical_validity=len(valid) / len(candidates),
        duplicate_rate=1 - len(set(expressions)) / len(expressions),
        unique_expressions=len(set(expressions)),
        evaluator_calls=len(expressions) + (2 if best else 0),
    )


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in sorted({r["method"] for r in rows}):
        result[method] = {}
        for group in ("representable", "outside_catalog", "all"):
            subset = [
                r
                for r in rows
                if r["method"] == method and (group == "all" or r["group"] == group)
            ]
            metrics = {}
            for key in ("observation_regret", "numerical_validity", "duplicate_rate"):
                values = [r[key] for r in subset if r[key] is not None]
                metrics["mean_" + key] = float(np.mean(values)) if values else None
            for regime in ("interpolation", "extrapolation"):
                for metric in ("nmse", "mse"):
                    values = [
                        r[regime][metric]
                        for r in subset
                        if r[regime] and r[regime]["valid"]
                    ]
                    metrics[f"mean_{regime}_{metric}"] = (
                        float(np.mean(values)) if values else None
                    )
            best = [r["best"] for r in subset if r["best"]]
            metrics.update(
                rows=len(subset),
                tasks=len({r["task"] for r in subset}),
                selected_exact_target_rate=sum(
                    r["best"] is not None and r["best"]["expression"] == r["target"]
                    for r in subset
                )
                / len(subset),
                mean_selected_complexity=float(np.mean([r["complexity"] for r in best]))
                if best
                else None,
                candidate_frequencies=dict(
                    Counter(c["expression"] for r in subset for c in r["candidates"])
                ),
                generation_seconds=sum(r["trace"]["seconds"] for r in subset),
                generated_tokens=sum(
                    r["trace"].get("generated_tokens", 0) for r in subset
                ),
                generation_calls=sum(
                    r["trace"].get("generation_calls", 0) for r in subset
                ),
                evaluator_calls=sum(r["evaluator_calls"] for r in subset),
                failed_selections=sum(r["best"] is None for r in subset),
            )
            result[method][group] = metrics
    return result


def comparisons(rows: list[dict]) -> dict:
    """Descriptive function-level intervals; tiny n and no confirmatory decision."""
    results = {}
    for comparator in ("no_observations", "uniform", "exhaustive"):
        for group in ("representable", "outside_catalog", "all"):
            for regime in ("interpolation", "extrapolation"):
                differences = []
                for task in sorted(
                    {r["task"] for r in rows if group == "all" or r["group"] == group}
                ):
                    values = {}
                    for method in ("conditioned", comparator):
                        rs = [
                            r
                            for r in rows
                            if r["task"] == task and r["method"] == method
                        ]
                        values[method] = [
                            r[regime]["nmse"]
                            for r in rs
                            if r[regime] and r[regime]["valid"]
                        ]
                    if values["conditioned"] and values[comparator]:
                        differences.append(
                            float(
                                np.mean(values["conditioned"])
                                - np.mean(values[comparator])
                            )
                        )
                if differences:
                    rng = np.random.default_rng(123)
                    means = rng.choice(
                        differences, (2000, len(differences)), replace=True
                    ).mean(axis=1)
                    results[f"conditioned_minus_{comparator}/{group}/{regime}"] = dict(
                        task_deltas=differences,
                        mean_delta=float(np.mean(differences)),
                        descriptive_ci95=np.quantile(means, [0.025, 0.975]).tolist(),
                        tasks=len(differences),
                    )
    return results


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(
    cfg: dict, generator: BaselineGenerator, output: Path, smoke: bool = False
) -> dict:
    if output.exists():
        raise ValueError("use a fresh output directory")
    output.mkdir(parents=True)
    tasks = make_tasks(cfg["dataset_seed"], cfg["points_per_regime"])
    if smoke:
        tasks = [tasks[0], tasks[5]]
    write(output / "config.json", cfg)
    write(output / "tasks.json", tasks)
    write(output / "effective-generation-config.json", generator.generation.to_dict())
    rows = []
    coverage = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        exhaustive_started = time.perf_counter()
        exhaustive = evaluate_candidates(task, cfg["inventory"], cfg["penalty"])
        exhaustive_seconds = time.perf_counter() - exhaustive_started
        if exhaustive["best"] is None:
            raise ValueError("no valid inventory candidates")
        ceiling = exhaustive["best"]["objective"]
        coverage.append(
            dict(
                task=task["id"],
                group=task["group"],
                target=task["target"],
                target_in_inventory=task["target"] in cfg["inventory"],
                min_observation_nmse=min(
                    c["nmse"] for c in exhaustive["candidates"] if c["valid"]
                ),
                observation_selector=exhaustive["best"],
                interpolation=exhaustive["interpolation"],
                extrapolation=exhaustive["extrapolation"],
            )
        )
        for method in (
            ("conditioned",)
            if smoke
            else ("conditioned", "no_observations", "uniform", "exhaustive")
        ):
            seeds = cfg["sampling_seeds"][:1] if smoke else cfg["sampling_seeds"]
            if method == "exhaustive":
                seeds = [None]
            for seed in seeds:
                effective_seed = seed + task_index * 1009 if seed is not None else None
                if method in ("conditioned", "no_observations"):
                    trace = generator.generate(
                        task["observations"] if method == "conditioned" else None,
                        2 if smoke else cfg["k"],
                        effective_seed,
                    )
                    expressions = trace.pop("expressions")
                    result = evaluate_candidates(task, expressions, cfg["penalty"])
                else:
                    begin = time.perf_counter()
                    expressions = (
                        cfg["inventory"]
                        if method == "exhaustive"
                        else np.random.default_rng(effective_seed)
                        .choice(cfg["inventory"], cfg["k"], replace=True)
                        .tolist()
                    )
                    result = (
                        exhaustive
                        if method == "exhaustive"
                        else evaluate_candidates(task, expressions, cfg["penalty"])
                    )
                    trace = dict(
                        seconds=exhaustive_seconds
                        if method == "exhaustive"
                        else time.perf_counter() - begin,
                        generation_calls=0,
                    )
                row = dict(
                    task=task["id"],
                    target=task["target"],
                    group=task["group"],
                    method=method,
                    seed=seed,
                    effective_seed=effective_seed,
                    **result,
                    trace=trace,
                    observation_regret=result["best"]["objective"] - ceiling
                    if result["best"]
                    else None,
                )
                rows.append(row)
                with (output / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                print(
                    f"{task['id']} {method} seed={seed}: {result['best']['expression'] if result['best'] else 'INVALID'} ({trace['seconds']:.2f}s)",
                    flush=True,
                )
    write(output / "coverage.json", coverage)
    summary = dict(
        schema=1,
        smoke=smoke,
        config_digest=digest(cfg),
        dataset_digest=digest(tasks),
        total_seconds=time.perf_counter() - started,
        parameter_count=generator.parameter_count,
        metrics=aggregate(rows),
        comparisons=comparisons(rows),
        interpretation="Development feasibility study; no weight edits, no independent final test, descriptive intervals only.",
    )
    write(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/pretrained-baseline-01/config.json")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    model = Path(args.model)
    manifest = json.loads((model / "download-manifest.json").read_text())
    if (
        manifest["model_id"] != cfg["model_id"]
        or manifest["revision"] != cfg["revision"]
    ):
        raise ValueError("model revision differs from protocol")
    for name, expected in manifest["sha256"].items():
        with (model / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError("model file hash mismatch: " + name)
    load_started = time.perf_counter()
    generator = BaselineGenerator(str(model), cfg)
    loading_seconds = time.perf_counter() - load_started
    output = Path(args.output)
    run(cfg, generator, output, args.smoke)
    write(output / "model-manifest.json", manifest)
    write(
        output / "environment.json",
        dict(
            python=platform.python_version(),
            platform=platform.platform(),
            packages={
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "numpy",
                    "safetensors",
                    "tokenizers",
                )
            },
            loading_seconds=loading_seconds,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            source_sha256={
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (
                    "src/symbolic_wsp/experiments/pretrained_baseline.py",
                    "src/symbolic_wsp/symbolic/grammar.py",
                    "src/symbolic_wsp/symbolic/evaluator.py",
                )
            },
            device="cpu",
            dtype=str(next(generator.model.parameters()).dtype),
            threads=generator.torch.get_num_threads(),
        ),
    )


if __name__ == "__main__":
    main()
