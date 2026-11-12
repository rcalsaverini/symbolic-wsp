"""First-class baseline diagnosis suite with frozen development/evaluation phases."""

import copy
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from ..inference.catalog import CatalogOracle
from ..symbolic.evaluator import evaluate, score
from .pretrained_baseline import digest, prompt_messages, validate_prompt, write


def tasks(targets, seed, points):
    result = []
    for i, target in enumerate(targets):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        row = dict(id=f"{seed}-{i}", target=target)
        for regime, lo, hi in [
            ("observations", -1, 1),
            ("interpolation", -1, 1),
            ("extrapolation", 1, 2),
        ]:
            x = rng.uniform(lo, hi, points)
            row[regime] = dict(x=x.tolist(), y=evaluate(target, x).tolist())
        result.append(row)
    return result


def observation_control(task, all_tasks, control, seed):
    obs = copy.deepcopy(task["observations"])
    if control == "correct":
        return obs
    if control == "absent":
        return None
    if control == "shuffled":
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(task["id"].split("-")[-1])])
        )
        obs["y"] = rng.permutation(obs["y"]).tolist()
    elif control == "other_function":
        index = next(i for i, t in enumerate(all_tasks) if t["id"] == task["id"])
        obs["y"] = evaluate(
            all_tasks[(index + 1) % len(all_tasks)]["target"], obs["x"]
        ).tolist()
    else:
        raise ValueError("unknown observation control")
    return obs


def prompt_variants(cfg):
    base = copy.deepcopy(cfg["prompt"])
    validate_prompt(base)
    listing = (
        "Allowed expressions: "
        + json.dumps(cfg["inventory"])
        + "\nChoose the expression that best fits the paired x/y observations. Return only that expression.\n{observations}"
    )
    catalog = dict(base, user_template=listing)
    demos = tasks(
        cfg["demonstration_targets"],
        cfg["demonstration_seed"],
        cfg["demonstration_points"],
    )
    examples = []
    for task in demos:
        payload = prompt_messages(task["observations"], base)[1]["content"].split(
            "\n", 1
        )[1]
        examples.append(
            "Example observations: " + payload + "\nExpression: " + task["target"]
        )
    few = dict(base, user_template="\n\n".join(examples) + "\n\n" + listing)
    return dict(minimal=base, catalog=catalog, few_shot=few), demos


def expected_selection(probabilities, task, inventory, k, penalty):
    """Distribution of the observation-selected best of K independent proposals."""
    if k < 1 or not np.isclose(sum(probabilities.values()), 1.0, atol=1e-7):
        raise ValueError("invalid K or probabilities")
    if set(probabilities) != set(inventory) or any(
        p < 0 for p in probabilities.values()
    ):
        raise ValueError("invalid catalog probabilities")
    observations = {
        e: score(e, **task["observations"], penalty=penalty) for e in inventory
    }
    if not all(s["valid"] for s in observations.values()):
        raise ValueError("inventory invalid on task")
    ordered = sorted(inventory, key=lambda e: (observations[e]["objective"], e))
    tail = 1.0
    chosen = {}
    for e in ordered:
        next_tail = max(0.0, tail - probabilities[e])
        chosen[e] = max(0.0, tail**k - next_tail**k)
        tail = next_tail
    if not np.isclose(sum(chosen.values()), 1.0, atol=1e-7):
        raise ValueError("selected probability mass")
    result = dict(
        selection_probabilities=chosen,
        target_probability=probabilities.get(task["target"], 0.0),
        exact_target_hit_probability=1
        - (1 - probabilities.get(task["target"], 0.0)) ** k,
        expected_observation_objective=sum(
            chosen[e] * observations[e]["objective"] for e in inventory
        ),
        observation_floor=observations[ordered[0]]["objective"],
        expected_complexity=sum(
            chosen[e] * observations[e]["complexity"] for e in inventory
        ),
    )
    result["expected_observation_regret"] = (
        result["expected_observation_objective"] - result["observation_floor"]
    )
    for regime in ("interpolation", "extrapolation"):
        evaluated = {e: score(e, **task[regime], penalty=penalty) for e in inventory}
        for metric in ("mse", "nmse"):
            result["expected_" + regime + "_" + metric] = sum(
                chosen[e] * evaluated[e][metric] for e in inventory
            )
    return result


def render(tokenizer, obs, spec):
    return tokenizer.apply_chat_template(
        prompt_messages(obs, spec),
        tokenize=False,
        chat_template=spec["chat_template"],
        add_generation_prompt=spec["add_generation_prompt"],
    )


def validate_model(path, identity):
    manifest = json.loads((path / "download-manifest.json").read_text())
    if any(manifest[k] != identity[k] for k in ("model_id", "revision")):
        raise ValueError("wrong model revision")
    for name, sha in manifest["sha256"].items():
        with (path / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != sha:
                raise ValueError("model hash mismatch")
    return manifest


def load_model(path, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        .float()
        .eval()
    )
    return model, tokenizer


def summarize(rows):
    summary = {}
    for key in sorted(
        {(r["model"], r["phase"], r["variant"], r["control"]) for r in rows}
    ):
        subset = [
            r
            for r in rows
            if (r["model"], r["phase"], r["variant"], r["control"]) == key
        ]
        for group in ("representable", "outside_catalog", "all"):
            rs = [r for r in subset if group == "all" or r["group"] == group]
            if not rs:
                continue
            metrics = {
                metric: float(np.mean([r["metrics"][metric] for r in rs]))
                for metric in rs[0]["metrics"]
                if metric != "selection_probabilities"
            }
            summary["/".join((*key, group))] = dict(
                tasks=len(rs),
                seconds=sum(r["distribution"]["seconds"] for r in rs),
                **metrics,
            )
    return summary


def collect_phase(oracle, tokenizer, cfg, phase, variants, output, model_key, existing):
    targets = (
        cfg["development_targets"]
        if phase == "development"
        else cfg["evaluation_targets"]
    )
    seed = cfg["development_seed"] if phase == "development" else cfg["evaluation_seed"]
    examples = tasks(targets, seed, cfg["points"])
    write(output / (phase + "-tasks.json"), examples)
    results = []
    for task in examples:
        for variant in cfg["prompt_variants"]:
            for control in (
                cfg["development_controls"]
                if phase == "development"
                else cfg["observation_controls"]
            ):
                key = (model_key, phase, task["id"], variant, control)
                if key in existing:
                    results.append(existing[key])
                    continue
                obs = observation_control(task, examples, control, cfg["control_seed"])
                distribution = oracle.distribution(
                    render(tokenizer, obs, variants[variant])
                )
                row = dict(
                    model=model_key,
                    phase=phase,
                    task=task["id"],
                    target=task["target"],
                    group="representable"
                    if task["target"] in cfg["inventory"]
                    else "outside_catalog",
                    variant=variant,
                    control=control,
                    prompt_config_digest=digest(variants[variant]),
                    distribution=distribution,
                    metrics=expected_selection(
                        distribution["probabilities"],
                        task,
                        cfg["inventory"],
                        cfg["k"],
                        cfg["penalty"],
                    ),
                )
                with (output / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                results.append(row)
                existing[key] = row
                print(
                    f"{model_key} {phase} {task['id']} {variant}/{control} p(target)={row['metrics']['target_probability']:.5g} ({distribution['seconds']:.2f}s)",
                    file=sys.stderr,
                    flush=True,
                )
    return results


def choose_prompt(rows, variants):
    correct = [r for r in rows if r["control"] == "correct"]
    means = {
        v: float(
            np.mean(
                [
                    r["metrics"]["expected_observation_regret"]
                    for r in correct
                    if r["variant"] == v
                ]
            )
        )
        for v in variants
    }
    return dict(
        variant=min(means, key=lambda v: (means[v], v)),
        development_objective=means,
        criterion="minimum development expected observation-objective regret; lexical tie-break",
    )


def source_fingerprint():
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("src/symbolic_wsp").rglob("*.py")
    }


def save_phase_environment(output, cfg, phase, seconds):
    import subprocess

    sources = source_fingerprint()
    if json.loads((output / (phase + "-source.json")).read_text()) != sources:
        raise ValueError(
            "source changed during execution; raw records retained but phase not finalized"
        )
    archive = output / "source-snapshots"
    archive.mkdir(exist_ok=True)
    for path, sha in sources.items():
        target = archive / (sha + ".py")
        if not target.exists():
            target.write_bytes(Path(path).read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
        dirty = None
    environment = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        packages={
            n: importlib.metadata.version(n)
            for n in ("torch", "transformers", "numpy", "safetensors", "tokenizers")
        },
        threads=cfg["threads"],
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        invocation_seconds=seconds,
        source_sha256=sources,
        git_commit=commit,
        worktree_dirty=dirty,
    )
    write(output / "environment.json", environment)
    write(output / ("environment-" + phase + ".json"), environment)
    path = output / "provenance.json"
    provenance = json.loads(path.read_text()) if path.exists() else {}
    provenance[phase] = commit
    write(path, provenance)


def run_study(cfg, model_paths, output, phase="all", resume=False):
    if phase == "all":
        result = None
        for index, stage in enumerate(("development", "intervention", "evaluation")):
            result = run_study(cfg, model_paths, output, stage, resume or index > 0)
        return result
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError("use a fresh output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and json.loads(
        (output / "config.json").read_text()
    ) != cfg:
        raise ValueError("resume configuration mismatch")
    phase_source = output / (phase + "-source.json")
    sources = source_fingerprint()
    if phase_source.exists() and json.loads(phase_source.read_text()) != sources:
        raise ValueError("source changed since phase started; use a fresh study output")
    write(phase_source, sources)
    write(output / "config.json", cfg)
    variants, demos = prompt_variants(cfg)
    write(output / "prompts.json", variants)
    write(output / "demonstrations.json", demos)
    if set(cfg["development_targets"]) & set(cfg["evaluation_targets"]) or set(
        cfg["demonstration_targets"]
    ) & (set(cfg["development_targets"]) | set(cfg["evaluation_targets"])):
        raise ValueError("task expression leakage")
    rows = (
        [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
        if (output / "rows.jsonl").exists()
        else []
    )
    existing = {
        (r["model"], r["phase"], r["task"], r["variant"], r["control"]): r for r in rows
    }
    selections = (
        json.loads((output / "selections.json").read_text())
        if (output / "selections.json").exists()
        else {}
    )
    for identity in cfg["models"]:
        key = identity["key"]
        path = Path(model_paths[key])
        if phase == "intervention" and key != cfg["intervention"]["model"]:
            continue
        manifest = validate_model(path, identity)
        write(output / (key + "-model-manifest.json"), manifest)
        model, tokenizer = load_model(path, cfg)
        oracle = CatalogOracle(model, tokenizer, cfg["inventory"])
        for selected_phase in ("development", "evaluation"):
            if phase not in ("all", selected_phase):
                continue
            if selected_phase == "evaluation" and key not in selections:
                raise ValueError(
                    "freeze development prompt selection before evaluation"
                )
            rs = collect_phase(
                oracle, tokenizer, cfg, selected_phase, variants, output, key, existing
            )
            if selected_phase == "development":
                selection = choose_prompt(rs, variants)
                if key in selections and selections[key] != selection:
                    raise ValueError("frozen selection mismatch")
                selections[key] = selection
                write(output / "selections.json", selections)
        if phase in ("all", "intervention") and key == cfg["intervention"]["model"]:
            from .intervention import run_intervention

            if key not in selections:
                raise ValueError("development required before intervention")
            dev = [
                r
                for r in existing.values()
                if r["model"] == key and r["phase"] == "development"
            ]
            run_intervention(
                oracle, tokenizer, model, cfg, variants, selections[key], dev, output
            )
        del oracle, model, tokenizer
        gc.collect()
    rows = list(existing.values())
    write(output / "summary.json", summarize(rows))
    save_phase_environment(output, cfg, phase, time.perf_counter() - started)
    return dict(
        output=str(output),
        rows=len(rows),
        config_digest=digest(cfg),
        selections=selections,
    )
