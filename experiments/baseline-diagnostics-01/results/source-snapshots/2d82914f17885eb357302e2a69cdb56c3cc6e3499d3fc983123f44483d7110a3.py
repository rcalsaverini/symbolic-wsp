"""Predeclared, development-only rank-one proposal and controlled evaluation."""

import hashlib
import json

import numpy as np

from ..wsp.checkpoints import Store
from ..wsp.editing import edit, rollback
from .diagnostics import expected_selection, render, tasks
from .pretrained_baseline import digest, write


def run_intervention(
    oracle, tokenizer, model, cfg, variants, selection, development, output
):
    import torch

    dest = output / "intervention"
    dest.mkdir(exist_ok=True)
    if (dest / "summary.json").exists():
        return json.loads((dest / "summary.json").read_text())
    rule = cfg["intervention"]
    selected = selection["variant"]
    spec = variants[selected]
    failures = []
    for target in cfg["development_targets"]:
        rs = [
            r
            for r in development
            if r["target"] == target and r["control"] == "correct"
        ]
        if len(rs) == len(cfg["prompt_variants"]) and all(
            r["metrics"]["target_probability"]
            < rule["repeated_failure_probability_threshold"]
            for r in rs
        ):
            failures.append(target)
    gate = dict(
        failing_targets=failures,
        required=rule["minimum_failing_development_tasks"],
        rule="target probability below threshold in every prompt variant on development only",
        selection=selection,
    )
    write(dest / "gate.json", gate)
    if len(failures) < rule["minimum_failing_development_tasks"]:
        result = dict(
            status="not_attempted",
            reason="predeclared repeated-failure gate not met",
            gate=gate,
        )
        write(dest / "summary.json", result)
        return result
    module_name, module = [
        (n, m) for n, m in model.named_modules() if n.endswith(rule["module_suffix"])
    ][-1]
    store = Store(dest / "tensor-checkpoints")
    original = module.weight.detach().cpu().numpy().copy()
    parent = store.put(
        {module_name: original},
        dict(scope="single tensor, not complete model", module=module_name),
    )
    baseline_sha = hashlib.sha256(original.tobytes()).hexdigest()
    development_tasks = tasks(
        cfg["development_targets"], cfg["development_seed"], cfg["points"]
    )
    records = []
    for task in development_tasks:
        capture = []
        hook = module.register_forward_pre_hook(
            lambda m, args: capture.append(args[0][0, -1].detach().cpu().numpy().copy())
        )
        try:
            ids = tokenizer.encode(
                render(tokenizer, task["observations"], spec),
                add_special_tokens=False,
                return_tensors="pt",
            )
            with torch.inference_mode():
                model(ids, use_cache=False)
        finally:
            hook.remove()
        records.append(capture[0])
    v = np.mean(records, axis=0)
    v /= np.linalg.norm(v)
    target_ids = sorted(
        {tokenizer.encode(e, add_special_tokens=False)[0] for e in failures}
    )
    embeddings = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    ref = [
        r for r in development if r["control"] == "correct" and r["variant"] == selected
    ]
    mean_root = {}
    for row in ref:
        for entry in row["distribution"]["root_allowed"]:
            mean_root[entry["token_id"]] = mean_root.get(
                entry["token_id"], 0.0
            ) + entry["probability"] / len(ref)
    u = embeddings[target_ids].mean(axis=0) - sum(
        prob * embeddings[token] for token, prob in mean_root.items()
    )
    u /= np.linalg.norm(u)
    proposal = dict(
        baseline_model_manifest_digest=digest(
            json.loads(
                (
                    output / (cfg["intervention"]["model"] + "-model-manifest.json")
                ).read_text()
            )
        ),
        hypothesis=rule["hypothesis"],
        module=module_name,
        parent_tensor=parent,
        original_tensor_sha256=baseline_sha,
        desired_first_tokens=[
            dict(id=t, text=tokenizer.decode([t])) for t in target_ids
        ],
        derived_from="development prompts and probabilities only",
        prompt_variant=selected,
        scope="baseline model manifest plus named tensor override defines edited state; tensor store is not a full model checkpoint",
    )
    write(dest / "proposal.json", proposal)
    candidates = []
    for relative in rule["relative_norms"]:
        event = edit(
            store,
            parent,
            dict(
                mode="rank_one",
                tensor=module_name,
                alpha=relative * float(np.linalg.norm(original)),
                u=u.tolist(),
                v=v.tolist(),
            ),
            rule["hypothesis"],
        )
        candidates.append(
            dict(
                name=f"rank_one_{relative}",
                identity=event["child"],
                event=event,
                relative=relative,
            )
        )

    def apply(identity):
        oracle._cache.clear()
        with torch.inference_mode():
            module.weight.copy_(torch.from_numpy(store.get(identity)[module_name]))

    def evaluate_set(examples, label, identity=None, bias=None):
        apply(identity or parent)
        rs = []
        for task in examples:
            distribution = oracle.distribution(
                render(tokenizer, task["observations"], spec), root_bias=bias
            )
            row = dict(
                condition=label,
                task=task["id"],
                target=task["target"],
                tensor_checkpoint=identity or parent,
                distribution=distribution,
                metrics=expected_selection(
                    distribution["probabilities"],
                    task,
                    cfg["inventory"],
                    cfg["k"],
                    cfg["penalty"],
                ),
            )
            with (dest / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            rs.append(row)
        return rs

    try:
        baseline = evaluate_set(development_tasks, "development/baseline")
        base_regret = float(
            np.mean([r["metrics"]["expected_observation_regret"] for r in baseline])
        )
        for candidate in candidates:
            rs = evaluate_set(
                development_tasks,
                "development/" + candidate["name"],
                candidate["identity"],
            )
            candidate["development_regret"] = float(
                np.mean([r["metrics"]["expected_observation_regret"] for r in rs])
            )
        chosen = min(candidates, key=lambda c: (c["development_regret"], c["name"]))
        accepted = (
            chosen["development_regret"]
            <= base_regret - rule["minimum_development_improvement"]
        )
        decision = dict(
            chosen=chosen["name"],
            candidate_tensor_checkpoint=chosen["identity"],
            development_baseline_regret=base_regret,
            development_candidate_regret=chosen["development_regret"],
            passed_development_screen=accepted,
            criterion="development expected observation-regret improvement >= configured minimum; no evaluation-based reselection",
        )
        # Freeze proposal and selection before creating or reading evaluation tasks.
        write(dest / "decision.json", decision)
        write(dest / "candidates.json", candidates)
        examples = tasks(
            cfg["evaluation_targets"], cfg["evaluation_seed"], cfg["points"]
        )
        conditions = [
            ("baseline", None, None),
            ("targeted", chosen["identity"], None),
            ("no_edit", None, None),
            (
                "root_logit_bias",
                None,
                {t: rule["token_bias_strength"] for t in target_ids},
            ),
        ]
        magnitude = chosen["event"]["delta"][module_name]["norm"]
        for seed in rule["random_seeds"]:
            event = edit(
                store,
                parent,
                dict(mode="random", tensor=module_name, norm=magnitude, seed=seed),
                "Norm-matched random control; no tuning",
            )
            conditions.append((f"random_{seed}", event["child"], None))
        evaluation = []
        for name, identity, bias in conditions:
            evaluation.extend(
                evaluate_set(examples, "evaluation/" + name, identity, bias)
            )
        metrics = {}
        for name, _, _ in conditions:
            rs = [r for r in evaluation if r["condition"] == "evaluation/" + name]
            for group in ("representable", "outside_catalog", "all"):
                subset = [
                    r
                    for r in rs
                    if group == "all"
                    or (r["target"] in cfg["inventory"]) == (group == "representable")
                ]
                metrics[name + "/" + group] = {
                    k: float(np.mean([r["metrics"][k] for r in subset]))
                    for k in subset[0]["metrics"]
                    if k != "selection_probabilities"
                }
        result = dict(
            status="completed",
            gate=gate,
            decision=decision,
            metrics=metrics,
            interpretation="Predeclared global first-token-access intervention; does not establish acquired search capability. Rejected proposals are evaluated as diagnostics, never retained on evaluation results.",
        )
        write(dest / "summary.json", result)
    finally:
        apply(parent)
        assert (
            hashlib.sha256(module.weight.detach().cpu().numpy().tobytes()).hexdigest()
            == baseline_sha
        )
        rollback(
            store,
            chosen["identity"] if "chosen" in locals() else candidates[0]["identity"],
        )
        write(
            dest / "rollback.json",
            dict(restored=True, tensor_sha256=baseline_sha, parent=parent),
        )
    return result
