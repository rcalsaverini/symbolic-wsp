"""Render the exact-probability diagnostic suite without rerunning inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from symbolic_wsp.experiments.diagnostics import expected_selection, tasks
from symbolic_wsp.experiments.pretrained_baseline import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.results / "config.json").read_text())
    summary = json.loads((args.results / "summary.json").read_text())
    choices = json.loads((args.results / "selections.json").read_text())
    rows = [
        json.loads(line)
        for line in (args.results / "rows.jsonl").read_text().splitlines()
    ]
    expected_rows = len(cfg["models"]) * (
        len(cfg["development_targets"])
        * len(cfg["prompt_variants"])
        * len(cfg["development_controls"])
        + len(cfg["evaluation_targets"])
        * len(cfg["prompt_variants"])
        * len(cfg["observation_controls"])
    )
    if len(rows) != expected_rows:
        raise ValueError(f"incomplete matrix: {len(rows)} != {expected_rows}")
    lines = [
        "# Baseline diagnostic suite: executed results",
        "",
        "These are exact finite-catalog distributions on public development/evaluation tasks. No private final test or structural synthesis claim is involved. Expected best-of-eight selects using original observation scores; reserved errors never choose expressions.",
        "",
        "## Frozen development selections",
        "",
        "| Model | Selected prompt | Minimal regret | Catalog regret | Few-shot regret |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model, choice in choices.items():
        m = choice["development_objective"]
        lines.append(
            f"| {model} | {choice['variant']} | {m['minimal']:.6g} | {m['catalog']:.6g} | {m['few_shot']:.6g} |"
        )
    lines += [
        "",
        "## Evaluation: correct observations, representable functions",
        "",
        "| Model | Prompt | Target probability | Expected target hit at K=8 | Expected interpolation NMSE | Expected extrapolation NMSE | Observation regret |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in choices:
        for variant in cfg["prompt_variants"]:
            m = summary[f"{model}/evaluation/{variant}/correct/representable"]
            lines.append(
                f"| {model} | {variant} | {m['target_probability']:.6g} | {m['exact_target_hit_probability']:.2%} | {m['expected_interpolation_nmse']:.6g} | {m['expected_extrapolation_nmse']:.6g} | {m['expected_observation_regret']:.6g} |"
            )
    lines += [
        "",
        "## Observation controls under the development-selected prompt",
        "",
        "| Model | Control | Target probability | Expected interpolation NMSE | Expected extrapolation NMSE | Observation regret |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model, choice in choices.items():
        for control in cfg["observation_controls"]:
            m = summary[
                f"{model}/evaluation/{choice['variant']}/{control}/representable"
            ]
            lines.append(
                f"| {model} | {control} | {m['target_probability']:.6g} | {m['expected_interpolation_nmse']:.6g} | {m['expected_extrapolation_nmse']:.6g} | {m['expected_observation_regret']:.6g} |"
            )
    lines += [
        "",
        "## Per-function target probability",
        "",
        "| Target | 135M minimal | 135M catalog | 135M few-shot | 360M minimal | 360M catalog | 360M few-shot |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in cfg["evaluation_targets"]:
        if target not in cfg["inventory"]:
            continue
        values = []
        for model in choices:
            for variant in cfg["prompt_variants"]:
                r = next(
                    r
                    for r in rows
                    if r["model"] == model
                    and r["phase"] == "evaluation"
                    and r["variant"] == variant
                    and r["control"] == "correct"
                    and r["target"] == target
                )
                values.append(f"{r['metrics']['target_probability']:.6g}")
        lines.append("| `" + target + "` | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Analytical nonneural references",
        "",
        "Uniform uses K=8 draws with replacement. Exhaustive selection evaluates all 14 entries. Both use the same correct observations and complexity penalty. These are derived without neural inference.",
        "",
        "| Group | Reference | Expected interpolation NMSE | Expected extrapolation NMSE | Expected observation regret |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    references = []
    from symbolic_wsp.symbolic.evaluator import score

    for task in tasks(cfg["evaluation_targets"], cfg["evaluation_seed"], cfg["points"]):
        best = min(
            cfg["inventory"],
            key=lambda e: (
                score(e, **task["observations"], penalty=cfg["penalty"])["objective"],
                e,
            ),
        )
        for method in ("uniform", "exhaustive"):
            probs = {
                e: 1 / len(cfg["inventory"])
                if method == "uniform"
                else float(e == best)
                for e in cfg["inventory"]
            }
            references.append(
                dict(
                    task=task["id"],
                    target=task["target"],
                    method=method,
                    group="representable"
                    if task["target"] in cfg["inventory"]
                    else "outside_catalog",
                    metrics=expected_selection(
                        probs, task, cfg["inventory"], cfg["k"], cfg["penalty"]
                    ),
                )
            )
    for group in ("representable", "outside_catalog"):
        for method in ("uniform", "exhaustive"):
            rs = [
                r["metrics"]
                for r in references
                if r["group"] == group and r["method"] == method
            ]
            means = {
                k: float(np.mean([r[k] for r in rs]))
                for k in rs[0]
                if k != "selection_probabilities"
            }
            lines.append(
                f"| {group} | {method} | {means['expected_interpolation_nmse']:.6g} | {means['expected_extrapolation_nmse']:.6g} | {means['expected_observation_regret']:.6g} |"
            )
    write(args.results / "analytical-references.json", references)
    lines += [
        "",
        "## Outside-catalog tasks, correct observations",
        "",
        "| Model | Prompt | Expected interpolation NMSE | Expected extrapolation NMSE | Observation regret |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model, choice in choices.items():
        m = summary[f"{model}/evaluation/{choice['variant']}/correct/outside_catalog"]
        lines.append(
            f"| {model} | {choice['variant']} | {m['expected_interpolation_nmse']:.6g} | {m['expected_extrapolation_nmse']:.6g} | {m['expected_observation_regret']:.6g} |"
        )
    inter = json.loads((args.results / "intervention/summary.json").read_text())
    lines += ["", "## Targeted intervention", "", f"Status: `{inter['status']}`."]
    if inter["status"] == "completed":
        d = inter["decision"]
        lines += [
            "",
            f"Development-selected candidate: `{d['chosen']}`. Development screen passed: **{d['passed_development_screen']}**.",
            f"Development baseline regret {d['development_baseline_regret']:.6g}; candidate regret {d['development_candidate_regret']:.6g}.",
            "",
            "A failed screen remains a rejected diagnostic candidate; evaluation does not promote it.",
            "",
            "| Condition | Expected interpolation NMSE | Expected extrapolation NMSE | Target probability | Observation regret |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for condition in (
            "baseline",
            "targeted",
            "no_edit",
            "root_logit_bias",
            *[f"random_{seed}" for seed in cfg["intervention"]["random_seeds"]],
        ):
            m = inter["metrics"][condition + "/representable"]
            lines.append(
                f"| {condition} | {m['expected_interpolation_nmse']:.6g} | {m['expected_extrapolation_nmse']:.6g} | {m['target_probability']:.6g} | {m['expected_observation_regret']:.6g} |"
            )
        rollback = json.loads((args.results / "intervention/rollback.json").read_text())
        lines += [
            "",
            f"Original tensor restored and hash-checked: **{rollback['restored']}**.",
        ]
    else:
        lines += ["", inter["reason"]]
    sensitivity = []
    for model, choice in choices.items():
        for target in cfg["evaluation_targets"]:
            rs = [
                r
                for r in rows
                if r["model"] == model
                and r["phase"] == "evaluation"
                and r["variant"] == choice["variant"]
                and r["target"] == target
            ]
            correct = next(r for r in rs if r["control"] == "correct")
            for control in ("shuffled", "other_function", "absent"):
                other = next(r for r in rs if r["control"] == control)
                tv = 0.5 * sum(
                    abs(
                        correct["distribution"]["probabilities"][e]
                        - other["distribution"]["probabilities"][e]
                    )
                    for e in cfg["inventory"]
                )
                sensitivity.append(
                    dict(
                        model=model,
                        target=target,
                        control=control,
                        total_variation=tv,
                        target_probability_delta=correct["metrics"][
                            "target_probability"
                        ]
                        - other["metrics"]["target_probability"],
                        interpolation_nmse_delta=correct["metrics"][
                            "expected_interpolation_nmse"
                        ]
                        - other["metrics"]["expected_interpolation_nmse"],
                    )
                )
    write(args.results / "observation-sensitivity.json", sensitivity)
    lines += [
        "",
        "## Accounting and limits",
        "",
        f"Completed matrix rows: {len(rows)}. Probability mass is checked at every catalog distribution. Sum of recorded noncached matrix inference times: {sum(r['distribution']['seconds'] for r in rows):.1f} s.",
        "Repeated identical prompts reuse exact distributions and are marked in the traces. Intervention work is separate; see its per-row timings. Wall timings are shared-CPU observations, not controlled latency benchmarks.",
        "Each group contains only five functions and one numerical realization per function. Exact probabilities remove Monte Carlo sampling uncertainty, not task/population uncertainty. No confirmatory significance claim or scaling law is established.",
        "All three prompt variants and all observation controls are reported; prompt choice and edit magnitude use development data only. Evaluation labels and reserved points never enter the generator. Historical baseline functions are reused with new points, so this is not an independent final test.",
        "Source/model hashes, effective prompts, prefix log probabilities, selection probabilities and per-function errors are retained in the raw artifacts. The README discusses scientific interpretation and implementation details.",
        "",
    ]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
