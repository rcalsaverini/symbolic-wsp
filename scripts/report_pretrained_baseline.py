"""Render the committed feasibility-run records into a compact Markdown report."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    env = json.loads((args.results / "environment.json").read_text())
    cfg = json.loads((args.results / "config.json").read_text())
    coverage = json.loads((args.results / "coverage.json").read_text())
    rows = [
        json.loads(line)
        for line in (args.results / "rows.jsonl").read_text().splitlines()
    ]
    lines = [
        "# Pretrained baseline: executed results",
        "",
        f"Model: `{cfg['model_id']}` at revision `{cfg['revision']}`.",
        f"Parameters: {summary['parameter_count']:,}. CPU float32, {env['threads']} threads.",
        f"Trial-loop wall time: {summary['total_seconds']:.1f} s; model loading: {env['loading_seconds']:.2f} s; peak process RSS: {env['peak_rss_mib']:.1f} MiB.",
        "",
        "No weight interventions were performed. These are public development tasks, not an independent final test.",
        "",
        "See [exact prompts and task configuration](PROMPTS.md) for the instruction, system message, both rendered conditions, and all task prompt files.",
        "",
        "## Metrics by group",
        "",
        "Each sampling method has three repeats per function. Values below average repeats and functions equally. Lower errors and regret are better.",
        "",
        "| Group | Method | Interpolation NMSE | Extrapolation NMSE | Observation-objective regret | Duplicate rate | Exact target selection |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ("representable", "outside_catalog", "all"):
        for method in ("conditioned", "no_observations", "uniform", "exhaustive"):
            m = summary["metrics"][method][group]
            lines.append(
                f"| {group} | {method} | {m['mean_interpolation_nmse']:.6g} | {m['mean_extrapolation_nmse']:.6g} | {m['mean_observation_regret']:.6g} | {m['mean_duplicate_rate']:.1%} | {m['selected_exact_target_rate']:.1%} |"
            )
    lines += [
        "",
        "Exact target selection is literal string recovery; it is not symbolic equivalence and is only informative for the representable group.",
        "",
        "## Error scale and validity",
        "",
        "| Method | Interpolation MSE | Extrapolation MSE | Numerical validity | Failed selections |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ("conditioned", "no_observations", "uniform", "exhaustive"):
        m = summary["metrics"][method]["all"]
        lines.append(
            f"| {method} | {m['mean_interpolation_mse']:.6g} | {m['mean_extrapolation_mse']:.6g} | {m['mean_numerical_validity']:.1%} | {m['failed_selections']} |"
        )
    lines += [
        "",
        "## Per-function errors",
        "",
        "Cells show mean interpolation / extrapolation NMSE. Sampling methods are averaged over three seeds.",
        "",
        "| Target | Conditioned | No observations | Uniform | Exhaustive |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task in coverage:
        cells = []
        for method in ("conditioned", "no_observations", "uniform", "exhaustive"):
            rs = [
                r for r in rows if r["task"] == task["task"] and r["method"] == method
            ]
            cells.append(
                " / ".join(
                    f"{np.mean([r[regime]['nmse'] for r in rs]):.5g}"
                    for regime in ("interpolation", "extrapolation")
                )
            )
        lines.append("| `" + task["target"] + "` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Coverage diagnostics",
        "",
        "| Target | In inventory | Minimum observation NMSE | Observation-selected catalog entry |",
        "| --- | --- | ---: | --- |",
    ]
    for task in coverage:
        lines.append(
            f"| `{task['target']}` | {task['target_in_inventory']} | {task['min_observation_nmse']:.6g} | `{task['observation_selector']['expression']}` |"
        )
    lines += [
        "",
        "The observation selector includes the complexity penalty. Neither the selector nor the coverage diagnostic uses reserved errors to choose an expression.",
        "",
        "## Proposal frequencies",
        "",
        "| Expression | Conditioned | No observations | Uniform |",
        "| --- | ---: | ---: | ---: |",
    ]
    for expression in cfg["inventory"]:
        counts = [
            summary["metrics"][method]["all"]["candidate_frequencies"].get(
                expression, 0
            )
            for method in ("conditioned", "no_observations", "uniform")
        ]
        lines.append("| `" + expression + "` | " + " | ".join(map(str, counts)) + " |")
    lines += [
        "",
        "## Cost accounting",
        "",
        "| Method | Candidate evaluations | All evaluator calls | Generation calls | Generated tokens | Method seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("conditioned", "no_observations", "uniform", "exhaustive"):
        m = summary["metrics"][method]["all"]
        candidate_count = sum(
            len(r["candidates"]) for r in rows if r["method"] == method
        )
        lines.append(
            f"| {method} | {candidate_count} | {m['evaluator_calls']} | {m['generation_calls']} | {m['generated_tokens']} | {m['generation_seconds']:.4f} |"
        )
    lines += [
        "",
        "Model timings include generation; nonneural timings include sampling/selection and numerical evaluation. They are not equal-compute conditions. Exhaustive search uses a different candidate budget. See the protocol for EOS and prefill accounting.",
        "",
        "## Paired descriptive comparisons",
        "",
        "Differences are conditioned minus comparator: negative favors conditioning. Intervals resample ten functions after averaging repeats; they are descriptive and do not establish population effects.",
        "",
        "| Comparator | Regime | Mean NMSE difference | Descriptive 95% interval |",
        "| --- | --- | ---: | --- |",
    ]
    for comparator in ("no_observations", "uniform", "exhaustive"):
        for regime in ("interpolation", "extrapolation"):
            c = summary["comparisons"][f"conditioned_minus_{comparator}/all/{regime}"]
            lo, hi = c["descriptive_ci95"]
            lines.append(
                f"| {comparator} | {regime} | {c['mean_delta']:.6g} | [{lo:.6g}, {hi:.6g}] |"
            )
    lines += [
        "",
        "## Reproducibility and scope",
        "",
        f"Configuration digest: `{summary['config_digest']}`.",
        f"Dataset digest: `{summary['dataset_digest']}`.",
        "",
        "Exact versions, source hashes, model-file hashes, rendered prompts, token IDs, per-candidate scores and task points are retained in the result directory.",
        "This report is generated from those records. Scientific interpretation and smoke-test history are in the experiment README. No private test, weight editing, or claim of new symbolic-search capability is involved.",
        "",
    ]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
