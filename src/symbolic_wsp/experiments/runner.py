"""Matched-budget paired runs. Held-out values never enter generation feedback."""

import json
import time

import numpy as np

from ..symbolic.evaluator import score


def run(
    backend,
    data,
    k=8,
    seeds=(1, 2, 3),
    condition="baseline",
    penalty=0.001,
    final=False,
):
    if k < 1 or not seeds:
        raise ValueError("positive budget and repeats required")
    if data["split"] == "final" and not final:
        raise ValueError("final data requires independent evaluator")
    if condition not in ("baseline", "feedback", "wsp", "random", "no_edit"):
        raise ValueError("unknown condition")
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        for task in data["tasks"]:
            entries = []
            if condition == "feedback":
                for i in range(k):
                    expr = backend.generate(
                        task["observations"], 1, seed + i, feedback=entries
                    )[0]
                    entries.append(
                        dict(
                            expression=expr,
                            **score(expr, **task["observations"], penalty=penalty),
                        )
                    )
            else:
                entries = [
                    dict(
                        expression=e,
                        **score(e, **task["observations"], penalty=penalty),
                    )
                    for e in backend.generate(task["observations"], k, seed)
                ]
            valid = [e for e in entries if e["valid"]]
            best = min(valid, key=lambda e: e["objective"]) if valid else None
            # Selection uses observations only; numerical held-out scores are measured afterwards.
            held = (
                score(best["expression"], **task["heldout"], penalty=penalty)
                if best
                else None
            )
            rows.append(
                dict(
                    task=task["id"],
                    seed=seed,
                    candidates=entries,
                    best=best,
                    heldout=held,
                    validity=len(valid) / k,
                    diversity=len({e["structure"] for e in valid}) / k,
                )
            )
    return dict(
        schema=1,
        generation_config={
            "expressions": list(getattr(backend, "expressions", ())),
            "prompt_version": 1,
        },
        checkpoint=backend.identity,
        condition=condition,
        split=data["split"],
        dataset_digest=__import__("hashlib")
        .sha256(json.dumps(data, sort_keys=True).encode())
        .hexdigest(),
        k=k,
        seeds=list(seeds),
        penalty=penalty,
        model_evaluations=k * len(rows),
        elapsed_seconds=time.perf_counter() - started,
        agent_tokens=None,
        edit_discovery_cost=None,
        rows=rows,
    )


def paired(a, b, seed=1, bootstrap=2000):
    for key in ("dataset_digest", "k", "seeds", "penalty", "generation_config"):
        if a[key] != b[key]:
            raise ValueError("unmatched evaluation: " + key)

    def values(result):
        return {(r["task"], r["seed"]): r for r in result["rows"]}

    aa, bb = values(a), values(b)
    if aa.keys() != bb.keys():
        raise ValueError("unpaired rows")
    by_task = {}
    failures = 0
    for key, ar in aa.items():
        br = bb[key]
        if (
            ar["heldout"]
            and br["heldout"]
            and ar["heldout"]["valid"]
            and br["heldout"]["valid"]
        ):
            by_task.setdefault(key[0], []).append(
                br["heldout"]["nmse"] - ar["heldout"]["nmse"]
            )
        else:
            failures += 1
    # Cluster repeats by function; do not treat repeated runs as independent tasks.
    deltas = np.array([np.mean(v) for v in by_task.values()])
    if not len(deltas):
        return dict(
            mean_delta=None,
            ci95=None,
            paired_tasks=0,
            failed_pairs=failures,
            retained=False,
        )
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(deltas, (bootstrap, len(deltas)), replace=True), axis=1)
    ci = np.quantile(means, [0.025, 0.975]).tolist()
    return dict(
        mean_delta=float(deltas.mean()),
        ci95=ci,
        paired_tasks=len(deltas),
        failed_pairs=failures,
        retained=bool(len(deltas) >= 5 and failures == 0 and ci[1] < 0),
        criterion="at least 5 tasks; no failed pairs; upper clustered bootstrap CI < 0",
    )


def record_decision(store, child, before, after, report):
    if before["split"] != "validation" or after["split"] != "validation":
        raise ValueError("decisions require validation")
    with (store.root / "history.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    action="decision",
                    child=child,
                    before=before["checkpoint"],
                    after=after["checkpoint"],
                    comparison=report,
                    status="retained" if report["retained"] else "rejected",
                )
            )
            + "\n"
        )
