"""Gradient-free interventions; formulas follow upstream wsp_engine.py.
Adaptation: immutable versions, float-preserving operations, strict validation.
Apache-2.0; see NOTICE.
"""

import json
import time

import numpy as np

from .checkpoints import compare
from .upstream_dsl import apply_edits_dsl


def edit(store, parent, spec, hypothesis):
    if not hypothesis.strip():
        raise ValueError("hypothesis required")
    base = store.get(parent)
    out = {k: a.copy() for k, a in base.items()}
    mode = spec["mode"]
    if mode == "dsl":
        report = apply_edits_dsl(out, spec["dsl"])
        if report["errors"]:
            raise ValueError(str(report["errors"]))
    else:
        w = out[spec["tensor"]]
        if mode == "rank_one":
            u, v = np.asarray(spec["u"]), np.asarray(spec["v"])
            if w.ndim != 2 or u.shape != (w.shape[0],) or v.shape != (w.shape[1],):
                raise ValueError("rank-one dimensions")
            w += (float(spec.get("alpha", 1)) * np.outer(u, v)).astype(w.dtype)
        elif mode == "project_out":
            u = np.asarray(spec["direction"], dtype=w.dtype)
            if w.ndim != 2 or u.shape != (w.shape[0],) or np.linalg.norm(u) == 0:
                raise ValueError("invalid output direction")
            u = u / np.linalg.norm(u)
            w -= float(spec.get("alpha", 1)) * np.outer(u, u @ w)
        elif mode == "ablate":
            axis = int(spec.get("axis", 0))
            if axis not in range(w.ndim):
                raise ValueError("invalid axis")
            idx = [slice(None)] * w.ndim
            idx[axis] = spec["indices"]
            w[tuple(idx)] *= float(spec.get("scale", 0))
        elif mode == "random":
            magnitude = float(spec["norm"])
            if magnitude < 0:
                raise ValueError("negative norm")
            delta = np.random.default_rng(spec["seed"]).normal(size=w.shape)
            w += (delta * (magnitude / np.linalg.norm(delta))).astype(w.dtype)
        else:
            raise ValueError("unknown edit mode")
    diff = compare(base, out)
    child = store.put(
        out, dict(parent=parent, spec=spec, hypothesis=hypothesis, delta=diff)
    )
    event = dict(
        time=time.time(),
        parent=parent,
        child=child,
        spec=spec,
        hypothesis=hypothesis,
        delta=diff,
        status="pending",
    )
    with (store.root / "history.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event


def rollback(store, child):
    parent = store.manifest(child).get("parent")
    if parent is None:
        raise ValueError("no parent for rollback")
    store.get(parent)
    with (store.root / "history.jsonl").open("a") as f:
        f.write(
            json.dumps(dict(action="rollback", child=child, restored=parent)) + "\n"
        )
    return parent
