"""Adapted from Felipe Penha's wsp_engine.py, Apache-2.0; see NOTICE.
Modification: extracted pure NumPy DSL helpers, without Pi or sandbox imports.
"""

import ast
import re
from typing import Any, Dict, List, Optional

import numpy as np


def _parse_slice_item(part: str):
    part = part.strip()
    if ":" not in part:
        return int(part)
    pieces = [p.strip() for p in part.split(":")]
    if len(pieces) == 2:
        start = int(pieces[0]) if pieces[0] else None
        stop = int(pieces[1]) if pieces[1] else None
        return slice(start, stop)
    if len(pieces) == 3:
        start = int(pieces[0]) if pieces[0] else None
        stop = int(pieces[1]) if pieces[1] else None
        step = int(pieces[2]) if pieces[2] else None
        return slice(start, stop, step)
    raise ValueError(f"Invalid slice syntax: {part!r}")


def _parse_slice_tuple(spec: str):
    items = [_parse_slice_item(p) for p in spec.split(",")]
    return items[0] if len(items) == 1 else tuple(items)


def _resolve_dsl_tensor_name(
    tname: str, tensors: Dict[str, np.ndarray]
) -> Optional[str]:
    if tname in tensors:
        return tname
    alt_u = tname.replace(".", "_")
    if alt_u in tensors:
        return alt_u
    alt_d = tname.replace("_", ".")
    if alt_d in tensors:
        return alt_d
    # Flax Actor-Critic RNN short aliases used in Experiment 4 (exp4_fix_car)
    flax_map: Dict[str, str] = {
        "embed.W": "params/Dense_0/kernel",
        "embed.b": "params/Dense_0/bias",
        "actor_head.W": "params/GeneralActorCriticRNN_0/Dense_0/kernel",
        "actor_head.b": "params/GeneralActorCriticRNN_0/Dense_0/bias",
        "critic_head.W": "params/GeneralActorCriticRNN_0/Dense_1/kernel",
        "critic_head.b": "params/GeneralActorCriticRNN_0/Dense_1/bias",
    }
    for k in range(5):
        flax_map[f"actor{k}.W"] = (
            f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k}/Dense_0/kernel"
        )
        flax_map[f"actor{k}.b"] = (
            f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k}/Dense_0/bias"
        )
        flax_map[f"critic{k}.W"] = (
            f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k + 5}/Dense_0/kernel"
        )
        flax_map[f"critic{k}.b"] = (
            f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k + 5}/Dense_0/bias"
        )
    mapped = flax_map.get(tname)
    if mapped and mapped in tensors:
        return mapped
    return None


def apply_edits_dsl(tensors: Dict[str, np.ndarray], dsl_text: str) -> Dict[str, Any]:
    """Parse and apply the `edits` block DSL from Experiments 4 and 5."""
    # Extract last ```edits ... ``` block if present, otherwise parse raw text
    blocks = re.findall(r"```edits\s*\n(.*?)```", dsl_text, flags=re.S)
    body = blocks[-1] if blocks else dsl_text

    # Join multiline bracket expressions while stripping comments
    clean_lines = []
    for raw_line in body.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            clean_lines.append(line)

    statements: List[str] = []
    buf = ""
    bracket_depth = 0
    for line in clean_lines:
        buf = (buf + " " + line).strip() if buf else line
        bracket_depth += line.count("[") - line.count("]")
        if bracket_depth <= 0:
            statements.append(buf)
            buf = ""
            bracket_depth = 0
    if buf:
        statements.append(buf)

    applied = 0
    errors = []
    touched = set()
    stmt_rx = re.compile(
        r"^([A-Za-z0-9_./]+)\s*\[\s*([^\]]+)\s*\]\s*(=|\+=|-=|\*=)\s*(.+)$"
    )

    for stmt in statements:
        m = stmt_rx.match(stmt)
        if not m:
            errors.append(
                {
                    "statement": stmt,
                    "error": "Syntax did not match TENSOR[slice] op value",
                }
            )
            continue
        raw_tname, slice_str, op, val_str = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4),
        )
        tname = _resolve_dsl_tensor_name(raw_tname, tensors)
        if tname is None:
            errors.append({"statement": stmt, "error": f"Unknown tensor {raw_tname!r}"})
            continue
        try:
            idx = _parse_slice_tuple(slice_str)
            rhs = ast.literal_eval(val_str.strip())
            rhs_arr = np.asarray(rhs, dtype=np.float32)
            if not np.all(np.abs(rhs_arr) <= 1e6):
                raise ValueError("Values exceed |v| <= 1e6 bound")
            arr = tensors[tname]
            if op == "=":
                arr[idx] = rhs_arr
            elif op == "+=":
                arr[idx] += rhs_arr
            elif op == "-=":
                arr[idx] -= rhs_arr
            elif op == "*=":
                arr[idx] *= rhs_arr
            touched.add(tname)
            applied += 1
        except Exception as exc:
            errors.append({"statement": stmt, "error": str(exc)})

    return {
        "statements_total": len(statements),
        "statements_applied": applied,
        "errors": errors,
        "tensors_touched": sorted(touched),
    }
