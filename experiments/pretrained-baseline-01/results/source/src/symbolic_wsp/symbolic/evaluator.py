"""Bounded AST interpreter. Never executes generated Python code."""

import ast
from dataclasses import dataclass

import numpy as np

FUNCTIONS = {
    k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs")
}
BINARY = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}


@dataclass(frozen=True)
class Grammar:
    functions: tuple[str, ...] = tuple(FUNCTIONS)
    operators: tuple[str, ...] = ("Add", "Sub", "Mult", "Div", "Pow")
    max_nodes: int = 80
    max_chars: int = 400
    max_points: int = 10000
    max_depth: int = 20

    def parse(self, text: str):
        if len(text) > self.max_chars:
            raise ValueError("expression too long")
        tree = ast.parse(text.replace("^", "**"), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
            raise ValueError("expression too large")

        def check(n, depth=0):
            if depth > self.max_depth:
                raise ValueError("expression too deep")
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                if not np.isfinite(n.value) or abs(n.value) > 1e6:
                    raise ValueError("constant out of bounds")
            elif isinstance(n, ast.Name) and n.id in ("x", "pi", "e"):
                pass
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                check(n.operand, depth + 1)
            elif (
                isinstance(n, ast.BinOp)
                and type(n.op) in BINARY
                and type(n.op).__name__ in self.operators
            ):
                check(n.left, depth + 1)
                check(n.right, depth + 1)
            elif (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in self.functions
                and len(n.args) == 1
                and not n.keywords
            ):
                check(n.args[0], depth + 1)
            else:
                raise ValueError("node outside grammar: " + type(n).__name__)

        check(tree.body)
        return tree


def evaluate(text, x, grammar=Grammar()):
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 1
        or not 0 < x.size <= grammar.max_points
        or not np.all(np.isfinite(x))
    ):
        raise ValueError("invalid evaluation inputs")
    tree = grammar.parse(text)

    def walk(n):
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            return {"x": x, "pi": np.pi, "e": np.e}[n.id]
        if isinstance(n, ast.UnaryOp):
            return (-1 if isinstance(n.op, ast.USub) else 1) * walk(n.operand)
        if isinstance(n, ast.Call):
            return FUNCTIONS[n.func.id](walk(n.args[0]))
        a, b = walk(n.left), walk(n.right)
        if isinstance(n.op, ast.Pow) and np.any(np.abs(b) > 32):
            raise ValueError("exponent exceeds resource bound")
        return BINARY[type(n.op)](a, b)

    with np.errstate(all="raise"):
        y = np.broadcast_to(walk(tree.body), x.shape).copy()
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite prediction")
    return y


def score(text, x, y, penalty=0.001, grammar=Grammar()):
    if penalty < 0:
        raise ValueError("negative complexity penalty")
    try:
        y = np.asarray(y, dtype=float)
        pred = evaluate(text, x, grammar)
        if y.shape != pred.shape or not np.all(np.isfinite(y)):
            raise ValueError("invalid targets")
        with np.errstate(all="raise"):
            mse = float(np.mean((pred - y) ** 2))
            nmse = mse / max(float(np.var(y)), 1e-12)
        tree = grammar.parse(text)
        complexity = sum(
            isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant))
            for n in ast.walk(tree)
        )
        return dict(
            valid=True,
            mse=mse,
            nmse=nmse,
            complexity=complexity,
            objective=nmse + penalty * complexity,
            structure=ast.dump(tree, include_attributes=False),
        )
    except (ValueError, SyntaxError, ArithmeticError, TypeError, RecursionError) as exc:
        return dict(
            valid=False,
            mse=None,
            nmse=None,
            complexity=None,
            objective=None,
            error=str(exc),
        )
