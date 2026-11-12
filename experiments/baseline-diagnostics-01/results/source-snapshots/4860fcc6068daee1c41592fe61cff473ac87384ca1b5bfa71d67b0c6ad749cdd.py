"""Portable generator protocol and synthetic CPU integration fixture."""

from typing import Protocol

import numpy as np

from ..symbolic.grammar import DEFAULT_EXPRESSIONS


class Backend(Protocol):
    identity: str

    def generate(
        self, observations: dict, k: int, seed: int, feedback=None
    ) -> list[str]: ...


class TinyBackend:
    """Synthetic categorical neural head. NOT a pretrained language model."""

    def __init__(self, store, identity, expressions=DEFAULT_EXPRESSIONS):
        self.identity = identity
        self.w = store.get(identity)["head"]
        self.expressions = tuple(expressions)
        if self.w.shape != (len(self.expressions), 2):
            raise ValueError("tiny head dimensions")

    def generate(self, observations, k, seed, feedback=None):
        x = np.asarray(observations["x"])
        y = np.asarray(observations["y"])
        features = np.array([1.0, np.tanh(np.mean(x * y))])
        logits = self.w @ features
        if feedback:
            # Fixed feedback penalty discourages exact repeats without weight updates.
            for entry in feedback:
                if entry["expression"] in self.expressions:
                    logits[self.expressions.index(entry["expression"])] -= 1
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return [
            self.expressions[i]
            for i in np.random.default_rng(seed).choice(len(p), k, p=p)
        ]

    def probe(self, observations):
        features = np.array(
            [1.0, np.tanh(np.mean(np.asarray(observations["x"]) * observations["y"]))]
        )
        return dict(
            checkpoint=self.identity,
            hidden=features.tolist(),
            logits=(self.w @ features).tolist(),
        )
