"""Exact finite-catalog probabilities under autoregressive grammar masking."""

import copy
import hashlib
import time

import numpy as np

from ..experiments.pretrained_baseline import digest
from ..symbolic.grammar import TokenGrammar


def probabilities_from_logits(sequences, prefix_logits, temperature=1.0):
    """Normalize over permitted next tokens separately at every trie prefix."""
    probabilities = {}
    branches = {}
    for sequence in sequences:
        for index in range(len(sequence)):
            prefix = tuple(sequence[:index])
            if prefix in branches:
                continue
            allowed = sorted(
                {
                    seq[index]
                    for seq in sequences
                    if len(seq) > index and tuple(seq[:index]) == prefix
                }
            )
            logits = (
                np.asarray(prefix_logits[prefix], dtype=np.float64)[allowed]
                / temperature
            )
            logsum = np.max(logits) + np.log(np.exp(logits - np.max(logits)).sum())
            branches[prefix] = dict(zip(allowed, logits - logsum))
    for sequence in sequences:
        probabilities[tuple(sequence)] = float(
            np.exp(
                sum(
                    branches[tuple(sequence[:i])][token]
                    for i, token in enumerate(sequence)
                )
            )
        )
    mass = sum(probabilities.values())
    if not np.isclose(mass, 1.0, atol=1e-7):
        raise ValueError("incomplete catalog probability mass")
    return probabilities, branches


class CatalogOracle:
    """One prompt prefill plus a batched teacher-forced suffix pass, no sampling."""

    def __init__(self, model, tokenizer, inventory):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar = TokenGrammar(tokenizer, inventory)
        self._cache = {}
        self.sequences = list(self.grammar.sequences)
        self.inventory = [self.grammar.sequences[seq] for seq in self.sequences]

    def distribution(self, prompt, root_bias=None):
        import torch
        from transformers import DynamicCache

        cache_key = digest({"prompt": prompt, "root_bias": root_bias})
        if cache_key in self._cache:
            result = copy.deepcopy(self._cache[cache_key])
            result["cache_hit"] = True
            result["seconds"] = 0.0
            return result
        start = time.perf_counter()
        ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        n = ids.shape[1]
        batch = len(self.sequences)
        longest = max(map(len, self.sequences)) - 1
        if n + longest + 1 > self.model.config.max_position_embeddings:
            raise ValueError("context limit; truncation forbidden")
        with torch.inference_mode():
            prefill = self.model(ids, use_cache=True, return_dict=True)
            root = prefill.logits[0, -1].float().cpu().numpy().copy()
            if root_bias:
                for token, bias in root_bias.items():
                    root[int(token)] += float(bias)
            legacy = prefill.past_key_values.to_legacy_cache()
            expanded = tuple(
                tuple(t.expand(batch, *t.shape[1:]).contiguous() for t in layer)
                for layer in legacy
            )
            cache = DynamicCache.from_legacy_cache(expanded)
            suffix = torch.full((batch, longest), self.grammar.eos, dtype=torch.long)
            mask = torch.zeros((batch, n + longest), dtype=torch.long)
            mask[:, :n] = 1
            for i, seq in enumerate(self.sequences):
                suffix[i, : len(seq) - 1] = torch.tensor(seq[:-1])
                mask[i, n : n + len(seq) - 1] = 1
            continuation = self.model(
                suffix,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=False,
                return_dict=True,
            )
            logits = continuation.logits.float().cpu().numpy()
        prefix_logits = {(): root}
        for i, seq in enumerate(self.sequences):
            for j in range(1, len(seq)):
                prefix_logits.setdefault(tuple(seq[:j]), logits[i, j - 1])
        probs, branches = probabilities_from_logits(self.sequences, prefix_logits)
        mass = sum(probs.values())
        ordered = np.array([probs[s] for s in self.sequences])
        ordered /= mass
        result = dict(
            probabilities=dict(zip(self.inventory, ordered.tolist())),
            probability_mass_before_roundoff=mass,
            root_allowed=[
                dict(
                    token_id=t,
                    text=self.tokenizer.decode([t]),
                    probability=float(np.exp(lp)),
                )
                for t, lp in branches[()].items()
            ],
            prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_tokens=n,
            suffix_token_slots=batch * longest,
            seconds=time.perf_counter() - start,
            branches=[
                dict(
                    prefix=list(prefix),
                    token_log_probabilities={
                        str(t): float(p) for t, p in values.items()
                    },
                )
                for prefix, values in branches.items()
            ],
            inventory_digest=digest(self.inventory),
            cache_hit=False,
        )
        self._cache[cache_key] = copy.deepcopy(result)
        return result
