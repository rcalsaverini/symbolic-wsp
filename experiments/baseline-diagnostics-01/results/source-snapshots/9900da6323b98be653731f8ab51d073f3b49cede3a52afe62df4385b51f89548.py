"""Optional Transformers worker. Uses real forward passes; no training."""

import hashlib
import json
from pathlib import Path

from ..symbolic.grammar import DEFAULT_EXPRESSIONS, TokenGrammar


class HFBackend:
    def __init__(
        self,
        model_path,
        device="cpu",
        expressions=DEFAULT_EXPRESSIONS,
        store=None,
        identity=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False)
            .to(device)
            .eval()
        )
        if store is not None:
            if identity is None:
                raise ValueError("checkpoint identity required")
            arrays = store.get(identity)
            state = self.model.state_dict()
            if set(arrays) != set(state):
                raise ValueError("checkpoint must contain complete model state")
            converted = {
                k: torch.from_numpy(a).to(dtype=state[k].dtype)
                for k, a in arrays.items()
            }
            self.model.load_state_dict(converted, strict=True)
        else:
            # Record resolved model configuration; remote model revisions should be pinned locally.
            digest = hashlib.sha256()
            for name, value in sorted(self.model.state_dict().items()):
                digest.update(name.encode())
                digest.update(str(value.shape).encode())
                digest.update(value.detach().float().cpu().numpy().tobytes())
            identity = digest.hexdigest()
        self.expressions = tuple(expressions)
        self.identity = identity
        self.grammar = TokenGrammar(self.tokenizer, expressions)

    def generate(self, observations, k, seed, feedback=None):
        torch = self.torch
        prompt = (
            "Find an expression in x for these observations. Return only the expression.\n"
            + json.dumps(observations)
        )
        if feedback:
            prompt += "\nPrevious candidates and observation scores: " + json.dumps(
                feedback
            )
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        width = ids.shape[1]

        def allowed(batch_id, tokens):
            return self.grammar.allowed(tokens[width:].tolist())

        torch.manual_seed(seed)
        with torch.inference_mode():
            output = self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                num_return_sequences=k,
                max_new_tokens=max(map(len, self.grammar.sequences)),
                prefix_allowed_tokens_fn=allowed,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        expressions = []
        for row in output[:, width:].tolist():
            row = row[: row.index(self.grammar.eos) + 1]
            expressions.append(self.grammar.expression(row))
        return expressions

    def probe(self, observations):
        torch = self.torch
        ids = self.tokenizer.encode(json.dumps(observations), return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            output = self.model(ids, output_hidden_states=True)
        values, indices = output.logits[0, -1].topk(min(10, output.logits.shape[-1]))
        return dict(
            checkpoint=self.identity,
            last_token_hidden=[
                h[0, -1].float().cpu().tolist() for h in output.hidden_states
            ],
            top_logits=dict(
                zip(map(str, indices.cpu().tolist()), values.float().cpu().tolist())
            ),
        )

    def snapshot(self, store):
        arrays = {
            k: v.detach().float().cpu().numpy()
            for k, v in self.model.state_dict().items()
        }
        return store.put(
            arrays,
            dict(
                source="transformers",
                config=self.model.config.to_dict(),
                dtype="float32",
            ),
        )

    def export(self, destination):
        if Path(destination).exists():
            raise ValueError("export destination exists")
        self.model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)
