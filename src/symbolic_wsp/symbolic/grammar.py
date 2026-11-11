"""A finite bounded grammar compiled to complete pretrained-token sequences.
Every leaf is parsed first. Terminals may span multiple tokenizer tokens.
"""

from .evaluator import Grammar

DEFAULT_EXPRESSIONS = (
    "x",
    "0",
    "1",
    "2*x",
    "x+1",
    "x*x",
    "x*x+1",
    "sin(x)",
    "cos(x)",
    "exp(x)",
    "1/(x*x+1)",
    "x/(x*x+1)",
    "sin(x)+x",
    "sin(x*x)",
)


class TokenGrammar:
    def __init__(
        self,
        tokenizer,
        expressions=DEFAULT_EXPRESSIONS,
        grammar=Grammar(),
        max_tokens=128,
    ):
        self.sequences = {}
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("tokenizer requires EOS")
        for expr in expressions:
            grammar.parse(expr)
            tokens = tuple(tokenizer.encode(expr, add_special_tokens=False))
            if not tokens or len(tokens) + 1 > max_tokens:
                raise ValueError("expression exceeds generation budget")
            decoded = tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if decoded.strip() != expr:
                raise ValueError("tokenizer roundtrip changed expression")
            self.sequences[tokens + (self.eos,)] = expr
        if not self.sequences:
            raise ValueError("empty grammar")

    def allowed(self, prefix):
        prefix = tuple(prefix)
        # Transformers still masks finished rows while longer batch mates run.
        # EOS is also our padding token; only an already valid leaf may absorb it.
        if self.eos in prefix:
            end = prefix.index(self.eos) + 1
            if prefix[:end] in self.sequences and all(
                token == self.eos for token in prefix[end:]
            ):
                return [self.eos]
            return []
        return sorted(
            {
                seq[len(prefix)]
                for seq in self.sequences
                if len(seq) > len(prefix) and seq[: len(prefix)] == prefix
            }
        )

    def expression(self, tokens):
        return self.sequences[tuple(tokens)]
