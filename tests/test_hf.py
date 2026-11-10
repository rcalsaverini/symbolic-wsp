"""Offline real forward/generation test; optional dependencies, zero downloads."""

import numpy as np
import pytest


def test_local_transformers(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from symbolic_wsp.inference.hf import HFBackend
    from symbolic_wsp.wsp.checkpoints import Store
    from symbolic_wsp.wsp.editing import edit

    vocab = {"[UNK]": 0, "[EOS]": 1, "x": 2, "1": 3, "0": 4}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    )
    path = tmp_path / "local"
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5, n_embd=8, n_layer=1, n_head=1, n_positions=512, eos_token_id=1
        )
    ).save_pretrained(path)
    fast.save_pretrained(path)
    backend = HFBackend(str(path), expressions=["x", "1", "0"])
    obs = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    assert set(backend.generate(obs, 4, 17)) <= {"x", "1", "0"}
    assert backend.probe(obs)["top_logits"]
    store = Store(tmp_path / "store")
    baseline = backend.snapshot(store)
    spec = {
        "mode": "ablate",
        "tensor": "transformer.h.0.mlp.c_proj.weight",
        "axis": 0,
        "indices": [0],
    }
    child = edit(store, baseline, spec, "offline test ablation")["child"]
    edited = HFBackend(
        str(path), expressions=["x", "1", "0"], store=store, identity=child
    )
    np.testing.assert_array_equal(
        edited.model.state_dict()[spec["tensor"]].detach().numpy()[0], 0
    )
    assert edited.identity == child
    assert len(edited.generate(obs, 2, 17)) == 2
    edited.export(tmp_path / "exported")
    assert (tmp_path / "exported/model.safetensors").exists()
