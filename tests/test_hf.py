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
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
    )
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


def test_explicit_baseline_generation(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from symbolic_wsp.experiments.pretrained_baseline import (
        BaselineGenerator,
        render_prompt,
    )

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[EOS]": 1, "x": 2, "+": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    )
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
    )
    path = tmp_path / "local"
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=512, eos_token_id=1
        )
    ).save_pretrained(path)
    fast.save_pretrained(path)
    cfg = dict(
        torch_threads=1,
        prompt=dict(
            version=1,
            system_message="Test system",
            user_template="Expression\n{observations}",
            no_observations="No data",
            significant_digits=6,
            chat_template=fast.chat_template,
            add_generation_prompt=True,
        ),
        inventory=["x", "x + x"],
        generation=dict(
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.0,
            use_cache=True,
        ),
    )
    model = BaselineGenerator(str(path), cfg)
    obs = dict(x=[0.0, 1.0], y=[0.0, 1.0])
    expected_prompt = render_prompt(model.tokenizer, obs, cfg["prompt"])
    model.tokenizer.chat_template = "This tokenizer default must be ignored"
    assert render_prompt(model.tokenizer, obs, cfg["prompt"]) == expected_prompt
    a = model.generate(obs, 16, 17)
    assert a["prompt"] == expected_prompt
    assert "prompt_config_digest" in a and "rendered_prompt_sha256" in a
    b = model.generate(obs, 16, 17)
    assert a["token_ids"] == b["token_ids"]
    assert set(a["expressions"]) == {"x", "x + x"}
    assert a["generated_tokens"] == sum(map(len, a["token_ids"]))
    assert a["prompt_tokens"] > model.generate(None, 2, 17)["prompt_tokens"]
    assert model.generation.top_k == 0


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_catalog_oracle_matches_uncached_full_forwards(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from symbolic_wsp.inference.catalog import CatalogOracle, probabilities_from_logits

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[EOS]": 1, "x": 2, "+": 3, "1": 4}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    )
    if architecture == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=5, n_embd=8, n_layer=1, n_head=1, n_positions=128)
        ).eval()
    else:
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=5,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
        ).eval()
    oracle = CatalogOracle(model, fast, ["x", "x + x", "1"])
    prompt = "x + 1"
    ids = fast.encode(prompt, add_special_tokens=False)
    prefix_logits = {}
    with torch.inference_mode():
        for seq in oracle.sequences:
            for j in range(len(seq)):
                inp = torch.tensor([ids + list(seq[:j])])
                prefix_logits[tuple(seq[:j])] = model(inp).logits[0, -1].numpy()
    reference, _ = probabilities_from_logits(oracle.sequences, prefix_logits)
    measured = oracle.distribution(prompt)
    for seq, expected in reference.items():
        assert measured["probabilities"][
            oracle.grammar.sequences[seq]
        ] == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert oracle.distribution(prompt)["cache_hit"]
    assert oracle.distribution(prompt)["seconds"] == 0
    biased = oracle.distribution(prompt, {2: 2.0})
    assert (
        biased["probabilities"]["x"] + biased["probabilities"]["x + x"]
        > measured["probabilities"]["x"] + measured["probabilities"]["x + x"]
    )


def test_real_tiny_llama_intervention_restores_weights(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import hashlib
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from symbolic_wsp.experiments.diagnostics import render, tasks
    from symbolic_wsp.experiments.intervention import run_intervention
    from symbolic_wsp.inference.catalog import CatalogOracle

    torch.set_num_threads(1)
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "x": 2, "2": 3, "*": 4, "1": 5}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=6,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=1024,
        )
    ).eval()
    inventory = ["x", "2 * x", "x * x", "1"]
    spec = dict(
        version=1,
        system_message="test",
        user_template="{observations}",
        no_observations="none",
        significant_digits=6,
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}",
        add_generation_prompt=True,
    )
    cfg = dict(
        inventory=inventory,
        development_targets=["x", "2 * x"],
        evaluation_targets=["x * x", "1"],
        development_seed=1,
        evaluation_seed=2,
        points=4,
        k=2,
        penalty=0.001,
        prompt_variants=["minimal"],
        intervention=dict(
            model="135m",
            module_suffix="mlp.down_proj",
            relative_norms=[0.001, 0.01],
            random_seeds=[3, 4],
            repeated_failure_probability_threshold=1.0,
            minimum_failing_development_tasks=1,
            minimum_development_improvement=0.001,
            hypothesis="offline intervention test",
            token_bias_strength=1.0,
        ),
    )
    oracle = CatalogOracle(model, fast, inventory)
    development = []
    for task in tasks(cfg["development_targets"], 1, 4):
        dist = oracle.distribution(render(fast, task["observations"], spec))
        development.append(
            dict(
                target=task["target"],
                control="correct",
                variant="minimal",
                distribution=dist,
                metrics=dict(target_probability=dist["probabilities"][task["target"]]),
            )
        )
    (tmp_path / "135m-model-manifest.json").write_text(
        json.dumps(dict(model_id="synthetic", revision="offline"))
    )
    original = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    result = run_intervention(
        oracle,
        fast,
        model,
        cfg,
        dict(minimal=spec),
        dict(variant="minimal"),
        development,
        tmp_path,
    )
    assert result["status"] == "completed"
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, original)
    rollback = json.loads((tmp_path / "intervention/rollback.json").read_text())
    assert (
        rollback["restored"]
        and rollback["tensor_sha256"]
        == hashlib.sha256(original.numpy().tobytes()).hexdigest()
    )
    assert result["metrics"]["baseline/all"] == result["metrics"]["no_edit/all"]
    assert (tmp_path / "intervention/decision.json").exists()
