"""Export recorded prompts and verify a configured replay with a local tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from symbolic_wsp.experiments.pretrained_baseline import digest, render_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/pretrained-baseline-01/results"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/pretrained-baseline-01/config.json"),
    )
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    historical = json.loads((args.results / "config.json").read_text())
    tasks = {t["id"]: t for t in json.loads((args.results / "tasks.json").read_text())}
    rows = [
        json.loads(line)
        for line in (args.results / "rows.jsonl").read_text().splitlines()
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        if row["method"] not in ("conditioned", "no_observations"):
            continue
        obs = (
            tasks[row["task"]]["observations"]
            if row["method"] == "conditioned"
            else None
        )
        prompt = render_prompt(tokenizer, obs, cfg["prompt"])
        if prompt != row["trace"]["prompt"]:
            raise ValueError(
                "configured prompt differs from historical input: " + row["task"]
            )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) != row["trace"]["prompt_tokens"]:
            raise ValueError("token count differs")
        name = row["task"] + ".txt" if obs is not None else "no-observations.txt"
        (args.output / name).write_text(prompt)
        entries.append(
            dict(
                task=row["task"],
                method=row["method"],
                seed=row["seed"],
                file=name,
                sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                tokens=len(tokens),
            )
        )
    manifest = dict(
        historical_config_digest=digest(historical),
        explicit_config_digest=digest(cfg),
        prompt_config_digest=digest(cfg["prompt"]),
        verified_model_calls=len(entries),
        claim="Configured replay matches every recorded prompt byte-for-byte and all prompt token counts; no model inference rerun.",
        calls=entries,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Verified {len(entries)} recorded model prompts; wrote {len(set(e['file'] for e in entries))} distinct prompt files."
    )


if __name__ == "__main__":
    main()
