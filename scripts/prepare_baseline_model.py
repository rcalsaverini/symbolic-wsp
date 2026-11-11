"""Download only pinned public model files, without Hub cache/HEAD dependencies."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "README.md",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/pretrained-baseline-01/config.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    prior_path = dest / "download-manifest.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    if prior is not None and (prior["model_id"], prior["revision"]) != (
        cfg["model_id"],
        cfg["revision"],
    ):
        raise ValueError("destination belongs to a different model revision")
    hashes = {}
    for name in FILES:
        target = dest / name
        if prior is not None and target.exists():
            with target.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != prior["sha256"].get(name):
                raise ValueError("existing file hash mismatch: " + name)
        # With no complete manifest, re-fetch rather than trusting unknown files.
        if not target.exists() or prior is None:
            url = f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['revision']}/{name}"
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                temporary = dest / (name + ".partial")
                with temporary.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
                temporary.rename(target)
        with target.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"{name}: {target.stat().st_size} bytes", flush=True)
    manifest = dict(model_id=cfg["model_id"], revision=cfg["revision"], sha256=hashes)
    (dest / "download-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
