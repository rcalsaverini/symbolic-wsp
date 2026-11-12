"""Content-addressed immutable NPZ/safetensors checkpoint store."""

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def digest(arrays: dict[str, np.ndarray], metadata: dict) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(metadata, sort_keys=True).encode())
        for k, a in sorted(arrays.items()):
            if a.dtype.kind != "f" or not np.all(np.isfinite(a)):
                raise ValueError("finite float tensors required")
            h.update(json.dumps([k, a.dtype.str, a.shape]).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def put(self, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> str:
        metadata = metadata or {}
        if "id" in metadata:
            raise ValueError("reserved metadata field: id")
        arrays = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
        identity = self.digest(arrays, metadata)
        target = self.root / identity
        if not target.exists():
            target.mkdir()
            save_file(arrays, str(target / "model.safetensors"))
            (target / "manifest.json").write_text(
                json.dumps(dict(id=identity, **(metadata or {})), indent=2)
            )
        return identity

    def get(self, identity):
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid checkpoint ID")
        arrays = load_file(str(self.root / identity / "model.safetensors"))
        metadata = json.loads((self.root / identity / "manifest.json").read_text())
        if metadata.pop("id") != identity or self.digest(arrays, metadata) != identity:
            raise ValueError("checkpoint integrity mismatch")
        return arrays

    def manifest(self, identity):
        self.get(identity)
        return json.loads((self.root / identity / "manifest.json").read_text())

    def import_file(self, path):
        if str(path).endswith(".safetensors"):
            arrays = load_file(str(path))
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        return self.put(arrays, dict(source=str(path)))

    def export(self, identity, path):
        if Path(path).exists():
            raise ValueError("export destination exists")
        save_file(self.get(identity), str(path))


def inspect(tensors):
    result = {}
    for k, a in tensors.items():
        s = (
            np.linalg.svd(a, compute_uv=False)
            if a.ndim == 2 and max(a.shape) <= 2048
            else None
        )
        result[k] = dict(
            shape=list(a.shape),
            dtype=str(a.dtype),
            norm=float(np.linalg.norm(a)),
            min=float(a.min()),
            max=float(a.max()),
            singular_values=s[:16].tolist() if s is not None else None,
        )
    return result


def compare(a, b):
    if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("checkpoint schemas differ")
    return {
        k: dict(
            norm=float(np.linalg.norm(b[k].astype(float) - a[k])),
            changed=int(np.count_nonzero(a[k] != b[k])),
        )
        for k in a
    }
