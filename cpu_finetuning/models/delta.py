"""Delta checkpoints.

A full fine-tune checkpoint used to be a complete fp32 copy of the model,
which is roughly twice the size of the original bf16 source model.  These
helpers store only ``current - base`` instead.  The base model is read lazily
from disk, one tensor at a time, so saving a checkpoint does not require a
second full model in RAM.

The default on-disk format is an int8 delta with per-row (or per-tensor for
1-D tensors) float32 scales.  The reconstruction is::

    current = base + int8_delta * scale

For a 1.7B model this is about half the size of the original bf16 model,
instead of twice its size.  An exact bf16/fp16 delta is also available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

FORMAT = "cpuft-delta-v1"
_META_KEYS = {"__metadata__"}


def _weight_files(base_source) -> list[Path]:
    source = Path(str(base_source))
    if source.is_file() and source.suffix == ".safetensors":
        return [source]
    if source.is_dir():
        single = source / "model.safetensors"
        if single.is_file():
            return [single]
        index = source / "model.safetensors.index.json"
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
            names = sorted(set((data.get("weight_map") or {}).values()))
            found = [source / name for name in names if (source / name).is_file()]
            if found:
                return found
        shards = sorted(source.glob("*.safetensors"))
        if shards:
            return shards
    # A repository id: reuse the already-downloaded local snapshot.  Never
    # trigger a fresh multi-gigabyte download while saving a checkpoint.
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=str(base_source),
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            local_files_only=True,
        )
        return _weight_files(local)
    except Exception:
        pass
    raise FileNotFoundError(
        "delta checkpoint needs the original base model weights; "
        f"could not find model.safetensors under {source}"
    )


def _open_shards(files: Iterable[Path]):
    from safetensors import safe_open
    return [safe_open(str(path), framework="pt", device="cpu") for path in files]


def _close_shards(handles) -> None:
    for handle in handles:
        try:
            handle.__exit__(None, None, None)
        except Exception:
            pass


def _base_key_map(handles) -> dict[str, object]:
    out = {}
    for handle in handles:
        for key in handle.keys():
            if key not in _META_KEYS:
                out.setdefault(key, handle)
    return out


def base_fingerprint(base_source) -> dict:
    """Cheap identity for the base model (no multi-gigabyte hashing)."""
    files = _weight_files(base_source)
    total = 0
    rows = []
    for path in files:
        stat = path.stat()
        total += stat.st_size
        rows.append({"name": path.name, "size": stat.st_size})
    return {"files": rows, "bytes": total}


def _row_scale(delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim >= 2:
        maxv = delta.abs().amax(dim=1, keepdim=True).float()
    elif delta.ndim == 0:
        maxv = delta.abs().float()
    else:
        maxv = delta.abs().amax().reshape(1).float()
    return (maxv / 127.0).clamp_min(1e-12)


def save_delta(model, base_source, out_path, dtype: str = "int8",
               extra_meta: dict | None = None) -> dict:
    """Save a delta relative to *base_source*.

    Returns a manifest dict suitable for writing to ``meta.json``.
    """
    from safetensors.torch import save_file

    mode = str(dtype or "int8").lower()
    if mode not in {"int8", "bfloat16", "float16", "float32"}:
        mode = "int8"
    files = _weight_files(base_source)
    handles = _open_shards(files)
    key_map = _base_key_map(handles)
    store: dict[str, torch.Tensor] = {}
    saved = 0
    exact = 0
    missing = []
    mismatched = []
    try:
        state = model.state_dict()
        for name, value in state.items():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            handle = key_map.get(name)
            if handle is None:
                missing.append(name)
                continue
            base = handle.get_tensor(name).to(torch.float32)
            current = value.detach().to(device="cpu", dtype=torch.float32)
            if tuple(base.shape) != tuple(current.shape):
                mismatched.append(name)
                continue
            delta = current - base
            if not torch.isfinite(delta).all():
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            if float(delta.abs().max().item()) == 0.0:
                exact += 1
                continue
            if mode == "int8":
                scale = _row_scale(delta)
                q = torch.round(delta / scale.clamp_min(1e-12))
                q = q.clamp(-127.0, 127.0).to(torch.int8).contiguous()
                store[f"delta.{name}"] = q
                store[f"scale.{name}"] = scale.contiguous()
            else:
                target = {"bfloat16": torch.bfloat16,
                          "float16": torch.float16,
                          "float32": torch.float32}[mode]
                store[f"delta.{name}"] = delta.to(target).contiguous()
            saved += 1
    finally:
        _close_shards(handles)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if store:
        save_file(store, str(out_path))
    else:
        # safetensors cannot write a completely empty file; use a marker.
        out_path.write_bytes(b"")
    manifest = {
        "format": FORMAT,
        "dtype": mode,
        "quantization": "per-row" if mode == "int8" else "tensor",
        "base_model": str(Path(str(base_source)).resolve()),
        "base_fingerprint": base_fingerprint(base_source),
        "saved_tensors": saved,
        "zero_tensors": exact,
        "missing_tensors": len(missing),
        "mismatched_tensors": len(mismatched),
        "file_bytes": out_path.stat().st_size if out_path.exists() else 0,
        "empty": not store,
    }
    if extra_meta:
        manifest.update(extra_meta)
    return manifest


def _iter_base(handles):
    key_map = _base_key_map(handles)
    return key_map


def load_delta(model, delta_path, base_source=None, manifest: dict | None = None) -> dict:
    """Load a delta checkpoint into *model* in place.

    The base model path is taken from the manifest when available.  Tensors
    that are absent from the delta are restored from the base too, which is
    important when rolling back from a later model state to step-0.
    """
    from safetensors import safe_open

    delta_path = Path(delta_path)
    manifest = manifest or {}
    base = base_source or manifest.get("base_model")
    if not base:
        raise ValueError("delta checkpoint has no base model reference")
    files = _weight_files(base)
    handles = _open_shards(files)
    key_map = _iter_base(handles)
    restored = 0
    missing_base = []
    try:
        state = model.state_dict()
        with torch.no_grad():
            if delta_path.exists() and delta_path.stat().st_size > 0:
                with safe_open(str(delta_path), framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    for name, value in state.items():
                        if not torch.is_tensor(value) or not value.is_floating_point():
                            continue
                        base_tensor = None
                        handle = key_map.get(name)
                        if handle is not None:
                            base_tensor = handle.get_tensor(name).to(torch.float32)
                        dkey = f"delta.{name}"
                        skey = f"scale.{name}"
                        if dkey in keys:
                            d = f.get_tensor(dkey)
                            if base_tensor is None:
                                missing_base.append(name)
                                continue
                            if d.dtype == torch.int8:
                                scale = f.get_tensor(skey).to(torch.float32)
                                new = base_tensor + d.to(torch.float32) * scale
                            else:
                                new = base_tensor + d.to(torch.float32)
                            value.copy_(new.to(device=value.device,
                                               dtype=value.dtype))
                            restored += 1
                        elif base_tensor is not None:
                            value.copy_(base_tensor.to(device=value.device,
                                                       dtype=value.dtype))
            else:
                # Empty delta is the step-0 baseline: restore the base exactly.
                for name, value in state.items():
                    if not torch.is_tensor(value) or not value.is_floating_point():
                        continue
                    handle = key_map.get(name)
                    if handle is None:
                        missing_base.append(name)
                        continue
                    value.copy_(handle.get_tensor(name).to(device=value.device,
                                                           dtype=value.dtype))
                    restored += 1
    finally:
        _close_shards(handles)
    return {"restored": restored, "missing_base": len(missing_base)}
