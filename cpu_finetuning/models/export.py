"""Full-weight export (requirement 11).

Writes a complete, ready-to-use HF-format model directory (safetensors +
config + tokenizer/generation config).  For multimodal source models the
vision/audio towers are preserved untouched; the fine-tuned text stack is
written back into the full checkpoint structure.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.jsonutil import dump as json_dump
from ..core.paths import EXPORT_DIR


def _write_meta(out_dir: Path, lm, metrics: dict | None):
    info = {
        "exported_by": "CPU-FineTuning",
        "method": "ASCOT (Anchor-State COmpressed Training)",
        "source_model": getattr(lm, "source", "") if lm is not None else "",
        "requires_trust_remote_code": bool(
            getattr(lm, "loaded_via_remote_code", False)
        ) if lm is not None else False,
        **(metrics or {}),
    }
    with open(out_dir / "cpu_finetuning_meta.json", "w", encoding="utf-8") as f:
        json_dump(info, f, indent=2)


def _save_tokenizer(tok, out_dir: Path):
    try:
        tok.save_pretrained(str(out_dir))
    except Exception:
        try:
            tok.tokenizer.save_pretrained(str(out_dir))
        except Exception:
            pass


def export_checkpoint(lm, checkpoint_dir, name: str,
                      metrics: dict | None = None) -> str:
    """Export a complete model from a checkpoint directory.

    A v8 full-HF checkpoint is copied without loading it again.  A v9 delta
    checkpoint is reconstructed into the live model first, then written as a
    complete HF model: the checkpoint itself stays small, while the export is
    always ready to use.
    """
    import shutil
    import json

    src = Path(checkpoint_dir)
    if not src.exists():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    delta_file = src / "delta.safetensors"
    if delta_file.exists():
        meta = {}
        try:
            meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        manifest = meta.get("delta") or {}
        base = manifest.get("base_model") or getattr(lm, "source", "")
        from .delta import load_delta
        load_delta(lm.model, delta_file, base_source=base, manifest=manifest)
        return export_model(lm, name, metrics=metrics)

    out_dir = Path(EXPORT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == "optimizer.pt":
            continue
        dst = out_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dst)
    _save_tokenizer(lm.tokenizer, out_dir)
    _write_meta(out_dir, lm, metrics)
    return str(out_dir)


def export_model(lm, name: str, metrics: dict | None = None) -> str:
    """Export ``lm`` (a LoadedModel) to ``exports/<name>``."""
    out_dir = Path(EXPORT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = lm.model, lm.tokenizer

    # shadow caches must not leak into the saved state
    from ..models.qlayer import set_shadow_enabled
    from ..models.loader import _strip_auto_map, persist_remote_code
    set_shadow_enabled(False)
    try:
        remote = persist_remote_code(model, out_dir)
        if not remote:
            _strip_auto_map(model.config)   # exports must never require remote code
        model.save_pretrained(str(out_dir), safe_serialization=True)
    finally:
        # A failed save must not leave the live trainer without int8 shadows.
        set_shadow_enabled(True)

    _save_tokenizer(tok, out_dir)
    _write_meta(out_dir, lm, metrics)
    return str(out_dir)
