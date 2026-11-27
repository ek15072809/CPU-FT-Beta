"""Checkpoint management (requirement 11).

Policy: save every ``every`` steps; keep the newest ``keep_last`` rolling
checkpoints; additionally keep every ``keep_every``-th milestone checkpoint
forever.  The final export always writes complete HF-format weights without
any user action.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from ..core.jsonutil import dump as json_dump
from ..core.paths import CHECKPOINT_DIR


class CheckpointManager:
    def __init__(
        self,
        run_name: str,
        every: int = 100,
        keep_last: int = 2,
        keep_every: int = 500,
        with_optimizer: bool = False,
        log=None,
        lang: str = "ja",
        base_model_source: str = "",
        delta_dtype: str = "int8",
        tokenizer=None,
    ):
        self.dir = Path(CHECKPOINT_DIR) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = max(0, every)
        self.keep_last = max(1, keep_last)
        self.keep_every = max(0, keep_every)
        self.with_optimizer = with_optimizer
        self.log = log or (lambda msg: print(msg, flush=True))
        self.lang = lang
        self.base_model_source = str(base_model_source or "")
        self.delta_dtype = str(delta_dtype or "int8")
        self.tokenizer = tokenizer
        self.best_step = 0
        self.has_best = False
        self._load_best_pointer()

    # ------------------------------------------------------------------
    def _best_file(self) -> Path:
        return self.dir / "best.json"

    def _load_best_pointer(self):
        try:
            d = json.loads(self._best_file().read_text(encoding="utf-8"))
            self.best_step = int(d.get("step") or 0)
            self.has_best = True       # step 0 (pre-training baseline) is valid
        except Exception:
            self.best_step = 0
            self.has_best = False

    def _write_best_pointer(self, step: int, val_ce=None, metrics=None):
        self.best_step = int(step)
        self.has_best = True
        payload = {"step": int(step), "val_ce": val_ce,
                   "metrics": metrics or {}, "time": time.time()}
        with open(self._best_file(), "w", encoding="utf-8") as f:
            json_dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    def _ckpt_path(self, step: int) -> Path:
        return self.dir / f"step_{step:06d}"

    @staticmethod
    def _has_weights(path: Path) -> bool:
        path = Path(path)
        if (path / "delta.safetensors").exists():
            return True
        return any(path.glob("model*.safetensors"))

    def _write_small_model_files(self, model, path: Path):
        """Write config/tokenizer/remote code without the weight tensors."""
        from ..models.loader import _strip_auto_map, persist_remote_code

        remote = persist_remote_code(model, path)
        if not remote:
            _strip_auto_map(model.config)
        try:
            model.config.save_pretrained(str(path))
        except Exception:
            pass
        gen = getattr(model, "generation_config", None)
        if gen is not None:
            try:
                gen.save_pretrained(str(path))
            except Exception:
                pass
        tok = self.tokenizer
        if tok is not None:
            try:
                tok.save_pretrained(str(path))
            except Exception:
                try:
                    tok.tokenizer.save_pretrained(str(path))
                except Exception:
                    pass
        return remote

    def should_save(self, step: int, total: int) -> bool:
        # The final step is handled by the final evaluation/best-model path.
        # Check this before the modulo so checkpoint_every=1 does not write the
        # same step twice (once here and once as a newly discovered best).
        if step == total:
            return False
        if self.every and step > 0 and step % self.every == 0:
            return True
        return False

    def is_milestone(self, step: int) -> bool:
        return self.keep_every > 0 and step > 0 and step % self.keep_every == 0

    def save(self, step: int, model, optimizer=None, metrics: dict | None = None,
             best_val_ce: float | None = None):
        from ..core.i18n import t
        from ..models.loader import _strip_auto_map, persist_remote_code

        path = self._ckpt_path(step)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"step": step, "time": time.time(), "metrics": metrics or {}}
        # A best checkpoint may already exist at this step (for example when
        # an evaluation lands exactly on checkpoint_every).  Do not rewrite
        # identical weights; just update its metadata/pointer.
        if not self._has_weights(path):
            self._write_small_model_files(model, path)
            delta_meta = None
            if self.base_model_source:
                try:
                    from ..models.delta import save_delta
                    delta_meta = save_delta(
                        model, self.base_model_source,
                        path / "delta.safetensors",
                        dtype=self.delta_dtype,
                        extra_meta={"step": int(step),
                                    "base_model": self.base_model_source},
                    )
                    meta["checkpoint_format"] = "delta"
                    meta["delta"] = delta_meta
                except Exception as e:
                    self.log(t("delta_checkpoint_failed", self.lang,
                               err=repr(e)[:200]))
                    delta_meta = None
                    # Never leave a partial/misleading delta file next to a
                    # full fallback checkpoint.
                    try:
                        (path / "delta.safetensors").unlink(missing_ok=True)
                    except Exception:
                        pass
            if delta_meta is None:
                # Legacy full checkpoint as a safety fallback (large, but it
                # is better than losing the checkpoint entirely).
                if persist_remote_code(model, path):
                    pass
                else:
                    _strip_auto_map(model.config)
                model.save_pretrained(str(path), safe_serialization=True)
                meta["checkpoint_format"] = "full"
        else:
            # Existing delta/full checkpoint: keep its format metadata.
            try:
                old = json.loads((path / "meta.json").read_text(encoding="utf-8"))
                for key in ("checkpoint_format", "delta"):
                    if key in old:
                        meta[key] = old[key]
            except Exception:
                pass
        if best_val_ce is not None:
            meta["val_ce"] = best_val_ce
            meta["best"] = True
            self._write_best_pointer(step, best_val_ce, metrics=metrics)
        if self.with_optimizer and optimizer is not None:
            torch_path = path / "optimizer.pt"
            import torch
            try:
                state_payload = optimizer.state_dict()
            except Exception:
                state_payload = {"state": optimizer.state, "lr": optimizer.lr,
                                 "step_count": optimizer.step_count}
            torch.save(state_payload, torch_path)
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json_dump(meta, f)
        self.log(t("checkpoint_saved", self.lang, path=path))
        self._prune(step)
        return path

    def save_best(self, step: int, model, val_ce: float,
                  optimizer=None, metrics: dict | None = None):
        """Save and pin the checkpoint with the lowest validation CE."""
        return self.save(step, model, optimizer=optimizer, metrics=metrics,
                         best_val_ce=float(val_ce))

    def best_path(self):
        if not self.has_best:
            return None
        p = self._ckpt_path(self.best_step)
        return p if p.exists() and self._has_weights(p) else None

    def has_step(self, step: int) -> bool:
        """True when a usable weights checkpoint exists for this exact step."""
        path = self._ckpt_path(int(step))
        return path.exists() and self._has_weights(path)

    def read_meta(self, step: int) -> dict:
        try:
            return json.loads(
                (self._ckpt_path(int(step)) / "meta.json").read_text(
                    encoding="utf-8")
            )
        except Exception:
            return {}

    def _prune(self, current_step: int):
        from ..core.i18n import t

        # Use the in-memory pointer.  Re-reading best.json here used to race
        # with the just-written pointer in some runs and could keep an old
        # baseline checkpoint forever (wasting 6.9 GB per run).
        entries = []
        for child in self.dir.iterdir():
            m = re.fullmatch(r"step_(\d+)", child.name)
            if m and child.is_dir():
                entries.append((int(m.group(1)), child))
        entries.sort()
        last_keep = {s for s, _ in entries[-self.keep_last:]}
        best_step = self.best_step if self.has_best else -1
        for step, path in entries:
            keep = (
                step == current_step
                or step == best_step
                or step in last_keep
                or (self.keep_every and step > 0
                    and step % self.keep_every == 0)
            )
            if not keep:
                shutil.rmtree(path, ignore_errors=True)
                self.log(t("checkpoint_pruned", self.lang, path=path))

    # ------------------------------------------------------------------
    def load_step_inplace(self, model, step: int) -> bool:
        """Copy one checkpoint's tensors into ``model`` (low peak memory).

        Delta checkpoints are reconstructed as ``base + quantized delta``;
        legacy v8 full checkpoints remain readable.  This walks tensors one at
        a time, so peak extra memory is one tensor.
        """
        path = self._ckpt_path(int(step))
        delta = path / "delta.safetensors"
        if delta.exists():
            from ..core.i18n import t
            try:
                meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            manifest = meta.get("delta") or {}
            base = manifest.get("base_model") or self.base_model_source
            if not base:
                return False
            from ..models.delta import load_delta
            try:
                load_delta(model, delta, base_source=base, manifest=manifest)
            except Exception as e:
                self.log(t("delta_checkpoint_load_failed", self.lang,
                           err=repr(e)[:200]))
                return False
            try:
                from ..models.qlayer import invalidate_shadows
                invalidate_shadows(model)
            except Exception:
                pass
            return True
        files = sorted(path.glob("*.safetensors"))
        if not files:
            return False
        import torch
        from safetensors import safe_open
        target = model.state_dict()
        seen = set()
        with torch.no_grad():
            for f in files:
                with safe_open(str(f), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        if key not in target:
                            continue
                        src = sf.get_tensor(key)
                        dst = target[key]
                        if dst.shape != src.shape:
                            continue
                        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
                        seen.add(key)
        if seen:
            # The int8 shadows describe the old weights; keeping them would
            # make the first post-resume/restart forward use a stale model.
            try:
                from ..models.qlayer import invalidate_shadows
                invalidate_shadows(model)
            except Exception:
                pass
        return bool(seen)

    def read_best(self) -> dict:
        try:
            return json.loads(self._best_file().read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def latest(self):
        """Newest checkpoint directory (single-file and sharded weights)."""
        best = None
        if not self.dir.exists():
            return None
        for child in self.dir.iterdir():
            m = re.fullmatch(r"step_(\d+)", child.name)
            if not m or not child.is_dir():
                continue
            if not self._has_weights(child):
                continue
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, child)
        return best

    def load_best(self, model, optimizer=None):
        """Restore the pinned best-val checkpoint (v7 resume policy).

        Returns the restored step, including 0 for a pre-training baseline,
        or ``None`` when no usable best checkpoint exists.  The old 0-on-
        failure convention was ambiguous and made the trainer fall back to a
        later, regressed checkpoint when the best was the step-0 baseline.
        """
        if not self.has_best:
            return None
        step = int(self.best_step or 0)
        path = self._ckpt_path(step)
        if not path.exists() or not self._has_weights(path):
            return None
        if not self.load_step_inplace(model, step):
            return None
        if optimizer is not None and (path / "optimizer.pt").exists():
            import torch
            d = torch.load(path / "optimizer.pt", map_location="cpu")
            if hasattr(optimizer, "load_state_dict"):
                try:
                    optimizer.load_state_dict(d)
                except Exception:
                    optimizer.state = d.get("state", {})
                    optimizer._step_count = d.get("step_count", 0)
                    optimizer.lr = d.get("lr", optimizer.lr)
            else:
                optimizer.state = d.get("state", {})
                optimizer._step_count = d.get("step_count", 0)
                optimizer.lr = d.get("lr", optimizer.lr)
        return step

    def load(self, model, optimizer=None):
        found = self.latest()
        if found is None:
            return 0
        step, path = found
        # v7: copy tensors into the already-loaded model instead of loading a
        # second 1.7B model.  The old path peaked at ~2x model size and is a
        # likely cause of the reported checkpoint-loading failures on 16 GB.
        ok = self.load_step_inplace(model, step)
        if not ok:
            return 0
        if optimizer is not None and (path / "optimizer.pt").exists():
            import torch
            d = torch.load(path / "optimizer.pt", map_location="cpu")
            if hasattr(optimizer, "load_state_dict"):
                try:
                    optimizer.load_state_dict(d)
                except Exception:
                    optimizer.state = d.get("state", {})
                    optimizer._step_count = d.get("step_count", 0)
                    optimizer.lr = d.get("lr", optimizer.lr)
            else:
                optimizer.state = d.get("state", {})
                optimizer._step_count = d.get("step_count", 0)
                optimizer.lr = d.get("lr", optimizer.lr)
        return step
