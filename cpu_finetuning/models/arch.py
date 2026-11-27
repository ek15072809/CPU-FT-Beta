"""Architecture adapters.

Provides a uniform interface over the supported model families:

* ``lfm2``   - LiquidAI LFM2 / LFM2.5 (short-conv + GQA hybrid)
* ``lfm3``   - LiquidAI LFM3 (tri-hybrid: conv + sliding + full attention)
* ``qwen3_5``- Qwen3.5 (gated delta net + full attention)
* ``gemma4`` - Gemma 4 text stack (sliding/full attention, PLE, shared KV)

The adapter exposes:

* ``full_forward``     - a standard forward of the underlying HF model
  (used for the frozen int8 passes; supports cache continuation),
* ``snapshot`` / ``restore`` - copy recurrent (conv/SSM) + attention states
  and (for Gemma4) shared KV between passes,
* ``slice_forward``    - run only a contiguous range of decoder layers with
  gradients, starting from a detached hidden state and restored states.

Any model whose ``config.layer_types`` uses the same vocabulary works as well,
regardless of parameter count.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)


def _layer_types_of(config) -> list[str]:
    lts = getattr(config, "layer_types", None)
    if lts:
        return list(lts)
    # Qwen3.5-style: full_attention_interval
    interval = getattr(config, "full_attention_interval", None)
    n = config.num_hidden_layers
    if interval:
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(n)
        ]
    return ["full_attention"] * n


class ArchAdapter:
    """Base adapter; subclasses override the arch-specific pieces."""

    family = "generic"

    def __init__(self, lm_model: nn.Module, causal_lm: nn.Module):
        # lm_model: the decoder stack (e.g. Lfm2Model); causal_lm: full model with lm_head
        self.lm_model = lm_model
        self.causal_lm = causal_lm
        self.config = lm_model.config
        self.layers = list(lm_model.layers)
        self.layer_types = _layer_types_of(self.config)
        self.embed_tokens = lm_model.embed_tokens
        self.final_norm = self._find_final_norm()
        self.num_layers = len(self.layers)
        self.head = causal_lm.lm_head if hasattr(causal_lm, "lm_head") else None
        self.supports_shared_kv = False

    def _find_final_norm(self) -> nn.Module:
        for name in ("embedding_norm", "norm", "final_norm"):
            if hasattr(self.lm_model, name):
                return getattr(self.lm_model, name)
        raise AttributeError("final norm not found")

    # ------------------------------------------------------------------
    # full forward (pass 1)
    # ------------------------------------------------------------------
    def full_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        position_ids: torch.Tensor | None = None,
    ):
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        return self.lm_model(**kwargs)

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    # ------------------------------------------------------------------
    # state snapshot / restore
    # ------------------------------------------------------------------
    @torch.no_grad()
    def snapshot(self, cache) -> dict:
        snap: dict[str, Any] = {"layers": []}
        for layer in cache.layers:
            entry: dict[str, Any] = {}
            conv = getattr(layer, "conv_states", None)
            conv_init = getattr(layer, "is_conv_states_initialized", None)
            if isinstance(conv, dict):
                entry["conv"] = {
                    idx: (t.clone() if torch.is_tensor(t) else None)
                    for idx, t in conv.items()
                    if conv_init.get(idx) and torch.is_tensor(t)
                }
            rec = getattr(layer, "recurrent_states", None)
            rec_init = getattr(layer, "is_recurrent_states_initialized", None)
            if isinstance(rec, dict):
                entry["recurrent"] = {
                    idx: (t.clone() if torch.is_tensor(t) else None)
                    for idx, t in rec.items()
                    if rec_init.get(idx) and torch.is_tensor(t)
                }
            keys = getattr(layer, "keys", None)
            if keys is not None:
                entry["keys"] = keys.clone()
                entry["values"] = layer.values.clone()
            snap["layers"].append(entry)
        return snap

    def restore(self, snap: dict):
        """Build a fresh DynamicCache holding the snapshotted states.

        State updates are patched to *replace* tensors instead of the default
        in-place ``copy_`` so that backward passes through gradient-carrying
        layers never see a version bump on saved tensors.
        """
        cache = DynamicCache(config=self.config)
        self._patch_cache_for_autograd(cache)
        for i, (layer, entry) in enumerate(zip(cache.layers, snap["layers"])):
            for idx, t in entry.get("conv", {}).items():
                if t is not None:
                    layer.update_conv_state(t, state_idx=idx)
            for idx, t in entry.get("recurrent", {}).items():
                if t is not None:
                    layer.update_recurrent_state(t, state_idx=idx)
            if "keys" in entry:
                layer.update(entry["keys"], entry["values"], layer_idx=i)
        return cache

    @staticmethod
    def _patch_cache_for_autograd(cache) -> None:
        import types
        import torch.nn.functional as F

        for layer in cache.layers:
            if isinstance(getattr(layer, "conv_states", None), dict):

                def safe_conv(self, conv_states, state_idx=0, conv_kernel_size=None, **kw):
                    if not self.is_conv_states_initialized[state_idx]:
                        self.lazy_initialization(
                            conv_states=conv_states, state_idx=state_idx,
                            conv_kernel_size=conv_kernel_size,
                        )
                    kernel = self.conv_kernel_size[state_idx]
                    if not self.has_previous_state[state_idx]:
                        full = conv_states
                        self.has_previous_state[state_idx] = True
                        if not self.record_past and full.shape[-1] < kernel:
                            full = F.pad(full, (kernel - full.shape[-1], 0))
                    else:
                        full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
                    if not self.record_past:
                        self.conv_states[state_idx] = full[..., -kernel:].clone()
                    else:
                        self.conv_states[state_idx] = full
                    return full

                layer.update_conv_state = types.MethodType(safe_conv, layer)

            if isinstance(getattr(layer, "recurrent_states", None), dict):

                def safe_rec(self, recurrent_states, state_idx=0, **kw):
                    if not self.is_recurrent_states_initialized[state_idx]:
                        self.lazy_initialization(
                            recurrent_states=recurrent_states, state_idx=state_idx
                        )
                    self.recurrent_states[state_idx] = recurrent_states
                    return self.recurrent_states[state_idx]

                layer.update_recurrent_state = types.MethodType(safe_rec, layer)

    # ------------------------------------------------------------------
    # masks / rope
    # ------------------------------------------------------------------
    def _mask_kwargs(self, hidden: torch.Tensor, cache, position_ids):
        return {
            "config": self.config,
            "inputs_embeds": hidden,
            "attention_mask": None,
            "past_key_values": cache,
            "position_ids": position_ids,
        }

    def make_masks(self, hidden, cache, position_ids) -> dict:
        kwargs = self._mask_kwargs(hidden, cache, position_ids)
        needed = set(self.layer_types)
        masks: dict[str, Any] = {}
        if "full_attention" in needed:
            masks["full_attention"] = create_causal_mask(**kwargs)
        if "sliding_attention" in needed:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
        if "conv" in needed or "linear_attention" in needed:
            m = create_recurrent_attention_mask(**kwargs)
            masks["conv"] = m
            masks["linear_attention"] = m
        return masks

    def rope(self, hidden, position_ids, layer_type=None):
        if layer_type is None:
            return self.lm_model.rotary_emb(hidden, position_ids=position_ids)
        return self.lm_model.rotary_emb(hidden, position_ids=position_ids, layer_type=layer_type)

    # ------------------------------------------------------------------
    # slice forward (pass 2) - arch specific
    # ------------------------------------------------------------------
    def slice_forward(
        self,
        h_start: torch.Tensor,
        layer_range: tuple[int, int],
        position_ids: torch.Tensor,
        cache,
        extra: dict | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Lfm2Adapter(ArchAdapter):
    family = "lfm2"

    def slice_forward(self, h_start, layer_range, position_ids, cache, extra=None):
        masks = self.make_masks(h_start, cache, position_ids)
        pos_emb = self.rope(h_start, position_ids)
        h = h_start
        j0, j1 = layer_range
        for i in range(j0, j1):
            lt = self.layer_types[i]
            h = self.layers[i](
                h,
                attention_mask=masks[lt],
                position_embeddings=pos_emb,
                position_ids=position_ids,
                past_key_values=cache,
            )
        return h


class Lfm3Adapter(ArchAdapter):
    family = "lfm3"

    def slice_forward(self, h_start, layer_range, position_ids, cache, extra=None):
        masks = self.make_masks(h_start, cache, position_ids)
        rope_types = getattr(self.lm_model.rotary_emb, "layer_types", [None])
        pos_emb = {
            lt: self.rope(h_start, position_ids, layer_type=lt) for lt in rope_types
        }
        h = h_start
        j0, j1 = layer_range
        for i in range(j0, j1):
            lt = self.layer_types[i]
            h = self.layers[i](
                h,
                attention_mask=masks[lt],
                position_embeddings=pos_emb.get(lt),
                position_ids=position_ids,
                past_key_values=cache,
            )
        return h


class Qwen35Adapter(ArchAdapter):
    family = "qwen3_5"

    def slice_forward(self, h_start, layer_range, position_ids, cache, extra=None):
        masks = self.make_masks(h_start, cache, position_ids)
        pos_emb = self.rope(h_start, position_ids)
        h = h_start
        j0, j1 = layer_range
        for i in range(j0, j1):
            lt = self.layer_types[i]
            h = self.layers[i](
                h,
                position_embeddings=pos_emb,
                attention_mask=masks[lt],
                position_ids=position_ids,
                past_key_values=cache,
            )
        return h


class Gemma4Adapter(ArchAdapter):
    family = "gemma4"
    supports_shared_kv = True

    def __init__(self, lm_model, causal_lm):
        super().__init__(lm_model, causal_lm)
        self.supports_shared_kv = True
        self.hidden_size_per_layer_input = getattr(
            self.config, "hidden_size_per_layer_input", 0
        )

    @torch.no_grad()
    def compute_per_layer_inputs(self, input_ids, inputs_embeds):
        """PLE inputs (constant w.r.t. gradients); shape [B, S, L, d_ple]."""
        if not self.hidden_size_per_layer_input:
            return None
        pli = self.lm_model.get_per_layer_inputs(input_ids, inputs_embeds)
        return self.lm_model.project_per_layer_inputs(inputs_embeds, pli)

    def slice_forward(self, h_start, layer_range, position_ids, cache, extra=None):
        from transformers.models.gemma4.modeling_gemma4 import KVSharingDict

        extra = extra or {}
        masks = self.make_masks(h_start, cache, position_ids)
        rope_types = ["full_attention", "sliding_attention"]
        pos_emb = {lt: self.rope(h_start, position_ids, layer_type=lt) for lt in rope_types}
        shared_kv = KVSharingDict(copy.deepcopy(extra.get("shared_kv", {})))
        per_layer_inputs = extra.get("per_layer_inputs")
        h = h_start
        j0, j1 = layer_range
        for i in range(j0, j1):
            lt = self.layer_types[i]
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            h = self.layers[i](
                h,
                pli,
                shared_kv_states=shared_kv,
                position_embeddings=pos_emb[lt],
                attention_mask=masks[lt],
                position_ids=position_ids,
                past_key_values=cache,
            )
        return h

    def full_forward(self, input_ids, past_key_values=None, use_cache=True,
                     output_hidden_states=False, position_ids=None):
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
            "return_shared_kv_states": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        return self.lm_model(**kwargs)

    def snapshot(self, cache) -> dict:
        snap = super().snapshot(cache)
        return snap

    def snapshot_shared_kv(self, out) -> dict:
        skv = getattr(out, "shared_kv_states", None)
        if not skv:
            return {}
        return {k: tuple(t.clone() for t in v) for k, v in skv.items()}


_ADAPTERS = {
    "lfm2": Lfm2Adapter,
    "lfm3": Lfm3Adapter,
    "qwen3_5": Qwen35Adapter,
    "qwen3_5_text": Qwen35Adapter,
    "gemma4": Gemma4Adapter,
    "gemma4_text": Gemma4Adapter,
}


def build_adapter(causal_lm: nn.Module) -> ArchAdapter:
    """Create the adapter for a (text) causal LM, detecting the family."""
    cfg = causal_lm.config
    model_type = getattr(cfg, "model_type", "")
    # multimodal wrappers: extract the text stack
    lm_model = None
    for path in ("model", "language_model", "model.model"):
        obj = causal_lm
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
            lm_model = obj
            break
    if lm_model is None:
        raise ValueError(f"Unsupported model structure: {type(causal_lm).__name__}")
    text_cfg = lm_model.config
    mt = getattr(text_cfg, "model_type", model_type)
    adapter_cls = _ADAPTERS.get(mt)
    if adapter_cls is None:
        # generic fallback: assume the transformers-standard layer interface
        adapter_cls = Lfm2Adapter if any(
            hasattr(l, "conv") for l in lm_model.layers
        ) else Qwen35Adapter
    return adapter_cls(lm_model, causal_lm)
