"""Quantized shadow layers for frozen forward passes.

The ASCOT method performs two kinds of compute per step:

* "shadow" forward passes over pre-roll and anchor tokens with the *frozen*
  model.  These run under ``torch.no_grad()`` and use dynamic int8 matmuls
  (``torch._int_mm``) against a cached int8 copy of each weight matrix.
* gradient passes through a small trainable slice of layers, which use the
  full-precision master parameters with normal autograd.

``install_shadows`` converts every ``nn.Linear`` / ``nn.Embedding`` inside a
model into ``QLinear`` / ``QEmbedding`` in place (parameters are kept), so the
same module automatically selects the fast int8 path when gradients are
disabled and the exact master path when autograd is active.  Derived int8
copies are *caches*: they are always smaller than the source weights and are
 invalidated whenever the master parameters change.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

_SHADOW_REGISTRY: dict[tuple[int, str], dict] = {}
_SHADOW_ENABLED = True
# Result of the one-time backend probe (None = not probed yet).
_INT8_BACKEND: dict | None = None


def shadow_enabled() -> bool:
    return _SHADOW_ENABLED


def _probe_int8_backend(dtype: torch.dtype = torch.float32) -> dict:
    """Measure ``_int_mm`` against the plain matmul on THIS machine.

    ``torch._int_mm`` is extremely fast on some builds and pathologically
    slow on others (Windows CPU builds of torch 2.6 dispatch it to a reference
    kernel: ~3.8 GFLOPS vs ~400 GFLOPS for fp32).  Blindly using the int8
    shadow path then makes every frozen forward ~100x slower - the reported
    "stuck at dataset construction for hours" symptom (the next phase is the
    initial validation).  This probe disables the int8 path when it is not
    actually beneficial.
    """
    global _INT8_BACKEND
    if _INT8_BACKEND is not None:
        return _INT8_BACKEND
    info: dict = {"enabled": False, "reason": "probe failed", "int8_ms": 0.0,
                  "ref_ms": 0.0, "speedup": 0.0}
    try:
        m, k, n = 128, 1024, 1024
        x = torch.randn(m, k)
        w = torch.randn(n, k)
        if dtype == torch.bfloat16:
            xr, wr = x.bfloat16(), w.bfloat16()
        else:
            xr, wr = x, w
        xq = torch.randint(-127, 127, (m, k), dtype=torch.int8)
        wq = torch.randint(-127, 127, (k, n), dtype=torch.int8)

        def _timed(fn, repeat=2):
            fn()
            t = time.perf_counter()
            for _ in range(repeat):
                fn()
            return (time.perf_counter() - t) / repeat

        t_int8 = _timed(lambda: torch._int_mm(xq, wq))
        t_ref = _timed(lambda: F.linear(xr, wr))
        speedup = t_ref / max(t_int8, 1e-9)
        info.update(
            int8_ms=round(t_int8 * 1000, 3),
            ref_ms=round(t_ref * 1000, 3),
            speedup=round(speedup, 2),
            enabled=bool(speedup >= 1.15),
            reason=("int8 is beneficial" if speedup >= 1.15
                    else "torch._int_mm is slower than the plain matmul"),
        )
    except Exception as e:  # unsupported op / OOM / exotic CPU
        info["reason"] = f"int8 probe unavailable: {e!r}"[:200]
    _INT8_BACKEND = info
    return info


def backend_info() -> dict:
    return dict(_INT8_BACKEND or {})


class _LinearShadow:
    __slots__ = ("wq_t", "scale", "stale")

    def __init__(self, weight: torch.Tensor):
        # per-output-channel symmetric int8 quantization of weight [N, K].
        # Chunked to bound the transient fp32 copy on large matrices.
        n = weight.shape[0]
        scale = torch.empty(n, dtype=torch.float32)
        wq_t = torch.empty(weight.shape[1], n, dtype=torch.int8)
        chunk = max(1, (1 << 22) // max(1, weight.shape[1]))  # ~4M elems/step
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            w = weight[s:e].detach().float()
            amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale[s:e] = (amax / 127.0).squeeze(1)
            q = torch.round(w / amax * 127.0).clamp_(-127, 127).to(torch.int8)
            wq_t[:, s:e] = q.t()
        self.scale = scale.contiguous()                        # [N] fp32
        self.wq_t = wq_t.contiguous()                          # [K, N] int8
        self.stale = False

    def bytes(self) -> int:
        return self.wq_t.numel() + self.scale.numel() * 4


class _EmbedShadow:
    __slots__ = ("wq", "scale", "stale")

    def __init__(self, weight: torch.Tensor):
        v = weight.shape[0]
        scale = torch.empty(v, dtype=torch.float32)
        wq = torch.empty_like(weight, dtype=torch.int8)
        chunk = max(1, (1 << 22) // max(1, weight.shape[1]))
        for s in range(0, v, chunk):
            e = min(s + chunk, v)
            w = weight[s:e].detach().float()
            amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale[s:e] = (amax / 127.0).squeeze(1)
            wq[s:e] = torch.round(w / amax * 127.0).clamp_(-127, 127).to(torch.int8)
        self.scale = scale                                        # [V] fp32
        self.wq = wq                                              # [V, D] int8
        self.stale = False

    def bytes(self) -> int:
        return self.wq.numel() + self.scale.numel() * 4


def _get_linear_shadow(weight: torch.Tensor) -> _LinearShadow:
    key = (weight.data_ptr(), "linear")
    sh = _SHADOW_REGISTRY.get(key)
    if sh is None or sh["obj"].stale:
        sh = {"obj": _LinearShadow(weight)}
        _SHADOW_REGISTRY[key] = sh
    return sh["obj"]


def _get_embed_shadow(weight: torch.Tensor) -> _EmbedShadow:
    key = (weight.data_ptr(), "embed")
    sh = _SHADOW_REGISTRY.get(key)
    if sh is None or sh["obj"].stale:
        sh = {"obj": _EmbedShadow(weight)}
        _SHADOW_REGISTRY[key] = sh
    return sh["obj"]


def _int8_matmul(x: torch.Tensor, shadow: _LinearShadow) -> torch.Tensor:
    """Dynamic int8 linear: x [.., K] -> [.., N]."""
    in_dtype = x.dtype
    xf = x.reshape(-1, x.shape[-1]).float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    sx = amax / 127.0
    xq = torch.round(xf / sx).clamp_(-127, 127).to(torch.int8)
    out = torch._int_mm(xq, shadow.wq_t).float()
    out = out * sx * shadow.scale
    return out.reshape(*x.shape[:-1], -1).to(in_dtype)


class QLinear(nn.Linear):
    """nn.Linear with a cached int8 shadow used when gradients are off."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _SHADOW_ENABLED and not torch.is_grad_enabled() and not self.training:
            return _int8_matmul(x, _get_linear_shadow(self.weight))
        return F.linear(x, self.weight, self.bias)


class QEmbedding(nn.Embedding):
    """nn.Embedding with a cached int8 shadow used when gradients are off."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if _SHADOW_ENABLED and not torch.is_grad_enabled() and not self.training:
            shadow = _get_embed_shadow(self.weight)
            ids = input_ids.reshape(-1)
            rows = shadow.wq.index_select(0, ids).float()          # [M, D]
            scales = shadow.scale.index_select(0, ids).unsqueeze(-1)
            out = rows * scales
            return out.reshape(*input_ids.shape, self.embedding_dim).to(self.weight.dtype)
        return F.embedding(
            input_ids, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )


def install_shadows(model: nn.Module) -> None:
    """Convert Linear/Embedding modules in-place to their quantized variants.

    The conversion is skipped automatically when the one-time benchmark shows
    that ``torch._int_mm`` is not faster than the plain matmul on this CPU.
    Skipping leaves the ordinary ``nn.Linear``/``nn.Embedding`` modules in
    place, so the exact same code runs at full speed.
    """
    params = list(model.parameters())
    dtype = params[0].dtype if params else torch.float32
    if any(p.device.type != "cpu" for p in params):
        global _INT8_BACKEND
        _INT8_BACKEND = {
            "enabled": False,
            "reason": "GPU device: CPU int8 shadow path disabled",
            "int8_ms": 0.0, "ref_ms": 0.0, "speedup": 0.0,
        }
        return
    info = _probe_int8_backend(dtype)
    if not info.get("enabled"):
        return
    for _, mod in model.named_modules():
        if mod.__class__ is nn.Linear:
            mod.__class__ = QLinear
        elif mod.__class__ is nn.Embedding:
            mod.__class__ = QEmbedding


def invalidate_shadows(model: nn.Module) -> None:
    """Mark shadows of every parameter inside ``model`` as stale (rebuild lazily)."""
    seen = set()
    for _, p in model.named_parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        for kind in ("linear", "embed"):
            sh = _SHADOW_REGISTRY.get((ptr, kind))
            if sh is not None:
                sh["obj"].stale = True


def clear_shadows() -> None:
    _SHADOW_REGISTRY.clear()


def shadow_report() -> dict:
    """Report size of derived int8 caches (must stay below the master size)."""
    total = 0
    entries = 0
    for sh in _SHADOW_REGISTRY.values():
        total += sh["obj"].bytes()
        entries += 1
    return {"shadow_entries": entries, "shadow_bytes": total}


def set_shadow_enabled(flag: bool) -> None:
    global _SHADOW_ENABLED
    _SHADOW_ENABLED = flag
