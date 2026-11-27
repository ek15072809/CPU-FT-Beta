"""Device and compute-dtype selection.

CPU remains the default fallback.  When a CUDA (or Apple MPS) device is
available, ``auto`` selects it, while an explicit ``cpu`` keeps the existing
CPU-only behaviour.  This module deliberately has no UI/CLI naming that ties
it to a particular hosted notebook service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceChoice:
    device: str
    kind: str
    label: str
    cuda_index: int = 0


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def resolve_device(requested: str = "auto", index: int | None = None) -> DeviceChoice:
    """Resolve ``auto|cpu|cuda|cuda:N|mps`` into a concrete device.

    ``auto`` uses CUDA first, then MPS, then CPU.  Requesting an unavailable
    accelerator falls back to CPU instead of failing a long run.
    """
    req = str(requested or "auto").strip().lower()
    if index is not None:
        req = f"cuda:{int(index)}"
    if req.startswith("cuda"):
        if _cuda_available():
            idx = 0
            if ":" in req:
                try:
                    idx = int(req.split(":", 1)[1])
                except ValueError:
                    idx = 0
            try:
                import torch
                count = int(torch.cuda.device_count())
                if count > 0 and (idx < 0 or idx >= count):
                    idx = 0
                name = torch.cuda.get_device_name(idx)
            except Exception:
                idx = max(0, idx)
                name = "CUDA GPU"
            return DeviceChoice(f"cuda:{idx}", "cuda",
                                f"{name} (cuda:{idx})", idx)
        return DeviceChoice("cpu", "cpu", "CPU (CUDA not available)")
    if req == "mps":
        if _mps_available():
            return DeviceChoice("mps", "mps", "Apple MPS")
        return DeviceChoice("cpu", "cpu", "CPU (MPS not available)")
    if req == "auto":
        if _cuda_available():
            try:
                import torch
                idx = int(os.environ.get("CPUFT_CUDA_INDEX", "0") or 0)
                count = int(torch.cuda.device_count())
                if count > 0 and (idx < 0 or idx >= count):
                    idx = 0
                name = torch.cuda.get_device_name(idx)
            except Exception:
                idx = 0
                name = "CUDA GPU"
            return DeviceChoice(f"cuda:{idx}", "cuda",
                                f"{name} (cuda:{idx})", idx)
        if _mps_available():
            return DeviceChoice("mps", "mps", "Apple MPS")
        return DeviceChoice("cpu", "cpu", "CPU")
    return DeviceChoice("cpu", "cpu", "CPU")


def numpy_or_torch_device(choice: DeviceChoice):
    import torch
    return torch.device(choice.device)


def preferred_model_dtype(requested: str = "auto", device: str | None = None) -> str:
    """Return a concrete model dtype for the selected accelerator.

    The loader accepts ``auto``, ``bfloat16``, ``float16`` and ``float32``.
    On CUDA, auto prefers bf16 only where the GPU reports native support;
    otherwise fp32 is the safe default (explicit float16 remains available).
    CPU auto remains fp32 (the existing, verified path).
    """
    req = str(requested or "auto").lower()
    if req != "auto":
        return req
    if str(device or "").startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
        except Exception:
            pass
        return "float32"
    return "float32"


def memory_hygiene(device: str | None = None) -> None:
    """Release accelerator cache after rollback/restart without touching CPU."""
    if str(device or "").startswith("cuda"):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
