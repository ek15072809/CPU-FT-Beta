"""Model loading: HuggingFace Hub or local folder, with project-local cache.

Remote-code handling (v3, fixes the Windows ``signal.SIGALRM`` crash):

Some repositories (e.g. LFM3) ship an ``auto_map`` in their config.json.
Depending on the installed transformers version, ``AutoConfig`` /
``AutoModelForCausalLM`` then demand ``trust_remote_code=True`` - and on
Windows the interactive check crashes with
``AttributeError: module 'signal' has no attribute 'SIGALRM'`` before raising
``ValueError: ... Please pass the argument trust_remote_code=True``.

The v2 resolver only handled the case where transformers *natively* knows the
architecture.  On machines with an older transformers (installed before the
architecture was upstreamed) the native class does not exist and loading
failed with exactly the error above.  v3 therefore resolves configs in three
stages:

1. plain ``AutoConfig`` with ``trust_remote_code=False`` (never prompts; works
   whenever transformers natively supports the model - regardless of
   ``auto_map``),
2. read the raw config.json, strip ``auto_map`` and instantiate the *native*
   config class for the model_type (no code execution; works even when the
   auto_map would normally trigger a prompt),
3. last resort: ``trust_remote_code=True`` (execute the repository's custom
   code, exactly what the transformers error message asks for).  The env var
   ``CPU_FT_TRUST_REMOTE_CODE=0`` disables this stage for locked-down
   machines.

Stage 3 keeps ``auto_map`` intact so the model/tokenizer load through the
same code path, and ``LoadedModel.loaded_via_remote_code`` tells checkpoints
and exports to preserve (and ship) the repository code files.

v12: stage 1 explicitly disables remote code.  On an interactive terminal
(Colab/Linux) ``trust_remote_code=None`` can prompt, execute the repository's
configuration code and return a *remote* config.  That config was mistaken for
a native one, stripped of ``auto_map`` and passed to
``AutoModelForCausalLM``, which then failed with
``Unrecognized configuration class``.  Stage 1 now always raises instead of
prompting, and remote models are loaded through their ``auto_map`` without
passing the pre-resolved config.

Local paths may be quoted (Windows "Copy as path" produces
``"C:\\models\\my-model"``); quotes are stripped automatically.

Nothing is ever written outside the project directory: HF_HOME and friends
are redirected by ``cpu_finetuning.core.paths`` before transformers is
imported.
"""

# Keep this statement immediately after the module docstring.  A duplicate
# future import inserted later in the file is a SyntaxError on every Python
# version (reported on Linux in v10).
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ..core.paths import looks_like_path, strip_path_quotes
from .qlayer import install_shadows, shadow_report, clear_shadows
from .arch import ArchAdapter, build_adapter


def _remote_code_allowed() -> bool:
    return os.environ.get("CPU_FT_TRUST_REMOTE_CODE", "1") != "0"


def total_ram_bytes() -> int:
    """Physical RAM of this machine (0 when unknown)."""
    try:
        if os.name == "nt":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        else:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        pass
    return 0


def estimate_params(source: str, cfg=None) -> int:
    """Approximate parameter count without loading the weights."""
    p = Path(source)
    if p.is_dir():
        total = 0
        for pattern in ("*.safetensors", "*.bin"):
            for f in p.glob(pattern):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
            if total:
                break
        if total:
            dtype = ""
            try:
                dtype = str(getattr(cfg, "dtype", "") or getattr(cfg, "torch_dtype", "") or "")
            except Exception:
                pass
            item = 2 if ("float16" in dtype or "bfloat16" in dtype or "half" in dtype) else 4
            return max(1, total // item)
    try:
        text_cfg = getattr(cfg, "text_config", None) or cfg
        h = int(getattr(text_cfg, "hidden_size", 2048))
        inter = int(getattr(text_cfg, "intermediate_size", h * 4))
        vocab = int(getattr(text_cfg, "vocab_size", 128000))
        layers = int(getattr(text_cfg, "num_hidden_layers", 24))
        return int(layers * (4 * h * h + 3 * h * inter) + vocab * h)
    except Exception:
        return 2_000_000_000


def resolve_compute_dtype(dtype, source: str, cfg=None,
                          device: str = "cpu") -> tuple[torch.dtype, str]:
    """Resolve ``"auto"`` into a dtype that fits the selected device.

    On a CPU without native bf16 GEMM (e.g. i7-1360P), fp32 matmuls are about
    4-5x faster than bf16 matmuls.  On CUDA, fp32 is the safe default for T4
    (16 GB); Ampere+ can use bf16 automatically.  fp16 is available explicitly
    for users who have measured it to be stable on their model.
    """
    if isinstance(dtype, torch.dtype):
        return dtype, "explicit"
    name = str(dtype or "auto").lower()
    if name in ("float32", "fp32", "float"):
        return torch.float32, "explicit"
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16, "explicit"
    if name in ("float16", "fp16", "half"):
        return torch.float16, "explicit"
    if str(device or "cpu").startswith("cuda"):
        try:
            import torch as _torch
            if _torch.cuda.is_bf16_supported():
                return torch.bfloat16, "auto: CUDA supports bf16"
        except Exception:
            pass
        return torch.float32, (
            "auto: CUDA fp32 (safe default for 16 GB T4; "
            "set model_dtype=float16 to opt in)"
        )
    if str(device or "cpu").startswith("mps"):
        # MPS has narrower dtype/op coverage than CUDA.  fp32 is slower but is
        # the only choice that keeps every LFM/Qwen/Gemma layer on the stable
        # path; explicit bfloat16/float16 still remain available.
        return torch.float32, "auto: MPS fp32 (widest operator support)"
    params = estimate_params(source, cfg)
    ram = total_ram_bytes()
    fp32_need = params * 4 + params * 2 + int(2.5e9)
    bf16_need = params * 2 + params * 2 + int(2.5e9)
    if ram and fp32_need <= ram * 0.85:
        return torch.float32, (f"auto: {params/1e9:.2f}B params, "
                               f"fp32 needs ~{fp32_need/1e9:.1f}GB of "
                               f"{ram/1e9:.1f}GB RAM")
    if ram and bf16_need <= ram * 0.95:
        return torch.bfloat16, (f"auto: {params/1e9:.2f}B params, "
                                f"fp32 needs ~{fp32_need/1e9:.1f}GB > "
                                f"{ram*0.85/1e9:.1f}GB budget -> bf16")
    return torch.bfloat16, "auto: memory unknown/tight -> bf16"


def _is_remote_code_error(e: BaseException) -> bool:
    msg = str(e)
    return ("trust_remote_code" in msg
            or "remote code" in msg.lower()
            or "SIGALRM" in msg)


def _is_unknown_arch_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("does not recognize this architecture" in msg
            or "not a recognised model type" in msg
            or "unrecognized model type" in msg
            or "unrecognized configuration class" in msg)


# re-exported so every call site shares one implementation
_strip_quotes = strip_path_quotes
_looks_like_path = looks_like_path


# native config classes for every supported architecture family
def _native_config_classes() -> dict:
    classes = {}
    try:
        from transformers import Lfm2Config
        classes["lfm2"] = Lfm2Config
    except ImportError:
        pass
    try:
        from transformers import Lfm3Config
        classes["lfm3"] = Lfm3Config
    except ImportError:
        pass
    try:
        from transformers import Gemma4Config
        classes["gemma4"] = Gemma4Config
    except ImportError:
        pass
    try:
        from transformers import Qwen3_5Config
        classes["qwen3_5"] = Qwen3_5Config
    except ImportError:
        pass
    return classes


def _read_config_dict(source: str) -> dict:
    import json
    p = Path(source)
    if p.exists() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    # from the hub (project-local cache)
    from huggingface_hub import hf_hub_download
    f = hf_hub_download(source, "config.json")
    return json.loads(Path(f).read_text(encoding="utf-8"))


def _strip_auto_map(cfg):
    """Remove every auto_map attribute (root and nested text config)."""
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is None:
            continue
        if "auto_map" in getattr(obj, "__dict__", {}):
            try:
                del obj.__dict__["auto_map"]
            except Exception:
                setattr(obj, "auto_map", None)
        elif hasattr(obj, "auto_map"):
            setattr(obj, "auto_map", None)
    return cfg


def resolve_config(source: str):
    """Load a model config without prompting, falling back to remote code.

    Stage 1: AutoConfig with ``trust_remote_code=False`` (native support
             wins over auto_map and remote code is never prompted/executed).
    Stage 2: native config class built from the raw config.json with
             ``auto_map`` stripped (no code execution).
    Stage 3: ``trust_remote_code=True`` (last resort; can be disabled with
             ``CPU_FT_TRUST_REMOTE_CODE=0``).  The returned config keeps its
             ``auto_map`` and is flagged with ``_cpuft_remote_code = True``.
    """
    from transformers import AutoConfig

    source = _resolve(source)
    # ---- stage 1: plain path (never prompt / never execute remote code)
    try:
        # trust_remote_code=False is essential: with the default None an
        # interactive terminal (Colab/Linux) prompts, runs the repository's
        # configuration code and returns a remote config.  Stripping its
        # auto_map made AutoModelForCausalLM report "Unrecognized
        # configuration class" instead of loading the custom model.
        cfg = AutoConfig.from_pretrained(source, trust_remote_code=False)
        return _strip_auto_map(cfg)
    except Exception as e:
        if not (_is_remote_code_error(e) or _is_unknown_arch_error(e)):
            raise
        first_error = e

    # ---- stage 2: native class, auto_map stripped
    try:
        d = _read_config_dict(source)
        mt = d.get("model_type")
        native = _native_config_classes().get(mt)
        if native is not None:
            d = {k: v for k, v in d.items() if k != "auto_map"}
            cfg = native.from_dict(d)
            return _strip_auto_map(cfg)
    except Exception:
        pass

    # ---- stage 3: trust the repository's custom code (v3)
    if not _remote_code_allowed():
        raise first_error
    cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
    try:
        cfg._cpuft_remote_code = True   # marker for load_causal_lm / export
    except Exception:
        pass
    return cfg


def load_tokenizer(source: str):
    """Load a tokenizer (or processor) with the same remote-code fallback."""
    from transformers import AutoTokenizer

    source = _resolve(source)
    first_error = None
    # Try native first with remote code disabled so a custom tokenizer can
    # never trigger the interactive prompt; the second pass is explicit opt-in.
    for kwargs in ({"trust_remote_code": False},
                   {"trust_remote_code": True}):
        try:
            return AutoTokenizer.from_pretrained(source, **kwargs)
        except Exception as e:
            if first_error is None:
                first_error = e
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(source, trust_remote_code=True)
    except Exception:
        raise first_error


def load_causal_lm(source: str, dtype=torch.bfloat16):
    """Load a causal LM with the remote-code-safe config path.

    Remote-code repositories must be loaded by the auto factory from their
    ``config.json`` / ``auto_map``.  Passing the pre-resolved config object
    bypasses that lookup (and is fatal if the object ever lost its
    ``auto_map``), so the remote path deliberately omits ``config=``.
    """
    from transformers import AutoModelForCausalLM

    source = _resolve(source)
    cfg = resolve_config(source)
    remote = bool(getattr(cfg, "_cpuft_remote_code", False))
    if remote:
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                source, dtype=dtype, low_cpu_mem_usage=True, config=cfg,
            )
        except Exception as e:
            # the config resolved natively but the weights still need the
            # repo's custom modeling code (older transformers without the
            # architecture)
            if not (_is_remote_code_error(e) or _is_unknown_arch_error(e)):
                raise
            model = AutoModelForCausalLM.from_pretrained(
                source, dtype=dtype, low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            remote = True
    try:
        model._cpuft_remote_code = remote
        model._cpuft_source_dir = source if Path(source).is_dir() else ""
    except Exception:
        pass
    return model


def persist_remote_code(model, out_dir) -> bool:
    """Ship repository code files next to a saved model when needed.

    When a model was loaded through the repository's custom code (stage 3),
    checkpoints and exports must keep their ``auto_map`` AND contain the
    repository's ``*.py`` files, otherwise they cannot be reloaded on a
    machine whose transformers does not (yet) know the architecture.

    A Hub model has no local source directory in ``_cpuft_source_dir``.  In
    that case look in the already-populated local Hugging Face cache (never
    trigger a new network download during a checkpoint save).  If no code can
    be found, return False so the caller strips ``auto_map`` instead of
    writing a checkpoint that points at missing modules.
    """
    import shutil

    out_dir = Path(out_dir)
    if not getattr(model, "_cpuft_remote_code", False):
        return False

    candidates: list[str] = []
    src = str(getattr(model, "_cpuft_source_dir", "") or "")
    if src:
        candidates.append(src)
    cfg = getattr(model, "config", None)
    for attr in ("_name_or_path", "name_or_path"):
        value = str(getattr(cfg, attr, "") or "")
        if value:
            candidates.append(value)

    source_dir = next((c for c in candidates if Path(c).is_dir()), None)
    if source_dir is None:
        # A Hub model: reuse the snapshot transformers already downloaded.
        try:
            from huggingface_hub import snapshot_download
            for repo_id in candidates:
                if not repo_id or Path(repo_id).exists():
                    continue
                source_dir = snapshot_download(
                    repo_id=repo_id, allow_patterns=["*.py"],
                    local_files_only=True,
                )
                if Path(source_dir).is_dir():
                    break
        except Exception:
            source_dir = None

    copied = 0
    if source_dir and Path(source_dir).is_dir():
        for f in Path(source_dir).glob("*.py"):
            try:
                shutil.copy2(f, out_dir / f.name)
                copied += 1
            except Exception:
                pass
    # If the checkpoint/export already contains the code files, it is safe to
    # keep auto_map even when this particular model object has no source path.
    already = list(out_dir.glob("*.py"))
    return bool(copied or already)


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    adapter: ArchAdapter
    source: str
    master_bytes: int
    shadow_bytes: int
    is_multimodal_wrapper: bool
    loaded_via_remote_code: bool = False
    compute_dtype: str = ""
    dtype_reason: str = ""
    int8_backend: dict | None = None


def _resolve(source: str) -> str:
    """Accept an HF repo id, an export/run name, or a local quoted path.

    v5: bare run names such as ``run1`` are resolved to their exported model
    instead of being sent to the Hub as a repository id.  This was the cause
    of the generation-test retry storm against ``huggingface.co/run1``.
    """
    from ..core.usage import resolve_model_source
    try:
        return resolve_model_source(source)
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception:
        # extremely defensive fallback: keep the old behaviour for exotic
        # strings, but still reject obviously local paths that do not exist
        s = _strip_quotes(source)
        if _looks_like_path(s) and not Path(s).exists():
            raise FileNotFoundError(
                f"local path not found: {s} "
                "(surrounding quotes are removed automatically; check the path)"
            )
        p = Path(s)
        if p.exists() and (p / "config.json").exists():
            return str(p.resolve())
        return s


def load_model(
    source: str,
    dtype: torch.dtype | str = "auto",
    device: str = "cpu",
) -> LoadedModel:
    # a fresh model load invalidates every cached int8 shadow: memory
    # addresses may be reused by the allocator, which would silently map an
    # old shadow onto a new weight tensor
    clear_shadows()

    source = _resolve(source)
    cfg = resolve_config(source)
    remote = bool(getattr(cfg, "_cpuft_remote_code", False))
    is_wrapper = hasattr(cfg, "text_config") and getattr(cfg, "text_config", None) is not None

    dtype, dtype_reason = resolve_compute_dtype(
        dtype, source, cfg, device=device)
    model = load_causal_lm(source, dtype=dtype)
    remote = remote or bool(getattr(model, "_cpuft_remote_code", False))
    model = model.to(device)
    model.eval()
    if not remote:
        _strip_auto_map(model.config)  # never persist auto_map into checkpoints
    try:
        model._cpuft_remote_code = remote
    except Exception:
        pass

    tokenizer = load_tokenizer(source)

    install_shadows(model)
    from .qlayer import backend_info
    int8_info = backend_info()
    adapter = build_adapter(model)

    master_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        source=source,
        master_bytes=master_bytes,
        shadow_bytes=0,
        is_multimodal_wrapper=is_wrapper,
        loaded_via_remote_code=remote,
        compute_dtype=str(dtype).replace("torch.", ""),
        dtype_reason=dtype_reason,
        int8_backend=int8_info,
    )


def shadow_budget_report(lm: LoadedModel) -> dict:
    """Verify the derived int8 caches stay below the master weights."""
    rep = shadow_report()
    from ..core.paths import human_size
    rep["master_bytes"] = lm.master_bytes
    rep["master_human"] = human_size(lm.master_bytes)
    rep["shadow_human"] = human_size(rep["shadow_bytes"])
    rep["ratio"] = round(rep["shadow_bytes"] / max(1, lm.master_bytes), 3)
    rep["within_budget"] = rep["shadow_bytes"] <= lm.master_bytes
    return rep
