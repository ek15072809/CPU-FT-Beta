"""Training configuration: everything the Web UI / CLI can set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..data.builder import BuildConfig
from ..data.formats import DatasetSpec
from ..train.ascot import AscotConfig
from .jsonutil import dumps as json_dumps
from .paths import PROJECT_ROOT, looks_like_path, strip_path_quotes


class ConfigError(ValueError):
    """A deterministic configuration error that must not be retried.

    The watchdog restarts a crashed training process.  Missing/empty dataset
    sources are configuration mistakes, not transient crashes, so they use
    this exception and a dedicated process exit code.
    """


PRESETS = {
    "speed": {
        "ascot": {
            "span_length": 112, "num_spans": 3, "sink_len": 16, "prefix_len": 48,
            "slice_size": 3, "mode": "global", "head_positions": 48,
            "slice_sampling": "balanced", "shadow_refresh_every": 4,
            "head_lr_scale": 0.10, "long_plan_tokens": 48,
        },
        "lr": 1e-5,
    },
    "balanced": {
        "ascot": {
            "span_length": 128, "num_spans": 4, "sink_len": 16, "prefix_len": 48,
            "slice_size": 4, "mode": "global", "head_positions": 64,
            "slice_sampling": "balanced", "shadow_refresh_every": 4,
            "head_lr_scale": 0.10, "long_plan_tokens": 64,
        },
        "lr": 5e-6,
    },
    "quality": {
        "ascot": {
            "span_length": 128, "num_spans": 8, "sink_len": 16, "prefix_len": 64,
            "slice_size": 4, "mode": "global", "head_positions": 96,
            "slice_sampling": "balanced", "shadow_refresh_every": 1,
            "head_lr_scale": 0.05, "long_plan_tokens": 64,
        },
        "lr": 2e-6,
    },
}


@dataclass
class TrainConfig:
    # model
    model_source: str = ""                # no default model name in the UI
    model_dtype: str = "auto"             # auto | bfloat16 | float16 | float32
    device: str = "auto"                  # auto | cpu | cuda | cuda:N | mps
    # data
    dataset_specs: list[dict] = field(default_factory=list)
    data: dict = field(default_factory=lambda: asdict(BuildConfig(mode="sft")))
    dataset_cache: str = ""               # name of a previously built cache
    val_tokens: int = 4096                # balanced validation tokens per evaluation (0 = all)
    # objective
    task: str = "sft"                     # sft | pt
    # ascot
    ascot: dict = field(default_factory=lambda: dict(PRESETS["balanced"]["ascot"]))
    # optimization
    lr: float = 5e-6
    weight_decay: float = 0.01
    scheduler: str = "cosine"             # cosine | linear | constant
    warmup_steps: int = 20
    total_steps: int = 0                  # 0 = follow stop conditions
    eval_every: int = 20
    seed: int = 0
    # v9: one unambiguous guard mode.  The legacy booleans are kept in the
    # serialized config for old configs, but normalize() makes them agree.
    guard_mode: str = "restart"           # restart | stop | off
    stop_on_val_regression: bool = True   # normalized from guard_mode
    val_patience: int = 1                 # bad evaluations before acting
    val_min_delta: float = 1e-3           # CE improvement smaller than this is noise
    export_best: bool = True              # export the best-val checkpoint, not the last
    auto_restart: bool = True             # normalized from guard_mode
    # v8: retry LR schedule.  "harmonic" is initial_lr / (retry+1):
    # 1/2, 1/3, 1/4, ...  "exponential" is initial_lr * decay**retry.
    restart_lr_mode: str = "harmonic"     # harmonic | exponential
    restart_lr_decay: float = 0.5         # exponential mode only
    max_restarts: int = 0                 # 0 = unlimited until the LR floor
    restart_lr_floor: float = 1e-8        # stop retrying below this LR
    train_probe_tokens: int = 512         # fixed full-forward training probe (0 = off)
    # gradient spike handling (skip / scale are mutually exclusive modes)
    grad_spike_mode: str = "off"          # off | skip | scale
    grad_spike_threshold_pct: float = 100.0   # trigger when norm > avg * (1 + pct/100)
    grad_spike_window: int = 20               # rolling window (steps)
    # early stop limits (0 = off)
    max_hours: float = 0.0                # stop after N hours of training
    max_epochs: int = 1                   # stop after N epochs (1 epoch = all train sequences)
    # logging (CUI)
    log_every: int = 1
    show_grad_norm: bool = True
    show_memory: bool = True
    lang: str = "ja"                      # ja | en
    # checkpoints
    checkpoint_every: int = 100
    checkpoint_keep_last: int = 2
    checkpoint_keep_every: int = 500      # milestone checkpoints (0 = off)
    checkpoint_optimizer: bool = False    # include 8-bit Adam states
    checkpoint_delta_dtype: str = "int8"  # int8 | bfloat16 | float16 | float32
    # export / plots
    auto_export: bool = True
    export_name: str = "finetuned"
    save_plots: bool = True               # write lr/loss/grad/CE PNGs
    # lifecycle
    resume: bool = True
    relaunch_ui: bool = True
    run_name: str = "run1"

    def normalize(self):
        """Make contradictory legacy guard settings impossible.

        ``guard_mode`` is the single source of truth.  A v8 state file could
        contain ``stop_on_val_regression=false`` and ``auto_restart=true``;
        that combination silently disabled the regression guard, so no restart
        or LR reduction ever happened.  Normalizing here (and again on the
        server) ensures the trainer always receives a consistent policy.
        """
        mode = str(self.guard_mode or "").strip().lower()
        if mode not in ("restart", "stop", "off"):
            mode = ("restart" if bool(self.auto_restart)
                    else "stop" if bool(self.stop_on_val_regression) else "off")
        self.guard_mode = mode
        if mode == "restart":
            self.stop_on_val_regression = True
            self.auto_restart = True
        elif mode == "stop":
            self.stop_on_val_regression = True
            self.auto_restart = False
        else:
            self.stop_on_val_regression = False
            self.auto_restart = False
        self.val_patience = max(1, int(self.val_patience or 1))
        self.max_restarts = max(0, int(self.max_restarts or 0))
        self.restart_lr_floor = max(0.0, float(self.restart_lr_floor or 0.0))
        self.restart_lr_decay = min(0.999, max(0.01, float(self.restart_lr_decay or 0.5)))
        self.restart_lr_mode = str(self.restart_lr_mode or "harmonic").lower()
        if self.restart_lr_mode not in ("harmonic", "exponential"):
            self.restart_lr_mode = "harmonic"
        self.checkpoint_delta_dtype = str(
            self.checkpoint_delta_dtype or "int8").lower()
        if self.checkpoint_delta_dtype not in (
                "int8", "bfloat16", "float16", "float32"):
            self.checkpoint_delta_dtype = "int8"
        self.device = str(self.device or "auto").lower()
        self.model_dtype = str(self.model_dtype or "auto").lower()
        if self.dataset_cache:
            self.dataset_cache = str(self.dataset_cache).strip()
            if self.dataset_cache.lower().endswith(".pt"):
                self.dataset_cache = self.dataset_cache[:-3]
        return self

    def apply_preset(self, preset: str):
        if preset in PRESETS:
            self.ascot.update(PRESETS[preset]["ascot"])
            self.lr = PRESETS[preset]["lr"]

    def to_json(self) -> str:
        return json_dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "TrainConfig":
        # Windows Notepad / PowerShell write UTF-8 with a BOM by default;
        # json.loads rejects the BOM instead of ignoring it.
        d = json.loads(str(text).lstrip("\ufeff"))
        cfg = TrainConfig()
        has_mode = "guard_mode" in d
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if not has_mode:
            # Derive a mode from the old pair before the booleans are
            # normalized.  This matters for v8 files that said "stop=false,
            # restart=true": the user's intent was clearly restart.
            if bool(cfg.auto_restart):
                cfg.guard_mode = "restart"
            elif bool(cfg.stop_on_val_regression):
                cfg.guard_mode = "stop"
            else:
                cfg.guard_mode = "off"
        cfg.normalize()
        cfg.validate_config_keys()
        return cfg

    def validate_config_keys(self) -> None:
        """Reject unknown nested settings deterministically.

        ``TrainConfig.from_json`` historically copied only known top-level
        fields and left ``data``/``ascot`` as opaque dictionaries.  A typo
        such as ``data.val_rati0`` was therefore silently ignored when an
        existing cache was selected, because ``build_config()`` was never
        called on the explicit-cache path.  Validate the nested dictionaries
        here so every entry point (train/resume/export/server/CLI ``--set``)
        reports a readable ``ConfigError`` before doing any work.
        """
        # Keep the v5 HTML aliases accepted; build_config() converts them.
        aliases = {
            "html_structure", "html_structure_min_chars",
            "html_structure_max_docs", "html_structure_tail_chars",
            "html_structure_prefix_tokens",
        }
        valid_data = {f.name for f in fields(BuildConfig)} | aliases
        unknown_data = sorted(set(self.data or {}) - valid_data)
        if unknown_data:
            raise ConfigError(
                "データ設定に不明な項目があります / unknown data setting(s): "
                + ", ".join(unknown_data)
            )
        valid_ascot = {f.name for f in fields(AscotConfig)}
        unknown_ascot = sorted(set(self.ascot or {}) - valid_ascot)
        if unknown_ascot:
            raise ConfigError(
                "ASCOT設定に不明な項目があります / unknown ASCOT setting(s): "
                + ", ".join(unknown_ascot)
            )

    def cleaned_dataset_specs(self) -> list[dict]:
        """Return non-empty, trimmed dataset specs.

        The Web UI can leave an extra blank row when the user adds a source
        and removes the text.  An empty source must never reach
        ``datasets.load_dataset`` (which raises a cryptic IndexError).
        """
        out: list[dict] = []
        for raw in self.dataset_specs or []:
            if not isinstance(raw, dict):
                continue
            spec = dict(raw)
            source = strip_path_quotes(spec.get("source") or "").strip()
            if not source:
                continue
            spec["source"] = source
            out.append(spec)
        return out

    def spec_objects(self) -> list[DatasetSpec]:
        return [DatasetSpec(**s) for s in self.cleaned_dataset_specs()]

    def validate_data_sources(self) -> None:
        """Fail before the model is loaded when data settings are invalid."""
        self.validate_config_keys()
        specs = self.spec_objects()
        if not specs:
            # A previously built cache can still be trained without sources.
            if self.dataset_cache:
                from .paths import DATASET_CACHE_DIR
                cache = Path(DATASET_CACHE_DIR) / f"{self.dataset_cache}.pt"
                if cache.exists():
                    return
                raise ConfigError(
                    "データセットキャッシュが見つかりません / "
                    f"dataset cache not found: {cache}"
                )
            raise ConfigError(
                "データセットソースが空です。データセットを指定してください / "
                "dataset source is empty; select a dataset"
            )

        local_exts = (
            ".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow",
            ".txt", ".md", ".yaml", ".yml",
        )
        for i, spec in enumerate(specs, 1):
            source = strip_path_quotes(spec.source).strip()
            if not source:
                raise ConfigError(
                    f"データセットソース #{i} が空です / "
                    f"dataset source #{i} is empty"
                )
            path = Path(source).expanduser()
            if not path.is_absolute() and (PROJECT_ROOT / path).exists():
                path = PROJECT_ROOT / path
            if path.exists():
                continue
            # A missing Windows/absolute path or a filename-looking source is
            # always a local path.  Bare HuggingFace ids (owner/name) remain
            # valid and are resolved later by datasets.
            if looks_like_path(source) or source.lower().endswith(local_exts):
                raise ConfigError(
                    f"データセットが見つかりません #{i}: {source} / "
                    f"dataset source not found: {source}"
                )

    def build_config(self) -> BuildConfig:
        d = dict(self.data)
        d["mode"] = self.task
        # Backward compatibility with the short-lived v5 HTML-only field names.
        # New code and configs use the generic long-document names.
        aliases = {
            "html_structure": "long_doc_structure",
            "html_structure_min_chars": "long_doc_min_chars",
            "html_structure_max_docs": "long_doc_max_docs",
            "html_structure_tail_chars": "long_doc_tail_chars",
            "html_structure_prefix_tokens": "long_doc_prefix_tokens",
        }
        for old_key, new_key in aliases.items():
            if old_key in d:
                if new_key not in d:
                    d[new_key] = d[old_key]
                d.pop(old_key, None)
        # Args from an older/newer UI or a hand-written --set must fail with a
        # readable configuration error instead of a retryable TypeError.
        valid = {f.name for f in fields(BuildConfig)}
        unknown = sorted(set(d) - valid)
        if unknown:
            raise ConfigError(
                "データ設定に不明な項目があります / unknown data setting(s): "
                + ", ".join(unknown)
            )
        # stored in the cache stats; the trainer refuses stale caches
        d["fingerprint"] = self.dataset_fingerprint()
        return BuildConfig(**d)

    def ascot_config(self) -> AscotConfig:
        valid = {f.name for f in fields(AscotConfig)}
        unknown = sorted(set(self.ascot or {}) - valid)
        if unknown:
            raise ConfigError(
                "ASCOT設定に不明な項目があります / unknown ASCOT setting(s): "
                + ", ".join(unknown)
            )
        acfg = AscotConfig(**self.ascot)
        # gradient-spike handling lives at the top level of TrainConfig but is
        # consumed by the stepper; forward it here so the two never diverge
        acfg.grad_spike_mode = self.grad_spike_mode
        acfg.grad_spike_threshold_pct = self.grad_spike_threshold_pct
        acfg.grad_spike_window = self.grad_spike_window
        return acfg

    def dataset_cache_name(self) -> str:
        """Stable cache name; includes examples_per_step when set.

        The default name embeds a fingerprint of the data sources and every
        build-relevant setting, so changing the dataset (or the packing
        options) never silently reuses an unrelated cache.  A user-supplied
        ``dataset_cache`` keeps its name; its fingerprint is still checked on
        load by the trainer.
        """
        base = self.dataset_cache or (
            f"{self.run_name}_{self.task}_{self.data.get('context_length', 32768)}"
        )
        eps = self.data.get("examples_per_step", 0)
        if eps and not self.dataset_cache:
            base += f"_eps{int(eps)}"
        if not self.dataset_cache:
            base += "_" + self.dataset_fingerprint()[:8]
        return base

    def dataset_fingerprint(self) -> str:
        """Hash of the data sources and every setting that changes packing."""
        keys = (
            "context_length", "examples_per_step", "val_ratio", "min_chars",
            "max_chars", "dedup", "seed", "max_sequences", "train_on_user",
            "add_generation_prompt", "join_text", "preview_examples",
            "dataset_order", "long_doc_structure", "long_doc_min_chars",
            "long_doc_min_sentences", "long_doc_max_docs",
            "long_doc_tail_chars", "long_doc_prefix_tokens",
        )
        payload = {
            "task": self.task,
            "specs": self.cleaned_dataset_specs(),
            "data": {k: self.data.get(k) for k in keys},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()
