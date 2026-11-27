"""Command line interface.

Every training setting can be supplied without starting the Web UI:
either edit a JSON config (``--config``), use the explicit flags below, or
pass any nested field with ``--set PATH=VALUE`` (for example
``--set data.val_ratio=0.03 --set ascot.num_spans=6``).

Commands:
  serve                start the local Web UI (opens the browser)
  train                supervised training with watchdog + CUI output
  _train_child         internal: one training attempt (used by watchdog)
  resume               train with resume forced
  export               export the best/newest checkpoint
  info                 environment and cache report
  generate             quick generation test with an exported model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Explicit flags for the frequently used top-level settings.  ``--set`` covers
# every remaining field (including nested data/ascot dictionaries), so no
# setting is Web-UI-only.
_TRAIN_FLAGS = (
    ("--model", "model_source", str, "model repo id or local folder"),
    ("--model-dtype", "model_dtype", str, "auto|bfloat16|float16|float32"),
    ("--device", "device", str, "auto|cpu|cuda|cuda:N|mps"),
    ("--dataset-cache", "dataset_cache", str, "name of an existing built cache"),
    ("--val-tokens", "val_tokens", int, "validation tokens per evaluation (0=all)"),
    ("--task", "task", str, "sft|pt"),
    ("--lr", "lr", float, "learning rate"),
    ("--weight-decay", "weight_decay", float, "weight decay"),
    ("--scheduler", "scheduler", str, "cosine|linear|constant"),
    ("--warmup-steps", "warmup_steps", int, "LR warmup steps"),
    ("--total-steps", "total_steps", int, "0 = follow stop conditions"),
    ("--eval-every", "eval_every", int, "validation interval (0=off)"),
    ("--seed", "seed", int, "random seed"),
    ("--guard-mode", "guard_mode", str, "restart|stop|off"),
    ("--val-patience", "val_patience", int, "bad validations before acting"),
    ("--val-min-delta", "val_min_delta", float, "minimum meaningful CE improvement"),
    ("--restart-lr-mode", "restart_lr_mode", str, "harmonic|exponential"),
    ("--restart-lr-decay", "restart_lr_decay", float, "exponential restart factor"),
    ("--restart-lr-floor", "restart_lr_floor", float, "stop restarting below this LR"),
    ("--max-restarts", "max_restarts", int, "0 = unlimited until LR floor"),
    ("--train-probe-tokens", "train_probe_tokens", int, "fixed training probe (0=off)"),
    ("--grad-spike-mode", "grad_spike_mode", str, "off|skip|scale"),
    ("--grad-spike-pct", "grad_spike_threshold_pct", float, "spike threshold percent"),
    ("--grad-spike-window", "grad_spike_window", int, "spike comparison window"),
    ("--max-hours", "max_hours", float, "time limit (0=off)"),
    ("--max-epochs", "max_epochs", int, "epoch limit (0=off)"),
    ("--log-every", "log_every", int, "console log interval"),
    ("--checkpoint-every", "checkpoint_every", int, "checkpoint interval (0=off)"),
    ("--checkpoint-keep-last", "checkpoint_keep_last", int, "rolling checkpoints"),
    ("--checkpoint-keep-every", "checkpoint_keep_every", int, "milestone interval"),
    ("--checkpoint-delta-dtype", "checkpoint_delta_dtype", str,
     "int8|bfloat16|float16|float32 delta storage"),
    ("--export-name", "export_name", str, "export folder name"),
    ("--run-name", "run_name", str, "run folder name"),
)

_BOOL_FLAGS = (
    ("--export-best", "export_best", "export the best validation checkpoint"),
    ("--show-grad-norm", "show_grad_norm", "show grad norm"),
    ("--show-memory", "show_memory", "show memory"),
    ("--checkpoint-optimizer", "checkpoint_optimizer", "save optimizer state"),
    ("--auto-export", "auto_export", "auto-export full weights"),
    ("--save-plots", "save_plots", "write PNG plots"),
    ("--relaunch-ui", "relaunch_ui", "relaunch Web UI when finished"),
)


def _add_lang(sp):
    sp.add_argument("--lang", default=None, help="ja | en")
    return sp


def _add_train_options(sp, *, config_required=True):
    sp.add_argument("--config", required=config_required,
                    help="JSON config file (all fields; optional with flags)")
    sp.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="PATH=VALUE",
                    help="override any field, e.g. --set data.val_ratio=0.03")
    sp.add_argument("--data-json", default=None,
                    help="JSON object merged into the data settings")
    sp.add_argument("--ascot-json", default=None,
                    help="JSON object merged into the ASCOT settings")
    for flag, dest, typ, helptext in _TRAIN_FLAGS:
        sp.add_argument(flag, dest=dest, type=typ, default=None, help=helptext)
    for flag, dest, helptext in _BOOL_FLAGS:
        sp.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction,
                        default=None, help=helptext)
    sp.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction,
                    default=None, help="resume from the newest checkpoint")
    sp.add_argument("--print-config", action="store_true",
                    help="print the normalized config and exit (no training)")
    sp.add_argument("--hold-console", action="store_true",
                    help="keep the console open after training")
    return sp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cpu_finetuning",
                                description="CPU/GPU LLM PT/SFT fine-tuning (ASCOT)")
    p.add_argument("--lang", default=None, help="ja | en")
    sub = p.add_subparsers(dest="cmd")

    sp = _add_lang(sub.add_parser("serve"))
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--config", default=None)
    sp.add_argument("--results", default=None)
    sp.add_argument("--no-browser", action="store_true")

    sp = _add_lang(sub.add_parser("train"))
    _add_train_options(sp, config_required=False)

    sp = _add_lang(sub.add_parser("_train_child"))
    sp.add_argument("--config", required=True)

    sp = _add_lang(sub.add_parser("resume"))
    _add_train_options(sp, config_required=False)

    sp = _add_lang(sub.add_parser("export"))
    sp.add_argument("--config", required=True)
    sp.add_argument("--name", default=None)

    sp = _add_lang(sub.add_parser("info"))

    sp = _add_lang(sub.add_parser("generate"))
    sp.add_argument("--model", required=True)
    sp.add_argument("--prompt", default="Hello!")
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    return p


def _parse_value(text: str):
    s = str(text).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return float(s)
    except Exception:
        return s


def _set_path(cfg, path: str, value):
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return
    target = cfg
    for part in parts[:-1]:
        if isinstance(target, dict):
            target = target.setdefault(part, {})
        else:
            cur = getattr(target, part, None)
            if cur is None:
                cur = {}
                setattr(target, part, cur)
            target = cur
    last = parts[-1]
    if isinstance(target, dict):
        target[last] = value
    elif hasattr(target, last):
        setattr(target, last, value)
    else:
        raise ValueError(f"unknown config field: {path}")


def _apply_args(cfg, args):
    for flag, dest, _typ, _help in _TRAIN_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            setattr(cfg, dest, value)
    for _flag, dest, _help in _BOOL_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            setattr(cfg, dest, value)
    if getattr(args, "lang", None):
        cfg.lang = args.lang
    if getattr(args, "resume", None) is not None:
        cfg.resume = bool(args.resume)
    if getattr(args, "data_json", None):
        d = json.loads(args.data_json)
        if not isinstance(d, dict):
            raise ValueError("--data-json must be an object")
        cfg.data.update(d)
    if getattr(args, "ascot_json", None):
        d = json.loads(args.ascot_json)
        if not isinstance(d, dict):
            raise ValueError("--ascot-json must be an object")
        cfg.ascot.update(d)
    for item in getattr(args, "overrides", []) or []:
        if "=" not in item:
            raise ValueError(f"--set needs KEY=VALUE: {item}")
        key, raw = item.split("=", 1)
        _set_path(cfg, key, _parse_value(raw))
    cfg.normalize()
    cfg.validate_config_keys()
    return cfg


def build_cli_command(cfg) -> str:
    """Return a PowerShell command that reproduces ``cfg`` with CLI flags.

    The Web UI uses this for its "export CLI command" button.  Every setting
    is represented by an explicit flag, ``--data-json``/``--ascot-json`` or a
    ``--set`` override, so the generated command does not depend on the
    saved state.json or on a config file.
    """
    from .core.jsonutil import json_safe

    cfg.normalize()
    cfg.validate_config_keys()

    def quote(value) -> str:
        # The command uses PowerShell's stop-parsing token (--%) after
        # ``train``.  Double quotes are therefore passed to Windows as-is;
        # doubling an embedded quote makes it literal.  Trailing backslashes
        # are doubled so they cannot escape the closing quote.
        text = str(value).replace('"', '""')
        trailing = len(text) - len(text.rstrip("\\"))
        if trailing:
            text = text[:-trailing] + "\\" * (trailing * 2)
        return '"' + text + '"'

    def json_arg(value) -> str:
        return quote(json.dumps(
            json_safe(value), ensure_ascii=False,
            separators=(",", ":"), allow_nan=False))

    parts = ["python", "run.py", "train", "--%"]
    for flag, dest, _typ, _help in _TRAIN_FLAGS:
        value = getattr(cfg, dest, None)
        if value is None:
            continue
        if dest == "dataset_cache" and value == "":
            continue
        parts.extend([flag, quote(value)])
    for flag, dest, _help in _BOOL_FLAGS:
        value = bool(getattr(cfg, dest, False))
        parts.append(flag if value else "--no-" + flag[2:])
    parts.extend(["--resume" if cfg.resume else "--no-resume",
                  "--lang", quote(cfg.lang or "ja")])
    specs = cfg.cleaned_dataset_specs()
    if specs:
        parts.extend(["--set", quote("dataset_specs=" + json.dumps(
            json_safe(specs), ensure_ascii=False,
            separators=(",", ":"), allow_nan=False))])
    if cfg.data:
        parts.extend(["--data-json", json_arg(cfg.data)])
    if cfg.ascot:
        parts.extend(["--ascot-json", json_arg(cfg.ascot)])
    return " ".join(parts)


def _prepare_config(args, *, force_resume: bool = False):
    from .core.config import TrainConfig

    if getattr(args, "config", None):
        cfg = TrainConfig.from_json(Path(args.config).read_text(encoding="utf-8"))
    else:
        cfg = TrainConfig()
    _apply_args(cfg, args)
    if force_resume:
        cfg.resume = True
    return cfg


def main(argv=None) -> int:
    from .core import paths  # noqa: F401  (env redirect happens here)
    from .core.config import ConfigError, TrainConfig
    from .core.i18n import t
    from .core.paths import (STATE_DIR, human_size, cache_size_bytes,
                             CACHE_DIR, DATASETS_DIR)

    args = build_parser().parse_args(argv)
    if not args.cmd:
        args = build_parser().parse_args(["serve"] + (argv or []))

    lang = args.lang or "ja"

    if args.cmd == "serve":
        from .webui.server import serve
        serve(port=args.port, config_path=args.config, results_run=args.results,
              lang=lang, open_browser=not args.no_browser)
        return 0

    if args.cmd in ("train", "resume"):
        try:
            cfg = _prepare_config(args, force_resume=(args.cmd == "resume"))
        except Exception as e:
            print(f"[config error] {e}", flush=True)
            return 2
        if getattr(args, "print_config", False):
            print(cfg.to_json())
            return 0
        try:
            cfg.validate_data_sources()
        except ConfigError as e:
            print(f"[config error] {e}", flush=True)
            return 2
        # The watchdog child must see exactly the effective config, including
        # all command-line overrides.
        cfg_path = Path(args.config).resolve() if getattr(args, "config", None) else None
        if (cfg_path is None or getattr(args, "overrides", None)
                or getattr(args, "data_json", None)
                or getattr(args, "ascot_json", None)
                or getattr(args, "resume", None) is not None
                or any(getattr(args, dest, None) is not None
                       for _f, dest, _t, _h in _TRAIN_FLAGS)
                or any(getattr(args, dest, None) is not None
                       for _f, dest, _h in _BOOL_FLAGS)):
            cfg_path = STATE_DIR / f"cli_{os.getpid()}.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(cfg.to_json(), encoding="utf-8")
        from .core.watchdog import run_with_watchdog
        return run_with_watchdog(str(cfg_path), lang=lang,
                                 hold_console=bool(getattr(args, "hold_console", False)))

    if args.cmd == "_train_child":
        # A malformed or unreadable JSON config is deterministic; returning
        # the same success/failure as a crash would make the watchdog retry it
        # several times.  Use the dedicated non-retryable exit code instead.
        try:
            raw_cfg = open(args.config, encoding="utf-8").read()
            cfg = TrainConfig.from_json(raw_cfg)
        except Exception as e:
            print(f"[config error] {e}", flush=True)
            return 2
        if args.lang:
            cfg.lang = args.lang
        from .train.trainer import Trainer
        try:
            Trainer(cfg).run()
        except ConfigError as e:
            print(f"[config error] {e}", flush=True)
            return 2          # watchdog: do not retry a deterministic error
        return 0

    if args.cmd == "export":
        cfg = TrainConfig.from_json(open(args.config, encoding="utf-8").read())
        from .models.loader import load_model
        from .models.export import export_checkpoint, export_model
        from .train.checkpoint import CheckpointManager
        from .core.paths import RUNS_DIR
        run_dir = Path(RUNS_DIR) / cfg.run_name
        metrics = {}
        try:
            metrics = json.loads((run_dir / "summary.json").read_text(
                encoding="utf-8"))
        except Exception:
            pass
        lm = load_model(cfg.model_source, dtype=cfg.model_dtype,
                        device=cfg.device)
        ckpt = CheckpointManager(
            cfg.run_name, every=cfg.checkpoint_every,
            keep_last=cfg.checkpoint_keep_last,
            keep_every=cfg.checkpoint_keep_every,
            base_model_source=cfg.model_source,
            delta_dtype=cfg.checkpoint_delta_dtype,
            tokenizer=lm.tokenizer, lang=cfg.lang,
        )
        chosen = None
        chosen_step = None
        best_path = ckpt.best_path()
        if cfg.export_best:
            chosen = best_path
            if chosen is not None:
                chosen_step = ckpt.best_step
        if chosen is None:
            found = ckpt.latest()
            if found:
                chosen_step, chosen = found
        out = args.name or cfg.export_name
        metrics["export_source"] = (
            "current" if chosen is None
            else "best" if best_path is not None
            and Path(chosen) == Path(best_path) else "latest"
        )
        if chosen is not None:
            chosen_meta = ckpt.read_meta(int(chosen_step or 0))
            chosen_metrics = chosen_meta.get("metrics") or {}
            metrics.update({
                "exported_step": int(chosen_step or 0),
                "exported_val_ce": chosen_metrics.get("val_ce"),
                "exported_loss": chosen_metrics.get("loss"),
                "exported_ema": chosen_metrics.get("ema"),
                "exported_checkpoint": str(chosen),
                "best_model_exported": bool(
                    best_path is not None
                    and Path(chosen) == Path(best_path)),
            })
            path = export_checkpoint(lm, chosen, out, metrics=metrics)
        else:
            metrics.update({
                "exported_step": None,
                "exported_val_ce": None,
                "exported_loss": None,
                "exported_ema": None,
                "exported_checkpoint": None,
                "best_model_exported": False,
            })
            path = export_model(lm, out, metrics=metrics)
        print(path)
        return 0

    if args.cmd == "info":
        import torch
        import transformers
        print(f"CPU-FineTuning v{__import__('cpu_finetuning').__version__}")
        print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
              f"transformers {transformers.__version__}")
        from .core.device import resolve_device
        choice = resolve_device("auto")
        print(f"device: {choice.device} ({choice.label}) | "
              f"cuda available: {torch.cuda.is_available()}")
        from .core.paths import MODELS_DIR
        print(f"threads: {torch.get_num_threads()}")
        print(f"downloaded models : {MODELS_DIR} ({human_size(cache_size_bytes())})")
        print(f"datasets          : {DATASETS_DIR}")
        return 0

    if args.cmd == "generate":
        import torch
        from .models.loader import (
            load_causal_lm, load_tokenizer, resolve_compute_dtype, resolve_config,
        )
        from .core.device import resolve_device
        tok = load_tokenizer(args.model)
        cfg = resolve_config(args.model)
        choice = resolve_device(args.device)
        device, dev_reason = choice.device, choice.label
        dtype, reason = resolve_compute_dtype("auto", args.model, cfg,
                                               device=device)
        print(f"[generate] device={device} ({dev_reason}) "
              f"dtype={str(dtype).replace('torch.', '')} ({reason})", flush=True)
        model = load_causal_lm(args.model, dtype=dtype).to(device)
        model.eval()
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.7, top_k=50)
        print(tok.decode(out[0][ids.shape[1]:].to("cpu"), skip_special_tokens=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
