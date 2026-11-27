"""Model usage history and generation-model suggestions (v5).

The generation tab can either take a free-form model path/HF id or pick a
model from a suggestion list.  Suggestions are built from:

* every model used by a training run (the base model), and
* every exported model under ``exports/`` (the trained model),

ordered by the last time the model was used for **training or generation**.
The history is stored in ``state/model_usage.json`` and is updated by the
trainer and by the Web UI generation endpoint.

The same module also resolves user-supplied generation names.  A bare name
such as ``run1`` used to be passed to transformers as an HF repo id, which
produced a long series of ``getaddrinfo`` retries for
``https://huggingface.co/run1/...``.  Run names and export folder names are
now resolved locally before anything is sent to the Hub.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .paths import EXPORT_DIR, PROJECT_ROOT, RUNS_DIR, STATE_DIR, strip_path_quotes

USAGE_FILE = STATE_DIR / "model_usage.json"


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _is_hf_id(s: str) -> bool:
    """A conservative HF repo-id test (``owner/name``)."""
    return "/" in s and not s.startswith(("/", ".", "~")) and "\\" not in s


def canon_path(path: str) -> str:
    """Normalize a local path for de-duplication; keep HF ids verbatim."""
    s = strip_path_quotes(str(path or ""))
    if not s:
        return ""
    p = Path(s)
    if p.exists() or p.is_absolute() or "\\" in s or s.startswith((".", "~")):
        try:
            return os.path.normcase(str(p.resolve()))
        except Exception:
            return os.path.normcase(os.path.abspath(s))
    return s


def load_usage() -> dict:
    d = _read_json(USAGE_FILE)
    models = d.get("models")
    return models if isinstance(models, dict) else {}


def _save_usage(models: dict) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USAGE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "models": models},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, USAGE_FILE)


def touch_model(path: str, kind: str = "base", mode: str = "generate",
                run_name: str = "", base_model: str = "",
                label: str = "") -> None:
    """Record that ``path`` was used for training or generation."""
    path = strip_path_quotes(str(path or ""))
    if not path:
        return
    key = canon_path(path)
    models = load_usage()
    e = models.get(key, {})
    e.update({
        "path": path,
        "kind": kind if kind in ("base", "trained") else e.get("kind", "base"),
        "last_used": time.time(),
        "last_mode": mode,
        "uses": int(e.get("uses", 0)) + 1,
    })
    if run_name:
        e["run_name"] = run_name
    if base_model:
        e["base_model"] = base_model
    if label:
        e["label"] = label
    models[key] = e
    _save_usage(models)


def _add(entries: dict, path: str, kind: str, *, run_name: str = "",
         base_model: str = "", timestamp: float = 0.0) -> None:
    path = strip_path_quotes(str(path or ""))
    if not path:
        return
    key = canon_path(path)
    old = entries.get(key)
    # A trained export is more useful than the same path as a generic base.
    if old and old.get("kind") == "trained" and kind != "trained":
        kind = "trained"
    e = old or {"path": path, "kind": kind, "last_used": 0.0}
    e["kind"] = kind
    if run_name and not e.get("run_name"):
        e["run_name"] = run_name
    if base_model and not e.get("base_model"):
        e["base_model"] = base_model
    if timestamp > float(e.get("last_used") or 0.0):
        e["last_used"] = timestamp
    # Prefer an existing local path over a stale quoted/relative spelling.
    p = Path(e["path"])
    if not p.exists() and Path(path).exists():
        e["path"] = path
    entries[key] = e


def _scan_runs(entries: dict) -> None:
    if not Path(RUNS_DIR).exists():
        return
    for run in sorted(Path(RUNS_DIR).iterdir()):
        if not run.is_dir():
            continue
        cfg = _read_json(run / "config.json")
        summary = _read_json(run / "summary.json")
        source = (summary.get("model") or cfg.get("model_source") or "").strip()
        run_name = summary.get("run_name") or cfg.get("run_name") or run.name
        ts = 0.0
        try:
            ts = (run / "summary.json").stat().st_mtime
        except OSError:
            try:
                ts = (run / "config.json").stat().st_mtime
            except OSError:
                pass
        if source:
            _add(entries, source, "base", run_name=run_name, timestamp=ts)
        exp = (summary.get("export_path") or "").strip()
        if not exp:
            # best-effort naming used by the trainer
            cand = Path(EXPORT_DIR) / f"finetuned-{run_name}"
            exp = str(cand) if (cand / "config.json").exists() else ""
        if exp and Path(exp).is_dir():
            _add(entries, exp, "trained", run_name=run_name,
                 base_model=source, timestamp=ts)


def _scan_exports(entries: dict) -> None:
    if not Path(EXPORT_DIR).exists():
        return
    for exp in sorted(Path(EXPORT_DIR).iterdir()):
        if not exp.is_dir() or not (exp / "config.json").exists():
            continue
        meta = _read_json(exp / "cpu_finetuning_meta.json")
        source = (meta.get("source_model") or meta.get("model") or "").strip()
        run_name = (meta.get("run_name") or "").strip()
        try:
            ts = max((exp / "config.json").stat().st_mtime,
                     (exp / "cpu_finetuning_meta.json").stat().st_mtime)
        except OSError:
            ts = 0.0
        _add(entries, str(exp), "trained", run_name=run_name,
             base_model=source, timestamp=ts)
        if source:
            _add(entries, source, "base", run_name=run_name, timestamp=ts)


def list_models() -> list[dict]:
    """Return generation suggestions, most recently used first."""
    entries: dict[str, dict] = {}
    _scan_runs(entries)
    _scan_exports(entries)

    # explicit usage history wins over mtime guesses
    for key, use in load_usage().items():
        path = use.get("path") or key
        kind = use.get("kind") or ("trained" if Path(path).is_dir()
                                   and Path(path).parent == Path(EXPORT_DIR)
                                   else "base")
        _add(entries, path, kind,
             run_name=use.get("run_name", ""),
             base_model=use.get("base_model", ""),
             timestamp=float(use.get("last_used") or 0.0))
        e = entries.get(canon_path(path))
        if e is not None:
            e["last_used"] = max(float(e.get("last_used") or 0.0),
                                 float(use.get("last_used") or 0.0))
            e["last_mode"] = use.get("last_mode", "")
            e["uses"] = int(use.get("uses", 0))

    # include the model currently configured for training, even before a run
    try:
        from .config import TrainConfig
        from .paths import STATE_DIR as _STATE
        state_file = _STATE / "state.json"
        if state_file.exists():
            cfg = TrainConfig.from_json(state_file.read_text(encoding="utf-8"))
            if cfg.model_source:
                _add(entries, cfg.model_source, "base", run_name=cfg.run_name)
    except Exception:
        pass

    out = []
    for e in entries.values():
        path = e.get("path", "")
        if not path:
            continue
        p = Path(path)
        base = e.get("base_model") or ""
        out.append({
            "path": path,
            "kind": e.get("kind", "base"),
            "run_name": e.get("run_name", ""),
            "base_model": base,
            "last_used": float(e.get("last_used") or 0.0),
            "last_mode": e.get("last_mode", ""),
            "exists": p.exists() if not _is_hf_id(path) else None,
        })
    out.sort(key=lambda x: (-x["last_used"],
                            0 if x["kind"] == "trained" else 1,
                            x["path"].lower()))
    return out


def base_model_for(path: str) -> str:
    """Return the pre-training model for an export, when known."""
    p = Path(strip_path_quotes(str(path or "")))
    try:
        meta = _read_json(p / "cpu_finetuning_meta.json")
        src = (meta.get("source_model") or meta.get("model") or "").strip()
        if src:
            return src
    except Exception:
        pass
    # a run name resolves to its summary/config
    run = Path(RUNS_DIR) / p.name
    cfg = _read_json(run / "config.json")
    return str(cfg.get("model_source") or "").strip()


def resolve_model_source(source: str) -> str:
    """Resolve a local run/export name before treating it as an HF repo id.

    Raises ``FileNotFoundError`` for a path-like string that does not exist,
    and for a known run name that has no exported model yet, with a helpful
    message instead of a Hub retry storm.
    """
    s = strip_path_quotes(str(source or ""))
    if not s:
        raise ValueError("モデルが指定されていません / model source is empty")

    p = Path(s)
    if p.exists():
        if (p / "config.json").exists():
            return str(p.resolve())
        raise FileNotFoundError(
            f"モデルフォルダに config.json がありません / not a model folder: {s}"
        )

    if p.is_absolute() or "\\" in s or s.startswith((".", "~")):
        raise FileNotFoundError(
            f"ローカルモデルが見つかりません / local model not found: {s} "
            "(引用符は自動で外されます / quotes are removed automatically)"
        )

    # project-relative paths (exports/foo, model, ...)
    cand = Path(PROJECT_ROOT) / s
    if cand.exists():
        if (cand / "config.json").exists():
            return str(cand.resolve())
        raise FileNotFoundError(
            f"モデルフォルダに config.json がありません / not a model folder: {cand}"
        )

    # exact export folder name
    cand = Path(EXPORT_DIR) / s
    if (cand / "config.json").exists():
        return str(cand.resolve())

    # run name -> the export recorded in its summary (or the default name)
    run = Path(RUNS_DIR) / s
    if run.is_dir():
        summary = _read_json(run / "summary.json")
        exp = (summary.get("export_path") or "").strip()
        if exp and Path(exp).is_dir() and (Path(exp) / "config.json").exists():
            return str(Path(exp).resolve())
        cand = Path(EXPORT_DIR) / f"finetuned-{s}"
        if (cand / "config.json").exists():
            return str(cand.resolve())
        cfg = _read_json(run / "config.json")
        name = (cfg.get("export_name") or "finetuned").strip()
        cand = Path(EXPORT_DIR) / name
        if (cand / "config.json").exists():
            return str(cand.resolve())
        raise FileNotFoundError(
            f"{s!r} は学習run名ですが、書き出し済みモデルがありません。"
            f" 生成には exports/ の学習後モデルを選んでください。"
            " / this is a training run without an exported model; "
            "select an export under exports/."
        )

    # A missing project-relative path must not be mistaken for an HF repo id.
    low = s.replace("\\", "/").lower()
    if any(low.startswith(prefix) for prefix in (
            "exports/", "runs/", "models/", "model/", "checkpoints/",
            "datasets/", "./", "../")):
        raise FileNotFoundError(
            f"ローカルモデルが見つかりません / local model not found: {s}"
        )

    # Bare names such as ``run1`` are not valid Hub repository ids.  Treating
    # them as one caused the old getaddrinfo retry storm against
    # https://huggingface.co/run1/... even though the intent was a local run.
    if not _is_hf_id(s):
        raise FileNotFoundError(
            f"モデルが見つかりません / model not found: {s!r}. "
            "学習run名なら exports/ の書き出し済みモデルを指定してください。"
            " / for a Hub model use the owner/name form."
        )

    # A known HF id is left untouched; the loader will fetch it.
    return s
