"""Dataset formats: auto-detection and normalization to a common schema.

Supported raw schemas (auto-detected per example, overridable):

* ``messages``      - [{"role": ..., "content": ...}] (OpenAI / HF chat)
* ``conversations`` - [{"from": "human"/"gpt", "value": ...}] (ShareGPT)
* ``instruction``   - Alpaca style (instruction / input? / output)
* ``prompt``        - prompt / response or question / answer pairs
* ``text``          - raw text (pre-training) or content field
* ``chosen``        - preference-style (only chosen used)

Any HF dataset that exposes one of these column sets works, regardless of
file format (json/jsonl/csv/tsv/parquet/arrow/...).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.paths import looks_like_path, strip_path_quotes

ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
    "system": "system", "prompter": "user",
}


@dataclass
class Example:
    """Normalized training example."""

    messages: list[dict[str, str]] | None = None   # chat format
    text: str | None = None                        # raw text (PT)
    source: str = ""

    def is_chat(self) -> bool:
        return self.messages is not None


def _as_messages(obj: Any) -> list[dict[str, str]] | None:
    if not isinstance(obj, (list, tuple)) or not obj:
        return None
    out = []
    for m in obj:
        if not isinstance(m, dict):
            return None
        role = m.get("role") or m.get("from") or m.get("speaker")
        content = m.get("content") if "content" in m else (
            m.get("value") or m.get("text") or m.get("message")
        )
        if role is None or content is None:
            return None
        out.append({"role": ROLE_MAP.get(str(role).lower(), str(role).lower()),
                    "content": str(content)})
    return out


def detect_format(example: dict) -> str:
    keys = set(example.keys())
    if "messages" in keys and _as_messages(example["messages"]):
        return "messages"
    if "conversations" in keys and _as_messages(example["conversations"]):
        return "conversations"
    if {"instruction"} & keys and ({"output", "response"} & keys):
        return "alpaca"
    if {"prompt", "question", "q", "input"} & keys and (
            {"response", "answer", "a", "output", "completion"} & keys):
        return "prompt"
    if {"chosen", "rejected"} <= keys:
        return "preference"
    if {"text", "content", "body", "document", "raw"} & keys:
        return "text"
    return "unknown"


def normalize(example: dict, fmt: str | None = None, source: str = "") -> Example | None:
    fmt = fmt or detect_format(example)
    if fmt in ("messages", "conversations"):
        msgs = _as_messages(example.get("messages") or example.get("conversations"))
        if msgs:
            return Example(messages=msgs, source=source)
        return None
    if fmt == "alpaca":
        prompt = example.get("instruction") or ""
        if example.get("input"):
            prompt = prompt + "\n\n" + str(example["input"])
        return Example(messages=[
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(example.get("output") or example.get("response") or "")},
        ], source=source)
    if fmt == "preference":
        return normalize({"messages": example["chosen"]}, "messages", source)
    if fmt == "prompt":
        q = example.get("prompt") or example.get("question") or example.get("q") or example.get("input")
        a = (example.get("response") or example.get("answer") or example.get("a")
             or example.get("output") or example.get("completion"))
        if q is None or a is None:
            return None
        return Example(messages=[
            {"role": "user", "content": str(q)},
            {"role": "assistant", "content": str(a)},
        ], source=source)
    if fmt == "text":
        for k in ("text", "content", "body", "document", "raw"):
            if k in example and example[k]:
                return Example(text=str(example[k]), source=source)
    # last resort: single string column
    for v in example.values():
        if isinstance(v, str) and len(v) > 20:
            return Example(text=v, source=source)
    return None


@dataclass
class DatasetSpec:
    """Declarative description of one data source."""

    source: str                          # HF repo id, or local file/dir path
    split: str = "train"
    name: str | None = None              # config name; None = auto/combine all
    format: str | None = None            # None = auto
    weight: float = 1.0                  # mixing weight
    subset: int | None = None            # use only first N examples
    columns: dict = field(default_factory=dict)  # optional column renaming


def _is_missing_config_error(e: BaseException) -> bool:
    """True when datasets.load_dataset asks for a config name."""
    msg = str(e).lower()
    return ("config name is missing" in msg
            or "please pick one among the available configs" in msg)


def _hub_config_names(src: str) -> list[str]:
    """Return the Hub dataset's config names, or [] when unavailable."""
    try:
        from datasets import get_dataset_config_names
        names = get_dataset_config_names(src)
    except Exception:
        return []
    return list(dict.fromkeys(
        str(n).strip() for n in names if str(n).strip()))


def _load_hub_dataset(src: str, spec: DatasetSpec):
    """Load a Hub dataset, auto-selecting/combining configs when name is blank.

    Some Hub datasets (e.g. Jackrong/DeepSeek-V4-Pro-Distilled-200K) expose
    multiple builder configs and require a config name.  A blank ``name`` used
    to fail with ``Config name is missing``.  v13 resolves the available
    configs and combines them; an explicit ``name`` still selects exactly one
    config.
    """
    from datasets import load_dataset, concatenate_datasets

    name = (spec.name or "").strip() or None
    split = (spec.split or "train").strip() or "train"
    try:
        return load_dataset(src, name=name, split=split)
    except Exception as e:
        if name is not None or not _is_missing_config_error(e):
            raise
        names = _hub_config_names(src)
        if not names:
            raise
        if len(names) == 1:
            return load_dataset(src, name=names[0], split=split)

        parts = []
        failed: list[tuple[str, BaseException]] = []
        for config_name in names:
            try:
                parts.append(load_dataset(src, name=config_name, split=split))
            except Exception as inner:
                failed.append((config_name, inner))
        if not parts:
            raise ValueError(
                f"dataset {src!r} has configs {names}, but none could be "
                f"loaded for split {split!r}; set the dataset config name "
                f"explicitly (one of: {', '.join(names)})"
            ) from failed[0][1]
        if failed:
            warnings.warn(
                f"dataset {src!r}: could not load config(s) "
                f"{', '.join(n for n, _ in failed)} for split {split!r}; "
                f"using {len(parts)} of {len(names)} configs",
                UserWarning, stacklevel=2,
            )
        if len(parts) == 1:
            return parts[0]
        try:
            return concatenate_datasets(parts)
        except Exception as e:
            raise ValueError(
                f"dataset {src!r} has multiple configs {names} with "
                f"incompatible schemas; set the dataset config name "
                f"explicitly (one of: {', '.join(names)})"
            ) from e


def load_raw(spec: DatasetSpec):
    """Load the raw rows of one spec (shared by the builder and the UI preview).

    Local paths are supported with or without surrounding quotes (Windows
    "Copy as path" pastes ``"C:\\...\\data.jsonl"``); local files are fed to
    the matching builder via ``data_files=`` instead of being mistaken for a
    hub dataset script.

    For a Hub dataset with multiple configs and a blank ``name``, all configs
    are loaded and concatenated.  Set ``name`` to use one config only.
    """
    from datasets import load_dataset, concatenate_datasets

    src = strip_path_quotes(spec.source).strip()
    if not src:
        raise ValueError(
            "dataset source is empty; select a dataset / "
            "データセットソースが空です。データセットを指定してください"
        )
    exts = (".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow",
            ".txt", ".md", ".yaml", ".yml")
    is_local = os.path.exists(src)

    if not is_local and looks_like_path(src):
        raise FileNotFoundError(
            f"local data source not found: {src} "
            "(surrounding quotes are removed automatically; check the path)"
        )

    if is_local and os.path.isdir(src):
        files = []
        for root, _dirs, fnames in os.walk(src):
            for fn in sorted(fnames):
                if fn.lower().endswith(exts):
                    files.append(os.path.join(root, fn))
        if not files:
            raise ValueError(f"no supported files under {src}")
        parts = []
        for f in files:
            kind = _file_kind(f)
            d = load_dataset(kind, data_files=f, split="train")
            parts.append(d)
        return concatenate_datasets(parts)
    if is_local:
        kind = _file_kind(src)
        return load_dataset(kind, data_files=src, split="train")
    return _load_hub_dataset(src, spec)


def load_examples(spec: DatasetSpec, bar: bool = True) -> list[Example]:
    """Load a dataset spec into normalized examples."""
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **kw: x  # noqa: E731

    src = strip_path_quotes(spec.source).strip()
    ds = load_raw(spec)

    if spec.subset:
        ds = ds.select(range(min(spec.subset, len(ds))))

    fmt = spec.format
    examples: list[Example] = []
    for row in tqdm(ds, desc=f"normalize {Path(src).name[:24]}",
                    total=len(ds), leave=False, mininterval=1.0,
                    disable=not bar, dynamic_ncols=True):
        if spec.columns:
            row = {spec.columns.get(k, k): v for k, v in row.items()}
        ex = normalize(dict(row), fmt=fmt, source=src)
        if ex is not None:
            examples.append(ex)
    return examples


def _file_kind(path: str) -> str:
    low = path.lower()
    if low.endswith((".json", ".jsonl")):
        return "json"
    if low.endswith((".csv", ".tsv")):
        return "csv"
    if low.endswith(".parquet"):
        return "parquet"
    if low.endswith(".arrow"):
        return "arrow"
    if low.endswith((".txt", ".md")):
        return "text"
    if low.endswith((".yaml", ".yml")):
        return "json"  # fallback
    return "json"
