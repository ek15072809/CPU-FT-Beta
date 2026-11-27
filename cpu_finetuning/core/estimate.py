"""Training-time estimates (v6, requirement 9).

The estimate is deliberately simple and calibrated against the verified
1.73B LFM3 / 24-layer / 24576-context run (~23 s per balanced ASCOT step).
It scales the main cost drivers: model size per layer, frozen pre-roll
tokens, trainable span tokens x slice size, and LM-head positions.

v6 fixes a misleading fallback: when the selected dataset could not be
counted, the old estimate used one sequence and displayed a one-step time
(e.g. 23 s) as the whole training time.  Local JSONL/JSON/CSV/TSV/Parquet/
Arrow files, directory trees, quoted Windows paths, and ``subset`` limits are
now counted before packing.  If no trustworthy count is available the API
returns an explicit error instead of inventing a duration.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from .paths import PROJECT_ROOT, strip_path_quotes

# calibration point: verified v4 balanced run
REF_PARAMS = 1.73e9
REF_LAYERS = 24
REF_PER_LAYER = REF_PARAMS / REF_LAYERS
REF_VOCAB_HIDDEN_UNITS = 128000 * 2048 / REF_PER_LAYER
REF_SEC = 23.0
REF_UNITS = (
    4 * (16 + 48 + 128) * REF_LAYERS
    + 3.0 * (4 * 128) * 4
    + (4 * 64) * REF_VOCAB_HIDDEN_UNITS
)

_LOCAL_EXTS = (
    ".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow",
    ".txt", ".md", ".yaml", ".yml",
)
_METADATA_FILES = {"dataset_info.json", "state.json"}


def _cfg_attr(cfg, name, default=None):
    text = getattr(cfg, "text_config", None)
    if text is not None and hasattr(text, name):
        return getattr(text, name)
    return getattr(cfg, name, default)


def estimate_step_seconds(n_params: int, n_layers: int, ascot,
                          vocab_size: int = 128000,
                          hidden_size: int = 2048,
                          dtype_name: str = "float32") -> float:
    """Approximate seconds for one ASCOT step on a CPU."""
    layers = max(1, int(n_layers or REF_LAYERS))
    params = max(1, int(n_params or REF_PARAMS))
    per_layer = params / layers
    spans = max(1, int(getattr(ascot, "num_spans", 4)))
    span = max(8, int(getattr(ascot, "span_length", 128)))
    sink = max(0, int(getattr(ascot, "sink_len", 16)))
    prefix = max(0, int(getattr(ascot, "prefix_len", 48)))
    # The plan slot is only added to long-document sequences (~1/8 of the cap
    # in practice), not to every normal step.  Folding it into every step
    # overestimated the v6 run by ~25%.  The fixed calibration below already
    # includes the ordinary pre-roll cost; the long-doc surcharge is small.
    slice_size = max(1, int(getattr(ascot, "slice_size", 4)))
    head_positions = int(getattr(ascot, "head_positions", 64) or span)
    head_positions = max(1, min(head_positions, span))
    vh_units = max(0.1, (vocab_size * hidden_size) / max(1.0, per_layer))

    frozen_tokens = spans * (sink + prefix + span)
    train_tokens = spans * span
    units = (
        frozen_tokens * layers
        + 3.0 * train_tokens * slice_size
        + (spans * head_positions) * vh_units
    )
    scale = per_layer / REF_PER_LAYER
    dtype = str(dtype_name).lower()
    if "bfloat" in dtype:
        dtype_factor = 3.2
    elif "float16" in dtype or "half" in dtype:
        dtype_factor = 2.0
    else:
        dtype_factor = 1.0
    sec = REF_SEC * (units / REF_UNITS) * scale * dtype_factor + 0.5
    return max(0.5, sec)


def _count_json_rows(path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except Exception:
        # A .json file may contain JSONL (one object per line).
        return sum(1 for line in text.splitlines() if line.strip())
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        # Common wrappers used by exported datasets.
        for key in ("data", "rows", "examples", "records", "train"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
        return 1
    return 0


def _count_local_file(path: Path) -> int:
    """Count rows in one supported local file without loading the dataset."""
    low = path.name.lower()
    try:
        if low.endswith(".jsonl"):
            n = 0
            with open(path, "rb") as f:
                for line in f:
                    if line.strip():
                        n += 1
            return n
        if low.endswith(".json"):
            return _count_json_rows(path)
        if low.endswith((".csv", ".tsv")):
            delimiter = "\t" if low.endswith(".tsv") else ","
            with open(path, "r", encoding="utf-8", errors="replace",
                      newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = sum(1 for row in reader if any(cell.strip() for cell in row))
            return max(0, rows - 1)       # subtract the header row
        if low.endswith(".parquet"):
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(str(path)).metadata.num_rows)
        if low.endswith(".arrow"):
            # HuggingFace Arrow datasets are best counted by the datasets
            # reader; fall back to the Arrow IPC reader for plain files.
            try:
                from datasets import Dataset
                return int(Dataset.from_file(str(path)).num_rows)
            except Exception:
                import pyarrow as pa
                with pa.memory_map(str(path), "r") as source:
                    try:
                        reader = pa.ipc.open_file(source)
                        return int(sum(reader.get_batch(i).num_rows
                                       for i in range(reader.num_record_batches)))
                    except Exception:
                        reader = pa.ipc.open_stream(source)
                        return int(sum(batch.num_rows for batch in reader))
        if low.endswith((".txt", ".md", ".yaml", ".yml")):
            return 1
    except Exception:
        return 0
    return 0


def _count_local(path: Path) -> int:
    if path.is_file():
        return _count_local_file(path)
    if path.is_dir():
        total = 0
        found = False
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.lower() in _METADATA_FILES:
                    continue
                p = Path(root) / name
                if not p.name.lower().endswith(_LOCAL_EXTS):
                    continue
                found = True
                total += _count_local_file(p)
        return total if found else 0
    return 0


def _count_remote(spec) -> tuple[int, bool]:
    """Best-effort row count for a HuggingFace dataset id.

    ``load_dataset_builder`` normally resolves split sizes from the dataset
    metadata without downloading all rows.  Offline/private datasets return
    unknown, and the API surfaces that instead of showing a fake 1-step time.
    """
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(spec.source, name=spec.name)
        info = getattr(builder, "info", None)
        splits = getattr(info, "splits", None) or {}
        split_name = spec.split or "train"
        if split_name in splits:
            n = getattr(splits[split_name], "num_examples", None)
            if n:
                return int(n), True
        total = sum(int(getattr(s, "num_examples", 0) or 0)
                    for s in splits.values())
        if total:
            return total, True
    except Exception:
        pass
    return 0, False


def _count_spec(spec) -> tuple[int, bool, str]:
    """Return (examples, known, reason)."""
    src = strip_path_quotes(str(getattr(spec, "source", "") or "")).strip()
    subset = int(getattr(spec, "subset", 0) or 0)
    if not src:
        return 0, False, "dataset source is empty"

    path = Path(src).expanduser()
    if not path.is_absolute() and (PROJECT_ROOT / path).exists():
        path = PROJECT_ROOT / path
    if path.exists():
        n = _count_local(path)
        if n > 0:
            return (min(n, subset) if subset else n), True, ""
        return 0, False, f"no countable rows in {src}"

    if subset:
        # The user explicitly capped the source; the cap is a safe estimate
        # even for a remote dataset that cannot be inspected offline.
        return subset, True, ""

    n, known = _count_remote(spec)
    if known:
        return n, True, ""
    return 0, False, f"could not count remote dataset {src}"


def _local_path(spec) -> Path | None:
    src = strip_path_quotes(str(getattr(spec, "source", "") or "")).strip()
    if not src:
        return None
    path = Path(src).expanduser()
    if not path.is_absolute() and (PROJECT_ROOT / path).exists():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def _local_bytes(path: Path) -> int:
    """Total bytes of a local source (files are summed for directories)."""
    try:
        if path.is_file():
            return int(path.stat().st_size)
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.lower() in _METADATA_FILES:
                    continue
                if not name.lower().endswith(_LOCAL_EXTS):
                    continue
                total += int((Path(root) / name).stat().st_size)
        return total
    except Exception:
        return 0


def _prebuild_sequence_estimate(cfg, examples: int, specs: list) -> tuple[int, int, float]:
    """Estimate packed sequences before the tokenizer cache exists.

    v6 assumed SFT always packs two examples per sequence.  On the reported
    run that predicted 668 sequences, while packing 29.5M tokens into 32k
    sequences actually produced 1701 (long documents fragment across many
    sequences).  The UI therefore promised ~8 h for a ~20 h run.

    This estimator uses the local source byte size (cheap, no tokenizer),
    a task-dependent bytes/token factor, and the same long-document plan-slot
    rule as the builder.  It is deliberately labelled provisional; after the
    cache exists the exact stats from the builder are used.
    """
    data = cfg.data or {}
    ctx = max(256, int(data.get("context_length", 32768) or 32768))
    eps = max(0, int(data.get("examples_per_step", 0) or 0))
    max_seq = max(0, int(data.get("max_sequences", 0) or 0))
    val_ratio = min(0.9, max(0.0, float(data.get("val_ratio", 0.0) or 0.0)))
    long_doc = bool(data.get("long_doc_structure", True))
    long_min = max(0, int(data.get("long_doc_min_chars", 1200) or 0))
    long_cap = max(0, int(data.get("long_doc_max_docs", 0) or 0))

    total_bytes = 0
    known_bytes = 0
    for spec in specs:
        p = _local_path(spec)
        if p is not None:
            b = _local_bytes(p)
            if b:
                total_bytes += b
                known_bytes += b

    # JSON/JSONL carries keys and quoting overhead; plain text does not.
    # 6 bytes/token matched the 29.5M-token / 177MB mixed English+Japanese
    # run; use a slightly more conservative factor for pure text.
    has_json = any(
        strip_path_quotes(str(getattr(s, "source", ""))).lower().endswith(
            (".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow"))
        for s in specs
    )
    bpt = 6.0 if has_json else 4.5
    est_tokens = int(total_bytes / bpt) if total_bytes else 0
    avg_tokens = (est_tokens / max(1, examples)) if est_tokens else 0.0

    if eps > 0:
        normal_seqs = int(math.ceil(max(1, examples) / eps))
    elif est_tokens:
        # Two lower bounds: token volume and example fragmentation.  The
        # second term dominates long-document data, where one document often
        # spans several context windows.
        by_tokens = int(math.ceil(est_tokens / ctx))
        pack_ratio = min(0.9, max(0.02, 1.5 * avg_tokens / ctx))
        by_examples = int(math.ceil(max(1, examples) * pack_ratio))
        normal_seqs = max(by_tokens, by_examples)
    else:
        # Remote/unknown sources: keep the old conservative fallback.
        per_seq = 1.0 if cfg.task == "pt" else 2.0
        normal_seqs = max(1, int(round(max(1, examples) / per_seq)))

    long_seqs = 0
    if long_doc and long_min > 0 and examples > 0:
        # A document is long if its average token count comfortably exceeds
        # the configured character threshold.  For mixed data the average is
        # dominated by the large source, matching the builder's plan slots.
        if avg_tokens <= 0 or avg_tokens * 2.0 >= long_min:
            plan_slots = (max_seq // 8) if max_seq > 0 else long_cap
            if plan_slots <= 0:
                plan_slots = long_cap or examples
            long_seqs = min(int(plan_slots), max(0, int(examples)))
    total_seqs = max(1, normal_seqs + long_seqs)
    if max_seq > 0:
        # The cap applies to final packed sequences (plan samples included).
        total_seqs = min(total_seqs, max_seq)

    n_val = max(1, int(total_seqs * val_ratio)) if val_ratio > 0 else 0
    if n_val >= total_seqs:
        n_val = max(0, total_seqs - 1)
    train_seqs = max(1, total_seqs - n_val)
    build_seconds = 2.0 + max(0, examples) * 0.04
    return total_seqs, train_seqs, build_seconds


def _read_cache_stats(cfg) -> dict:
    try:
        from .paths import DATASET_CACHE_DIR
        path = Path(DATASET_CACHE_DIR) / (cfg.dataset_cache_name() + ".pt.json")
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            from ..data.builder import CACHE_FMT
            if isinstance(d, dict) and d.get("fmt") == CACHE_FMT:
                return d
    except Exception:
        pass
    return {}


def _estimate_error(message: str, note: str = "") -> dict:
    return {
        "error": message,
        "examples": None,
        "sequences": None,
        "train_sequences": None,
        "steps": None,
        "sec_per_step": None,
        "build_seconds": None,
        "train_seconds": None,
        "total_seconds": None,
        "known_dataset": False,
        "provisional": True,
        "note": note,
    }


def estimate_for_config(cfg) -> dict:
    """Return a JSON-friendly estimate for a TrainConfig."""
    from ..models.loader import (
        estimate_params, resolve_compute_dtype, resolve_config,
    )

    notes = []
    n_params = 2_000_000_000
    n_layers = 24
    vocab = 128000
    hidden = 2048
    dtype_name = "float32"
    try:
        mcfg = resolve_config(cfg.model_source)
        n_params = estimate_params(cfg.model_source, mcfg)
        n_layers = int(_cfg_attr(mcfg, "num_hidden_layers", 24) or 24)
        vocab = int(_cfg_attr(mcfg, "vocab_size", 128000) or 128000)
        hidden = int(_cfg_attr(mcfg, "hidden_size", 2048) or 2048)
        dtype, _reason = resolve_compute_dtype(cfg.model_dtype, cfg.model_source, mcfg)
        dtype_name = str(dtype).replace("torch.", "")
    except Exception as e:
        notes.append(f"model config unavailable: {repr(e)[:120]}")

    stats = _read_cache_stats(cfg)
    if stats and stats.get("fingerprint") != cfg.dataset_fingerprint():
        stats = {}
    examples = int(stats.get("examples_kept")
                   or stats.get("examples_used")
                   or stats.get("examples_total") or 0)
    # A stats file with only a fingerprint is not a usable count.
    known = bool(stats.get("train_sequences") or examples)
    count_reasons = []
    if not examples:
        examples = 0
        specs = cfg.spec_objects()
        all_known = bool(specs)
        for spec in specs:
            n, ok, reason = _count_spec(spec)
            examples += n
            all_known = all_known and ok
            if reason:
                count_reasons.append(reason)
        known = known or all_known
        if not all_known:
            notes.append(
                "データセット件数を一部取得できません / "
                "some dataset counts are estimates"
            )

    if not examples and not known:
        detail = "; ".join(dict.fromkeys(count_reasons)) or (
            "dataset source is empty"
        )
        return _estimate_error(
            "データセット件数を取得できません。ローカルファイルを指定するか、"
            "「先頭N件」を設定してください / "
            "could not count the dataset; select a local file or set 'first N'"
            f" ({detail})",
            " / ".join(notes),
        )

    if stats.get("train_sequences"):
        train_seqs = int(stats["train_sequences"])
        sequences = int(stats.get("sequences") or train_seqs)
        build_seconds = float(stats.get("build_seconds") or 0.0)
        provisional = False
    else:
        sequences, train_seqs, build_seconds = _prebuild_sequence_estimate(
            cfg, examples, cfg.spec_objects())
        notes.append(
            "シーケンス数はデータ構築前の概算です（構築後に実測値へ更新） / "
            "sequence count is a pre-build estimate (exact after building)"
        )
        provisional = True

    ascot = cfg.ascot_config()
    sec_step = estimate_step_seconds(n_params, n_layers, ascot, vocab, hidden,
                                     dtype_name)

    total_steps = int(cfg.total_steps or 0)
    max_epochs = int(cfg.max_epochs or 0)
    max_hours = float(cfg.max_hours or 0.0)
    if total_steps > 0:
        steps = total_steps
        if max_epochs > 0:
            # match the trainer: a huge total_steps does not override a
            # smaller epoch limit (v4 default was 1e15)
            steps = min(steps, max_epochs * train_seqs)
    elif max_epochs > 0:
        steps = max_epochs * train_seqs
    elif max_hours > 0:
        steps = max(1, int(max_hours * 3600 / sec_step))
    else:
        steps = train_seqs  # requirement: exactly one epoch

    train_seconds = steps * sec_step
    total_seconds = train_seconds + build_seconds
    return {
        "params": int(n_params),
        "layers": int(n_layers),
        "examples": int(examples) if examples else None,
        "sequences": int(sequences),
        "train_sequences": int(train_seqs),
        "steps": int(steps),
        "sec_per_step": round(sec_step, 2),
        "build_seconds": round(build_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "dtype": dtype_name,
        "known_dataset": bool(known),
        "provisional": bool(provisional),
        "note": " / ".join(notes) if notes else "",
    }
