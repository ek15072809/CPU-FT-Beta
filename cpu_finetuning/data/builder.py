"""Dataset builder: tokenize, mask, pack, filter, split, save (v2).

v5 additions:
* ``dataset_order="front"`` keeps source order instead of shuffling, and
  mixes smaller sources randomly into the first N examples of the largest
  source.  Training then consumes the data from the front, so a small dataset
  mixed into the prefix is fully used even if training stops early.
* long documents get a document-level learning sample: a compact plan
  (title/outline/key facts/recurring entities) as loss-masked context followed
  by a long tail of the same document (trained).  ASCOT uses the plan as the
  span prefix, so late spans learn to use distant information and maintain
  names, facts, formatting and terminology across the whole document.
  HTML is only one special case: headings/tags/classes are extracted when
  present, but ordinary articles, books, Markdown, reports and code documents
  use the same generic path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from array import array
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm always present via transformers
    def tqdm(x, **kw):
        return x

from ..core.paths import DATASET_CACHE_DIR
from .formats import DatasetSpec, Example, load_examples

CACHE_FMT = 5
_TOKEN_BATCH = 64          # examples per tokenization batch
_PROGRESS_EVERY = 20       # UI progress callback cadence (batches)


def _emit(progress, msg: str, phase: str | None = None,
          current: int | float | None = None,
          total: int | float | None = None) -> None:
    """Call a progress callback with optional structured fields.

    Older callbacks accept only ``msg``; structured kwargs are best-effort so
    both the console and the Web UI keep working.
    """
    if not progress:
        return
    try:
        progress(msg, phase=phase, current=current, total=total)
    except TypeError:
        try:
            progress(msg)
        except Exception:
            pass
    except Exception:
        pass


@contextmanager
def _heartbeat(progress, label: str, every: float = 2.0):
    """Report elapsed time for an opaque operation (torch.load/save).

    The console normally shows a tqdm bar; this heartbeat is what proves that
    a long cache load is still alive instead of frozen.
    """
    stop = threading.Event()
    t0 = time.time()

    def run():
        while not stop.wait(every):
            _emit(progress, f"{label} ... {time.time() - t0:.0f}s",
                  phase="heartbeat", current=round(time.time() - t0, 1), total=None)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=0.2)


@dataclass
class BuildConfig:
    mode: str = "sft"                 # "sft" | "pt"
    context_length: int = 32768
    val_ratio: float = 0.02
    min_chars: int = 10
    max_chars: int = 2_000_000
    dedup: bool = True
    seed: int = 0
    examples_per_step: int = 0        # 0 = auto (fill context); N = exactly N examples per step
    max_sequences: int = 4000         # safety cap on packed sequences (0 = unlimited)
    # SFT extras
    train_on_user: bool = False
    add_generation_prompt: bool = False
    # PT extras
    join_text: str = "\n\n"
    # v5: data order / small-source mixing
    dataset_order: str = "front"      # front | shuffle
    # v5: generic long-document plan/consistency learning
    long_doc_structure: bool = True
    long_doc_min_chars: int = 1200
    long_doc_min_sentences: int = 4
    long_doc_max_docs: int = 512
    long_doc_tail_chars: int = 16000
    long_doc_prefix_tokens: int = 64
    # preview
    preview_examples: int = 3         # first N examples stored in stats
    # cache identity (hash of sources + packaging settings)
    fingerprint: str = ""


@dataclass
class PackedDataset:
    """Variable-length packed sequences (each <= context_length)."""

    train_ids: list                   # list[Tensor[n_i]] long
    train_mask: list                  # list[Tensor[n_i]] uint8/float
    val_ids: torch.Tensor             # 1-D long (concatenated val tokens)
    val_mask: torch.Tensor            # 1-D
    stats: dict
    train_meta: list = field(default_factory=list)  # per-sequence structure metadata

    def save(self, path: str):
        torch.save({
            "fmt": CACHE_FMT,
            "train_ids": self.train_ids,
            "train_mask": [m.to(torch.uint8) for m in self.train_mask],
            "val_ids": self.val_ids,
            "val_mask": self.val_mask.to(torch.uint8),
            "stats": self.stats,
            "train_meta": self.train_meta,
        }, path)

    @classmethod
    def load(cls, path: str, progress: Callable | None = None) -> "PackedDataset":
        _emit(progress, f"cache load: {Path(path).name}", phase="cache",
              current=0, total=None)
        with _heartbeat(progress, f"cache load {Path(path).name}"):
            d = torch.load(path, map_location="cpu")
        if d.get("fmt") != CACHE_FMT:
            raise ValueError(
                f"cache format {d.get('fmt')} is incompatible (expected {CACHE_FMT}); "
                "the dataset will be rebuilt"
            )
        _emit(progress, f"cache loaded: {len(d['train_ids'])} sequences",
              phase="cache", current=1, total=1)
        return cls(
            d["train_ids"],
            list(d["train_mask"]),
            d["val_ids"],
            d["val_mask"],
            d["stats"],
            d.get("train_meta") or [None] * len(d["train_ids"]),
        )


def _hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()


def _tqdm(iterable, desc: str, total=None, enabled=True):
    return tqdm(
        iterable, desc=desc, total=total, leave=False,
        mininterval=1.0, disable=not enabled, dynamic_ncols=True,
    )


def _tpl_ids_len(tokenizer, messages: list[dict]) -> int | None:
    """Token count of one rendered conversation (robust across versions).

    transformers 5.x may return a ``BatchEncoding`` (a ``UserDict``, NOT a
    ``dict`` subclass) from ``apply_chat_template`` even with
    ``tokenize=True``, so the raw result must never be ``len()``-ed directly;
    always read ``["input_ids"]`` and unwrap batches.
    """
    try:
        out = tokenizer.apply_chat_template(messages, tokenize=True,
                                            return_dict=True)
        try:
            ids = out["input_ids"]
        except Exception:
            ids = out if isinstance(out, (list, tuple)) else None
        if ids is None:
            return None
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)
    except Exception:
        return None


def _probe_assistant_mask(tokenizer, messages: list[dict], n_ids: int) -> list[int] | None:
    """Assistant-token mask for templates without ``{% generation %}`` markers.

    Some widely used chat templates (e.g. the Qwen family) do not contain
    ``{% generation %}`` markers.  ``return_assistant_tokens_mask=True`` then
    yields an **all-zero** mask, which would silently train SFT on nothing
    (loss == 0 for every token, EOS/stop tokens included).

    Fallback: render increasing prefixes of the conversation and use the
    token-length differences as message boundaries.  ChatML-style templates
    are prefix consistent, so assistant turns (including their turn-end /
    EOS token) are recovered exactly.  Returns ``None`` when the template is
    not prefix consistent (the caller then falls back to train-on-everything
    and reports it in the build stats).
    """
    bounds = [0]
    for i in range(1, len(messages) + 1):
        b = _tpl_ids_len(tokenizer, messages[:i])
        if b is None or b < bounds[-1] or b > n_ids:
            return None
        bounds.append(b)
    if bounds[-1] != n_ids:
        return None
    mask = [0] * n_ids
    for j, m in enumerate(messages):
        if str(m.get("role", "")).lower() == "assistant":
            for t in range(bounds[j], bounds[j + 1]):
                mask[t] = 1
    return mask


def _no_template_error() -> ValueError:
    return ValueError(
        "this tokenizer has no chat template, so chat data cannot be rendered "
        "for SFT. Use the PT (continued pre-training) task with text data, or a "
        "model that ships a chat template (instruct models)."
    )


def _unwrap_ids(x):
    """``[[ids]]`` (batched) -> ``[ids]``; leaves flat lists untouched."""
    if x and isinstance(x[0], (list, tuple)):
        return list(x[0])
    return list(x)


def _render_one_chat(tokenizer, messages: list[dict], cfg: BuildConfig,
                     counters: dict) -> tuple[list[int], list[int]]:
    """Render one conversation to (ids, assistant mask) with safe fallbacks."""
    try:
        e = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = list(map(int, _unwrap_ids(e["input_ids"])))
        mask = e.get("assistant_masks")
        if mask is not None:
            mask = list(map(int, _unwrap_ids(mask)))
    except ValueError as e:
        if "chat_template" in str(e) or "chat template" in str(e).lower():
            raise _no_template_error()
        raise
    if cfg.train_on_user:
        return ids, [1] * len(ids)
    if mask is not None and any(mask):
        return ids, mask                      # markers present (LFM/Gemma style)
    probed = _probe_assistant_mask(tokenizer, messages, len(ids))
    if probed is not None:
        # Qwen-style template without markers: exact boundaries recovered by
        # prefix probing (the assistant turn-end / EOS token is included).
        counters["assistant_mask_probed"] = counters.get("assistant_mask_probed", 0) + 1
        return ids, probed
    # last resort: template is not prefix consistent.  Training on the full
    # conversation keeps EOS/stop-token learning intact (unlike an all-zero
    # mask, which would train on nothing at all); reported in build stats.
    counters["assistant_mask_fallback"] = counters.get("assistant_mask_fallback", 0) + 1
    return ids, [1] * len(ids)


def _render_batch(tokenizer, examples: list[Example], cfg: BuildConfig,
                  counters: dict):
    """Tokenize a batch of examples, preserving the input order.

    v7 fixes a data-order bug: the previous implementation emitted every chat
    example before every text example.  With mixed sources that silently
    reordered the front-mix window and made the visible "first 20" order
    differ from what the model actually consumed.  Results are now placed
    back at their original list positions.  Returns ``(ids, mask, source)``.
    """
    out: list[tuple[list[int], list[int], str] | None] = [None] * len(examples)
    chat_pos = [i for i, ex in enumerate(examples) if ex.is_chat()]
    text_pos = [i for i, ex in enumerate(examples) if not ex.is_chat()]

    if chat_pos:
        chats = [examples[i] for i in chat_pos]
        convs = [ex.messages for ex in chats]
        rendered = None
        try:
            enc = tokenizer.apply_chat_template(
                convs,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            ids_list = enc["input_ids"]
            masks_list = enc.get("assistant_masks")
            if masks_list is None:
                raise ValueError("no assistant mask support")
            # an all-zero mask (templates without {% generation %} markers,
            # e.g. Qwen) would silently train on nothing -> per-example probe
            if any(any(int(x) for x in _unwrap_ids(m)) for m in masks_list):
                rendered = []
                for i in range(len(chats)):
                    ids = _unwrap_ids(ids_list[i])
                    mask = _unwrap_ids(masks_list[i])
                    if cfg.train_on_user:
                        mask = [1] * len(ids)
                    rendered.append((list(map(int, ids)), list(map(int, mask))))
        except ValueError as e:
            if "chat_template" in str(e) or "chat template" in str(e).lower():
                raise _no_template_error()
            rendered = None
        except Exception:
            rendered = None
        if rendered is None:
            # per-example fallback (rare templates without batch support,
            # or marker-less templates that need prefix probing)
            rendered = [
                _render_one_chat(tokenizer, ex.messages, cfg, counters)
                for ex in chats
            ]
        for pos, (ids, mask) in zip(chat_pos, rendered):
            out[pos] = (ids, mask, examples[pos].source)

    if text_pos:
        texts = [examples[i] for i in text_pos]
        tlist = [(ex.text or "").strip() for ex in texts]
        enc = tokenizer(tlist, add_special_tokens=False)
        for pos, ids in zip(text_pos, enc["input_ids"]):
            ids = list(map(int, ids))
            out[pos] = (ids, [1] * len(ids), examples[pos].source)

    return [x for x in out if x is not None]


# ---------------------------------------------------------------------------
# v5: source-order merging and generic long-document plan samples
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<([a-zA-Z][\w:-]*)([^<>]*)>", re.S)
_HTML_ATTR_RE = re.compile(
    r"\b(class|id)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))", re.I
)
_HTML_HEADING_RE = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.I | re.S)
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+\S.{0,120}"
    r"|\d+(?:\.\d+)*[.)]\s+\S.{0,120}"
    r"|[A-Z][A-Z0-9][A-Z0-9 _\-:]{2,80})\s*$",
    re.M,
)
_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？])\s*"
    r"|(?<=[\.\!\?])"
    r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bProf\.)"
    r"(?<!\bSt\.)(?<!\bvs\.)(?<!\bNo\.)(?<!\bFig\.)"
    r"(?<!\bInc\.)(?<!\bLtd\.)(?<!\bCo\.)(?<!\bU\.S\.)(?<!\bU\.K\.)"
    r"\s+"
    r"|\n+"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_CJK_SEQ_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_CJK_RUN_RE = re.compile(r"[\u30a0-\u30ff\u30fc]{2,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "were", "have", "has", "had", "but", "not", "you", "your", "they",
    "their", "which", "will", "would", "can", "could", "into", "than",
    "then", "there", "these", "those", "been", "being", "about", "after",
    "before", "also", "more", "most", "such", "some", "when", "where",
    "what", "who", "how", "its", "our", "out", "all", "any", "one",
}
_GENERIC_CJK = {
    "\u5831\u544a", "\u6587\u66f8", "\u5185\u5bb9", "\u65b9\u6cd5",
    "\u7d50\u679c", "\u8003\u5bdf", "\u7d50\u8ad6", "\u80cc\u666f",
    "\u7814\u7a76", "\u5206\u6790", "\u8ffd\u52a0", "\u8ffd\u52a0\u5206\u6790",
    "\u4ee5\u4e0a", "\u4ee5\u4e0b", "\u4eca\u56de", "\u57fa\u76e4",
    "\u8aac\u660e", "\u5fc5\u8981", "\u91cd\u8981", "\u53ef\u80fd",
    "\u5834\u5408", "\u69cb\u6210", "\u7528\u8a9e", "\u7bc0", "\u7ae0",
    "\u672c\u6587", "\u6982\u8981",
}
_GENERIC_TERMS = {
    "additional", "analysis", "text", "document", "section", "report",
    "introduction", "background", "method", "methods", "result", "results",
    "discussion", "conclusion", "chapter", "figure", "table", "example",
    "content", "page", "paragraph", "summary", "overview", "details",
    "class", "id", "html", "body", "head", "div", "span", "href", "src",
    "style", "script", "title", "article", "main", "header", "footer",
    "step", "operator", "operators",
}


def _plain_text(text: str) -> str:
    """Strip markup for sentence/plan extraction (HTML cues stay separate)."""
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _clean_title(s: str) -> str:
    s = re.sub(r"^\s*#{1,6}\s+", "", s or "")
    s = re.sub(r"^\s*(?://+|/\*+|\*+|<!--)\s*", "", s)
    s = re.sub(r"^\s*[`'\"]{1,3}\s*", "", s)
    s = re.sub(r"^\s*\d+(?:\.\d+)*[.)]\s+", "", s)
    s = re.sub(r"\s*[`'\"]{1,3}\s*$", "", s)
    s = re.sub(r"\s*(?:\*/|-->)\s*$", "", s)
    return " ".join(s.split())[:260].strip()


def _looks_like_html(text: str) -> bool:
    """HTML is a structural hint, not a requirement for long-doc learning."""
    if not text:
        return False
    low = text.lower()
    if not any(tag in low for tag in (
            "<html", "<body", "<div", "<section", "<article", "<main",
            "<table", "<ul", "<ol", "<form", "<style", "<script")):
        return False
    return low.count("</") >= 3


def _looks_like_long_doc(text: str, min_chars: int,
                         min_sentences: int) -> bool:
    """Generic long-document test: prose, headings, HTML or structured text."""
    if not text or len(text) < max(1, int(min_chars)):
        return False
    if _looks_like_html(text):
        return True
    sentences = _split_sentences(text)
    if len(sentences) >= max(1, int(min_sentences)):
        return True
    if len(_outline(text)) >= 2 and len(text) >= min_chars:
        return True
    # Long, line-structured documents (logs, code explanations, reports).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 8 and len(text) >= min_chars


def _example_document(ex: Example, min_chars: int,
                      min_sentences: int) -> str | None:
    if ex.is_chat():
        candidates = []
        for m in ex.messages or []:
            c = str(m.get("content") or "")
            if _looks_like_long_doc(c, min_chars, min_sentences):
                candidates.append((str(m.get("role") or ""), c))
        if not candidates:
            return None
        # Prefer the assistant's document: that is the part the model must
        # actually generate and keep consistent.
        assistants = [c for role, c in candidates if role == "assistant"]
        pool = assistants or [c for _, c in candidates]
        return max(pool, key=len)
    text = ex.text or ""
    return text if _looks_like_long_doc(text, min_chars, min_sentences) else None


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = _SENT_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = " ".join(p.split())
        if p:
            out.append(p)
    return out


def _outline(text: str) -> list[str]:
    """Title/headings/list labels for any format (Markdown, HTML, numbers)."""
    out: list[str] = []
    seen = set()

    def add(s: str):
        s = " ".join(str(s).split())[:120].strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)

    for m in _HTML_HEADING_RE.finditer(text[:60000]):
        inner = re.sub(r"<[^>]+>", " ", m.group(1))
        add(inner)
    for m in _HEADING_RE.finditer(text[:60000]):
        line = m.group(0).strip()
        line = re.sub(r"^#{1,6}\s+", "", line)
        add(line)
    lines = [ln.strip() for ln in text[:20000].splitlines() if ln.strip()]
    if lines:
        first = _clean_title(lines[0])
        if len(first) <= 100 and not first.endswith((".", "。", "!", "！", "?", "？")):
            add(first)
    for ln in lines:
        if re.match(r"^[-*•]\s+\S", ln) and len(ln) <= 120:
            add(ln)
        if len(out) >= 8:
            break
    return out[:8]


def _doc_units(text: str) -> set[str]:
    """Cheap multilingual units for distant-reference sentence matching."""
    units: set[str] = set()
    low = text.lower()
    for w in _WORD_RE.findall(low):
        if w not in _STOP_WORDS and len(w) >= 3:
            units.add(w)
    for seq in _CJK_SEQ_RE.findall(text):
        # character n-grams stand in for words in Japanese/Chinese
        for n in (2, 3):
            for i in range(min(len(seq) - n + 1, 6000)):
                g = seq[i:i + n]
                if not all("\u3040" <= c <= "\u309f" for c in g):
                    units.add(g)
    return units


def _key_sentences(text: str, limit: int = 3) -> list[str]:
    """Early sentences that share vocabulary with the document's late part.

    These are the facts/names a writer must remember while finishing a long
    document.  The same overlap also makes the samples useful for ordinary
    articles and Markdown, not just HTML.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    n = len(text)
    early_end = min(max(0, n - 1), max(600, int(n * 0.30)))
    late_start = min(max(early_end, int(n * 0.55)), max(0, n - 1))
    late_units = _doc_units(text[late_start:late_start + 30000])
    # Count near-identical sentence patterns (cards, table rows, repeated
    # procedure steps).  Without this penalty, a template repeated dozens of
    # times outranks the short opening fact that actually defines the plan.
    pattern_counts: dict[str, int] = {}
    for s in sentences:
        p = re.sub(r"\d+", "#", s.lower()).strip()
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    early: list[tuple[float, int, str, set[str]]] = []
    pos = 0
    for i, s in enumerate(sentences):
        if pos >= early_end:
            break
        if i == 0:
            continue  # the title is already in the plan
        pos += len(s) + 1
        units = _doc_units(s)
        if not units:
            continue
        overlap = len(units & late_units) if late_units else 0
        # Prefer early sentences with repeated material; the position prior
        # prevents a long run of repeated boilerplate from dominating.
        # Cap the overlap term: a long boilerplate sentence that repeats every
        # name must not outrank a concise opening fact merely by length.
        score = min(2.5, overlap / max(1.0, len(units) ** 0.5))
        # Opening facts are usually the document's plan.
        score += 0.80 if i < 8 else 0.0
        score += 0.45 * (1.0 - min(1.0, pos / max(1, early_end)))
        pattern = re.sub(r"\d+", "#", s.lower()).strip()
        repeats = pattern_counts.get(pattern, 1)
        if repeats > 1:
            score -= 0.55 * math.log1p(repeats - 1)
        if overlap:
            early.append((score, i, s, units))
    if not early:
        # The first sentence is the title, so start the fact fallback after it.
        return sentences[1:limit + 1]
    early.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    used = set()
    selected_units: list[set[str]] = []
    for _, _, s, units in early:
        k = s.lower()
        if k in used:
            continue
        if any((len(units & u) / max(1, len(units | u))) >= 0.70
               for u in selected_units):
            continue
        used.add(k)
        selected_units.append(units)
        out.append(s[:300])
        if len(out) >= limit:
            break
    return out


def _key_terms(text: str, limit: int = 12) -> list[str]:
    """Recurring names/terms/classes shared by the head and the tail."""
    is_html = _looks_like_html(text)
    plain = _plain_text(text) if is_html else text
    early = plain[:max(1200, int(len(plain) * 0.45))]
    late = plain[max(0, int(len(plain) * 0.5)):][:30000]
    scores: dict[str, float] = {}
    covered: set[str] = set()

    def ok_term(term: str) -> bool:
        low = term.lower().strip()
        return (len(low) >= 2 and low not in _GENERIC_TERMS
                and low not in _GENERIC_CJK and low not in _STOP_WORDS)

    def appears_late(term: str) -> bool:
        if not term:
            return False
        if term.isascii():
            pat = (r"(?<![A-Za-z0-9_])" + re.escape(term)
                   + r"(?![A-Za-z0-9_])")
            return re.search(pat, late, re.I) is not None
        return term in late

    # English words and capitalised phrases: names/entities are the most
    # reusable kind of distant reference.
    for m in re.finditer(
            r"\b[A-Z][A-Za-z0-9_\-]{2,}"
            r"(?:[ \t]+[A-Z][A-Za-z0-9_\-]{2,}){0,2}\b",
            early[:20000]):
        phrase = m.group(0).strip()
        low = phrase.lower()
        if (low in _STOP_WORDS or low in _GENERIC_TERMS
                or low.split()[0] in _STOP_WORDS):
            continue
        if appears_late(phrase):
            scores[phrase] = scores.get(phrase, 0) + 3
            covered.add(low)
    # Lower-case common words are intentionally not added here: the plan's
    # key-sentence section already carries the facts, and entity/class terms
    # are far less noisy across languages and document genres.  Code-like
    # identifiers and backticked terms are the exception: they are exact,
    # reusable names in Markdown/code documentation.
    ident_re = re.compile(
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+)+\b"
        r"|\b[a-z][A-Za-z0-9]*(?:\.[A-Za-z0-9_]+)+\b"
        r"|\b[A-Za-z_][A-Za-z0-9_]*[0-9][A-Za-z0-9_]*\b"
    )
    for m in ident_re.finditer(early[:20000]):
        term = m.group(0)
        if ok_term(term) and appears_late(term):
            scores[term] = scores.get(term, 0) + 2
    for m in re.finditer(r"`([^`\n]{2,60})`", early[:20000]):
        term = m.group(1).strip()
        if ok_term(term) and appears_late(term):
            scores[term] = scores.get(term, 0) + 2
    # CJK names/terms: use katakana and kanji runs rather than raw n-grams.
    # This avoids emitting particles/substrings such as hiragana fragments and is much
    # cleaner for Japanese while still working for Chinese kanji compounds.
    cjk_counts: dict[str, int] = {}
    for m in _CJK_RUN_RE.finditer(early[:60000]):
        g = m.group(0)
        if len(g) > 24:
            continue
        if g in late:
            cjk_counts[g] = cjk_counts.get(g, 0) + 1
    chosen_cjk: list[str] = []
    for g, c in sorted(cjk_counts.items(),
                       key=lambda kv: (-(len(kv[0]) * kv[1]), -len(kv[0]), kv[0])):
        if c < 2:
            continue
        if any(g in x or x in g for x in chosen_cjk):
            continue
        chosen_cjk.append(g)
        if len(chosen_cjk) >= 8:
            break
    for g in chosen_cjk:
        scores[g] = scores.get(g, 0) + cjk_counts[g] * max(1, len(g) - 1)
    # HTML class/id values are first-class reusable tokens when present.
    if is_html:
        for a in _HTML_ATTR_RE.finditer(early[:30000]):
            val = (a.group(2) or a.group(3) or a.group(4) or "").strip()
            if val and val in late:
                for part in val.split():
                    if 2 <= len(part) <= 60 and ok_term(part):
                        scores[part] = scores.get(part, 0) + 2
    out: list[str] = []
    seen: set[str] = set()
    for term, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].lower())):
        low = term.lower()
        if low in seen or not ok_term(term):
            continue
        if any(low in x or x in low for x in seen if len(x) >= 3):
            continue
        seen.add(low)
        out.append(term[:80])
        if len(out) >= limit:
            break
    return out

def _html_cues(text: str) -> str:
    """Compact opening-tag/class-id cue list (optional for any document)."""
    head = text[:60000]
    parts: list[str] = []
    seen = set()
    m = re.match(r"\s*(<!doctype[^>]*>)", head, re.I)
    if m:
        parts.append(m.group(1).strip())
    for tag_m in _HTML_TAG_RE.finditer(head):
        tag = tag_m.group(1).lower()
        attrs = tag_m.group(2) or ""
        found = []
        for a in _HTML_ATTR_RE.finditer(attrs):
            name = a.group(1).lower()
            val = a.group(2) or a.group(3) or a.group(4) or ""
            val = " ".join(val.split())[:80]
            if val:
                found.append((name, val))
        key = (tag, tuple(found))
        if key in seen:
            continue
        seen.add(key)
        s = "<" + tag
        for name, val in found:
            s += f' {name}="{val}"'
        s += ">"
        parts.append(s)
        if len(parts) >= 60 or sum(len(x) for x in parts) >= 1800:
            break
    return " ".join(parts)


def _document_summary(text: str) -> tuple[str, list[str]]:
    """Build a compact, generic plan for the beginning of a long document.

    The plan is deliberately structured but format-agnostic:
      * title/first sentence
      * outline/headings
      * compact HTML tag/class cues (when applicable)
      * early key facts that are referenced later
      * recurring names/terms
    """
    head = text[:60000]
    plain = _plain_text(head) if _looks_like_html(text) else head
    parts: list[str] = []
    sentences = _split_sentences(plain)
    title = _clean_title(sentences[0]) if sentences else ""
    if title:
        parts.append(title)
    outline = [o for o in _outline(head)
               if _clean_title(o).lower() != title.lower()]
    if outline:
        parts.append("OUTLINE: " + " | ".join(outline[:6]))
    if _looks_like_html(text):
        cues = _html_cues(text)
        if cues:
            parts.append("STRUCT: " + cues[:1200])
    outline_keys = {_clean_title(o).lower() for o in outline}
    keys = [k for k in _key_sentences(plain, limit=6)
            if _clean_title(k).lower() != title.lower()
            and _clean_title(k).lower() not in outline_keys]
    if keys:
        parts.append("KEY: " + " | ".join(
            " ".join(k.split()) for k in keys[:3]))
    terms = _key_terms(text)
    if terms:
        parts.append("TERMS: " + ", ".join(terms[:10]))
    summary = "\n".join(dict.fromkeys(
        " ".join(p.split()) for p in parts if p.strip()
    ))
    return summary[:12000], terms


def _encode_ids(tokenizer, text: str) -> list[int]:
    try:
        e = tokenizer(text, add_special_tokens=False)
        ids = e["input_ids"] if hasattr(e, "__getitem__") else e
        return list(map(int, _unwrap_ids(ids)))
    except Exception:
        return []


def _find_reference_spans(tokenizer, text: str, ids: list[int],
                          terms: list[str], offset: int = 0,
                          max_spans: int = 32) -> list[list[int]]:
    """Locate later mentions of early plan terms and return token ranges.

    ASCOT boosts these ranges: a span that repeats a name/class/fact from the
    document plan gets extra probability, which directly trains distant
    reference use instead of only generic tail continuation.
    """
    if not terms or not ids:
        return []
    positions: list[int] = []
    try:
        enc = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)
        offs = enc.get("offset_mapping") if hasattr(enc, "get") else None
        enc_ids = enc.get("input_ids") if hasattr(enc, "get") else None
        if offs and enc_ids is not None and len(offs) == len(enc_ids):
            shift = max(0, len(enc_ids) - len(ids))
            starts = [int(a) for a, _ in offs]
            for term in terms:
                for m in re.finditer(re.escape(term), text,
                                     re.I if term.isascii() else 0):
                    s, e = m.span()
                    # Only walk tokens near the match, not the whole tail.
                    i = max(0, bisect_left(starts, s) - 1)
                    while i < len(offs) and starts[i] < e:
                        a, b = offs[i]
                        if b > s:
                            j = i - shift
                            if j >= 0:
                                positions.append(j)
                        i += 1
                    if len(positions) >= 400:
                        break
    except Exception:
        positions = []
    if not positions:
        for term in terms:
            variants = [term, " " + term, term.strip()]
            for v in variants:
                tids = _encode_ids(tokenizer, v)
                if not tids:
                    continue
                for i in range(0, max(0, len(ids) - len(tids) + 1)):
                    if ids[i:i + len(tids)] == tids:
                        positions.extend(range(i, i + len(tids)))
                        break
                if len(positions) >= 400:
                    break
    if not positions:
        return []
    positions = sorted(set(positions))
    ranges: list[list[int]] = []
    start = prev = positions[0]
    for p in positions[1:]:
        if p <= prev + 1:
            prev = p
        else:
            ranges.append([start + offset, prev + offset + 1])
            start = prev = p
    ranges.append([start + offset, prev + offset + 1])
    # Keep the longest/highest-value ranges within a small budget.
    ranges.sort(key=lambda r: -(r[1] - r[0]))
    return ranges[:max_spans]


def _tail_region(text: str, tail_chars: int) -> str:
    if len(text) <= tail_chars:
        # Keep the ending and most of the middle; the opening is in the plan.
        cut = max(600, len(text) // 4)
    else:
        cut = len(text) - tail_chars
    cut = max(0, min(cut, len(text) - 1))
    # Do not start in the middle of a word/sentence merely because a character
    # budget fell there.  Shift to a nearby natural boundary; the shift is
    # bounded so the tail still contains the intended late region.
    probe = text[cut:cut + 400]
    m = re.search(r"[\n。！？]|[.!?](?=\s)", probe)
    if m:
        cut += m.end()
    else:
        space = probe.find(" ")
        if 0 <= space < 120:
            cut += space + 1
    return text[cut:]


def _build_long_doc_sequences(tokenizer, examples: list[Example],
                              cfg: BuildConfig, report,
                              use_bars: bool = True,
                              max_docs: int | None = None) -> list[dict]:
    """Create plan+long-tail samples for arbitrary long documents.

    ``max_docs`` is an internal build cap (the plan-slot reservation); the
    configured ``long_doc_max_docs`` still applies as an upper bound.
    """
    if not cfg.long_doc_structure:
        return []
    eos = getattr(tokenizer, "eos_token_id", None)
    configured = int(cfg.long_doc_max_docs or 0)
    if max_docs is None:
        max_docs = configured
    else:
        max_docs = max(0, int(max_docs))
        if configured > 0:
            max_docs = min(max_docs, configured)
    prefix_budget = max(8, int(cfg.long_doc_prefix_tokens or 64))
    prefix_budget = min(prefix_budget, max(8, cfg.context_length // 6))
    tail_chars = max(1000, min(int(cfg.long_doc_tail_chars or 16000),
                               cfg.context_length * 2))
    min_chars = int(cfg.long_doc_min_chars or 1200)
    min_sentences = int(cfg.long_doc_min_sentences or 4)
    out: list[dict] = []
    seen = set()
    report("長文の文書プラン・一貫性サンプルを抽出中... / extracting long-document plan samples",
           phase="structure", current=0, total=len(examples))
    for i, ex in enumerate(examples, start=1):
        doc = _example_document(ex, min_chars, min_sentences)
        if doc:
            key = hashlib.sha1(doc.encode("utf-8", "ignore")).hexdigest()
            if key not in seen:
                seen.add(key)
                summary, terms = _document_summary(doc)
                head_ids = _encode_ids(tokenizer, summary + "\n")[:prefix_budget]
                if len(head_ids) >= 4:
                    tail = _tail_region(doc, tail_chars)
                    max_tail = max(2, cfg.context_length - len(head_ids) - 2)
                    tail_ids = _encode_ids(tokenizer, tail)
                    if len(tail_ids) > max_tail:
                        tail_ids = tail_ids[-max_tail:]  # keep the ending
                    if len(tail_ids) >= 256:
                        ids = head_ids + tail_ids
                        mask = [0] * len(head_ids) + [1] * len(tail_ids)
                        if eos is not None:
                            ids.append(int(eos))
                            mask.append(1)
                        refs = _find_reference_spans(
                            tokenizer, tail, tail_ids, terms,
                            offset=len(head_ids))
                        out.append({
                            "ids": ids,
                            "mask": mask,
                            "meta": {
                                "source": ex.source,
                                "long_docs": [{
                                    "start": 0,
                                    "head_end": len(head_ids),
                                    "end": len(ids),
                                    "kind": "html" if _looks_like_html(doc) else "text",
                                    "key_terms": terms[:10],
                                    "reference_spans": refs,
                                }],
                            },
                        })
        if i % 50 == 0 or i == len(examples):
            report(f"長文構造サンプル {len(out)}件 / {i}例",
                   phase="structure", current=i, total=len(examples))
        if max_docs > 0 and len(out) >= max_docs:
            break
    return out


def _merge_front(loaded: list[list[Example]], specs: list[DatasetSpec],
                 rng: random.Random) -> tuple[list[Example], dict]:
    """Keep source order and mix smaller sources into the main prefix.

    ``A`` is the source with the most examples.  If it has ``subset=N``, the
    mix window is its first N examples; otherwise the whole A list.  Every
    example from the smaller sources is inserted at a random position inside
    that window, preserving A's own order.
    """
    if not loaded:
        return [], {}
    main_i = max(range(len(loaded)), key=lambda i: len(loaded[i]))
    main = list(loaded[main_i])
    window = specs[main_i].subset or len(main)
    window = max(1, min(int(window), len(main)))
    inserts: list[tuple[int, float, Example]] = []
    mixed = {}
    for i, exs in enumerate(loaded):
        if i == main_i:
            continue
        mixed[specs[i].source] = len(exs)
        for ex in exs:
            inserts.append((rng.randrange(window), rng.random(), ex))
    inserts.sort(key=lambda x: (x[0], x[1]))
    by_pos: dict[int, list[Example]] = {}
    for pos, _, ex in inserts:
        by_pos.setdefault(pos, []).append(ex)
    result: list[Example] = []
    for idx, ex in enumerate(main):
        if idx in by_pos:
            result.extend(by_pos[idx])
        result.append(ex)
    info = {
        "mode": "front",
        "main_source": specs[main_i].source,
        "window": window,
        "mixed": mixed,
        "mixed_total": sum(mixed.values()),
    }
    return result, info


def build(
    tokenizer,
    specs: list[DatasetSpec],
    cfg: BuildConfig | None = None,
    cache_name: str | None = None,
    progress: Callable[[str], None] | None = None,
    use_bars: bool = True,
) -> PackedDataset:
    """Load examples, tokenize in batches and pack into training sequences."""
    specs = list(specs or [])
    if not specs:
        raise ValueError(
            "no dataset sources / データセットソースが指定されていません"
        )
    cfg = cfg or BuildConfig()
    rng = random.Random(cfg.seed)
    eos = getattr(tokenizer, "eos_token_id", None)
    t_build0 = time.time()

    def report(msg: str, phase: str | None = None,
               current: int | float | None = None,
               total: int | float | None = None):
        _emit(progress, msg, phase=phase, current=current, total=total)

    # ------------------------------------------------ 1. load + normalize
    loaded: list[list[Example]] = []
    per_source = {}
    for i, spec in enumerate(specs):
        report(f"loading source {i + 1}/{len(specs)}: {spec.source}",
               phase="load", current=i, total=len(specs))
        t0 = time.time()
        exs = load_examples(spec, bar=use_bars)
        per_source[spec.source] = len(exs)
        loaded.append(exs)
        report(f"loaded {len(exs)} examples from {spec.source} "
               f"({time.time() - t0:.1f}s)",
               phase="load", current=i + 1, total=len(specs))
    front_info = {}
    if cfg.dataset_order == "front" and len(loaded) > 1:
        examples, front_info = _merge_front(loaded, specs, rng)
        report(f"front mix: main={front_info.get('main_source')} "
               f"window={front_info.get('window')} "
               f"mixed={front_info.get('mixed_total')}", phase="load",
               current=len(specs), total=len(specs))
    else:
        examples = [ex for exs in loaded for ex in exs]
        if cfg.dataset_order == "front" and loaded:
            front_info = {
                "mode": "front",
                "main_source": specs[0].source,
                "window": specs[0].subset or len(loaded[0]),
                "mixed": {},
                "mixed_total": 0,
            }
    report(f"loaded {len(examples)} examples from {len(specs)} source(s)",
           phase="load", current=len(specs), total=len(specs))

    # ------------------------------------------------ 2. filter + dedup
    seen = set()
    kept: list[Example] = []
    dropped_len = dropped_dup = 0
    for idx, ex in enumerate(_tqdm(examples, "filter", total=len(examples),
                                   enabled=use_bars), start=1):
        text = json.dumps(ex.messages, ensure_ascii=False) if ex.is_chat() else ex.text
        if text is None or not (cfg.min_chars <= len(text) <= cfg.max_chars):
            dropped_len += 1
            continue
        if cfg.dedup:
            h = _hash_text(text)
            if h in seen:
                dropped_dup += 1
                continue
            seen.add(h)
        kept.append(ex)
        if idx % 200 == 0:
            report(f"filtering {idx}/{len(examples)}", phase="filter",
                   current=idx, total=len(examples))
    if cfg.dataset_order != "front":
        rng.shuffle(kept)
    report(f"kept {len(kept)} examples "
           f"(dropped {dropped_len} by length, {dropped_dup} duplicates)",
           phase="filter", current=len(examples), total=len(examples))

    # first normalized examples for the UI preview (input/output table)
    first_examples = []
    for ex in kept[: max(0, cfg.preview_examples)]:
        if ex.is_chat():
            first_examples.append({
                "kind": "chat",
                "source": ex.source,
                "messages": [
                    {"role": m["role"], "content": (m["content"] or "")[:600]}
                    for m in ex.messages
                ],
            })
        else:
            first_examples.append({
                "kind": "text",
                "source": ex.source,
                "text": (ex.text or "")[:1200],
            })

    # ------------------------------------------------ 3. tokenize (batched)
    ctx = cfg.context_length
    eps = max(0, int(cfg.examples_per_step))
    max_seq = max(0, int(cfg.max_sequences))
    counters: dict = {}              # assistant-mask fallback statistics

    # per-sequence buffers of compact int32/uint8
    # Long-document plan samples are valuable but must not starve the normal
    # stream (especially when max_sequences is small).  Reserve at most 1/8
    # of the cap for them; normal tokenization fills the remaining slots.
    # Do not extract more documents than can be used.
    plan_slots = (max_seq // 8) if max_seq > 0 else None
    if plan_slots == 0:
        long_doc_samples = []
    else:
        long_doc_samples = _build_long_doc_sequences(
            tokenizer, kept, cfg, report, use_bars=use_bars,
            max_docs=plan_slots)
    seqs_ids: list[array] = [array("i", s["ids"]) for s in long_doc_samples]
    seqs_mask: list[array] = [array("b", s["mask"]) for s in long_doc_samples]
    seqs_meta: list[dict | None] = [s["meta"] for s in long_doc_samples]
    seqs_sources: list[dict[str, int]] = [
        {str((s.get("meta") or {}).get("source") or "?"): 1}
        for s in long_doc_samples
    ]
    cur_ids = array("i")
    cur_mask = array("b")
    cur_sources: dict[str, int] = {}
    cur_count = 0           # examples in the current sequence
    tokens_total = 0
    examples_used = 0

    def close_seq():
        nonlocal cur_ids, cur_mask, cur_sources, cur_count
        if len(cur_ids) > 0:
            seqs_ids.append(cur_ids)
            seqs_mask.append(cur_mask)
            seqs_meta.append(None)
            seqs_sources.append(dict(cur_sources))
        cur_ids = array("i")
        cur_mask = array("b")
        cur_sources = {}
        cur_count = 0

    def cap_reached() -> bool:
        return max_seq > 0 and len(seqs_ids) >= max_seq

    n_batches = (len(kept) + _TOKEN_BATCH - 1) // _TOKEN_BATCH
    bar = _tqdm(range(n_batches), "tokenize", enabled=use_bars)
    bi = 0
    done = False
    report(f"tokenizing {len(kept)} examples in {n_batches} batches",
           phase="tokenize", current=0, total=len(kept))
    for b0 in range(0, len(kept), _TOKEN_BATCH):
        if done or cap_reached():
            break
        batch = kept[b0:b0 + _TOKEN_BATCH]
        rendered = _render_batch(tokenizer, batch, cfg, counters)
        for (ids, mask, src) in rendered:
            if cap_reached():
                done = True
                break
            examples_used += 1
            src = str(src or "?")
            # append EOS separator (loss masked off in SFT unless user-tuned)
            if eos is not None:
                ids = ids + [int(eos)]
                mask = mask + [1 if (cfg.mode == "pt" or cfg.train_on_user) else 0]
            if len(ids) > ctx:
                # a single example longer than the context: chunk it
                for c0 in range(0, len(ids), ctx):
                    if cap_reached():
                        done = True
                        break
                    close_seq()
                    cur_ids.extend(ids[c0:c0 + ctx])
                    cur_mask.extend(mask[c0:c0 + ctx])
                    cur_sources[src] = 1
                    cur_count = eps if eps else cur_count  # chunk fills a step
                close_seq()
                continue
            would_overflow = len(cur_ids) + len(ids) > ctx
            count_reached = eps > 0 and cur_count >= eps
            if (cur_ids and (would_overflow or count_reached)):
                close_seq()
            cur_ids.extend(ids)
            cur_mask.extend(mask)
            cur_sources[src] = cur_sources.get(src, 0) + 1
            cur_count += 1
        bi += 1
        bar.update(1)
        done_examples = min(len(kept), bi * _TOKEN_BATCH)
        report(f"tokenized ~{done_examples}/{len(kept)} examples, "
               f"{len(seqs_ids)} sequences",
               phase="tokenize", current=done_examples, total=len(kept))
    bar.close()
    if not done and not cap_reached():
        close_seq()
    report(f"tokenization done: {examples_used} examples, {len(seqs_ids)} sequences"
           + (f", assistant-mask fallback: {counters}" if counters else ""),
           phase="tokenize", current=len(kept), total=len(kept))

    if not seqs_ids:
        raise ValueError(
            f"not enough tokens for one training sequence "
            f"(context_length={ctx}); check the dataset size and filters"
        )

    # ------------------------------------------------ 4. to tensors + split
    report("packing sequences into tensors", phase="pack", current=0,
           total=len(seqs_ids))
    tensors = []
    masks_t = []
    metas_t = []
    for i, (a, m, meta) in enumerate(
            _tqdm(list(zip(seqs_ids, seqs_mask, seqs_meta)), "pack",
                  total=len(seqs_ids), enabled=use_bars), start=1):
        tensors.append(torch.tensor(a, dtype=torch.long))
        masks_t.append(torch.tensor(m, dtype=torch.long))
        metas_t.append(meta)
        tokens_total += tensors[-1].numel()
        if i % 200 == 0 or i == len(seqs_ids):
            report(f"packing sequences {i}/{len(seqs_ids)}", phase="pack",
                   current=i, total=len(seqs_ids))

    n_seq = len(tensors)
    n_val = max(1, int(n_seq * cfg.val_ratio)) if cfg.val_ratio > 0 else 0
    if n_val >= n_seq:
        n_val = max(0, n_seq - 1)
    n_structured = sum(1 for m in metas_t if m)
    if cfg.dataset_order == "front" and n_structured:
        # Long-document plan samples must stay in training; keep at least one
        # normal sequence for training and take validation from the normal end.
        n_normal = n_seq - n_structured
        n_val = min(n_val, max(0, n_normal - 1))

    if cfg.dataset_order == "front":
        # Do not shuffle the training sequences: training consumes the front
        # first, so a small source mixed into the prefix is fully used before
        # a long main source is exhausted.  Validation comes from the end of
        # the normal stream; plan samples are never used for validation.
        normal_positions = [i for i, m in enumerate(metas_t) if not m]
        n_val = min(n_val, max(0, len(normal_positions) - 1))
        val_positions = normal_positions[-n_val:] if n_val else []
        val_set = set(val_positions)
        val_idx = torch.tensor(val_positions, dtype=torch.long)
        train_idx = torch.tensor(
            [i for i in range(n_seq) if i not in val_set], dtype=torch.long)
    else:
        g = torch.Generator().manual_seed(cfg.seed)
        # Long-document plan samples are training material: choose validation
        # windows from normal sequences only, never from the plan set.
        normal_idx = [i for i, m in enumerate(metas_t) if not m]
        struct_idx = [i for i, m in enumerate(metas_t) if m]
        if normal_idx:
            n_val = min(n_val, max(0, len(normal_idx) - 1))
            order = torch.randperm(len(normal_idx), generator=g).tolist()
            normal_order = [normal_idx[i] for i in order]
            val_idx = torch.tensor(normal_order[:n_val], dtype=torch.long)
            train_pool = normal_order[n_val:] + struct_idx
            tr = torch.randperm(len(train_pool), generator=g).tolist()
            train_idx = torch.tensor([train_pool[i] for i in tr],
                                     dtype=torch.long)
        else:
            n_val = 0
            val_idx = torch.tensor([], dtype=torch.long)
            train_idx = torch.tensor(struct_idx, dtype=torch.long)

    train_list = [tensors[i] for i in train_idx.tolist()]
    train_mask_list = [masks_t[i] for i in train_idx.tolist()]
    train_meta_list = [metas_t[i] for i in train_idx.tolist()]
    train_sources_list = [seqs_sources[i] for i in train_idx.tolist()]

    # v5: put the first normal (front-mixed) sequence before the long-doc
    # plan samples.  The plan samples are still early enough to be learned in
    # a short run, while a small dataset mixed into A's prefix is not pushed
    # behind every plan sample.
    if any(train_meta_list):
        items = list(zip(train_list, train_mask_list, train_meta_list,
                         train_sources_list))
        normal = [x for x in items if not x[2]]
        structured = [x for x in items if x[2]]
        ordered = []
        for i, item in enumerate(normal):
            ordered.append(item)
            if i < len(structured):
                ordered.append(structured[i])
        ordered.extend(structured[len(normal):])
        train_list = [x[0] for x in ordered]
        train_mask_list = [x[1] for x in ordered]
        train_meta_list = [x[2] for x in ordered]
        train_sources_list = [x[3] for x in ordered]

    # validation: concatenate a couple of sequences into one 1-D stream
    val_stream_ids = array("i")
    val_stream_mask = array("b")
    val_seq_lengths: list[int] = []
    for i in val_idx.tolist():
        n_add = tensors[i].numel()
        val_stream_ids.extend(tensors[i].tolist())
        val_stream_mask.extend(masks_t[i].tolist())
        val_seq_lengths.append(n_add)
        if len(val_stream_ids) >= 65536:
            break
    val_ids = torch.tensor(val_stream_ids, dtype=torch.long)
    val_mask = torch.tensor(val_stream_mask, dtype=torch.long)

    # v7: human-readable provenance for the first 20 examples/sequences.
    # The UI draws these as coloured A/B/C badges so the actual consumption
    # order (including front mixing and long-document slots) is visible.
    all_sources: list[str] = list(per_source.keys())
    for counts in seqs_sources:
        for src in counts:
            if src not in all_sources:
                all_sources.append(src)

    def _label(i: int) -> str:
        # A..Z, then AA, AB, ... (more than 26 sources is rare but supported)
        out = ""
        n = i
        while True:
            out = chr(65 + (n % 26)) + out
            n = n // 26 - 1
            if n < 0:
                return out

    source_legend = {src: _label(i) for i, src in enumerate(all_sources)}
    legend = [{"label": source_legend[s], "source": s} for s in all_sources]

    def _seq_labels(counts: dict[str, int]) -> list[str]:
        return sorted(
            {source_legend.get(s, "?") for s in (counts or {})},
            key=lambda x: (len(x), x),
        )

    example_order = []
    # Only examples actually consumed before the packing cap are "passed to
    # the model"; showing 20 when max_sequences stopped after 10 is misleading.
    for i, ex in enumerate(kept[:min(20, max(0, examples_used))], start=1):
        example_order.append({
            "index": i,
            "label": source_legend.get(ex.source, "?"),
            "source": ex.source,
            "kind": "chat" if ex.is_chat() else "text",
        })
    sequence_sources = []
    for i, counts in enumerate(train_sources_list[:20], start=1):
        labels = _seq_labels(counts)
        primary = (max(counts.items(), key=lambda kv: kv[1])[0]
                   if counts else "?")
        sequence_sources.append({
            "index": i,
            "label": source_legend.get(primary, "?"),
            "labels": labels,
            "source": primary,
            "counts": counts,
        })

    stats = {
        "fmt": CACHE_FMT,
        "examples_total": len(examples),
        "examples_kept": len(kept),
        "examples_used": examples_used,
        "dropped_length": dropped_len,
        "dropped_duplicate": dropped_dup,
        "tokens_total": tokens_total,
        "sequences": n_seq,
        "train_sequences": len(train_list),
        "val_sequences": int(val_idx.numel()),
        "val_seq_lengths": val_seq_lengths,
        "context_length": ctx,
        "examples_per_step_setting": eps,
        "avg_examples_per_sequence": round(examples_used / max(1, n_seq), 2),
        "avg_tokens_per_sequence": round(tokens_total / max(1, n_seq), 1),
        "loss_token_ratio": round(
            float(torch.cat(masks_t).float().mean().item()), 4
        ) if masks_t else 0.0,
        "max_sequences_cap": max_seq,
        "per_source": per_source,
        "source_legend": legend,
        "example_order": example_order,
        "sequence_sources": sequence_sources,
        "sequence_order": "".join(x["label"] for x in sequence_sources),
        "dataset_order": cfg.dataset_order,
        "front_mix": front_info,
        "long_doc_sequences": len(long_doc_samples),
        "long_doc_enabled": bool(cfg.long_doc_structure),
        "first_examples": first_examples,
        "mode": cfg.mode,
        "eos_token_id": eos,
        "assistant_mask_probed": counters.get("assistant_mask_probed", 0),
        "assistant_mask_fallback": counters.get("assistant_mask_fallback", 0),
        # hash of sources + every build-relevant setting; the trainer refuses
        # to reuse a cache whose fingerprint no longer matches the config
        "fingerprint": cfg.fingerprint,
        "build_seconds": round(time.time() - t_build0, 2),
    }

    packed = PackedDataset(
        train_ids=train_list,
        train_mask=train_mask_list,
        val_ids=val_ids,
        val_mask=val_mask,
        stats=stats,
        train_meta=train_meta_list,
    )
    if cache_name:
        path = os.path.join(DATASET_CACHE_DIR, cache_name + ".pt")
        _emit(progress, f"saving cache: {cache_name}.pt", phase="save",
              current=0, total=1)
        with _heartbeat(progress, "saving cache"):
            packed.save(path)
            with open(path + ".json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        _emit(progress, f"cache saved: {path}", phase="save", current=1, total=1)
    return packed
