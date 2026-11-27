"""Project path management (v2 folder layout).

All caches, downloaded models and datasets live INSIDE the project folder so
that nothing is written to the user profile (no ~/.cache and friends).

v2 layout (requirement: downloaded models / datasets / built datasets are
grouped into clearly named folders):

    models/huggingface/...     <- HF hub cache (downloaded models)
    datasets/huggingface/...   <- HF datasets cache (downloaded raw datasets)
    datasets/built/...         <- packed, ready-to-train dataset caches
    checkpoints/               <- training checkpoints
    exports/                   <- exported full weights
    runs/                      <- per-run history and configs
    logs/                      <- logs
    state/                     <- app state (UI settings, pid files)
    .cache/tmp/                <- transient temp files only

When an old v1 ``.cache`` layout is found, its contents are migrated
automatically (best effort).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_project_root() -> Path:
    """Return the directory that contains the cpu_finetuning package."""
    root = Path(__file__).resolve().parent.parent.parent
    if (root / "cpu_finetuning").is_dir():
        return root
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "cpu_finetuning").is_dir():
            return cand
    return Path.cwd()


PROJECT_ROOT: Path = find_project_root()

CACHE_DIR: Path = PROJECT_ROOT / ".cache"          # transient only (tmp)
MODELS_DIR: Path = PROJECT_ROOT / "models"         # downloaded models
HF_CACHE_DIR: Path = MODELS_DIR / "huggingface"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"     # all dataset artifacts
HF_DATASETS_DIR: Path = DATASETS_DIR / "huggingface"
DATASET_CACHE_DIR: Path = DATASETS_DIR / "built"   # packed caches
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
EXPORT_DIR: Path = PROJECT_ROOT / "exports"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
STATE_DIR: Path = PROJECT_ROOT / "state"
LOG_DIR: Path = PROJECT_ROOT / "logs"


def _migrate_v1() -> None:
    """Move v1 ``.cache`` artifacts into the v2 folders (once, best effort).

    Runs BEFORE the v2 directories are created so the plain-move fast path is
    available; when a v2 folder already exists the children are merged.
    """
    old_cache = PROJECT_ROOT / ".cache"
    moves = [
        (old_cache / "huggingface", HF_CACHE_DIR),
        (old_cache / "datasets", DATASET_CACHE_DIR),
        (old_cache / "app", STATE_DIR),
    ]
    for src, dst in moves:
        try:
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.move(str(src), str(dst))
            else:
                for child in src.iterdir():
                    target = dst / child.name
                    if not target.exists():
                        shutil.move(str(child), str(target))
                # drop the old folder when nothing (but empty dirs) remains
                try:
                    if not any(p.is_file() for p in src.rglob("*")):
                        shutil.rmtree(src, ignore_errors=True)
                    else:
                        src.rmdir()
                except OSError:
                    pass
        except Exception:
            pass


_migrate_v1()

for _d in (CACHE_DIR, MODELS_DIR, HF_CACHE_DIR, DATASETS_DIR,
           HF_DATASETS_DIR, DATASET_CACHE_DIR, CHECKPOINT_DIR,
           EXPORT_DIR, RUNS_DIR, STATE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def configure_environment() -> None:
    """Redirect every known cache location into the project folder.

    Must be called before transformers / datasets / huggingface_hub are
    imported for the first time.
    """
    env = os.environ
    # HuggingFace stack -> models/huggingface, datasets/huggingface
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    env["HF_DATASETS_CACHE"] = str(HF_DATASETS_DIR)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
    env["XDG_CACHE_HOME"] = str(CACHE_DIR)
    env["TMPDIR"] = str(CACHE_DIR / "tmp")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # plain http for stability
    # Tokenizers / sentencepiece
    env["TOKENIZERS_PARALLELISM"] = "false"
    # keep torch from grabbing every core when the OS needs responsiveness
    if "OMP_NUM_THREADS" not in env:
        env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) - 1))
    os.makedirs(env["TMPDIR"], exist_ok=True)


configure_environment()


def cache_size_bytes() -> int:
    """Total size of downloaded models + datasets + built caches."""
    total = 0
    for base in (MODELS_DIR, DATASETS_DIR):
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
    return total


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def strip_path_quotes(s: str) -> str:
    """Strip whitespace and surrounding quotes from a pasted path.

    Windows Explorer's "Copy as path" wraps paths in double quotes because of
    spaces; users also paste values quoted by shells or file dialogs.  Both
    straight quotes ("x", 'x') and directional curly quotes (\u201cx\u201d,
    \u2018x\u2019) are handled.
    """
    s = str(s).strip()
    pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]
    for a, b in pairs:
        if len(s) >= 2 and s.startswith(a) and s.endswith(b):
            s = s[1:-1].strip()
            break
    return s


def looks_like_path(s: str) -> bool:
    """Heuristic: does this look like a filesystem path rather than a repo id?"""
    if not s:
        return False
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return True                       # C:\... or C:/...
    if s.startswith(("\\\\", "/", "./", "../", "~")):
        return True                       # UNC, absolute, relative, home
    return "\\" in s
