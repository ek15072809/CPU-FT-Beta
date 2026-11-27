"""Local Web UI server (requirement 3/10/13).

Standard-library HTTP server; no web framework needed on Windows.  The UI is
used for configuration and dataset building.  When training starts, the
server spawns the training process (with its own console on Windows),
responds, and then SHUTS ITSELF DOWN so the browser can be closed while the
CUI shows progress.  When training finishes, the trainer relaunches the
server with the results view.

v2: model info goes through the remote-code-safe config resolver; the
dataset preview endpoint also returns the first *normalized* example so the
UI can show an input/output table.

v3: every model/tokenizer/dataset access goes through the shared loader
helpers, so quoted local paths and repositories that require
``trust_remote_code`` (LFM3-style ``auto_map``) work from the Web UI too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..core.config import PRESETS, ConfigError, TrainConfig
from ..core.jsonutil import dumps as json_dumps
from ..core.paths import (
    DATASETS_DIR, DATASET_CACHE_DIR, EXPORT_DIR, RUNS_DIR, STATE_DIR, LOG_DIR,
    cache_size_bytes, human_size,
)

STATIC_DIR = Path(__file__).parent / "static"
STATE_FILE = STATE_DIR / "state.json"


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _safe_run_name(run):
    """Return a single safe directory name, or None.

    Run names are directory names, not paths.  Keeping this check local means a
    legacy config with a separator still fails the same way it did before, but
    API reads and new training runs cannot escape ``runs/``.
    """
    if not run:
        return None
    name = str(run)
    if name in (".", ".."):
        return None
    if any(ch in name for ch in '/\\<>:"|?*\x00'):
        return None
    if name.endswith((" ", ".")):
        return None
    return name


def _run_dir(run):
    name = _safe_run_name(run)
    return None if name is None else Path(RUNS_DIR) / name


BUILD_STATUS = {
    "running": False, "progress": [], "done": None, "error": None,
    "stats": None, "phase": None, "message": "", "current": 0,
    "total": 0, "percent": 0.0,
}


def _load_state_config() -> TrainConfig:
    if STATE_FILE.exists():
        try:
            return TrainConfig.from_json(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return TrainConfig()


def _save_state_config(cfg: TrainConfig):
    STATE_FILE.write_text(cfg.to_json(), encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    lang = "ja"

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # keep the console clean

    def _json(self, obj, code=200):
        # Never emit NaN/Infinity: JavaScript's JSON.parse rejects them.
        body = json_dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._file("index.html", "text/html")
        if url.path.startswith("/static/"):
            name = url.path[len("/static/"):]
            ctype = {"js": "text/javascript", "css": "text/css",
                     "html": "text/html", "svg": "image/svg+xml"}.get(name.rsplit(".", 1)[-1], "text/plain")
            return self._file(name, ctype)
        if url.path == "/api/state":
            return self._api_state(parse_qs(url.query))
        if url.path == "/api/build/status":
            return self._json({k: v for k, v in BUILD_STATUS.items()})
        if url.path == "/api/models":
            return self._api_models()
        if url.path == "/api/results":
            q = parse_qs(url.query)
            return self._api_results(q.get("run", [None])[0])
        if url.path == "/api/log":
            q = parse_qs(url.query)
            return self._api_log(q.get("run", [None])[0])
        if url.path == "/api/plot":
            q = parse_qs(url.query)
            return self._api_plot(q.get("run", [None])[0],
                                  q.get("name", [""])[0])
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/config":
                return self._api_config()
            if url.path == "/api/model/info":
                return self._api_model_info()
            if url.path == "/api/dataset/preview":
                return self._api_dataset_preview()
            if url.path == "/api/dataset/build":
                return self._api_dataset_build()
            if url.path == "/api/cli-command":
                return self._api_cli_command()
            if url.path == "/api/train/start":
                return self._api_train_start()
            if url.path == "/api/generate":
                return self._api_generate()
            if url.path == "/api/estimate":
                return self._api_estimate()
        except ConfigError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": repr(e)[:400]}, 500)
        return self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    def _file(self, name, ctype):
        root = Path(STATIC_DIR).resolve()
        path = (root / name).resolve()
        # Keep ``/static/../../...`` inside the static directory; without this
        # check a local browser request could read arbitrary project files.
        if not path.is_relative_to(root) or not path.is_file():
            return self._json({"error": "missing"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def _api_state(self, q):
        cfg = _load_state_config()
        runs = []
        if Path(RUNS_DIR).exists():
            for child in sorted(Path(RUNS_DIR).iterdir()):
                if (child / "summary.json").exists():
                    runs.append(child.name)
        datasets = []
        if Path(DATASET_CACHE_DIR).exists():
            for f in sorted(Path(DATASET_CACHE_DIR).glob("*.pt")):
                stats = {}
                # Cache stats live next to the .pt as "<full filename>.json"
                # (e.g. run_pt_256_abcd.pt.json).  Using f.stem dropped the
                # ".pt" and silently made every built dataset show "?" with no
                # source/order visualization.
                jf = f.parent / (f.name + ".json")
                if jf.exists():
                    try:
                        stats = json.loads(jf.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                datasets.append({"name": f.stem, "file": f.name,
                                 "mb": round(f.stat().st_size / 1e6, 1),
                                 "mtime": f.stat().st_mtime,
                                 "stats": stats})
        return self._json({
            "config": json.loads(cfg.to_json()),
            "presets": {k: {"ascot": v["ascot"], "lr": v["lr"]} for k, v in PRESETS.items()},
            "runs": runs,
            "datasets": datasets,
            "exports": [d.name for d in sorted(Path(EXPORT_DIR).iterdir())] if Path(EXPORT_DIR).exists() else [],
            "cache": {"dir": str(DATASETS_DIR.parent),
                      "size": human_size(cache_size_bytes())},
            "lang": self.lang,
            "version": __import__("cpu_finetuning").__version__,
            "training_pid_file": str(STATE_DIR / "training.pid"),
        })

    def _api_config(self):
        body = self._body()
        cfg = TrainConfig.from_json(json.dumps(body.get("config", body)))
        _save_state_config(cfg)
        return self._json({"ok": True, "saved": str(STATE_FILE)})

    def _api_model_info(self):
        body = self._body()
        source = body.get("source", "")
        from ..models.loader import resolve_config, _resolve
        cfg = resolve_config(_resolve(source))
        text_cfg = getattr(cfg, "text_config", cfg) if getattr(cfg, "text_config", None) else cfg
        info = {
            "model_type": getattr(text_cfg, "model_type", "?"),
            "hidden_size": getattr(text_cfg, "hidden_size", None),
            "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
            "vocab_size": getattr(text_cfg, "vocab_size", None),
            "layer_types": getattr(text_cfg, "layer_types", None),
            "max_position_embeddings": getattr(text_cfg, "max_position_embeddings", None),
            "tie_word_embeddings": getattr(text_cfg, "tie_word_embeddings", None),
            "supported": True,
        }
        return self._json(info)

    @staticmethod
    def _preview_one(raw_spec: dict, n: int) -> dict:
        from ..data.formats import DatasetSpec, detect_format, normalize, load_raw
        source = str(raw_spec.get("source") or "").strip()
        if not source:
            return {"source": source, "error": "dataset source is empty"}
        try:
            spec = DatasetSpec(**raw_spec)
            ds = load_raw(spec)
            if spec.subset:
                ds = ds.select(range(min(spec.subset, len(ds))))
            # Do not materialize a large dataset just to show ``n`` rows.
            try:
                preview_rows = ds.select(range(min(n, len(ds))))
            except Exception:
                preview_rows = list(ds)[:n]
            rows = []
            first_examples = []
            for row in preview_rows:
                d = dict(row)
                fmt = detect_format(d)
                rows.append({
                    "format": fmt,
                    "row": {k: str(v)[:200] for k, v in d.items()},
                })
                if not first_examples:
                    ex = normalize(d, fmt=fmt, source=spec.source)
                    if ex is not None:
                        if ex.is_chat():
                            first_examples.append({
                                "kind": "chat",
                                "messages": [
                                    {"role": m["role"],
                                     "content": (m["content"] or "")[:600]}
                                    for m in ex.messages
                                ],
                            })
                        else:
                            first_examples.append({
                                "kind": "text",
                                "text": (ex.text or "")[:1200],
                            })
            return {
                "source": source,
                "total": len(ds),
                "columns": list(ds.column_names),
                "rows": rows,
                "first_examples": first_examples,
            }
        except Exception as e:
            return {"source": source, "error": repr(e)[:500]}

    def _api_dataset_preview(self):
        body = self._body()
        # v7: preview every configured source, not only the first one.  The
        # old UI sent specs[0] and therefore hid all additional datasets.
        raw_specs = body.get("specs")
        if not isinstance(raw_specs, list):
            raw_specs = [body.get("spec") or {}]
        raw_specs = [s for s in raw_specs if isinstance(s, dict)
                     and str(s.get("source") or "").strip()]
        if not raw_specs:
            return self._json({
                "error": "データセットソースが空です / dataset source is empty"
            }, 400)
        n = max(1, min(20, int(body.get("n", 3) or 3)))
        sources = [self._preview_one(s, n) for s in raw_specs]
        ok = [s for s in sources if not s.get("error")]
        first = sources[0]
        return self._json({
            "sources": sources,
            "count": len(sources),
            "total": first.get("total"),
            "columns": first.get("columns", []),
            "rows": first.get("rows", []),
            "first_examples": first.get("first_examples", []),
            "error": (None if ok else " / ".join(
                f"{s.get('source')}: {s.get('error')}" for s in sources)),
        })

    def _api_dataset_build(self):
        if BUILD_STATUS["running"]:
            return self._json({"ok": False, "error": "build already running"})
        body = self._body()
        cfg = TrainConfig.from_json(json.dumps(body.get("config", body)))
        try:
            cfg.validate_data_sources()
        except ConfigError as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        _save_state_config(cfg)

        def work():
            BUILD_STATUS.update(running=True, progress=[], done=None,
                                error=None, stats=None, phase=None,
                                message="", current=0, total=0, percent=0.0)
            try:
                from ..data.builder import build as build_ds

                def progress(msg, phase=None, current=None, total=None):
                    BUILD_STATUS["message"] = str(msg)
                    BUILD_STATUS["progress"].append(str(msg))
                    BUILD_STATUS["progress"] = BUILD_STATUS["progress"][-50:]
                    if phase:
                        BUILD_STATUS["phase"] = phase
                    if current is not None:
                        BUILD_STATUS["current"] = current
                    if total:
                        BUILD_STATUS["total"] = total
                        BUILD_STATUS["percent"] = round(
                            min(100.0, 100.0 * float(current) / float(total)), 1)

                from ..models.loader import load_tokenizer
                tok = load_tokenizer(cfg.model_source)
                cache_name = cfg.dataset_cache_name()
                packed = build_ds(tok, cfg.spec_objects(), cfg.build_config(),
                                  cache_name=cache_name, progress=progress,
                                  use_bars=False)  # bars are for consoles
                BUILD_STATUS["stats"] = packed.stats
                BUILD_STATUS["done"] = cache_name
            except Exception as e:
                BUILD_STATUS["error"] = repr(e)[:500]
            finally:
                BUILD_STATUS["running"] = False

        threading.Thread(target=work, daemon=True).start()
        return self._json({"ok": True})

    def _api_cli_command(self):
        body = self._body()
        cfg = TrainConfig.from_json(json.dumps(body.get("config", body)))
        cfg.resume = _as_bool(body.get("resume"), cfg.resume)
        from ..cli import build_cli_command
        return self._json({"command": build_cli_command(cfg)})

    def _api_train_start(self):
        body = self._body()
        cfg = TrainConfig.from_json(json.dumps(body.get("config", body)))
        cfg.resume = _as_bool(body.get("resume"), cfg.resume)
        run_name = _safe_run_name(cfg.run_name)
        if run_name is None:
            return self._json({
                "error": "run名にパス区切り文字は使えません / "
                         "run name must be a single directory name"
            }, 400)
        cfg.run_name = run_name
        if not str(cfg.model_source or "").strip():
            return self._json({
                "error": "モデルが指定されていません / model source is empty"
            }, 400)
        try:
            cfg.validate_data_sources()
        except ConfigError as e:
            return self._json({"error": str(e)}, 400)
        _save_state_config(cfg)
        run_dir = Path(RUNS_DIR) / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(cfg.to_json(), encoding="utf-8")
        try:
            from ..core.usage import touch_model
            touch_model(cfg.model_source, kind="base", mode="train",
                        run_name=cfg.run_name)
        except Exception:
            pass

        pkg_root = Path(__file__).resolve().parents[2]
        cmd = [sys.executable, "-m", "cpu_finetuning", "train",
               "--config", str(cfg_path), "--lang", cfg.lang,
               "--hold-console"]
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
        proc = subprocess.Popen(cmd, cwd=str(pkg_root), creationflags=flags)
        (STATE_DIR / "training.pid").write_text(str(proc.pid))
        # shut the web UI down (browser can be closed while CUI shows progress)
        threading.Timer(1.5, lambda: os._exit(0)).start()
        return self._json({"ok": True, "config": str(cfg_path), "pid": proc.pid,
                           "message": "training started; web UI will close now"})

    def _api_results(self, run):
        run_dir = _run_dir(run)
        if run_dir is None:
            return self._json({"error": "run?"}, 400)
        summary, history, val = {}, [], []
        try:
            if (run_dir / "summary.json").exists():
                summary = json.loads(
                    (run_dir / "summary.json").read_text(encoding="utf-8"))
            if (run_dir / "history.json").exists():
                d = json.loads(
                    (run_dir / "history.json").read_text(encoding="utf-8"))
                history = d.get("train", [])[-2000:]
                val = d.get("val", [])
        except Exception as e:
            return self._json({"error": repr(e)[:300]}, 500)
        return self._json({"summary": summary, "history": history, "val": val})

    def _api_log(self, run):
        run_dir = _run_dir(run)
        if run_dir is None:
            return self._json({"error": "run?"}, 400)
        # v10 writes training logs to logs/<run>.log.  Keep reading the v9
        # runs/<run>/train.log location so old runs still show their log.
        path = Path(LOG_DIR) / f"{run_dir.name}.log"
        if not path.exists():
            path = run_dir / "train.log"
        if not path.exists():
            return self._json({"text": "", "exists": False})
        try:
            # Keep the tail bounded: a 1000-step run is small, but resumed
            # runs can append for many hours.
            text = path.read_text(encoding="utf-8", errors="replace")
            return self._json({"text": text[-120000:], "exists": True})
        except Exception as e:
            return self._json({"error": repr(e)[:300]}, 500)

    def _api_plot(self, run, name):
        run_dir = _run_dir(run)
        if run_dir is None or not name:
            return self._json({"error": "run/name?"}, 400)
        import re
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not name.endswith(".png"):
            return self._json({"error": "invalid plot name"}, 400)
        path = (run_dir / "plots" / name).resolve()
        plots_root = (run_dir / "plots").resolve()
        try:
            path.relative_to(plots_root)
        except ValueError:
            return self._json({"error": "invalid plot path"}, 400)
        if not path.exists():
            return self._json({"error": "plot not found"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _api_estimate(self):
        body = self._body()
        cfg = TrainConfig.from_json(json.dumps(body.get("config", body)))
        try:
            cfg.validate_data_sources()
        except ConfigError as e:
            return self._json({"error": str(e)}, 400)
        from ..core.estimate import estimate_for_config
        est = estimate_for_config(cfg)
        if est.get("error"):
            return self._json(est, 400)
        return self._json(est)

    def _api_models(self):
        from ..core.usage import list_models
        return self._json({"models": list_models()})

    def _api_generate(self):
        """Generate with one model, or sequentially compare base vs tuned.

        The comparison deliberately loads one model at a time and frees it
        before the next one starts.  Loading both 1.7B models at once does not
        fit on a 16 GB CPU machine and was a common source of failures.
        """
        body = self._body()
        from ..core.usage import (
            base_model_for, canon_path, resolve_model_source, touch_model,
        )

        raw_target = str(body.get("model", "") or "").strip()
        if not raw_target:
            return self._json({
                "error": "モデルを選択してください / select a model"
            }, 400)
        try:
            target = resolve_model_source(raw_target)
        except Exception as e:
            return self._json({"error": str(e)[:500]}, 400)

        compare = _as_bool(body.get("compare"), False)
        base = ""
        if compare:
            base = str(body.get("base_model") or base_model_for(target) or "")
            if base:
                try:
                    base = resolve_model_source(base)
                except Exception:
                    base = ""
            if base and canon_path(base) == canon_path(target):
                base = ""

        prompt = str(body.get("prompt", "Hello!"))
        opts = {
            "max_new_tokens": max(1, min(512, int(body.get("max_new_tokens", 96)))),
            "chat": _as_bool(body.get("chat"), True),
            "temperature": float(body.get("temperature", 0.7)),
            "top_k": int(body.get("top_k", 50)),
            "top_p": float(body.get("top_p", 0.95)),
            "device": str(body.get("device", "auto") or "auto"),
        }

        try:
            if compare and base:
                before = self._generate_one(base, prompt, opts)
                after = self._generate_one(target, prompt, opts)
                touch_model(base, kind="base", mode="generate",
                            base_model=base)
                touch_model(target, kind="trained", mode="generate",
                            base_model=base)
                return self._json({
                    "compare": True,
                    "base": before,
                    "tuned": after,
                    "text": after["text"],
                })
            out = self._generate_one(target, prompt, opts)
            touch_model(target,
                        kind="trained" if out.get("kind") == "trained" else "base",
                        mode="generate")
            return self._json(out)
        except Exception as e:
            return self._json({"error": repr(e)[:500]}, 500)

    @staticmethod
    def _free_model_memory():
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("msvcrt")._heapmin()
        except Exception:
            pass

    def _generate_one(self, source: str, prompt: str, opts: dict) -> dict:
        import torch
        from ..models.loader import (
            load_causal_lm, load_tokenizer, resolve_compute_dtype, resolve_config,
        )
        from ..core.device import resolve_device
        from pathlib import Path

        tok = load_tokenizer(source)
        cfg = resolve_config(source)
        dev_choice = resolve_device(opts.get("device", "auto"))
        device, dev_reason = dev_choice.device, dev_choice.label
        dtype, reason = resolve_compute_dtype("auto", source, cfg,
                                               device=device)
        model = load_causal_lm(source, dtype=dtype).to(device)
        model.eval()
        try:
            if opts.get("chat", True):
                try:
                    enc = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=True,
                        return_dict=True, add_generation_prompt=True)
                    raw = enc["input_ids"]
                    ids = torch.tensor(raw, dtype=torch.long)
                except Exception:
                    ids = tok(prompt, return_tensors="pt").input_ids
            else:
                ids = tok(prompt, return_tensors="pt").input_ids
            # apply_chat_template may return a flat list, [1, N], or even a
            # nested batch.  v8 assumed shape[1] existed and crashed with
            # IndexError('tuple index out of range') for flat token ids.
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            if ids.dim() == 0:
                ids = ids.reshape(1, 1)
            elif ids.dim() == 1:
                ids = ids.unsqueeze(0)
            elif ids.dim() > 2:
                ids = ids.reshape(-1, ids.shape[-1])[:1]
            ids = ids.to(device)
            prompt_len = int(ids.shape[-1])

            temp = float(opts.get("temperature", 0.7))
            gen_kwargs = {
                "max_new_tokens": int(opts.get("max_new_tokens", 96)),
                "do_sample": temp > 0,
                "repetition_penalty": 1.05,
            }
            if temp > 0:
                gen_kwargs.update(temperature=temp,
                                  top_k=int(opts.get("top_k", 50)),
                                  top_p=float(opts.get("top_p", 0.95)))
            with torch.no_grad():
                out = model.generate(ids, **gen_kwargs)
            new_ids = out[0][prompt_len:].detach().to("cpu")
            text = tok.decode(new_ids, skip_special_tokens=True)
            p = Path(source)
            kind = "trained" if p.is_dir() and p.parent.name.lower() == "exports" else "base"
            return {"text": text, "model": source, "kind": kind,
                    "device": str(device),
                    "device_reason": dev_reason,
                    "dtype": str(dtype).replace("torch.", ""),
                    "dtype_reason": reason}
        finally:
            try:
                del model
            except Exception:
                pass
            del tok
            self._free_model_memory()


def serve(port=8765, config_path=None, results_run=None, lang="ja",
          open_browser=True):
    if config_path and Path(config_path).exists():
        try:
            cfg = TrainConfig.from_json(Path(config_path).read_text(encoding="utf-8"))
            _save_state_config(cfg)
        except Exception:
            pass
    Handler.lang = lang
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    if results_run:
        url += f"#results:{results_run}"
    print(f"CPU-FineTuning Web UI: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
