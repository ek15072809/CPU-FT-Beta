"""The training process (requirement 10/11).

Runs detached from the Web UI.  All progress goes to the console (CUI) in the
configured language at a configurable interval; long build phases show
tqdm progress bars; checkpoints follow the configured policy; on completion
the full weights are exported automatically and the Web UI is relaunched.

v2 additions:
* time limit (``max_hours``) and epoch limit (``max_epochs``) cutoffs,
* gradient-spike skip/scale actions (logged per step),
* variable-length packed sequences (examples_per_step support).
"""

from __future__ import annotations

import gc
import json
import math
import os
import signal
import time
from pathlib import Path

import torch

from ..core.config import TrainConfig
from ..core.device import memory_hygiene, resolve_device
from ..core.jsonutil import dump as json_dump
from ..core.i18n import t
from ..core.paths import EXPORT_DIR, RUNS_DIR, STATE_DIR, human_size, LOG_DIR
from ..core.sleep import SleepGuard
from ..data.builder import build as build_dataset
from ..data.formats import DatasetSpec
from ..models.loader import load_model, shadow_budget_report
from ..models.qlayer import clear_shadows, invalidate_shadows, set_shadow_enabled
from ..train.ascot import AscotStepper
from ..train.checkpoint import CheckpointManager
from ..train.optimizer import Adam8Bit, LRSchedule


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _mem_hygiene(device: str | None = None):
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    memory_hygiene(device)


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return -1.0


class Trainer:
    def __init__(self, cfg: TrainConfig):
        cfg.normalize()
        self.cfg = cfg
        self.lang = cfg.lang
        self.run_dir = Path(RUNS_DIR) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Persist the effective config for both CLI and Web UI runs.  The
        # relaunched results UI must describe this run, not mutable state.json.
        try:
            (self.run_dir / "config.json").write_text(
                cfg.to_json(), encoding="utf-8")
        except Exception:
            pass
        # Keep every training log under logs/ (the run directory still holds
        # config.json, history.json, summary.json and plots).
        self.log_path = Path(LOG_DIR) / f"{cfg.run_name}.log"
        self._log_file = None
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "a",
                                  encoding="utf-8", buffering=1)
        except Exception:
            self._log_file = None

        def _log(msg):
            print(msg, flush=True)
            if self._log_file is not None:
                try:
                    self._log_file.write(str(msg) + "\n")
                except Exception:
                    pass

        self.log = _log
        self.history: list[dict] = []
        self.val_history: list[dict] = []
        self.device_choice = None
        self.device = "cpu"
        self._selected_cache = ""
        self._events: list[dict] = []
        self._prior_events: list[dict] = []
        self._prior_restarts: list[dict] = []

    def _event(self, kind: str, **data):
        rec = {"kind": str(kind), "at": round(time.time(), 3)}
        rec.update(data)
        self._events.append(rec)
        return rec

    # ------------------------------------------------------------------
    def _load_history(self):
        p = self.run_dir / "history.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                self.history = d.get("train", [])
                self.val_history = d.get("val", [])
            except Exception:
                pass
        # A resumed run should keep the milestones that still belong to the
        # restored part of the curve (baseline/best/regression/restart).  The
        # events are otherwise only in memory and the final summary would
        # silently lose every pre-resume event.
        summary_path = self.run_dir / "summary.json"
        if summary_path.exists():
            try:
                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                self._prior_events = list(prev.get("events") or [])
                self._prior_restarts = list(prev.get("restarts") or [])
            except Exception:
                self._prior_events = []
                self._prior_restarts = []

    def _save_history(self):
        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json_dump({"train": self.history, "val": self.val_history}, f)

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        lang = self.lang
        # Deterministic configuration errors must fail before the 1.7B model
        # is loaded (and before the watchdog starts retrying them).
        if not str(cfg.model_source or "").strip():
            from ..core.config import ConfigError
            raise ConfigError(
                "モデルが指定されていません / model source is empty"
            )
        cfg.validate_data_sources()
        guard = SleepGuard(log=self.log)
        guard.start(lang)

        torch.manual_seed(cfg.seed)
        self.device_choice = resolve_device(cfg.device)
        self.device = self.device_choice.device
        # Keep the v8 CPU throughput default even when ``device=auto`` falls
        # back to CPU; only skip it for an actual accelerator.
        if self.device.startswith("cpu"):
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self.log(t("device_selected", lang, device=self.device_choice.label))

        # ---------------- model
        self.log(t("loading_model", lang, src=cfg.model_source))
        # "auto" lets the loader pick a dtype that fits the selected device.
        lm = load_model(cfg.model_source, dtype=cfg.model_dtype,
                        device=self.device)
        model, adapter = lm.model, lm.adapter
        n_params = sum(p.numel() for p in model.parameters())
        try:
            from ..core.usage import touch_model
            touch_model(lm.source, kind="base", mode="train",
                        run_name=cfg.run_name)
        except Exception:
            pass
        self.log(t("model_loaded", lang, params=f"{n_params/1e6:.1f}M",
                   adapter=type(adapter).__name__, size=human_size(lm.master_bytes)))
        backend = lm.int8_backend or {}
        self.log(t("compute_setup", lang, dtype=lm.compute_dtype,
                   reason=lm.dtype_reason,
                   int8="on" if backend.get("enabled") else "off",
                   why=backend.get("reason", "-")))
        self.compute_info = {
            "compute_dtype": lm.compute_dtype,
            "dtype_reason": lm.dtype_reason,
            "int8_backend": backend,
        }

        # ---------------- data
        packed = self._prepare_data(lm.tokenizer)
        train_ids, train_mask = packed.train_ids, packed.train_mask
        n_train = len(train_ids)
        val_ids, val_mask = self._validation_view(packed)
        self.data_stats = {k: v for k, v in packed.stats.items()
                           if k != "first_examples"}
        self._estimate = None
        try:
            from ..core.estimate import estimate_for_config
            est = estimate_for_config(cfg)
            self._estimate = est
            self.log(t("estimate_line", lang,
                       steps=est["steps"], sps=est["sec_per_step"],
                       train=_fmt_time(est["train_seconds"]),
                       build=_fmt_time(est["build_seconds"]),
                       total=_fmt_time(est["total_seconds"])))
        except Exception:
            pass

        # ---------------- training objects
        stepper = AscotStepper(model, adapter, cfg.ascot_config())

        # v5: total_steps=0 means "follow the stop conditions".  The LR
        # schedule needs a horizon: use max_epochs*n_train when epochs are
        # configured, otherwise a first-step time estimate for max_hours.
        # If every stop condition is 0, train exactly one epoch.
        n_train = max(1, len(train_ids))
        one_epoch_default = False
        if int(cfg.total_steps or 0) > 0:
            planned_steps = int(cfg.total_steps)
            # A huge step budget (the v4 default was 1e15) must not disable
            # LR decay when an epoch limit stops the run much earlier.
            if int(cfg.max_epochs or 0) > 0:
                planned_steps = min(planned_steps,
                                    int(cfg.max_epochs) * n_train)
        elif int(cfg.max_epochs or 0) > 0:
            planned_steps = int(cfg.max_epochs) * n_train
        elif float(cfg.max_hours or 0) > 0:
            planned_steps = None
        else:
            planned_steps = n_train
            one_epoch_default = True
        # The LR horizon must match when training actually stops.  v8 planned
        # a 1667-step cosine for an 8-hour run that reached only 1000 steps, so
        # the LR was still at 43% of peak when the time limit hit.  When a
        # time limit is configured, use the estimator (and a measured
        # correction after the first validation) to shorten the horizon.
        est_sps = 0.0
        if self._estimate:
            try:
                est_sps = float(self._estimate.get("sec_per_step") or 0.0)
            except Exception:
                est_sps = 0.0
        if planned_steps is not None:
            loop_limit = planned_steps
            schedule_total = planned_steps
            if float(cfg.max_hours or 0) > 0 and est_sps > 0:
                time_steps = max(
                    10, int(float(cfg.max_hours) * 3600 * 0.95 / est_sps)
                )
                schedule_total = min(planned_steps, time_steps)
        else:
            # The time-based horizon is corrected after the first measured
            # validation; 20s/step is a conservative placeholder.
            loop_limit = 10 ** 12
            if est_sps > 0:
                schedule_total = max(
                    10, int(float(cfg.max_hours) * 3600 * 0.95 / est_sps)
                )
            else:
                schedule_total = max(
                    10, int(float(cfg.max_hours) * 3600 * 0.95 / 20.0)
                )
        self._planned_steps = planned_steps
        self._schedule_total = int(schedule_total)
        self._time_schedule = float(cfg.max_hours or 0) > 0
        self._sched_time_adjusted = False
        self._loop_limit = loop_limit
        planned_epochs = (None if planned_steps is None
                          else max(1, (planned_steps + n_train - 1) // n_train))
        if int(cfg.max_epochs or 0) > 0:
            # If an explicit step budget ends before the epoch limit, show
            # the number of epochs that can actually be reached (e.g. 10
            # steps with 35 sequences must not say "1/2").
            self._epochs_total = (
                min(int(cfg.max_epochs), planned_epochs)
                if (planned_epochs is not None and int(cfg.total_steps or 0) > 0)
                else int(cfg.max_epochs)
            )
        else:
            self._epochs_total = planned_epochs

        opt = Adam8Bit([p for p in model.parameters()], lr=cfg.lr,
                       weight_decay=cfg.weight_decay)
        sched = LRSchedule(opt, total_steps=schedule_total,
                           warmup_steps=cfg.warmup_steps,
                           schedule=cfg.scheduler)
        ckpt = CheckpointManager(
            cfg.run_name, every=cfg.checkpoint_every, keep_last=cfg.checkpoint_keep_last,
            keep_every=cfg.checkpoint_keep_every, with_optimizer=cfg.checkpoint_optimizer,
            log=self.log, lang=lang, base_model_source=lm.source,
            delta_dtype=cfg.checkpoint_delta_dtype, tokenizer=lm.tokenizer,
        )

        start_step = 0
        resumed_from_checkpoint = False
        if cfg.resume:
            # v7: when a best-val checkpoint exists, resume from it rather
            # than from a later (possibly regressed) regular checkpoint.
            # ``load_best`` returns None on failure and 0 for a real step-0
            # baseline, so check explicitly instead of using truthiness.
            loaded_best = None
            # export_best also means "the best weights are the canonical run
            # result"; resuming from a later regressed step would contradict
            # the exported model.  v8 only did this when the guard flag was
            # set, which made resume/export disagree.
            if (cfg.export_best
                    or cfg.guard_mode in ("stop", "restart")
                    or cfg.stop_on_val_regression) and ckpt.has_best:
                loaded_best = ckpt.load_best(
                    model, opt if cfg.checkpoint_optimizer else None)
            if loaded_best is not None:
                start_step = int(loaded_best)
                resumed_from_checkpoint = True
                self._load_history()
            else:
                start_step = ckpt.load(
                    model, opt if cfg.checkpoint_optimizer else None) or 0
                if start_step:
                    resumed_from_checkpoint = True
                    self._load_history()
            if resumed_from_checkpoint:
                if start_step:
                    self.log(t("resume_from", lang, step=start_step))
                    sched = LRSchedule(opt, total_steps=schedule_total,
                                       warmup_steps=0, schedule=cfg.scheduler)
                    for s in range(start_step):
                        sched.set(s)
                # A best-checkpoint resume intentionally discards the regressed
                # part of the old curve; keep the history aligned with the
                # restored weights so step numbers and the results chart agree.
                # A step-0 best checkpoint is a real result too: clear the old
                # curve completely and let the baseline evaluation below
                # create the single step-0 record again.
                self.history = self.history[:start_step]
                self.val_history = [
                    h for h in self.val_history
                    if int(h.get("step") or 0) <= start_step
                ]
                if start_step == 0:
                    self.val_history = []

        # v8: the best-val checkpoint is the export source.  The baseline
        # evaluation is step 0 and is always eligible as the best.
        # Keep the user's original LR for the whole retry sequence; each
        # retry is computed from this value, not from the previous retry.
        self._initial_lr = float(cfg.lr)
        self._sched_offset = 0
        self._best_val = None
        self._best_step = 0
        self._bad_evals = 0
        self._restarts: list[dict] = []
        # Keep only milestones that belong to the restored part of the curve.
        # A resume from the best step intentionally discards later history;
        # showing later regressions/restarts would contradict the new chart.
        if resumed_from_checkpoint and start_step > 0:
            self._events = [
                e for e in self._prior_events
                if int(e.get("step") or 0) <= start_step
            ]
            self._restarts = [
                r for r in self._prior_restarts
                if int(r.get("from_step") or 0) <= start_step
            ]
        self._val_baseline = None
        self._probe_ids = None
        self._probe_mask = None
        probe_n = min(max(0, int(cfg.train_probe_tokens or 0)),
                      max(0, train_ids[0].numel() - 1)) if train_ids else 0
        if probe_n >= 16:
            self._probe_ids = train_ids[0][:probe_n]
            self._probe_mask = train_mask[0][:probe_n].float()

        def _probe_ce() -> float | None:
            if self._probe_ids is None or self._probe_ids.numel() < 2:
                return None
            n = int(self._probe_ids.numel())
            value = stepper.evaluate(
                self._probe_ids, self._probe_mask,
                window=n, stride=n, show_bar=False)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            # A probe window with no loss tokens yields NaN.  Treat it as a
            # missing measurement so summary/history remain valid JSON.
            return value if math.isfinite(value) else None

        ce_before = None
        if start_step == 0 and val_ids.numel() > 1:
            self.log(t("val_start", lang, tokens=val_ids.numel()))
            ce_before = stepper.evaluate(
                val_ids, val_mask.float(), progress=self.log,
                **self._eval_kwargs(val_ids.numel()))
            self._val_baseline = ce_before
            self._best_val = ce_before
            self._best_step = 0
            probe = _probe_ce()
            self.val_history.append({
                "step": 0, "val_ce": ce_before,
                "probe_ce": probe,
                "best_ce": self._best_val, "best_step": 0,
            })
            self.log(t("val_baseline", lang, ce=ce_before))
            self._event("baseline", step=0, val_ce=ce_before,
                        probe_ce=probe, lr=float(cfg.lr))
            # The baseline is a legitimate "best" model when training never
            # improves.  Pin a full checkpoint so export_best can always
            # return the lowest-CE weights instead of the final regressed ones.
            try:
                ckpt.save_best(
                    0, model, ce_before,
                    metrics={"loss": None, "ema": None, "val_ce": ce_before,
                             "probe_ce": probe, "baseline": True},
                )
                self.log(t("best_checkpoint_saved", lang,
                           path=ckpt.best_path(), ce=ce_before))
            except Exception as e:
                self.log(t("best_checkpoint_failed", lang,
                           err=repr(e)[:200]))
        elif self.val_history:
            # Resume: restore the guard's notion of the best baseline.
            self._val_baseline = self.val_history[0].get("val_ce")
            best = min(
                (h for h in self.val_history if h.get("val_ce") is not None),
                key=lambda h: h["val_ce"], default=None)
            if best is not None:
                self._best_val = best["val_ce"]
                self._best_step = int(best.get("step") or 0)
            probe = _probe_ce()
            if probe is not None and self.val_history:
                self.val_history[-1].setdefault("probe_ce", probe)

        # ---------------- loop
        eps = packed.stats.get("examples_per_step_setting", 0)
        steps_label = str(planned_steps) if planned_steps is not None else "∞"
        self.log(t("train_start", lang, steps=steps_label,
                   ctx=packed.stats["context_length"],
                   mode=cfg.ascot.get("mode", "auto"),
                   seqs=n_train, eps=eps))
        if one_epoch_default:
            self.log(t("one_epoch_default", lang))
        budget = shadow_budget_report(lm)
        self.log(t("cache_budget", lang, shadow=budget["shadow_human"],
                   master=budget["master_human"]))
        if cfg.max_hours > 0:
            self.log(t("time_limit_set", lang, hours=cfg.max_hours))
        if cfg.max_epochs > 0:
            self.log(t("epoch_limit_set", lang, epochs=cfg.max_epochs))

        t_train0 = time.time()
        ema = None
        step_times: list[float] = []
        stop_requested = False
        stop_reason = "completed"

        def _handle_sig(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        try:
            signal.signal(signal.SIGINT, _handle_sig)
            signal.signal(signal.SIGTERM, _handle_sig)
        except Exception:
            pass

        try:
            step = start_step
            while step < loop_limit:
                if stop_requested:
                    stop_reason = "stopped"
                    self.log("[STOP] saving checkpoint...")
                    self._event("stop", step=step, reason=stop_reason)
                    break
                # ---- time limit (requirement 6)
                if cfg.max_hours > 0 and (time.time() - t_train0) >= cfg.max_hours * 3600:
                    stop_reason = "time_limit"
                    self.log(t("time_limit_reached", lang, hours=cfg.max_hours))
                    self._event("stop", step=step, reason=stop_reason)
                    break
                # ---- epoch limit (requirement 6): 1 epoch = all train sequences
                if cfg.max_epochs > 0 and (step + 1) > cfg.max_epochs * n_train:
                    stop_reason = "epoch_limit"
                    self.log(t("epoch_limit_reached", lang, epochs=cfg.max_epochs))
                    self._event("stop", step=step, reason=stop_reason)
                    break
                if one_epoch_default and (step + 1) > n_train:
                    stop_reason = "one_epoch_default"
                    self.log(t("one_epoch_reached", lang))
                    self._event("stop", step=step, reason=stop_reason)
                    break

                idx = step % n_train
                lr = sched.set(step - self._sched_offset)
                meta = None
                if getattr(packed, "train_meta", None):
                    meta = packed.train_meta[idx] if idx < len(packed.train_meta) else None
                long_docs = (meta or {}).get("long_docs") if meta else None
                m = stepper.step(
                    train_ids[idx], train_mask[idx].float(),
                    optimizer=opt, lr=lr,
                    doc_contexts=long_docs,
                )
                ema = m.loss if ema is None else 0.9 * ema + 0.1 * m.loss
                step_times.append(m.seconds)
                epoch = step // n_train + 1
                epochs_label = (self._epochs_total
                                if self._epochs_total is not None else "?")
                rec = {
                    "step": step + 1, "loss": m.loss, "ema": ema,
                    "true": m.true_loss, "lr": lr, "sec": round(m.seconds, 3),
                    "slice": [m.slice_start, m.slice_end], "mode": m.mode,
                    "epoch": epoch, "grad_norm": round(float(m.grad_norm), 6),
                }
                if m.spike:
                    rec["spike"] = m.spike
                self.history.append(rec)
                if step > start_step and step % n_train == 0:
                    self.log(t("epoch_start", lang, epoch=epoch,
                               epochs=epochs_label, step=step + 1,
                               total=steps_label))
                if (step == start_step and planned_steps is None
                        and cfg.max_hours > 0):
                    est = max(1, int(cfg.max_hours * 3600 / max(m.seconds, 1e-6)))
                    sched.total = est
                    self.log(t("schedule_horizon", lang, steps=est,
                               hours=cfg.max_hours))
                if cfg.log_every > 0 and (step + 1) % cfg.log_every == 0:
                    avg_s = sum(step_times[-20:]) / max(1, len(step_times[-20:]))
                    elapsed = time.time() - t_train0
                    if planned_steps is not None:
                        eta = avg_s * max(0, planned_steps - step - 1)
                    elif cfg.max_hours > 0:
                        eta = max(0.0, cfg.max_hours * 3600 - elapsed)
                    else:
                        eta = 0.0
                    self.log(t(
                        "step_line", lang,
                        epoch=epoch, epochs=epochs_label,
                        step=step + 1, total=steps_label, loss=m.loss, ema=ema,
                        true=m.true_loss, lr=lr, sps=avg_s,
                        sph=3600 / max(avg_s, 1e-9),
                        elapsed=_fmt_time(elapsed), eta=_fmt_time(eta),
                    ))
                    if m.spike:
                        action = (
                            t("spike_action_scale", lang, scale=f"{m.spike_scale:.3f}")
                            if m.spike == "scale"
                            else t("spike_action_skip", lang)
                        )
                        self.log(t("grad_spike", lang, step=step + 1,
                                   action=action, avg=m.spike_avg_norm,
                                   ratio=m.spike_ratio))
                    if cfg.show_grad_norm or cfg.show_memory:
                        mem = f"{_rss_mb():.0f}MB" if cfg.show_memory else "-"
                        gn = f"{m.grad_norm:.3f}" if cfg.show_grad_norm else "-"
                        self.log(t("step_detail", lang, j0=m.slice_start, j1=m.slice_end,
                                   mode=m.mode, gamma=f"{m.gamma:.3f}", gn=gn, mem=mem))
                saved_best_this_step = False
                if (self._time_schedule and not self._sched_time_adjusted
                        and step_times
                        and step + 1 >= min(20, max(5, cfg.eval_every or 20))):
                    # v9: the first validation gives a measured sec/step value
                    # (including evaluation overhead).  Re-anchor the LR
                    # horizon to the remaining wall-clock budget so a time
                    # limited run still anneals instead of stopping at 40% LR.
                    avg_measured = sum(step_times) / max(1, len(step_times))
                    elapsed = time.time() - t_train0
                    remaining = max(0.0, float(cfg.max_hours) * 3600 - elapsed)
                    if avg_measured > 0:
                        target = step + 1 + int(remaining * 0.95 / avg_measured)
                        target = max(step + 11, target)
                        # A finite step/epoch budget still ends the loop; the
                        # LR horizon must not be stretched past that budget.
                        if planned_steps is not None:
                            target = min(target, int(planned_steps))
                        if abs(target - int(sched.total)) >= max(5, int(sched.total * 0.02)):
                            old_total = int(sched.total)
                            sched.set_total_steps(target)
                            self._schedule_total = int(target)
                            self.log(t("schedule_adjusted", lang,
                                       old=old_total, new=target,
                                       sps=avg_measured))
                            self._event("schedule", step=step + 1,
                                        old_total=old_total, new_total=target,
                                        sec_per_step=round(avg_measured, 3))
                    self._sched_time_adjusted = True
                if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
                    self.log(t("val_start", lang, tokens=val_ids.numel()))
                    ce = stepper.evaluate(
                        val_ids, val_mask.float(), progress=self.log,
                        **self._eval_kwargs(val_ids.numel()))
                    probe = _probe_ce()
                    base = (self.val_history[0].get("val_ce")
                            if self.val_history else None)
                    prev_best = self._best_val
                    # The pinned best must be the true minimum.  The noise
                    # threshold controls patience, not checkpoint selection:
                    # a 1e-6 "improvement" must not reset a 2-evaluation
                    # regression counter.
                    min_delta = max(0.0, float(cfg.val_min_delta or 0.0))
                    improved = (prev_best is None
                                or float(ce) < float(prev_best))
                    meaningful = (prev_best is None
                                  or float(ce) < float(prev_best) - min_delta)
                    if improved:
                        self._best_val = float(ce)
                        self._best_step = step + 1
                    if meaningful:
                        self._bad_evals = 0
                    elif not improved:
                        self._bad_evals += 1
                    # A tiny improvement below min_delta is neither a
                    # meaningful reset nor another worsening: keep the
                    # existing counter unchanged.

                    rec = {
                        "step": step + 1, "val_ce": ce,
                        "probe_ce": probe,
                        "best_ce": self._best_val,
                        "best_step": self._best_step,
                        "meaningful_improvement": bool(meaningful),
                        "delta_base": (None if base is None else base - ce),
                        "delta_best": (None if prev_best is None
                                       else prev_best - ce),
                    }
                    self.val_history.append(rec)
                    self.log(t("val_result", lang, before=base if base is not None else ce,
                               after=ce, delta=(ce if base is None else base - ce),
                               best=self._best_val,
                               best_step=self._best_step,
                               probe=("-" if probe is None else f"{probe:.4f}")))
                    self._event("val", step=step + 1, val_ce=ce,
                                best=self._best_val, best_step=self._best_step,
                                probe=probe, lr=float(lr),
                                meaningful_improvement=bool(meaningful))
                    if improved:
                        self.log(t("val_best", lang, ce=ce, step=step + 1))
                        self._event("best", step=step + 1, val_ce=ce,
                                    probe=probe, lr=float(lr))
                        try:
                            ckpt.save_best(
                                step + 1, model, ce,
                                optimizer=(opt if cfg.checkpoint_optimizer else None),
                                metrics={"loss": m.loss, "ema": ema,
                                         "val_ce": ce, "probe_ce": probe},
                            )
                            self.log(t("best_checkpoint_saved", lang,
                                       path=ckpt.best_path(), ce=ce))
                            saved_best_this_step = True
                        except Exception as e:
                            self.log(t("best_checkpoint_failed", lang,
                                       err=repr(e)[:200]))
                    self._save_history()
                    _mem_hygiene(self.device)

                    # ---- validation guard (requirements 4/5/9)
                    base_bad = (base is not None
                                and ce > float(base) + float(cfg.val_min_delta))
                    best_bad = (self._best_val is not None
                                and ce > self._best_val + float(cfg.val_min_delta))
                    patience = max(1, int(cfg.val_patience or 1))
                    guard_active = (cfg.guard_mode in ("stop", "restart")
                                    or bool(cfg.stop_on_val_regression))
                    restart_enabled = (bool(cfg.auto_restart)
                                       or cfg.guard_mode == "restart")
                    trigger = (guard_active
                               and self._bad_evals >= patience
                               and (base_bad or best_bad))
                    if trigger:
                        # v8: retry as many times as needed (until the LR
                        # floor).  Rewinding is also allowed to the step-0
                        # baseline, so a model that never improved can still
                        # be retried at a lower LR.
                        self._event("regression", step=step + 1, val_ce=ce,
                                    best=self._best_val, best_step=self._best_step,
                                    patience=patience, lr=float(lr),
                                    restart_enabled=bool(restart_enabled))
                        restart_limit = max(0, int(cfg.max_restarts or 0))
                        can_restart = (
                            restart_enabled
                            and self._best_step >= 0
                            and ckpt.best_path() is not None
                        )
                        if can_restart and restart_limit > 0 \
                                and len(self._restarts) >= restart_limit:
                            self.log(t("restart_limit_reached", lang,
                                       limit=restart_limit))
                            can_restart = False
                        if can_restart:
                            retry_no = len(self._restarts) + 1
                            mode = str(cfg.restart_lr_mode or "harmonic").lower()
                            new_lr = restart_lr(
                                self._initial_lr, retry_no, mode,
                                float(cfg.restart_lr_decay))
                            lr_floor = max(
                                0.0, float(cfg.restart_lr_floor or 0.0))
                            if new_lr <= 0.0 or (
                                    lr_floor > 0.0 and new_lr < lr_floor):
                                # The best model is already pinned.  Retrying
                                # below the floor cannot lower the LR further;
                                # stop instead of looping at the same LR.
                                stop_reason = "restart_lr_floor"
                                self.log(t("restart_lr_floor", lang,
                                           retry=retry_no, lr=new_lr,
                                           floor=lr_floor,
                                           best=self._best_val,
                                           best_step=self._best_step))
                                self._event("restart_floor", step=step + 1,
                                            retry=retry_no, lr=new_lr,
                                            floor=lr_floor,
                                            best=self._best_val,
                                            best_step=self._best_step)
                                self._save_history()
                                break
                            self.log(t("restarting_from_best", lang,
                                       step=self._best_step, ce=self._best_val,
                                       lr=new_lr, retry=retry_no,
                                       mode=mode))
                            ok = ckpt.load_step_inplace(model, self._best_step)
                            if ok:
                                # drop the post-best part of the curve so the
                                # results/chart show the actual retried run
                                self.history = self.history[:self._best_step]
                                self.val_history = [
                                    h for h in self.val_history
                                    if int(h.get("step") or 0) <= self._best_step
                                ]
                                # step_times drives the final "N steps" line
                                # and throughput.  Keep it aligned with history;
                                # otherwise a restart reports one phantom step.
                                step_times = step_times[:self._best_step]
                                opt.reset_state()
                                invalidate_shadows(model)
                                clear_shadows()
                                _mem_hygiene(self.device)
                                self._restarts.append({
                                    "retry": retry_no,
                                    "from_step": self._best_step,
                                    "val_ce": self._best_val,
                                    "lr": new_lr,
                                    "mode": mode,
                                    "at_wall": time.time() - t_train0,
                                })
                                self._event("restart", step=self._best_step,
                                            from_step=self._best_step,
                                            val_ce=self._best_val,
                                            retry=retry_no, lr=new_lr,
                                            mode=mode)
                                opt.lr = new_lr
                                elapsed = time.time() - t_train0
                                if cfg.max_hours > 0:
                                    avg0 = (sum(step_times[-20:]) /
                                            max(1, len(step_times[-20:])))
                                    remaining = max(1, int(
                                        (cfg.max_hours * 3600 - elapsed) /
                                        max(avg0, 1e-6)))
                                else:
                                    remaining = max(1, loop_limit - self._best_step)
                                sched = LRSchedule(
                                    opt, total_steps=remaining,
                                    warmup_steps=0, schedule=cfg.scheduler)
                                self._sched_offset = self._best_step
                                self._sched_time_adjusted = False
                                self._bad_evals = 0
                                ema = None
                                self._save_history()
                                step = self._best_step
                                continue
                            self.log(t("restart_failed", lang,
                                       step=self._best_step))
                        stop_reason = "val_regression"
                        self.log(t("val_regression_stop", lang,
                                   step=step + 1, ce=ce,
                                   best=self._best_val,
                                   best_step=self._best_step))
                        self._event("stop", step=step + 1, reason=stop_reason,
                                    val_ce=ce, best=self._best_val,
                                    best_step=self._best_step)
                        self._save_history()
                        break
                if not saved_best_this_step and ckpt.should_save(
                        step + 1, planned_steps or 0):
                    ckpt.save(step + 1, model,
                              opt if cfg.checkpoint_optimizer else None,
                              metrics={"loss": m.loss, "ema": ema})
                    self._save_history()
                    _mem_hygiene(self.device)
                if (step + 1) % 25 == 0:
                    _mem_hygiene(self.device)
                step += 1
        except Exception as e:
            emergency = ckpt.save(max(start_step, len(self.history)),
                                  model, None, metrics={"error": str(e)})
            self._save_history()
            self.log(t("train_error", lang, err=repr(e)[:300]))
            guard.stop()
            raise

        if stop_reason == "completed":
            if one_epoch_default:
                stop_reason = "one_epoch_default"
            elif int(cfg.total_steps or 0) == 0 and int(cfg.max_epochs or 0) > 0:
                stop_reason = "epoch_limit"
            elif int(cfg.total_steps or 0) > 0:
                stop_reason = (
                    "epoch_limit"
                    if (int(cfg.max_epochs or 0) > 0
                        and planned_steps is not None
                        and planned_steps >= int(cfg.max_epochs) * n_train)
                    else "step_limit"
                )

        # ---------------- final: export best-val model (v7)
        wall = time.time() - t_train0
        final_ce = None
        if val_ids.numel() > 1:
            # avoid a duplicate evaluation when the loop just evaluated
            if (self.val_history
                    and self.val_history[-1].get("step") == len(self.history)
                    and self.val_history[-1].get("val_ce") is not None):
                final_ce = self.val_history[-1]["val_ce"]
            else:
                self.log(t("val_start", lang, tokens=val_ids.numel()))
                final_ce = stepper.evaluate(
                    val_ids, val_mask.float(), progress=self.log,
                    **self._eval_kwargs(val_ids.numel()))
                self.val_history.append({
                    "step": len(self.history), "val_ce": final_ce,
                    "probe_ce": None,
                    "best_ce": self._best_val,
                    "best_step": self._best_step,
                    "final": True,
                })
                self._event("final_eval", step=len(self.history),
                            val_ce=final_ce)
                # The final evaluation can land after the last checkpoint
                # interval.  If it is the true minimum it must become the
                # pinned/exported model, not the previous "best".
                if (final_ce is not None
                        and (self._best_val is None
                             or float(final_ce) < float(self._best_val))):
                    self._best_val = float(final_ce)
                    self._best_step = len(self.history)
                    try:
                        ckpt.save_best(
                            self._best_step, model, self._best_val,
                            metrics={
                                "loss": (self.history[-1].get("loss")
                                         if self.history else None),
                                # Keep EMA metadata usable for the exported
                                # best model; final evaluation happens after
                                # the normal checkpoint path.
                                "ema": ema,
                                "val_ce": self._best_val,
                                "probe_ce": None,
                                "final": True,
                            },
                        )
                        self.log(t("best_checkpoint_saved", lang,
                                   path=ckpt.best_path(),
                                   ce=self._best_val))
                    except Exception as e:
                        self.log(t("best_checkpoint_failed", lang,
                                   err=repr(e)[:200]))

        # A normal finish does not hit the periodic checkpoint path when the
        # final step is not a new best.  Keep a final (delta) checkpoint so
        # "latest" resume/export really means the last trained weights.
        final_step = len(self.history)
        if final_step > 0 and not ckpt.has_step(final_step):
            try:
                ckpt.save(
                    final_step, model,
                    opt if cfg.checkpoint_optimizer else None,
                    metrics={
                        "loss": (self.history[-1].get("loss")
                                 if self.history else None),
                        "ema": ema,
                        "val_ce": final_ce,
                        "final": True,
                    },
                )
                self._save_history()
            except Exception as e:
                self.log(t("checkpoint_failed", lang, err=repr(e)[:200]))

        best_path = ckpt.best_path()
        use_best = bool(cfg.auto_export and cfg.export_best
                        and best_path is not None)
        export_path = None
        export_source = None
        exported_meta = {}
        if cfg.auto_export:
            self.log(t("exporting", lang))
            export_source = "best" if use_best else "current"
            from ..models.export import export_checkpoint, export_model
            # exports are additive: never overwrite an existing model folder
            name = cfg.export_name or "finetuned"
            target = Path(EXPORT_DIR) / name
            try:
                if target.exists() and any(target.iterdir()):
                    name = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
            except OSError:
                pass
            metrics = self._summary(final_ce, wall, step_times, None,
                                    stop_reason, best_path=best_path,
                                    best_exported=use_best)
            exported_step = self._best_step if use_best else len(self.history)
            exported_val = self._best_val if use_best else final_ce
            bmeta = ckpt.read_meta(self._best_step) if use_best else {}
            exported_meta = {
                "run_steps": len(self.history),
                "exported_step": int(exported_step or 0),
                "exported_val_ce": exported_val,
                "exported_loss": (bmeta.get("metrics") or {}).get("loss")
                                 if use_best else (
                                     self.history[-1].get("loss")
                                     if self.history else None),
                "exported_ema": (bmeta.get("metrics") or {}).get("ema")
                                if use_best else (
                                    self.history[-1].get("ema")
                                    if self.history else None),
                "exported_checkpoint": str(best_path) if use_best else None,
                "export_source": export_source,
            }
            metrics.update(exported_meta)
            try:
                if use_best:
                    self.log(t("exporting_best", lang,
                               step=self._best_step, ce=self._best_val))
                    export_path = export_checkpoint(
                        lm, best_path, name, metrics=metrics)
                else:
                    export_path = export_model(lm, name, metrics=metrics)
                self.log(t("export_done", lang, path=export_path))
                self._event("export", step=len(self.history),
                            path=str(export_path), best=bool(use_best))
                try:
                    from ..core.usage import touch_model
                    touch_model(export_path, kind="trained", mode="train",
                                run_name=cfg.run_name,
                                base_model=lm.source)
                except Exception:
                    pass
            except Exception as e:
                # An export failure must not throw away a completed training
                # run: the history, log, summary and best/latest checkpoints
                # are still valuable, and the UI can show the exact error.
                export_error = repr(e)[:400]
                use_best = False
                exported_meta.update({
                    "exported_step": None,
                    "exported_val_ce": None,
                    "exported_loss": None,
                    "exported_ema": None,
                    "exported_checkpoint": None,
                    "export_source": "failed",
                    "export_error": export_error,
                })
                self.log(t("export_failed", lang, err=export_error))
                self._event("export", step=len(self.history),
                            path=None, best=False,
                            error=export_error)

        self._save_history()
        if cfg.save_plots:
            try:
                from ..train.plots import generate_training_plots
                self._event("plots", step=len(self.history), status="start")
                summary_for_plots = self._summary(
                    final_ce, wall, step_times, export_path, stop_reason,
                    best_path=best_path, best_exported=use_best)
                plots = generate_training_plots(
                    self.run_dir, summary=summary_for_plots)
                self._plots = plots
                self._event("plots", step=len(self.history), status="done",
                            files=len(plots))
            except Exception as e:
                self._plots = {}
                self.log(t("plots_failed", lang, err=repr(e)[:200]))
        summary = self._summary(final_ce, wall, step_times, export_path,
                                stop_reason, best_path=best_path,
                                best_exported=use_best)
        # Persist which weights were actually exported.  The UI must not infer
        # that the final step was exported when export_best chose an earlier
        # checkpoint.
        summary.update(exported_meta)
        if getattr(self, "_plots", None):
            summary["plots"] = self._plots
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json_dump(summary, f, indent=2)
        avg = (sum(step_times) / len(step_times)) if step_times else 0.0
        self.log(t("train_done", lang, steps=len(self.history), wall=int(wall),
                   sph=3600 / avg if avg > 0 else 0.0))
        guard.stop()

        if cfg.relaunch_ui:
            self._relaunch_ui()
        return summary

    # ------------------------------------------------------------------
    def _prepare_data(self, tokenizer):
        cfg = self.cfg
        cache_name = cfg.dataset_cache_name()
        from ..core.paths import DATASET_CACHE_DIR
        cache_path = Path(DATASET_CACHE_DIR) / f"{cache_name}.pt"
        want_fp = cfg.dataset_fingerprint()
        explicit = bool(cfg.dataset_cache)

        def log_progress(msg, **kw):
            self.log("  " + str(msg))

        if explicit:
            if not cache_path.exists():
                from ..core.config import ConfigError
                raise ConfigError(
                    "選択した構築済みデータセットが見つかりません / "
                    f"selected dataset cache not found: {cache_path}"
                )
            from ..data.builder import PackedDataset
            try:
                self.log(t("building_dataset", lang=self.lang) +
                         f" [selected cache: {cache_name}]")
                packed = PackedDataset.load(str(cache_path),
                                            progress=log_progress)
                have_fp = packed.stats.get("fingerprint")
                if have_fp and have_fp != want_fp:
                    self.log(t("cache_selected_diff", lang=self.lang,
                               cache=cache_name))
                else:
                    self.log(t("cache_reused", lang=self.lang,
                               seqs=len(packed.train_ids),
                               tokens=packed.stats.get("tokens_total", 0)))
                self._selected_cache = cache_name
                return packed
            except Exception as e:
                if isinstance(e, ConfigError):
                    raise
                from ..core.config import ConfigError
                raise ConfigError(
                    "選択したデータセットを読み込めません / "
                    f"could not load selected dataset cache {cache_name}: "
                    f"{repr(e)[:200]}"
                )

        if cache_path.exists():
            from ..data.builder import PackedDataset
            try:
                self.log(t("building_dataset", lang=self.lang) +
                         f" [cache: {cache_name}]")
                packed = PackedDataset.load(str(cache_path),
                                            progress=log_progress)
                have_fp = packed.stats.get("fingerprint")
                # the fingerprint covers sources and every packing setting;
                # a cache from another dataset/setting must never be reused
                stale = (have_fp != want_fp) if have_fp else False
                if not stale:
                    self.log(t("cache_reused", lang=self.lang,
                               seqs=len(packed.train_ids),
                               tokens=packed.stats.get("tokens_total", 0)))
                    self._selected_cache = cache_name
                    return packed
                self.log(t("cache_stale", lang=self.lang))
            except Exception as e:
                self.log(t("cache_incompatible", lang=self.lang,
                           err=repr(e)[:200]))
        self.log(t("building_dataset", lang=self.lang))

        packed = build_dataset(
            tokenizer, cfg.spec_objects(), cfg.build_config(),
            cache_name=cache_name, progress=log_progress,
        )
        s = packed.stats
        self.log(t("dataset_built", lang=self.lang,
                   seqs=s["sequences"], tokens=s["tokens_total"],
                   ratio=s["loss_token_ratio"],
                   eps=s.get("examples_per_step_setting", 0)))
        if s.get("dataset_order") == "front":
            fm = s.get("front_mix") or {}
            self.log(t("data_order_front", lang=self.lang,
                       main=fm.get("main_source", "-"),
                       window=fm.get("window", "-"),
                       mixed=fm.get("mixed_total", 0)))
        if s.get("long_doc_sequences"):
            self.log(t("long_doc_built", lang=self.lang,
                       n=s.get("long_doc_sequences")))
        self._selected_cache = cache_name
        _mem_hygiene(self.device)
        return packed

    # ------------------------------------------------------------------
    def _validation_view(self, packed):
        """Return a validation stream that samples across held-out documents.

        v8 evaluated only the first 4,096 tokens of a 96,648-token validation
        stream.  With 32k-token packed sequences that is usually one document,
        so a single unusually easy/hard document could make the guard fire (or
        hide a regression).  New caches record per-sequence lengths; build a
        balanced set of fixed windows from the beginning/middle/end of several
        validation sequences.  Legacy v8 caches fall back to the old prefix.
        """
        val_ids, val_mask = packed.val_ids, packed.val_mask
        if val_ids.numel() <= 1:
            val_ids = packed.train_ids[-1]
            val_mask = packed.train_mask[-1]
            return val_ids, val_mask
        cap = int(self.cfg.val_tokens or 0)
        if cap <= 0 or val_ids.numel() <= cap:
            return val_ids, val_mask
        lengths = (packed.stats or {}).get("val_seq_lengths") or []
        lengths = [max(0, int(x)) for x in lengths]
        if not lengths or sum(lengths) <= 0:
            return val_ids[:cap], val_mask[:cap]
        window = max(2, min(2048, cap))
        slots = max(1, cap // window)
        # Round-robin across validation sequences, so a run with many short
        # held-out documents does not evaluate only the first one.
        chunks_ids, chunks_mask = [], []
        offsets = []
        pos = 0
        for n in lengths:
            offsets.append(pos)
            pos += n
        used = 0
        per_seq = max(1, slots // max(1, len(lengths)))
        order = list(range(len(lengths)))
        # If there are more sequences than slots, spread the slots evenly.
        if slots < len(lengths):
            order = [round(i * (len(lengths) - 1) / max(1, slots - 1))
                     for i in range(slots)] if slots > 1 else [0]
        for idx in order:
            n = lengths[idx]
            if n <= 1:
                continue
            take = min(window, n)
            count = per_seq if slots >= len(lengths) else 1
            for k in range(count):
                if used >= slots:
                    break
                # start/middle/end anchors for this sequence
                if count == 1:
                    start = max(0, (n - take) // 2)
                else:
                    start = int((n - take) * k / max(1, count - 1))
                start = max(0, min(start, n - take))
                a = offsets[idx] + start
                chunks_ids.append(val_ids[a:a + take])
                chunks_mask.append(val_mask[a:a + take])
                used += 1
            if used >= slots:
                break
        if not chunks_ids:
            return val_ids[:cap], val_mask[:cap]
        out_ids = torch.cat(chunks_ids)[:cap]
        out_mask = torch.cat(chunks_mask)[:cap]
        return out_ids, out_mask

    def _eval_kwargs(self, n):
        # v7: always evaluate fixed 2048-token windows.  The old v6 path
        # evaluated the whole capped prefix in one forward; 8192 tokens already
        # needed ~9 GB for the attention mask/allocation and 96k OOMs outright
        # on a 16 GB machine.  Fixed windows are also a less noisy estimate
        # than a single 2048-token prefix.
        w = max(2, min(2048, int(n)))
        return {"window": w, "stride": w}

    def _relaunch_ui(self):
        self.log(t("relaunch_ui", self.lang))
        try:
            import subprocess
            import sys
            # Use the run's own config, not the mutable UI state.  v8 launched
            # the results UI with state/state.json, which could describe a
            # completely different dataset and resume policy.
            run_config = self.run_dir / "config.json"
            config_path = run_config if run_config.exists() else (STATE_DIR / "state.json")
            subprocess.Popen(
                [sys.executable, "-m", "cpu_finetuning", "serve",
                 "--config", str(config_path),
                 "--lang", self.lang, "--results", self.cfg.run_name],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except Exception as e:
            self.log(f"[UI relaunch failed: {e}]")

    def _summary(self, final_ce, wall, step_times, export_path=None,
                 stop_reason="completed", best_path=None,
                 best_exported=None):
        avg = (sum(step_times) / len(step_times)) if step_times else 0.0
        steady_pool = (step_times[1:] if len(step_times) > 1 else step_times)[-20:]
        steady = (sum(steady_pool) / len(steady_pool)) if steady_pool else 0.0
        # first/last windows must not overlap on short runs (e.g. the 10-step
        # verification): k = floor(n/3), capped at 5.
        k = max(1, min(len(self.history) // 3, 5)) if self.history else 0
        first = self.history[:k] or [{"loss": 0}]
        last = self.history[-k:] or [{"loss": 0}]
        ctx = self.cfg.data.get("context_length", 32768)
        info = getattr(self, "compute_info", {})
        dstats = getattr(self, "data_stats", {})
        vce = [h["val_ce"] for h in self.val_history
               if h.get("val_ce") is not None]
        probes = [h["probe_ce"] for h in self.val_history
                  if h.get("probe_ce") is not None]
        train_seqs = int(dstats.get("train_sequences") or 0) or 1
        best_val_step = (self._best_step
                         if getattr(self, "_best_step", None) is not None
                         else None)
        best_val = (self._best_val if getattr(self, "_best_val", None) is not None
                    else (min(vce) if vce else None))
        if best_val_step is None and vce:
            best_val_step = min(
                self.val_history,
                key=lambda h: (h["val_ce"]
                               if h.get("val_ce") is not None
                               else float("inf")),
            ).get("step")
        last_val = vce[-1] if vce else None
        grad_values = [float(h.get("grad_norm") or 0.0) for h in self.history
                       if h.get("grad_norm") is not None]
        spike_count = sum(1 for h in self.history if h.get("spike"))
        return {
            "run_name": self.cfg.run_name,
            "model": self.cfg.model_source,
            "task": self.cfg.task,
            "steps": len(self.history),
            "run_steps": len(self.history),
            "epochs": round(len(self.history) / train_seqs, 2),
            "planned_steps": getattr(self, "_planned_steps", None),
            "schedule_total": getattr(self, "_schedule_total", None),
            "epochs_total": getattr(self, "_epochs_total", None),
            "stop_reason": stop_reason,
            "context_length": ctx,
            "examples_per_step": self.cfg.data.get("examples_per_step", 0),
            "device": getattr(self, "device", "cpu"),
            "device_label": (getattr(self, "device_choice", None).label
                             if getattr(self, "device_choice", None) else ""),
            "dataset_cache": getattr(self, "_selected_cache", "") or self.cfg.dataset_cache,
            "sec_per_step": round(avg, 3),
            "sec_per_step_steady": round(steady, 3),
            "steps_per_hour": round(3600 / avg, 1) if avg > 0 else 0.0,
            "steps_per_hour_steady": (
                round(3600 / steady, 1) if steady > 0 else 0.0),
            "tokens_per_second": (
                round(ctx / steady, 1) if steady > 0 else 0.0),
            "compute_dtype": info.get("compute_dtype"),
            "dtype_reason": info.get("dtype_reason"),
            "int8_shadow": bool((info.get("int8_backend") or {}).get("enabled")),
            "int8_reason": (info.get("int8_backend") or {}).get("reason"),
            "loss_first": round(sum(h["loss"] for h in first) / len(first), 4),
            "loss_last": round(sum(h["loss"] for h in last) / len(last), 4),
            "loss_window": k,
            "loss_min": round(min(h["loss"] for h in self.history), 4)
                        if self.history else None,
            "loss_ema_first": self.history[0].get("ema") if self.history else None,
            "loss_ema_last": self.history[-1].get("ema") if self.history else None,
            "grad_norm_first": (grad_values[0] if grad_values else None),
            "grad_norm_last": (grad_values[-1] if grad_values else None),
            "grad_norm_mean": (round(sum(grad_values) / len(grad_values), 6)
                               if grad_values else None),
            "grad_norm_max": max(grad_values) if grad_values else None,
            "grad_spikes": spike_count,
            "val_ce_first": vce[0] if vce else None,
            "val_ce_best": best_val,
            "val_ce_best_step": best_val_step,
            "val_ce_last": last_val,
            "val_ce_final": final_ce,
            "val_delta_base": ((vce[0] - last_val)
                               if len(vce) >= 2 and last_val is not None
                               and vce[0] is not None else None),
            "guard_mode": str(self.cfg.guard_mode),
            "stop_on_val_regression": bool(self.cfg.stop_on_val_regression),
            "val_patience": int(self.cfg.val_patience),
            "val_min_delta": float(self.cfg.val_min_delta),
            "train_probe_first": probes[0] if probes else None,
            "train_probe_last": probes[-1] if probes else None,
            "best_checkpoint": str(best_path) if best_path else None,
            "best_model_exported": (
                bool(best_exported) if best_exported is not None
                else bool(best_path)),
            "restarts": getattr(self, "_restarts", []),
            "restart_count": len(getattr(self, "_restarts", [])),
            "auto_restart": bool(self.cfg.auto_restart),
            "max_restarts": int(self.cfg.max_restarts or 0),
            "restart_lr_mode": str(self.cfg.restart_lr_mode or "harmonic"),
            "restart_lr_decay": float(self.cfg.restart_lr_decay),
            "restart_lr_floor": float(self.cfg.restart_lr_floor),
            "events": list(getattr(self, "_events", [])),
            "val_tokens": int(validate_val_tokens(self.cfg.val_tokens)),
            "val_window": min(2048, int(self.cfg.val_tokens or 2048)),
            "dataset": {k2: dstats.get(k2) for k2 in (
                "sequences", "train_sequences", "val_sequences", "tokens_total",
                "loss_token_ratio", "fingerprint", "build_seconds",
                "dataset_order", "long_doc_sequences", "front_mix",
                "source_legend", "example_order", "sequence_sources",
                "sequence_order", "first_examples")},
            "wall_seconds": round(wall, 1),
            "export_path": str(export_path) if export_path else None,
            "history_file": str(self.run_dir / "history.json"),
            "log_file": str(self.log_path),
        }


def restart_lr(initial_lr: float, retry_number: int,
               mode: str = "harmonic", decay: float = 0.5) -> float:
    """Learning rate for retry number ``retry_number`` (1-based).

    ``harmonic`` follows the requested 1/n sequence: base/2, base/3, base/4...
    ``exponential`` keeps the old configurable factor semantics:
    base*decay, base*decay**2, ...  The original base LR is used every time,
    so repeated restarts do not accidentally compound the reduction.
    """
    base = float(initial_lr)
    n = max(1, int(retry_number))
    if str(mode or "").lower() in ("exponential", "factor", "decay"):
        # A factor >= 1 would never reach the floor.  Clamp it to a value
        # that strictly decreases the LR.
        factor = min(0.999, max(0.01, float(decay)))
        return base * (factor ** n)
    return base / float(n + 1)


def validate_val_tokens(v) -> int:
    try:
        return max(0, int(v))
    except Exception:
        return 0


