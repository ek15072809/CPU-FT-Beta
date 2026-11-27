"""ASCOT step logic: Anchor-State COmpressed Training.

One step consumes a full packed sequence (``context_length`` tokens, e.g.
32768) and produces an update whose expected value equals the update of a
dense fine-tuning step on the same sequence, at a small fraction of the
compute.  The step has three parts:

1. **Select**  - cheap importance scores over all blocks of the sequence;
   sample ``num_spans`` anchor spans with probabilities ``pi`` and reweight
   their losses by ``1/(R*pi)`` (Horvitz-Thompson => unbiased).
2. **Transplant** - one frozen (int8 shadow) forward over
   ``[sink | prefix | span]`` rows produces recurrent/attention states and
   reference hidden states; states are snapshotted at the span boundary and
   *transplanted* into the gradient pass.
3. **Update**  - a contiguous *slice* of ``slice_size`` decoder layers (sampled
   by layer-roulette) plus the LM head run with autograd from the transplanted
   state; the loss is a (calibrated) next-token CE on the span positions,
   reweighted by span and slice probabilities.  Optionally, when the slice
   sits in the upper half of the stack, the layers above the trainable slice
   are run with data-gradients only ("global" mode) so the objective is the
   exact full CE.

The per-step compute budget is roughly
``num_spans*(sink+prefix+span)`` frozen token-forwards plus
``span_tokens*(slice_size/L)*(fwd+bwd)`` trainable token-equivalents, which is
how a 32k-token step fits into the "40 tokens/s of compute" envelope.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x

from .selector import SpanSelector
from .spike_guard import GradSpikeGuard
from ..models.arch import ArchAdapter
from ..models.qlayer import invalidate_shadows


@dataclass
class AscotConfig:
    # span / context geometry
    block_size: int = 512
    span_length: int = 128
    num_spans: int = 4
    sink_len: int = 16
    prefix_len: int = 64
    uniform_floor: float = 0.25
    mask_weight: float = 1.0
    novelty_weight: float = 1.0
    # slice (layer roulette)
    slice_size: int = 3
    # v7: "balanced" walks a shuffled permutation of every possible slice
    # start.  The old adaptive sampler followed gradient mass and gave the
    # first slice starts 2-3x more updates than the last ones, which left the
    # upper layers (and therefore the input distribution seen by the tied
    # LM head) permanently behind the lower layers.
    slice_sampling: str = "balanced"   # balanced | uniform | top | adaptive
    slice_reweight: bool = False       # 1/p rescale (SGD theory; off for Adam)
    mode: str = "global"               # global | local | auto
    # v7: long-document rows contain [sink | document plan | local prefix].
    # Keeping a local prefix is essential for next-token prediction; v5/v6
    # replaced it with the plan, making every late span unnaturally hard.
    long_plan_tokens: int = 64
    # objective
    head_positions: int = 64           # CE positions per span row (0 = all)
    calib_ema: float = 0.95
    # The tied LM head gets an exact sparse-row Adam update.  Its gradient is
    # normalized per element, so a handful of occurrences move a row by ~lr
    # regardless of gradient magnitude; a milder head LR keeps the output
    # distribution (and the tied input embeddings) stable.
    head_lr_scale: float = 0.10
    # frozen-reference freshness: 1 = refresh int8 shadows of updated layers
    # every step (exact); N>1 lets the frozen pass use an N-step-old snapshot
    # of updated layers, amortizing requantization (asynchronous-SGD style).
    shadow_refresh_every: int = 1
    # gradient spike handling: off | skip | scale (mutually exclusive modes)
    grad_spike_mode: str = "off"
    grad_spike_threshold_pct: float = 100.0
    grad_spike_window: int = 20
    # optimization defaults (overridden by trainer config)
    max_grad_norm: float = 1.0
    seed: int = 0


@dataclass
class StepMetrics:
    loss: float = 0.0
    true_loss: float = 0.0             # frozen-pass CE on span tokens
    lr: float = 0.0
    grad_norm: float = 0.0
    slice_start: int = 0
    slice_end: int = 0
    mode: str = "local"
    seconds: float = 0.0
    span_starts: list = field(default_factory=list)
    gamma: float = 1.0
    spike: str = ""                     # "" | "skip" | "scale" when triggered
    spike_avg_norm: float = 0.0
    spike_scale: float = 1.0
    spike_ratio: float = 1.0


class AscotStepper:
    """Executes ASCOT steps on one model/adapter pair."""

    def __init__(
        self,
        model,
        adapter: ArchAdapter,
        cfg: AscotConfig | None = None,
    ):
        self.model = model
        self.adapter = adapter
        self.cfg = cfg or AscotConfig()
        self.selector = SpanSelector(
            block_size=self.cfg.block_size,
            span_length=self.cfg.span_length,
            num_spans=self.cfg.num_spans,
            uniform_floor=self.cfg.uniform_floor,
            mask_weight=self.cfg.mask_weight,
            novelty_weight=self.cfg.novelty_weight,
            seed=self.cfg.seed,
        )
        self.n_layers = adapter.num_layers
        # calibration (diagonal, per-dimension) per slice-exit layer
        self.gamma: dict[int, torch.Tensor] = {}
        # adaptive layer sampling distribution over slice starts.  A mild
        # top bias makes the first steps update layers whose hidden states are
        # already close to the output (fast, visible loss movement); after
        # that the EMA follows the observed gradient mass.
        idx = torch.arange(self.n_layers, dtype=torch.float32)
        self.layer_scores = 1.0 + idx / max(1, self.n_layers - 1)
        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        # balanced sampling state: a fresh permutation of every valid start
        self._slice_queue: list[int] = []
        self.slice_counts: dict[str, int] = {}
        self.spike_guard = GradSpikeGuard(
            mode=self.cfg.grad_spike_mode,
            threshold_pct=self.cfg.grad_spike_threshold_pct,
            window=self.cfg.grad_spike_window,
        )
        self._last_extra: dict | None = None
        model.eval()

    # ------------------------------------------------------------------
    def _slice_start_probs(self) -> torch.Tensor:
        K = self.cfg.slice_size
        n_starts = max(1, self.n_layers - K + 1)
        s = self.cfg.slice_sampling
        if s == "uniform":
            p = torch.ones(n_starts)
        elif s == "top":
            idx = torch.arange(n_starts, dtype=torch.float32)
            p = 1.0 + idx * 2.0
        else:  # adaptive: follow accumulated gradient-mass EMA of layers
            sc = self.layer_scores[:n_starts].clone()
            p = sc.clamp_min(1e-3)
        return p / p.sum()

    @torch.no_grad()
    def _sample_slice(self) -> tuple[int, int]:
        K = min(self.cfg.slice_size, self.n_layers)
        n_starts = max(1, self.n_layers - K + 1)
        if self.cfg.slice_sampling == "balanced":
            if not self._slice_queue:
                self._slice_queue = torch.randperm(
                    n_starts, generator=self.gen
                ).tolist()
            j0 = int(self._slice_queue.pop())
        else:
            probs = self._slice_start_probs()
            j0 = int(torch.multinomial(probs, 1, generator=self.gen).item())
        j0 = min(j0, self.n_layers - K)
        key = f"{j0}:{j0 + K}"
        self.slice_counts[key] = self.slice_counts.get(key, 0) + 1
        return j0, j0 + K

    # ------------------------------------------------------------------
    def _build_rows(self, ids: torch.Tensor, plan, doc_contexts=None):
        """Build uniform pre-roll rows and contiguous-position spans.

        Every pre-roll row has exactly ``sink_len + plan_len + prefix_len``
        tokens so the rows batch cleanly, and its ``position_ids`` are the
        contiguous range ``0..maxlen-1``.  Absolute document positions are
        deliberately NOT used here: transformers 5.x LFM3 interprets any
        position restart as a packed document and then refuses cache
        continuation (``A cache filled by a packed-document forward cannot be
        continued``).  Relative positions *inside* the span
        (``maxlen..maxlen+L-1``) stay exact, which is what the transplanted
        conv/attention state needs.

        v7: a long-document row is ``[sink | document plan | local prefix]``.
        v5/v6 used ``[sink | plan]`` and therefore removed the immediate
        local tokens from late-span prediction.  That made every distant
        span artificially hard (its first token had no local continuation
        context), which is why the training loss stalled around 1.5-1.8 while
        validation CE was much lower.  Keeping both the plan and the local
        prefix lets the model learn distant-reference consistency *and*
        ordinary next-token prediction with the same row budget.
        """
        cfg = self.cfg
        n = ids.numel()
        L = min(cfg.span_length, max(1, n - 1))
        S0 = min(cfg.sink_len, n)
        P = max(0, int(cfg.prefix_len))
        docs = doc_contexts or []
        # Only long-document samples need the plan slot.  Giving every normal
        # sequence 64 zero-token slots would inject fake context into ordinary
        # next-token prediction and inflate the frozen pass for no benefit.
        PLAN = max(0, int(cfg.long_plan_tokens or 0)) if docs else 0
        maxlen = S0 + PLAN + P
        zero = ids[:0]

        def context_for(start: int):
            for d in docs:
                try:
                    if int(d.get("head_end", 0)) <= start < int(d.get("end", 0)):
                        return d
                except Exception:
                    continue
            return None

        def pad_left(x: torch.Tensor, length: int) -> torch.Tensor:
            if x.numel() >= length:
                return x[-length:] if length else x[:0]
            return torch.cat([
                torch.zeros(length - x.numel(), dtype=ids.dtype,
                            device=ids.device), x,
            ])

        rows = []
        for start in plan.starts.tolist():
            start = min(int(start), max(0, n - L - 1))
            sd = context_for(start)
            if sd is not None:
                base = max(0, int(sd.get("start", 0)))
                head_end = max(0, int(sd.get("head_end", 0)))
                # Sink: the beginning of the document (document identity).
                sink = ids[base:min(base + S0, head_end)]
                if sink.numel() < S0:
                    sink = torch.cat([
                        sink, torch.zeros(S0 - sink.numel(), dtype=ids.dtype,
                                          device=ids.device),
                    ])
                # Plan: the loss-masked document summary / outline / key facts.
                plan_tokens = (
                    ids[base:min(base + PLAN, head_end)] if PLAN else zero
                )
                if plan_tokens.numel() < PLAN:
                    plan_tokens = torch.cat([
                        plan_tokens,
                        torch.zeros(PLAN - plan_tokens.numel(), dtype=ids.dtype,
                                    device=ids.device),
                    ])
                # Local prefix: contiguous tokens immediately before the span.
                # This is the context a normal next-token objective needs.
                pre = pad_left(ids[max(0, start - P):start], P)
            else:
                sink = ids[:S0]
                plan_tokens = (
                    torch.zeros(PLAN, dtype=ids.dtype, device=ids.device)
                    if PLAN else zero
                )
                pre = pad_left(ids[max(0, start - P):start], P)
            rows.append(torch.cat([sink, plan_tokens, pre]))
        rows_pre = torch.stack(rows)
        pos_pre = torch.arange(maxlen, dtype=torch.long,
                               device=ids.device).unsqueeze(0).expand(
            rows_pre.shape[0], -1)
        starts = [min(int(s), max(0, n - L - 1)) for s in plan.starts.tolist()]
        span_rows = torch.stack([ids[s:s + L] for s in starts])
        span_pos = torch.arange(maxlen, maxlen + L, dtype=torch.long,
                                device=ids.device).unsqueeze(
            0).expand(span_rows.shape[0], -1)
        return rows_pre, pos_pre, span_rows, span_pos

    # ------------------------------------------------------------------
    def step(
        self,
        ids: torch.Tensor,          # [N] packed sequence
        loss_mask: torch.Tensor,    # [N] 1 where the token trains
        optimizer=None,
        lr: float | None = None,
        train: bool = True,
        doc_contexts: list | None = None,
    ) -> StepMetrics:
        cfg = self.cfg
        t_start = time.perf_counter()
        metrics = StepMetrics()
        adapter = self.adapter
        device = next(self.model.parameters()).device
        ids = ids.to(device)
        loss_mask = loss_mask.to(device)
        N = ids.numel()

        # ---- 1. score and sample spans
        with torch.no_grad():
            embeds = adapter.embed_tokens(ids.unsqueeze(0))[0].float()
            block_scores = self.selector.block_scores(embeds, loss_mask.float())
        boosts = []
        for d in (doc_contexts or []):
            try:
                hs, he = int(d.get("head_end", 0)), int(d.get("end", N))
                if he > hs:
                    # Focus on the later part, but keep probability mass on
                    # the whole tail for general long-form continuation.
                    boosts.append((hs + int(0.50 * (he - hs)), he, 3.0))
                    # Distinct extra boost for later mentions of early-plan
                    # names/classes/facts: direct distant-reference training.
                    for ref in d.get("reference_spans", []) or []:
                        rs, rend = int(ref[0]), int(ref[1])
                        rs = max(0, min(N, rs))
                        rend = max(rs + 1, min(N, rend))
                        if rend > rs:
                            boosts.append((rs, rend, 4.0))
            except Exception:
                pass
        plan = self.selector.plan(
            N, block_scores, min_prefix=cfg.prefix_len,
            boost_ranges=boosts or None,
        )
        metrics.span_starts = plan.starts.tolist()

        rows_pre, pos_pre, span_rows, span_pos = self._build_rows(
            ids, plan, doc_contexts=doc_contexts)
        R, L = span_rows.shape
        starts = [min(int(s), max(0, N - L - 1)) for s in plan.starts.tolist()]

        # loss positions are selected BEFORE the 262M-parameter head GEMM;
        # the reference pass then needs ONE head call for all rows instead of
        # one call per span/chunk (each call streams the whole head once).
        if cfg.head_positions and cfg.head_positions < L:
            keep = torch.linspace(0, L - 1, cfg.head_positions).long().unique()
        else:
            keep = None
        S = L if keep is None else int(keep.numel())
        if keep is not None:
            keep = keep.to(device)
        weights = plan.weights.float().to(device)
        shift_labels = torch.stack([ids[s + 1:s + L + 1] for s in starts])
        m_rows = torch.stack([
            loss_mask[s + 1:s + L + 1] for s in starts
        ]).float()
        lab_sel = shift_labels if keep is None else shift_labels[:, keep]
        m_sel = m_rows if keep is None else m_rows[:, keep]

        # ---- 2. frozen pass: pre-roll states + reference hidden states
        head_rows, head_vals = None, None
        with torch.no_grad():
            out_a = adapter.full_forward(
                rows_pre, use_cache=True, position_ids=pos_pre,
            )
            cache_a = out_a.past_key_values
            snap = adapter.snapshot(cache_a)
            out_b = adapter.full_forward(
                span_rows,
                past_key_values=cache_a,
                use_cache=True,
                output_hidden_states=True,
                position_ids=span_pos,
            )
            h_ref = [h for h in out_b.hidden_states]     # L+1 entries [R, L, D]
            top_hidden = out_b.last_hidden_state         # after final norm

            # reference CE on the chosen positions (true loss of the frozen
            # model), in bounded chunks: the output dominates transient RAM
            h_top = top_hidden if keep is None else top_hidden[:, keep]
            flat = h_top.reshape(-1, h_top.shape[-1])
            ref = []
            for c0 in range(0, flat.shape[0], 256):
                ref.append(adapter.head_logits(flat[c0:c0 + 256]).float())
            ref = torch.cat(ref, 0)
            ce_ref = F.cross_entropy(
                ref, lab_sel.reshape(-1), reduction="none"
            ).view(R, S)
            per_row_ref = (ce_ref * m_sel).sum(1) / m_sel.sum(1).clamp_min(1.0)
            metrics.true_loss = float((per_row_ref * weights).sum().clamp(min=0))

            # Exact CE gradient w.r.t. the *observed* LM-head rows, taken from
            # the true frozen top representation (not from the calibrated
            # slice proxy).  The head is tied to the embeddings, so this is
            # also the embedding update; vocabulary rows that never occur get
            # no update (their softmax gradient is tiny, their Adam noise is
            # not, and a random walk over 128k rows degrades the model).
            if train and adapter.head is not None:
                # Only *loss-bearing* label positions have a CE gradient on
                # the output side.  v6 collected every label (including the
                # user/context tokens with mask 0) and then handed zero rows
                # to the sparse head optimizer.  Those rows still advanced
                # their Adam moments and received weight decay, so embeddings
                # that never occurred as targets slowly drifted.  Filter
                # positions before taking the gradient.
                vocab_n = adapter.head.weight.shape[0]
                valid = m_sel.reshape(-1) > 0
                lab_flat = lab_sel.reshape(-1)
                valid &= lab_flat < vocab_n
                if bool(valid.any()):
                    lab_valid = lab_flat[valid]
                    present = torch.unique(lab_valid)
                    h_valid = h_top.detach().float().reshape(
                        -1, h_top.shape[-1])[valid]
                    ref_valid = ref.reshape(-1, ref.shape[-1])[valid]
                    lse = torch.logsumexp(ref_valid, dim=-1, keepdim=True)
                    p_sel = (
                        ref_valid.index_select(-1, present) - lse
                    ).exp()
                    onehot = torch.zeros_like(p_sel)
                    idx = torch.searchsorted(present, lab_valid)
                    onehot.scatter_(1, idx.unsqueeze(1), 1.0)
                    row_w = (
                        weights[:, None] * m_sel
                        / m_sel.sum(1, keepdim=True).clamp_min(1.0)
                    ).reshape(-1)[valid]
                    coef = (p_sel - onehot) * row_w.reshape(-1, 1)
                    head_rows = present
                    head_vals = (coef.T @ h_valid).contiguous()
                    del p_sel, onehot, coef, lse, h_valid, ref_valid
                else:
                    head_rows, head_vals = None, None
            del ref, ce_ref, flat, h_top

        # ---- 3. choose slice and mode
        j0, j1 = self._sample_slice()
        metrics.slice_start, metrics.slice_end = j0, j1
        mode = cfg.mode
        if mode == "auto":
            # The local proxy is only meaningful after its diagonal
            # calibration has converged.  Without a prior, an early local
            # slice produces a near-uniform CE (~ln V) whose gradient is
            # noise; auto therefore uses the exact global objective.  The
            # local mode remains available as an explicit opt-in.
            mode = "global"
        metrics.mode = mode

        # ---- 4. gradient pass
        cache_g = adapter.restore(snap)
        extra: dict[str, Any] = {}
        if adapter.supports_shared_kv:
            extra["shared_kv"] = adapter.snapshot_shared_kv(out_b)
            if adapter.hidden_size_per_layer_input:
                emb_span = adapter.embed_tokens(span_rows)
                extra["per_layer_inputs"] = adapter.compute_per_layer_inputs(
                    span_rows, emb_span
                )

        trainable = []
        for p in self.model.parameters():
            p.requires_grad_(False)
        head_w = adapter.head.weight if adapter.head is not None else None
        head_b = getattr(adapter.head, "bias", None)
        if head_w is None:
            raise RuntimeError("the model has no LM head; cannot compute CE")
        if train:
            for i in range(j0, j1):
                trainable += list(adapter.layers[i].parameters())
            for p in trainable:
                p.requires_grad_(True)
        # NOTE: the LM head is deliberately not enabled for autograd.  Its
        # dense gradient is a 1 GB tensor (262M params); we build the exact
        # gradient of the *observed* vocabulary rows instead and hand it to
        # Adam8Bit.step_sparse_rows.

        h_in = h_ref[j0].detach()
        h_slice = None
        if mode == "global":
            h_out = adapter.slice_forward(
                h_in, (j0, self.n_layers), span_pos, cache_g, extra)
            h_final = adapter.final_norm(h_out)
            gamma = 1.0
        else:
            h_slice = adapter.slice_forward(
                h_in, (j0, j1), span_pos, cache_g, extra)
            h_norm = adapter.final_norm(h_slice)
            # Fit the diagonal calibration BEFORE the loss: a slice exit that
            # has never been sampled would otherwise start from an
            # uncalibrated representation (near-uniform CE) and send a
            # meaningless gradient into the head/slice on its first update.
            gamma = self.gamma.get(j1)
            if train:
                with torch.no_grad():
                    h_det = h_norm.detach().float().reshape(
                        -1, h_norm.shape[-1])
                    t_det = top_hidden.detach().float().reshape_as(h_det)
                    num = (h_det * t_det).sum(0)
                    den = (h_det * h_det).sum(0) + 1e-3
                    new_g = (num / den).clamp(0.05, 20.0)
                    if gamma is None:
                        gamma = new_g
                    else:
                        gamma = (
                            cfg.calib_ema * gamma
                            + (1 - cfg.calib_ema) * new_g
                        ).clamp(0.05, 20.0)
                    self.gamma[j1] = gamma
            if gamma is None:
                gamma = torch.ones(h_norm.shape[-1], device=h_norm.device)
            h_final = h_norm * gamma.to(device=h_norm.device, dtype=h_norm.dtype)

        h_sel_g = h_final if keep is None else h_final[:, keep]
        logits = F.linear(
            h_sel_g, head_w.detach(),
            head_b.detach() if head_b is not None else None,
        )
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        ce_l = -log_probs.gather(-1, lab_sel.unsqueeze(-1)).squeeze(-1)
        per_row_l = (ce_l * m_sel).sum(1) / m_sel.sum(1).clamp_min(1.0)
        loss = (per_row_l * weights).sum()   # ~ full-sequence mean CE (unbiased)

        # NOTE: head_rows / head_vals were computed in the frozen pass from
        # the true top representation.  Here only the slice proxy is
        # backpropagated (the head weight is detached).
        metrics.gamma = float(self.gamma[j1].mean()) if j1 in self.gamma else 1.0

        metrics.loss = float(loss.detach())

        # slice reweighting: in expectation each layer receives a full update.
        # Applied only to the backward loss (the metric stays interpretable).
        loss_scaled = loss
        if cfg.slice_reweight:
            n_starts = max(1, self.n_layers - cfg.slice_size + 1)
            # Balanced sampling is uniform over the queue even though the
            # adaptive probabilities below may be non-uniform.  Using the
            # adaptive p here would apply the wrong importance weight.
            if cfg.slice_sampling == "balanced":
                p = 1.0 / n_starts
            else:
                p = self._slice_start_probs()[min(j0, n_starts - 1)].item()
            loss_scaled = loss * (1.0 / max(p, 1e-9))

        if train:
            loss_scaled.backward()

            # Per-layer grad norms must be read BEFORE zero_grad() (the old
            # code read them after the update, so adaptive sampling was
            # silently frozen at uniform).
            layer_gn = [0.0] * self.n_layers
            with torch.no_grad():
                for i in range(j0, j1):
                    s = 0.0
                    for p in adapter.layers[i].parameters():
                        if p.grad is not None:
                            s += float(p.grad.detach().float().pow(2).sum())
                    layer_gn[i] = math.sqrt(s) / max(1, j1 - j0)
                dense_sq = 0.0
                for p in trainable:
                    if p.grad is not None:
                        dense_sq += float(p.grad.detach().float().pow(2).sum())
                if head_vals is not None:
                    dense_sq += float(head_vals.pow(2).sum())
                gn = math.sqrt(dense_sq)
            metrics.grad_norm = gn

            # --- gradient spike guard (skip / scale, mutually exclusive) ---
            decision = self.spike_guard.check(gn)
            if decision.is_spike:
                metrics.spike = decision.action
                metrics.spike_avg_norm = decision.avg_norm
                metrics.spike_scale = decision.scale
                metrics.spike_ratio = decision.ratio
                if decision.action == "scale":
                    for p in trainable:
                        if p.grad is not None:
                            p.grad.mul_(decision.scale)
                    if head_vals is not None:
                        head_vals.mul_(decision.scale)
                    gn = gn * decision.scale
                    metrics.grad_norm = gn
                elif decision.action == "skip":
                    optimizer.zero_grad() if optimizer is not None else None
                    for p in trainable:
                        if p.grad is not None:
                            p.grad = None
            if optimizer is not None and not (decision.is_spike and decision.action == "skip"):
                if cfg.max_grad_norm > 0 and gn > cfg.max_grad_norm:
                    scale = cfg.max_grad_norm / (gn + 1e-6)
                    for p in trainable:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                    if head_vals is not None:
                        head_vals.mul_(scale)
                if lr is not None:
                    optimizer.lr = lr
                optimizer.step()
                if head_vals is not None:
                    head_lr = (
                        (optimizer.lr if lr is None else lr)
                        * cfg.head_lr_scale
                    )
                    optimizer.step_sparse_rows(
                        head_w, head_rows, head_vals, lr=head_lr)
                optimizer.zero_grad()
                self._steps_done = getattr(self, "_steps_done", 0) + 1
                # derived int8 caches of updated modules are stale; refresh them
                # lazily, optionally amortized over several steps
                if cfg.shadow_refresh_every <= 1 or self._steps_done % cfg.shadow_refresh_every == 0:
                    invalidate_shadows(adapter.head)
                    for li in range(j0, j1):
                        invalidate_shadows(adapter.layers[li])
                # adaptive layer scores from the observed gradient mass
                with torch.no_grad():
                    for i in range(j0, j1):
                        self.layer_scores[i] = (
                            0.9 * self.layer_scores[i] + 0.1 * layer_gn[i]
                        )
                    self.layer_scores.clamp_min_(1e-4)
                    self.layer_scores /= self.layer_scores.mean()
        else:
            # release graph
            loss.detach()
        del logits, log_probs, loss_scaled

        for p in trainable:
            p.requires_grad_(False)

        metrics.seconds = time.perf_counter() - t_start
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        ids: torch.Tensor,
        loss_mask: torch.Tensor,
        window: int = 2048,
        stride: int = 2048,
        head_chunk: int = 512,
        progress=None,
        show_bar: bool = True,
    ) -> float:
        """Frozen sliding-window CE (uses int8 shadows) for validation.

        The head matmul is chunked over positions to bound peak memory.
        ``progress`` receives one message per window so a caller (console or
        Web UI) can show that a long validation is alive, and a tqdm bar is
        drawn on the console.
        """
        adapter = self.adapter
        device = next(self.model.parameters()).device
        ids = ids.to(device)
        loss_mask = loss_mask.to(device)
        n = ids.numel()
        offsets = list(range(0, max(1, n - 1), stride))
        it = tqdm(offsets, desc="validation", total=len(offsets), leave=False,
                  mininterval=1.0, disable=not show_bar, dynamic_ncols=True)
        losses, counts = [], []
        for wi, s in enumerate(it, start=1):
            e = min(s + window, n)
            chunk = ids[s:e]
            if chunk.numel() < 2:
                break
            out = adapter.full_forward(chunk.unsqueeze(0), use_cache=False)
            hidden = out.last_hidden_state[0]           # [T, D]
            labels = ids[s + 1:e + 1]
            m = loss_mask[s + 1:e + 1].float()
            T = min(labels.numel(), hidden.shape[0])
            total, cnt = 0.0, 0.0
            for c0 in range(0, T, head_chunk):
                c1 = min(c0 + head_chunk, T)
                logits = adapter.head_logits(hidden[c0:c1])
                ce = F.cross_entropy(logits.float(), labels[c0:c1], reduction="none")
                total += float((ce * m[c0:c1]).sum())
                cnt += float(m[c0:c1].sum())
            losses.append(total)
            counts.append(cnt)
            if progress:
                try:
                    progress(f"validation window {wi}/{len(offsets)} "
                             f"({chunk.numel()} tokens)")
                except Exception:
                    pass
        total = sum(counts)
        if total == 0:
            return float("nan")
        return sum(losses) / total
