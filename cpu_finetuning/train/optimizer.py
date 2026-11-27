"""CPU-friendly optimizer: block-wise 8-bit Adam with stochastic rounding.

Design notes (informed by the classic pitfalls of quantized optimizers):

* moments are stored in int8 with **per-row (per output channel) scales**.
  A single per-tensor scale zeroes small second-moment entries, which makes
  sqrt(v_hat) collapse to zero and turns the Adam update into
  lr * m / eps -> catastrophic divergence.  Per-row scales keep enough
  resolution for realistic gradient distributions.
* the update ratio m_hat / sqrt(v_hat) is additionally clamped to +-5 by
  default as a safety net against residual quantization artifacts.
* master parameters may live in bfloat16; updates are written back with
  unbiased stochastic rounding so tiny updates are not lost
  (E[SR_bF16(x)] = x).
* all state math is row-chunked: even a 250M-parameter tied embedding head
  updates with only a few tens of MB of transient memory.
"""

from __future__ import annotations

import math

import torch

_ROW_ELEMS = 1 << 22  # process ~4M elements per chunk


def _sr_to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unbiased stochastic rounding from fp32 into dtype (bf16 aware)."""
    if dtype != torch.bfloat16:
        return x.to(dtype)
    nearest = x.to(torch.bfloat16)
    err = x - nearest.float()
    step = x.abs().clamp_min(1e-30) * (2.0 ** -8)
    u = torch.rand_like(x)
    bump = (err.abs() > u * step).to(x.dtype) * torch.sign(err) * step
    return (nearest.float() + bump).to(torch.bfloat16)


class Adam8Bit:
    """AdamW with per-row int8 moments, chunked updates and optional SR."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        stochastic_rounding: bool = True,
        ratio_clip: float = 5.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.sr = stochastic_rounding
        self.ratio_clip = ratio_clip
        self.state: dict[int, dict] = {}
        self._step_count = 0

    # ------------------------------------------------------------------
    def _state_for(self, p: torch.Tensor) -> dict:
        key = p.data_ptr()
        st = self.state.get(key)
        if st is None:
            rows = p.shape[0] if p.dim() >= 2 else 1
            cols = p.numel() // rows
            st = {
                "step": 0,
                "rows": rows,
                "cols": cols,
                "m_q": torch.zeros(p.numel(), dtype=torch.int8,
                                   device=p.device),
                "v_q": torch.zeros(p.numel(), dtype=torch.int8,
                                   device=p.device),
                "m_s": torch.zeros(rows, dtype=torch.float32,
                                   device=p.device),
                "v_s": torch.zeros(rows, dtype=torch.float32,
                                   device=p.device),
            }
            self.state[key] = st
        return st

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.reshape(-1)
            st = self._state_for(p)
            st["step"] += 1
            t = st["step"]
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            flat_master = p.reshape(-1)
            rows, cols = st["rows"], st["cols"]
            row_chunk = max(1, _ROW_ELEMS // max(1, cols))

            for r0 in range(0, rows, row_chunk):
                r1 = min(r0 + row_chunk, rows)
                s0, s1 = r0 * cols, r1 * cols
                g_c = g[s0:s1].float().view(r1 - r0, cols)
                # Standard Adam moments.  The old t==1 special case stored
                # the *undecayed* g / g^2 and still divided by (1-b1)/(1-b2)
                # below, so the first update was ~3x too small and the stored
                # moments were ~10x/~1000x too large (visible as a one-step
                # dip followed by a distorted second step).
                if t == 1:
                    m_c = g_c * (1 - b1)
                    v_c = (g_c * g_c) * (1 - b2)
                else:
                    m_prev = (
                        st["m_q"][s0:s1].view(r1 - r0, cols).float()
                        * st["m_s"][r0:r1].unsqueeze(1)
                    )
                    v_prev = (
                        st["v_q"][s0:s1].view(r1 - r0, cols).float()
                        * st["v_s"][r0:r1].unsqueeze(1)
                    )
                    m_c = m_prev * b1 + g_c * (1 - b1)
                    v_c = v_prev * b2 + (g_c * g_c) * (1 - b2)
                # store moments (per-row scales)
                am = m_c.abs().amax(dim=1).clamp_min(1e-12)
                st["m_s"][r0:r1] = (am / 127.0)
                st["m_q"][s0:s1] = (
                    torch.round(m_c / am.unsqueeze(1) * 127.0)
                    .clamp_(-127, 127).to(torch.int8).view(-1)
                )
                av = v_c.abs().amax(dim=1).clamp_min(1e-12)
                st["v_s"][r0:r1] = (av / 127.0)
                st["v_q"][s0:s1] = (
                    torch.round(v_c / av.unsqueeze(1) * 127.0)
                    .clamp_(-127, 127).to(torch.int8).view(-1)
                )
                # Adam update with quantization-safe denominator
                denom = (v_c / bc2).sqrt()
                floor = (v_c.mean(dim=1, keepdim=True) * 1e-4).sqrt()
                denom = torch.maximum(denom, floor).add_(self.eps)
                ratio = (m_c / bc1) / denom
                if self.ratio_clip > 0:
                    ratio = ratio.clamp_(-self.ratio_clip, self.ratio_clip)
                update = self.lr * ratio
                if self.weight_decay > 0:
                    master_c = flat_master[s0:s1].view(r1 - r0, cols)
                    update = update + self.lr * self.weight_decay * master_c.float()
                new_c = flat_master[s0:s1].view(r1 - r0, cols).float() - update
                if self.sr and p.dtype == torch.bfloat16:
                    flat_master[s0:s1] = _sr_to_dtype(
                        new_c.view(-1), torch.bfloat16
                    )
                else:
                    flat_master[s0:s1] = new_c.view(-1).to(p.dtype)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_sparse_rows(self, p: torch.Tensor, rows: torch.Tensor,
                         grads: torch.Tensor, lr: float | None = None) -> None:
        """AdamW update for a subset of rows of a 2-D parameter.

        The (tied) LM head is a 128k x 2048 matrix.  Its dense gradient is
        1 GB in fp32 and a dense update touches every one of the 262M rows x
        columns every step, although ASCOT trains on a few hundred observed
        tokens: the softmax gradient of all other vocabulary rows is pure
        Adam noise and is zeroed by the caller anyway.  Updating only the
        observed rows keeps the exact same math for them at ~1/500 of the
        memory traffic.
        """
        if rows.numel() == 0:
            return
        st = self._state_for(p)
        st["step"] += 1
        t = st["step"]
        b1, b2 = self.betas
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        rows = rows.to(device=p.device, dtype=torch.long)
        n_rows, cols = st["rows"], st["cols"]
        g = grads.to(device=p.device, dtype=torch.float32)
        m_q = st["m_q"].view(n_rows, cols)
        v_q = st["v_q"].view(n_rows, cols)

        m_prev = m_q.index_select(0, rows).float() * st["m_s"][rows].unsqueeze(1)
        v_prev = v_q.index_select(0, rows).float() * st["v_s"][rows].unsqueeze(1)
        m_c = m_prev * b1 + g * (1 - b1)
        v_c = v_prev * b2 + (g * g) * (1 - b2)

        am = m_c.abs().amax(dim=1).clamp_min(1e-12)
        av = v_c.abs().amax(dim=1).clamp_min(1e-12)
        st["m_s"][rows] = am / 127.0
        st["v_s"][rows] = av / 127.0
        m_q.index_copy_(0, rows, torch.round(m_c / am.unsqueeze(1) * 127.0)
                        .clamp_(-127, 127).to(torch.int8))
        v_q.index_copy_(0, rows, torch.round(v_c / av.unsqueeze(1) * 127.0)
                        .clamp_(-127, 127).to(torch.int8))

        denom = (v_c / bc2).sqrt()
        floor = (v_c.mean(dim=1, keepdim=True) * 1e-4).sqrt()
        denom = torch.maximum(denom, floor).add_(self.eps)
        ratio = (m_c / bc1) / denom
        if self.ratio_clip > 0:
            ratio = ratio.clamp_(-self.ratio_clip, self.ratio_clip)
        lr = self.lr if lr is None else float(lr)
        update = lr * ratio

        master = p.view(n_rows, cols)
        cur = master.index_select(0, rows).float()
        if self.weight_decay > 0:
            update = update + lr * self.weight_decay * cur
        new = cur - update
        if self.sr and p.dtype == torch.bfloat16:
            new = _sr_to_dtype(new.reshape(-1), torch.bfloat16).view(-1, cols)
        master.index_copy_(0, rows, new.to(p.dtype))

    def state_dict(self) -> dict:
        """Serialize moments by parameter index (data_ptr is process-local)."""
        states = []
        for i, p in enumerate(self.params):
            st = self.state.get(p.data_ptr())
            if st is None:
                continue
            states.append({
                "index": i,
                "state": {
                    "step": int(st.get("step", 0)),
                    "rows": int(st["rows"]),
                    "cols": int(st["cols"]),
                    "m_q": st["m_q"].detach().cpu(),
                    "v_q": st["v_q"].detach().cpu(),
                    "m_s": st["m_s"].detach().cpu(),
                    "v_s": st["v_s"].detach().cpu(),
                },
            })
        return {"format": "adam8bit-v9", "lr": self.lr,
                "step_count": self._step_count, "states": states}

    def load_state_dict(self, payload: dict) -> None:
        self.state.clear()
        if not isinstance(payload, dict):
            return
        self.lr = float(payload.get("lr", self.lr))
        self._step_count = int(payload.get("step_count", 0))
        rows = payload.get("states")
        if rows is None:
            # Legacy v8 optimizer.pt: state keyed by old data_ptr values.
            # Those pointers are not stable across processes, so it cannot be
            # restored safely; ignore it rather than corrupting the run.
            return
        for row in rows:
            try:
                idx = int(row.get("index", -1))
                if idx < 0 or idx >= len(self.params):
                    continue
                p = self.params[idx]
                st = dict(row.get("state") or {})
                for key in ("m_q", "v_q"):
                    if key in st:
                        st[key] = st[key].to(device=p.device)
                for key in ("m_s", "v_s"):
                    if key in st:
                        st[key] = st[key].to(device=p.device,
                                               dtype=torch.float32)
                self.state[p.data_ptr()] = st
            except Exception:
                continue

    def reset_state(self) -> None:
        """Drop Adam moments after a rollback to a best checkpoint.

        The moments describe the abandoned trajectory; reusing them would
        immediately pull the restored weights away from the best point.
        """
        self.state.clear()
        self._step_count = 0

    def state_bytes(self) -> int:
        total = 0
        for st in self.state.values():
            total += st["m_q"].numel() + st["v_q"].numel()
        return total


class LRSchedule:
    """Warmup + cosine/linear decay schedule."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0,
                 schedule: str = "cosine", min_lr_ratio: float = 0.1):
        self.opt = optimizer
        self.total = max(1, total_steps)
        # a 20-step warmup is 10% of a normal run but 200% of a 10-step run,
        # which would make the configured LR unreachable; cap it at 10%
        self.warmup = min(int(warmup_steps), max(0, int(self.total * 0.1)))
        self.schedule = schedule
        self.min_ratio = min_lr_ratio
        self.base_lr = optimizer.lr

    def at(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * (step + 1) / max(1, self.warmup)
        prog = (step - self.warmup) / max(1, self.total - self.warmup)
        prog = min(1.0, prog)
        if self.schedule == "linear":
            factor = 1.0 - prog
        else:
            factor = 0.5 * (1 + math.cos(math.pi * prog))
        return self.base_lr * (self.min_ratio + (1 - self.min_ratio) * factor)

    def set_total_steps(self, total_steps: int) -> None:
        """Shorten/extend the horizon after a measured time budget.

        ``base_lr`` is intentionally preserved: this is a schedule correction,
        not a new run or a restart.
        """
        self.total = max(1, int(total_steps))
        self.warmup = min(self.warmup, max(0, int(self.total * 0.1)))

    def set(self, step: int) -> float:
        lr = self.at(step)
        self.opt.lr = lr
        return lr
