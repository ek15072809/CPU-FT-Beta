"""Anchor span selection for ASCOT.

A training step consumes one packed sequence of ``context_length`` tokens
(e.g. 32768).  Instead of backpropagating through all of them, ASCOT samples a
small number of *anchor spans* with probabilities proportional to a cheap
importance score and re-weights their losses so that the resulting gradient is
an unbiased (Horvitz-Thompson) estimate of the full-sequence gradient.

Importance signals (all computable in O(N) cheap vector ops):

* **novelty**   - relative embedding distance between consecutive tokens
  (a surprisal proxy available before any model forward),
* **mask density** - fraction of tokens that carry loss (SFT assistant
  tokens), which is where supervised learning value lives,
* **uniform floor** - a mixture weight that keeps every token's inclusion
  probability strictly positive, guaranteeing unbiasedness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SpanPlan:
    """Sampled anchor spans for one step."""

    starts: torch.Tensor          # [R] int64, span start positions
    prefix_lens: torch.Tensor     # [R] int64, usable prefix length per span
    probs: torch.Tensor           # [R] float32, inclusion probability per span
    weights: torch.Tensor         # [R] float32, 1 / (R * probs) reweighting


class SpanSelector:
    def __init__(
        self,
        block_size: int = 512,
        span_length: int = 128,
        num_spans: int = 4,
        uniform_floor: float = 0.25,
        mask_weight: float = 1.0,
        novelty_weight: float = 1.0,
        seed: int = 0,
    ):
        self.block_size = block_size
        self.span_length = span_length
        self.num_spans = num_spans
        self.uniform_floor = uniform_floor
        self.mask_weight = mask_weight
        self.novelty_weight = novelty_weight
        self.gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def block_scores(
        self,
        embeds: torch.Tensor,      # [N, D] float32 token embeddings
        loss_mask: torch.Tensor,   # [N] float32 (1 where token trains)
    ) -> torch.Tensor:
        n = embeds.shape[0]
        bs = self.block_size
        nb = (n + bs - 1) // bs
        pad = nb * bs - n
        if pad:
            embeds = torch.cat([embeds, embeds[-1:].expand(pad, -1)])
            loss_mask = torch.cat([
                loss_mask,
                torch.zeros(pad, dtype=loss_mask.dtype,
                            device=loss_mask.device),
            ])
        e = embeds.view(nb, bs, -1)
        m = loss_mask.view(nb, bs)
        # novelty: relative displacement of consecutive embeddings
        d = (e[:, 1:] - e[:, :-1]).norm(dim=-1) / (e[:, 1:].norm(dim=-1) + 1e-6)
        novelty = torch.cat(
            [d.new_zeros(nb, 1), d], dim=1
        ).mean(dim=1)
        mask_density = m.mean(dim=1)
        score = (
            self.novelty_weight * (novelty / (novelty.mean() + 1e-6))
            + self.mask_weight * (mask_density + 0.05)
        )
        return score

    # ------------------------------------------------------------------
    @torch.no_grad()
    def plan(
        self,
        seq_len: int,
        block_scores: torch.Tensor,
        min_prefix: int,
        boost_ranges: list[tuple[int, int, float]] | None = None,
    ) -> SpanPlan:
        """Sample span starts proportional to block importance.

        Spans are aligned so that each has at least ``min_prefix`` tokens of
        local context available before it and fits inside the sequence.
        ``boost_ranges`` = [(start, end, multiplier)] can bias sampling toward
        e.g. the later region and distant-reference spans of a long document.
        """
        L = self.span_length
        R = self.num_spans
        lo = min_prefix
        # Selection is tiny (one score per 512-token block).  Keeping it on
        # CPU also lets the CPU generator be used for reproducible sampling
        # regardless of the training device.
        block_scores = block_scores.detach().to(device="cpu", dtype=torch.float32)
        # the last span position predicts ids[s+L], so s may be at most
        # seq_len-L-1 (using seq_len-L reads the label out of bounds and
        # silently zero-pads it in torch indexing)
        hi = seq_len - L - 1
        if hi < lo:
            # degenerate short sequence: allow any start
            lo = 0
            hi = max(0, seq_len - L - 1)
        if hi <= lo:
            starts = torch.tensor([lo], dtype=torch.long)
            probs = torch.ones(1)
        else:
            nb = block_scores.numel()
            # span-start -> block index mapping
            centers = torch.arange(lo, hi + 1, dtype=torch.float32)
            block_idx = ((centers + L / 2) / self.block_size).long().clamp(0, nb - 1)
            raw = block_scores[block_idx].clone()
            if boost_ranges:
                for bs, be, mult in boost_ranges:
                    mask = (centers >= float(bs)) & (centers < float(be))
                    if bool(mask.any()):
                        raw[mask] = raw[mask] * float(mult)
            raw = raw / (raw.sum() + 1e-6)
            # mixture with uniform floor for unbiasedness
            n_opts = hi - lo + 1
            alpha = self.uniform_floor
            pi = alpha / n_opts + (1 - alpha) * (raw / raw.sum())
            picked = torch.multinomial(
                pi, R, replacement=True, generator=self.gen
            )
            starts = picked + lo
            probs = pi[picked]
        # Horvitz-Thompson weights estimating the *mean* loss over the
        # sequence: E[sum_r w_r * loss_r] = mean loss over all span options.
        n_opts = max(1, int(hi - lo + 1)) if hi > lo else 1
        weights = 1.0 / (R * probs * n_opts)
        prefix_lens = starts.clamp(min=1)  # prefix length decided by caller budget
        return SpanPlan(starts=starts, prefix_lens=prefix_lens, probs=probs, weights=weights)
