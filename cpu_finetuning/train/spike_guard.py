"""Gradient spike detection (requirement: skip or rescale abnormal steps).

Maintains a rolling window of recent gradient norms.  A step is flagged as a
*spike* when its grad norm exceeds the window average by more than a
configured percentage.  Flagged steps can then be:

* ``skip``  - the optimizer update is skipped entirely (weights unchanged),
* ``scale`` - the gradients are scaled by ``avg / current`` so the weight
  update magnitude matches a typical recent step,
* ``off``   - disabled.

The two actions are mutually exclusive by construction (single mode select).
Spike norms are excluded from the rolling average so that one outlier does
not mask the following ones.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class SpikeDecision:
    is_spike: bool = False
    action: str = "none"          # none | skip | scale
    scale: float = 1.0
    avg_norm: float = 0.0
    ratio: float = 1.0


class GradSpikeGuard:
    def __init__(
        self,
        mode: str = "off",
        threshold_pct: float = 100.0,
        window: int = 20,
        min_samples: int = 5,
    ):
        self.mode = mode if mode in ("off", "skip", "scale") else "off"
        self.threshold_pct = max(1.0, float(threshold_pct))
        self.window = max(2, int(window))
        # min_samples can never exceed the window, otherwise a small window
        # would disable detection entirely
        self.min_samples = min(max(1, int(min_samples)), self.window)
        self._history: deque[float] = deque(maxlen=self.window)

    def check(self, grad_norm: float) -> SpikeDecision:
        """Decide what to do with this step's gradient norm."""
        d = SpikeDecision(avg_norm=self._average())
        if self.mode == "off" or grad_norm <= 0:
            self._record(grad_norm)
            return d
        avg = self._average()
        if avg <= 0 or len(self._history) < self.min_samples:
            self._record(grad_norm)
            return d
        d.avg_norm = avg
        d.ratio = grad_norm / avg
        if grad_norm > avg * (1.0 + self.threshold_pct / 100.0):
            d.is_spike = True
            if self.mode == "skip":
                d.action = "skip"
            elif self.mode == "scale":
                d.action = "scale"
                d.scale = min(1.0, avg / grad_norm)
            # spike norms are NOT recorded: they must not contaminate the
            # baseline that future spikes are compared against
            return d
        self._record(grad_norm)
        return d

    def _record(self, gn: float):
        if gn > 0:
            self._history.append(gn)

    def _average(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
