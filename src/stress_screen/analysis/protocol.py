"""Protocol metadata — describes the test protocol so analysis thresholds
can scale with C-rate, chemistry, and voltage window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ProtocolMetadata:
    """Describes the test protocol that produced the data being analysed.

    Used by detection methods to scale thresholds that would otherwise be
    hard-coded for a single protocol.
    """

    chemistry: Literal["LFP", "NMC", "LCO", "LTO"] = "LFP"
    c_rate: float = 0.5
    """Cell-level C-rate during the charge phase (1.0 = 1C)."""
    nominal_capacity_ah: float = 2.5
    voltage_window: tuple[float, float] = (3.0, 3.65)
    """(V_min, V_max) — used as fit bounds for OCV models."""

    def dqdv_prominence_pct(self) -> float:
        """C-rate-aware peak prominence threshold for dQ/dV peak detection.

        Faster charges produce noisier dQ/dV curves because incremental
        capacity is computed over a smaller voltage window per sample.
        Empirical scaling: 0.05 baseline at 0.5C, +0.02 per C above 0.5.
        """
        return 0.05 + 0.02 * max(0.0, self.c_rate - 0.5)

    def dt_late_noise_floor_k(self) -> float:
        """C-rate-aware noise floor for late-charge ΔT.

        Faster charges produce larger absolute thermal signatures, so the
        noise floor must scale up to preserve specificity.
        Empirical scaling: 0.3 K baseline at 0.5C, +0.2 K per C above 0.5.
        """
        return 0.3 + 0.2 * max(0.0, self.c_rate - 0.5)
