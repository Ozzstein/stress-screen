"""
analysis/li_plating.py — Li-plating detection module for stress_screen.

Analyses charge-phase and early-rest-phase data to detect Lithium plating
in individual cell-groups using three complementary sub-methods:

  1. dV/dQ extra peak detection (anomalous inflections near end-of-charge)
  2. Post-charge voltage relaxation speed (faster drop → re-intercalation)
  3. Charge-time anomaly (unusually long charge duration)

Public API
----------
run_li_plating_analysis(charge_cell_df, rest_cell_df, params) -> dict[int, MethodResult]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import OptimizeWarning, curve_fit

from stress_screen.models import MethodResult
from stress_screen.analysis.util import robust_z


# ---------------------------------------------------------------------------
# LiPlatingParams
# ---------------------------------------------------------------------------

@dataclass
class LiPlatingParams:
    """Tunable parameters for the Li-plating detection analysis."""

    z_thresh: float = 2.0
    """Robust z threshold for a HIGH verdict."""

    min_charge_points: int = 60
    """Minimum charge-phase samples per cell to run analysis."""

    relaxation_window_h: float = 0.5
    """Hours of post-charge rest to analyse (first N hours)."""

    dv_smooth_window: int = 11
    """Savitzky-Golay smoothing window length for dV/dQ."""

    peak_prominence_pct: float = 0.05
    """Minimum peak prominence as fraction of the dV/dQ range."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exp_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Simple exponential decay model: V(t) = A * exp(-t/tau) + C."""
    return A * np.exp(-t / tau) + C


def _verdict(z: float, z_thresh: float) -> str:
    if np.isnan(z):
        return "NORMAL"
    if z >= z_thresh:
        return "HIGH"
    if z >= 1.0:
        return "ELEVATED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Sub-method 1: dV/dQ extra peak detection
# ---------------------------------------------------------------------------

def _compute_dvdq_peak_sum(
    voltage: np.ndarray,
    smooth_window: int,
    peak_prominence_pct: float,
) -> float:
    """Return the sum of peak prominences in the smoothed dV/dQ signal.

    Uses voltage gradient w.r.t. index for relative comparison across cells
    (all cells share the same time resolution).
    """
    if len(voltage) < 3:
        return 0.0

    dv_dq = np.gradient(voltage)

    # Apply Savitzky-Golay smoothing if we have enough points
    if len(dv_dq) >= smooth_window:
        try:
            dv_dq_smooth = signal.savgol_filter(
                dv_dq, window_length=smooth_window, polyorder=2
            )
        except Exception:
            dv_dq_smooth = dv_dq
    else:
        dv_dq_smooth = dv_dq

    ptp = np.ptp(dv_dq_smooth)
    if ptp < 1e-12:
        return 0.0

    min_prominence = peak_prominence_pct * ptp

    try:
        _, props = signal.find_peaks(
            dv_dq_smooth, prominence=min_prominence
        )
        prominences = props.get("prominences", np.array([]))
        return float(np.sum(prominences)) if len(prominences) > 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Sub-method 2: Post-charge voltage relaxation speed
# ---------------------------------------------------------------------------

def _compute_tau_inv(
    time_hours: np.ndarray,
    voltage: np.ndarray,
    min_points: int,
) -> float:
    """Fit an exponential decay to post-charge rest voltage.

    Returns 1/tau (faster relaxation = higher value).
    Returns np.nan on failure or insufficient data.
    """
    if len(time_hours) < min_points:
        return np.nan

    # Shift time to start from 0
    t = time_hours - time_hours.min()
    v = voltage

    v_min = float(np.nanmin(v))
    v_max = float(np.nanmax(v))
    v_mean = float(np.nanmean(v))

    if v_max - v_min < 1e-6:
        return np.nan

    p0 = [0.01, 0.1, v_mean]
    bounds = (
        [0.0,   0.001, v_min],
        [0.5,  10.0,  v_max],
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _exp_decay, t, v,
                p0=p0, bounds=bounds,
                maxfev=20_000, method="trf",
            )
        _A, tau, _C = popt
        return float(1.0 / tau) if tau > 1e-9 else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_li_plating_analysis(
    charge_cell_df: pd.DataFrame,
    rest_cell_df: pd.DataFrame,
    params: LiPlatingParams | None = None,
) -> dict[int, MethodResult]:
    """Detect Li-plating signatures.

    Parameters
    ----------
    charge_cell_df:
        Long-format DataFrame with columns ``time_hours``, ``channel_index``,
        ``voltage``, ``temperature`` — charge segment only.
    rest_cell_df:
        Long-format DataFrame with same columns — rest segment only
        (first 2 hours recommended).
    params:
        Tunable parameters; uses defaults if None.

    Returns
    -------
    dict mapping ``channel_index`` (int) to a ``MethodResult`` with
    ``method_name="li_plating"``.
    """
    if params is None:
        params = LiPlatingParams()

    channels = sorted(
        set(charge_cell_df["channel_index"].unique())
        | set(rest_cell_df["channel_index"].unique())
    )

    # ------------------------------------------------------------------
    # Sub-method 1: dV/dQ peak prominence sum (charge phase)
    # ------------------------------------------------------------------
    dv_metrics: dict[int, float] = {}

    for ch in channels:
        ch_charge = (
            charge_cell_df[charge_cell_df["channel_index"] == ch]
            .sort_values("time_hours")
        )
        if len(ch_charge) < params.min_charge_points:
            dv_metrics[ch] = np.nan
        else:
            dv_metrics[ch] = _compute_dvdq_peak_sum(
                ch_charge["voltage"].values,
                smooth_window=params.dv_smooth_window,
                peak_prominence_pct=params.peak_prominence_pct,
            )

    # ------------------------------------------------------------------
    # Sub-method 2: Post-charge voltage relaxation speed (rest phase)
    # ------------------------------------------------------------------
    relax_metrics: dict[int, float] = {}

    for ch in channels:
        ch_rest = rest_cell_df[rest_cell_df["channel_index"] == ch].sort_values("time_hours")
        if len(ch_rest) > 0:
            ch_start = ch_rest["time_hours"].min()
            ch_rest = ch_rest[ch_rest["time_hours"] <= ch_start + params.relaxation_window_h]
        relax_metrics[ch] = _compute_tau_inv(
            ch_rest["time_hours"].values,
            ch_rest["voltage"].values,
            min_points=params.min_charge_points,
        )

    # ------------------------------------------------------------------
    # Sub-method 3: Charge-time anomaly (charge phase)
    # ------------------------------------------------------------------
    time_metrics: dict[int, float] = {}

    for ch in channels:
        ch_charge = charge_cell_df[charge_cell_df["channel_index"] == ch]
        if len(ch_charge) == 0:
            time_metrics[ch] = np.nan
        else:
            t = ch_charge["time_hours"]
            time_metrics[ch] = float(t.max() - t.min())

    # ------------------------------------------------------------------
    # Compute robust z-scores for each sub-method
    # ------------------------------------------------------------------
    dv_arr = np.array([dv_metrics[ch] for ch in channels])
    relax_arr = np.array([relax_metrics[ch] for ch in channels])
    time_arr = np.array([time_metrics[ch] for ch in channels])

    dv_scores = robust_z(dv_arr)
    relax_scores = robust_z(relax_arr)
    time_scores = robust_z(time_arr)

    # ------------------------------------------------------------------
    # Assemble MethodResult per channel
    # ------------------------------------------------------------------
    results: dict[int, MethodResult] = {}

    for i, ch in enumerate(channels):
        dv_z = float(dv_scores[i])
        relax_z = float(relax_scores[i])
        time_z = float(time_scores[i])

        valid = [s for s in [dv_z, relax_z, time_z] if not np.isnan(s)]
        z = float(np.nanmean([dv_z, relax_z, time_z])) if valid else np.nan

        verdict = _verdict(z, params.z_thresh)

        results[ch] = MethodResult(
            method_name="li_plating",
            z_score=z,
            verdict=verdict,
            metadata={
                "dv_dq_z": dv_z,
                "relaxation_z": relax_z,
                "charge_time_z": time_z,
                "peak_prominence_sum": float(dv_metrics[ch]),
                "tau_inv": float(relax_metrics[ch]),
                "charge_duration_h": float(time_metrics[ch]),
            },
        )

    return results
