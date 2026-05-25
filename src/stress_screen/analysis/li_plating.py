"""
analysis/li_plating.py — Li-plating detection module for stress_screen.

Analyses charge-phase and early-rest-phase data to detect Lithium plating
in individual cell-groups using five complementary sub-methods:

  1. dV/dQ extra peak detection (anomalous inflections near end-of-charge)
  2. Post-charge voltage relaxation speed (faster drop → re-intercalation)
  3. Charge-time anomaly (unusually long charge duration)
  4. Cold-charge anomaly (cells charging colder than the fleet)
  5. Late-charge ΔT — heat-of-plating signature (exothermic plating)

The three electrical signatures (1–3) are gated by temperature: a cell
warmer than ``T_plating_threshold_c`` cannot meaningfully plate, so its
electrical z-scores are scaled down. Cold cells receive full (or boosted)
weight on the electrical signatures.

Public API
----------
run_li_plating_analysis(charge_cell_df, rest_cell_df, params) -> dict[int, MethodResult]
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm.auto import tqdm

from stress_screen.models import MethodResult
from stress_screen.analysis.util import robust_z
from stress_screen._progress import get as _get_progress


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

    T_plating_threshold_c: float = 20.0
    """Charge-mean temperature (°C) above which plating is essentially
    impossible; electrical signatures are gated out above this value."""

    T_default_gate: float = 0.5
    """Gate value used when temperature data is missing for a channel
    (mid-risk — neither fully trust nor fully gate)."""


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
    q_axis: np.ndarray | None = None,
) -> float:
    """Return the sum of peak prominences in the smoothed dV/dQ signal.

    Uses voltage gradient w.r.t. q_axis when provided (physically correct
    Q-domain), otherwise falls back to gradient w.r.t. sample index.
    """
    if len(voltage) < 3:
        return 0.0

    if q_axis is not None:
        dv_dq = np.gradient(voltage, q_axis)
    else:
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
    top_charge_df: pd.DataFrame | None = None,
    n_parallel: int = 1,
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

    # Build pack-level Q axis from charge current when available
    q_pack_time: np.ndarray | None = None
    q_pack_cumul: np.ndarray | None = None
    if top_charge_df is None:
        warnings.warn(
            "run_li_plating_analysis: top_charge_df not provided; dV/dQ computed vs "
            "sample index (physically incorrect). Pass top_charge_df for Q-domain analysis.",
            UserWarning,
            stacklevel=2,
        )
    elif len(top_charge_df) >= 2:
        _top = top_charge_df.sort_values("time_hours")
        q_pack_time = _top["time_hours"].values
        q_pack_cumul = np.concatenate([
            [0.0],
            cumulative_trapezoid(np.abs(_top["current"].values), q_pack_time),
        ]) / max(n_parallel, 1)
    else:  # len == 1
        warnings.warn(
            "run_li_plating_analysis: top_charge_df has fewer than 2 rows; dV/dQ computed vs "
            "sample index (physically incorrect). Pass a valid top_charge_df for Q-domain analysis.",
            UserWarning,
            stacklevel=2,
        )

    channels = sorted(
        set(charge_cell_df["channel_index"].unique())
        | set(rest_cell_df["channel_index"].unique())
    )

    _disable_progress = _get_progress().quiet

    # ------------------------------------------------------------------
    # Sub-method 1: dV/dQ peak prominence sum (charge phase)
    # ------------------------------------------------------------------
    dv_metrics: dict[int, float] = {}

    for ch in tqdm(
        channels,
        desc="  Li-plating (dV/dQ peaks)",
        unit="ch",
        leave=False,
        file=sys.stderr,
        disable=_disable_progress,
    ):
        ch_charge = (
            charge_cell_df[charge_cell_df["channel_index"] == ch]
            .sort_values("time_hours")
        )
        if len(ch_charge) < params.min_charge_points:
            dv_metrics[ch] = np.nan
        else:
            q_ch: np.ndarray | None = None
            if q_pack_time is not None:
                q_ch = np.interp(
                    ch_charge["time_hours"].values, q_pack_time, q_pack_cumul
                )
            dv_metrics[ch] = _compute_dvdq_peak_sum(
                ch_charge["voltage"].values,
                smooth_window=params.dv_smooth_window,
                peak_prominence_pct=params.peak_prominence_pct,
                q_axis=q_ch,
            )

    # ------------------------------------------------------------------
    # Sub-method 2: Post-charge voltage relaxation speed (rest phase)
    # ------------------------------------------------------------------
    relax_metrics: dict[int, float] = {}

    for ch in tqdm(
        channels,
        desc="  Li-plating (relaxation fit)",
        unit="ch",
        leave=False,
        file=sys.stderr,
        disable=_disable_progress,
    ):
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
    # Sub-method 4 & 5: Temperature-based signatures (charge phase)
    # ------------------------------------------------------------------
    T_mean_charge_metrics: dict[int, float] = {}
    dT_late_metrics: dict[int, float] = {}

    has_temperature = "temperature" in charge_cell_df.columns

    for ch in channels:
        ch_charge = (
            charge_cell_df[charge_cell_df["channel_index"] == ch]
            .sort_values("time_hours")
        )

        if not has_temperature or len(ch_charge) == 0:
            T_mean_charge_metrics[ch] = np.nan
            dT_late_metrics[ch] = np.nan
            continue

        temps = ch_charge["temperature"].values.astype(float)

        # All-NaN safeguard
        if np.all(np.isnan(temps)):
            T_mean_charge_metrics[ch] = np.nan
            dT_late_metrics[ch] = np.nan
            continue

        T_mean_charge_metrics[ch] = float(np.nanmean(temps))

        # Late-charge ΔT: last 20% vs 60-80% window
        n = len(temps)
        if n < 5:
            dT_late_metrics[ch] = np.nan
            continue

        i_mid_lo = int(np.floor(0.60 * n))
        i_mid_hi = int(np.floor(0.80 * n))
        i_late_lo = int(np.floor(0.80 * n))
        # Inclusive of end → use n as upper bound

        T_mid = temps[i_mid_lo:i_mid_hi]
        T_late = temps[i_late_lo:n]

        if len(T_mid) == 0 or len(T_late) == 0:
            dT_late_metrics[ch] = np.nan
            continue

        T_mid_mean = float(np.nanmean(T_mid))
        T_late_mean = float(np.nanmean(T_late))

        if np.isnan(T_mid_mean) or np.isnan(T_late_mean):
            dT_late_metrics[ch] = np.nan
        else:
            dt_late = T_late_mean - T_mid_mean
            dT_late_metrics[ch] = dt_late if abs(dt_late) >= 0.3 else np.nan

    # ------------------------------------------------------------------
    # Compute robust z-scores for each sub-method
    # ------------------------------------------------------------------
    dv_arr = np.array([dv_metrics[ch] for ch in channels])
    relax_arr = np.array([relax_metrics[ch] for ch in channels])
    time_arr = np.array([time_metrics[ch] for ch in channels])
    T_mean_arr = np.array([T_mean_charge_metrics[ch] for ch in channels])
    dT_late_arr = np.array([dT_late_metrics[ch] for ch in channels])

    dv_scores = robust_z(dv_arr)
    relax_scores = robust_z(relax_arr)
    time_scores = robust_z(time_arr)

    # Cold-charge anomaly: INVERTED — cold (low T) means high z (suspicious)
    cold_scores = -robust_z(T_mean_arr)
    # Heat-of-plating: higher ΔT_late → higher z (suspicious)
    heat_scores = robust_z(dT_late_arr)

    # ------------------------------------------------------------------
    # Cold-temperature gating for electrical signatures
    # ------------------------------------------------------------------
    T_thr = params.T_plating_threshold_c
    gates: list[float] = []
    for T_mean in T_mean_arr:
        if np.isnan(T_mean):
            gate = params.T_default_gate
        else:
            gate = max(0.0, T_thr - float(T_mean)) / T_thr
            gate = min(1.5, gate)
        gates.append(gate)

    # ------------------------------------------------------------------
    # Assemble MethodResult per channel
    # ------------------------------------------------------------------
    results: dict[int, MethodResult] = {}

    for i, ch in enumerate(channels):
        dv_z = float(dv_scores[i])
        relax_z = float(relax_scores[i])
        time_z = float(time_scores[i])
        cold_z = float(cold_scores[i])
        heat_z = float(heat_scores[i])
        gate = float(gates[i])

        # Force temperature signatures to NaN if no T data for this channel
        if np.isnan(T_mean_arr[i]):
            cold_z = float("nan")
            heat_z = float("nan")

        # Apply gating to electrical signatures
        gated_dv_z = dv_z * gate if not np.isnan(dv_z) else float("nan")
        gated_relax_z = relax_z * gate if not np.isnan(relax_z) else float("nan")
        gated_time_z = time_z * gate if not np.isnan(time_z) else float("nan")

        five = [gated_dv_z, gated_relax_z, gated_time_z, cold_z, heat_z]
        valid = [s for s in five if not np.isnan(s)]
        z = float(np.nanmean(five)) if valid else float("nan")

        verdict = _verdict(z, params.z_thresh)

        results[ch] = MethodResult(
            method_name="li_plating",
            z_score=z,
            verdict=verdict,
            metadata={
                # Existing electrical signatures
                "dv_dq_z": dv_z,
                "relaxation_z": relax_z,
                "charge_time_z": time_z,
                "peak_prominence_sum": float(dv_metrics[ch]),
                "tau_inv": float(relax_metrics[ch]),
                "charge_duration_h": float(time_metrics[ch]),
                # New temperature signatures
                "cold_z": cold_z,
                "heat_z": heat_z,
                "T_mean_charge": float(T_mean_charge_metrics[ch]),
                "dT_late": float(dT_late_metrics[ch]),
                "temperature_gate": gate,
                # Gated electrical z-scores (after T-gating)
                "gated_dv_dq_z": gated_dv_z,
                "gated_relaxation_z": gated_relax_z,
                "gated_charge_time_z": gated_time_z,
            },
        )

    return results
