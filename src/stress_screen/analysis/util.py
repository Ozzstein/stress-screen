"""
analysis/util.py — Shared statistical helpers for stress_screen detection methods.

No I/O, no plotting.  Imported by rest.py and li_plating.py.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# OCV relaxation model
# ---------------------------------------------------------------------------

def ocv_model(t: np.ndarray, V_ocv: float, a: float, tau: float, k: float) -> np.ndarray:
    """Physical OCV relaxation model: V(t) = V_ocv + a*exp(-t/tau) - k*t."""
    return V_ocv + a * np.exp(-t / tau) - k * t


# ---------------------------------------------------------------------------
# Robust z-score
# ---------------------------------------------------------------------------

def robust_z(values: np.ndarray) -> np.ndarray:
    """Compute median-MAD robust z-scores.

    Parameters
    ----------
    values:
        1-D array of floats (NaNs are ignored in median/MAD computation but
        propagate in the output).

    Returns
    -------
    np.ndarray of the same length as *values*.
    """
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.full_like(values, np.nan, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return (values - median) / (1.4826 * mad + 1e-12)


# ---------------------------------------------------------------------------
# Two-sided CUSUM
# ---------------------------------------------------------------------------

def cusum_2sided(
    residuals: np.ndarray,
    k_sigma: float = 0.5,
    h_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, int | None, int]:
    """Two-sided CUSUM change-point detector.

    Parameters
    ----------
    residuals:
        Residual series (zero-mean recommended; the function uses nanstd of
        the raw residuals as its noise estimate).
    k_sigma:
        Allowance as a multiple of sigma.  Default 0.5 (detect 1-sigma shift).
    h_sigma:
        Decision threshold as a multiple of sigma.  Default 4.0.

    Returns
    -------
    S_pos : np.ndarray
        Positive-direction CUSUM statistic (reset to 0 on alarm).
    S_neg : np.ndarray
        Negative-direction CUSUM statistic (reset to 0 on alarm).
    first_alarm_idx : int | None
        Index of the first alarm (any direction), or None.
    n_alarms : int
        Total number of alarms (positive + negative combined).
    """
    residuals = np.asarray(residuals, dtype=float)
    sigma = np.nanstd(residuals)

    if sigma < 1e-10:
        return (
            np.zeros_like(residuals),
            np.zeros_like(residuals),
            None,
            0,
        )

    k = k_sigma * sigma
    h = h_sigma * sigma

    n = len(residuals)
    S_pos = np.zeros(n)
    S_neg = np.zeros(n)
    alarms_pos: list[int] = []
    alarms_neg: list[int] = []

    for i in range(1, n):
        r = residuals[i] if not np.isnan(residuals[i]) else 0.0
        S_pos[i] = max(0.0, S_pos[i - 1] + r - k)
        S_neg[i] = max(0.0, S_neg[i - 1] - r - k)
        if S_pos[i] > h:
            alarms_pos.append(i)
            S_pos[i] = 0.0
        if S_neg[i] > h:
            alarms_neg.append(i)
            S_neg[i] = 0.0

    all_alarms = sorted(alarms_pos + alarms_neg)
    first = all_alarms[0] if all_alarms else None
    return S_pos, S_neg, first, len(all_alarms)
