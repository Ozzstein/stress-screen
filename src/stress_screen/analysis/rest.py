"""
analysis/rest.py — M1–M6 OCV-rest detection methods for stress_screen.

Each method operates on a single rest segment (post-settling) of the long-format
cell voltage DataFrame and returns a MethodResult per channel.

Public API
----------
run_rest_analysis(rest_cell_df, topology, params) -> dict[int, list[MethodResult]]
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm.auto import tqdm

from stress_screen.models import MethodResult, PackTopology
from stress_screen.analysis.util import cusum_2sided, ocv_model, robust_z
from stress_screen._progress import get as _get_progress


# ---------------------------------------------------------------------------
# RestParams
# ---------------------------------------------------------------------------

@dataclass
class RestParams:
    """Tunable parameters for the six rest-phase detection methods."""

    settling_h: float = 2.0
    """Hours of rest to discard at the start (OCV settling transient)."""

    z_thresh: float = 2.0
    """Robust z threshold for a HIGH verdict."""

    voltage_bounds: tuple[float, float] = (3.0, 3.65)
    """(V_low, V_high) — LFP-default OCV bounds for the M1 curve fit."""

    cusum_k_sigma: float = 0.5
    """CUSUM allowance as a multiple of sigma."""

    cusum_h_sigma: float = 4.0
    """CUSUM decision threshold as a multiple of sigma."""

    min_points: int = 60
    """Minimum data points required per channel to run analysis."""

    k_max: float = 0.05
    """Upper bound for the self-discharge rate k in the OCV curve fit.

    Default 0.05 h⁻¹ is wide enough to capture severely discharging cells
    (the original hardcoded 0.005 was too tight for some real-world cases).
    """


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------

def _verdict(z: float, z_thresh: float) -> str:
    if np.isnan(z):
        return "NORMAL"
    if z >= z_thresh:
        return "HIGH"
    if z >= 1.0:
        return "ELEVATED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# NaN result helper
# ---------------------------------------------------------------------------

def _nan_result(method_name: str, metadata: dict[str, Any]) -> MethodResult:
    return MethodResult(
        method_name=method_name,
        z_score=float("nan"),
        verdict="NORMAL",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_rest_analysis(
    rest_cell_df: pd.DataFrame,
    topology: PackTopology,
    params: RestParams | None = None,
) -> dict[int, list[MethodResult]]:
    """Run M1–M6 detection methods on rest-phase cell data.

    Parameters
    ----------
    rest_cell_df:
        Long-format DataFrame with columns ``time_hours``, ``channel_index``,
        ``voltage``, ``temperature`` — rest segment only.
    topology:
        Pack topology (used to enumerate valid channel indices).
    params:
        Tunable parameters; uses defaults if None.

    Returns
    -------
    dict mapping ``channel_index`` (int) to a list of six ``MethodResult``
    objects (one per method, M1–M6).
    """
    if params is None:
        params = RestParams()

    channels = sorted(rest_cell_df["channel_index"].unique())

    # ------------------------------------------------------------------ #
    # Pre-process: build per-channel data dictionaries                     #
    # ------------------------------------------------------------------ #
    # Shift time to start from 0 at the first rest sample, then apply
    # settling cutoff so each method works on the settled window only.
    t0 = rest_cell_df["time_hours"].min()

    chan_data: dict[int, dict] = {}
    for ch in channels:
        cd = (
            rest_cell_df[rest_cell_df["channel_index"] == ch]
            .sort_values("time_hours")
            .copy()
        )
        # Relative time (0 = start of this rest segment)
        cd["_t_rel"] = cd["time_hours"] - t0

        # Settled window (after OCV settling transient)
        settled = cd[cd["_t_rel"] >= params.settling_h]

        t_all = cd["_t_rel"].values
        v_all = cd["voltage"].values
        t_set = settled["_t_rel"].values
        v_set = settled["voltage"].values
        temp_set = settled["temperature"].values

        chan_data[ch] = {
            "t_all": t_all,
            "v_all": v_all,
            "t_set": t_set,
            "v_set": v_set,
            "temp_set": temp_set,
            "n_set": len(t_set),
        }

    # ------------------------------------------------------------------ #
    # M1 — OCV-fit self-discharge rate k                                   #
    # ------------------------------------------------------------------ #
    m1_k: dict[int, float] = {}
    m1_V_ocv: dict[int, float] = {}
    m1_tau: dict[int, float] = {}
    m1_fit_ok: dict[int, bool] = {}
    m1_popt: dict[int, tuple] = {}

    V_low, V_high = params.voltage_bounds

    _disable_progress = _get_progress().quiet
    for ch in tqdm(
        channels,
        desc="  M1 (OCV fit)",
        unit="ch",
        leave=False,
        file=sys.stderr,
        disable=_disable_progress,
    ):
        d = chan_data[ch]
        if d["n_set"] < params.min_points:
            m1_k[ch] = np.nan
            m1_V_ocv[ch] = np.nan
            m1_tau[ch] = np.nan
            m1_fit_ok[ch] = False
            continue

        t = d["t_set"]
        v = d["v_set"]

        p0 = [np.nanmean(v), 0.05, 1.0, 0.0001]
        bounds = (
            [V_low, 0.0,  0.05, 0.0        ],
            [V_high, 0.4, 24.0, params.k_max],
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    ocv_model, t, v,
                    p0=p0, bounds=bounds,
                    maxfev=20_000, method="trf",
                )
            V_ocv_f, a_f, tau_f, k_f = popt
            m1_k[ch] = float(k_f)
            m1_V_ocv[ch] = float(V_ocv_f)
            m1_tau[ch] = float(tau_f)
            m1_fit_ok[ch] = True
            m1_popt[ch] = tuple(popt)
        except Exception:
            m1_k[ch] = np.nan
            m1_V_ocv[ch] = np.nan
            m1_tau[ch] = np.nan
            m1_fit_ok[ch] = False

    # Compute robust_z for M1 across all channels
    k_arr = np.array([m1_k[ch] for ch in channels])
    k_z_arr = robust_z(k_arr)
    m1_z: dict[int, float] = {ch: float(k_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # Build wide-format voltage pivot (settled window, all channels)       #
    # — used by M3 and M6                                                  #
    # ------------------------------------------------------------------ #
    # We need a common time axis; build from the settled slice of the df.
    settled_df = rest_cell_df[
        (rest_cell_df["time_hours"] - t0) >= params.settling_h
    ].copy()
    settled_df["_t_rel"] = settled_df["time_hours"] - t0

    pivot_v: pd.DataFrame = settled_df.pivot_table(
        index="_t_rel", columns="channel_index", values="voltage"
    )

    # ------------------------------------------------------------------ #
    # M2 — Temperature–OCV residual correlation                            #
    # ------------------------------------------------------------------ #
    m2_r: dict[int, float] = {}

    for ch in channels:
        d = chan_data[ch]
        if d["n_set"] < params.min_points:
            m2_r[ch] = np.nan
            continue

        t = d["t_set"]
        v = d["v_set"]
        temp = d["temp_set"]

        # Compute OCV residuals using M1 fit if available; else mean-centre.
        if m1_fit_ok.get(ch, False):
            popt = m1_popt[ch]
            v_resid = v - ocv_model(t, *popt)
        else:
            v_resid = v - np.nanmean(v)

        valid = ~np.isnan(temp) & ~np.isnan(v_resid)
        if valid.sum() >= 5 and np.nanstd(temp[valid]) > 1e-6:
            r, _ = _stats.pearsonr(temp[valid], v_resid[valid])
            m2_r[ch] = float(r)
        else:
            m2_r[ch] = 0.0

    # z-score on |r| (both positive and negative T-V coupling are suspicious)
    r_abs_arr = np.array([abs(m2_r[ch]) for ch in channels])
    r_z_arr = robust_z(r_abs_arr)
    m2_z: dict[int, float] = {ch: float(r_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # M3 — Voltage spread / divergence slope                               #
    # ------------------------------------------------------------------ #
    m3_spread: dict[int, float] = {}

    if not pivot_v.empty:
        fleet_med = pivot_v.median(axis=1)
        for ch in channels:
            if ch in pivot_v.columns and chan_data[ch]["n_set"] >= params.min_points:
                deviation = pivot_v[ch] - fleet_med
                m3_spread[ch] = float(np.nanstd(deviation.values))
            else:
                m3_spread[ch] = np.nan
    else:
        for ch in channels:
            m3_spread[ch] = np.nan

    spread_arr = np.array([m3_spread[ch] for ch in channels])
    spread_z_arr = robust_z(spread_arr)
    m3_z: dict[int, float] = {ch: float(spread_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # M4 — CUSUM on M1 OCV residuals                                       #
    # ------------------------------------------------------------------ #
    m4_n_alarms: dict[int, int] = {}
    m4_first_alarm_h: dict[int, float | None] = {}

    for ch in channels:
        d = chan_data[ch]
        if d["n_set"] < params.min_points:
            m4_n_alarms[ch] = 0
            m4_first_alarm_h[ch] = None
            continue

        t = d["t_set"]
        v = d["v_set"]

        if m1_fit_ok.get(ch, False):
            popt = m1_popt[ch]
            residuals = v - ocv_model(t, *popt)
        else:
            slope, intercept, *_ = _stats.linregress(t, v)
            residuals = v - (slope * t + intercept)

        _, _, first_idx, n_alarms = cusum_2sided(
            residuals,
            k_sigma=params.cusum_k_sigma,
            h_sigma=params.cusum_h_sigma,
        )

        m4_n_alarms[ch] = n_alarms
        m4_first_alarm_h[ch] = float(t[first_idx]) if first_idx is not None else None

    alarms_arr = np.array([float(m4_n_alarms[ch]) for ch in channels])
    alarms_z_arr = robust_z(alarms_arr)
    m4_z: dict[int, float] = {ch: float(alarms_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # M5 — Temperature-corrected self-discharge                            #
    # ------------------------------------------------------------------ #
    m5_k_corr: dict[int, float] = {}
    m5_T_mean: dict[int, float] = {}

    for ch in channels:
        d = chan_data[ch]
        if d["n_set"] < params.min_points or np.isnan(m1_k.get(ch, np.nan)):
            m5_k_corr[ch] = np.nan
            m5_T_mean[ch] = np.nan
            continue

        temp = d["temp_set"]
        k_raw = m1_k[ch]

        valid_temp = temp[~np.isnan(temp)]
        if len(valid_temp) > 0:
            T_mean = float(np.nanmean(valid_temp))
        else:
            T_mean = np.nan

        m5_T_mean[ch] = T_mean

        if not np.isnan(T_mean):
            # Arrhenius approximation: k_corrected = k / (1 + 0.02*(T-25))
            denom = 1.0 + 0.02 * (T_mean - 25.0)
            if abs(denom) > 1e-6:
                m5_k_corr[ch] = k_raw / denom
            else:
                m5_k_corr[ch] = k_raw
        else:
            # No temperature data — fall back to raw k
            m5_k_corr[ch] = k_raw

    k_corr_arr = np.array([m5_k_corr[ch] for ch in channels])
    k_corr_z_arr = robust_z(k_corr_arr)
    m5_z: dict[int, float] = {ch: float(k_corr_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # M6 — Percentile rank tracking                                         #
    # ------------------------------------------------------------------ #
    m6_mean_rank: dict[int, float] = {}
    m6_frac_bot20: dict[int, float] = {}
    m6_rank_slope: dict[int, float] = {}

    if not pivot_v.empty:
        rank_pct = pivot_v.rank(axis=1, pct=True, method="average") * 100.0

        for ch in channels:
            if ch in rank_pct.columns and chan_data[ch]["n_set"] >= params.min_points:
                rs = rank_pct[ch].dropna()
                t_r = rs.index.values
                r_r = rs.values

                mean_rank = float(r_r.mean())
                frac_bot20 = float((r_r < 20.0).mean())

                if len(t_r) >= 2:
                    slope_r, *_ = _stats.linregress(t_r, r_r)
                else:
                    slope_r = 0.0

                m6_mean_rank[ch] = mean_rank
                m6_frac_bot20[ch] = frac_bot20
                m6_rank_slope[ch] = float(slope_r)
            else:
                m6_mean_rank[ch] = np.nan
                m6_frac_bot20[ch] = np.nan
                m6_rank_slope[ch] = np.nan
    else:
        for ch in channels:
            m6_mean_rank[ch] = np.nan
            m6_frac_bot20[ch] = np.nan
            m6_rank_slope[ch] = np.nan

    frac_arr = np.array([m6_frac_bot20[ch] for ch in channels])
    frac_z_arr = robust_z(frac_arr)
    m6_z: dict[int, float] = {ch: float(frac_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # Assemble results                                                      #
    # ------------------------------------------------------------------ #
    results: dict[int, list[MethodResult]] = {}

    for ch in channels:
        d = chan_data[ch]
        insufficient = d["n_set"] < params.min_points

        # M1
        if insufficient or np.isnan(m1_k.get(ch, np.nan)):
            m1_res = _nan_result("M1_ocv_k", {"k": np.nan, "V_ocv": np.nan, "tau": np.nan})
        else:
            z1 = m1_z[ch]
            m1_res = MethodResult(
                method_name="M1_ocv_k",
                z_score=z1,
                verdict=_verdict(z1, params.z_thresh),
                metadata={
                    "k": m1_k[ch],
                    "V_ocv": m1_V_ocv[ch],
                    "tau": m1_tau[ch],
                },
            )

        # M2
        if insufficient:
            m2_res = _nan_result("M2_thermal", {"pearson_r": np.nan})
        else:
            z2 = m2_z[ch]
            m2_res = MethodResult(
                method_name="M2_thermal",
                z_score=z2,
                verdict=_verdict(z2, params.z_thresh),
                metadata={"pearson_r": m2_r[ch]},
            )

        # M3
        if insufficient or np.isnan(m3_spread.get(ch, np.nan)):
            m3_res = _nan_result("M3_spread", {"spread_std": np.nan})
        else:
            z3 = m3_z[ch]
            m3_res = MethodResult(
                method_name="M3_spread",
                z_score=z3,
                verdict=_verdict(z3, params.z_thresh),
                metadata={"spread_std": m3_spread[ch]},
            )

        # M4
        if insufficient:
            m4_res = _nan_result("M4_cusum", {"n_alarms": 0, "first_alarm_h": None})
        else:
            z4 = m4_z[ch]
            m4_res = MethodResult(
                method_name="M4_cusum",
                z_score=z4,
                verdict=_verdict(z4, params.z_thresh),
                metadata={
                    "n_alarms": m4_n_alarms[ch],
                    "first_alarm_h": m4_first_alarm_h[ch],
                },
            )

        # M5
        if insufficient or np.isnan(m5_k_corr.get(ch, np.nan)):
            m5_res = _nan_result(
                "M5_temp_k",
                {"k_corrected": np.nan, "T_mean": np.nan, "temp_correction_applied": False},
            )
        else:
            z5 = m5_z[ch]
            m5_res = MethodResult(
                method_name="M5_temp_k",
                z_score=z5,
                verdict=_verdict(z5, params.z_thresh),
                metadata={
                    "k_corrected": m5_k_corr[ch],
                    "T_mean": m5_T_mean[ch],
                    "temp_correction_applied": not np.isnan(m5_T_mean[ch]),
                },
            )

        # M6
        if insufficient or np.isnan(m6_frac_bot20.get(ch, np.nan)):
            m6_res = _nan_result(
                "M6_rank",
                {"mean_rank_pct": np.nan, "frac_bot20": np.nan, "rank_slope": np.nan},
            )
        else:
            z6 = m6_z[ch]
            m6_res = MethodResult(
                method_name="M6_rank",
                z_score=z6,
                verdict=_verdict(z6, params.z_thresh),
                metadata={
                    "mean_rank_pct": m6_mean_rank[ch],
                    "frac_bot20": m6_frac_bot20[ch],
                    "rank_slope": m6_rank_slope[ch],
                },
            )

        results[ch] = [m1_res, m2_res, m3_res, m4_res, m5_res, m6_res]

    return results
