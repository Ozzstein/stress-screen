"""
analysis/short_circuit.py — Soft/incipient internal short-circuit (ISC) detection.

Three methods:
  S1: Excess self-discharge rate (k well above fleet median+MAD threshold)
  S2: Thermal anomaly during rest (positive dT/dt vs cooling peers)
  S3: Charge-acceptance shape deficit (reduced dV/dQ area in Q domain)

Public API
----------
run_isc_analysis(rest_cell_df, rest_results, charge_cell_df, params,
                 top_charge_df, n_parallel) -> dict[int, MethodResult]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.integrate import cumulative_trapezoid, trapezoid

from stress_screen.models import MethodResult
from stress_screen.analysis.util import robust_z


@dataclass
class ShortCircuitParams:
    """Tunable parameters for the ISC detection analysis."""

    z_thresh: float = 2.0
    isc_k_sigma: float = 3.0
    settling_h: float = 2.0
    min_points: int = 60
    n_q_resample: int = 500
    """Number of uniformly-spaced Q points for S3 dV/dQ resampling.
    Pre-resampling replaces post-gradient Savitzky-Golay smoothing."""
    peak_prominence_pct: float = 0.05


def _verdict(z: float, z_thresh: float) -> str:
    if np.isnan(z):
        return "NORMAL"
    if z >= z_thresh:
        return "HIGH"
    if z >= 1.0:
        return "ELEVATED"
    return "NORMAL"


def run_isc_analysis(
    rest_cell_df: pd.DataFrame,
    rest_results: dict[int, list[MethodResult]],
    charge_cell_df: pd.DataFrame,
    params: ShortCircuitParams | None = None,
    top_charge_df: pd.DataFrame | None = None,
    n_parallel: int = 1,
) -> dict[int, MethodResult]:
    """Detect soft/incipient internal short-circuit (ISC) signatures.

    Parameters
    ----------
    rest_cell_df:
        Long-format rest-phase data with columns ``time_hours``, ``channel_index``,
        ``voltage``, ``temperature``.
    rest_results:
        Output of ``run_rest_analysis``. M1 (index 0) metadata must contain ``"k"``.
    charge_cell_df:
        Long-format charge-phase data (same schema). Used for S3.
    params:
        Tunable parameters; defaults used if None.
    top_charge_df:
        Pack-level charge DataFrame with ``time_hours`` and ``current``. S3 returns
        nan for all channels when absent.
    n_parallel:
        Parallel cell count per group (topology.parallel). Scales the Q axis.

    Returns
    -------
    dict mapping ``channel_index`` to a ``MethodResult`` with ``method_name="isc"``.
    """
    if params is None:
        params = ShortCircuitParams()

    channels = sorted(rest_cell_df["channel_index"].unique())
    t0 = rest_cell_df["time_hours"].min()

    # Pre-process: settled per-channel data
    chan_data: dict[int, dict] = {}
    for ch in channels:
        cd = (
            rest_cell_df[rest_cell_df["channel_index"] == ch]
            .sort_values("time_hours")
            .copy()
        )
        cd["_t_rel"] = cd["time_hours"] - t0
        settled = cd[cd["_t_rel"] >= params.settling_h]
        has_temp = "temperature" in settled.columns
        chan_data[ch] = {
            "t_set": settled["_t_rel"].values,
            "temp_set": settled["temperature"].values if has_temp else np.array([]),
            "n_set": len(settled),
        }

    # S1: Excess self-discharge rate
    m1_k: dict[int, float] = {}
    for ch in channels:
        if ch in rest_results and rest_results[ch]:
            m1_k[ch] = float(rest_results[ch][0].metadata.get("k", np.nan))
        else:
            m1_k[ch] = np.nan

    k_arr = np.array([m1_k.get(ch, np.nan) for ch in channels])
    valid_k = k_arr[~np.isnan(k_arr)]

    s1_excess: dict[int, float] = {}
    if len(valid_k) >= 3:
        med_k = float(np.median(valid_k))
        mad_k = float(np.median(np.abs(valid_k - med_k)))
        threshold = med_k + params.isc_k_sigma * mad_k
        for i, ch in enumerate(channels):
            s1_excess[ch] = (
                max(0.0, float(k_arr[i]) - threshold)
                if not np.isnan(k_arr[i])
                else np.nan
            )
    else:
        for ch in channels:
            s1_excess[ch] = np.nan

    s1_arr = np.array([s1_excess[ch] for ch in channels])
    s1_z_arr = robust_z(s1_arr)
    s1_z: dict[int, float] = {ch: float(s1_z_arr[i]) for i, ch in enumerate(channels)}

    # S2: Thermal anomaly during rest (positive dT/dt slope)
    s2_slope: dict[int, float] = {}
    for ch in channels:
        d = chan_data[ch]
        if d["n_set"] < params.min_points or len(d["temp_set"]) == 0:
            s2_slope[ch] = np.nan
            continue
        temp = d["temp_set"]
        t = d["t_set"]
        valid = ~np.isnan(temp)
        if valid.sum() < 5 or np.nanstd(temp[valid]) < 0.01:
            s2_slope[ch] = np.nan
            continue
        slope, *_ = _stats.linregress(t[valid], temp[valid])
        s2_slope[ch] = float(slope)

    s2_arr = np.array([s2_slope[ch] for ch in channels])
    s2_z_arr = robust_z(s2_arr)
    s2_z: dict[int, float] = {ch: float(s2_z_arr[i]) for i, ch in enumerate(channels)}

    # S3: Charge-acceptance shape (dV/dQ area deficit)
    s3_area: dict[int, float] = {}

    if top_charge_df is not None and len(top_charge_df) >= 2 and not charge_cell_df.empty:
        _top = top_charge_df.sort_values("time_hours")
        t_pack = _top["time_hours"].values
        i_pack = np.abs(_top["current"].values)
        Q_pack = np.concatenate([
            [0.0], cumulative_trapezoid(i_pack, t_pack)
        ]) / max(n_parallel, 1)

        for ch in channels:
            ch_charge = (
                charge_cell_df[charge_cell_df["channel_index"] == ch]
                .sort_values("time_hours")
            )
            if len(ch_charge) < params.min_points:
                s3_area[ch] = np.nan
                continue
            q_ch = np.interp(ch_charge["time_hours"].values, t_pack, Q_pack)
            voltage = ch_charge["voltage"].values
            # Keep strictly increasing Q for interpolation
            mono = np.concatenate([[True], np.diff(q_ch) > 0])
            if mono.sum() < 3:
                s3_area[ch] = np.nan
                continue
            q_mono = q_ch[mono]
            v_mono = voltage[mono]
            # Resample at uniform ΔQ (implicit smoothing — no post-gradient filter)
            q_grid = np.linspace(q_mono[0], q_mono[-1], params.n_q_resample)
            v_interp = np.interp(q_grid, q_mono, v_mono)
            dv_dq = np.gradient(v_interp, q_grid)
            s3_area[ch] = float(trapezoid(np.abs(dv_dq), q_grid))
    else:
        for ch in channels:
            s3_area[ch] = np.nan

    # Invert: low area = high z (area deficit is the suspicious direction)
    s3_arr = np.array([s3_area[ch] for ch in channels])
    s3_z_arr = -robust_z(s3_arr)
    s3_z: dict[int, float] = {ch: float(s3_z_arr[i]) for i, ch in enumerate(channels)}

    # Assemble MethodResult per channel
    results: dict[int, MethodResult] = {}
    for i, ch in enumerate(channels):
        z1, z2, z3 = s1_z[ch], s2_z[ch], s3_z[ch]
        valid_zs = [s for s in (z1, z2, z3) if not np.isnan(s)]
        z = float(np.mean(valid_zs)) if valid_zs else float("nan")
        results[ch] = MethodResult(
            method_name="isc",
            z_score=z,
            verdict=_verdict(z, params.z_thresh),
            metadata={
                "s1_excess_k_z": z1,
                "s2_dT_dt_z": z2,
                "s3_area_deficit_z": z3,
                "s1_excess_k": float(s1_excess[ch]),
                "s2_dT_dt_slope": float(s2_slope[ch]),
                "s3_dvdq_area": float(s3_area[ch]),
            },
        )
    return results
