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

from stress_screen.models import MethodResult, PackTopology
from stress_screen.analysis.util import arrhenius_correction, robust_z


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

    isc_ea_ev: float = 0.1
    """Activation energy (eV) for ISC-specific temperature correction.
    Lower than the 0.5 eV used for benign SEI self-discharge because
    metallic-Li bridging is largely electronic (T-insensitive)."""


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
    topology: PackTopology | None = None,
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

    # S1: Excess self-discharge rate (uses ISC-Ea-corrected k, not raw)
    m1_k_corr: dict[int, float] = {}
    for ch in channels:
        if ch in rest_results and rest_results[ch]:
            k_raw = float(rest_results[ch][0].metadata.get("k", np.nan))
            # Prefer M5 metadata when present (T_mean is computed there)
            T_mean_ch = np.nan
            m5_mr = next(
                (mr for mr in rest_results[ch] if mr.method_name == "temp_k"),
                None,
            )
            if m5_mr is not None:
                T_mean_ch = float(m5_mr.metadata.get("T_mean", np.nan))
            if np.isnan(T_mean_ch):
                # Fall back to rest-segment temperature
                ch_rest = rest_cell_df[rest_cell_df["channel_index"] == ch]
                if "temperature" in ch_rest.columns and len(ch_rest) > 0:
                    T_mean_ch = float(np.nanmean(ch_rest["temperature"].values))
            correction = arrhenius_correction(T_celsius=T_mean_ch, ea_ev=params.isc_ea_ev)
            m1_k_corr[ch] = k_raw * correction if not np.isnan(k_raw) else np.nan
        else:
            m1_k_corr[ch] = np.nan

    k_arr = np.array([m1_k_corr[ch] for ch in channels])
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
    s1_z_arr = robust_z(s1_arr, min_mad=1e-6)
    s1_z: dict[int, float] = {ch: float(s1_z_arr[i]) for i, ch in enumerate(channels)}

    # S2: Thermal anomaly during rest — excess dT/dt above module-median baseline.
    # Subtracting the module-median slope removes common-mode ambient drift that
    # would otherwise make every cell in a warm module look like an ISC candidate.
    s2_raw_slope: dict[int, float] = {}
    for ch in channels:
        d = chan_data[ch]
        if d["n_set"] < params.min_points or len(d["temp_set"]) == 0:
            s2_raw_slope[ch] = np.nan
            continue
        temp = d["temp_set"]
        t = d["t_set"]
        valid = ~np.isnan(temp)
        if valid.sum() < 5 or np.nanstd(temp[valid]) < 0.01:
            s2_raw_slope[ch] = np.nan
            continue
        slope, *_ = _stats.linregress(t[valid], temp[valid])
        s2_raw_slope[ch] = float(slope)

    # Subtract module-median slope when topology is available.
    # Without topology, fall back to using raw slopes (same as before).
    s2_slope: dict[int, float] = {}
    if topology is not None:
        for ch in channels:
            mid = topology.module_for_channel(ch)
            mod_ch = topology.channels_in_module(mid)
            mod_slopes = [s2_raw_slope[c] for c in mod_ch
                          if c in s2_raw_slope and not np.isnan(s2_raw_slope.get(c, np.nan))]
            mod_median = float(np.nanmedian(mod_slopes)) if mod_slopes else 0.0
            raw = s2_raw_slope.get(ch, np.nan)
            s2_slope[ch] = float(raw - mod_median) if not np.isnan(raw) else np.nan
    else:
        s2_slope = dict(s2_raw_slope)

    s2_arr = np.array([s2_slope[ch] for ch in channels])
    s2_z_arr = robust_z(s2_arr, min_mad=1e-4)
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
    s3_z_arr = -robust_z(s3_arr, min_mad=1e-3)
    s3_z: dict[int, float] = {ch: float(s3_z_arr[i]) for i, ch in enumerate(channels)}

    # S3 temperature gate: cold cells naturally show reduced charge acceptance
    # (higher impedance → smaller dV/dQ area). Apply a mild Arrhenius gate
    # (Ea = isc_ea_ev ≈ 0.1 eV) using mean charge-phase temperature.
    # Gate < 1 at cold T scales down S3 z; gate > 1 at warm T slightly raises it.
    s3_T_mean: dict[int, float] = {}
    if not charge_cell_df.empty and "temperature" in charge_cell_df.columns:
        for ch in channels:
            ch_t = charge_cell_df[charge_cell_df["channel_index"] == ch]["temperature"].values.astype(float)
            valid_t = ch_t[~np.isnan(ch_t)]
            s3_T_mean[ch] = float(np.nanmean(valid_t)) if len(valid_t) > 0 else np.nan
    else:
        for ch in channels:
            s3_T_mean[ch] = np.nan

    s3_gate: dict[int, float] = {}
    for ch in channels:
        T = s3_T_mean.get(ch, np.nan)
        if np.isnan(T):
            s3_gate[ch] = 1.0  # no data → no correction
        else:
            # arrhenius_correction normalises a rate at T back to T_ref.
            # For cold T it returns > 1 (rate is slower at cold, so multiply up).
            # For the S3 gate we want the inverse: cold T → gate < 1 (suppress z),
            # warm T → gate > 1 (amplify z).
            corr = arrhenius_correction(T_celsius=T, ea_ev=params.isc_ea_ev)
            g = 1.0 / corr if corr > 0 else 1.0
            s3_gate[ch] = min(g, 1.5)  # cap warm-T amplification at 1.5
        z = s3_z.get(ch, np.nan)
        s3_z[ch] = float(z * s3_gate[ch]) if not np.isnan(z) else np.nan

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
                "s1_k_corrected_isc": float(m1_k_corr[ch]),
                "s2_dT_dt_slope": float(s2_slope[ch]),
                "s2_dT_dt_raw_slope": float(s2_raw_slope[ch]),
                "s3_dvdq_area": float(s3_area[ch]),
                "s3_temperature_gate": float(s3_gate.get(ch, 1.0)),
            },
        )
    return results
