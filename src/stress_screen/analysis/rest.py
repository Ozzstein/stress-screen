"""
analysis/rest.py — OCV-rest detection methods for stress_screen.

Scored methods: ocv_k, thermal_corr, spread, cusum, temp_k.
Report-only statistics: rank (percentile-rank trajectory; never scored).

Each method operates on a single rest segment (post-settling) of the long-format
cell voltage DataFrame and returns a MethodResult per channel.

Public API
----------
run_rest_analysis(rest_cell_df, topology, params) -> dict[int, list[MethodResult]]
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm.auto import tqdm

from stress_screen.models import MethodResult, PackTopology
from stress_screen.analysis.util import (
    arrhenius_correction,
    cusum_2sided,
    ocv_model,
    robust_z,
)
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

    arrhenius_ea_ev: float = 0.5
    """Activation energy (eV) for Arrhenius self-discharge temperature correction (LFP default)."""

    m3_min_slope_mad_v_per_h: float = 5e-6
    """Lower bound of the MAD scale (V/h) for M3 divergence-slope z-scores.

    When the fleet is healthy, empirical MAD ≈ 0 and dividing by it inflates
    any nonzero slope into a millions-scale z-score. The effective floor is
    max(this, q / T_settled), where q is the tester's voltage resolution
    detected from the rest data (1 mV on integer-millivolt loggers) and
    T_settled the settled rest length: a single quantisation step across the
    window then scores z ≈ 1 at most, and HIGH needs ≥ ~3 steps of genuine
    divergence. On loggers with fine (sub-µV) resolution this value governs;
    5 µV/h is ~0.27 mV of total divergence over a 54 h rest. The effective
    floor and q are reported in the method metadata (``mad_floor_v_per_h``,
    ``voltage_resolution_v``).
    """

    dv_dt_coeff_mv_per_c: float = -0.2
    """OCV temperature coefficient (mV/°C) used to compensate M3 and M6 for
    within-module thermal gradients before computing fleet-relative metrics.

    A cell that is warmer than its module-mates naturally has a slightly lower
    OCV (for LFP the coefficient is approximately −0.2 to −0.4 mV/°C in the
    flat plateau region). Without correction, a 5°C gradient produces ~1 mV
    of apparent divergence that M3 and M6 would misread as self-discharge.
    Set to 0.0 to disable temperature compensation (e.g., when temperature
    data is unreliable or absent).
    """

    edge_sensor_deflation: float = 1.4142135623730951
    """Deflation factor applied to the M2 thermal_corr z of single-sensor
    edge cells (G1/G8). Interior groups average two bracketing sensors (√2
    noise reduction); without this penalty the noisier edge cells produce
    structural false positives. Default √2."""


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


def _voltage_resolution(values: np.ndarray) -> float:
    """Smallest step between distinct voltage values (V) — the tester's
    quantisation step (1 mV on integer-millivolt loggers). NaN when there are
    fewer than two distinct values."""
    arr = np.asarray(values, dtype=float)
    distinct = np.unique(np.round(arr[np.isfinite(arr)], 7))
    steps = np.diff(distinct)
    steps = steps[steps > 0]
    return round(float(steps.min()), 9) if steps.size else float("nan")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_rest_analysis(
    rest_cell_df: pd.DataFrame,
    topology: PackTopology,
    params: RestParams | None = None,
) -> dict[int, list[MethodResult]]:
    """Run the rest-phase detection methods on cell data.

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
    objects (ocv_k, thermal_corr, spread, cusum, temp_k, rank). ``rank`` is
    report-only: its z-score is always NaN and it never enters a verdict.
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

    # Temperature-compensated voltage pivot for M3 and M6.
    # Subtracts the expected OCV offset due to each cell's temperature
    # deviation from the fleet median at each time step.
    # V_adj = V - (dV/dT) * (T_cell - T_fleet_median)
    # A sample without temperature gets a NaN V_adj and is thereby excluded
    # from the fleet median and from this cell's M3/M6 fit. Whole modules
    # without temperature are rejected upstream (cli), so this only covers
    # sporadic sensor gaps.
    has_temp_col = "temperature" in settled_df.columns
    if has_temp_col and params.dv_dt_coeff_mv_per_c != 0.0:
        pivot_t: pd.DataFrame = settled_df.pivot_table(
            index="_t_rel", columns="channel_index", values="temperature"
        )
        # Reindex to match pivot_v (same time axis)
        pivot_t = pivot_t.reindex(pivot_v.index)
        T_fleet_med = pivot_t.median(axis=1)
        T_excess = pivot_t.sub(T_fleet_med, axis=0)  # deviation from fleet median
        dv_dt_v = params.dv_dt_coeff_mv_per_c / 1000.0  # convert mV/°C → V/°C
        pivot_v_adj: pd.DataFrame = pivot_v - dv_dt_v * T_excess
    else:
        pivot_v_adj = pivot_v

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

    # Edge-cell noise penalty:
    # G1 and G8 of each module are covered by a single un-averaged temperature
    # sensor, whereas G2-G7 are the mean of two sensors (√2 noise reduction).
    # Without this correction the higher-noise edge cells systematically yield
    # inflated |r| and z, producing structural false positives.  We deflate
    # the z by 1/√n_sensors so cells with fewer sensors face a higher
    # effective threshold.
    for ch in channels:
        module_id = topology.module_for_channel(ch)
        group_idx = topology.group_index_in_module(ch)
        n_sensors = len(topology._temp_sensor_map.get((module_id, group_idx), []))
        if n_sensors == 1 and not np.isnan(m2_z[ch]):
            m2_z[ch] = m2_z[ch] / params.edge_sensor_deflation

    # ------------------------------------------------------------------ #
    # M3 — Slope of |V_cell − V_fleet_median| (divergence rate)            #
    # ------------------------------------------------------------------ #
    m3_spread: dict[int, float] = {}
    # Noise floor for the slope z-scores: at least one quantisation step of
    # the tester's voltage resolution across the settled window, so a single
    # rounding step cannot reach HIGH (see RestParams.m3_min_slope_mad_v_per_h).
    m3_resolution_v = float("nan")
    m3_floor = params.m3_min_slope_mad_v_per_h

    if not pivot_v_adj.empty:
        fleet_med_adj = pivot_v_adj.median(axis=1)
        t_index = pivot_v_adj.index.values  # _t_rel in hours
        # Measure on the raw samples: the pivot averages duplicated
        # timestamps, which would halve the apparent resolution.
        m3_resolution_v = _voltage_resolution(settled_df["voltage"].to_numpy())
        t_span = float(t_index[-1] - t_index[0]) if len(t_index) > 1 else 0.0
        if np.isfinite(m3_resolution_v) and t_span > 0:
            m3_floor = max(m3_floor, m3_resolution_v / t_span)
        for ch in channels:
            if ch in pivot_v_adj.columns and chan_data[ch]["n_set"] >= params.min_points:
                deviation_abs = np.abs((pivot_v_adj[ch] - fleet_med_adj).values)
                valid = ~np.isnan(deviation_abs)
                if valid.sum() >= 5:
                    slope_dev, *_ = _stats.linregress(t_index[valid], deviation_abs[valid])
                    # Sign matters: positive slope = diverging away from fleet.
                    # Use max(0, slope) so converging cells aren't flagged.
                    m3_spread[ch] = float(max(0.0, slope_dev))
                else:
                    m3_spread[ch] = np.nan
            else:
                m3_spread[ch] = np.nan
    else:
        for ch in channels:
            m3_spread[ch] = np.nan

    spread_arr = np.array([m3_spread[ch] for ch in channels])
    spread_z_arr = robust_z(spread_arr, min_mad=m3_floor)
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

        correction = arrhenius_correction(T_celsius=T_mean, ea_ev=params.arrhenius_ea_ev)
        m5_k_corr[ch] = k_raw * correction

    k_corr_arr = np.array([m5_k_corr[ch] for ch in channels])
    k_corr_z_arr = robust_z(k_corr_arr)
    m5_z: dict[int, float] = {ch: float(k_corr_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # M6 — Percentile rank statistics (report-only, never scored)          #
    # ------------------------------------------------------------------ #
    # Percentile rank has no resolution on tester data: voltages are 1 mV
    # quantised and a healthy fleet sits within a few mV, so ranks are
    # dominated by ties, and the fleet median moves through the bulk as the
    # slowest-relaxing module settles, reshuffling healthy cells by tens of
    # percentile points. The bottom-20 % share flags static SOC ordering,
    # not drift. The statistics are kept for the rank chart and the per-cell
    # table; the method's z-score is NaN so it never enters a verdict
    # (see aggregate.ADVISORY_METHODS).
    m6_mean_rank: dict[int, float] = {}
    m6_frac_bot20: dict[int, float] = {}
    m6_rank_slope: dict[int, float] = {}

    rank_pct = (
        pivot_v_adj.rank(axis=1, pct=True, method="average") * 100.0
        if not pivot_v_adj.empty else pd.DataFrame()
    )
    for ch in channels:
        if ch in rank_pct.columns and chan_data[ch]["n_set"] >= params.min_points:
            rs = rank_pct[ch].dropna()
            t_r = rs.index.values
            r_r = rs.values
            m6_mean_rank[ch] = float(r_r.mean())
            m6_frac_bot20[ch] = float((r_r < 20.0).mean())
            if len(t_r) >= 2:
                slope_r, *_ = _stats.linregress(t_r, r_r)
                m6_rank_slope[ch] = float(slope_r)
            else:
                m6_rank_slope[ch] = 0.0
        else:
            m6_mean_rank[ch] = np.nan
            m6_frac_bot20[ch] = np.nan
            m6_rank_slope[ch] = np.nan

    # ------------------------------------------------------------------ #
    # Assemble results                                                      #
    # ------------------------------------------------------------------ #
    results: dict[int, list[MethodResult]] = {}

    for ch in channels:
        d = chan_data[ch]
        insufficient = d["n_set"] < params.min_points

        # M1
        if insufficient or np.isnan(m1_k.get(ch, np.nan)):
            m1_res = _nan_result("ocv_k", {"k": np.nan, "V_ocv": np.nan, "tau": np.nan})
        else:
            z1 = m1_z[ch]
            m1_res = MethodResult(
                method_name="ocv_k",
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
            m2_res = _nan_result("thermal_corr", {"pearson_r": np.nan})
        else:
            z2 = m2_z[ch]
            m2_res = MethodResult(
                method_name="thermal_corr",
                z_score=z2,
                verdict=_verdict(z2, params.z_thresh),
                metadata={"pearson_r": m2_r[ch]},
            )

        # M3
        m3_meta = {
            "voltage_resolution_v": m3_resolution_v,
            "mad_floor_v_per_h": m3_floor,
        }
        if insufficient or np.isnan(m3_spread.get(ch, np.nan)):
            m3_res = _nan_result("spread", {"divergence_slope_v_per_h": np.nan, **m3_meta})
        else:
            z3 = m3_z[ch]
            m3_res = MethodResult(
                method_name="spread",
                z_score=z3,
                verdict=_verdict(z3, params.z_thresh),
                metadata={"divergence_slope_v_per_h": m3_spread[ch], **m3_meta},
            )

        # M4
        if insufficient:
            m4_res = _nan_result("cusum", {"n_alarms": 0, "first_alarm_h": None})
        else:
            z4 = m4_z[ch]
            m4_res = MethodResult(
                method_name="cusum",
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
                "temp_k",
                {"k_corrected": np.nan, "T_mean": np.nan, "temp_correction_applied": False},
            )
        else:
            z5 = m5_z[ch]
            m5_res = MethodResult(
                method_name="temp_k",
                z_score=z5,
                verdict=_verdict(z5, params.z_thresh),
                metadata={
                    "k_corrected": m5_k_corr[ch],
                    "T_mean": m5_T_mean[ch],
                    "temp_correction_applied": not np.isnan(m5_T_mean[ch]),
                },
            )

        # M6 — report-only statistics, never scored (see the M6 block).
        m6_res = _nan_result(
            "rank",
            {
                "mean_rank_pct": m6_mean_rank.get(ch, np.nan),
                "frac_bot20": m6_frac_bot20.get(ch, np.nan),
                "rank_slope": m6_rank_slope.get(ch, np.nan),
            },
        )

        results[ch] = [m1_res, m2_res, m3_res, m4_res, m5_res, m6_res]

    return results
