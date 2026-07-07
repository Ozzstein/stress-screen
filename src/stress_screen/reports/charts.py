"""
reports/charts.py — Shared Plotly figure builders for stress_screen.

Pure functions: take analysis result objects and return plotly Figure objects.
No I/O; no .show() calls. Used by both html.py (interactive) and pdf.py (static PNG).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stress_screen.models import AnalysisResult, MethodResult, Segment
from stress_screen.reports.findings import fmt_num, k_to_mv_per_h

#: Rest settling transient discarded by the OCV fit (RestParams default);
#: fitted-model overlays are drawn from here onward.
_SETTLING_H = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _downsample_df(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """Return a uniformly-thinned copy of *df* capped at *max_points* rows."""
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step]


def _rolling_median(y: np.ndarray, t_hours: np.ndarray,
                    window_h: float = 0.5) -> np.ndarray:
    """Display-only rolling-median smoothing (~*window_h* hours wide).

    The tester logs integer-millivolt values, so raw traces are dominated by
    quantization staircases. Detection always runs on raw data; this
    smoothing exists purely so the charts show the trend the detectors see.
    """
    n = len(y)
    if n < 5:
        return y
    dt = np.nanmedian(np.diff(t_hours)) if n > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        window = 5
    else:
        window = max(5, int(round(window_h / dt)))
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    return (
        pd.Series(y).rolling(window, center=True, min_periods=1).median().values
    )


def _reconstruct_ocv_amplitude(
    t: np.ndarray, v: np.ndarray, V_ocv: float, tau: float, k: float
) -> float:
    """Conditional least-squares estimate of the relaxation amplitude ``a``.

    The analysis stores V_ocv, tau, k but not ``a`` from the fit
    V(t) = V_ocv + a·exp(−t/τ) − k·t. Given the other three parameters the
    optimal amplitude has the closed form a* = Σe·(v − V_ocv + k·t) / Σe²
    with e = exp(−t/τ) — exact, deterministic, display-side only.
    """
    if not (np.isfinite(V_ocv) and np.isfinite(tau) and np.isfinite(k)) or tau <= 0:
        return float("nan")
    e = np.exp(-t / tau)
    valid = np.isfinite(v) & np.isfinite(e)
    den = float(np.sum(e[valid] ** 2))
    if den <= 0:
        return float("nan")
    return float(np.sum(e[valid] * (v[valid] - V_ocv + k * t[valid])) / den)


def _cell_lookup(result: AnalysisResult, module_id: int):
    """(verdict_map, label_map, method_map) for one module's channels."""
    mv = next((m for m in result.module_verdicts if m.module_id == module_id), None)
    verdict_map: dict[int, str] = {}
    label_map: dict[int, str] = {}
    method_map: dict[int, dict[str, MethodResult]] = {}
    if mv is not None:
        for cv in mv.all_cells:
            verdict_map[cv.channel_index] = cv.verdict
            label_map[cv.channel_index] = cv.label
            method_map[cv.channel_index] = {
                mr.method_name: mr for mr in cv.method_results
            }
    return verdict_map, label_map, method_map


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    "HIGH": "red",
    "ELEVATED": "orange",
    "NORMAL": "gray",
}


def _trace_style(verdict: str) -> dict[str, Any]:
    """Line style + legend policy for a cell trace by its verdict.

    HIGH and ELEVATED cells are highlighted and legend-named so a specific
    flagged trace is identifiable in both HTML and static PDF renders.
    """
    if verdict == "HIGH":
        return dict(line=dict(color="red", width=2.5), opacity=1.0,
                    showlegend=True)
    if verdict == "ELEVATED":
        return dict(line=dict(color="orange", width=2.0), opacity=1.0,
                    showlegend=True)
    return dict(line=dict(color="lightgray", width=1), opacity=0.5,
                showlegend=False)


def _add_color_key(fig: go.Figure, n_normal: int) -> None:
    """Key the HIGH/ELEVATED/normal color convention in the legend."""
    if n_normal > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="lightgray", width=1),
            name=f"normal cells (n={n_normal})",
            showlegend=True, hoverinfo="skip",
        ))


# Discrete color bands anchored to the verdict gates (composite_z):
#   [0, 0.5) deep green   — well below every gate
#   [0.5, 1) pale green   — ELEVATED floor region
#   [1, 2)   amber        — ELEVATED outright gate region
#   [2, 3)   orange-red   — above HIGH gate
#   [3, 4]   red          — far above (display clamps at 4)
_HEATMAP_CLAMP = 4.0
_HEATMAP_COLORSCALE = [
    [0.000, "rgb(0,150,60)"], [0.125, "rgb(0,150,60)"],
    [0.125, "rgb(150,205,120)"], [0.250, "rgb(150,205,120)"],
    [0.250, "rgb(255,200,60)"], [0.500, "rgb(255,200,60)"],
    [0.500, "rgb(240,90,40)"], [0.750, "rgb(240,90,40)"],
    [0.750, "rgb(190,0,0)"], [1.000, "rgb(190,0,0)"],
]


# ---------------------------------------------------------------------------
# 1. pack_heatmap
# ---------------------------------------------------------------------------

def pack_heatmap(result: AnalysisResult) -> go.Figure:
    """Heatmap of composite_z per cell-group, organised as modules × groups.

    Rows = modules, M1 at the TOP (reading order). Columns = cell-groups.
    Color bands are anchored to the verdict gates (see _HEATMAP_COLORSCALE);
    each cell shows its numeric composite_z, the verdict lives in the hover,
    and flagged cells are marked with an open-square outline.
    """
    topo = result.topology
    n_modules = topo.module_count
    n_groups = topo.series  # cell-groups per module

    z_matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []
    verdict_matrix: list[list[str]] = []
    y_labels: list[str] = []
    flagged_points: list[tuple[str, str]] = []  # (x_label, y_label)

    for mid in range(1, n_modules + 1):  # M1 first → top row
        mv = next((m for m in result.module_verdicts if m.module_id == mid), None)
        verdict_label = mv.verdict if mv is not None else "OK"
        y_label = f"M{mid} {verdict_label}"
        y_labels.append(y_label)
        row_z: list[float] = []
        row_txt: list[str] = []
        row_verdict: list[str] = []
        for gidx in range(1, n_groups + 1):
            cv = None
            if mv is not None:
                cv = next((c for c in mv.all_cells if c.group_in_module == gidx), None)
            if cv is not None:
                row_z.append(float(np.clip(cv.composite_z, 0.0, _HEATMAP_CLAMP)))
                row_txt.append(f"{cv.composite_z:.2f}")
                row_verdict.append(cv.verdict)
                if cv.verdict in ("HIGH", "ELEVATED"):
                    flagged_points.append((f"G{gidx}", y_label))
            else:
                row_z.append(0.0)
                row_txt.append("")
                row_verdict.append("")
        z_matrix.append(row_z)
        text_matrix.append(row_txt)
        verdict_matrix.append(row_verdict)

    x_labels = [f"G{g}" for g in range(1, n_groups + 1)]

    heatmap = go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        zmin=0.0,
        zmax=_HEATMAP_CLAMP,
        colorscale=_HEATMAP_COLORSCALE,
        colorbar=dict(
            title="Composite Z",
            tickvals=[0.25, 0.75, 1.5, 2.5, 3.5],
            ticktext=["<0.5", "0.5–1", "1–2 ELEV", "2–3 HIGH", "≥3"],
        ),
        text=text_matrix,
        customdata=verdict_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate=("Module: %{y}<br>Group: %{x}<br>"
                       "Composite z: %{text}<br>Verdict: %{customdata}"
                       "<extra></extra>"),
    )

    fig = go.Figure(data=[heatmap])

    # Outline flagged cells so they pop even in a static PDF render
    if flagged_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in flagged_points],
            y=[p[1] for p in flagged_points],
            mode="markers",
            marker=dict(symbol="square-open", size=26,
                        line=dict(color="black", width=2.5)),
            name="flagged cell",
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title="Pack Overview — Composite Z-Score Heatmap "
              "(flagged cells outlined)",
        xaxis_title="Cell Group",
        yaxis_title="Module",
        yaxis=dict(autorange="reversed"),  # M1 at top
        template="plotly_white",
        height=max(300, 60 * n_modules + 120),
    )

    return fig


# ---------------------------------------------------------------------------
# 2. ocv_fit_overlay
# ---------------------------------------------------------------------------

def ocv_fit_overlay(
    result: AnalysisResult,
    module_id: int,
    rest_cell_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """OCV voltage curves during rest for all cell-groups in a module.

    Flagged cells (HIGH red / ELEVATED orange) are highlighted and named in
    the legend; for each flagged cell the fitted decay model
    V(t) = V_ocv + a·exp(−t/τ) − k·t is drawn dashed over the settled window
    with the fitted k (mV/h) and τ in the legend name.

    Parameters
    ----------
    result:
        Full analysis result.
    module_id:
        1-based module identifier.
    rest_cell_df:
        Long-format cell DataFrame restricted to the rest phase.
        Columns: time_hours, channel_index, voltage.
        If None an empty figure with an annotation is returned.
    """
    fig = go.Figure()
    fig.update_layout(
        title=f"Module M{module_id} — OCV Curves During Rest",
        xaxis_title="Time from rest start (h)",
        yaxis_title="Voltage (V)",
        template="plotly_white",
    )

    if rest_cell_df is None or rest_cell_df.empty:
        fig.add_annotation(text="No rest data provided", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    topo = result.topology
    channels = sorted(topo.channels_in_module(module_id))
    verdict_map, label_map, method_map = _cell_lookup(result, module_id)

    t_min = rest_cell_df["time_hours"].min()
    n_normal = 0
    fit_traces: list[go.Scatter] = []

    for ch in channels:
        ch_df = rest_cell_df[rest_cell_df["channel_index"] == ch].sort_values("time_hours")
        if ch_df.empty:
            continue
        ch_df = _downsample_df(ch_df)
        t_rel = (ch_df["time_hours"] - t_min).values
        v = ch_df["voltage"].values

        verdict = verdict_map.get(ch, "NORMAL")
        lbl = label_map.get(ch, f"Ch{ch}")
        style = _trace_style(verdict)
        if verdict == "NORMAL":
            n_normal += 1

        fig.add_trace(go.Scatter(
            x=t_rel, y=v, mode="lines", name=lbl, **style,
        ))

        # Fitted decay model overlay for flagged cells
        if verdict in ("HIGH", "ELEVATED"):
            ocv_meta = method_map.get(ch, {}).get("ocv_k")
            if ocv_meta is not None:
                k = ocv_meta.metadata.get("k", float("nan"))
                v_ocv = ocv_meta.metadata.get("V_ocv", float("nan"))
                tau = ocv_meta.metadata.get("tau", float("nan"))
                settled = t_rel >= _SETTLING_H
                if (np.isfinite(k) and np.isfinite(v_ocv) and np.isfinite(tau)
                        and settled.sum() >= 5):
                    a = _reconstruct_ocv_amplitude(
                        t_rel[settled], v[settled], v_ocv, tau, k)
                    if np.isfinite(a):
                        t_fit = np.linspace(_SETTLING_H, float(t_rel.max()), 200)
                        v_fit = v_ocv + a * np.exp(-t_fit / tau) - k * t_fit
                        fit_traces.append(go.Scatter(
                            x=t_fit, y=v_fit, mode="lines",
                            name=(f"{lbl} fit — k = {k_to_mv_per_h(k)}, "
                                  f"τ = {fmt_num(tau, 2, 'h')}"),
                            line=dict(color="black", width=1.5, dash="dash"),
                            showlegend=True,
                        ))

    for tr in fit_traces:  # drawn last so they sit on top
        fig.add_trace(tr)
    _add_color_key(fig, n_normal)

    return fig


# ---------------------------------------------------------------------------
# 3. dv_dq_chart
# ---------------------------------------------------------------------------

def _build_q_axis(
    top_charge_df: Optional[pd.DataFrame],
    n_parallel: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build cumulative charge Q axis from pack-level charge data.

    Returns (q_pack_time, q_pack_cumul) arrays, or (None, None) when
    top_charge_df is None / too short.
    """
    if top_charge_df is None or len(top_charge_df) < 2:
        return None, None
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
    _top = top_charge_df.sort_values("time_hours")
    t = _top["time_hours"].values
    q = np.concatenate([
        [0.0],
        _cumtrapz(np.abs(_top["current"].values), t),
    ]) / max(n_parallel, 1)
    return t, q


def _compute_dqdv(
    voltage: np.ndarray,
    time_h: np.ndarray,
    q_pack_time: Optional[np.ndarray],
    q_pack_cumul: Optional[np.ndarray],
    dv_step: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dQ/dV via voltage-domain resampling at *dv_step* V intervals.

    Resamples Q(V) onto a uniform voltage grid, applies Savitzky-Golay
    smoothing on Q before differentiating — ICA best practice. Smoothing
    Q rather than dQ/dV preserves peak positions while removing measurement
    noise. Returns empty arrays when Q data is unavailable.
    """
    if q_pack_time is None or q_pack_cumul is None:
        return np.array([]), np.array([])

    q_ch = np.interp(time_h, q_pack_time, q_pack_cumul)

    # Sort by voltage, remove duplicate V values
    idx = np.argsort(voltage)
    v_s = voltage[idx]
    q_s = q_ch[idx]
    mono = np.concatenate([[True], np.diff(v_s) > 0])
    v_s, q_s = v_s[mono], q_s[mono]

    if len(v_s) < 5:
        return np.array([]), np.array([])

    v_grid = np.arange(v_s[0], v_s[-1] + dv_step, dv_step)
    if len(v_grid) < 5:
        return np.array([]), np.array([])

    q_interp = np.interp(v_grid, v_s, q_s)

    # Smooth Q(V) before differentiation: odd window ≤21 points (≤ 42 mV)
    from scipy.signal import savgol_filter as _sg
    wl = min(21, len(q_interp))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 5)
    q_interp = _sg(q_interp, window_length=wl, polyorder=3)

    dqdv = np.gradient(q_interp, v_grid)
    dqdv = np.clip(dqdv, 0.0, None)  # non-physical negative values removed
    return v_grid, dqdv


def dv_dq_chart(
    result: AnalysisResult,
    module_id: int,
    charge_cell_df: Optional[pd.DataFrame] = None,
    top_charge_df: Optional[pd.DataFrame] = None,
    n_parallel: int = 1,
) -> go.Figure:
    """dQ/dV (Incremental Capacity Analysis) curves for each cell-group in a module during charge.

    HIGH cells: red, thick line.  Others: gray, thin, semi-transparent.
    Requires top_charge_df to build the Q axis; returns an annotated empty
    figure when Q data is absent.

    Parameters
    ----------
    result:
        Full analysis result.
    module_id:
        1-based module identifier.
    charge_cell_df:
        Long-format cell DataFrame restricted to the charge phase.
        Columns: time_hours, channel_index, voltage.
        If None an empty figure with an annotation is returned.
    top_charge_df:
        Pack-level DataFrame for the charge segment (time_hours, current).
        Used to build Q axis; optional.
    n_parallel:
        Number of parallel strings — divides pack current to per-string Q.
    """
    q_pack_time, q_pack_cumul = _build_q_axis(top_charge_df, n_parallel)

    fig = go.Figure()
    fig.update_layout(
        title=f"Module M{module_id} — dQ/dV (Incremental Capacity)",
        xaxis_title="Voltage (V)",
        yaxis_title="dQ/dV (Ah/V)",
        template="plotly_white",
    )

    if charge_cell_df is None or charge_cell_df.empty:
        fig.add_annotation(text="No charge data provided", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    if q_pack_time is None:
        fig.add_annotation(
            text="Q data required for dQ/dV — pass top_charge_df",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    topo = result.topology
    channels = sorted(topo.channels_in_module(module_id))
    verdict_map, label_map, method_map = _cell_lookup(result, module_id)

    n_normal = 0
    for ch in channels:
        ch_df = charge_cell_df[charge_cell_df["channel_index"] == ch].sort_values("time_hours")
        if len(ch_df) < 5:
            continue

        v_grid, dqdv = _compute_dqdv(
            ch_df["voltage"].values,
            ch_df["time_hours"].values,
            q_pack_time,
            q_pack_cumul,
        )
        if len(v_grid) == 0:
            continue

        verdict = verdict_map.get(ch, "NORMAL")
        lbl = label_map.get(ch, f"Ch{ch}")
        style = _trace_style(verdict)
        if verdict == "NORMAL":
            n_normal += 1

        fig.add_trace(go.Scatter(
            x=v_grid, y=dqdv, mode="lines", name=lbl, **style,
        ))

        # Mark the detected extra peak (li-plating signature) for flagged cells
        if verdict in ("HIGH", "ELEVATED"):
            lp = method_map.get(ch, {}).get("li_plating")
            if lp is not None:
                peak_v = lp.metadata.get("dqdv_extra_peak_voltage", float("nan"))
                if np.isfinite(peak_v) and v_grid[0] <= peak_v <= v_grid[-1]:
                    peak_y = float(np.interp(peak_v, v_grid, dqdv))
                    fig.add_trace(go.Scatter(
                        x=[peak_v], y=[peak_y], mode="markers",
                        marker=dict(symbol="triangle-down", size=12,
                                    color=style["line"]["color"]),
                        name=f"{lbl} extra peak at {peak_v:.2f} V",
                        showlegend=True,
                    ))

    _add_color_key(fig, n_normal)
    return fig


# ---------------------------------------------------------------------------
# 4. cell_detail_card
# ---------------------------------------------------------------------------

def cell_detail_card(
    result: AnalysisResult,
    channel_index: int,
    rest_cell_df: Optional[pd.DataFrame] = None,
    charge_cell_df: Optional[pd.DataFrame] = None,
    top_charge_df: Optional[pd.DataFrame] = None,
    n_parallel: int = 1,
) -> go.Figure:
    """1×3 subplot detail card for a single flagged cell.

    Subplot 1: OCV voltage vs time (rest phase).
    Subplot 2: CUSUM trace from M4 metadata (if available).
    Subplot 3: dQ/dV (charge phase) — voltage-domain ICA when top_charge_df provided.

    Parameters
    ----------
    result:
        Full analysis result.
    channel_index:
        0-based channel index for the cell of interest.
    rest_cell_df:
        Long-format cell DataFrame restricted to the rest phase.
    charge_cell_df:
        Long-format cell DataFrame restricted to the charge phase.
    top_charge_df:
        Pack-level DataFrame for the charge segment (time_hours, current).
    n_parallel:
        Number of parallel strings.
    """
    from scipy import signal as _signal  # local import

    topo = result.topology
    mid = topo.module_for_channel(channel_index)
    gidx = topo.group_index_in_module(channel_index)
    cell_label = f"M{mid}/G{gidx}"

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["OCV (rest)", "CUSUM (cusum method)", "dQ/dV (charge)"],
        horizontal_spacing=0.10,
    )
    fig.update_layout(
        title=f"Cell {cell_label} — Detail",
        template="plotly_white",
        showlegend=False,
        height=350,
    )

    # Method metadata for this cell (fit parameters + CUSUM stats)
    _, _, method_map = _cell_lookup(result, mid)
    cell_methods = method_map.get(channel_index, {})

    # ---- Subplot 1: OCV voltage during rest + fitted decay model -----------
    if rest_cell_df is not None and not rest_cell_df.empty:
        ch_rest = rest_cell_df[rest_cell_df["channel_index"] == channel_index].sort_values("time_hours")
        if not ch_rest.empty:
            ch_rest = _downsample_df(ch_rest)
            t_min = ch_rest["time_hours"].min()
            t_rel = (ch_rest["time_hours"] - t_min).values
            v = ch_rest["voltage"].values
            fig.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=v,
                    mode="lines",
                    line=dict(color="steelblue", width=1.5),
                    name="OCV",
                ),
                row=1, col=1,
            )
            ocv_mr = cell_methods.get("ocv_k")
            if ocv_mr is not None:
                k = ocv_mr.metadata.get("k", float("nan"))
                v_ocv = ocv_mr.metadata.get("V_ocv", float("nan"))
                tau = ocv_mr.metadata.get("tau", float("nan"))
                settled = t_rel >= _SETTLING_H
                if (np.isfinite(k) and np.isfinite(v_ocv) and np.isfinite(tau)
                        and settled.sum() >= 5):
                    a = _reconstruct_ocv_amplitude(
                        t_rel[settled], v[settled], v_ocv, tau, k)
                    if np.isfinite(a):
                        t_fit = np.linspace(_SETTLING_H, float(t_rel.max()), 200)
                        fig.add_trace(
                            go.Scatter(
                                x=t_fit,
                                y=v_ocv + a * np.exp(-t_fit / tau) - k * t_fit,
                                mode="lines",
                                line=dict(color="black", width=1.5, dash="dash"),
                                name="fit",
                            ),
                            row=1, col=1,
                        )
                if np.isfinite(k):
                    fig.add_annotation(
                        text=(f"k = {k_to_mv_per_h(k)}<br>"
                              f"τ = {fmt_num(tau, 2, 'h')}<br>"
                              f"V_ocv = {fmt_num(v_ocv, 4, 'V')}"),
                        xref="x domain", yref="y domain",
                        x=0.98, y=0.98, xanchor="right", yanchor="top",
                        showarrow=False, align="right",
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.7)",
                        row=1, col=1,
                    )
    fig.update_xaxes(title_text="Time from rest start (h)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)

    # ---- Subplot 2: CUSUM from M4 metadata ---------------------------------
    # Find M4 MethodResult for this channel
    mv = next((m for m in result.module_verdicts if m.module_id == mid), None)
    cusum_plotted = False
    if mv is not None:
        cv = next((c for c in mv.all_cells if c.channel_index == channel_index), None)
        if cv is not None:
            m4_res = next(
                (mr for mr in cv.method_results if mr.method_name == "cusum"),
                None,
            )
            if m4_res is not None and "cusum_pos" in m4_res.metadata:
                # If the caller has stored the CUSUM series in metadata
                cusum_pos = m4_res.metadata.get("cusum_pos", [])
                cusum_neg = m4_res.metadata.get("cusum_neg", [])
                if cusum_pos:
                    t_ax = list(range(len(cusum_pos)))
                    fig.add_trace(
                        go.Scatter(x=t_ax, y=cusum_pos, mode="lines",
                                   line=dict(color="red", width=1.5), name="C+"),
                        row=1, col=2,
                    )
                if cusum_neg:
                    t_ax = list(range(len(cusum_neg)))
                    fig.add_trace(
                        go.Scatter(x=t_ax, y=cusum_neg, mode="lines",
                                   line=dict(color="blue", width=1.5), name="C-"),
                        row=1, col=2,
                    )
                cusum_plotted = True

    if not cusum_plotted:
        # Compute CUSUM on the fly from residuals if rest data is available
        if rest_cell_df is not None and not rest_cell_df.empty:
            ch_rest = rest_cell_df[rest_cell_df["channel_index"] == channel_index].sort_values("time_hours")
            if len(ch_rest) >= 10:
                from stress_screen.analysis.util import cusum_2sided
                v = ch_rest["voltage"].values
                resid = v - np.nanmean(v)
                c_pos, c_neg, _, _ = cusum_2sided(resid)
                t_hours = (ch_rest["time_hours"].values
                           - ch_rest["time_hours"].values[0])
                step = max(1, len(t_hours) // 2000)
                fig.add_trace(
                    go.Scatter(x=t_hours[::step], y=c_pos[::step], mode="lines",
                               line=dict(color="red", width=1.5), name="C+"),
                    row=1, col=2,
                )
                fig.add_trace(
                    go.Scatter(x=t_hours[::step], y=c_neg[::step], mode="lines",
                               line=dict(color="blue", width=1.5), name="C-"),
                    row=1, col=2,
                )
                # Decision threshold: ±h_sigma·σ̂ (RestParams default h=4)
                sigma = float(np.nanstd(resid))
                if np.isfinite(sigma) and sigma > 0:
                    for sign in (1.0, -1.0):
                        fig.add_hline(
                            y=sign * 4.0 * sigma, line_dash="dot",
                            line_color="dimgray", row=1, col=2,
                        )
                    fig.add_annotation(
                        text="±4σ alarm threshold",
                        xref="x2 domain", yref="y2 domain",
                        x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                        showarrow=False, font=dict(size=9),
                        row=1, col=2,
                    )
                cusum_mr = cell_methods.get("cusum")
                if cusum_mr is not None:
                    n_alarms = cusum_mr.metadata.get("n_alarms", float("nan"))
                    first = cusum_mr.metadata.get("first_alarm_h", float("nan"))
                    if np.isfinite(n_alarms):
                        note = f"{int(n_alarms)} alarm(s)"
                        if np.isfinite(first):
                            note += f", first at {first:.1f} h"
                        fig.add_annotation(
                            text=note,
                            xref="x2 domain", yref="y2 domain",
                            x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                            showarrow=False, font=dict(size=10),
                            bgcolor="rgba(255,255,255,0.7)",
                            row=1, col=2,
                        )
        else:
            fig.add_annotation(
                text="CUSUM N/A", showarrow=False,
                xref="x2 domain", yref="y2 domain", x=0.5, y=0.5,
            )
    fig.update_xaxes(title_text="Time from rest start (h)", row=1, col=2)
    fig.update_yaxes(title_text="CUSUM (V)", row=1, col=2)

    # ---- Subplot 3: dQ/dV during charge ------------------------------------
    q_pack_time, q_pack_cumul = _build_q_axis(top_charge_df, n_parallel)
    if charge_cell_df is not None and not charge_cell_df.empty:
        ch_chg = charge_cell_df[charge_cell_df["channel_index"] == channel_index].sort_values("time_hours")
        if len(ch_chg) >= 5:
            v_grid, dqdv = _compute_dqdv(
                ch_chg["voltage"].values,
                ch_chg["time_hours"].values,
                q_pack_time,
                q_pack_cumul,
            )
            if len(v_grid) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=v_grid,
                        y=dqdv,
                        mode="lines",
                        line=dict(color="darkred", width=1.5),
                        name="dQ/dV",
                    ),
                    row=1, col=3,
                )
    fig.update_xaxes(title_text="Voltage (V)", row=1, col=3)
    fig.update_yaxes(title_text="dQ/dV (Ah/V)", row=1, col=3)

    return fig


# ---------------------------------------------------------------------------
# 5. divergence_chart  — M3 visualisation
# ---------------------------------------------------------------------------

def divergence_chart(
    result: AnalysisResult,
    module_id: int,
    rest_cell_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """|V_cell − V_module_median| vs time for every cell in a module.

    Traces are rolling-median smoothed for display (the tester's integer-mV
    quantization otherwise dominates); detection uses raw data. Flagged
    cells keep a faint raw trace and get the detector's fitted
    (temperature-compensated) divergence trend drawn dashed.
    """
    fig = go.Figure()
    fig.update_layout(
        title=(f"Module M{module_id} — Voltage Divergence from Fleet Median "
               f"(M3; display-smoothed, detection uses raw data)"),
        xaxis_title="Time from rest start (h)",
        yaxis_title="|V − fleet median| (mV)",
        template="plotly_white",
    )

    if rest_cell_df is None or rest_cell_df.empty:
        fig.add_annotation(text="No rest data", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    topo = result.topology
    channels = sorted(topo.channels_in_module(module_id))
    verdict_map, label_map, method_map = _cell_lookup(result, module_id)

    mod_df = rest_cell_df[rest_cell_df["channel_index"].isin(channels)].copy()
    if mod_df.empty:
        return fig

    t_min = mod_df["time_hours"].min()
    mod_df["t_rel"] = mod_df["time_hours"] - t_min

    # Pivot → time × channel, compute fleet median at each timestamp
    pivot = mod_df.pivot_table(index="t_rel", columns="channel_index",
                                values="voltage", aggfunc="mean")
    fleet_median = pivot.median(axis=1)

    t_vals = pivot.index.values
    step = max(1, len(t_vals) // 1000)

    n_normal = 0
    for ch in channels:
        if ch not in pivot.columns:
            continue
        dev_mv = (pivot[ch] - fleet_median).abs().values * 1000.0  # mV
        smoothed = _rolling_median(dev_mv, t_vals)
        verdict = verdict_map.get(ch, "NORMAL")
        lbl = label_map.get(ch, f"Ch{ch}")
        style = _trace_style(verdict)
        is_flagged = verdict in ("HIGH", "ELEVATED")
        if not is_flagged:
            n_normal += 1
        else:
            # faint raw trace behind the smoothed one
            fig.add_trace(go.Scatter(
                x=t_vals[::step], y=dev_mv[::step], mode="lines",
                name=f"{lbl} raw",
                line=dict(color=style["line"]["color"], width=0.8),
                opacity=0.3, showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=t_vals[::step], y=smoothed[::step], mode="lines",
            name=lbl, **style,
        ))

        # Detector's fitted divergence trend for flagged cells
        if is_flagged:
            spread_mr = method_map.get(ch, {}).get("spread")
            if spread_mr is not None:
                slope = spread_mr.metadata.get(
                    "divergence_slope_v_per_h", float("nan"))
                if np.isfinite(slope):
                    t_line = np.array([0.0, float(t_vals.max())])
                    offset = float(np.nanmedian(smoothed[: max(5, len(smoothed) // 20)]))
                    fig.add_trace(go.Scatter(
                        x=t_line,
                        y=offset + slope * 1000.0 * t_line,
                        mode="lines",
                        name=(f"{lbl} fitted trend "
                              f"{k_to_mv_per_h(slope)} (T-compensated)"),
                        line=dict(color="black", width=1.5, dash="dash"),
                        showlegend=True,
                    ))

    _add_color_key(fig, n_normal)
    return fig


# ---------------------------------------------------------------------------
# 6. rank_chart  — M6 visualisation
# ---------------------------------------------------------------------------

def rank_chart(
    result: AnalysisResult,
    module_id: int,
    rest_cell_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Rank percentile vs time for each cell in a module during rest.

    Traces are rolling-median smoothed for display (integer-mV quantization
    makes raw ranks oscillate wildly); detection uses raw data. The 0–20 %
    danger band is shaded; flagged-cell legends carry the measured rank
    statistics.
    """
    fig = go.Figure()
    fig.update_layout(
        title=(f"Module M{module_id} — Voltage Rank Percentile Over Rest "
               f"(M6; display-smoothed, detection uses raw data)"),
        xaxis_title="Time from rest start (h)",
        yaxis_title="Rank percentile (%)",
        template="plotly_white",
    )
    fig.add_hrect(y0=0.0, y1=20.0, fillcolor="orange", opacity=0.12,
                  line_width=0)
    fig.add_hline(y=20.0, line_dash="dash", line_color="orange",
                  annotation_text="bottom-20 % band",
                  annotation_position="top right")

    if rest_cell_df is None or rest_cell_df.empty:
        fig.add_annotation(text="No rest data", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    topo = result.topology
    channels = sorted(topo.channels_in_module(module_id))
    verdict_map, label_map, method_map = _cell_lookup(result, module_id)

    mod_df = rest_cell_df[rest_cell_df["channel_index"].isin(channels)].copy()
    if mod_df.empty:
        return fig

    t_min = mod_df["time_hours"].min()
    mod_df["t_rel"] = mod_df["time_hours"] - t_min

    pivot = mod_df.pivot_table(index="t_rel", columns="channel_index",
                                values="voltage", aggfunc="mean")
    rank_pct = pivot.rank(axis=1, pct=True, method="average") * 100.0

    t_vals = rank_pct.index.values
    step = max(1, len(t_vals) // 1000)

    n_normal = 0
    for ch in channels:
        if ch not in rank_pct.columns:
            continue
        r_vals = _rolling_median(rank_pct[ch].values, t_vals)
        verdict = verdict_map.get(ch, "NORMAL")
        lbl = label_map.get(ch, f"Ch{ch}")
        style = _trace_style(verdict)
        if verdict in ("HIGH", "ELEVATED"):
            rank_mr = method_map.get(ch, {}).get("rank")
            if rank_mr is not None:
                mean_rank = rank_mr.metadata.get("mean_rank_pct", float("nan"))
                frac = rank_mr.metadata.get("frac_bot20", float("nan"))
                if np.isfinite(mean_rank):
                    lbl = f"{lbl} — mean rank {mean_rank:.0f}th pct"
                    if np.isfinite(frac):
                        lbl += f", {frac * 100:.0f}% in bottom 20%"
        else:
            n_normal += 1
        fig.add_trace(go.Scatter(
            x=t_vals[::step], y=r_vals[::step], mode="lines",
            name=lbl, **style,
        ))

    _add_color_key(fig, n_normal)
    return fig


# ---------------------------------------------------------------------------
# 7. temperature_chart  — M2 / Li-plating / ISC S2 visualisation
# ---------------------------------------------------------------------------

def temperature_chart(
    result: AnalysisResult,
    module_id: int,
    rest_cell_df: Optional[pd.DataFrame] = None,
    charge_cell_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Two-panel temperature chart: rest phase (left) + charge phase (right).

    Shows per-cell temperature vs time for a module. Helps interpret M2
    (T–OCV correlation), ISC S2 (dT/dt anomaly), and Li-plating cold/heat
    signatures. HIGH/ELEVATED cells coloured; others in gray.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Temperature during rest", "Temperature during charge"],
        horizontal_spacing=0.10,
    )
    fig.update_layout(
        title=f"Module M{module_id} — Temperature (Rest & Charge)",
        template="plotly_white",
        height=380,
    )

    topo = result.topology
    channels = sorted(topo.channels_in_module(module_id))
    verdict_map, label_map, method_map = _cell_lookup(result, module_id)

    def _add_traces(df, col, x_title):
        if df is None or df.empty or "temperature" not in df.columns:
            fig.add_annotation(
                text="No temperature data", showarrow=False,
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.5, y=0.5, row=1, col=col,
            )
            return
        mod_df = df[df["channel_index"].isin(channels)].copy()
        if mod_df.empty:
            return
        t_min = mod_df["time_hours"].min()
        t_span = float(mod_df["time_hours"].max() - t_min)
        for ch in channels:
            ch_df = mod_df[mod_df["channel_index"] == ch].sort_values("time_hours")
            if ch_df.empty or ch_df["temperature"].isna().all():
                continue
            ch_df = _downsample_df(ch_df)
            verdict = verdict_map.get(ch, "NORMAL")
            lbl = label_map.get(ch, f"Ch{ch}")
            style = _trace_style(verdict)
            is_flagged = verdict in ("HIGH", "ELEVATED")
            fig.add_trace(go.Scatter(
                x=ch_df["time_hours"] - t_min,
                y=ch_df["temperature"],
                mode="lines",
                name=lbl,
                line=style["line"],
                opacity=style["opacity"],
                showlegend=is_flagged and col == 1,
                legendgroup=str(ch),
            ), row=1, col=col)

            # Rest panel: draw the ISC S2 fitted thermal slope for flagged cells
            if is_flagged and col == 1:
                isc_mr = method_map.get(ch, {}).get("isc")
                if isc_mr is not None:
                    slope = isc_mr.metadata.get("s2_dT_dt_raw_slope", float("nan"))
                    if np.isfinite(slope) and t_span > 0:
                        t_med = ch_df["temperature"].median()
                        t_line = np.array([0.0, t_span])
                        y_line = t_med + slope * (t_line - t_span / 2)
                        fig.add_trace(go.Scatter(
                            x=t_line, y=y_line, mode="lines",
                            name=f"{lbl} dT/dt = {slope:.3f} °C/h",
                            line=dict(color="black", width=1.5, dash="dash"),
                            showlegend=True,
                        ), row=1, col=col)
        fig.update_xaxes(title_text=x_title, row=1, col=col)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=col)

    _add_traces(rest_cell_df, 1, "Time from rest start (h)")
    _add_traces(charge_cell_df, 2, "Time from charge start (h)")

    n_normal = sum(1 for ch in channels
                   if verdict_map.get(ch, "NORMAL") == "NORMAL")
    _add_color_key(fig, n_normal)

    return fig


# ---------------------------------------------------------------------------
# 8. method_zscore_heatmap  — all-methods overview
# ---------------------------------------------------------------------------

def method_zscore_heatmap(
    result: AnalysisResult,
    module_id: int,
) -> go.Figure:
    """Heatmap of method z-scores grouped by evidence cluster.

    Rows are ordered by cluster (the aggregation's CLUSTERS map): each
    cluster contributes a bold summary row with its cluster score followed by
    its member-method rows. Columns = cell groups. Cell text is the numeric
    z (verdict is encoded by color and shown in hover). Color clamps at ±6
    so scores beyond the ±2 gates remain distinguishable.
    """
    from stress_screen.analysis.aggregate import CLUSTERS

    mv = next((m for m in result.module_verdicts if m.module_id == module_id), None)
    if mv is None or not mv.all_cells:
        fig = go.Figure()
        fig.add_annotation(text="No cell data", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    cells = sorted(mv.all_cells, key=lambda c: c.group_in_module)
    x_labels = [c.label for c in cells]
    present_methods = {mr.method_name for mr in cells[0].method_results}

    def _cell_z(cv, method: str) -> float:
        mr = next((m for m in cv.method_results if m.method_name == method), None)
        return mr.z_score if mr is not None else float("nan")

    # Row plan, top to bottom: per cluster a bold score row then members;
    # any method outside the known clusters is appended at the end.
    rows: list[tuple[str, list[float]]] = []  # (label, values)
    covered: set[str] = set()
    for cluster, members in CLUSTERS.items():
        members_here = [m for m in members if m in present_methods]
        if not members_here:
            continue
        cluster_vals = [
            (cv.cluster_scores or {}).get(cluster, float("nan")) for cv in cells
        ]
        rows.append((f"<b>{cluster.replace('_', ' ').upper()}</b>", cluster_vals))
        for method in members_here:
            rows.append((f"  · {method.replace('_', ' ')}",
                         [_cell_z(cv, method) for cv in cells]))
            covered.add(method)
    for method in sorted(present_methods - covered):
        rows.append((method.replace("_", " "),
                     [_cell_z(cv, method) for cv in cells]))

    y_labels = [label for label, _ in rows]
    z_matrix = [
        [float(np.clip(v, -6.0, 6.0)) if np.isfinite(v) else 0.0 for v in vals]
        for _, vals in rows
    ]
    text_matrix = [
        [f"{v:.1f}" if np.isfinite(v) else "—" for v in vals]
        for _, vals in rows
    ]

    colorscale = [
        [0.0,  "rgb(0,180,0)"],
        [0.50, "rgb(255,255,100)"],
        [1.0,  "rgb(220,0,0)"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        zmin=-6.0, zmax=6.0,
        colorscale=colorscale,
        colorbar=dict(title="Z-score",
                      tickvals=[-6, -2, 0, 2, 6],
                      ticktext=["−6", "−2", "0", "2 (gate)", "≥6"]),
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Cell: %{x}<br>Row: %{y}<br>z = %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=(f"Module M{module_id} — Method Z-Scores by Evidence Cluster "
               f"(bold rows = cluster scores)"),
        template="plotly_white",
        yaxis=dict(autorange="reversed"),  # first cluster at top
        margin=dict(l=170),
        height=max(300, 40 * len(rows) + 140),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. phase_timeline
# ---------------------------------------------------------------------------

def phase_timeline(
    top_df: pd.DataFrame,
    segments: list[Segment],
) -> go.Figure:
    """Pack current vs time with shaded phase segments.

    Shading:
    - charge:    light blue
    - discharge: light red / salmon
    - rest:      light green

    Parameters
    ----------
    top_df:
        Pack-level DataFrame with columns ``time_hours`` and ``current``.
    segments:
        List of Segment objects (as returned by segmentation.segment()).
    """
    _PHASE_COLORS = {
        "charge":    "rgba(100, 180, 255, 0.25)",
        "discharge": "rgba(255, 120, 120, 0.25)",
        "rest":      "rgba(100, 220, 100, 0.25)",
    }

    top_ds = _downsample_df(top_df)

    fig = go.Figure()

    # -- Shaded, labeled phase rectangles (behind the current trace) --
    total_span = max(
        (seg.end_time_h for seg in segments), default=0.0
    ) - min((seg.start_time_h for seg in segments), default=0.0)

    for seg in segments:
        color = _PHASE_COLORS.get(seg.phase, "rgba(200,200,200,0.2)")
        # Label each region directly; skip labels on slivers (< 2 % of span)
        wide_enough = total_span > 0 and seg.duration_h / total_span >= 0.02
        fig.add_vrect(
            x0=seg.start_time_h,
            x1=seg.end_time_h,
            fillcolor=color,
            line_width=0,
            layer="below",
            annotation_text=(
                f"{seg.phase.capitalize()} {seg.duration_h:.1f} h"
                if wide_enough else ""
            ),
            annotation_position="top left",
            annotation_font_size=10,
        )

    # -- Pack current line ----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=top_ds["time_hours"],
        y=top_ds["current"],
        mode="lines",
        name="Pack current (A)",
        line=dict(color="black", width=1.5),
    ))

    fig.update_layout(
        title="Charge/Discharge/Rest Phase Timeline",
        xaxis_title="Time (h)",
        yaxis_title="Current (A)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
