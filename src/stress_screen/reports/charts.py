"""
reports/charts.py — Shared Plotly figure builders for stress_screen.

Pure functions: take analysis result objects and return plotly Figure objects.
No I/O; no .show() calls. Used by both html.py (interactive) and pdf.py (static PNG).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stress_screen.models import AnalysisResult, Segment


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _downsample_df(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """Return a uniformly-thinned copy of *df* capped at *max_points* rows."""
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    "HIGH": "red",
    "ELEVATED": "orange",
    "NORMAL": "gray",
}

# Green(0) → Yellow(2) → Red(5) colorscale for composite_z in [0, 5]
_HEATMAP_COLORSCALE = [
    [0.0, "rgb(0,180,0)"],
    [0.4, "rgb(255,220,0)"],
    [1.0, "rgb(220,0,0)"],
]


# ---------------------------------------------------------------------------
# 1. pack_heatmap
# ---------------------------------------------------------------------------

def pack_heatmap(result: AnalysisResult) -> go.Figure:
    """Heatmap of composite_z per cell-group, organised as modules × groups.

    Rows = modules (M1..MN, bottom to top on plot).
    Columns = cell-groups within module (G1..GS, left to right).
    Cell colour = composite_z clamped to [0, 5].
    Each cell is annotated with its verdict label (HIGH / ELEVATED / NORMAL).
    Right-side row annotation shows module verdict (OK / NOK).
    """
    topo = result.topology
    n_modules = topo.module_count
    n_groups = topo.series  # cell-groups per module

    # Build z-matrix (rows = modules bottom-to-top, cols = groups)
    # We store rows in bottom-to-top order so the plot y-axis reads M1 at bottom
    z_matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []
    y_labels: list[str] = []  # bottom-to-top

    for mid in range(1, n_modules + 1):
        mv = next((m for m in result.module_verdicts if m.module_id == mid), None)
        row_z: list[float] = []
        row_txt: list[str] = []
        for gidx in range(1, n_groups + 1):
            if mv is not None:
                cv = next((c for c in mv.all_cells if c.group_in_module == gidx), None)
            else:
                cv = None
            if cv is not None:
                cz = float(np.clip(cv.composite_z, 0.0, 5.0))
                row_z.append(cz)
                row_txt.append(cv.verdict)
            else:
                row_z.append(0.0)
                row_txt.append("")
        z_matrix.append(row_z)
        text_matrix.append(row_txt)
        nok = (mv.verdict == "NOK") if mv is not None else False
        y_labels.append(f"M{mid} {'NOK' if nok else 'OK'}")

    x_labels = [f"G{g}" for g in range(1, n_groups + 1)]

    heatmap = go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        zmin=0.0,
        zmax=5.0,
        colorscale=_HEATMAP_COLORSCALE,
        colorbar=dict(title="Composite Z", tickvals=[0, 1, 2, 3, 4, 5]),
        text=text_matrix,
        texttemplate="%{text}",
        hovertemplate="Module: %{y}<br>Group: %{x}<br>Z-score: %{z:.2f}<br>Verdict: %{text}<extra></extra>",
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="Pack Overview — Composite Z-Score Heatmap",
        xaxis_title="Cell Group",
        yaxis_title="Module",
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

    HIGH cells: red, thick line, shown in legend with their label.
    Normal/Elevated cells: gray, thin, semi-transparent.

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
    mv = next((m for m in result.module_verdicts if m.module_id == module_id), None)
    channels = sorted(topo.channels_in_module(module_id))

    # Build verdict lookup: channel → verdict string
    verdict_map: dict[int, str] = {}
    label_map: dict[int, str] = {}
    if mv is not None:
        for cv in mv.all_cells:
            verdict_map[cv.channel_index] = cv.verdict
            label_map[cv.channel_index] = cv.label

    t_min = rest_cell_df["time_hours"].min()

    for ch in channels:
        ch_df = rest_cell_df[rest_cell_df["channel_index"] == ch].sort_values("time_hours")
        if ch_df.empty:
            continue
        ch_df = _downsample_df(ch_df)
        t_rel = ch_df["time_hours"] - t_min
        v = ch_df["voltage"]

        verdict = verdict_map.get(ch, "NORMAL")
        lbl = label_map.get(ch, f"Ch{ch}")

        if verdict == "HIGH":
            fig.add_trace(go.Scatter(
                x=t_rel,
                y=v,
                mode="lines",
                name=lbl,
                line=dict(color="red", width=2.5),
                showlegend=True,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=t_rel,
                y=v,
                mode="lines",
                name=lbl,
                line=dict(color="lightgray", width=1),
                opacity=0.5,
                showlegend=False,
            ))

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

    Follows ICA best-practice: resample before differentiating (implicit
    smoothing), no post-gradient filter. Returns empty arrays when Q data
    is unavailable — caller must handle the empty-array case.
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
    mv = next((m for m in result.module_verdicts if m.module_id == module_id), None)
    channels = sorted(topo.channels_in_module(module_id))

    verdict_map: dict[int, str] = {}
    label_map: dict[int, str] = {}
    if mv is not None:
        for cv in mv.all_cells:
            verdict_map[cv.channel_index] = cv.verdict
            label_map[cv.channel_index] = cv.label

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

        if verdict == "HIGH":
            fig.add_trace(go.Scatter(
                x=v_grid,
                y=dqdv,
                mode="lines",
                name=lbl,
                line=dict(color="red", width=2.5),
                showlegend=True,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=v_grid,
                y=dqdv,
                mode="lines",
                name=lbl,
                line=dict(color="lightgray", width=1),
                opacity=0.5,
                showlegend=False,
            ))

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
        subplot_titles=["OCV (rest)", "CUSUM (M4)", "dQ/dV (charge)"],
        horizontal_spacing=0.10,
    )
    fig.update_layout(
        title=f"Cell {cell_label} — Detail",
        template="plotly_white",
        showlegend=False,
        height=350,
    )

    # ---- Subplot 1: OCV voltage during rest --------------------------------
    if rest_cell_df is not None and not rest_cell_df.empty:
        ch_rest = rest_cell_df[rest_cell_df["channel_index"] == channel_index].sort_values("time_hours")
        if not ch_rest.empty:
            ch_rest = _downsample_df(ch_rest)
            t_min = ch_rest["time_hours"].min()
            fig.add_trace(
                go.Scatter(
                    x=ch_rest["time_hours"] - t_min,
                    y=ch_rest["voltage"],
                    mode="lines",
                    line=dict(color="steelblue", width=1.5),
                    name="OCV",
                ),
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
                (mr for mr in cv.method_results if mr.method_name == "M4_cusum"),
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
                t_ax = np.arange(len(c_pos))
                step = max(1, len(t_ax) // 2000)
                fig.add_trace(
                    go.Scatter(x=t_ax[::step], y=c_pos[::step], mode="lines",
                               line=dict(color="red", width=1.5), name="C+"),
                    row=1, col=2,
                )
                fig.add_trace(
                    go.Scatter(x=t_ax[::step], y=c_neg[::step], mode="lines",
                               line=dict(color="blue", width=1.5), name="C-"),
                    row=1, col=2,
                )
        else:
            fig.add_annotation(
                text="CUSUM N/A", showarrow=False,
                xref="x2 domain", yref="y2 domain", x=0.5, y=0.5,
            )
    fig.update_xaxes(title_text="Sample index", row=1, col=2)
    fig.update_yaxes(title_text="CUSUM", row=1, col=2)

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
# 5. phase_timeline
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

    # -- Shaded phase rectangles (added before the line so they sit behind) --
    # Collect unique phases for legend de-duplication
    shown_phases: set[str] = set()

    for seg in segments:
        color = _PHASE_COLORS.get(seg.phase, "rgba(200,200,200,0.2)")
        show_legend = seg.phase not in shown_phases
        shown_phases.add(seg.phase)
        fig.add_trace(go.Scatter(
            x=[seg.start_time_h, seg.end_time_h, seg.end_time_h, seg.start_time_h, seg.start_time_h],
            y=[None, None, None, None, None],  # placeholder; shapes used instead
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            mode="lines",
            name=seg.phase.capitalize(),
            showlegend=show_legend,
            legendgroup=seg.phase,
            hoverinfo="skip",
        ))
        # Use layout shapes for the actual shading (cleaner approach)
        fig.add_vrect(
            x0=seg.start_time_h,
            x1=seg.end_time_h,
            fillcolor=color,
            line_width=0,
            layer="below",
            annotation_text="",
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
