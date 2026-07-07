"""Chart-builder tests — inspect fig.data/fig.layout, no Kaleido rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_screen.analysis.aggregate import CLUSTERS
from stress_screen.models import (
    AnalysisResult, CellVerdict, MethodResult, ModuleVerdict, Segment,
)
from stress_screen.reports.charts import (
    cell_detail_card, method_zscore_heatmap, ocv_fit_overlay, pack_heatmap,
    phase_timeline, divergence_chart, rank_chart,
)
from stress_screen.topology import derive_topology

N_CH = 8  # one 4P8S module


def _cell(ch: int, verdict: str, composite: float) -> CellVerdict:
    mrs = [
        MethodResult("ocv_k", composite, verdict,
                     {"k": 8e-4, "V_ocv": 3.34, "tau": 1.2}),
        MethodResult("thermal_corr", 0.1, "NORMAL", {"pearson_r": 0.2}),
        MethodResult("spread", composite, verdict,
                     {"divergence_slope_v_per_h": 6e-5}),
        MethodResult("cusum", 0.2, "NORMAL",
                     {"n_alarms": 1, "first_alarm_h": 5.0}),
        MethodResult("temp_k", 0.3, "NORMAL",
                     {"k_corrected": 7e-4, "T_mean": 24.0,
                      "temp_correction_applied": True}),
        MethodResult("rank", composite, verdict,
                     {"mean_rank_pct": 15.0, "frac_bot20": 0.7,
                      "rank_slope": -0.2}),
        MethodResult("li_plating", 0.1, "NORMAL",
                     {"dqdv_extra_peak_voltage": float("nan")}),
        MethodResult("isc", 0.1, "NORMAL",
                     {"s2_dT_dt_raw_slope": 0.05}),
    ]
    return CellVerdict(
        channel_index=ch, module_id=1, group_in_module=ch + 1,
        composite_z=composite, n_methods_high=0, verdict=verdict,
        method_results=mrs,
        cluster_scores={c: composite / (i + 1) for i, c in enumerate(CLUSTERS)},
        composite_z_legacy=composite * 0.8,
    )


@pytest.fixture
def result():
    topo = derive_topology(N_CH, 1)
    cells = [
        _cell(0, "HIGH", 2.6),
        _cell(1, "ELEVATED", 1.3),
        *[_cell(i, "NORMAL", 0.1) for i in range(2, N_CH)],
    ]
    flagged = [c for c in cells if c.verdict in ("HIGH", "ELEVATED")]
    mv = ModuleVerdict(1, "NOK", flagged, cells)
    return AnalysisResult(
        csv_path=Path("Synth_D01032026_M1.csv"),
        topology=topo,
        segments=[Segment("charge", 0.0, 3.0, 0, 30),
                  Segment("rest", 3.0, 53.0, 31, 400)],
        module_verdicts=[mv],
    )


@pytest.fixture
def rest_df():
    rng = np.random.default_rng(1)
    t = np.arange(0, 50, 0.05)
    frames = []
    for ch in range(N_CH):
        v = 3.35 + 0.015 * np.exp(-t) - 1e-4 * ch * t + rng.normal(0, 3e-4, len(t))
        frames.append(pd.DataFrame({
            "time_hours": t + 3.0, "channel_index": ch,
            "voltage": v, "temperature": 22.0 + rng.normal(0, 0.3, len(t)),
        }))
    return pd.concat(frames, ignore_index=True)


def _legend_names(fig):
    return [tr.name for tr in fig.data if getattr(tr, "showlegend", None)]


def test_ocv_overlay_highlights_elevated_and_draws_fits(result, rest_df):
    fig = ocv_fit_overlay(result, 1, rest_df)
    names = _legend_names(fig)
    # ELEVATED cell now visible and named (the old code hid it)
    assert any(n == "M1/G2" for n in names)
    elevated_trace = next(tr for tr in fig.data if tr.name == "M1/G2")
    assert elevated_trace.line.color == "orange"
    # fitted-model overlays with physical units in the legend
    fit_names = [n for n in names if "fit — k =" in n]
    assert len(fit_names) == 2  # one per flagged cell
    assert any("mV/h" in n and "τ =" in n for n in fit_names)
    # dashes on fit traces
    fit_traces = [tr for tr in fig.data if tr.name in fit_names]
    assert all(tr.line.dash == "dash" for tr in fit_traces)
    # color key present
    assert any(n.startswith("normal cells (n=") for n in names)


def test_divergence_chart_smooths_and_draws_trend(result, rest_df):
    fig = divergence_chart(result, 1, rest_df)
    names = _legend_names(fig)
    assert any("fitted trend" in n and "mV/h" in n for n in names)
    assert any("T-compensated" in n for n in names)
    assert any(n.startswith("normal cells") for n in names)


def test_rank_chart_band_and_stats(result, rest_df):
    fig = rank_chart(result, 1, rest_df)
    names = _legend_names(fig)
    assert any("mean rank" in n and "bottom 20%" in n for n in names)
    # shaded band exists
    assert any(getattr(s, "y1", None) == 20.0 for s in fig.layout.shapes)


def test_pack_heatmap_numeric_text_gate_bands_m1_top(result):
    fig = pack_heatmap(result)
    hm = fig.data[0]
    # numeric composite text, not verdict words
    assert hm.text[0][0] == "2.60"
    assert "NORMAL" not in str(hm.text)
    # gate-anchored colorbar
    assert "ELEV" in str(hm.colorbar.ticktext) and "HIGH" in str(hm.colorbar.ticktext)
    # M1 at top via reversed axis
    assert fig.layout.yaxis.autorange == "reversed"
    # flagged cells outlined
    assert any(getattr(tr, "mode", "") == "markers" for tr in fig.data[1:])


def test_zscore_heatmap_cluster_rows(result):
    fig = method_zscore_heatmap(result, 1)
    y = list(fig.data[0].y)
    # bold cluster rows present, in CLUSTERS order
    cluster_rows = [lbl for lbl in y if lbl.startswith("<b>")]
    assert cluster_rows == [
        f"<b>{c.replace('_', ' ').upper()}</b>" for c in CLUSTERS
    ]
    # member rows follow their cluster row
    assert y.index("  · ocv k") == y.index("<b>SELF DISCHARGE</b>") + 1
    # wider clamp with the gate labeled
    assert "2 (gate)" in str(fig.data[0].colorbar.ticktext)


def test_phase_timeline_no_dummy_traces_and_labels():
    top_df = pd.DataFrame({
        "time_hours": np.arange(0, 53, 0.1),
        "current": np.where(np.arange(0, 53, 0.1) < 3.0, 5.0, 0.02),
    })
    segments = [Segment("charge", 0.0, 3.0, 0, 30),
                Segment("rest", 3.0, 53.0, 31, 529)]
    fig = phase_timeline(top_df, segments)
    # no all-None placeholder traces (the old legend hack)
    for tr in fig.data:
        assert tr.y is None or not all(v is None for v in tr.y)
    # phases labeled with duration
    ann_text = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "Rest 50.0 h" in ann_text
    assert "Charge 3.0 h" in ann_text


def test_cell_detail_card_fit_and_cusum_thresholds(result, rest_df):
    fig = cell_detail_card(result, 0, rest_df)
    ann_text = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "k = 0.80 mV/h" in ann_text
    assert "V_ocv" in ann_text
    assert "alarm threshold" in ann_text
    assert "1 alarm(s), first at 5.0 h" in ann_text
    # CUSUM x-axis now in hours
    assert fig.layout.xaxis2.title.text == "Time from rest start (h)"
