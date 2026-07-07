"""
reports/figures.py — build every report figure exactly once.

The HTML and PDF writers previously each rebuilt the same Plotly figures
(pack heatmap, phase timeline, six charts per module). ``build_figures``
produces one shared :class:`FigureSet` that both writers consume, halving
figure-construction time when both reports are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from stress_screen.models import AnalysisResult
from stress_screen.reports.charts import (
    cell_detail_card,
    divergence_chart,
    dv_dq_chart,
    method_zscore_heatmap,
    ocv_fit_overlay,
    pack_heatmap,
    phase_timeline,
    rank_chart,
    temperature_chart,
)


@dataclass
class ModuleFigures:
    """The six per-module detail figures."""

    ocv: Any
    dvdq: Any
    divergence: Any
    rank: Any
    temperature: Any
    zscore_heatmap: Any


@dataclass
class FigureSet:
    """All figures for one analysis run, built once and shared by writers."""

    pack_heatmap: Any
    phase_timeline: Any
    per_module: dict[int, ModuleFigures] = field(default_factory=dict)
    #: channel_index → detail card (flagged HIGH cells only; HTML report)
    flagged_cell_details: dict[int, Any] = field(default_factory=dict)


def build_figures(
    result: AnalysisResult,
    rest_cell_df: pd.DataFrame,
    charge_cell_df: pd.DataFrame,
    top_df: pd.DataFrame,
    top_charge_df: Optional[pd.DataFrame] = None,
    n_parallel: int = 1,
) -> FigureSet:
    """Build the full figure set for *result*."""
    figures = FigureSet(
        pack_heatmap=pack_heatmap(result),
        phase_timeline=phase_timeline(top_df, result.segments),
    )

    for mv in result.module_verdicts:
        mid = mv.module_id
        figures.per_module[mid] = ModuleFigures(
            ocv=ocv_fit_overlay(result, mid, rest_cell_df),
            dvdq=dv_dq_chart(
                result, mid, charge_cell_df,
                top_charge_df=top_charge_df, n_parallel=n_parallel,
            ),
            divergence=divergence_chart(result, mid, rest_cell_df),
            rank=rank_chart(result, mid, rest_cell_df),
            temperature=temperature_chart(result, mid, rest_cell_df, charge_cell_df),
            zscore_heatmap=method_zscore_heatmap(result, mid),
        )
        for fc in mv.flagged_cells:
            figures.flagged_cell_details[fc.channel_index] = cell_detail_card(
                result,
                fc.channel_index,
                rest_cell_df,
                charge_cell_df,
                top_charge_df=top_charge_df,
                n_parallel=n_parallel,
            )

    return figures
