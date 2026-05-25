"""
reports/html.py — Standalone HTML report writer for stress_screen.

Generates a single self-contained HTML file (all Plotly JS inlined).
No external dependencies required to open the report.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from stress_screen.models import AnalysisResult
from stress_screen.reports.charts import (
    cell_detail_card,
    dv_dq_chart,
    ocv_fit_overlay,
    pack_heatmap,
    phase_timeline,
)

# ---------------------------------------------------------------------------
# Version string
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("stress_screen")
except Exception:
    _VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _templates_dir() -> Path:
    """Return the path to the Jinja2 templates directory.

    Supports both PyInstaller bundles and normal development layouts.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "reports" / "templates"  # type: ignore[attr-defined]
    return Path(__file__).parent / "templates"


def _fig_to_html(fig: Any) -> str:
    """Render a Plotly figure to an HTML fragment with inlined JS."""
    return fig.to_html(
        include_plotlyjs="inline",
        full_html=False,
        config={"responsive": True},
    )


def _fig_to_html_cdn(fig: Any) -> str:
    """Render a Plotly figure to an HTML fragment without the JS bundle.

    Used for every figure after the first when the JS has already been
    inlined once — saves significant file size.
    """
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_html_report(
    result: AnalysisResult,
    rest_cell_df: pd.DataFrame,
    charge_cell_df: pd.DataFrame,
    top_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write a standalone HTML report to *out_path*.

    All Plotly JS is inlined — no external dependencies required to open
    the resulting file.

    Parameters
    ----------
    result:
        Full analysis result (topology, segments, module verdicts).
    rest_cell_df:
        Long-format cell DataFrame restricted to the first rest segment.
    charge_cell_df:
        Long-format cell DataFrame restricted to the first charge segment.
    top_df:
        Pack-level DataFrame (time_hours, current, voltage, …).
    out_path:
        Destination file path; parent directory must already exist.
    """
    topo = result.topology

    # ------------------------------------------------------------------
    # 1. Header metadata
    # ------------------------------------------------------------------
    pack_id = result.csv_path.stem
    test_date = (
        pd.to_datetime(top_df["time_hours"].iloc[0], unit="h").strftime("%Y-%m-%d")
        if not top_df.empty and "time_hours" in top_df.columns
        else "unknown"
    )
    # Prefer absolute wall-clock time if the CSV carries it; fall back to
    # the test date extracted from the filename (DDMMYYYY pattern).
    import re as _re
    date_match = _re.search(r"_P(\d{2})(\d{2})(\d{4})_", result.csv_path.name)
    if date_match:
        day, month, year = date_match.group(1), date_match.group(2), date_match.group(3)
        test_date = f"{year}-{month}-{day}"

    config_str = (
        f"{topo.module_count} modules, "
        f"{topo.config_name}, "
        f"{topo.active_channels} active cell-groups"
    )
    report_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overall_verdict = "NOK" if result.any_nok else "OK"

    # ------------------------------------------------------------------
    # 2. Module summary table rows
    # ------------------------------------------------------------------
    summary_table_rows: list[dict[str, Any]] = []
    for mv in result.module_verdicts:
        flagged_labels = [c.label for c in mv.flagged_cells]
        # Collect distinct method names that fired HIGH on any flagged cell
        methods_fired_set: list[str] = []
        seen_methods: set[str] = set()
        for fc in mv.flagged_cells:
            for mr in fc.method_results:
                if mr.verdict == "HIGH" and mr.method_name not in seen_methods:
                    methods_fired_set.append(mr.method_name)
                    seen_methods.add(mr.method_name)
        summary_table_rows.append(
            {
                "module_id": mv.module_id,
                "verdict": mv.verdict,
                "flagged_cell_labels": flagged_labels,
                "methods_fired": ", ".join(methods_fired_set),
            }
        )

    # ------------------------------------------------------------------
    # 3. Render pack-level charts
    # ------------------------------------------------------------------
    # The first figure to_html call inlines the Plotly JS bundle (~3 MB).
    # All subsequent figures omit it to keep the file size manageable.
    plotly_js_included = False

    def _render(fig: Any) -> str:
        nonlocal plotly_js_included
        if not plotly_js_included:
            plotly_js_included = True
            return _fig_to_html(fig)
        return _fig_to_html_cdn(fig)

    charts = {
        "pack_heatmap": _render(pack_heatmap(result)),
        "phase_timeline": _render(phase_timeline(top_df, result.segments)),
    }

    # ------------------------------------------------------------------
    # 4. Per-module detail
    # ------------------------------------------------------------------
    module_details: list[dict[str, Any]] = []
    for mv in result.module_verdicts:
        mid = mv.module_id

        ocv_fig = ocv_fit_overlay(result, mid, rest_cell_df)
        dvdq_fig = dv_dq_chart(result, mid, charge_cell_df)

        # Flagged cell detail cards
        flagged_cells_data: list[dict[str, Any]] = []
        for fc in mv.flagged_cells:
            detail_fig = cell_detail_card(
                result,
                fc.channel_index,
                rest_cell_df,
                charge_cell_df,
            )
            flagged_cells_data.append(
                {
                    "label": fc.label,
                    "composite_z": fc.composite_z,
                    "method_results": fc.method_results,
                    "detail_chart": _render(detail_fig),
                }
            )

        module_details.append(
            {
                "module_id": mid,
                "verdict": mv.verdict,
                "ocv_chart": _render(ocv_fig),
                "dvdq_chart": _render(dvdq_fig),
                "flagged_cells": flagged_cells_data,
            }
        )

    # ------------------------------------------------------------------
    # 5. Render Jinja2 template
    # ------------------------------------------------------------------
    templates_dir = _templates_dir()
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,  # we handle safe HTML ourselves
    )
    template = env.get_template("report.html.j2")

    html_content = template.render(
        pack_id=pack_id,
        test_date=test_date,
        config_str=config_str,
        report_date=report_date,
        overall_verdict=overall_verdict,
        summary_table_rows=summary_table_rows,
        charts=charts,
        module_details=module_details,
        version=_VERSION,
    )

    # ------------------------------------------------------------------
    # 6. Write to disk
    # ------------------------------------------------------------------
    out_path.write_text(html_content, encoding="utf-8")
