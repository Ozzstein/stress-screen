"""
reports/html.py — Standalone HTML report writer for stress_screen.

Generates a single self-contained HTML file (all Plotly JS inlined).
No external dependencies required to open the report.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from stress_screen.models import AnalysisResult
from stress_screen.reports.figures import FigureSet, build_figures

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
    top_charge_df: Optional[pd.DataFrame] = None,
    n_parallel: int = 1,
    figures: Optional[FigureSet] = None,
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
    top_charge_df:
        Pack-level DataFrame restricted to the charge segment — used to build
        the Q axis for dV/dQ charts. Optional; falls back to sample index.
    n_parallel:
        Number of parallel strings (divides pack current to per-string Q).
    figures:
        Pre-built figure set (shared with the PDF writer). Built on demand
        when None.
    """
    topo = result.topology

    if figures is None:
        figures = build_figures(
            result, rest_cell_df, charge_cell_df, top_df,
            top_charge_df=top_charge_df, n_parallel=n_parallel,
        )

    # ------------------------------------------------------------------
    # 1. Header metadata
    # ------------------------------------------------------------------
    pack_id = result.csv_path.stem
    # Extract the test date from the filename (_D<DDMMYYYY>_ or legacy
    # _P<DDMMYYYY>_ pattern). Never fall back to today's date — a wrong date
    # on a QA document is worse than an explicit "unknown".
    from stress_screen.serialize import extract_test_date
    _test_date = extract_test_date(result.csv_path.name)
    test_date = _test_date.isoformat() if _test_date else "unknown"

    config_str = (
        f"{topo.module_count} modules, "
        f"{topo.config_name}, "
        f"{topo.active_channels} active cell-groups"
    )
    report_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if result.any_nok:
        overall_verdict = "NOK"
    elif any(m.verdict == "MARGINAL" for m in result.module_verdicts):
        overall_verdict = "MARGINAL"
    else:
        overall_verdict = "OK"

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
        "pack_heatmap": _render(figures.pack_heatmap),
        "phase_timeline": _render(figures.phase_timeline),
    }

    # ------------------------------------------------------------------
    # 4. Per-module detail
    # ------------------------------------------------------------------
    module_details: list[dict[str, Any]] = []
    for mv in result.module_verdicts:
        mid = mv.module_id
        mod_figs = figures.per_module[mid]

        # Flagged cell detail cards
        flagged_cells_data: list[dict[str, Any]] = []
        for fc in mv.flagged_cells:
            detail_fig = figures.flagged_cell_details[fc.channel_index]
            isc_mr = next((mr for mr in fc.method_results if mr.method_name == "isc"), None)
            isc_detail = None
            if isc_mr:
                meta = isc_mr.metadata
                def _fmt(v):
                    return f"{v:.3f}" if (v == v) else "—"
                isc_detail = {
                    "s1_z": _fmt(meta.get("s1_excess_k_z", float("nan"))),
                    "s1_excess_k": _fmt(meta.get("s1_excess_k", float("nan"))),
                    "s2_z": _fmt(meta.get("s2_dT_dt_z", float("nan"))),
                    "s2_slope": _fmt(meta.get("s2_dT_dt_slope", float("nan"))),
                    "s3_z": _fmt(meta.get("s3_area_deficit_z", float("nan"))),
                    "s3_area": _fmt(meta.get("s3_dvdq_area", float("nan"))),
                }
            flagged_cells_data.append(
                {
                    "label": fc.label,
                    "composite_z": fc.composite_z,
                    "method_results": fc.method_results,
                    "isc_detail": isc_detail,
                    "detail_chart": _render(detail_fig),
                }
            )

        # Build all-cell method data for the module table
        all_cells_data: list[dict[str, Any]] = []
        for cv in mv.all_cells:
            method_rows = []
            for mr in cv.method_results:
                z = mr.z_score
                z_str = f"{z:.2f}" if (z == z) else "—"  # NaN check
                method_rows.append({
                    "method_name": mr.method_name,
                    "z_str": z_str,
                    "verdict": mr.verdict,
                    "metadata": mr.metadata,
                })
            all_cells_data.append({
                "label": cv.label,
                "composite_z": cv.composite_z,
                "verdict": cv.verdict,
                "method_rows": method_rows,
            })

        method_names = [mr["method_name"] for mr in all_cells_data[0]["method_rows"]] if all_cells_data else []

        module_details.append(
            {
                "module_id": mid,
                "verdict": mv.verdict,
                "ocv_chart": _render(mod_figs.ocv),
                "dvdq_chart": _render(mod_figs.dvdq),
                "divergence_chart": _render(mod_figs.divergence),
                "rank_chart": _render(mod_figs.rank),
                "temperature_chart": _render(mod_figs.temperature),
                "zscore_heatmap": _render(mod_figs.zscore_heatmap),
                "flagged_cells": flagged_cells_data,
                "all_cells": all_cells_data,
                "method_names": method_names,
            }
        )

    # ------------------------------------------------------------------
    # 5. Render Jinja2 template
    # ------------------------------------------------------------------
    templates_dir = _templates_dir()
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "j2"]),
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
