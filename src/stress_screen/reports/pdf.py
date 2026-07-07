"""
reports/pdf.py — Static PDF report writer for stress_screen.

Uses Plotly + Kaleido for chart rasterisation and ReportLab for layout.

Public API
----------
    write_pdf_report(result, rest_cell_df, charge_cell_df, top_df, out_path)
"""

from __future__ import annotations

import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from stress_screen.models import AnalysisResult
from stress_screen.reports.figures import FigureSet, build_figures


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig_to_image(fig, width_pt: float, height_pt: float) -> Image:
    """Render a Plotly figure to a ReportLab Image at the given pt dimensions."""
    # Disable Kaleido's MathJax stage: we render no LaTeX, it slows every
    # figure down, and it is a known cause of intermittent renderer hangs on
    # Windows CI runners.
    try:
        import plotly.io as pio
        pio.kaleido.scope.mathjax = None
    except Exception:
        pass
    # Scale factor: 72 pt/inch, kaleido works in pixels; use 1.5× for crispness
    px_w = int(width_pt * 1.5)
    px_h = int(height_pt * 1.5)
    try:
        img_bytes = fig.to_image(format="png", width=px_w, height=px_h, scale=1)
    except Exception:
        # Fall back to plotly.io if the fig method isn't available
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=px_w, height=px_h, scale=1)
    return Image(BytesIO(img_bytes), width=width_pt, height=height_pt)


def _make_styles():
    """Return a namespace of ReportLab paragraph styles."""
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "TitleStyle",
        parent=base["Title"],
        fontSize=28,
        spaceAfter=12,
    )
    styles["h1"] = ParagraphStyle(
        "H1Style",
        parent=base["Heading1"],
        fontSize=18,
        spaceAfter=8,
    )
    styles["h2"] = ParagraphStyle(
        "H2Style",
        parent=base["Heading2"],
        fontSize=14,
        spaceAfter=6,
    )
    styles["normal"] = base["Normal"]
    styles["verdict_ok"] = ParagraphStyle(
        "VerdictOK",
        parent=base["Normal"],
        fontSize=36,
        textColor=colors.HexColor("#1a7a1a"),
        spaceAfter=12,
        alignment=1,  # center
    )
    styles["verdict_nok"] = ParagraphStyle(
        "VerdictNOK",
        parent=base["Normal"],
        fontSize=36,
        textColor=colors.HexColor("#cc0000"),
        spaceAfter=12,
        alignment=1,  # center
    )
    styles["center"] = ParagraphStyle(
        "CenterStyle",
        parent=base["Normal"],
        alignment=1,
    )
    styles["finding"] = ParagraphStyle(
        "FindingStyle",
        parent=base["Normal"],
        fontSize=9.5,
        leading=13,
        spaceAfter=6,
        leftIndent=8,
    )
    styles["small"] = ParagraphStyle(
        "SmallStyle",
        parent=base["Normal"],
        fontSize=8.5,
        leading=11,
    )
    return styles


# ---------------------------------------------------------------------------
# Page builders (return lists of Flowables)
# ---------------------------------------------------------------------------

def _page_exec_summary(result: AnalysisResult, findings, styles: dict) -> list:
    """Page 1 — executive summary: verdict, headline, and per-flagged-cell
    factual finding sentences."""
    topo = result.topology
    pack_id = result.csv_path.stem
    config_str = (
        f"{topo.module_count} modules, "
        f"{topo.config_name}, "
        f"{topo.active_channels} active cell-groups"
    )

    # Derive test date from filename (_D<DDMMYYYY>_ or legacy _P<DDMMYYYY>_)
    from stress_screen.serialize import extract_test_date
    _test_date = extract_test_date(result.csv_path.name)
    test_date = _test_date.strftime("%d/%m/%Y") if _test_date else "unknown"

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if result.any_nok:
        nok_ids = ", ".join(f"M{m.module_id}" for m in result.module_verdicts
                            if m.verdict == "NOK")
        verdict_text = f"NOK: {nok_ids}"
        verdict_style = styles["verdict_nok"]
    else:
        verdict_text = "ALL OK"
        verdict_style = styles["verdict_ok"]

    rest_str = (f"{findings.rest_duration_h:.1f} h"
                if findings.rest_duration_h else "—")

    flowables = [
        Spacer(1, 1.2 * cm),
        Paragraph("Stress Screen Report", styles["title"]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"<b>Pack ID:</b> {pack_id}", styles["normal"]),
        Paragraph(f"<b>Configuration:</b> {config_str}", styles["normal"]),
        Paragraph(f"<b>Test date:</b> {test_date} · "
                  f"<b>Longest rest:</b> {rest_str} (protocol ≥ 48 h)",
                  styles["normal"]),
        Paragraph(f"<b>Report generated:</b> {generated_at}", styles["normal"]),
        Spacer(1, 1.0 * cm),
        Paragraph(verdict_text, verdict_style),
        Spacer(1, 0.4 * cm),
        Paragraph(findings.headline, styles["center"]),
        Spacer(1, 0.8 * cm),
    ]

    if findings.cell_findings:
        flowables.append(Paragraph("Findings", styles["h2"]))
        for f in findings.cell_findings:
            color = "#cc0000" if f.verdict == "HIGH" else "#b8860b"
            flowables.append(Paragraph(
                f'<font color="{color}"><b>{f.label}</b></font> — {f.sentence}',
                styles["finding"],
            ))
        flowables.append(Spacer(1, 0.5 * cm))

    for mv in result.module_verdicts:
        color = "#1a7a1a" if mv.verdict == "OK" else "#cc0000"
        flowables.append(
            Paragraph(
                f'<font color="{color}">{mv.summary_line}</font>',
                styles["small"],
            )
        )

    flowables.append(PageBreak())
    return flowables


def _page_methodology(styles: dict) -> list:
    """One-page condensed methodology: the 8 detectors + cluster/gate rules.
    Text comes from findings.METHOD_DESCRIPTIONS so HTML and PDF can't drift."""
    from stress_screen.reports.findings import (
        METHOD_DESCRIPTIONS, VERDICT_RULES_TEXT,
    )

    flowables = [
        Paragraph("Methodology", styles["h1"]),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Each cell-group is scored by eight detection methods; every "
            "method produces a robust z-score (median/MAD against this "
            "pack's own fleet). Divergence and rank charts are smoothed "
            "with a ~30-min rolling median for display only — detection "
            "always runs on raw data.",
            styles["small"],
        ),
        Spacer(1, 0.3 * cm),
    ]
    for name, (descr, detects) in METHOD_DESCRIPTIONS.items():
        flowables.append(Paragraph(
            f"<b>{name}</b> — {descr} <i>Detects:</i> {detects}",
            styles["small"],
        ))
        flowables.append(Spacer(1, 0.15 * cm))
    flowables += [
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Verdict aggregation</b>", styles["h2"]),
        Paragraph(VERDICT_RULES_TEXT, styles["small"]),
        PageBreak(),
    ]
    return flowables


def _page2_module_table(result: AnalysisResult, styles: dict) -> list:
    """Module summary table page."""
    header = ["Module", "Verdict", "Flagged Cells", "Methods Fired"]

    data = [header]
    row_styles: list[tuple] = []

    for idx, mv in enumerate(result.module_verdicts):
        row_idx = idx + 1  # 0 is header row

        flagged = ", ".join(c.label for c in mv.flagged_cells) if mv.flagged_cells else "—"
        methods_fired: set[str] = set()
        for cv in mv.flagged_cells:
            for mr in cv.method_results:
                if mr.verdict in ("HIGH", "ELEVATED"):
                    methods_fired.add(mr.method_name)
        methods_str = ", ".join(sorted(methods_fired)) if methods_fired else "—"

        data.append([
            f"M{mv.module_id}",
            mv.verdict,
            flagged,
            methods_str,
        ])

        bg = colors.HexColor("#d4edda" if mv.verdict == "OK" else "#f8d7da")
        row_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    col_widths = [2.5 * cm, 2.5 * cm, 8 * cm, 8 * cm]

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    base_style = TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])
    for style_cmd in row_styles:
        base_style.add(*style_cmd)

    tbl.setStyle(base_style)

    flowables = [
        Paragraph("Module Summary", styles["h1"]),
        Spacer(1, 0.5 * cm),
        tbl,
    ]
    return flowables


def _page_pack_heatmap(
    fig,
    result: AnalysisResult,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
) -> list:
    """Pack heatmap sized to share the page with the module summary table."""
    usable_w = page_w - 2 * margin
    n_modules = result.topology.module_count
    # Scale with module count but keep room for the summary table above
    usable_h = min(page_h * 0.55, (4.5 + 1.8 * n_modules) * cm)

    img = _fig_to_image(fig, usable_w, usable_h)
    return [
        Spacer(1, 0.6 * cm),
        Paragraph("Pack Overview — Composite Z-Score Heatmap "
                  "(flagged cells outlined)", styles["h2"]),
        Spacer(1, 0.2 * cm),
        img,
        PageBreak(),
    ]


def _page_phase_timeline(
    fig,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
) -> list:
    """Full-page phase timeline."""
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - 2 * cm

    img = _fig_to_image(fig, usable_w, usable_h)
    return [
        Paragraph("Phase Timeline", styles["h1"]),
        Spacer(1, 0.3 * cm),
        img,
        PageBreak(),
    ]


def _pages_per_module(
    result: AnalysisResult,
    figures: FigureSet,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
) -> list:
    """Six charts across three pages for each module.

    Page 1: OCV relaxation + M3 divergence (rest-phase voltage)
    Page 2: dQ/dV + M6 rank percentile (charge + rank)
    Page 3: Temperature (rest & charge) + All-method z-score heatmap
    """
    flowables = []
    usable_w = page_w - 2 * margin
    chart_h = (page_h - 2 * margin - 4 * cm) / 2

    for mv in result.module_verdicts:
        mid = mv.module_id
        mod_figs = figures.per_module[mid]

        # --- Page 1: rest-phase voltage charts ---
        flowables.append(Paragraph(f"Module M{mid} — Rest Phase Analysis", styles["h2"]))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(_fig_to_image(mod_figs.ocv, usable_w, chart_h))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_fig_to_image(mod_figs.divergence, usable_w, chart_h))
        flowables.append(PageBreak())

        # --- Page 2: charge-phase + rank charts ---
        flowables.append(Paragraph(f"Module M{mid} — Charge Phase &amp; Rank Analysis", styles["h2"]))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(_fig_to_image(mod_figs.dvdq, usable_w, chart_h))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_fig_to_image(mod_figs.rank, usable_w, chart_h))
        flowables.append(PageBreak())

        # --- Page 3: temperature + method z-score overview ---
        flowables.append(Paragraph(f"Module M{mid} — Temperature &amp; Method Overview", styles["h2"]))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(_fig_to_image(mod_figs.temperature, usable_w, chart_h))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_fig_to_image(mod_figs.zscore_heatmap, usable_w, chart_h))
        flowables.append(PageBreak())

        # --- Page 4: per-cell composite + cluster scores ---
        flowables.append(Paragraph(
            f"Module M{mid} — Per-Cell Composite &amp; Cluster Scores",
            styles["h2"],
        ))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_module_ztable(mv))
        flowables.append(PageBreak())

    return flowables


def _module_ztable(mv) -> Table:
    """Compact per-cell table: verdict, composite, and the 5 cluster scores."""
    from stress_screen.analysis.aggregate import CLUSTERS

    cluster_names = list(CLUSTERS)
    header = ["Cell", "Verdict", "Composite Z"] + [
        c.replace("_", " ") for c in cluster_names
    ]
    data = [header]
    row_styles: list[tuple] = []
    for idx, cv in enumerate(sorted(mv.all_cells, key=lambda c: c.group_in_module)):
        row_idx = idx + 1
        scores = cv.cluster_scores or {}
        data.append([
            cv.label,
            cv.verdict,
            f"{cv.composite_z:.2f}",
            *[f"{scores[c]:.2f}" if c in scores else "—" for c in cluster_names],
        ])
        if cv.verdict == "HIGH":
            row_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                               colors.HexColor("#f8d7da")))
        elif cv.verdict == "ELEVATED":
            row_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                               colors.HexColor("#fff3cd")))

    tbl = Table(data, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    for cmd in row_styles:
        style.add(*cmd)
    tbl.setStyle(style)
    return tbl


def _param_table(rows) -> Table:
    """Sectioned physical-parameters table from findings.ParamRow list."""
    data: list[list] = [["Physical parameter", "Value", "z"]]
    group_rows: list[int] = []
    prev_group = None
    for row in rows:
        if row.group != prev_group:
            data.append([row.group, "", ""])
            group_rows.append(len(data) - 1)
            prev_group = row.group
        data.append([f"  {row.label}", row.value_str, row.z_str])

    tbl = Table(data, colWidths=[9.5 * cm, 4.5 * cm, 2.5 * cm], repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ])
    for r in group_rows:
        style.add("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e9ecef"))
        style.add("FONTNAME", (0, r), (-1, r), "Helvetica-Bold")
        style.add("SPAN", (0, r), (-1, r))
    tbl.setStyle(style)
    return tbl


def _pages_flagged_cells(
    result: AnalysisResult,
    figures: FigureSet,
    findings,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
) -> list:
    """One detail page per flagged cell: finding sentence, detail-card chart,
    and the full physical-parameters table."""
    flowables: list = []
    usable_w = page_w - 2 * margin

    for f in findings.cell_findings:
        color = "#cc0000" if f.verdict == "HIGH" else "#b8860b"
        flowables.append(Paragraph(
            f'Flagged Cell {f.label} — '
            f'<font color="{color}">{f.verdict}</font> '
            f'(composite z = {f.composite_z:.2f})',
            styles["h1"],
        ))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(Paragraph(f.sentence, styles["finding"]))
        flowables.append(Spacer(1, 0.3 * cm))

        detail_fig = figures.flagged_cell_details.get(f.channel_index)
        if detail_fig is not None:
            flowables.append(_fig_to_image(detail_fig, usable_w, 6.0 * cm))
            flowables.append(Spacer(1, 0.4 * cm))

        flowables.append(_param_table(f.param_rows))
        flowables.append(PageBreak())

    return flowables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_pdf_report(
    result: AnalysisResult,
    rest_cell_df: pd.DataFrame,
    charge_cell_df: pd.DataFrame,
    top_df: pd.DataFrame,
    out_path: Path,
    top_charge_df: "pd.DataFrame | None" = None,
    n_parallel: int = 1,
    figures: "FigureSet | None" = None,
    findings=None,
) -> None:
    """Write a PDF report to *out_path*.

    Parameters
    ----------
    result:
        Full analysis result produced by ``analysis.aggregate``.
    rest_cell_df:
        Long-format cell DataFrame restricted to the rest phase.
    charge_cell_df:
        Long-format cell DataFrame restricted to the charge phase.
    top_df:
        Pack-level (top-level) DataFrame with ``time_hours`` and ``current``
        columns, covering the full test duration.
    out_path:
        Destination path; parent directory must exist.
    figures:
        Pre-built figure set (shared with the HTML writer). Built on demand
        when None.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if figures is None:
        figures = build_figures(
            result, rest_cell_df, charge_cell_df, top_df,
            top_charge_df=top_charge_df, n_parallel=n_parallel,
        )
    if findings is None:
        from stress_screen.reports.findings import build_findings
        findings = build_findings(result)

    styles = _make_styles()

    # A4 portrait — use landscape for charts to maximise width
    PAGE_W, PAGE_H = A4          # 595.27 × 841.89 pt
    MARGIN = 1.5 * cm

    # Build a single-column document with full-page frames
    frame = Frame(
        MARGIN, MARGIN,
        PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )
    page_template = PageTemplate(id="main", frames=[frame])

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        pageTemplates=[page_template],
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    story: list = []

    # Page 1 — executive summary (verdict + factual findings)
    story += _page_exec_summary(result, findings, styles)

    # Page 2 — module summary table + pack heatmap
    story += _page2_module_table(result, styles)
    story += _page_pack_heatmap(figures.pack_heatmap, result,
                                PAGE_W, PAGE_H, MARGIN, styles)

    # Page 3 — methodology (shared text with the HTML report)
    story += _page_methodology(styles)

    # Page 4 — phase timeline
    story += _page_phase_timeline(figures.phase_timeline, PAGE_W, PAGE_H, MARGIN, styles)

    # Pages 5+ — per-module charts + per-cell cluster tables
    story += _pages_per_module(result, figures, PAGE_W, PAGE_H, MARGIN, styles)

    # Final pages — one detail page per flagged cell
    story += _pages_flagged_cells(result, figures, findings,
                                  PAGE_W, PAGE_H, MARGIN, styles)

    doc.build(story)
