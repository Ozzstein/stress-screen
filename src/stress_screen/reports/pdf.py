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
from stress_screen.reports.charts import (
    divergence_chart,
    dv_dq_chart,
    method_zscore_heatmap,
    ocv_fit_overlay,
    pack_heatmap,
    phase_timeline,
    rank_chart,
    temperature_chart,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig_to_image(fig, width_pt: float, height_pt: float) -> Image:
    """Render a Plotly figure to a ReportLab Image at the given pt dimensions."""
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
    styles["verdict_marginal"] = ParagraphStyle(
        "VerdictMarginal",
        parent=base["Normal"],
        fontSize=36,
        textColor=colors.HexColor("#b8860b"),
        spaceAfter=12,
        alignment=1,  # center
    )
    styles["center"] = ParagraphStyle(
        "CenterStyle",
        parent=base["Normal"],
        alignment=1,
    )
    return styles


# ---------------------------------------------------------------------------
# Page builders (return lists of Flowables)
# ---------------------------------------------------------------------------

def _page1_title(result: AnalysisResult, styles: dict) -> list:
    """Title page flowables."""
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

    # Verdict summary
    nok_modules = [m for m in result.module_verdicts if m.verdict == "NOK"]
    marginal_modules = [m for m in result.module_verdicts if m.verdict == "MARGINAL"]
    if nok_modules:
        nok_ids = ", ".join(f"M{m.module_id}" for m in nok_modules)
        verdict_text = f"NOK: {nok_ids}"
        verdict_style = styles["verdict_nok"]
    elif marginal_modules:
        from stress_screen.models import MARGINAL_DISPLAY
        marginal_ids = ", ".join(f"M{m.module_id}" for m in marginal_modules)
        verdict_text = f"{MARGINAL_DISPLAY}: {marginal_ids}"
        verdict_style = styles["verdict_marginal"]
    else:
        verdict_text = "ALL OK"
        verdict_style = styles["verdict_ok"]

    flowables = [
        Spacer(1, 3 * cm),
        Paragraph("Stress Screen Report", styles["title"]),
        Spacer(1, 0.5 * cm),
        Paragraph(f"<b>Pack ID:</b> {pack_id}", styles["normal"]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"<b>Configuration:</b> {config_str}", styles["normal"]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"<b>Test date:</b> {test_date}", styles["normal"]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"<b>Report generated:</b> {generated_at}", styles["normal"]),
        Spacer(1, 2 * cm),
        Paragraph(verdict_text, verdict_style),
    ]

    # Brief per-module summary lines below the verdict
    flowables.append(Spacer(1, 1 * cm))
    for mv in result.module_verdicts:
        if mv.verdict == "OK":
            color = "#1a7a1a"
        elif mv.verdict == "MARGINAL":
            color = "#b8860b"
        else:
            color = "#cc0000"
        flowables.append(
            Paragraph(
                f'<font color="{color}">{mv.summary_line}</font>',
                styles["normal"],
            )
        )

    flowables.append(PageBreak())
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

        from stress_screen.models import MARGINAL_DISPLAY
        verdict_label = MARGINAL_DISPLAY if mv.verdict == "MARGINAL" else mv.verdict
        data.append([
            f"M{mv.module_id}",
            verdict_label,
            flagged,
            methods_str,
        ])

        if mv.verdict == "OK":
            bg = colors.HexColor("#d4edda")
        elif mv.verdict == "MARGINAL":
            bg = colors.HexColor("#fff3cd")
        else:
            bg = colors.HexColor("#f8d7da")
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
        PageBreak(),
    ]
    return flowables


def _page_pack_heatmap(
    result: AnalysisResult,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
) -> list:
    """Full-page pack heatmap."""
    fig = pack_heatmap(result)
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - 2 * cm  # leave room for heading

    img = _fig_to_image(fig, usable_w, usable_h)
    return [
        Paragraph("Pack Overview — Composite Z-Score Heatmap", styles["h1"]),
        Spacer(1, 0.3 * cm),
        img,
        PageBreak(),
    ]


def _page_phase_timeline(
    result: AnalysisResult,
    top_df: pd.DataFrame,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
) -> list:
    """Full-page phase timeline."""
    fig = phase_timeline(top_df, result.segments)
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
    rest_cell_df: pd.DataFrame,
    charge_cell_df: pd.DataFrame,
    page_w: float,
    page_h: float,
    margin: float,
    styles: dict,
    top_charge_df: "pd.DataFrame | None" = None,
    n_parallel: int = 1,
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

        # --- Page 1: rest-phase voltage charts ---
        flowables.append(Paragraph(f"Module M{mid} — Rest Phase Analysis", styles["h2"]))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(_fig_to_image(ocv_fit_overlay(result, mid, rest_cell_df), usable_w, chart_h))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_fig_to_image(divergence_chart(result, mid, rest_cell_df), usable_w, chart_h))
        flowables.append(PageBreak())

        # --- Page 2: charge-phase + rank charts ---
        flowables.append(Paragraph(f"Module M{mid} — Charge Phase &amp; Rank Analysis", styles["h2"]))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(_fig_to_image(
            dv_dq_chart(result, mid, charge_cell_df, top_charge_df=top_charge_df, n_parallel=n_parallel),
            usable_w, chart_h,
        ))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_fig_to_image(rank_chart(result, mid, rest_cell_df), usable_w, chart_h))
        flowables.append(PageBreak())

        # --- Page 3: temperature + method z-score overview ---
        flowables.append(Paragraph(f"Module M{mid} — Temperature &amp; Method Overview", styles["h2"]))
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(_fig_to_image(
            temperature_chart(result, mid, rest_cell_df, charge_cell_df), usable_w, chart_h,
        ))
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(_fig_to_image(method_zscore_heatmap(result, mid), usable_w, chart_h))
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
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Page 1 — title
    story += _page1_title(result, styles)

    # Page 2 — module summary table
    story += _page2_module_table(result, styles)

    # Page 3 — pack heatmap
    story += _page_pack_heatmap(result, PAGE_W, PAGE_H, MARGIN, styles)

    # Page 4 — phase timeline
    story += _page_phase_timeline(result, top_df, PAGE_W, PAGE_H, MARGIN, styles)

    # Pages 5+ — per-module charts
    story += _pages_per_module(
        result, rest_cell_df, charge_cell_df,
        PAGE_W, PAGE_H, MARGIN, styles,
        top_charge_df=top_charge_df, n_parallel=n_parallel,
    )

    doc.build(story)
