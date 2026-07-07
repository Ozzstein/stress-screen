"""
reports/findings.py — structured, strictly factual findings for reports.

Turns an :class:`AnalysisResult` into display-ready findings shared by the
HTML and PDF writers: a pack headline, one factual sentence per flagged cell
(measured and fitted values only — no failure-mode speculation), and an
exhaustive physical-parameters table per flagged cell built from the method
metadata that the analysis pipeline already computes.

Units note: the OCV rest model is ``V(t) = V_ocv + a·exp(−t/τ) − k·t``
(analysis/util.py), so the fitted ``k`` is a linear voltage-decay slope in
V/h — displayed as mV/h by a direct ×1000 conversion.

No I/O and no Plotly here; pure data preparation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from stress_screen.analysis.aggregate import CLUSTERS
from stress_screen.models import AnalysisResult, CellVerdict, MethodResult

#: Cluster score at/above this counts as "above gate" (method z threshold).
GATE = 2.0

#: Clusters scoring below this are omitted from the finding sentence
#: (the top-scoring cluster is always included).
SENTENCE_MIN_SCORE = 0.5


# ---------------------------------------------------------------------------
# Shared methodology text (single source for HTML + PDF, so they can't drift)
# ---------------------------------------------------------------------------

#: method name → (one-line description, what it detects)
METHOD_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "ocv_k": (
        "Fits each cell's rest voltage to V(t) = V_ocv + a·exp(−t/τ) − k·t "
        "and extracts the linear self-discharge decay slope k (mV/h).",
        "Slow voltage decay during rest — primary capacity-loss / leakage signature.",
    ),
    "thermal_corr": (
        "Pearson correlation between the OCV-fit residuals and cell temperature.",
        "Cells whose voltage tracks heat — internal-resistance imbalance.",
    ),
    "spread": (
        "Slope of |V_cell − V_module_median| over rest, temperature-compensated.",
        "Cells diverging from the fleet over time.",
    ),
    "cusum": (
        "Two-sided CUSUM alarms on the OCV-fit residuals.",
        "Step changes / persistent biases the decay fit would miss.",
    ),
    "temp_k": (
        "The ocv_k decay slope Arrhenius-normalized to 25 °C.",
        "Excess self-discharge after removing temperature bias.",
    ),
    "rank": (
        "Voltage-rank percentile of the cell within its module over rest.",
        "Cells sliding down the within-module ranking.",
    ),
    "li_plating": (
        "dQ/dV extra-peak detection above the main plateau, post-charge "
        "relaxation speed, charge-time anomaly, and two temperature "
        "signatures, gated by an Arrhenius cold-temperature likelihood.",
        "Li-metal plating on the anode (cold/fast-charge induced).",
    ),
    "isc": (
        "Three sub-signals: S1 excess self-discharge above the module "
        "threshold, S2 rest-phase thermal slope above the module median, "
        "S3 dV/dQ area deficit during charge (temperature-gated).",
        "Internal short-circuit paths — simultaneous excess self-discharge, "
        "localised heating, and reduced charge acceptance.",
    ),
}

VERDICT_RULES_TEXT = (
    "The eight methods group into five evidence clusters — self_discharge "
    "(ocv_k, temp_k), residual_dynamics (thermal_corr, cusum), "
    "fleet_divergence (spread, rank), li_plating, and isc — because cluster "
    "members are redundant readings of the same physical signal. Member "
    "z-scores are averaged within their cluster, each cluster score is "
    "winsorized to ±8, and the cell composite is the weighted mean of the "
    "cluster scores. A cell is HIGH when composite_z > 2.0 or "
    "(≥2 clusters ≥ 2.0 and composite_z ≥ 1.0); ELEVATED when "
    "composite_z > 1.0 or (≥1 cluster ≥ 2.0 and composite_z ≥ 0.5); "
    "otherwise NORMAL. The module verdict is binary: NOK if any cell is "
    "HIGH or ELEVATED, else OK."
)


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def fmt_num(v: Any, nd: int = 2, unit: str = "", scale: float = 1.0) -> str:
    """Format a numeric value with unit, or an em-dash for missing/NaN.

    Every value displayed in the reports goes through this function so that
    missing data always renders as "—" rather than "nan" or "None".
    """
    if _is_missing(v):
        return "—"
    x = float(v) * scale
    if x != 0 and abs(x) < 10 ** (-nd):
        s = f"{x:.1e}"
    else:
        s = f"{x:.{nd}f}"
    return f"{s} {unit}".strip()


def k_to_mv_per_h(k_v_per_h: Any) -> str:
    """Fitted decay slope k (V/h) → display string in mV/h."""
    return fmt_num(k_v_per_h, nd=2, unit="mV/h", scale=1000.0)


# ---------------------------------------------------------------------------
# Structured findings
# ---------------------------------------------------------------------------

@dataclass
class ParamRow:
    """One row of the physical-parameters table."""

    group: str       # section header, e.g. "Self-discharge (rest)"
    label: str       # e.g. "Decay slope k"
    value_str: str   # e.g. "0.92 mV/h"
    z_str: str = ""  # associated z-score when one exists


@dataclass
class ClusterSummary:
    name: str
    score: float          # NaN-safe; float("nan") when absent
    above_gate: bool
    detail: str           # factual fragment ("" when nothing to report)


@dataclass
class CellFinding:
    channel_index: int
    label: str
    module_id: int
    verdict: str
    composite_z: float
    composite_z_legacy: float | None
    n_methods_high: int
    cluster_summaries: list[ClusterSummary] = field(default_factory=list)
    sentence: str = ""
    param_rows: list[ParamRow] = field(default_factory=list)


@dataclass
class PackFindings:
    overall_verdict: str
    headline: str
    rest_duration_h: float | None
    n_cells: int
    cell_findings: list[CellFinding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metadata access helpers
# ---------------------------------------------------------------------------

def _methods_by_name(cv: CellVerdict) -> dict[str, MethodResult]:
    return {mr.method_name: mr for mr in cv.method_results}


def _meta(methods: dict[str, MethodResult], method: str, key: str) -> Any:
    mr = methods.get(method)
    return mr.metadata.get(key) if mr is not None else None


def _z(methods: dict[str, MethodResult], method: str) -> Any:
    mr = methods.get(method)
    return mr.z_score if mr is not None else None


# ---------------------------------------------------------------------------
# Cluster detail fragments (strictly factual — measured values only)
# ---------------------------------------------------------------------------

def _detail_self_discharge(m: dict[str, MethodResult]) -> str:
    parts = []
    k = _meta(m, "ocv_k", "k")
    if not _is_missing(k):
        frag = f"k = {k_to_mv_per_h(k)}"
        kc = _meta(m, "temp_k", "k_corrected")
        t_mean = _meta(m, "temp_k", "T_mean")
        if not _is_missing(kc):
            frag += f" ({k_to_mv_per_h(kc)} at 25 °C"
            if not _is_missing(t_mean):
                frag += f", mean T = {fmt_num(t_mean, 1, '°C')}"
            frag += ")"
        parts.append(frag)
    return "; ".join(parts)


def _detail_residual_dynamics(m: dict[str, MethodResult]) -> str:
    parts = []
    n_alarms = _meta(m, "cusum", "n_alarms")
    if not _is_missing(n_alarms) and float(n_alarms) > 0:
        frag = f"CUSUM {int(n_alarms)} alarm(s)"
        first = _meta(m, "cusum", "first_alarm_h")
        if not _is_missing(first):
            frag += f", first at {fmt_num(first, 1, 'h')}"
        parts.append(frag)
    r = _meta(m, "thermal_corr", "pearson_r")
    if not _is_missing(r):
        parts.append(f"residual–T correlation r = {fmt_num(r, 2)}")
    return "; ".join(parts)


def _detail_fleet_divergence(
    m: dict[str, MethodResult], rest_duration_h: float | None
) -> str:
    parts = []
    slope = _meta(m, "spread", "divergence_slope_v_per_h")
    if not _is_missing(slope):
        frag = f"|V − module median| slope {k_to_mv_per_h(slope)}"
        if rest_duration_h:
            frag += f" over {fmt_num(rest_duration_h, 0, 'h')}"
        parts.append(frag)
    mean_rank = _meta(m, "rank", "mean_rank_pct")
    if not _is_missing(mean_rank):
        frag = f"mean rank {fmt_num(mean_rank, 0)}th pct"
        frac = _meta(m, "rank", "frac_bot20")
        if not _is_missing(frac):
            frag += f", {fmt_num(frac, 0, '%', scale=100)} of samples in bottom 20 %"
        parts.append(frag)
    return "; ".join(parts)


def _detail_li_plating(m: dict[str, MethodResult]) -> str:
    parts = []
    peak_v = _meta(m, "li_plating", "dqdv_extra_peak_voltage")
    gated_dv = _meta(m, "li_plating", "gated_dqdv_z")
    if not _is_missing(peak_v):
        frag = f"extra dQ/dV peak at {fmt_num(peak_v, 2, 'V')}"
        if not _is_missing(gated_dv):
            frag += f" (gated z = {fmt_num(gated_dv, 1)})"
        parts.append(frag)
    tau_inv = _meta(m, "li_plating", "tau_inv")
    if not _is_missing(tau_inv):
        parts.append(f"relaxation 1/τ = {fmt_num(tau_inv, 2, '1/h')}")
    gate = _meta(m, "li_plating", "temperature_gate")
    t_charge = _meta(m, "li_plating", "T_mean_charge")
    if not _is_missing(gate):
        frag = f"T-gate {fmt_num(gate, 2)}"
        if not _is_missing(t_charge):
            frag += f" (mean charge T = {fmt_num(t_charge, 1, '°C')})"
        parts.append(frag)
    return "; ".join(parts)


def _detail_isc(m: dict[str, MethodResult]) -> str:
    parts = []
    s1 = _meta(m, "isc", "s1_excess_k")
    s1z = _meta(m, "isc", "s1_excess_k_z")
    if not _is_missing(s1):
        frag = f"S1 excess k = {k_to_mv_per_h(s1)}"
        if not _is_missing(s1z):
            frag += f" (z {fmt_num(s1z, 1)})"
        parts.append(frag)
    s2 = _meta(m, "isc", "s2_dT_dt_slope")
    s2z = _meta(m, "isc", "s2_dT_dt_z")
    if not _is_missing(s2):
        frag = f"S2 dT/dt = {fmt_num(s2, 3, '°C/h')}"
        if not _is_missing(s2z):
            frag += f" (z {fmt_num(s2z, 1)})"
        parts.append(frag)
    s3z = _meta(m, "isc", "s3_area_deficit_z")
    gate = _meta(m, "isc", "s3_temperature_gate")
    if not _is_missing(s3z):
        frag = f"S3 area z {fmt_num(s3z, 1)}"
        if not _is_missing(gate):
            frag += f" (gate {fmt_num(gate, 2)})"
        parts.append(frag)
    return "; ".join(parts)


_DETAIL_BUILDERS = {
    "self_discharge": lambda m, rest_h: _detail_self_discharge(m),
    "residual_dynamics": lambda m, rest_h: _detail_residual_dynamics(m),
    "fleet_divergence": _detail_fleet_divergence,
    "li_plating": lambda m, rest_h: _detail_li_plating(m),
    "isc": lambda m, rest_h: _detail_isc(m),
}


# ---------------------------------------------------------------------------
# Physical-parameters table
# ---------------------------------------------------------------------------

def _param_rows(m: dict[str, MethodResult]) -> list[ParamRow]:
    """Exhaustive grouped physical-parameters table for one cell."""

    def row(group: str, label: str, value: str, z: Any = None) -> ParamRow:
        return ParamRow(group, label, value,
                        fmt_num(z, 2) if z is not None else "")

    rows: list[ParamRow] = []

    g = "Self-discharge (rest)"
    rows.append(row(g, "Decay slope k", k_to_mv_per_h(_meta(m, "ocv_k", "k")),
                    _z(m, "ocv_k")))
    rows.append(row(g, "k at 25 °C (Arrhenius)",
                    k_to_mv_per_h(_meta(m, "temp_k", "k_corrected")),
                    _z(m, "temp_k")))
    rows.append(row(g, "Mean rest temperature",
                    fmt_num(_meta(m, "temp_k", "T_mean"), 1, "°C")))
    rows.append(row(g, "Relaxation time constant τ",
                    fmt_num(_meta(m, "ocv_k", "tau"), 2, "h")))
    rows.append(row(g, "Fitted OCV plateau V_ocv",
                    fmt_num(_meta(m, "ocv_k", "V_ocv"), 4, "V")))

    g = "Residual dynamics (rest)"
    rows.append(row(g, "Residual–temperature correlation r",
                    fmt_num(_meta(m, "thermal_corr", "pearson_r"), 3),
                    _z(m, "thermal_corr")))
    rows.append(row(g, "CUSUM alarms",
                    fmt_num(_meta(m, "cusum", "n_alarms"), 0),
                    _z(m, "cusum")))
    rows.append(row(g, "First CUSUM alarm",
                    fmt_num(_meta(m, "cusum", "first_alarm_h"), 1, "h")))

    g = "Fleet divergence (rest)"
    rows.append(row(g, "Divergence slope (T-compensated)",
                    k_to_mv_per_h(_meta(m, "spread", "divergence_slope_v_per_h")),
                    _z(m, "spread")))
    rows.append(row(g, "Mean rank percentile",
                    fmt_num(_meta(m, "rank", "mean_rank_pct"), 1, "%"),
                    _z(m, "rank")))
    rows.append(row(g, "Fraction of samples in bottom 20 %",
                    fmt_num(_meta(m, "rank", "frac_bot20"), 0, "%", scale=100)))
    rows.append(row(g, "Rank slope",
                    fmt_num(_meta(m, "rank", "rank_slope"), 3, "%/h")))

    g = "Li-plating (charge + relaxation)"
    rows.append(row(g, "dQ/dV extra-peak prominence sum",
                    fmt_num(_meta(m, "li_plating", "dqdv_extra_peak_sum"), 3),
                    _meta(m, "li_plating", "dqdv_z")))
    rows.append(row(g, "dQ/dV extra-peak voltage",
                    fmt_num(_meta(m, "li_plating", "dqdv_extra_peak_voltage"), 3, "V")))
    rows.append(row(g, "Post-charge relaxation 1/τ",
                    fmt_num(_meta(m, "li_plating", "tau_inv"), 2, "1/h"),
                    _meta(m, "li_plating", "relaxation_z")))
    rows.append(row(g, "Charge duration",
                    fmt_num(_meta(m, "li_plating", "charge_duration_h"), 2, "h"),
                    _meta(m, "li_plating", "charge_time_z")))
    rows.append(row(g, "Mean charge temperature",
                    fmt_num(_meta(m, "li_plating", "T_mean_charge"), 1, "°C"),
                    _meta(m, "li_plating", "cold_z")))
    rows.append(row(g, "Late-charge delta-T",
                    fmt_num(_meta(m, "li_plating", "dT_late"), 2, "°C"),
                    _meta(m, "li_plating", "heat_z")))
    rows.append(row(g, "Cold-temperature gate",
                    fmt_num(_meta(m, "li_plating", "temperature_gate"), 2)))
    rows.append(row(g, "Gated dQ/dV z",
                    fmt_num(_meta(m, "li_plating", "gated_dqdv_z"), 2)))
    rows.append(row(g, "Gated relaxation z",
                    fmt_num(_meta(m, "li_plating", "gated_relaxation_z"), 2)))
    rows.append(row(g, "Gated charge-time z",
                    fmt_num(_meta(m, "li_plating", "gated_charge_time_z"), 2)))

    g = "Internal short circuit"
    rows.append(row(g, "S1 excess self-discharge",
                    k_to_mv_per_h(_meta(m, "isc", "s1_excess_k")),
                    _meta(m, "isc", "s1_excess_k_z")))
    rows.append(row(g, "S1 k corrected (ISC Ea)",
                    k_to_mv_per_h(_meta(m, "isc", "s1_k_corrected_isc"))))
    rows.append(row(g, "S2 rest dT/dt (vs module median)",
                    fmt_num(_meta(m, "isc", "s2_dT_dt_slope"), 3, "°C/h"),
                    _meta(m, "isc", "s2_dT_dt_z")))
    rows.append(row(g, "S2 rest dT/dt (raw)",
                    fmt_num(_meta(m, "isc", "s2_dT_dt_raw_slope"), 3, "°C/h")))
    rows.append(row(g, "S3 dV/dQ area",
                    fmt_num(_meta(m, "isc", "s3_dvdq_area"), 3, "V"),
                    _meta(m, "isc", "s3_area_deficit_z")))
    rows.append(row(g, "S3 temperature gate",
                    fmt_num(_meta(m, "isc", "s3_temperature_gate"), 2)))

    return rows


# ---------------------------------------------------------------------------
# Sentence + findings assembly
# ---------------------------------------------------------------------------

def _cluster_summaries(
    cv: CellVerdict, methods: dict[str, MethodResult],
    rest_duration_h: float | None,
) -> list[ClusterSummary]:
    scores = cv.cluster_scores or {}
    summaries: list[ClusterSummary] = []
    for name in CLUSTERS:
        score = scores.get(name)
        if score is None or _is_missing(score):
            continue
        builder = _DETAIL_BUILDERS.get(name)
        detail = builder(methods, rest_duration_h) if builder else ""
        summaries.append(ClusterSummary(
            name=name,
            score=float(score),
            above_gate=float(score) >= GATE,
            detail=detail,
        ))
    summaries.sort(key=lambda s: s.score, reverse=True)
    return summaries


def _sentence(cv: CellVerdict, summaries: list[ClusterSummary]) -> str:
    lead = f"{cv.label} {cv.verdict} (composite {fmt_num(cv.composite_z, 2)})"
    fragments: list[str] = []
    for i, s in enumerate(summaries):
        if i > 0 and s.score < SENTENCE_MIN_SCORE:
            continue
        gate_note = "above gate" if s.above_gate else "below gate"
        frag = f"{s.name} {fmt_num(s.score, 1)}σ {gate_note}"
        if s.detail:
            frag += f" — {s.detail}"
        fragments.append(frag)
    if not fragments:
        return f"{lead}."
    return f"{lead}: " + "; ".join(fragments) + "."


def build_findings(result: AnalysisResult) -> PackFindings:
    """Build the display-ready findings for one analysis result."""
    rest_h = max(
        (s.duration_h for s in result.segments if s.phase == "rest"),
        default=None,
    )

    cell_findings: list[CellFinding] = []
    n_cells = result.topology.active_channels
    for mv in result.module_verdicts:
        for cv in mv.flagged_cells:
            methods = _methods_by_name(cv)
            summaries = _cluster_summaries(cv, methods, rest_h)
            cell_findings.append(CellFinding(
                channel_index=int(cv.channel_index),
                label=cv.label,
                module_id=mv.module_id,
                verdict=cv.verdict,
                composite_z=cv.composite_z,
                composite_z_legacy=cv.composite_z_legacy,
                n_methods_high=cv.n_methods_high,
                cluster_summaries=summaries,
                sentence=_sentence(cv, summaries),
                param_rows=_param_rows(methods),
            ))

    cell_findings.sort(key=lambda f: f.composite_z, reverse=True)

    overall = "NOK" if result.any_nok else "OK"
    nok_modules = [f"M{mv.module_id}" for mv in result.module_verdicts
                   if mv.verdict == "NOK"]
    if cell_findings:
        flagged_desc = ", ".join(
            f"{f.label} ({f.verdict})" for f in cell_findings
        )
        headline = (
            f"Pack {overall} — {len(cell_findings)} of {n_cells} cell-groups "
            f"flagged ({flagged_desc}); modules {', '.join(nok_modules)} NOK."
        )
    else:
        headline = f"Pack OK — all {n_cells} cell-groups NORMAL."

    return PackFindings(
        overall_verdict=overall,
        headline=headline,
        rest_duration_h=rest_h,
        n_cells=n_cells,
        cell_findings=cell_findings,
    )
