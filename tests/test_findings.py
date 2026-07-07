"""Unit tests for reports/findings.py — factual sentences and param tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from stress_screen.models import (
    AnalysisResult, CellVerdict, MethodResult, ModuleVerdict, PackTopology,
    Segment,
)
from stress_screen.reports.findings import (
    build_findings, fmt_num, k_to_mv_per_h,
)

NAN = float("nan")

#: Words that must never appear in generated findings (speculation ban).
FORBIDDEN = ("suspect", "likely", "probable", "failure", "defect", "broken")


def _full_metadata_cell(verdict="HIGH", composite=2.6) -> CellVerdict:
    mrs = [
        MethodResult("ocv_k", 3.1, "HIGH",
                     {"k": 1.1e-3, "V_ocv": 3.3312, "tau": 1.3}),
        MethodResult("thermal_corr", 0.4, "NORMAL", {"pearson_r": 0.71}),
        MethodResult("spread", 2.8, "HIGH",
                     {"divergence_slope_v_per_h": 5.8e-5}),
        MethodResult("cusum", 2.2, "HIGH",
                     {"n_alarms": 3, "first_alarm_h": 6.2}),
        MethodResult("temp_k", 3.0, "HIGH",
                     {"k_corrected": 0.9e-3, "T_mean": 27.3,
                      "temp_correction_applied": True}),
        MethodResult("rank", 2.4, "HIGH",
                     {"mean_rank_pct": 12.0, "frac_bot20": 0.84,
                      "rank_slope": -0.5, "slope_contribution_raw": 2.0,
                      "slope_contribution_capped": 1.0}),
        MethodResult("li_plating", 1.1, "ELEVATED",
                     {"dqdv_z": 2.3, "relaxation_z": 0.5, "charge_time_z": 0.1,
                      "dqdv_extra_peak_sum": 0.12,
                      "dqdv_extra_peak_voltage": 3.42, "tau_inv": 1.8,
                      "charge_duration_h": 2.1, "cold_z": 0.9, "heat_z": 0.2,
                      "T_mean_charge": 8.4, "dT_late": 0.6,
                      "temperature_gate": 0.6, "gated_dqdv_z": 2.3,
                      "gated_relaxation_z": 0.3, "gated_charge_time_z": 0.06}),
        MethodResult("isc", 0.8, "NORMAL",
                     {"s1_excess_k_z": 1.1, "s1_excess_k": 4.0e-4,
                      "s1_k_corrected_isc": 1.0e-3,
                      "s2_dT_dt_z": 0.7, "s2_dT_dt_slope": 0.04,
                      "s2_dT_dt_raw_slope": 0.05,
                      "s3_area_deficit_z": 0.2, "s3_dvdq_area": 0.31,
                      "s3_temperature_gate": 1.0}),
    ]
    return CellVerdict(
        channel_index=6, module_id=1, group_in_module=7,
        composite_z=composite, n_methods_high=2, verdict=verdict,
        method_results=mrs,
        cluster_scores={"self_discharge": 3.05, "residual_dynamics": 1.3,
                        "fleet_divergence": 2.6, "li_plating": 1.1,
                        "isc": 0.8},
        composite_z_legacy=2.1,
    )


def _nan_metadata_cell() -> CellVerdict:
    mrs = [
        MethodResult(name, NAN, "NORMAL", {k: NAN for k in keys})
        for name, keys in [
            ("ocv_k", ["k", "V_ocv", "tau"]),
            ("thermal_corr", ["pearson_r"]),
            ("spread", ["divergence_slope_v_per_h"]),
            ("cusum", ["n_alarms", "first_alarm_h"]),
            ("temp_k", ["k_corrected", "T_mean"]),
            ("rank", ["mean_rank_pct", "frac_bot20", "rank_slope"]),
            ("li_plating", ["tau_inv", "temperature_gate"]),
            ("isc", ["s1_excess_k", "s2_dT_dt_slope"]),
        ]
    ]
    return CellVerdict(
        channel_index=0, module_id=1, group_in_module=1,
        composite_z=1.2, n_methods_high=1, verdict="ELEVATED",
        method_results=mrs,
        cluster_scores={"self_discharge": 2.2},
        composite_z_legacy=None,
    )


def _result_with(cells: list[CellVerdict]) -> AnalysisResult:
    topo = PackTopology(module_count=1, series=8, parallel=4,
                        config_name="4P8S", active_channels=8)
    all_cells = list(cells)
    flagged = [c for c in cells if c.verdict in ("HIGH", "ELEVATED")]
    mv = ModuleVerdict(module_id=1, verdict="NOK" if flagged else "OK",
                       flagged_cells=flagged, all_cells=all_cells)
    return AnalysisResult(
        csv_path=Path("Synth_D01032026_M1.csv"),
        topology=topo,
        segments=[Segment("charge", 0.0, 3.0, 0, 10),
                  Segment("rest", 3.0, 51.0, 11, 100)],
        module_verdicts=[mv],
    )


def test_fmt_num_handles_missing():
    assert fmt_num(None) == "—"
    assert fmt_num(NAN) == "—"
    assert fmt_num(0.923, 2, "mV/h") == "0.92 mV/h"
    assert fmt_num(1.1e-3, 2, "mV/h", scale=1000) == "1.10 mV/h"
    assert fmt_num(0.0) == "0.00"


def test_k_conversion():
    assert k_to_mv_per_h(1.1e-3) == "1.10 mV/h"
    assert k_to_mv_per_h(None) == "—"


def test_sentence_contains_facts():
    findings = build_findings(_result_with([_full_metadata_cell()]))
    assert len(findings.cell_findings) == 1
    s = findings.cell_findings[0].sentence
    assert s.startswith("M1/G7 HIGH (composite 2.60)")
    assert "self_discharge" in s and "above gate" in s
    assert "1.10 mV/h" in s          # k converted to physical units
    assert "0.90 mV/h at 25 °C" in s
    assert "CUSUM 3 alarm(s), first at 6.2 h" in s
    assert "bottom 20 %" in s
    # top cluster first
    assert s.index("self_discharge") < s.index("fleet_divergence")


def test_sentence_never_speculates():
    findings = build_findings(_result_with([_full_metadata_cell()]))
    text = " ".join(f.sentence for f in findings.cell_findings).lower()
    for word in FORBIDDEN:
        assert word not in text, f"speculative word {word!r} in findings"


def test_nan_metadata_renders_dashes_not_nan():
    findings = build_findings(_result_with([_nan_metadata_cell()]))
    f = findings.cell_findings[0]
    assert "nan" not in f.sentence.lower()
    assert "None" not in f.sentence
    for row in f.param_rows:
        assert "nan" not in row.value_str.lower()
        assert "None" not in row.value_str
    # all-NaN metadata → all values are dashes
    assert all(r.value_str == "—" for r in f.param_rows)


def test_param_rows_cover_all_groups_and_fields():
    findings = build_findings(_result_with([_full_metadata_cell()]))
    rows = findings.cell_findings[0].param_rows
    groups = {r.group for r in rows}
    assert groups == {
        "Self-discharge (rest)", "Residual dynamics (rest)",
        "Fleet divergence (rest)", "Li-plating (charge + relaxation)",
        "Internal short circuit",
    }
    labels = [r.label for r in rows]
    # li_plating richness: gates + gated z's + both temperature signatures
    assert "Cold-temperature gate" in labels
    assert "Late-charge delta-T" in labels
    assert "Gated dQ/dV z" in labels
    # all ISC sub-signals incl. the previously hidden ones
    assert "S1 k corrected (ISC Ea)" in labels
    assert "S2 rest dT/dt (raw)" in labels
    assert "S3 temperature gate" in labels
    by_label = {r.label: r for r in rows}
    assert by_label["Decay slope k"].value_str == "1.10 mV/h"
    assert by_label["Fraction of samples in bottom 20 %"].value_str == "84 %"


def test_missing_cluster_scores_legacy_mode():
    cell = _full_metadata_cell()
    cell.cluster_scores = None
    findings = build_findings(_result_with([cell]))
    f = findings.cell_findings[0]
    assert f.cluster_summaries == []
    assert f.sentence.startswith("M1/G7 HIGH")
    assert f.sentence.endswith(".")


def test_headline_counts():
    findings = build_findings(_result_with([_full_metadata_cell()]))
    assert "1 of 8 cell-groups flagged" in findings.headline
    assert "M1/G7 (HIGH)" in findings.headline
    assert findings.rest_duration_h == pytest.approx(48.0)


def test_headline_clean_pack():
    healthy = _full_metadata_cell(verdict="NORMAL")
    healthy.cluster_scores = {"self_discharge": 0.1}
    findings = build_findings(_result_with([healthy]))
    assert findings.cell_findings == []
    assert "all 8 cell-groups NORMAL" in findings.headline
    assert findings.overall_verdict == "OK"


def test_findings_sorted_by_composite_desc():
    c1 = _full_metadata_cell(verdict="ELEVATED", composite=1.2)
    c2 = _full_metadata_cell(verdict="HIGH", composite=3.4)
    c2.group_in_module = 3
    c2.channel_index = 2
    findings = build_findings(_result_with([c1, c2]))
    assert [f.composite_z for f in findings.cell_findings] == [3.4, 1.2]
