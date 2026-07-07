"""Tests for the calibrate harness and the CLI subcommand dispatcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stress_screen.calibrate import load_labels, run_calibration


def _fake_result(pack_id: str, cells_spec: list[tuple[int, int, str, float]]) -> dict:
    """cells_spec: (module_id, group, verdict, composite_z) per cell."""
    modules: dict[int, dict] = {}
    for module_id, group, verdict, z in cells_spec:
        m = modules.setdefault(module_id, {"module_id": module_id, "cells": []})
        m["cells"].append({
            "channel_index": (module_id - 1) * 8 + group - 1,
            "label": f"M{module_id}/G{group}",
            "group_in_module": group,
            "composite_z": z,
            "composite_z_legacy": z,
            "cluster_scores": {"self_discharge": z, "isc": z / 2},
            "n_methods_high": 0,
            "verdict": verdict,
            "methods": [
                {"name": "ocv_k", "z": z, "verdict": verdict, "metadata": {}},
                {"name": "isc", "z": z / 2, "verdict": "NORMAL", "metadata": {}},
            ],
        })
    for m in modules.values():
        # Binary module verdict, matching the real aggregation (strict)
        any_flagged = any(c["verdict"] in ("HIGH", "ELEVATED") for c in m["cells"])
        m["verdict"] = "NOK" if any_flagged else "OK"
    return {
        "schema_version": 2,
        "input": {"pack_id": pack_id, "csv_name": f"{pack_id}_D01012026_M2.csv"},
        "verdict": {"overall": "OK", "exit_code": 0},
        "modules": list(modules.values()),
    }


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    r = _fake_result("PackA", [
        (1, 1, "HIGH", 3.0),
        (1, 2, "NORMAL", 0.1),
        (2, 1, "ELEVATED", 1.2),
        (2, 2, "NORMAL", -0.2),
    ])
    (d / "PackA_D01012026_M2_result.json").write_text(json.dumps(r))
    return d


@pytest.fixture
def labels_file(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "pack_id;module_id;group;outcome\n"
        "PackA;1;1;bad\n"
        "PackA;1;2;good\n"
        "PackA;2;1;good\n"    # ELEVATED but actually good → lenient FP
        "PackA;2;;good\n"     # module-level label
        "PackB;1;1;bad\n"     # no matching result → unmatched warning
    )
    return p


def test_load_labels_parses_cell_and_module_rows(labels_file):
    rows = load_labels(labels_file)
    assert len(rows) == 5
    assert rows[0].group == 1 and rows[0].outcome == "bad"
    assert rows[3].group is None


def test_load_labels_rejects_bad_header(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("pack;module;grp;result\nA;1;1;bad\n")
    with pytest.raises(ValueError, match="header"):
        load_labels(p)


def test_load_labels_rejects_bad_outcome(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("pack_id;module_id;group;outcome\nA;1;1;broken\n")
    with pytest.raises(ValueError, match="outcome"):
        load_labels(p)


def test_run_calibration_report(results_dir, labels_file):
    report = run_calibration(results_dir, labels_file)
    # Confusion at strict gate: HIGH cell labeled bad → TP=1; others good & not HIGH → TN
    assert "Cells, strict (HIGH = bad)" in report
    assert "TP=  1" in report
    # Separation: bad cell has the highest composite → AUC 1.0
    assert "composite_z" in report and "AUC=1.000" in report
    # per-cluster and per-method lines present
    assert "cluster:self_discharge" in report
    assert "method:ocv_k" in report
    # unmatched PackB row reported
    assert "no matching result" in report and "PackB" in report


def test_run_calibration_sweep(results_dir, labels_file):
    report = run_calibration(results_dir, labels_file, sweep=True)
    assert "Threshold sweep" in report
    assert "Suggested operating point" in report


def _run_cli(*argv: str):
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "stress_screen", *argv],
        capture_output=True, text=True, env=env,
    )


def test_cli_calibrate_subcommand(results_dir, labels_file):
    result = _run_cli("calibrate", "--results", str(results_dir),
                      "--labels", str(labels_file))
    assert result.returncode == 0, result.stderr
    assert "Separation power" in result.stdout


def test_cli_bare_csv_still_works_as_run(tmp_path):
    """Backward compat: `stress_screen missing.csv` must behave like `run`
    (here: a clean exit-2 'file not found', not an argparse usage error)."""
    result = _run_cli(str(tmp_path / "nope_M2.csv"), "--no-html")
    assert result.returncode == 2
    assert "CSV file not found" in result.stderr


def test_cli_subcommand_helps():
    # Bare --help routes to the implicit "run" command (compat with the
    # single-command CLI) and must exit cleanly.
    bare = _run_cli("--help")
    assert bare.returncode == 0
    assert "csv" in bare.stdout.lower()
    assert _run_cli("run", "--help").returncode == 0
    assert _run_cli("calibrate", "--help").returncode == 0