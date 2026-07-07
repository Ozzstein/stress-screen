"""Tests for the history store, trend analysis, and fleet subcommands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stress_screen.history import (
    HistoryStore,
    analyze_cell_trend,
    summarize_result,
    theil_sen_slope,
)


def _fake_result(pack_id: str, test_date: str, k_by_cell: dict[str, float],
                 verdict_by_cell: dict[str, str] | None = None) -> dict:
    """Minimal schema-v2 result with one module and the given cells."""
    verdict_by_cell = verdict_by_cell or {}
    cells = []
    for i, (label, k) in enumerate(sorted(k_by_cell.items())):
        group = int(label.split("/")[1][1:])
        v = verdict_by_cell.get(label, "NORMAL")
        cells.append({
            "channel_index": i, "label": label, "group_in_module": group,
            "composite_z": 0.1, "composite_z_legacy": 0.1,
            "cluster_scores": {"self_discharge": 0.1},
            "n_methods_high": 0, "verdict": v,
            "methods": [
                {"name": "ocv_k", "z": 0.1, "verdict": v, "metadata": {"k": k}},
                {"name": "temp_k", "z": 0.1, "verdict": "NORMAL",
                 "metadata": {"k_corrected": k * 0.9}},
            ],
        })
    any_high = any(c["verdict"] == "HIGH" for c in cells)
    any_elev = any(c["verdict"] == "ELEVATED" for c in cells)
    mod_verdict = "NOK" if any_high else ("MARGINAL" if any_elev else "OK")
    return {
        "schema_version": 2,
        "generated_at": f"{test_date}T10:00:00+00:00",
        "input": {"pack_id": pack_id, "test_date": test_date,
                  "csv_name": f"{pack_id}_M1.csv"},
        "verdict": {"overall": mod_verdict, "exit_code": 1 if any_high else 0},
        "modules": [{"module_id": 1, "verdict": mod_verdict, "cells": cells}],
    }


def _write_result(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_store_add_and_entries(tmp_path):
    store = HistoryStore(tmp_path / "store")
    r1 = _write_result(tmp_path, "a_result.json",
                       _fake_result("PackA", "2026-01-01", {"M1/G1": 1e-5}))
    r2 = _write_result(tmp_path, "b_result.json",
                       _fake_result("PackB", "2026-02-01", {"M1/G1": 2e-5}))
    store.add(r1)
    store.add(r2)
    assert len(store.entries()) == 2
    assert store.packs() == ["PackA", "PackB"]
    only_a = store.entries("PackA")
    assert len(only_a) == 1 and only_a[0].overall == "OK"
    assert only_a[0].cells["M1/G1"]["k"] == pytest.approx(1e-5)


def test_store_add_same_run_twice_is_idempotent(tmp_path):
    store = HistoryStore(tmp_path / "store")
    r = _write_result(tmp_path, "a_result.json",
                      _fake_result("PackA", "2026-01-01", {"M1/G1": 1e-5}))
    store.add(r)
    store.add(r)
    assert len(store.entries()) == 1


def test_cell_series_ordered_by_date(tmp_path):
    store = HistoryStore(tmp_path / "store")
    for i, (date, k) in enumerate([("2026-03-01", 3e-5), ("2026-01-01", 1e-5),
                                   ("2026-02-01", 2e-5)]):
        r = _write_result(tmp_path, f"run{i}_result.json",
                          _fake_result("PackA", date, {"M1/G1": k}))
        store.add(r)
    series = store.cell_series("PackA", 1, 1)
    assert [s["test_date"] for s in series] == \
        ["2026-01-01", "2026-02-01", "2026-03-01"]
    assert [s["k"] for s in series] == pytest.approx([1e-5, 2e-5, 3e-5])


def test_rebuild_index_recovers_from_corruption(tmp_path):
    store = HistoryStore(tmp_path / "store")
    r = _write_result(tmp_path, "a_result.json",
                      _fake_result("PackA", "2026-01-01", {"M1/G1": 1e-5}))
    store.add(r)
    # Corrupt the index; entries() skips garbage, rebuild restores it
    store.index_path.write_text("this is not json\n")
    assert store.entries() == []
    assert store.rebuild_index() == 1
    assert len(store.entries()) == 1


def test_theil_sen_slope():
    assert theil_sen_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert theil_sen_slope([1.0, None, 3.0]) == pytest.approx(1.0)
    assert theil_sen_slope([5.0]) is None
    # Robust to one outlier
    assert theil_sen_slope([1.0, 2.0, 100.0, 4.0, 5.0]) == pytest.approx(1.0, abs=0.5)


def test_analyze_cell_trend_flags_k_growth():
    series = [{"k": 1e-5, "verdict": "NORMAL"},
              {"k": 2e-5, "verdict": "NORMAL"},
              {"k": 3e-5, "verdict": "NORMAL"}]
    trend = analyze_cell_trend(series, k_slope_floor=1e-6)
    assert trend["k_worsening"] and trend["worsening"]
    assert trend["k_slope_per_run"] == pytest.approx(1e-5)


def test_analyze_cell_trend_flags_verdict_transition():
    series = [{"k": 1e-5, "verdict": "NORMAL"},
              {"k": 1e-5, "verdict": "ELEVATED"}]
    trend = analyze_cell_trend(series)
    assert trend["entered_flagged"] and trend["worsening"]


def test_analyze_cell_trend_stable_cell_not_flagged():
    series = [{"k": 1e-5, "verdict": "NORMAL"}] * 4
    trend = analyze_cell_trend(series)
    assert not trend["worsening"]


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def _run_cli(*argv: str):
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "stress_screen", *argv],
        capture_output=True, text=True, env=env,
    )


@pytest.fixture
def populated_store(tmp_path):
    store_dir = tmp_path / "store"
    store = HistoryStore(store_dir)
    for i, (date, k, verdict) in enumerate([
        ("2026-01-01", 1e-5, "NORMAL"),
        ("2026-02-01", 2e-5, "NORMAL"),
        ("2026-03-01", 3e-5, "ELEVATED"),
    ]):
        r = _write_result(
            tmp_path, f"run{i}_result.json",
            _fake_result("PackA", date, {"M1/G1": k, "M1/G2": 1e-5},
                         {"M1/G1": verdict}),
        )
        store.add(r)
    return store_dir


def test_cli_history_lists_runs(populated_store):
    result = _run_cli("history", "--history", str(populated_store))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PackA") == 3


def test_cli_trend_flags_worsening_cell(populated_store):
    result = _run_cli("trend", "--history", str(populated_store), "--pack", "PackA")
    assert result.returncode == 1, result.stdout  # worsening → exit 1
    assert "WORSENING M1/G1" in result.stdout
    assert "M1/G2" not in result.stdout.split("WORSENING", 1)[1]


def test_cli_trend_cell_detail(populated_store):
    result = _run_cli("trend", "--history", str(populated_store),
                      "--pack", "PackA", "--module", "1", "--group", "1")
    assert "per-run history" in result.stdout
    assert result.stdout.count("k=") >= 3


def test_cli_trend_unknown_pack_errors(populated_store):
    result = _run_cli("trend", "--history", str(populated_store), "--pack", "Nope")
    assert result.returncode == 2
    assert "Known packs" in result.stderr


def test_cli_run_with_history_and_batch(tmp_path):
    """End-to-end: batch over synthetic CSVs feeding a history store."""
    from tests.synth import make_synthetic_csv

    data_dir = tmp_path / "data"
    make_synthetic_csv(data_dir, leaky_channels={4: 0.8},
                       filename="PackX_D01032026_080000_M2.csv")
    make_synthetic_csv(data_dir, rest_hours=20.0,
                       filename="Short_D02032026_080000_M2.csv")  # protocol-invalid
    store_dir = tmp_path / "store"

    result = _run_cli("batch", str(data_dir), "--no-html", "--no-pdf",
                      "--history", str(store_dir))
    # Worst case exit: the short-rest file errors (2)
    assert result.returncode == 2
    assert "PackX_D01032026_080000_M2.csv: NOK" in result.stdout
    assert "Short_D02032026_080000_M2.csv: ERROR" in result.stdout

    # The valid run landed in the history store
    listing = _run_cli("history", "--history", str(store_dir))
    assert "PackX" in listing.stdout
