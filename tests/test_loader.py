"""Unit tests for loader.py — CSV parsing, unit conversion, time handling."""

from __future__ import annotations

import numpy as np
import pytest

from stress_screen.loader import load_csv, active_channel_count, _parse_time_to_hours
import pandas as pd


def _write_csv(tmp_path, rows: list[str], name: str = "Mini_M1.csv"):
    """Write a minimal tester CSV: 2 comment lines, header, given data rows."""
    header = (
        "TimeStamp;Current;Voltage;SOC %;Warning;Fault;"
        "Cell_1_Volt;Cell_1_Temp;Cell_2_Volt;Cell_2_Temp"
    )
    content = "\n".join(["# comment line 1", "# comment 2", header] + rows) + "\n"
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_basic_parse_comments_decimals_trailing_semicolon(tmp_path):
    p = _write_csv(tmp_path, [
        "01_03_26_08_00_00;5,5;26,45;100,0;OK;OK;3494;17;3500;18;",
        "01_03_26_08_01_00;5,5;26,50;100,0;OK;OK;3495;17;3501;18;",
    ])
    top_df, cell_df = load_csv(p)
    assert len(top_df) == 2
    # decimal commas parsed
    assert top_df["current"].iloc[0] == pytest.approx(5.5)
    assert top_df["pack_voltage"].iloc[0] == pytest.approx(26.45)
    assert top_df["soc_pct"].iloc[0] == pytest.approx(100.0)
    assert top_df["warning"].iloc[0] == "OK"
    # elapsed time from timestamps
    assert top_df["time_hours"].iloc[1] == pytest.approx(1 / 60)
    # trailing semicolon must not create a phantom channel
    assert active_channel_count(cell_df) == 2


def test_millivolt_to_volt_conversion(tmp_path):
    p = _write_csv(tmp_path, [
        "01_03_26_08_00_00;0,0;26,0;100,0;OK;OK;3494;17;3500;18;",
    ])
    _, cell_df = load_csv(p)
    v = cell_df[cell_df["channel_index"] == 0]["voltage"].iloc[0]
    assert v == pytest.approx(3.494)


def test_inactive_channel_dropped(tmp_path):
    p = _write_csv(tmp_path, [
        "01_03_26_08_00_00;0,0;26,0;100,0;OK;OK;3494;17;0;0;",
        "01_03_26_08_01_00;0,0;26,0;100,0;OK;OK;3495;17;0;0;",
    ])
    _, cell_df = load_csv(p)
    assert active_channel_count(cell_df) == 1
    assert set(cell_df["channel_index"].unique()) == {0}


def test_zero_temperature_becomes_nan(tmp_path):
    p = _write_csv(tmp_path, [
        "01_03_26_08_00_00;0,0;26,0;100,0;OK;OK;3494;0;3500;18;",
    ])
    _, cell_df = load_csv(p)
    t0 = cell_df[cell_df["channel_index"] == 0]["temperature"].iloc[0]
    t1 = cell_df[cell_df["channel_index"] == 1]["temperature"].iloc[0]
    assert np.isnan(t0)
    assert t1 == pytest.approx(18.0)


def test_missing_required_columns_raises(tmp_path):
    p = tmp_path / "Bad_M1.csv"
    p.write_text("TimeStamp;Current;Cell_1_Volt\n01_03_26_08_00_00;0,0;3494\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_csv(p)


def test_downsample_validation():
    with pytest.raises(ValueError, match="downsample"):
        load_csv("whatever.csv", downsample=0)
    with pytest.raises(ValueError, match="downsample"):
        load_csv("whatever.csv", downsample=1.5)  # type: ignore[arg-type]


def test_downsample_keeps_every_nth(tmp_path):
    rows = [
        f"01_03_26_08_{m:02d}_00;0,0;26,0;100,0;OK;OK;3494;17;3500;18;"
        for m in range(10)
    ]
    p = _write_csv(tmp_path, rows)
    top_df, _ = load_csv(p, downsample=3)
    assert len(top_df) == 4  # rows 0, 3, 6, 9


def test_hh_mm_ss_format_and_midnight_rollover():
    s = pd.Series(["23:59:00", "23:59:30", "00:00:00", "00:00:30", "00:01:00"])
    hours = _parse_time_to_hours(s)
    assert hours.iloc[0] == 0.0
    # strictly increasing across midnight
    assert (np.diff(hours.values) > 0).all()
    assert hours.iloc[2] == pytest.approx(60 / 3600)
    assert hours.iloc[4] == pytest.approx(120 / 3600)


def test_double_midnight_rollover():
    s = pd.Series(["23:00:00", "01:00:00", "23:00:00", "01:00:00"])
    hours = _parse_time_to_hours(s)
    assert hours.iloc[1] == pytest.approx(2.0)
    assert hours.iloc[2] == pytest.approx(24.0)
    assert hours.iloc[3] == pytest.approx(26.0)


def test_c_engine_matches_legacy_loader(tmp_path, monkeypatch):
    """The fast C-engine path must produce identical frames to the legacy
    python-engine path (STRESS_SCREEN_LEGACY_LOADER=1 escape hatch)."""
    from tests.synth import make_synthetic_csv

    p = make_synthetic_csv(tmp_path)
    monkeypatch.delenv("STRESS_SCREEN_LEGACY_LOADER", raising=False)
    top_new, cell_new = load_csv(p)
    monkeypatch.setenv("STRESS_SCREEN_LEGACY_LOADER", "1")
    top_old, cell_old = load_csv(p)

    pd.testing.assert_frame_equal(top_new, top_old)
    pd.testing.assert_frame_equal(cell_new, cell_old)


def test_synthetic_fixture_roundtrip(tmp_path):
    """The synthetic fixture must load with the expected shape."""
    from tests.synth import make_synthetic_csv

    p = make_synthetic_csv(tmp_path)
    top_df, cell_df = load_csv(p)
    assert active_channel_count(cell_df) == 16
    assert top_df["time_hours"].iloc[-1] == pytest.approx(53.0, abs=0.1)
    # gap-column temperatures (dead sensors) must be NaN
    ch14 = cell_df[cell_df["channel_index"] == 14]["temperature"]
    assert ch14.isna().all()
