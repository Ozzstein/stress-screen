"""Unit tests for serialize.py — date/pack-id extraction and JSON safety."""

from __future__ import annotations

import json
from datetime import date

import numpy as np

from stress_screen.serialize import _jsonable, extract_pack_id, extract_test_date


def test_extract_test_date_d_pattern():
    assert extract_test_date(
        "DataLogging_C1_I01_D22062026_091000_M4 1.csv"
    ) == date(2026, 6, 22)


def test_extract_test_date_legacy_p_pattern():
    assert extract_test_date(
        "DataLogging_C1_I01_P18052026_M6.csv"
    ) == date(2026, 5, 18)


def test_extract_test_date_year_first_pattern():
    # Some testers write the date year-first (YYYYMMDD) rather than DDMMYYYY.
    assert extract_test_date(
        "DataLogging_C2_I04_D20261002_120054_M4.csv"
    ) == date(2026, 10, 2)
    assert extract_test_date("Pack_P20260518_M6.csv") == date(2026, 5, 18)


def test_extract_test_date_missing_or_invalid():
    assert extract_test_date("NoDateHere_M6.csv") is None
    # digits that form a real calendar date in neither DDMMYYYY nor YYYYMMDD
    assert extract_test_date("Pack_D99992026_M6.csv") is None


def test_extract_pack_id_strips_date_suffix():
    assert extract_pack_id(
        "DataLogging_C1_I01_D22062026_091000_M4 1.csv"
    ) == "DataLogging_C1_I01"
    assert extract_pack_id("DataLogging_C1_I01_P18052026_M6.csv") == "DataLogging_C1_I01"
    assert extract_pack_id("Custom_M6.csv") == "Custom_M6"


def test_jsonable_handles_numpy_and_nan():
    data = {
        "i16": np.int16(4),
        "f64": np.float64(1.5),
        "nan": float("nan"),
        "inf": np.inf,
        "arr": np.array([1.0, np.nan]),
        "tup": (3.0, 3.65),
        "nested": {"z": np.float32(2.0)},
    }
    out = _jsonable(data)
    # must be dumpable with allow_nan=False (strict JSON)
    dumped = json.dumps(out, allow_nan=False)
    assert out["i16"] == 4
    assert out["f64"] == 1.5
    assert out["nan"] is None
    assert out["inf"] is None
    assert out["arr"] == [1.0, None]
    assert out["tup"] == [3.0, 3.65]
    assert out["nested"]["z"] == 2.0
    assert "NaN" not in dumped
