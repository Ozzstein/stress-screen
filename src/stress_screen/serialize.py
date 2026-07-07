"""
serialize.py — machine-readable JSON export of analysis results.

Turns the in-memory :class:`~stress_screen.models.AnalysisResult` tree into a
versioned, JSON-safe dict and writes it next to the HTML/PDF reports. The JSON
result is the foundation for calibration, regression testing, and fleet/trend
tracking — everything the HTML report shows is preserved here as data.

Schema (version 1)
------------------
::

    {
      "schema_version": 1,
      "tool_version": "0.1.0",
      "generated_at": "2026-07-07T12:00:00+00:00",
      "input": {
        "csv_name": "..._M6.csv",
        "csv_path": "/abs/path/..._M6.csv",
        "file_size_bytes": 123,
        "pack_id": "..._M6",
        "test_date": "2026-05-18",        # null when not derivable
        "module_count": 6
      },
      "config": { ... resolved analysis configuration ... },
      "topology": {"module_count": 6, "series": 8, "parallel": 4,
                   "config_name": "4P8S", "active_channels": 48},
      "segments": [{"phase": "charge", "start_time_h": 0.0,
                    "end_time_h": 2.1, "duration_h": 2.1}, ...],
      "verdict": {"overall": "OK|MARGINAL|NOK", "exit_code": 0},
      "modules": [
        {"module_id": 1, "verdict": "OK",
         "cells": [
           {"channel_index": 0, "label": "M1/G1", "group_in_module": 1,
            "composite_z": 0.42, "n_methods_high": 0, "verdict": "NORMAL",
            "methods": [
              {"name": "ocv_k", "z": 0.31, "verdict": "NORMAL",
               "metadata": {"k": 1.2e-05, ...}},
              ...
            ]}]}]
    }

Method ``metadata`` dicts pass through untouched (apart from JSON-safety
conversion), so every sub-z, fitted parameter, and gate value produced by the
detection methods is preserved. NaN/inf map to ``null`` — bare ``NaN`` is not
valid JSON.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stress_screen.models import AnalysisResult

SCHEMA_VERSION = 1

#: Filename date components: newer tester firmware writes ``_D<DDMMYYYY>_``
#: (optionally followed by ``_HHMMSS``); older files use ``_P<DDMMYYYY>_``.
_DATE_PATTERNS = (
    re.compile(r"_D(\d{2})(\d{2})(\d{4})[_.]"),
    re.compile(r"_P(\d{2})(\d{2})(\d{4})[_.]"),
)


def tool_version() -> str:
    """Return the installed stress-screen version, or "unknown" outside a package."""
    try:
        from importlib.metadata import version

        return version("stress-screen")
    except Exception:
        return "unknown"


def extract_test_date(filename: str) -> date | None:
    """Extract the test date from a tester CSV filename.

    Tries the ``_D<DDMMYYYY>_`` pattern first (current firmware), then the
    legacy ``_P<DDMMYYYY>_`` pattern. Returns None when neither matches or the
    digits do not form a real calendar date.
    """
    for pattern in _DATE_PATTERNS:
        m = pattern.search(filename)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(year, month, day)
            except ValueError:
                continue
    return None


def extract_pack_id(filename: str) -> str:
    """Derive a stable pack identifier from a tester CSV filename.

    Strips the date (``_D...``/``_P...``) component and anything after it, so
    the same pack tested on different dates maps to the same id. Falls back to
    the bare stem when no date component is present.
    """
    stem = Path(filename).stem
    m = re.search(r"_[DP]\d{8}", stem)
    if m:
        return stem[: m.start()]
    # No date component — strip the module-count suffix noise conservatively
    return stem


def _jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-safe primitives.

    numpy scalars → Python scalars; NaN/inf → None; tuples/sets → lists;
    Path → str; dataclass-like leftovers → str as a last resort.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def result_to_dict(
    result: AnalysisResult,
    config: dict[str, Any] | None = None,
    run_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize an :class:`AnalysisResult` into the versioned schema dict.

    Parameters
    ----------
    result:
        The full analysis result tree.
    config:
        The resolved analysis configuration as a plain dict (chemistry,
        thresholds, ...). Stored verbatim so a JSON result always records
        which parameters produced its verdicts.
    run_info:
        Extra run metadata merged into the ``input`` section (e.g.
        ``downsample``).
    """
    csv_path = result.csv_path
    try:
        file_size = csv_path.stat().st_size
    except OSError:
        file_size = None

    test_date = extract_test_date(csv_path.name)

    input_section: dict[str, Any] = {
        "csv_name": csv_path.name,
        "csv_path": str(csv_path),
        "file_size_bytes": file_size,
        "pack_id": extract_pack_id(csv_path.name),
        "test_date": test_date.isoformat() if test_date else None,
        "module_count": result.topology.module_count,
    }
    if run_info:
        input_section.update(_jsonable(run_info))

    topo = result.topology
    any_marginal = any(m.verdict == "MARGINAL" for m in result.module_verdicts)
    overall = "NOK" if result.any_nok else ("MARGINAL" if any_marginal else "OK")

    modules = []
    for mv in result.module_verdicts:
        cells = []
        for cv in mv.all_cells:
            cells.append({
                "channel_index": _jsonable(cv.channel_index),
                "label": cv.label,
                "group_in_module": _jsonable(cv.group_in_module),
                "composite_z": _jsonable(cv.composite_z),
                "n_methods_high": _jsonable(cv.n_methods_high),
                "verdict": cv.verdict,
                "methods": [
                    {
                        "name": mr.method_name,
                        "z": _jsonable(mr.z_score),
                        "verdict": mr.verdict,
                        "metadata": _jsonable(mr.metadata),
                    }
                    for mr in cv.method_results
                ],
            })
        modules.append({
            "module_id": _jsonable(mv.module_id),
            "verdict": mv.verdict,
            "cells": cells,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": tool_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": input_section,
        "config": _jsonable(config or {}),
        "topology": {
            "module_count": topo.module_count,
            "series": topo.series,
            "parallel": topo.parallel,
            "config_name": topo.config_name,
            "active_channels": topo.active_channels,
        },
        "segments": [
            {
                "phase": s.phase,
                "start_time_h": _jsonable(s.start_time_h),
                "end_time_h": _jsonable(s.end_time_h),
                "duration_h": _jsonable(s.duration_h),
            }
            for s in result.segments
        ],
        "verdict": {
            "overall": overall,
            "exit_code": 1 if result.any_nok else 0,
        },
        "modules": modules,
    }


def write_json_result(
    result: AnalysisResult,
    out_path: Path,
    config: dict[str, Any] | None = None,
    run_info: dict[str, Any] | None = None,
) -> Path:
    """Write the JSON result to *out_path* and return the path."""
    payload = result_to_dict(result, config=config, run_info=run_info)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return out_path
