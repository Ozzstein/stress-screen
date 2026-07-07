"""
history.py — local fleet history store for stress_screen results.

A history store is a plain directory of JSON result files plus an
append-only ``index.jsonl`` with one summary line per run. This keeps the
store inspectable, diffable, and safe on the network shares bench PCs
actually use (no database locking); the index makes listing and trending
fast without re-parsing full result files, and it is always rebuildable
from the JSONs (``stress_screen history --history DIR --rebuild-index``).

Trend analysis is deliberately based on RAW PHYSICAL METRICS (self-discharge
rate k, its temperature-corrected variant, relaxation tau_inv, ISC excess k)
rather than z-scores: every z is fleet-relative *within one run* (that pack,
that day), so a uniformly degrading pack shows flat z-scores over time while
its raw k trajectory climbs.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INDEX_NAME = "index.jsonl"

#: Per-cell raw metrics captured in the index: (index key, method, metadata key)
_CELL_METRICS = (
    ("k", "ocv_k", "k"),
    ("k_corrected", "temp_k", "k_corrected"),
    ("tau_inv", "li_plating", "tau_inv"),
    ("s1_excess_k", "isc", "s1_excess_k"),
)


@dataclass
class RunSummary:
    """One line of the index — the summary of one analysis run."""

    pack_id: str
    test_date: str | None
    generated_at: str
    json_file: str                       # filename within the store root
    overall: str
    modules: dict[int, str] = field(default_factory=dict)          # id → verdict
    cells: dict[str, dict[str, Any]] = field(default_factory=dict) # "M1/G3" → snapshot

    def to_json_line(self) -> str:
        return json.dumps({
            "pack_id": self.pack_id,
            "test_date": self.test_date,
            "generated_at": self.generated_at,
            "json_file": self.json_file,
            "overall": self.overall,
            "modules": {str(k): v for k, v in self.modules.items()},
            "cells": self.cells,
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunSummary":
        return cls(
            pack_id=d["pack_id"],
            test_date=d.get("test_date"),
            generated_at=d.get("generated_at", ""),
            json_file=d["json_file"],
            overall=d.get("overall", "?"),
            modules={int(k): v for k, v in (d.get("modules") or {}).items()},
            cells=d.get("cells") or {},
        )


def summarize_result(data: dict[str, Any], json_file: str) -> RunSummary:
    """Build a RunSummary from a parsed *_result.json dict."""
    cells: dict[str, dict[str, Any]] = {}
    modules: dict[int, str] = {}
    for module in data.get("modules", []):
        modules[int(module["module_id"])] = module["verdict"]
        for cell in module.get("cells", []):
            methods = {m["name"]: m for m in cell.get("methods", [])}
            snapshot: dict[str, Any] = {
                "composite_z": cell.get("composite_z"),
                "verdict": cell.get("verdict"),
            }
            for key, method, meta_key in _CELL_METRICS:
                m = methods.get(method)
                snapshot[key] = (m.get("metadata") or {}).get(meta_key) if m else None
            cells[cell["label"]] = snapshot

    return RunSummary(
        pack_id=data.get("input", {}).get("pack_id", "unknown"),
        test_date=data.get("input", {}).get("test_date"),
        generated_at=data.get("generated_at", ""),
        json_file=json_file,
        overall=data.get("verdict", {}).get("overall", "?"),
        modules=modules,
        cells=cells,
    )


class HistoryStore:
    """Directory of result JSONs + append-only index.jsonl."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / INDEX_NAME

    # ------------------------------------------------------------------

    def add(self, json_result_path: Path) -> RunSummary:
        """Copy a result JSON into the store and append its index line.

        Adding the same run twice (same filename and generated_at) is a
        no-op returning the existing summary.
        """
        json_result_path = Path(json_result_path)
        data = json.loads(json_result_path.read_text(encoding="utf-8"))

        dest = self.root / json_result_path.name
        if dest.exists():
            existing = json.loads(dest.read_text(encoding="utf-8"))
            if existing.get("generated_at") == data.get("generated_at"):
                summary = summarize_result(data, dest.name)
                if any(e.json_file == dest.name for e in self.entries()):
                    return summary
            else:
                # Same filename, different run — disambiguate by timestamp
                stamp = (data.get("generated_at") or "run").replace(":", "").replace("-", "")
                dest = self.root / f"{json_result_path.stem}_{stamp}.json"

        if json_result_path.resolve() != dest.resolve():
            shutil.copy2(json_result_path, dest)

        summary = summarize_result(data, dest.name)
        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(summary.to_json_line() + "\n")
        return summary

    # ------------------------------------------------------------------

    def entries(self, pack_id: str | None = None) -> list[RunSummary]:
        """All index entries (oldest first), optionally filtered by pack.

        Corrupt index lines are skipped (rebuild with ``rebuild_index``).
        """
        if not self.index_path.exists():
            return []
        out: list[RunSummary] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                summary = RunSummary.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if pack_id is None or summary.pack_id == pack_id:
                out.append(summary)
        out.sort(key=lambda s: (s.test_date or "", s.generated_at))
        return out

    def packs(self) -> list[str]:
        return sorted({e.pack_id for e in self.entries()})

    # ------------------------------------------------------------------

    def cell_series(self, pack_id: str, module_id: int, group: int) -> list[dict[str, Any]]:
        """Per-run snapshots of one cell (oldest first)."""
        label = f"M{module_id}/G{group}"
        series = []
        for e in self.entries(pack_id):
            snap = e.cells.get(label)
            if snap is None:
                continue
            series.append({
                "test_date": e.test_date,
                "generated_at": e.generated_at,
                "json_file": e.json_file,
                **snap,
            })
        return series

    # ------------------------------------------------------------------

    def rebuild_index(self) -> int:
        """Rebuild index.jsonl from the JSON files in the store. Returns the
        number of runs indexed."""
        summaries = []
        for f in sorted(self.root.glob("*.json")):
            if f.name == INDEX_NAME:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "modules" not in data:
                continue
            summaries.append(summarize_result(data, f.name))
        summaries.sort(key=lambda s: (s.test_date or "", s.generated_at))
        with open(self.index_path, "w", encoding="utf-8") as fh:
            for s in summaries:
                fh.write(s.to_json_line() + "\n")
        return len(summaries)


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def theil_sen_slope(values: list[float]) -> float | None:
    """Median of pairwise slopes per run-index step. None for < 2 points."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return None
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(pts)
        for x2, y2 in pts[i + 1:]
    ]
    slopes.sort()
    n = len(slopes)
    mid = n // 2
    return slopes[mid] if n % 2 else 0.5 * (slopes[mid - 1] + slopes[mid])


def analyze_cell_trend(
    series: list[dict[str, Any]],
    k_slope_floor: float = 1e-6,
) -> dict[str, Any]:
    """Flag a worsening cell from its run history.

    Worsening =
      - Theil–Sen slope of raw k across >= 3 runs above *k_slope_floor*
        (units: h⁻¹ per run), or
      - the verdict entering ELEVATED/HIGH after previously being NORMAL.
    """
    k_values = [s.get("k") for s in series]
    slope = theil_sen_slope(k_values) if len(series) >= 3 else None
    verdicts = [s.get("verdict") for s in series]

    entered_flagged = (
        len(verdicts) >= 2
        and verdicts[-1] in ("ELEVATED", "HIGH")
        and all(v == "NORMAL" for v in verdicts[:-1] if v is not None)
    )
    k_worsening = slope is not None and slope > k_slope_floor

    return {
        "n_runs": len(series),
        "k_slope_per_run": slope,
        "k_worsening": k_worsening,
        "entered_flagged": entered_flagged,
        "worsening": k_worsening or entered_flagged,
    }
