#!/usr/bin/env python3
"""
composite_ab.py — offline A/B of legacy vs clustered composite.

Reads stress_screen JSON results (which carry every per-method z-score),
recomputes both composites with current aggregation code, and prints the
per-cell verdict transition matrix plus per-pack module verdict changes.
No pipeline re-runs needed.

Usage::

    PYTHONPATH=src python scripts/composite_ab.py results_dir_or_json [more...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from stress_screen.analysis.aggregate import (
    AggregateParams,
    CompositeParams,
    _composite_clustered,
    _composite_legacy,
)
from stress_screen.models import MethodResult


def _verdict(composite_z: float, n_high: int, p: AggregateParams) -> str:
    if composite_z > p.high_composite or (
        n_high >= p.high_n_methods and composite_z >= p.high_composite_floor
    ):
        return "HIGH"
    if composite_z > p.elevated_composite or (
        n_high >= p.elevated_n_methods and composite_z >= p.elevated_composite_floor
    ):
        return "ELEVATED"
    return "NORMAL"


def _iter_json_files(args: list[str]):
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            yield from sorted(p.glob("*_result.json"))
        elif p.suffix == ".json":
            yield p


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    params = AggregateParams()
    comp = CompositeParams()

    cell_transitions: Counter[tuple[str, str]] = Counter()
    module_changes: list[str] = []
    n_cells = 0

    for path in _iter_json_files(argv):
        data = json.loads(path.read_text(encoding="utf-8"))
        pack = data.get("input", {}).get("csv_name", path.name)
        for module in data.get("modules", []):
            old_module, new_module = "OK", "OK"
            for cell in module.get("cells", []):
                mrs = [
                    MethodResult(
                        method_name=m["name"],
                        z_score=float("nan") if m["z"] is None else float(m["z"]),
                        verdict=m["verdict"],
                        metadata=m.get("metadata") or {},
                    )
                    for m in cell.get("methods", [])
                ]
                lz, ln = _composite_legacy(mrs, params)
                cz, cn, _scores = _composite_clustered(mrs, params, comp)
                v_old = _verdict(lz, ln, params)
                v_new = _verdict(cz, cn, params)
                cell_transitions[(v_old, v_new)] += 1
                n_cells += 1
                if v_old == "HIGH":
                    old_module = "NOK"
                elif v_old == "ELEVATED" and old_module != "NOK":
                    old_module = "MARGINAL"
                if v_new == "HIGH":
                    new_module = "NOK"
                elif v_new == "ELEVATED" and new_module != "NOK":
                    new_module = "MARGINAL"
            if old_module != new_module:
                module_changes.append(
                    f"  {pack} M{module['module_id']}: {old_module} -> {new_module}"
                )

    order = ["NORMAL", "ELEVATED", "HIGH"]
    print(f"Cells analysed: {n_cells}\n")
    print("Cell verdict transitions (legacy -> clustered):")
    label = "legacy / clustered"
    print(f"{label:>20s} " + " ".join(f"{v:>9s}" for v in order))
    for old in order:
        row = " ".join(f"{cell_transitions.get((old, new), 0):>9d}" for new in order)
        print(f"{old:>20s} {row}")

    changed = sum(v for (a, b), v in cell_transitions.items() if a != b)
    print(f"\nCells changing verdict: {changed} / {n_cells}")
    if module_changes:
        print("\nModule verdict changes:")
        print("\n".join(module_changes))
    else:
        print("\nNo module verdict changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
