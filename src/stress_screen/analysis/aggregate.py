"""
aggregate.py — verdict aggregation for stress_screen.

Takes per-cell MethodResult dicts from rest.py and li_plating.py, combines
them into a single CellVerdict per cell, and rolls up to ModuleVerdict objects.
"""

from __future__ import annotations

import numpy as np

from stress_screen.analysis.util import winsorize_clip
from stress_screen.models import CellVerdict, MethodResult, ModuleVerdict, PackTopology


def aggregate(
    rest_results: dict[int, list[MethodResult]],
    li_plating_results: dict[int, MethodResult],
    topology: PackTopology,
    isc_results: dict[int, MethodResult] | None = None,
    z_thresh: float = 2.0,
) -> list[ModuleVerdict]:
    """
    Aggregate per-cell method results into module-level verdicts.

    Parameters
    ----------
    rest_results:
        Mapping of channel_index → list of MethodResult (6 rest methods).
    li_plating_results:
        Mapping of channel_index → MethodResult (1 li_plating method).
    topology:
        Pack topology describing module/channel layout.
    isc_results:
        Optional mapping of channel_index → MethodResult from ISC analysis.
        When provided, the ISC z-score and MethodResult are appended to each
        cell's composite (8 total methods). Defaults to None (7-method composite).
    z_thresh:
        Z-score threshold above which a method is counted as firing HIGH.

    Returns
    -------
    list[ModuleVerdict] ordered by module_id (1..N).
    """
    # --- Cell-level aggregation -------------------------------------------------
    cell_verdicts: dict[int, CellVerdict] = {}

    _missing_in_li = set(rest_results.keys()) - set(li_plating_results.keys())
    _missing_in_rest = set(li_plating_results.keys()) - set(rest_results.keys())
    if _missing_in_li:
        import warnings
        warnings.warn(
            f"aggregate(): {len(_missing_in_li)} channels in rest_results missing from "
            f"li_plating_results — skipped: {sorted(_missing_in_li)[:5]}..."
        )
    if _missing_in_rest:
        import warnings
        warnings.warn(
            f"aggregate(): {len(_missing_in_rest)} channels in li_plating_results missing "
            f"from rest_results — skipped"
        )

    all_channels = sorted(set(rest_results.keys()) & set(li_plating_results.keys()))

    for ch in all_channels:
        # 1. Collect all z-scores (6 rest + 1 li_plating + optional 1 isc)
        isc_mr = isc_results.get(ch) if isc_results is not None else None

        # 2. Confidence-weighted composite z-score.
        # Each MethodResult may publish metadata["confidence"] in [0, 1]; when
        # absent, confidence defaults to 1.0 — preserving the legacy unweighted
        # behaviour. Methods with confidence <= 0 are dropped.
        method_results_for_aggregation = (
            rest_results[ch] + [li_plating_results[ch]] +
            ([isc_mr] if isc_mr is not None else [])
        )
        z_with_conf: list[tuple[float, float]] = []
        for mr in method_results_for_aggregation:
            if np.isnan(mr.z_score):
                continue
            conf = float(mr.metadata.get("confidence", 1.0))
            if conf <= 0.0:
                continue
            z_with_conf.append((mr.z_score, conf))

        if z_with_conf:
            zs = np.array([z for z, _ in z_with_conf])
            ws = np.array([w for _, w in z_with_conf])
            clipped_z = winsorize_clip(zs, low=-8.0, high=8.0)
            composite_z = float(np.sum(clipped_z * ws) / np.sum(ws))
        else:
            composite_z = 0.0

        # valid_z used by the n_high count below — same set of z's
        valid_z = [z for z, _ in z_with_conf]

        # 3. Count methods firing HIGH
        n_high = sum(1 for z in valid_z if z >= z_thresh)

        # 4. Cell verdict
        if n_high >= 2 or composite_z > 2.0:
            verdict = "HIGH"
        elif n_high >= 1 or composite_z > 1.0:
            verdict = "ELEVATED"
        else:
            verdict = "NORMAL"

        # 5. Build CellVerdict
        method_results_list = rest_results[ch] + [li_plating_results[ch]]
        if isc_mr is not None:
            method_results_list = method_results_list + [isc_mr]
        cell_verdicts[ch] = CellVerdict(
            channel_index=ch,
            module_id=topology.module_for_channel(ch),
            group_in_module=topology.group_index_in_module(ch),
            composite_z=composite_z,
            n_methods_high=n_high,
            verdict=verdict,
            method_results=method_results_list,
        )

    # --- Module-level rollup ----------------------------------------------------
    module_verdicts: list[ModuleVerdict] = []

    for module_id in range(1, topology.module_count + 1):
        module_channels = topology.channels_in_module(module_id)
        cells = [cell_verdicts[ch] for ch in module_channels if ch in cell_verdicts]
        cells.sort(key=lambda cv: cv.group_in_module)

        flagged_cells = [cv for cv in cells if cv.verdict == "HIGH"]
        elevated_cells = [cv for cv in cells if cv.verdict == "ELEVATED"]
        if flagged_cells:
            verdict = "NOK"
        elif elevated_cells:
            verdict = "MARGINAL"
        else:
            verdict = "OK"

        module_verdicts.append(
            ModuleVerdict(
                module_id=module_id,
                verdict=verdict,
                flagged_cells=flagged_cells,
                all_cells=cells,
            )
        )

    return module_verdicts
