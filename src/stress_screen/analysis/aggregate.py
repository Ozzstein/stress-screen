"""
aggregate.py — verdict aggregation for stress_screen.

Takes per-cell MethodResult dicts from rest.py and li_plating.py, combines
them into a single CellVerdict per cell, and rolls up to ModuleVerdict objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stress_screen.analysis.util import winsorize_clip
from stress_screen.models import CellVerdict, MethodResult, ModuleVerdict, PackTopology


#: Cluster membership of the detection methods. The six rest methods are NOT
#: independent measurements — they share fitted inputs by construction:
#:   - temp_k is ocv_k's k times an Arrhenius scalar          → self_discharge
#:   - thermal_corr and cusum both analyse ocv_k's residuals  → residual_dynamics
#:   - spread and rank both derive from the same T-compensated
#:     fleet voltage pivot                                    → fleet_divergence
#: Treating them as eight independent votes overweights self-discharge and
#: dilutes the genuinely independent li_plating / isc signals. Membership is
#: structural (it follows from the code), so it is fixed here; the per-cluster
#: WEIGHTS are the calibratable knobs (config section `composite`).
CLUSTERS: dict[str, tuple[str, ...]] = {
    "self_discharge": ("ocv_k", "temp_k"),
    "residual_dynamics": ("thermal_corr", "cusum"),
    "fleet_divergence": ("spread", "rank"),
    "li_plating": ("li_plating",),
    "isc": ("isc",),
}

_METHOD_TO_CLUSTER: dict[str, str] = {
    m: c for c, members in CLUSTERS.items() for m in members
}


@dataclass
class CompositeParams:
    """How per-method z-scores combine into the official composite."""

    mode: str = "clustered"
    """"clustered" (default): reduce correlated methods within each cluster
    first, then combine cluster scores. "legacy": plain mean of all method
    z-scores (pre-cluster behavior)."""

    reduce: str = "mean"
    """Within-cluster reduction. "mean" (default): cluster members are
    redundant noisy readings of the same physical signal, and averaging
    reduces that noise — when members disagree (one fires, its twin is
    quiet) the disagreement itself argues against a real defect. "max"
    (strongest reading wins) is more sensitive and is available for fleets
    where calibration data justifies it."""

    weight_self_discharge: float = 1.0
    weight_residual_dynamics: float = 1.0
    weight_fleet_divergence: float = 1.0
    weight_li_plating: float = 1.0
    weight_isc: float = 1.0
    """Per-cluster weights. All 1.0 until the calibrate command justifies
    otherwise — weights must be earned from labeled outcomes, not guessed."""


@dataclass
class AggregateParams:
    """Tunable parameters for composite-z aggregation and verdict gates."""

    z_thresh: float = 2.0
    """Z-score threshold above which a single method counts as firing HIGH."""

    winsor_z: float = 8.0
    """Per-method z-scores are clipped to ±winsor_z before averaging, so one
    extreme method cannot dominate the composite."""

    high_composite: float = 2.0
    """Cell is HIGH when composite_z exceeds this outright."""

    high_n_methods: int = 2
    """... or when at least this many methods fire HIGH and the composite
    clears high_composite_floor."""

    high_composite_floor: float = 1.0

    elevated_composite: float = 1.0
    """Cell is ELEVATED when composite_z exceeds this outright."""

    elevated_n_methods: int = 1
    """... or when at least this many methods fire HIGH and the composite
    clears elevated_composite_floor."""

    elevated_composite_floor: float = 0.5


def _composite_legacy(
    method_results: list[MethodResult],
    params: AggregateParams,
) -> tuple[float, int]:
    """Plain confidence-weighted mean of all method z-scores (pre-cluster
    behavior, bit-identical). Returns (composite_z, n_methods_high)."""
    z_with_conf: list[tuple[float, float]] = []
    for mr in method_results:
        if np.isnan(mr.z_score):
            continue
        conf = float(mr.metadata.get("confidence", 1.0))
        if conf <= 0.0:
            continue
        z_with_conf.append((mr.z_score, conf))

    if z_with_conf:
        zs = np.array([z for z, _ in z_with_conf])
        ws = np.array([w for _, w in z_with_conf])
        clipped_z = winsorize_clip(zs, low=-params.winsor_z, high=params.winsor_z)
        composite_z = float(np.sum(clipped_z * ws) / np.sum(ws))
    else:
        composite_z = 0.0

    n_high = sum(1 for z, _ in z_with_conf if z >= params.z_thresh)
    return composite_z, n_high


def _composite_clustered(
    method_results: list[MethodResult],
    params: AggregateParams,
    composite: CompositeParams,
) -> tuple[float, int, dict[str, float]]:
    """Cluster-aware composite.

    Methods are grouped into their evidence clusters (see CLUSTERS); each
    cluster's member z-scores reduce to one score (mean by default — the
    members are redundant noisy readings of the same physical signal), the
    score is winsorized, and the composite is the weighted mean of cluster
    scores.
    ``n_high`` counts CLUSTERS at or above z_thresh, so e.g. ocv_k and temp_k
    both firing no longer double-counts one physical signal toward the
    HIGH gate.

    Returns (composite_z, n_clusters_high, cluster_scores).
    """
    by_cluster: dict[str, list[float]] = {}
    for mr in method_results:
        if np.isnan(mr.z_score):
            continue
        # A method outside every known cluster forms its own singleton
        # cluster (weight 1.0) rather than being silently dropped.
        cluster = _METHOD_TO_CLUSTER.get(mr.method_name, mr.method_name)
        by_cluster.setdefault(cluster, []).append(mr.z_score)

    cluster_scores: dict[str, float] = {}
    for cluster, zs in by_cluster.items():
        score = max(zs) if composite.reduce == "max" else float(np.mean(zs))
        cluster_scores[cluster] = float(
            np.clip(score, -params.winsor_z, params.winsor_z)
        )

    if cluster_scores:
        weights = {
            c: float(getattr(composite, f"weight_{c}", 1.0))
            for c in cluster_scores
        }
        total_w = sum(weights.values())
        composite_z = (
            sum(cluster_scores[c] * weights[c] for c in cluster_scores) / total_w
            if total_w > 0 else 0.0
        )
    else:
        composite_z = 0.0

    n_high = sum(1 for s in cluster_scores.values() if s >= params.z_thresh)
    return float(composite_z), n_high, cluster_scores


def aggregate(
    rest_results: dict[int, list[MethodResult]],
    li_plating_results: dict[int, MethodResult],
    topology: PackTopology,
    isc_results: dict[int, MethodResult] | None = None,
    z_thresh: float = 2.0,
    params: AggregateParams | None = None,
    composite: CompositeParams | None = None,
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
        Ignored when *params* is given (use ``params.z_thresh`` instead).
    params:
        Full aggregation parameter set. Defaults to ``AggregateParams`` built
        from *z_thresh* (which reproduces the historical hardcoded gates).
    composite:
        Composite-mode parameters (clustered vs legacy, reduction, cluster
        weights). Defaults to ``CompositeParams()`` — clustered mode.

    Returns
    -------
    list[ModuleVerdict] ordered by module_id (1..N).
    """
    if params is None:
        params = AggregateParams(z_thresh=z_thresh)
    if composite is None:
        composite = CompositeParams()
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
        # 1. Collect all method results (6 rest + 1 li_plating + optional isc)
        isc_mr = isc_results.get(ch) if isc_results is not None else None
        method_results_list = rest_results[ch] + [li_plating_results[ch]]
        if isc_mr is not None:
            method_results_list = method_results_list + [isc_mr]

        # 2. Both composites are always computed: the mode-selected one
        # drives the verdict; the other is kept for comparability (JSON,
        # calibration, A/B analysis across tool versions).
        legacy_z, legacy_n_high = _composite_legacy(method_results_list, params)
        clustered_z, clusters_high, cluster_scores = _composite_clustered(
            method_results_list, params, composite
        )

        if composite.mode == "legacy":
            composite_z, n_high = legacy_z, legacy_n_high
        else:
            composite_z, n_high = clustered_z, clusters_high

        # 3. Cell verdict
        # n_high gates require composite evidence above a floor so that a
        # single method/cluster barely crossing z_thresh cannot overrule the
        # rest saying NORMAL (defaults):
        #   HIGH:     composite > 2.0, OR (n_high >= 2 AND composite >= 1.0)
        #   ELEVATED: composite > 1.0, OR (n_high >= 1 AND composite >= 0.5)
        if composite_z > params.high_composite or (
            n_high >= params.high_n_methods
            and composite_z >= params.high_composite_floor
        ):
            verdict = "HIGH"
        elif composite_z > params.elevated_composite or (
            n_high >= params.elevated_n_methods
            and composite_z >= params.elevated_composite_floor
        ):
            verdict = "ELEVATED"
        else:
            verdict = "NORMAL"

        cell_verdicts[ch] = CellVerdict(
            channel_index=ch,
            module_id=topology.module_for_channel(ch),
            group_in_module=topology.group_index_in_module(ch),
            composite_z=composite_z,
            n_methods_high=n_high,
            verdict=verdict,
            method_results=method_results_list,
            cluster_scores=cluster_scores,
            composite_z_legacy=legacy_z,
        )

    # --- Module-level rollup ----------------------------------------------------
    module_verdicts: list[ModuleVerdict] = []

    for module_id in range(1, topology.module_count + 1):
        module_channels = topology.channels_in_module(module_id)
        cells = [cell_verdicts[ch] for ch in module_channels if ch in cell_verdicts]
        cells.sort(key=lambda cv: cv.group_in_module)

        # Binary module verdict (strict): any cell above NORMAL — HIGH or
        # ELEVATED — fails the module. Cell-level granularity is kept for
        # diagnostics; the module-level answer is OK or NOK, nothing between.
        flagged_cells = [cv for cv in cells if cv.verdict in ("HIGH", "ELEVATED")]
        verdict = "NOK" if flagged_cells else "OK"

        module_verdicts.append(
            ModuleVerdict(
                module_id=module_id,
                verdict=verdict,
                flagged_cells=flagged_cells,
                all_cells=cells,
            )
        )

    return module_verdicts
