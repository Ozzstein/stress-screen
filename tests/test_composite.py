"""Unit tests for the cluster-aware composite (aggregate._composite_clustered)."""

from __future__ import annotations

import math

import pytest

from stress_screen.analysis.aggregate import (
    AggregateParams,
    CompositeParams,
    _composite_clustered,
    _composite_legacy,
)
from stress_screen.models import MethodResult


def _mr(name: str, z: float) -> MethodResult:
    return MethodResult(method_name=name, z_score=z, verdict="NORMAL", metadata={})


REAL_METHODS = ["ocv_k", "thermal_corr", "spread", "cusum", "temp_k", "rank",
                "li_plating", "isc"]


def _cell(zs: dict[str, float]) -> list[MethodResult]:
    return [_mr(name, zs.get(name, 0.0)) for name in REAL_METHODS]


def test_correlated_pair_counts_once_in_n_high():
    """ocv_k and temp_k both firing is ONE cluster vote, not two."""
    mrs = _cell({"ocv_k": 3.0, "temp_k": 3.0})
    _, n_high, scores = _composite_clustered(mrs, AggregateParams(), CompositeParams())
    assert n_high == 1
    assert scores["self_discharge"] == pytest.approx(3.0)


def test_mean_reduce_averages_disagreeing_members():
    """rank firing while spread is quiet must not carry the whole cluster."""
    mrs = _cell({"rank": 2.8, "spread": -0.1})
    _, n_high, scores = _composite_clustered(mrs, AggregateParams(), CompositeParams())
    assert scores["fleet_divergence"] == pytest.approx((2.8 - 0.1) / 2)
    assert n_high == 0


def test_max_reduce_takes_strongest_member():
    mrs = _cell({"rank": 2.8, "spread": -0.1})
    comp = CompositeParams(reduce="max")
    _, n_high, scores = _composite_clustered(mrs, AggregateParams(), comp)
    assert scores["fleet_divergence"] == pytest.approx(2.8)
    assert n_high == 1


def test_nan_members_skipped():
    mrs = _cell({"ocv_k": 2.0})
    mrs[4] = _mr("temp_k", float("nan"))  # temp_k NaN → cluster = ocv_k alone
    _, _, scores = _composite_clustered(mrs, AggregateParams(), CompositeParams())
    assert scores["self_discharge"] == pytest.approx(2.0)


def test_all_nan_cluster_absent():
    mrs = [_mr("ocv_k", float("nan")), _mr("temp_k", float("nan")),
           _mr("li_plating", 1.0)]
    composite, _, scores = _composite_clustered(
        mrs, AggregateParams(), CompositeParams()
    )
    assert "self_discharge" not in scores
    assert composite == pytest.approx(1.0)  # only li_plating cluster remains


def test_missing_isc_handled():
    mrs = [m for m in _cell({"ocv_k": 1.0}) if m.method_name != "isc"]
    composite, _, scores = _composite_clustered(
        mrs, AggregateParams(), CompositeParams()
    )
    assert "isc" not in scores
    assert not math.isnan(composite)


def test_unknown_method_becomes_singleton_cluster():
    mrs = _cell({}) + [_mr("future_method", 4.0)]
    _, n_high, scores = _composite_clustered(mrs, AggregateParams(), CompositeParams())
    assert scores["future_method"] == pytest.approx(4.0)
    assert n_high == 1


def test_cluster_weights_shift_composite():
    mrs = _cell({"li_plating": 4.0})
    base, _, _ = _composite_clustered(mrs, AggregateParams(), CompositeParams())
    boosted, _, _ = _composite_clustered(
        mrs, AggregateParams(), CompositeParams(weight_li_plating=2.0)
    )
    assert boosted > base


def test_cluster_scores_winsorized():
    mrs = _cell({"isc": 100.0})
    _, _, scores = _composite_clustered(mrs, AggregateParams(), CompositeParams())
    assert scores["isc"] == pytest.approx(8.0)  # winsor_z default


def test_independent_signals_not_diluted_vs_legacy():
    """A strong li_plating signal carries more weight per-cluster (1 of 5)
    than per-method (1 of 8) — the dilution fix."""
    mrs = _cell({"li_plating": 5.0})
    legacy_z, _ = _composite_legacy(mrs, AggregateParams())
    clustered_z, _, _ = _composite_clustered(
        mrs, AggregateParams(), CompositeParams()
    )
    assert clustered_z > legacy_z
    assert clustered_z == pytest.approx(1.0)   # 5.0 / 5 clusters
    assert legacy_z == pytest.approx(5.0 / 8)  # 5.0 / 8 methods


def test_legacy_mode_bit_compatible():
    """_composite_legacy must reproduce the historical plain mean exactly."""
    mrs = _cell({"ocv_k": 3.0, "rank": -1.0, "isc": 0.5})
    legacy_z, n_high = _composite_legacy(mrs, AggregateParams())
    assert legacy_z == pytest.approx((3.0 - 1.0 + 0.5) / 8)
    assert n_high == 1
