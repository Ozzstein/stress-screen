import numpy as np
import pytest
from stress_screen.analysis.aggregate import aggregate
from stress_screen.models import MethodResult, PackTopology
from stress_screen.topology import derive_topology


def _make_method(name, z):
    return MethodResult(method_name=name, z_score=z,
                        verdict="NORMAL", metadata={})


def _build_inputs(z_per_channel):
    """z_per_channel: dict[ch] -> list of 6 rest method z-scores + 1 li_plating z."""
    rest_results = {}
    li_results = {}
    for ch, z_list in z_per_channel.items():
        rest_results[ch] = [
            _make_method(f"M{i+1}", z_list[i]) for i in range(6)
        ]
        li_results[ch] = _make_method("li_plating", z_list[6])
    return rest_results, li_results


def test_composite_preserves_extreme_pathological_z():
    """A cell with one method at z=10 must still register strongly in the
    composite (old behaviour clipped to 5 then averaged with 6 zeros gave 0.71;
    new behaviour clips to 8 → 8/7 ≈ 1.14)."""
    n = 8
    z_map = {ch: [0.0] * 7 for ch in range(n)}
    z_map[0][0] = 10.0  # M1 catastrophic
    rest_r, li_r = _build_inputs(z_map)
    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_r, li_r, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.composite_z > 1.0, (
        f"Pathological z=10 should still register strongly; got composite_z={ch0_cv.composite_z}"
    )


def test_composite_does_not_clip_healthy_negative_z_to_zero():
    """A consistently healthy cell (all z = -2) should produce composite < 0,
    not 0 like before."""
    n = 8
    z_map = {ch: [0.0] * 7 for ch in range(n)}
    z_map[0] = [-2.0] * 7
    rest_r, li_r = _build_inputs(z_map)
    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_r, li_r, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.composite_z < 0.0, (
        f"All-negative-z cell should have negative composite_z; got {ch0_cv.composite_z}"
    )


def test_composite_weights_methods_by_confidence():
    """LEGACY MODE ONLY: methods that publish a `confidence` metadata field
    are weighted accordingly. The clustered composite (default) uses explicit
    per-cluster config weights instead and ignores the confidence hook."""
    n = 8
    rest_results = {}
    li_results = {}
    for ch in range(n):
        # ch0: M1 publishes confidence=1.0 with z=4 (catastrophic, trusted);
        # other methods publish confidence=0.1 with z=0 (noisy, untrusted).
        # Other channels: all zeros, all confidence=1.0.
        if ch == 0:
            m1 = MethodResult("M1", 4.0, "NORMAL", metadata={"confidence": 1.0})
            others = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={"confidence": 0.1})
                for i in range(1, 6)
            ]
            rest_results[ch] = [m1] + others
            li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL",
                                          metadata={"confidence": 0.1})
        else:
            rest_results[ch] = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={"confidence": 1.0})
                for i in range(6)
            ]
            li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL",
                                          metadata={"confidence": 1.0})

    from stress_screen.analysis.aggregate import CompositeParams

    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_results, li_results, topo,
                         composite=CompositeParams(mode="legacy"))
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    # Weighted mean ch0: (4*1.0 + 0*0.1*6) / (1.0 + 6*0.1) = 4/1.6 = 2.5
    # Unweighted (broken) would give: 4/7 ≈ 0.57
    assert ch0_cv.composite_z > 2.0, (
        f"High-confidence z=4 should dominate; got composite_z={ch0_cv.composite_z:.3f}. "
        f"Unweighted would give 0.57; weighted should give ~2.5."
    )


def test_composite_default_confidence_is_unity():
    """When no method publishes 'confidence' metadata, the composite should
    match the unweighted (winsorized) mean — backward-compatible behaviour."""
    n = 8
    z_map = {ch: [0.0] * 7 for ch in range(n)}
    z_map[0] = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rest_r, li_r = _build_inputs(z_map)  # no 'confidence' key
    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_r, li_r, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    # Unweighted mean of 1 method at z=3 + 6 at z=0 = 3/7 ≈ 0.43
    # With default confidence=1.0, weighted result is identical.
    assert abs(ch0_cv.composite_z - 3.0/7.0) < 0.01, (
        f"Default confidence=1.0 must give unweighted mean; got {ch0_cv.composite_z:.3f}, "
        f"expected {3.0/7.0:.3f}"
    )


def test_module_marginal_verdict_for_elevated_only_real():
    """A module with at least one ELEVATED cell (composite_z > 1.0, no method >= 2.0)
    and no HIGH cells should be MARGINAL."""
    n = 8
    rest_results = {}
    li_results = {}
    for ch in range(n):
        if ch == 0:
            # 4 rest methods at z=1.9 (each < 2.0 → n_high=0), 2 at z=0, 1 li at 0
            # composite_z = (1.9*4 + 0*3) / 7 ≈ 1.086 > 1.0 → ELEVATED
            mrs = [
                MethodResult("M1", 1.9, "ELEVATED", metadata={}),
                MethodResult("M2", 1.9, "ELEVATED", metadata={}),
                MethodResult("M3", 1.9, "ELEVATED", metadata={}),
                MethodResult("M4", 1.9, "ELEVATED", metadata={}),
                MethodResult("M5", 0.0, "NORMAL", metadata={}),
                MethodResult("M6", 0.0, "NORMAL", metadata={}),
            ]
            rest_results[ch] = mrs
        else:
            rest_results[ch] = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={}) for i in range(6)
            ]
        li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL", metadata={})

    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_results, li_results, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.verdict == "ELEVATED", (
        f"Setup sanity: ch0 should be ELEVATED. Got {ch0_cv.verdict}, composite_z={ch0_cv.composite_z:.3f}"
    )
    assert verdicts[0].verdict == "MARGINAL", (
        f"Module with one ELEVATED cell expected MARGINAL, got {verdicts[0].verdict}"
    )


def test_module_nok_overrides_marginal_when_any_high():
    """Any HIGH cell still produces NOK regardless of how many MARGINAL cells.

    Uses 4 methods at z=2.5 so that composite_z = (4*2.5)/7 ≈ 1.43 ≥ 1.0
    and n_high=4 ≥ 2 — satisfying both conditions of the n_high gate.
    """
    n = 8
    rest_results = {}
    li_results = {}
    for ch in range(n):
        if ch == 0:
            # 4 methods above z_thresh, composite_z = 4*2.5/7 ≈ 1.43 → HIGH
            rest_results[ch] = [
                MethodResult("M1", 2.5, "HIGH", metadata={}),
                MethodResult("M2", 2.5, "HIGH", metadata={}),
                MethodResult("M3", 2.5, "HIGH", metadata={}),
                MethodResult("M4", 2.5, "HIGH", metadata={}),
                MethodResult("M5", 0.0, "NORMAL", metadata={}),
                MethodResult("M6", 0.0, "NORMAL", metadata={}),
            ]
        elif ch == 1:
            rest_results[ch] = [
                MethodResult("M1", 1.9, "ELEVATED", metadata={}),
                MethodResult("M2", 1.9, "ELEVATED", metadata={}),
                MethodResult("M3", 1.9, "ELEVATED", metadata={}),
                MethodResult("M4", 1.9, "ELEVATED", metadata={}),
                MethodResult("M5", 0.0, "NORMAL", metadata={}),
                MethodResult("M6", 0.0, "NORMAL", metadata={}),
            ]
        else:
            rest_results[ch] = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={}) for i in range(6)
            ]
        li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL", metadata={})

    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_results, li_results, topo)
    assert verdicts[0].verdict == "NOK", f"Expected NOK with HIGH cell, got {verdicts[0].verdict}"


def test_two_borderline_high_methods_with_low_composite_z_is_elevated_not_nok():
    """Two methods barely above z_thresh but composite_z < 1.0 must be ELEVATED.

    This guards against two weakly-firing methods overruling five methods
    saying NORMAL.  composite_z = (2.1 + 2.1 + 0*5) / 7 ≈ 0.60 < 1.0.
    """
    n = 8
    rest_results = {}
    li_results = {}
    for ch in range(n):
        if ch == 0:
            rest_results[ch] = [
                MethodResult("M1", 2.1, "HIGH", metadata={}),
                MethodResult("M2", 2.1, "HIGH", metadata={}),
                MethodResult("M3", 0.0, "NORMAL", metadata={}),
                MethodResult("M4", 0.0, "NORMAL", metadata={}),
                MethodResult("M5", 0.0, "NORMAL", metadata={}),
                MethodResult("M6", 0.0, "NORMAL", metadata={}),
            ]
        else:
            rest_results[ch] = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={}) for i in range(6)
            ]
        li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL", metadata={})

    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_results, li_results, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.verdict == "ELEVATED", (
        f"Two borderline HIGH methods with composite_z={ch0_cv.composite_z:.2f} < 1.0 "
        f"should be ELEVATED, not {ch0_cv.verdict}"
    )
    assert verdicts[0].verdict == "MARGINAL", (
        f"Module with only ELEVATED cells should be MARGINAL, got {verdicts[0].verdict}"
    )


def test_single_borderline_high_with_low_composite_is_normal_not_elevated():
    """One method just above z_thresh with composite < 0.5 is NORMAL.

    Guard against single-method threshold trips driving false ELEVATED.
    composite_z = (2.1 + 0*6) / 7 ≈ 0.30 < 0.5.
    """
    n = 8
    rest_results = {}
    li_results = {}
    for ch in range(n):
        if ch == 0:
            rest_results[ch] = [
                MethodResult("M1", 2.1, "HIGH", metadata={}),
                MethodResult("M2", 0.0, "NORMAL", metadata={}),
                MethodResult("M3", 0.0, "NORMAL", metadata={}),
                MethodResult("M4", 0.0, "NORMAL", metadata={}),
                MethodResult("M5", 0.0, "NORMAL", metadata={}),
                MethodResult("M6", 0.0, "NORMAL", metadata={}),
            ]
        else:
            rest_results[ch] = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={}) for i in range(6)
            ]
        li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL", metadata={})

    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_results, li_results, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.verdict == "NORMAL", (
        f"Single borderline HIGH with composite_z={ch0_cv.composite_z:.2f} < 0.5 "
        f"should be NORMAL, not {ch0_cv.verdict}"
    )


def test_single_high_with_moderate_composite_is_elevated():
    """One HIGH method with composite >= 0.5 produces ELEVATED.

    Boundary check: composite = (2.5 + 1.0 + 1.0 + 0*4) / 7 ≈ 0.64 > 0.5.
    """
    n = 8
    rest_results = {}
    li_results = {}
    for ch in range(n):
        if ch == 0:
            rest_results[ch] = [
                MethodResult("M1", 2.5, "HIGH", metadata={}),
                MethodResult("M2", 1.0, "ELEVATED", metadata={}),
                MethodResult("M3", 1.0, "ELEVATED", metadata={}),
                MethodResult("M4", 0.0, "NORMAL", metadata={}),
                MethodResult("M5", 0.0, "NORMAL", metadata={}),
                MethodResult("M6", 0.0, "NORMAL", metadata={}),
            ]
        else:
            rest_results[ch] = [
                MethodResult(f"M{i+1}", 0.0, "NORMAL", metadata={}) for i in range(6)
            ]
        li_results[ch] = MethodResult("li_plating", 0.0, "NORMAL", metadata={})

    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_results, li_results, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.verdict == "ELEVATED", (
        f"n_high=1 + composite_z={ch0_cv.composite_z:.2f} ≥ 0.5 should be ELEVATED, "
        f"got {ch0_cv.verdict}"
    )
