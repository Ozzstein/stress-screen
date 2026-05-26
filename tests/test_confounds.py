"""Cross-mode and cross-protocol confound tests.

Each test injects ONE confounder that historically caused false positives
and asserts the analysis does NOT escalate that cell to HIGH.
"""
import numpy as np
import pandas as pd
import warnings

from stress_screen.analysis.rest import run_rest_analysis, RestParams
from stress_screen.analysis.li_plating import run_li_plating_analysis, LiPlatingParams
from stress_screen.analysis.short_circuit import run_isc_analysis, ShortCircuitParams
from stress_screen.analysis.aggregate import aggregate
from stress_screen.analysis.protocol import ProtocolMetadata
from stress_screen.topology import derive_topology


def _make_charge_df(n_channels, n_points=200, base_temp_c=25.0):
    rows = []
    for ch in range(n_channels):
        t = np.linspace(0, 2.0, n_points)
        frac = t / t[-1]
        V = np.where(
            frac < 0.10, 3.0 + 3.0 * frac,
            np.where(frac < 0.80,
                     3.3 + (frac - 0.10) / 0.70 * 0.10,
                     3.4 + (frac - 0.80) / 0.20 * 0.25)
        )
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": base_temp_c})
    return pd.DataFrame(rows)


def _make_rest_df(n_channels, n_points=500, base_temp_c=25.0):
    rows = []
    t = np.linspace(0, 10.0, n_points)
    rng = np.random.default_rng(42)
    for ch in range(n_channels):
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - 1e-4 * t + rng.normal(0, 1e-4, n_points)
        T = base_temp_c + rng.normal(0, 0.1, n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)


def _make_top_charge(n_points=200, current=5.0):
    return pd.DataFrame({
        "time_hours": np.linspace(0, 2.0, n_points),
        "current": np.full(n_points, current),
    })


def test_warm_cell_not_flagged_as_isc():
    """Hot cell with normal underlying k must not be HIGH on ISC.

    This is the confound the Task 3 fix is designed to address. With Task 3
    applied, the warm cell's k is temperature-corrected before being scored
    by S1, so it should not be the top S1 alarm just because it's warmer.
    """
    n = 8
    rest = _make_rest_df(n, base_temp_c=25.0)
    # Make ch3 warm (+7°C) but with the SAME normal voltage curve
    rest.loc[rest["channel_index"] == 3, "temperature"] = 32.0
    charge = _make_charge_df(n)
    topo = derive_topology(n, 1)
    rest_results = run_rest_analysis(rest, topo)
    isc_results = run_isc_analysis(rest, rest_results, charge)
    # ch3 should not produce a HIGH ISC verdict from temperature alone.
    assert isc_results[3].verdict != "HIGH", (
        f"Warm-only cell should not be ISC HIGH, got {isc_results[3].verdict} "
        f"(s1_z={isc_results[3].metadata['s1_excess_k_z']:.2f})"
    )


def test_cold_cell_without_electrical_signature_does_not_escalate():
    """A cold cell with no electrical anomaly should be ELEVATED at most, not HIGH."""
    n = 8
    rest = _make_rest_df(n)
    charge = _make_charge_df(n)
    # ch3 charges cold (10°C) but is otherwise identical
    charge.loc[charge["channel_index"] == 3, "temperature"] = 10.0
    top_charge = _make_top_charge()
    topo = derive_topology(n, 1)

    rest_results = run_rest_analysis(rest, topo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        li_results = run_li_plating_analysis(charge, rest, top_charge_df=top_charge, n_parallel=1)
    isc_results = run_isc_analysis(rest, rest_results, charge)
    verdicts = aggregate(rest_results, li_results, topo, isc_results=isc_results)

    ch3 = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 3)
    assert ch3.verdict != "HIGH", (
        f"Cold-but-otherwise-healthy cell should not be HIGH; got {ch3.verdict} "
        f"composite_z={ch3.composite_z:.2f}"
    )


def test_high_c_rate_does_not_inflate_dt_late_false_positives():
    """Under 2C protocol, normal small thermal drift should not register as
    heat-of-plating. This validates the c-rate-aware ΔT noise floor (Task 7)."""
    n = 8
    rest = _make_rest_df(n)
    charge = _make_charge_df(n)
    # Add small natural thermal drift to all channels (0.4 K end-of-charge)
    for ch in range(n):
        mask = charge["channel_index"] == ch
        order = charge[mask].sort_values("time_hours").index
        n_per = len(order)
        late_lo = int(0.80 * n_per)
        charge.loc[order[late_lo:], "temperature"] = 25.4  # 0.4 K drift
    top_charge = _make_top_charge()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Under 2C: noise floor = 0.3 + 0.2*1.5 = 0.6 K → 0.4 K is filtered
        results = run_li_plating_analysis(
            charge, rest, top_charge_df=top_charge, n_parallel=1,
            protocol=ProtocolMetadata(c_rate=2.0),
        )
    # No cell should have heat_z populated when ΔT < c-rate-scaled noise floor
    for ch in range(n):
        assert np.isnan(results[ch].metadata["heat_z"]), (
            f"ch{ch}: heat_z should be nan under 2C with 0.4 K drift; "
            f"got {results[ch].metadata['heat_z']}"
        )


def test_simultaneous_plating_and_isc_both_detected():
    """A cell with both plating signatures and high self-discharge should
    show HIGH on the composite (multiple modes firing)."""
    n = 8
    rng = np.random.default_rng(7)
    # Build rest with ch4 having 30x baseline k → HIGH on M1
    t = np.linspace(0, 10, 500)
    rows = []
    for ch in range(n):
        k = 3e-3 if ch == 4 else 1e-4
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - k * t + rng.normal(0, 1e-4, len(t))
        T = 25.0 + rng.normal(0, 0.1, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest = pd.DataFrame(rows)
    # Build charge with ch4 also cold (additional plating signature)
    charge = _make_charge_df(n)
    charge.loc[charge["channel_index"] == 4, "temperature"] = 5.0

    top_charge = _make_top_charge()
    topo = derive_topology(n, 1)
    rest_results = run_rest_analysis(rest, topo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        li_results = run_li_plating_analysis(charge, rest, top_charge_df=top_charge, n_parallel=1)
    isc_results = run_isc_analysis(rest, rest_results, charge)
    verdicts = aggregate(rest_results, li_results, topo, isc_results=isc_results)
    ch4 = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 4)
    assert ch4.verdict == "HIGH", (
        f"Dual-failure cell (plating + ISC) expected HIGH; got {ch4.verdict} "
        f"composite_z={ch4.composite_z:.2f}, n_high={ch4.n_methods_high}"
    )
