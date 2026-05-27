import numpy as np
import pandas as pd
import pytest
from stress_screen.analysis.rest import run_rest_analysis, RestParams
from stress_screen.topology import derive_topology

def _make_rest_cell_df(n_channels=8, n_points=500, bad_channels=None, k_bad=0.002, rng=None):
    """Synthetic OCV rest data. bad_channels is a list of channel_indices with high k (self-discharge)."""
    if rng is None:
        rng = np.random.default_rng(42)
    bad_channels = bad_channels or []
    rows = []
    t = np.linspace(0, 10, n_points)  # 10 hours of rest
    for ch in range(n_channels):
        k = k_bad if ch in bad_channels else 0.0001
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - k * t + rng.normal(0, 0.0001, n_points)
        T = 25.0 + rng.normal(0, 0.1, n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch, "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)

def test_rest_analysis_returns_all_channels():
    rng = np.random.default_rng(42)
    rest_df = _make_rest_cell_df(n_channels=8, rng=rng)
    topo = derive_topology(8, 1)  # 8 channels, 1 module -> 4P8S
    results = run_rest_analysis(rest_df, topo)
    assert len(results) == 8

def test_m1_flags_high_k_cells():
    rng = np.random.default_rng(42)
    bad = [2, 5]
    rest_df = _make_rest_cell_df(n_channels=16, bad_channels=bad, k_bad=0.003, rng=rng)
    topo = derive_topology(16, 1)  # 16 channels, 1 module -> 2P16S
    results = run_rest_analysis(rest_df, topo)
    m1_verdicts = {ch: next(mr.verdict for mr in mrs if mr.method_name == "ocv_k")
                   for ch, mrs in results.items()}
    # Both bad channels should be HIGH
    for ch in bad:
        assert m1_verdicts[ch] in ("HIGH", "ELEVATED"), f"ch {ch} expected HIGH/ELEVATED, got {m1_verdicts[ch]}"


def test_m5_arrhenius_vs_linear():
    """M5 Arrhenius-corrected k at 35°C must differ from the old linear approx by >1%."""
    rng = np.random.default_rng(42)
    rest_df = _make_rest_cell_df(n_channels=8, rng=rng)
    rest_df["temperature"] = 35.0
    topo = derive_topology(8, 1)
    results = run_rest_analysis(rest_df, topo)

    checked = False
    for ch, mrs in results.items():
        m1 = next((mr for mr in mrs if mr.method_name == "ocv_k"), None)
        m5 = next((mr for mr in mrs if mr.method_name == "temp_k"), None)
        if m1 is None or m5 is None:
            continue
        k_raw = m1.metadata.get("k", float("nan"))
        k_corr = m5.metadata.get("k_corrected", float("nan"))
        if np.isnan(k_raw) or np.isnan(k_corr) or k_raw < 1e-10:
            continue
        k_linear = k_raw / (1.0 + 0.02 * (35.0 - 25.0))  # old formula: k / 1.2
        assert abs(k_corr - k_linear) / k_linear > 0.01, (
            f"ch{ch}: Arrhenius {k_corr:.8f} vs linear {k_linear:.8f} — "
            f"difference {100 * abs(k_corr - k_linear) / k_linear:.2f}% must be >1%"
        )
        # At T > T_ref, Arrhenius normalisation reduces k toward 25°C baseline
        assert k_corr < k_raw, (
            f"ch{ch}: k_corr={k_corr:.8f} should be < k_raw={k_raw:.8f} at 35°C (above T_ref=25°C)"
        )
        checked = True
        break

    assert checked, "No channel produced valid M1+M5 data — check _make_rest_cell_df or min_points"


def test_m6_slope_penalises_trending_cell():
    """Channel with declining rank but frac_bot20=0 gets higher M6 z after slope fix."""
    rng = np.random.default_rng(0)
    n_channels = 8
    t = np.linspace(0, 10, 600)
    rows = []
    for ch in range(n_channels):
        if ch == 0:
            # starts highest (~87th pct), drifts toward 3rd rank — never below 20th pct
            V = 3.450 + 0.050 * np.exp(-t / 2.0) - 0.004 * t + rng.normal(0, 1e-4, len(t))
        elif ch in (1, 2):
            # permanently lowest (rank 1 and 2 → always below 20th pct with 8 channels)
            V = 3.395 + (ch - 1) * 0.002 + 0.050 * np.exp(-t / 2.0) + rng.normal(0, 1e-4, len(t))
        else:
            # stable middle channels, slope ≈ 0
            V = 3.420 + (ch - 3) * 0.003 + 0.050 * np.exp(-t / 2.0) + rng.normal(0, 1e-4, len(t))
        T = 25.0 + rng.normal(0, 0.05, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest_df = pd.DataFrame(rows)
    topo = derive_topology(n_channels, 1)
    results = run_rest_analysis(rest_df, topo)

    m6_z_ch0 = next(mr.z_score for mr in results[0] if mr.method_name == "rank")
    # Only compare against stable middle channels (ch3–ch7); they have near-zero slope
    m6_z_stable = [
        next(mr.z_score for mr in results[ch] if mr.method_name == "rank")
        for ch in range(3, 8)
    ]
    assert m6_z_ch0 > float(np.nanmedian(m6_z_stable)) + 0.5, (
        f"Trending ch0 M6 z={m6_z_ch0:.3f} should be > stable median "
        f"{np.nanmedian(m6_z_stable):.3f} + 0.5"
    )

    # Verify the slope term is actually contributing — disable it and check ch0 z drops
    from stress_screen.analysis.rest import RestParams
    results_no_slope = run_rest_analysis(rest_df, topo, params=RestParams(m6_slope_weight=0.0))
    m6_z_ch0_no_slope = next(mr.z_score for mr in results_no_slope[0] if mr.method_name == "rank")
    assert m6_z_ch0 > m6_z_ch0_no_slope, (
        f"Slope term must be raising ch0 M6 z: with_slope={m6_z_ch0:.3f}, "
        f"no_slope={m6_z_ch0_no_slope:.3f}"
    )


def test_m5_uses_arrhenius_helper(monkeypatch):
    """M5 must delegate temperature correction to arrhenius_correction()."""
    rng = np.random.default_rng(42)
    rest_df = _make_rest_cell_df(n_channels=8, rng=rng)
    rest_df["temperature"] = 35.0
    topo = derive_topology(8, 1)

    calls = []
    from stress_screen.analysis import rest as rest_module
    real_helper = rest_module.arrhenius_correction

    def spy(T_celsius, ea_ev):
        calls.append((T_celsius, ea_ev))
        return real_helper(T_celsius, ea_ev)

    monkeypatch.setattr(rest_module, "arrhenius_correction", spy)
    run_rest_analysis(rest_df, topo, params=RestParams(arrhenius_ea_ev=0.5))
    assert len(calls) > 0, "M5 must call arrhenius_correction()"
    # Every call must pass the configured Ea
    assert all(abs(c[1] - 0.5) < 1e-9 for c in calls)


def test_m3_flags_drifting_cell_over_static_offset():
    """A cell drifting away from fleet during rest should get higher M3 z than
    a cell that's just statically offset by the same average distance."""
    rng = np.random.default_rng(123)
    n = 8
    t = np.linspace(0, 10, 600)
    rows = []
    for ch in range(n):
        if ch == 0:
            # Drifting cell: smooth linear drift away from fleet, LOW noise.
            # Under NEW M3 the |deviation| has a clear positive slope.
            V = 3.420 + 0.050 * np.exp(-t / 2.0) - 0.002 * t + rng.normal(0, 1e-4, len(t))
        elif ch == 1:
            # Static offset with HIGH wander: large noise around a fixed offset.
            # Under OLD M3 (static std), this cell has high spread.
            # Under NEW M3 (divergence slope), the wander averages out → low slope.
            V = 3.400 + 0.050 * np.exp(-t / 2.0) + rng.normal(0, 5e-3, len(t))
        else:
            V = 3.420 + (ch - 2) * 0.001 + 0.050 * np.exp(-t / 2.0) + rng.normal(0, 1e-4, len(t))
        T = 25.0 + rng.normal(0, 0.05, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest_df = pd.DataFrame(rows)
    topo = derive_topology(n, 1)
    results = run_rest_analysis(rest_df, topo)
    m3_z = {ch: next(mr.z_score for mr in results[ch] if mr.method_name == "spread")
            for ch in range(n)}
    assert m3_z[0] > m3_z[1], (
        f"Drifting ch0 M3 z={m3_z[0]:.3f} must exceed static-offset ch1 M3 z={m3_z[1]:.3f}"
    )


def test_m3_noise_floor_prevents_false_high_on_healthy_fleet():
    """When all cells have near-zero divergence slopes (healthy fleet), no cell
    should reach HIGH on M3 — even if a few have tiny nonzero slopes that are
    well within measurement noise (< 5 µV/h = default min_mad floor)."""
    rng = np.random.default_rng(99)
    n = 16
    t = np.linspace(0, 10.0, 500)
    rows = []
    for ch in range(n):
        # All cells follow the same OCV model; ch0 has a tiny 2 µV/h divergence
        # (well below the 5 µV/h floor) that would produce z~millions without a floor.
        tiny_slope = 2e-6 if ch == 0 else 0.0
        V = 3.42 + 0.05 * np.exp(-t / 2.0) - 1e-4 * t - tiny_slope * t + rng.normal(0, 1e-4, len(t))
        T = 25.0 + rng.normal(0, 0.05, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest_df = pd.DataFrame(rows)
    topo = derive_topology(n, 1)
    results = run_rest_analysis(rest_df, topo)
    m3_verdict_ch0 = next(mr.verdict for mr in results[0] if mr.method_name == "spread")
    m3_z_ch0 = next(mr.z_score for mr in results[0] if mr.method_name == "spread")
    assert m3_verdict_ch0 != "HIGH", (
        f"Noise-level 2 µV/h slope should not be HIGH; got verdict={m3_verdict_ch0}, z={m3_z_ch0:.2f}"
    )


def test_m3_temperature_gradient_not_flagged():
    """A cell that progressively warms relative to its fleet-mates during rest must
    NOT be flagged as HIGH by M3 when temperature compensation is enabled.

    Without correction the growing temperature gap looks like accelerating
    self-discharge (slope ≈ 125 µV/h); with −0.2 mV/°C compensation the
    corrected voltage tracks the fleet and the divergence slope drops to ≈ 0.
    """
    rng = np.random.default_rng(12)
    n = 8
    t = np.linspace(0, 10.0, 500)
    rows = []
    for ch in range(n):
        # ch0: starts +5°C above fleet and warms a further +5°C over 10 h.
        # All cells have the same underlying self-discharge rate (1e-4 V/h).
        # ch0's measured voltage is depressed by the OCV-temperature effect.
        T = (25.0 + 5.0 + 0.625 * t) if ch == 0 else np.full(len(t), 25.0)
        V_true = 3.4 + 0.05 * np.exp(-t / 2.0) - 1e-4 * t
        V_measured = V_true - 0.0002 * (T - 25.0) + rng.normal(0, 5e-5, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": float(V_measured[i]),
                         "temperature": float(T if np.isscalar(T) else T[i])})
    rest_df = pd.DataFrame(rows)
    topo = derive_topology(n, 1)

    results_with = run_rest_analysis(rest_df, topo, params=RestParams(dv_dt_coeff_mv_per_c=-0.2))
    results_without = run_rest_analysis(rest_df, topo, params=RestParams(dv_dt_coeff_mv_per_c=0.0))

    z_with = next(mr.z_score for mr in results_with[0] if mr.method_name == "spread")
    z_without = next(mr.z_score for mr in results_without[0] if mr.method_name == "spread")

    assert z_without > z_with, (
        f"T-compensation must reduce ch0 M3 z: without={z_without:.3f}, with={z_with:.3f}"
    )
    verdict_with = next(mr.verdict for mr in results_with[0] if mr.method_name == "spread")
    assert verdict_with != "HIGH", (
        f"ch0 must not be HIGH after T-compensation, got {verdict_with} (z={z_with:.3f})"
    )


def test_m6_slope_cap_score_formula():
    """The cap must bound the raw slope contribution stored in M6 metadata.

    For any cell whose raw_slope_contrib exceeds the cap, the capped value must
    be strictly less than the raw value.  For cells that don't hit the cap,
    capped == raw.  This tests the score-level computation directly and is
    unaffected by the z-score normalisation step.
    """
    rng = np.random.default_rng(42)
    n = 16
    t = np.linspace(2.5, 60.0, 800)

    rows = []
    for ch in range(n):
        k = 1.3e-4 if ch == 0 else 1e-4
        V = 3.40 - k * t + rng.normal(0, 3e-5, len(t))
        T = 25.0 + rng.normal(0, 0.05, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})

    rest_df = pd.DataFrame(rows)
    topo = derive_topology(n, 2)  # 2 modules of 8

    cap = 0.05
    results = run_rest_analysis(rest_df, topo, params=RestParams(m6_max_slope_contribution=cap))

    cap_active_on_any = False
    for ch in range(n):
        m6 = next(mr for mr in results[ch] if mr.method_name == "rank")
        raw = m6.metadata.get("slope_contribution_raw", float("nan"))
        capped = m6.metadata.get("slope_contribution_capped", float("nan"))
        if np.isnan(raw):
            continue
        if raw > cap:
            assert capped == pytest.approx(cap, abs=1e-9), (
                f"ch{ch}: raw={raw:.4f} > cap={cap} but capped={capped:.4f} != cap"
            )
            cap_active_on_any = True
        else:
            assert capped == pytest.approx(raw, abs=1e-9), (
                f"ch{ch}: raw={raw:.4f} ≤ cap={cap} but capped={capped:.4f} != raw"
            )

    assert cap_active_on_any, (
        "No channel had raw_slope_contrib > cap — increase drift or lower cap"
    )
