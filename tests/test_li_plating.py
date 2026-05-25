import pandas as pd
import numpy as np
from stress_screen.analysis.li_plating import run_li_plating_analysis, LiPlatingParams

def _make_charge_df(n_channels=5, n_points=200):
    """LFP-like charge profile: main plateau + fast end-of-charge rise.

    Channel 0 has a secondary plateau at 3.5 V injected into the
    end-of-charge region — simulating a plating-induced dQ/dV extra peak
    above the main LFP plateau.
    """
    rows = []
    for ch in range(n_channels):
        t = np.linspace(0, 2.0, n_points)
        frac = t / t[-1]
        # Pre-plateau (0–10 %): 3.0 → 3.3 V
        # Main LFP plateau (10–80 %): 3.3 → 3.4 V (slow rise)
        # End-of-charge (80–100 %): 3.4 → 3.65 V (fast rise)
        V = np.where(
            frac < 0.10,
            3.0 + 3.0 * frac,
            np.where(
                frac < 0.80,
                3.3 + (frac - 0.10) / 0.70 * 0.10,
                3.4 + (frac - 0.80) / 0.20 * 0.25,
            ),
        )
        if ch == 0:
            # Replace first half of end-of-charge with a flat plateau at 3.5 V.
            # In dQ/dV (voltage-domain) this creates a secondary peak above the
            # main plateau — the expected Li-plating signature.
            plateau_start = int(0.80 * n_points)
            plateau_end = int(0.90 * n_points)
            V = V.copy()
            V[plateau_start:plateau_end] = 3.5
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": 25.0})
    return pd.DataFrame(rows)

def _make_rest_df(n_channels=5, n_points=100):
    rows = []
    for ch in range(n_channels):
        t = np.linspace(0, 0.5, n_points)
        V = 3.5 * np.exp(-t / 0.2) + 0.0 + 3.2  # fast decay for all
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch, "voltage": V[i], "temperature": 25.0})
    return pd.DataFrame(rows)

def test_li_plating_returns_all_channels():
    charge = _make_charge_df()
    rest = _make_rest_df()
    results = run_li_plating_analysis(charge, rest)
    assert len(results) == 5
    assert all(r.method_name == "li_plating" for r in results.values())

def test_li_plating_metadata_keys():
    charge = _make_charge_df()
    rest = _make_rest_df()
    results = run_li_plating_analysis(charge, rest)
    for mr in results.values():
        assert "dqdv_z" in mr.metadata
        assert "relaxation_z" in mr.metadata
        assert "charge_time_z" in mr.metadata
        # New temperature-related metadata keys
        assert "cold_z" in mr.metadata
        assert "heat_z" in mr.metadata
        assert "T_mean_charge" in mr.metadata
        assert "dT_late" in mr.metadata
        assert "temperature_gate" in mr.metadata
        assert "gated_dqdv_z" in mr.metadata
        assert "gated_relaxation_z" in mr.metadata
        assert "gated_charge_time_z" in mr.metadata


def test_cold_cell_flagged():
    """A cell charging at low T should fire the cold_z signature."""
    charge = _make_charge_df(n_channels=8)
    rest = _make_rest_df(n_channels=8)
    # Make channel 3 cold during charge: 5°C instead of 25°C
    charge.loc[charge["channel_index"] == 3, "temperature"] = 5.0
    results = run_li_plating_analysis(charge, rest)
    ch3 = results[3]
    others = [results[c] for c in range(8) if c != 3]
    # cold_z for ch3 should be much higher than the median of others
    other_cold = [r.metadata["cold_z"] for r in others]
    assert ch3.metadata["cold_z"] > float(np.nanmedian(other_cold)) + 0.5, \
        f"Cold cell cold_z={ch3.metadata['cold_z']:.3f} not significantly above others median {np.nanmedian(other_cold):.3f}"
    # Temperature gate should be active for ch3
    assert ch3.metadata["temperature_gate"] > 0.5, f"gate={ch3.metadata['temperature_gate']}"


def test_injected_cell_has_higher_dqdv_z():
    """Channel 0's secondary plateau creates a higher dqdv_z than healthy peers."""
    charge = _make_charge_df(n_channels=8)
    rest = _make_rest_df(n_channels=8)
    top_charge = _make_top_charge_df(current=5.0)
    results = run_li_plating_analysis(charge, rest, top_charge_df=top_charge, n_parallel=1)
    ch0_dqdv_z = results[0].metadata["dqdv_z"]
    other_dqdv_z = [results[ch].metadata["dqdv_z"] for ch in range(1, 8)]
    assert ch0_dqdv_z > float(np.nanmedian(other_dqdv_z)), (
        f"Expected ch0 dqdv_z={ch0_dqdv_z:.3f} > median others "
        f"{np.nanmedian(other_dqdv_z):.3f}"
    )


import warnings


def _make_top_charge_df(n_points=200, current=5.0):
    t = np.linspace(0, 2.0, n_points)
    return pd.DataFrame({"time_hours": t, "current": np.full(n_points, current)})


def test_dvdq_uses_q_domain():
    """With Q data, ch0's secondary plateau produces a non-zero dqdv_extra_peak_sum;
    without Q data the metric returns 0 (requires Q for voltage-domain dQ/dV)."""
    charge = _make_charge_df(n_channels=5)
    rest = _make_rest_df(n_channels=5)
    top_charge = _make_top_charge_df(current=5.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_no_q = run_li_plating_analysis(charge, rest)
    results_q = run_li_plating_analysis(charge, rest, top_charge_df=top_charge, n_parallel=1)

    # Without Q data the metric is undefined → 0
    assert results_no_q[0].metadata["dqdv_extra_peak_sum"] == 0.0, (
        "dqdv_extra_peak_sum must be 0 when top_charge_df is absent"
    )
    # With Q data, ch0's secondary plateau creates a detectable extra peak
    assert results_q[0].metadata["dqdv_extra_peak_sum"] > 0.0, (
        "Injected secondary plateau on ch0 should produce non-zero dqdv_extra_peak_sum"
    )


def test_dvdq_fallback_no_top_df():
    """Omitting top_charge_df issues a warning but returns normally."""
    charge = _make_charge_df()
    rest = _make_rest_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run_li_plating_analysis(charge, rest)
    assert len(results) == 5
    messages = " ".join(str(w.message) for w in caught).lower()
    assert "top_charge_df" in messages or "q data" in messages or "q-domain" in messages, (
        f"Expected Q-domain fallback warning, got: {[str(w.message) for w in caught]}"
    )


def test_t_threshold_20c():
    """Default T_plating_threshold_c=20°C anchors gate=1.0; 19°C slightly higher,
    21°C slightly lower; warm cell (30°C) gate is heavily suppressed."""
    charge = _make_charge_df(n_channels=4)
    rest = _make_rest_df(n_channels=4)
    charge.loc[charge["channel_index"] == 0, "temperature"] = 19.0
    charge.loc[charge["channel_index"] == 1, "temperature"] = 21.0
    charge.loc[charge["channel_index"] == 2, "temperature"] = 30.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    g19 = results[0].metadata["temperature_gate"]
    g21 = results[1].metadata["temperature_gate"]
    g30 = results[2].metadata["temperature_gate"]
    assert g19 > g21, f"19°C gate {g19:.4f} should exceed 21°C gate {g21:.4f}"
    assert g30 < 0.5, f"30°C gate {g30:.4f} should be heavily suppressed (<0.5)"


def test_dt_late_noise_guard():
    """dT_late < 0.3°C should produce heat_z = nan (sub-noise signal discarded)."""
    charge = _make_charge_df(n_channels=5)  # uniform T=25°C → dT_late=0.0 for all
    rest = _make_rest_df(n_channels=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    for ch in range(5):
        assert np.isnan(results[ch].metadata["heat_z"]), (
            f"ch{ch}: expected heat_z=nan for |dT_late|<0.3°C, "
            f"got heat_z={results[ch].metadata['heat_z']}"
        )


def test_dt_late_noise_guard_positive_case():
    """dT_late = 0.5°C (>= 0.3°C guard) should produce finite heat_z for that channel."""
    import warnings
    charge = _make_charge_df(n_channels=5)
    n_per_ch = len(charge[charge["channel_index"] == 0])
    i_mid_lo = int(np.floor(0.60 * n_per_ch))
    i_mid_hi = int(np.floor(0.80 * n_per_ch))
    i_late_lo = int(np.floor(0.80 * n_per_ch))

    # Set channel 0: mid window = 25°C, late window = 25.5°C → dT_late = 0.5°C
    mask_ch0 = charge["channel_index"] == 0
    ch0_sorted_idx = charge[mask_ch0].sort_values("time_hours").index
    charge.loc[ch0_sorted_idx[i_mid_lo:i_mid_hi], "temperature"] = 25.0
    charge.loc[ch0_sorted_idx[i_late_lo:], "temperature"] = 25.5

    rest = _make_rest_df(n_channels=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)

    assert not np.isnan(results[0].metadata["heat_z"]), (
        "ch0: dT_late=0.5°C (>= 0.3°C guard) should produce finite heat_z"
    )


def test_arrhenius_gate_at_threshold_is_unity():
    """At the plating threshold temperature (default 20°C), gate must equal 1.0."""
    charge = _make_charge_df(n_channels=3)
    rest = _make_rest_df(n_channels=3)
    charge.loc[charge["channel_index"] == 0, "temperature"] = 20.0  # exactly threshold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    gate = results[0].metadata["temperature_gate"]
    assert abs(gate - 1.0) < 1e-6, f"Gate at threshold expected 1.0, got {gate:.6f}"


def test_arrhenius_gate_above_threshold_is_suppressed():
    """Cells warmer than threshold by >5K should have gate substantially <1."""
    charge = _make_charge_df(n_channels=3)
    rest = _make_rest_df(n_channels=3)
    charge.loc[charge["channel_index"] == 0, "temperature"] = 30.0  # 10°C above
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    gate = results[0].metadata["temperature_gate"]
    assert gate < 0.5, f"Gate >5K above threshold expected <0.5, got {gate:.4f}"


def test_arrhenius_gate_cold_cell_boosted_vs_mild_cold():
    """Cold cell (5°C, 15K below threshold) should produce gate > 2x the gate at 15°C."""
    charge = _make_charge_df(n_channels=3)
    rest = _make_rest_df(n_channels=3)
    charge.loc[charge["channel_index"] == 0, "temperature"] = 5.0
    charge.loc[charge["channel_index"] == 1, "temperature"] = 15.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    gate_5 = results[0].metadata["temperature_gate"]
    gate_15 = results[1].metadata["temperature_gate"]
    assert gate_5 > 2.0 * gate_15, (
        f"Arrhenius gate should be much higher at 5°C than 15°C: "
        f"gate_5={gate_5:.3f}, gate_15={gate_15:.3f}"
    )


def test_protocol_scales_dt_noise_guard():
    """Same ΔT_late signal that's discarded under 0.2C protocol should be kept
    under 0.5C protocol (different noise floors)."""
    from stress_screen.analysis.protocol import ProtocolMetadata
    charge = _make_charge_df(n_channels=5)
    # Inject ch0 ΔT_late = 0.4 K (between 0.3 K and 0.6 K)
    n_per_ch = len(charge[charge["channel_index"] == 0])
    i_mid_lo = int(np.floor(0.60 * n_per_ch))
    i_mid_hi = int(np.floor(0.80 * n_per_ch))
    i_late_lo = int(np.floor(0.80 * n_per_ch))
    mask_ch0 = charge["channel_index"] == 0
    ch0_idx = charge[mask_ch0].sort_values("time_hours").index
    charge.loc[ch0_idx[i_mid_lo:i_mid_hi], "temperature"] = 25.0
    charge.loc[ch0_idx[i_late_lo:], "temperature"] = 25.4
    rest = _make_rest_df(n_channels=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Default 0.5C protocol: noise floor 0.3 K → 0.4 K passes
        normal_protocol = run_li_plating_analysis(
            charge, rest, params=LiPlatingParams(),
            protocol=ProtocolMetadata(c_rate=0.5),
        )
        # Fast 2.0C protocol: noise floor 0.3 + 0.2*1.5 = 0.6 K → 0.4 K is filtered
        fast_protocol = run_li_plating_analysis(
            charge, rest, params=LiPlatingParams(),
            protocol=ProtocolMetadata(c_rate=2.0),
        )
    assert not np.isnan(normal_protocol[0].metadata["heat_z"]), \
        "0.5C protocol: 0.4 K ΔT must pass 0.3 K noise floor → finite heat_z"
    assert np.isnan(fast_protocol[0].metadata["heat_z"]), \
        "2.0C protocol: 0.4 K ΔT below 0.6 K floor → heat_z must be nan"
