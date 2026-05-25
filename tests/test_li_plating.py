import pandas as pd
import numpy as np
from stress_screen.analysis.li_plating import run_li_plating_analysis, LiPlatingParams

def _make_charge_df(n_channels=5, n_points=200):
    rows = []
    for ch in range(n_channels):
        t = np.linspace(0, 2.0, n_points)
        V = 3.2 + 0.4 * (t / t.max())  # healthy: smooth ramp
        if ch == 0:
            # inject extra prominence mid-charge
            V = V.copy()
            V[120:130] += 0.05  # artificial bump -> extra dV/dQ peak
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch, "voltage": V[i], "temperature": 25.0})
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
        assert "dv_dq_z" in mr.metadata
        assert "relaxation_z" in mr.metadata
        assert "charge_time_z" in mr.metadata
        # New temperature-related metadata keys
        assert "cold_z" in mr.metadata
        assert "heat_z" in mr.metadata
        assert "T_mean_charge" in mr.metadata
        assert "dT_late" in mr.metadata
        assert "temperature_gate" in mr.metadata
        assert "gated_dv_dq_z" in mr.metadata
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


def test_injected_cell_has_higher_dv_dq():
    charge = _make_charge_df(n_channels=8)
    rest = _make_rest_df(n_channels=8)
    results = run_li_plating_analysis(charge, rest)
    ch0_dv_z = results[0].metadata["dv_dq_z"]
    other_dv_z = [results[ch].metadata["dv_dq_z"] for ch in range(1, 8)]
    import numpy as np
    # Channel 0 has an injected peak, should have higher dv_dq_z than the median
    assert ch0_dv_z > float(np.nanmedian(other_dv_z)), \
        f"Expected ch0 dv_dq_z={ch0_dv_z:.3f} > median others {np.nanmedian(other_dv_z):.3f}"


import warnings


def _make_top_charge_df(n_points=200, current=5.0):
    t = np.linspace(0, 2.0, n_points)
    return pd.DataFrame({"time_hours": t, "current": np.full(n_points, current)})


def test_dvdq_uses_q_domain():
    """Peak prominence sum must differ between Q-domain and index-domain calls."""
    charge = _make_charge_df(n_channels=5)
    rest = _make_rest_df(n_channels=5)
    top_charge = _make_top_charge_df(current=5.0)  # dq ≈ 0.05 Ah/sample → ~20× scale factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_idx = run_li_plating_analysis(charge, rest)
    results_q = run_li_plating_analysis(charge, rest, top_charge_df=top_charge, n_parallel=1)

    # Channel 0 has an injected bump → non-zero prominence in both calls
    idx_sum = results_idx[0].metadata["peak_prominence_sum"]
    q_sum = results_q[0].metadata["peak_prominence_sum"]
    assert idx_sum > 0, "Injected peak on ch0 should give non-zero index-based prominence"
    assert abs(q_sum - idx_sum) / (idx_sum + 1e-10) > 0.01, (
        f"Q-domain prominence {q_sum:.6f} too similar to index-based {idx_sum:.6f}"
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
    assert "index" in messages or "top_charge_df" in messages, (
        f"Expected Q-domain fallback warning, got: {[str(w.message) for w in caught]}"
    )


def test_t_threshold_20c():
    """Default T_plating_threshold_c=20°C: 19°C cell gate ≥0.05; 21°C cell gate=0."""
    charge = _make_charge_df(n_channels=4)
    rest = _make_rest_df(n_channels=4)
    charge.loc[charge["channel_index"] == 0, "temperature"] = 19.0
    charge.loc[charge["channel_index"] == 1, "temperature"] = 21.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    assert results[0].metadata["temperature_gate"] >= 0.05, (
        f"19°C cell gate={results[0].metadata['temperature_gate']:.4f} must be ≥0.05 with 20°C threshold"
    )
    assert results[1].metadata["temperature_gate"] == 0.0, (
        f"21°C cell gate={results[1].metadata['temperature_gate']:.4f} must be 0.0"
    )


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
