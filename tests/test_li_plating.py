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
