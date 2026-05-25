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
        assert "dqdv_z" in mr.metadata
        assert "relaxation_z" in mr.metadata
        assert "charge_time_z" in mr.metadata
