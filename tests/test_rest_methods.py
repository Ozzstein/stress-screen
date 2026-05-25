import numpy as np
import pandas as pd
from stress_screen.analysis.rest import run_rest_analysis, RestParams
from stress_screen.topology import derive_topology

def _make_rest_cell_df(n_channels=8, n_points=500, bad_channels=None, k_bad=0.002):
    """Synthetic OCV rest data. bad_channels is a list of channel_indices with high k (self-discharge)."""
    bad_channels = bad_channels or []
    rows = []
    t = np.linspace(0, 10, n_points)  # 10 hours of rest
    for ch in range(n_channels):
        k = k_bad if ch in bad_channels else 0.0001
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - k * t + np.random.normal(0, 0.0001, n_points)
        T = 25.0 + np.random.normal(0, 0.1, n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch, "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)

def test_rest_analysis_returns_all_channels():
    np.random.seed(42)
    rest_df = _make_rest_cell_df(n_channels=8)
    topo = derive_topology(8, 1)  # 8 channels, 1 module -> 4P8S
    results = run_rest_analysis(rest_df, topo)
    assert len(results) == 8

def test_m1_flags_high_k_cells():
    np.random.seed(42)
    bad = [2, 5]
    rest_df = _make_rest_cell_df(n_channels=16, bad_channels=bad, k_bad=0.003)
    topo = derive_topology(16, 1)  # 16 channels, 1 module -> 2P16S
    results = run_rest_analysis(rest_df, topo)
    m1_verdicts = {ch: next(mr.verdict for mr in mrs if mr.method_name == "M1_ocv_k")
                   for ch, mrs in results.items()}
    # Both bad channels should be HIGH
    for ch in bad:
        assert m1_verdicts[ch] in ("HIGH", "ELEVATED"), f"ch {ch} expected HIGH/ELEVATED, got {m1_verdicts[ch]}"
