import numpy as np
import pandas as pd
from stress_screen.models import MethodResult
from stress_screen.analysis.short_circuit import run_isc_analysis, ShortCircuitParams


def _make_isc_rest_df(n_channels=8, n_points=400):
    """Stable OCV rest data, 8 hours, all channels normal."""
    rows = []
    t = np.linspace(0, 8.0, n_points)
    for ch in range(n_channels):
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - 0.0001 * t
        T = 25.0 * np.ones(n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)


def _make_rest_results_with_k(n_channels=8, k_values=None):
    """Minimal rest_results dict — M1 at index 0 with only a 'k' metadata key."""
    k_values = k_values or {ch: 0.0001 for ch in range(n_channels)}
    results = {}
    for ch in range(n_channels):
        k = k_values.get(ch, 0.0001)
        results[ch] = [
            MethodResult(
                method_name="M1_ocv_k",
                z_score=0.0,
                verdict="NORMAL",
                metadata={"k": k, "V_ocv": 3.4, "tau": 1.0},
            )
        ]
    return results


def _make_charge_df_for_isc(n_channels=8, n_points=200):
    """Charge data for S3 area tests. ch4 has half the normal voltage rise."""
    rows = []
    for ch in range(n_channels):
        t = np.linspace(0, 2.0, n_points)
        if ch == 4:
            V = 3.2 + 0.2 * (t / t.max())  # half rise → smaller dV/dQ area
        else:
            V = 3.2 + 0.4 * (t / t.max())
        T = 25.0 * np.ones(n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)


def test_isc_returns_all_channels():
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k()
    charge_df = pd.DataFrame(columns=["time_hours", "channel_index", "voltage", "temperature"])
    results = run_isc_analysis(rest_df, rest_results, charge_df)
    assert len(results) == 8
    assert all(r.method_name == "isc" for r in results.values())


def test_s1_high_k_cell_flagged():
    """Channel with 10× fleet-median k fires HIGH/ELEVATED on S1 and has higher S1 z."""
    k_values = {ch: 0.0001 for ch in range(8)}
    k_values[3] = 0.001  # 10× the fleet median
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k(k_values=k_values)
    charge_df = pd.DataFrame(columns=["time_hours", "channel_index", "voltage", "temperature"])
    results = run_isc_analysis(rest_df, rest_results, charge_df)
    assert results[3].metadata["s1_excess_k_z"] > results[0].metadata["s1_excess_k_z"], (
        "High-k channel should have higher S1 z than normal channel"
    )
    assert results[3].verdict in ("HIGH", "ELEVATED"), (
        f"High-k channel verdict expected HIGH/ELEVATED, got {results[3].verdict}"
    )


def test_s2_warming_cell_flagged():
    """Channel with rising temperature during rest gets elevated S2 z."""
    n_channels = 8
    rows = []
    t = np.linspace(0, 8.0, 400)
    for ch in range(n_channels):
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - 0.0001 * t
        T = 25.0 + 0.05 * t if ch == 2 else 26.0 - 0.02 * t
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest_df = pd.DataFrame(rows)
    rest_results = _make_rest_results_with_k(n_channels=n_channels)
    charge_df = pd.DataFrame(columns=["time_hours", "channel_index", "voltage", "temperature"])
    results = run_isc_analysis(rest_df, rest_results, charge_df)
    s2_z_ch2 = results[2].metadata["s2_dT_dt_z"]
    s2_z_others = [results[ch].metadata["s2_dT_dt_z"] for ch in range(n_channels) if ch != 2]
    assert s2_z_ch2 > float(np.nanmedian(s2_z_others)), (
        f"Warming ch2 S2 z={s2_z_ch2:.3f} should be above peers median "
        f"{np.nanmedian(s2_z_others):.3f}"
    )


def test_s3_area_deficit_cell_flagged():
    """Channel with half the dV/dQ area gets higher (inverted) S3 z than peers."""
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k()
    charge_df = _make_charge_df_for_isc()
    top_charge_df = pd.DataFrame({
        "time_hours": np.linspace(0, 2.0, 200),
        "current": np.full(200, 5.0),
    })
    results = run_isc_analysis(rest_df, rest_results, charge_df,
                               top_charge_df=top_charge_df)
    s3_z_ch4 = results[4].metadata["s3_area_deficit_z"]
    s3_z_others = [results[ch].metadata["s3_area_deficit_z"]
                   for ch in range(8) if ch != 4]
    assert s3_z_ch4 > float(np.nanmedian(s3_z_others)), (
        f"Area-deficit ch4 S3 z={s3_z_ch4:.3f} should be above peers median "
        f"{np.nanmedian(s3_z_others):.3f}"
    )


def test_s3_fallback_no_top_df():
    """Absent top_charge_df: all S3 area values are nan, no exception raised."""
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k()
    charge_df = _make_charge_df_for_isc()
    results = run_isc_analysis(rest_df, rest_results, charge_df)  # no top_charge_df
    assert len(results) == 8
    for ch, mr in results.items():
        assert np.isnan(mr.metadata["s3_dvdq_area"]), (
            f"ch{ch} s3_dvdq_area should be nan when top_charge_df absent, "
            f"got {mr.metadata['s3_dvdq_area']}"
        )


def test_isc_aggregate_integration():
    """Full aggregate with ISC produces CellVerdict with 8 method_results."""
    import warnings
    from stress_screen.analysis.rest import run_rest_analysis
    from stress_screen.analysis.li_plating import run_li_plating_analysis
    from stress_screen.analysis.aggregate import aggregate
    from stress_screen.topology import derive_topology

    n_channels = 8
    rest_df = _make_isc_rest_df()
    charge_df = _make_charge_df_for_isc()
    topo = derive_topology(n_channels, 1)

    full_rest_results = run_rest_analysis(rest_df, topo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        li_results = run_li_plating_analysis(charge_df, rest_df)
    isc_results = run_isc_analysis(rest_df, full_rest_results, charge_df)

    module_verdicts = aggregate(
        full_rest_results, li_results, topo, isc_results=isc_results
    )

    for mv in module_verdicts:
        for cv in mv.all_cells:
            assert len(cv.method_results) == 8, (
                f"{cv.label}: expected 8 method_results "
                f"(6 rest + 1 li_plating + 1 isc), got {len(cv.method_results)}"
            )
            isc_mrs = [mr for mr in cv.method_results if mr.method_name == "isc"]
            assert len(isc_mrs) == 1, (
                f"{cv.label}: expected exactly 1 isc result, got {len(isc_mrs)}"
            )
