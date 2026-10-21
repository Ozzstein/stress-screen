"""M3 (spread) noise floor on quantised data.

Tester voltages are integer millivolts, so on a healthy pack most cells have a
divergence slope of exactly 0 and the fleet MAD collapses to 0. The z-score
is then slope / floor, and the floor must be large enough that a single
quantisation step cannot reach HIGH while genuine multi-mV divergence can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress_screen.analysis.rest import RestParams, run_rest_analysis
from stress_screen.topology import derive_topology

# Symmetric integer-mV offsets with median 0 so the fleet median is stable
# whatever the extreme cell (index 4, offset -2 mV) does.
_OFFSETS_MV = [0, 1, -1, 2, -2, 1, -1, 0, 1, -1, 2, -2, 1, -1, 0, 1]
_DIVERGING_CH = 4


def _quantised_rest_df(diverge_mv) -> pd.DataFrame:
    """16-channel 2P16S rest of 50 h at 60 s, constant 25 °C, integer-mV
    voltages like the tester's. ``diverge_mv(t_hours)`` is the extra drop
    (in mV, >= 0) applied to channel 4, which starts at the fleet's low end
    so any drop moves it *away* from the median."""
    t = np.arange(0, 50 * 60) / 60.0  # hours, 3000 samples
    frames = []
    for ch, off in enumerate(_OFFSETS_MV):
        mv = np.full(t.size, 3350 + off, dtype=float)
        if ch == _DIVERGING_CH:
            mv = mv - diverge_mv(t)
        frames.append(pd.DataFrame({
            "time_hours": t,
            "channel_index": ch,
            "voltage": np.rint(mv) / 1000.0,  # exactly what load_csv produces
            "temperature": 25.0,
        }))
    return pd.concat(frames, ignore_index=True)


def _spread(results, ch):
    return next(mr for mr in results[ch] if mr.method_name == "spread")


def test_voltage_resolution_is_smallest_step_between_distinct_values():
    from stress_screen.analysis.rest import _voltage_resolution

    assert _voltage_resolution(np.array([3.300, 3.301, 3.305, 3.301, np.nan])) == pytest.approx(0.001)
    assert np.isnan(_voltage_resolution(np.array([3.3, 3.3, 3.3])))


def test_single_lsb_step_cannot_reach_high():
    # One quantisation step halfway through the rest: the largest slope a
    # pure rounding artefact can produce. Must stay below the HIGH gate.
    df = _quantised_rest_df(lambda t: np.where(t >= 25.0, 1.0, 0.0))
    results = run_rest_analysis(df, derive_topology(16, 1))
    m3 = _spread(results, _DIVERGING_CH)
    assert m3.z_score < 2.0, f"1-LSB step gave spread z={m3.z_score:.2f}"


def test_spread_floor_is_one_lsb_over_settled_window():
    df = _quantised_rest_df(lambda t: np.where(t >= 25.0, 1.0, 0.0))
    params = RestParams()
    results = run_rest_analysis(df, derive_topology(16, 1), params)
    t = df.loc[df["channel_index"] == 0, "time_hours"].to_numpy()
    t_settled = t[t >= params.settling_h]
    expected_floor = 0.001 / (t_settled[-1] - t_settled[0])
    assert expected_floor > params.m3_min_slope_mad_v_per_h  # the derived floor governs here

    meta = _spread(results, _DIVERGING_CH).metadata
    assert meta["voltage_resolution_v"] == pytest.approx(0.001)
    assert meta["mad_floor_v_per_h"] == pytest.approx(expected_floor)


def test_resolution_is_measured_on_raw_samples_not_pivot_means():
    # Real logs contain the odd duplicated timestamp; the wide pivot averages
    # those samples, which would halve the apparent resolution. q must come
    # from the raw samples (1 mV), not from the averaged pivot (0.5 mV).
    df = _quantised_rest_df(lambda t: np.zeros_like(t))
    dup = df[(df["channel_index"] == 0) & (df["time_hours"] == 10.0)].copy()
    dup["voltage"] += 0.001
    df = pd.concat([df, dup], ignore_index=True)

    results = run_rest_analysis(df, derive_topology(16, 1))
    assert _spread(results, 0).metadata["voltage_resolution_v"] == pytest.approx(0.001)


def test_multi_lsb_divergence_is_still_high():
    # Regression guard for detection power: a 5 mV linear drift away from
    # the fleet over the rest must remain HIGH on the method.
    df = _quantised_rest_df(lambda t: 5.0 * t / t[-1])
    results = run_rest_analysis(df, derive_topology(16, 1))
    m3 = _spread(results, _DIVERGING_CH)
    assert m3.z_score >= 2.0, f"5-LSB drift gave spread z={m3.z_score:.2f}"
