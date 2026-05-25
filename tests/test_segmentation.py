import pandas as pd
import numpy as np
import warnings
from stress_screen.segmentation import segment, rest_segments, charge_segments

def _make_top_df(currents: list, dt_h: float = 1.0/3600) -> pd.DataFrame:
    n = len(currents)
    return pd.DataFrame({
        "time_hours": np.arange(n) * dt_h,
        "current": currents,
        "pack_voltage": [3.5] * n,
        "soc_pct": [50.0] * n,
        "warning": [""] * n,
        "fault": [""] * n,
    })

def test_basic_segmentation():
    # charge 2h, discharge 2h, rest 40h
    currents = [10.0] * 7200 + [-10.0] * 7200 + [0.0] * 144000
    top = _make_top_df(currents)
    segs = segment(top)
    phases = [s.phase for s in segs]
    assert "charge" in phases
    assert "rest" in phases

def test_rest_duration():
    currents = [10.0] * 7200 + [0.0] * 144000
    top = _make_top_df(currents)
    segs = segment(top)
    rs = rest_segments(segs)
    assert rs[0].duration_h > 35.0

def test_empty_dataframe():
    top = _make_top_df([])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = segment(top)
    assert result == []
    assert len(w) == 1
