import numpy as np
import pytest
from stress_screen.analysis.util import arrhenius_correction, winsorize_clip


def test_arrhenius_correction_at_t_ref_is_unity():
    # 25°C reference: correction must equal 1.0
    c = arrhenius_correction(T_celsius=25.0, ea_ev=0.5)
    assert abs(c - 1.0) < 1e-9


def test_arrhenius_correction_at_higher_t_below_unity():
    # 35°C with Ea=0.5 eV: correction < 1 (warmer cell is faster, so we
    # divide the raw rate by something > 1 -> correction factor < 1)
    c = arrhenius_correction(T_celsius=35.0, ea_ev=0.5)
    assert c < 1.0
    # Sanity: ratio should be around exp(-Ea/k_B * (1/298 - 1/308)) ≈ 0.516
    assert 0.45 < c < 0.60


def test_arrhenius_correction_zero_ea_is_unity_at_all_t():
    # ISC (electronic short) has Ea ≈ 0 → temperature-insensitive
    for T in (-10.0, 0.0, 25.0, 50.0):
        c = arrhenius_correction(T_celsius=T, ea_ev=0.0)
        assert abs(c - 1.0) < 1e-9


def test_arrhenius_correction_nan_returns_unity():
    assert arrhenius_correction(T_celsius=np.nan, ea_ev=0.5) == 1.0


def test_winsorize_clip_symmetric_bounds():
    # Default symmetric clip should preserve sign of extreme negative z
    values = np.array([-7.0, -3.0, 0.0, 3.0, 7.0])
    out = winsorize_clip(values, low=-5.0, high=5.0)
    np.testing.assert_array_equal(out, np.array([-5.0, -3.0, 0.0, 3.0, 5.0]))


def test_winsorize_clip_preserves_nans():
    values = np.array([np.nan, 2.0, 10.0])
    out = winsorize_clip(values, low=-5.0, high=5.0)
    assert np.isnan(out[0])
    assert out[1] == 2.0
    assert out[2] == 5.0
