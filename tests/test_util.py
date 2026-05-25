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


def test_protocol_metadata_defaults_lfp():
    from stress_screen.analysis.protocol import ProtocolMetadata
    p = ProtocolMetadata()
    assert p.chemistry == "LFP"
    assert 0.1 <= p.c_rate <= 3.0
    assert p.nominal_capacity_ah > 0
    assert p.voltage_window == (3.0, 3.65)


def test_protocol_metadata_c_rate_scales_dqdv_threshold():
    from stress_screen.analysis.protocol import ProtocolMetadata
    p_slow = ProtocolMetadata(c_rate=0.2)
    p_fast = ProtocolMetadata(c_rate=2.0)
    assert p_fast.dqdv_prominence_pct() > p_slow.dqdv_prominence_pct()


def test_protocol_metadata_c_rate_scales_dt_noise_floor():
    from stress_screen.analysis.protocol import ProtocolMetadata
    p_slow = ProtocolMetadata(c_rate=0.2)
    p_fast = ProtocolMetadata(c_rate=2.0)
    assert p_fast.dt_late_noise_floor_k() > p_slow.dt_late_noise_floor_k()


def test_protocol_metadata_baseline_thresholds_at_default_c_rate():
    """At the default C-rate (0.5), the scaling helpers should return the
    baseline thresholds matching the historical hard-coded values."""
    from stress_screen.analysis.protocol import ProtocolMetadata
    p = ProtocolMetadata()  # default c_rate=0.5
    assert p.dqdv_prominence_pct() == pytest.approx(0.05, abs=1e-9)
    assert p.dt_late_noise_floor_k() == pytest.approx(0.3, abs=1e-9)
