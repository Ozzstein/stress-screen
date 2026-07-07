"""Unit tests for config.py — merge order, presets, key validation."""

from __future__ import annotations

import pytest

from stress_screen.analysis.aggregate import AggregateParams
from stress_screen.analysis.li_plating import LiPlatingParams
from stress_screen.analysis.rest import RestParams
from stress_screen.analysis.short_circuit import ShortCircuitParams
from stress_screen.analysis.protocol import ProtocolMetadata
from stress_screen.config import AnalysisConfig, load_config


def test_default_config_matches_dataclass_defaults():
    """The lfp preset must be a no-op: resolved config == dataclass defaults."""
    config = load_config(chem="lfp")
    assert config.rest == RestParams()
    assert config.li_plating == LiPlatingParams()
    assert config.short_circuit == ShortCircuitParams()
    assert config.aggregate == AggregateParams()
    assert config.protocol == ProtocolMetadata()
    assert config.preset == "lfp"


def test_nmc_preset_sets_voltage_bounds():
    config = load_config(chem="nmc")
    assert config.rest.voltage_bounds == (3.0, 4.25)
    assert config.protocol.voltage_window == (3.0, 4.25)
    assert config.protocol.chemistry == "NMC"
    # Everything else stays at baseline
    assert config.rest.settling_h == RestParams().settling_h
    assert config.li_plating == LiPlatingParams()


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="Unknown chemistry preset"):
        load_config(chem="lto")


def test_user_config_overrides_preset(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text(
        "rest:\n  settling_h: 4.0\n  voltage_bounds: [3.1, 3.6]\n"
        "aggregate:\n  high_composite: 2.5\n"
    )
    config = load_config(config_path=cfg, chem="lfp")
    assert config.rest.settling_h == 4.0
    assert config.rest.voltage_bounds == (3.1, 3.6)
    assert config.aggregate.high_composite == 2.5
    # untouched keys keep defaults
    assert config.rest.z_thresh == 2.0


def test_cli_overrides_beat_user_config(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text("protocol:\n  c_rate: 0.8\n")
    config = load_config(
        config_path=cfg, chem="lfp",
        cli_overrides={"protocol": {"c_rate": 1.5}},
    )
    assert config.protocol.c_rate == 1.5


def test_unknown_key_rejected(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text("rest:\n  setling_h: 4.0\n")  # typo
    with pytest.raises(ValueError, match="setling_h"):
        load_config(config_path=cfg, chem="lfp")


def test_unknown_section_rejected(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text("restt:\n  settling_h: 4.0\n")
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(config_path=cfg, chem="lfp")


def test_missing_config_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(config_path=tmp_path / "nope.yaml", chem="lfp")


def test_to_dict_roundtrips_all_sections():
    d = load_config(chem="nmc").to_dict()
    assert d["preset"] == "nmc"
    for section in ("protocol", "rest", "li_plating", "short_circuit", "aggregate"):
        assert isinstance(d[section], dict) and d[section]
    assert d["rest"]["voltage_bounds"] == (3.0, 4.25)


def test_config_is_immutable_across_loads():
    """Two loads must not share mutated state."""
    a = load_config(chem="lfp")
    b = load_config(chem="nmc")
    assert a.rest.voltage_bounds == (3.0, 3.65)
    assert b.rest.voltage_bounds == (3.0, 4.25)
    assert AnalysisConfig().rest.voltage_bounds == (3.0, 3.65)
