"""
config.py — resolved analysis configuration for stress_screen.

Bundles every tunable parameter set (protocol, rest methods, Li-plating, ISC,
aggregation gates) into one :class:`AnalysisConfig` and resolves it from, in
increasing precedence:

1. Dataclass defaults (the source of truth for baseline behavior)
2. The chemistry preset selected with ``--chem`` (from
   ``configs/analysis_defaults.yaml``)
3. A user-supplied ``--config my_config.yaml`` file
4. Explicit CLI flags (``--c-rate``, ``--capacity-ah``)

User config files use one section per parameter group::

    rest:
      settling_h: 2.0
      arrhenius_ea_ev: 0.5
    li_plating:
      T_plating_threshold_c: 20.0
    short_circuit:
      isc_k_sigma: 3.0
    aggregate:
      high_composite: 2.0
    protocol:
      c_rate: 0.5

Unknown sections or keys are rejected loudly (typo protection). The bundled
``configs/analysis_defaults.yaml`` documents every available key.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from stress_screen.analysis.aggregate import AggregateParams, CompositeParams
from stress_screen.analysis.li_plating import LiPlatingParams
from stress_screen.analysis.protocol import ProtocolMetadata
from stress_screen.analysis.rest import RestParams
from stress_screen.analysis.short_circuit import ShortCircuitParams

#: Dataclass fields that must be coerced from YAML lists back to tuples.
_TUPLE_FIELDS = {"voltage_bounds", "voltage_window"}

_SECTION_TYPES: dict[str, type] = {
    "protocol": ProtocolMetadata,
    "rest": RestParams,
    "li_plating": LiPlatingParams,
    "short_circuit": ShortCircuitParams,
    "aggregate": AggregateParams,
    "composite": CompositeParams,
}


@dataclass
class AnalysisConfig:
    """The fully resolved parameter set for one analysis run."""

    protocol: ProtocolMetadata = field(default_factory=ProtocolMetadata)
    rest: RestParams = field(default_factory=RestParams)
    li_plating: LiPlatingParams = field(default_factory=LiPlatingParams)
    short_circuit: ShortCircuitParams = field(default_factory=ShortCircuitParams)
    aggregate: AggregateParams = field(default_factory=AggregateParams)
    composite: CompositeParams = field(default_factory=CompositeParams)
    preset: str = "lfp"

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict dump for the JSON result's ``config`` section."""
        out: dict[str, Any] = {"preset": self.preset}
        for name in _SECTION_TYPES:
            out[name] = asdict(getattr(self, name))
        return out


def find_defaults_file() -> Path:
    """Locate the bundled analysis_defaults.yaml (same pattern as topology).

    The file ships inside the package (``stress_screen/configs/``) so wheel
    installs, editable installs, and PyInstaller bundles all resolve it.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).parent
    return base / "configs" / "analysis_defaults.yaml"


def _validate_keys(section: str, data: dict[str, Any], cls: type) -> None:
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section '{section}'. "
            f"Valid keys: {sorted(valid)}"
        )


def _coerce(section_data: dict[str, Any]) -> dict[str, Any]:
    return {
        k: tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v
        for k, v in section_data.items()
    }


def _apply_sections(config: AnalysisConfig, overrides: dict[str, Any],
                    source: str) -> AnalysisConfig:
    """Return a new AnalysisConfig with *overrides* applied section-wise."""
    unknown_sections = set(overrides) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ValueError(
            f"Unknown config section(s) {sorted(unknown_sections)} in {source}. "
            f"Valid sections: {sorted(_SECTION_TYPES)}"
        )
    updates: dict[str, Any] = {}
    for section, data in overrides.items():
        if data is None:
            continue
        if not isinstance(data, dict):
            raise ValueError(
                f"Config section '{section}' in {source} must be a mapping, "
                f"got {type(data).__name__}"
            )
        cls = _SECTION_TYPES[section]
        _validate_keys(section, data, cls)
        updates[section] = replace(getattr(config, section), **_coerce(data))
    return replace(config, **updates)


def _load_presets(defaults_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = defaults_path or find_defaults_file()
    if not path.exists():
        raise FileNotFoundError(f"Bundled analysis defaults not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"'presets' in {path} must be a mapping")
    return presets


def load_config(
    config_path: Path | None = None,
    chem: str = "lfp",
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    defaults_path: Path | None = None,
) -> AnalysisConfig:
    """Resolve the analysis configuration.

    Parameters
    ----------
    config_path:
        Optional user YAML config file (section → key → value).
    chem:
        Chemistry preset name from ``--chem`` (must exist under ``presets:``
        in the bundled defaults file).
    cli_overrides:
        Highest-precedence overrides from explicit CLI flags, in the same
        section → key → value shape (e.g. ``{"protocol": {"c_rate": 1.0}}``).
    defaults_path:
        Override the bundled defaults/presets file (mainly for tests).
    """
    config = AnalysisConfig(preset=chem)

    presets = _load_presets(defaults_path)
    if chem not in presets:
        raise ValueError(
            f"Unknown chemistry preset '{chem}'. Available: {sorted(presets)}"
        )
    config = _apply_sections(config, presets[chem] or {}, f"preset '{chem}'")

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
        if not isinstance(user_data, dict):
            raise ValueError(f"Config file {config_path} must be a YAML mapping")
        user_data.pop("_version", None)
        config = _apply_sections(config, user_data, str(config_path))

    if cli_overrides:
        config = _apply_sections(config, cli_overrides, "CLI flags")

    if config.composite.mode not in ("clustered", "legacy"):
        raise ValueError(
            f"composite.mode must be 'clustered' or 'legacy', "
            f"got {config.composite.mode!r}"
        )
    if config.composite.reduce not in ("max", "mean"):
        raise ValueError(
            f"composite.reduce must be 'max' or 'mean', "
            f"got {config.composite.reduce!r}"
        )

    return config
