"""
topology.py — Pack topology derivation for the stress_screen battery analysis tool.

Given the number of active voltage channels and the number of modules, derives
the cell configuration (series/parallel groups per module), builds the
channel-to-module map and temperature-sensor map, and returns a PackTopology.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from stress_screen.models import PackTopology

# ---------------------------------------------------------------------------
# Supported configurations
# ---------------------------------------------------------------------------

_SUPPORTED_CONFIGS = {"1P32S", "2P16S", "4P8S"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_mapping_file(override: Path | None = None) -> Path:
    """Locate the temp_mapping.yaml file.

    Search order:
    1. ``override`` (if given)
    2. PyInstaller bundle (``sys._MEIPASS``)
    3. Development layout: walk up from this file to the project root
    """
    if override is not None:
        return override
    # PyInstaller bundle
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # src/stress_screen/topology.py → project root (3 levels up)
        base = Path(__file__).parent.parent.parent
    return base / "configs" / "temp_mapping.yaml"


def _load_group_to_sensors(config_name: str, mapping_file: Path) -> dict[int, list[int]]:
    """Load the group_index → sensor_list mapping for *config_name* from YAML."""
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Temperature mapping file not found: {mapping_file}"
        )
    with mapping_file.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    if config_name not in data:
        raise KeyError(
            f"Config '{config_name}' not found in {mapping_file}. "
            f"Available: {[k for k in data if not k.startswith('_')]}"
        )
    raw: dict[Any, Any] = data[config_name]
    # YAML keys are already integers (bare integer keys in YAML are parsed as int)
    return {int(k): list(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_topology(
    active_channels: int,
    module_count: int,
    mapping_file: Path | None = None,
) -> PackTopology:
    """Derive pack topology from active channel count and module count.

    Parameters
    ----------
    active_channels:
        Total number of active voltage channels across all modules.
    module_count:
        Number of modules in the pack (parsed from the ``M{n}`` filename suffix).
    mapping_file:
        Optional override for the ``configs/temp_mapping.yaml`` path.
        Useful in tests.

    Returns
    -------
    PackTopology
        Fully-populated topology dataclass including channel→module and
        (module, group)→sensor-list maps.

    Raises
    ------
    ValueError
        If ``active_channels`` is not evenly divisible by ``module_count``, or
        if the resulting ``parallel * series`` does not equal 32, or if the
        derived config is not one of the supported configs.
    """
    # ------------------------------------------------------------------
    # 1. Validate divisibility
    # ------------------------------------------------------------------
    if active_channels % module_count != 0:
        quotient = active_channels / module_count
        raise ValueError(
            f"Cannot derive topology: {active_channels} active channels / "
            f"{module_count} modules = {quotient:.2f} groups per module — "
            f"not an integer"
        )

    series = active_channels // module_count
    parallel = 32 // series

    # Verify 32 cells per module invariant
    if parallel * series != 32:
        raise ValueError(
            f"Cannot derive topology: parallel ({parallel}) × series ({series}) "
            f"= {parallel * series}, expected 32. "
            f"Derived from {active_channels} active channels / {module_count} modules."
        )

    config_name = f"{parallel}P{series}S"

    if config_name not in _SUPPORTED_CONFIGS:
        raise ValueError(
            f"Unsupported configuration '{config_name}'. "
            f"Supported: {sorted(_SUPPORTED_CONFIGS)}"
        )

    # ------------------------------------------------------------------
    # 2. Build channel-to-module map (0-based channel → 1-based module)
    # ------------------------------------------------------------------
    channel_module_map: dict[int, int] = {
        ch: (ch // series) + 1 for ch in range(active_channels)
    }

    # ------------------------------------------------------------------
    # 3. Build temperature sensor map
    # ------------------------------------------------------------------
    resolved_file = _find_mapping_file(mapping_file)
    group_to_sensors = _load_group_to_sensors(config_name, resolved_file)

    temp_sensor_map: dict[tuple[int, int], list[int]] = {}
    for module_id in range(1, module_count + 1):
        for group_idx, sensor_list in group_to_sensors.items():
            temp_sensor_map[(module_id, group_idx)] = sensor_list

    # ------------------------------------------------------------------
    # 4. Assemble and return
    # ------------------------------------------------------------------
    return PackTopology(
        module_count=module_count,
        series=series,
        parallel=parallel,
        config_name=config_name,
        active_channels=active_channels,
        _channel_module_map=channel_module_map,
        _temp_sensor_map=temp_sensor_map,
    )
