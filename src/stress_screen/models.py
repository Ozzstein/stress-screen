"""
Shared dataclasses for the stress_screen battery-pack analysis tool.

No business logic, no I/O.  Every other module imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# PackTopology
# ---------------------------------------------------------------------------

@dataclass
class PackTopology:
    """Describes the physical cell-group layout of a battery pack.

    Populated by topology.py; the two private maps are filled there and must
    not be written directly by callers.
    """

    module_count: int          # number of modules (from filename M{n})
    series: int                # cell-groups per module in series
    parallel: int              # cells per group in parallel
    config_name: str           # "1P32S", "2P16S", or "4P8S"
    active_channels: int       # total active voltage channels across all modules

    # channel_index (0-based) → module_id (1-based)
    _channel_module_map: dict[int, int] = field(default_factory=dict)

    # (module_id, group_index_within_module 1-based) → list of sensor indices
    _temp_sensor_map: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def module_for_channel(self, channel_index: int) -> int:
        """Return the 1-based module that owns *channel_index* (0-based)."""
        return self._channel_module_map[channel_index]

    def channels_in_module(self, module_id: int) -> list[int]:
        """Return all 0-based channel indices that belong to *module_id*."""
        return [
            ch for ch, mid in self._channel_module_map.items() if mid == module_id
        ]

    def group_index_in_module(self, channel_index: int) -> int:
        """Return the 1-based group position of *channel_index* within its module."""
        module_id = self.module_for_channel(channel_index)
        module_channels = sorted(self.channels_in_module(module_id))
        return module_channels.index(channel_index) + 1

    def temp_sensors_for_group(
        self, module_id: int, group_index: int
    ) -> list[int]:
        """Return sensor indices for the given (module, group) pair."""
        return self._temp_sensor_map.get((module_id, group_index), [])


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A contiguous phase slice of the test time-series."""

    phase: str           # "charge", "discharge", or "rest"
    start_time_h: float
    end_time_h: float
    duration_h: float    # end_time_h - start_time_h
    start_row: int       # integer row index into the full DataFrame
    end_row: int


# ---------------------------------------------------------------------------
# MethodResult
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """One detection method's output for a single cell-group."""

    method_name: str     # e.g. "M1_ocv_k", "M4_cusum", "li_plating"
    z_score: float
    verdict: str         # "HIGH", "ELEVATED", or "NORMAL"
    metadata: dict       # method-specific extras (e.g. {"k": 0.0023, "n_alarms": 5})


# ---------------------------------------------------------------------------
# CellVerdict
# ---------------------------------------------------------------------------

@dataclass
class CellVerdict:
    """Aggregated verdict for one cell-group across all detection methods."""

    channel_index: int   # 0-based global channel index
    module_id: int       # 1-based module
    group_in_module: int # 1-based group within module
    composite_z: float
    n_methods_high: int
    verdict: str         # "HIGH", "ELEVATED", or "NORMAL"
    method_results: list[MethodResult]

    @property
    def label(self) -> str:
        """Human-readable identifier, e.g. 'M3/G5'."""
        return f"M{self.module_id}/G{self.group_in_module}"


# ---------------------------------------------------------------------------
# ModuleVerdict
# ---------------------------------------------------------------------------

@dataclass
class ModuleVerdict:
    """Pass/fail verdict for an entire module."""

    module_id: int
    verdict: str                  # "OK" or "NOK"
    flagged_cells: list[CellVerdict]   # only HIGH cells
    all_cells: list[CellVerdict]

    @property
    def summary_line(self) -> str:
        """One-line human-readable summary.

        Examples::
            "M3: OK"
            "M3: NOK  [cells flagged: M3/G5 (HIGH), M3/G7 (HIGH)]"
        """
        if self.verdict == "OK":
            return f"M{self.module_id}: OK"

        flagged_labels = ", ".join(
            f"{c.label} ({c.verdict})" for c in self.flagged_cells
        )
        return f"M{self.module_id}: NOK  [cells flagged: {flagged_labels}]"


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Top-level result object passed to the CLI and report generators."""

    csv_path: Path
    topology: PackTopology
    segments: list[Segment]
    module_verdicts: list[ModuleVerdict]   # ordered by module_id

    @property
    def any_nok(self) -> bool:
        return any(m.verdict == "NOK" for m in self.module_verdicts)
