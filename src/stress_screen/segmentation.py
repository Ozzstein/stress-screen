"""
segmentation.py — Phase segmentation for stress_screen battery analysis.

Takes the pack-level DataFrame (top_df from loader.py) and splits it into
Segment objects with phase in {"charge", "discharge", "rest"}.
"""

from __future__ import annotations

import warnings
from typing import List

import pandas as pd

from stress_screen.models import Segment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

I_REST = 0.5        # A — |current| below this → rest (when sustained)
I_ACTIVE = 1.0      # A — |current| above this → active (charge or discharge)
REST_MIN_HOURS = 30.0  # minimum duration to label a segment as "rest"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment(
    top_df: pd.DataFrame,
    i_rest: float = I_REST,
    i_active: float = I_ACTIVE,
    rest_min_hours: float = REST_MIN_HOURS,
) -> List[Segment]:
    """
    Split pack-level DataFrame into charge/discharge/rest Segment objects.

    Parameters
    ----------
    top_df : pd.DataFrame
        Pack-level time series with at least ``time_hours`` and ``current``
        columns, as returned by ``loader.load_csv``.
    i_rest : float
        Current magnitude (A) below which a row is a candidate for rest.
    i_active : float
        Current magnitude (A) above which a row is definitively charge or
        discharge.
    rest_min_hours : float
        Minimum duration (hours) for a rest candidate to be kept as "rest".
        Shorter candidates are re-labelled as "transition" and then merged
        into neighbouring segments.

    Returns
    -------
    list[Segment]
        Segments ordered by ``start_time_h``.
        Emits a ``warnings.warn`` if no rest segment >= 24 h is found.
    """
    current: pd.Series = top_df["current"].reset_index(drop=True)
    time_h: pd.Series = top_df["time_hours"].reset_index(drop=True)
    n = len(current)

    # ------------------------------------------------------------------
    # Step 1 — assign raw phase with hysteresis
    # ------------------------------------------------------------------
    raw: list[str] = [""] * n
    prev = "rest"  # default initial state

    abs_current = current.abs()

    for i in range(n):
        c = current.iat[i]
        a = abs_current.iat[i]
        if a < i_rest:
            prev = "rest"
        elif c > i_active:
            prev = "charge"
        elif c < -i_active:
            prev = "discharge"
        # else: between thresholds — inherit previous (hysteresis)
        raw[i] = prev

    # ------------------------------------------------------------------
    # Step 2 — group consecutive rows with the same raw phase into
    #           candidate segments
    # ------------------------------------------------------------------
    candidates: list[tuple[str, int, int]] = []  # (phase, start_row, end_row)
    seg_start = 0
    seg_phase = raw[0]

    for i in range(1, n):
        if raw[i] != seg_phase:
            candidates.append((seg_phase, seg_start, i - 1))
            seg_start = i
            seg_phase = raw[i]
    candidates.append((seg_phase, seg_start, n - 1))

    # ------------------------------------------------------------------
    # Step 3 — post-process: re-label short rest candidates as
    #           "transition", then merge consecutive same-phase segments
    # ------------------------------------------------------------------
    processed: list[tuple[str, int, int]] = []
    for phase, sr, er in candidates:
        if phase == "rest":
            duration = time_h.iat[er] - time_h.iat[sr]
            if duration < rest_min_hours:
                phase = "transition"
        processed.append((phase, sr, er))

    # Merge consecutive segments of the same label (handles transitions
    # that collapse into surrounding charge/discharge, and any other
    # adjacent-same-phase pairs created by the relabelling).
    # We do multiple passes until stable to handle chains like:
    #   charge, transition, charge → charge (two passes needed)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[str, int, int]] = []
        i = 0
        while i < len(processed):
            if merged and merged[-1][0] == processed[i][0]:
                # extend last segment to cover current one
                last = merged[-1]
                merged[-1] = (last[0], last[1], processed[i][2])
                changed = True
            else:
                merged.append(processed[i])
            i += 1
        processed = merged

    # Any remaining "transition" segments (isolated, not adjacent to a
    # same-phase neighbour) keep their label for safety — but we convert
    # them to the dominant surrounding phase if possible.
    final: list[tuple[str, int, int]] = []
    for idx, (phase, sr, er) in enumerate(processed):
        if phase == "transition":
            # Pick the phase of the nearest non-transition neighbour
            prev_phase = None
            next_phase = None
            for j in range(idx - 1, -1, -1):
                if processed[j][0] != "transition":
                    prev_phase = processed[j][0]
                    break
            for j in range(idx + 1, len(processed)):
                if processed[j][0] != "transition":
                    next_phase = processed[j][0]
                    break
            # Prefer the previous phase; fall back to next; then "rest"
            resolved = prev_phase or next_phase or "rest"
            final.append((resolved, sr, er))
        else:
            final.append((phase, sr, er))

    # One more merge pass after transition resolution
    changed = True
    while changed:
        changed = False
        merged = []
        for item in final:
            if merged and merged[-1][0] == item[0]:
                last = merged[-1]
                merged[-1] = (last[0], last[1], item[2])
                changed = True
            else:
                merged.append(item)
        final = merged

    # ------------------------------------------------------------------
    # Step 4 — build Segment objects
    # ------------------------------------------------------------------
    segments: list[Segment] = []
    for phase, sr, er in final:
        segments.append(Segment(
            phase=phase,
            start_time_h=float(time_h.iat[sr]),
            end_time_h=float(time_h.iat[er]),
            start_row=int(sr),
            end_row=int(er),
        ))

    # ------------------------------------------------------------------
    # Step 5 — sanity warning
    # ------------------------------------------------------------------
    max_rest = max(
        (s.duration_h for s in segments if s.phase == "rest"),
        default=0.0,
    )
    if max_rest < 24.0:
        warnings.warn(
            f"No rest segment >= 24 h found (longest rest = {max_rest:.1f} h). "
            "Check that the data contains a complete rest period.",
            stacklevel=2,
        )

    return segments


def rest_segments(segments: list[Segment]) -> list[Segment]:
    """Filter to rest segments only, longest first."""
    return sorted(
        [s for s in segments if s.phase == "rest"],
        key=lambda s: s.duration_h,
        reverse=True,
    )


def charge_segments(segments: list[Segment]) -> list[Segment]:
    """Filter to charge segments only, ordered by start time."""
    return [s for s in segments if s.phase == "charge"]
