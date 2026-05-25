"""
loader.py — CSV data loader for the stress_screen battery analysis tool.

Reads the semicolon-delimited, comma-decimal CSV produced by the pack tester
and returns two tidy DataFrames:

  top_df  — pack-level time series (one row per timestamp)
  cell_df — long-format per-cell time series (active channels only)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(
    filepath: Path,
    min_voltage_v: float = 0.1,
    downsample: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a stress-test CSV.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file produced by the battery pack tester.
    min_voltage_v : float
        Minimum peak voltage (volts) a channel must reach to be considered
        active.  Channels whose max converted voltage is below this are
        dropped from *cell_df*.  Default is 0.1 V (= 100 mV raw).
    downsample : int
        Keep every Nth row (1 = no downsampling, 60 = ~1 sample per minute
        at 1 Hz logging rate).

    Returns
    -------
    top_df : pd.DataFrame
        Pack-level time series with columns:
        ``time_hours``, ``current``, ``pack_voltage``, ``soc_pct``,
        ``warning``, ``fault``.
    cell_df : pd.DataFrame
        Long-format per-cell time series with columns:
        ``time_hours``, ``channel_index``, ``voltage``, ``temperature``.
        Only active channels (max voltage >= *min_voltage_v*) are included.
        Temperatures of 0.0 are replaced with NaN (no sensor connected).
    """
    filepath = Path(filepath)

    # ------------------------------------------------------------------
    # 1. Count leading comment lines (#) so we can skip them.
    # ------------------------------------------------------------------
    skip_rows = 0
    with open(filepath, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                skip_rows += 1
            else:
                break

    # ------------------------------------------------------------------
    # 2. Read CSV.
    #    - sep=';'           semicolon delimiter
    #    - decimal=','       European decimal separator
    #    - comment='#'       skip in-body comment lines
    #    - index_col=False   prevent pandas from treating the first column
    #                        as the index (the data rows have one trailing
    #                        semicolon, giving them one extra field vs header)
    # ------------------------------------------------------------------
    df = pd.read_csv(
        filepath,
        sep=";",
        decimal=",",
        skiprows=skip_rows,
        comment="#",
        index_col=False,
        skipinitialspace=True,
        engine="python",
    )

    # Strip whitespace from column names (some have trailing spaces in file)
    df.columns = df.columns.str.strip()

    # Drop fully-empty trailing columns (artifact of trailing semicolons)
    df = df.dropna(axis=1, how="all")

    # ------------------------------------------------------------------
    # 3. Optional downsampling
    # ------------------------------------------------------------------
    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Parse time → elapsed hours
    # ------------------------------------------------------------------
    time_hours = _parse_time_to_hours(df.iloc[:, 0].astype(str).str.strip())
    time_hours = time_hours.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Build top_df (pack-level)
    # ------------------------------------------------------------------
    top_df = pd.DataFrame({
        "time_hours":   time_hours,
        "current":      pd.to_numeric(df["Current"], errors="coerce"),
        "pack_voltage": pd.to_numeric(df["Voltage"], errors="coerce"),
        "soc_pct":      pd.to_numeric(df["SOC %"], errors="coerce"),
        "warning":      df["Warning"].astype(str).str.strip(),
        "fault":        df["Fault"].astype(str).str.strip(),
    })

    # ------------------------------------------------------------------
    # 6. Identify cell voltage and temperature columns
    # ------------------------------------------------------------------
    volt_re = re.compile(r"^Cell_(\d+)_Volt$")
    temp_re = re.compile(r"^Cell_(\d+)_Temp$")

    volt_cols: dict[int, str] = {}  # cell_number → column name
    temp_cols: dict[int, str] = {}

    for col in df.columns:
        m = volt_re.match(col)
        if m:
            volt_cols[int(m.group(1))] = col
            continue
        m = temp_re.match(col)
        if m:
            temp_cols[int(m.group(1))] = col

    # ------------------------------------------------------------------
    # 7. Convert mV → V and filter active channels
    # ------------------------------------------------------------------
    min_voltage_raw = min_voltage_v  # already in V after /1000 below

    active_cell_numbers: list[int] = []
    for cell_num in sorted(volt_cols):
        raw = pd.to_numeric(df[volt_cols[cell_num]], errors="coerce")
        max_v = (raw / 1000.0).max()
        if pd.notna(max_v) and max_v >= min_voltage_raw:
            active_cell_numbers.append(cell_num)

    # ------------------------------------------------------------------
    # 8. Build cell_df (long format, active channels only)
    #    channel_index is 0-based (cell_number - 1)
    # ------------------------------------------------------------------
    frames: list[pd.DataFrame] = []
    for cell_num in active_cell_numbers:
        voltage_v = (
            pd.to_numeric(df[volt_cols[cell_num]], errors="coerce") / 1000.0
        ).values

        if cell_num in temp_cols:
            temp = pd.to_numeric(
                df[temp_cols[cell_num]], errors="coerce"
            ).values.astype(float)
            # 0.0 means "no sensor connected" → NaN
            temp[temp == 0.0] = np.nan
        else:
            temp = np.full(len(df), np.nan)

        frames.append(pd.DataFrame({
            "time_hours":    time_hours.values,
            "channel_index": cell_num - 1,      # 0-based
            "voltage":       voltage_v,
            "temperature":   temp,
        }))

    if frames:
        cell_df = pd.concat(frames, ignore_index=True)
    else:
        # No active channels — return empty but correctly-typed DataFrame
        cell_df = pd.DataFrame(columns=[
            "time_hours", "channel_index", "voltage", "temperature"
        ]).astype({
            "time_hours": float,
            "channel_index": int,
            "voltage": float,
            "temperature": float,
        })

    return top_df, cell_df


def active_channel_count(cell_df: pd.DataFrame) -> int:
    """Return the number of distinct active channels in *cell_df*."""
    return cell_df["channel_index"].nunique()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_time_to_hours(time_series: pd.Series) -> pd.Series:
    """
    Convert a Series of timestamp strings to elapsed-hours from the first row.

    Supported formats
    -----------------
    ``DD_MM_YY_HH_MM_SS``  (e.g. ``26_05_21_10_36_51``)
    ``HH:MM:SS``           (e.g. ``10:36:51``)

    .. note::
        The task spec documents the format as ``YY_MM_DD_HH_MM_SS`` but the
        actual tester output uses **day-first** ordering (``DD_MM_YY``).
        Both interpretations are tried; the one that produces the shorter
        elapsed-hour span (i.e. plausible test duration) is kept.

    Midnight wraparound is handled for the ``HH:MM:SS`` format: whenever a
    timestamp is strictly earlier than the previous one the entire tail is
    shifted forward by 24 h.
    """
    first = time_series.dropna().iloc[0]

    if "_" in first:
        # Try DD_MM_YY_HH_MM_SS first (day-first, used by the real tester).
        # Fall back to YY_MM_DD_HH_MM_SS if the day-first parse fails.
        try:
            parsed_dmy = pd.to_datetime(time_series, format="%d_%m_%y_%H_%M_%S")
            parsed_ymd = pd.to_datetime(time_series, format="%y_%m_%d_%H_%M_%S")
            # Pick whichever gives a shorter (more plausible) time span
            span_dmy = (parsed_dmy.iloc[-1] - parsed_dmy.iloc[0]).total_seconds()
            span_ymd = (parsed_ymd.iloc[-1] - parsed_ymd.iloc[0]).total_seconds()
            parsed = parsed_dmy if span_dmy <= span_ymd else parsed_ymd
        except Exception:
            # If one format fails entirely, fall back to the other
            try:
                parsed = pd.to_datetime(time_series, format="%d_%m_%y_%H_%M_%S")
            except Exception:
                parsed = pd.to_datetime(time_series, format="%y_%m_%d_%H_%M_%S")
        elapsed = (parsed - parsed.iloc[0]).dt.total_seconds() / 3600.0
    else:
        # HH:MM:SS → seconds since midnight, then handle midnight rollover
        parsed = pd.to_datetime(time_series, format="%H:%M:%S")
        seconds = (
            parsed.dt.hour * 3600
            + parsed.dt.minute * 60
            + parsed.dt.second
        ).astype(float)

        # Cumulative correction for midnight crossings
        arr = seconds.values.copy()
        offset = 0.0
        for i in range(1, len(arr)):
            if arr[i] + offset < arr[i - 1] + offset:
                offset += 86400.0
            arr[i] += offset

        elapsed = pd.Series((arr - arr[0]) / 3600.0, index=time_series.index)

    return elapsed.reset_index(drop=True)
