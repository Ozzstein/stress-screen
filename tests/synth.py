"""
synth.py — deterministic synthetic tester-CSV fixtures for tests.

Generates a small but protocol-valid pack CSV in the exact tester format:
semicolon-delimited, decimal commas, leading ``#`` comment lines, a trailing
semicolon on every data row, integer millivolt cell voltages, integer °C
temperatures, and a ``_D<DDMMYYYY>_ ... _M<n>`` filename.

The default file is a 2-module 4P8S pack (16 channels) with a ~3 h
conditioning charge followed by a 50 h rest — long enough to satisfy the 48 h
protocol gate while keeping the full pipeline run at a few seconds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

#: 4P8S structural constants (see loader._sensor_csv_channel): each module
#: pair occupies 14 temp columns + a 2-column gap with no sensor wired.
_PAIR_GAP_COLS = (15, 16)


def make_synthetic_csv(
    out_dir: Path,
    n_modules: int = 2,
    series: int = 8,
    charge_hours: float = 3.0,
    rest_hours: float = 50.0,
    dt_s: float = 60.0,
    leaky_channels: dict[int, float] | None = None,
    baseline_leak_mv_per_h: tuple[float, float] = (0.02, 0.08),
    seed: int = 42,
    filename: str | None = None,
) -> Path:
    """Write a synthetic tester CSV and return its path.

    Parameters
    ----------
    out_dir:
        Directory to write into (created if missing).
    n_modules, series:
        Pack layout; channels = n_modules * series. With series=8 the derived
        topology is 4P8S (parallel = 32 // series).
    charge_hours, rest_hours:
        Segment durations. rest_hours must be >= 48 for a protocol-valid file.
    dt_s:
        Sampling interval in seconds.
    leaky_channels:
        Mapping of 0-based channel index -> extra self-discharge slope in
        mV/h applied during rest (e.g. ``{3: 0.8}`` makes channel 3 lose an
        extra ~40 mV over a 50 h rest — a clear fleet outlier).
    baseline_leak_mv_per_h:
        (low, high) bounds of the uniform self-discharge slope every cell
        gets. A real fleet is never perfectly uniform; without this spread
        the fleet MAD collapses to ~0 and quantization noise produces false
        flags on "healthy" cells. Bounded uniform (not gaussian) so a
        healthy fleet cannot contain an accidental extreme outlier.
    seed:
        RNG seed; the file is fully deterministic for a given argument set.
    filename:
        Override the generated ``SynthPack_D01032026_HHMMSS_M<n>.csv`` name.
    """
    rng = np.random.default_rng(seed)
    leaky_channels = leaky_channels or {}
    n_channels = n_modules * series

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"SynthPack_D01032026_080000_M{n_modules}.csv"
    out_path = out_dir / filename

    n_charge = int(charge_hours * 3600 / dt_s)
    n_rest = int(rest_hours * 3600 / dt_s)
    n_rows = n_charge + n_rest
    t_h = np.arange(n_rows) * dt_s / 3600.0  # elapsed hours per row

    start = datetime(2026, 3, 1, 8, 0, 0)
    timestamps = [
        (start + timedelta(seconds=i * dt_s)).strftime("%d_%m_%y_%H_%M_%S")
        for i in range(n_rows)
    ]

    # --- Pack-level traces -------------------------------------------------
    current = np.full(n_rows, 0.02)
    current[:n_charge] = 5.0
    soc = np.full(n_rows, 100.0)
    soc[:n_charge] = np.linspace(62.0, 100.0, n_charge)

    # --- Per-channel voltage (mV) and temperature (°C) ----------------------
    # Charge: common ramp 3200 -> 3448 mV plus a fixed per-cell offset.
    # Rest: settling exponential (~15 mV, tau ~1 h) on a flat OCV plateau,
    # plus a per-channel leak slope for injected anomalies.
    cell_offset_mv = rng.normal(0.0, 2.0, size=n_channels)
    base_leak = rng.uniform(
        baseline_leak_mv_per_h[0], baseline_leak_mv_per_h[1], n_channels
    )
    volt_mv = np.empty((n_rows, n_channels))
    temp_c = np.empty((n_rows, n_channels))

    charge_ramp = np.linspace(3200.0, 3448.0, n_charge)
    t_rest = t_h[n_charge:] - t_h[n_charge]
    sensor_offset = rng.normal(0.0, 0.5, size=n_channels)

    for ch in range(n_channels):
        leak_mv_h = base_leak[ch] + leaky_channels.get(ch, 0.0)
        noise = rng.normal(0.0, 0.4, size=n_rows)
        volt_mv[:n_charge, ch] = charge_ramp + cell_offset_mv[ch] + noise[:n_charge]
        volt_mv[n_charge:, ch] = (
            3350.0
            + cell_offset_mv[ch]
            + 15.0 * np.exp(-t_rest / 1.0)
            - leak_mv_h * t_rest
            + noise[n_charge:]
        )
        # Warm slightly during charge, cool to ambient during rest.
        temp_c[:n_charge, ch] = 22.0 + 3.0 * np.linspace(0, 1, n_charge) + sensor_offset[ch]
        temp_c[n_charge:, ch] = 19.0 + 6.0 * np.exp(-t_rest / 2.0) + sensor_offset[ch]

    volt_mv = np.rint(volt_mv).astype(int)
    temp_c = np.rint(temp_c).astype(int)

    pack_voltage = volt_mv.mean(axis=1) * series / 1000.0

    # --- Write the file ------------------------------------------------------
    # Temp columns follow the paired 4P8S sensor layout: sensors live in the
    # first 14 columns of each module pair; the structural gap columns read 0
    # (dead sensor -> NaN in the loader).
    header_cells = ";".join(
        f"Cell_{n}_Volt;Cell_{n}_Temp" for n in range(1, n_channels + 1)
    )
    lines = [
        "# synthetic stress_screen test fixture",
        f"# seed={seed} modules={n_modules} series={series}",
        f"TimeStamp;Current;Voltage;SOC %;Warning;Fault;{header_cells}",
    ]

    def _num(x: float, nd: int = 2) -> str:
        return f"{x:.{nd}f}".replace(".", ",")

    gap_cols = set()
    if series == 8:  # 4P8S paired layout
        n_pairs = (n_modules + 1) // 2
        for pair in range(n_pairs):
            for col in _PAIR_GAP_COLS:
                gap_cols.add(pair * 16 + col)  # 1-based temp column number

    for i in range(n_rows):
        cells = []
        for n in range(1, n_channels + 1):
            t_val = 0 if n in gap_cols else temp_c[i, n - 1]
            cells.append(f"{volt_mv[i, n - 1]};{t_val}")
        lines.append(
            f"{timestamps[i]};{_num(current[i])};{_num(pack_voltage[i])};"
            f"{_num(soc[i], 1)};OK;OK;" + ";".join(cells) + ";"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
