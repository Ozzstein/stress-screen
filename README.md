# stress-screen

Battery pack stress-test module screener. Given a tester CSV from a multi-module battery pack (4P8S / 2P16S / 1P32S topologies), `stress-screen` runs eight independent detection methods over the rest and conditioning-charge phases and produces a per-module **OK / MARGINAL / NOK** verdict plus an HTML and PDF report.

The screener is designed to catch the early-stage degradation modes that matter in pack qualification — self-discharge drift, thermal-coupled voltage drift, lithium plating signatures, and internal short circuits — without flagging the structural noise that comes from sensor placement and module-to-module temperature gradients.

## What it detects

Each cell-group is scored independently by eight methods. Method z-scores are robust (median / MAD) and aggregated into a confidence-weighted **composite z**.

| Method | What it measures | Detects |
|---|---|---|
| `ocv_k` | OCV-fit self-discharge rate `k` from `V(t) = V_ocv + V·exp(−k·t)` | Slow voltage decay during rest — primary capacity-loss / leakage signature |
| `thermal_corr` | Pearson r between OCV residuals and cell temperature | Cells whose voltage tracks heat — internal-resistance imbalance |
| `spread` | Slope of `\|V_cell − V_fleet_median\|` over rest (T-compensated) | Cells diverging from the fleet over time |
| `cusum` | Two-sided CUSUM alarms on OCV residuals | Step changes / persistent biases the linear fit would miss |
| `temp_k` | Arrhenius-corrected `ocv_k` normalized to 25 °C | Excess self-discharge after removing temperature bias |
| `rank` | Voltage-rank percentile drift within the module | Cells sliding down the within-module ranking |
| `li_plating` | dQ/dV peak shifts + post-charge relaxation fit | Lithium plating on the anode (cold/fast-charge induced) |
| `isc` | S1 excess `k` + S2 thermal slope + S3 dV/dQ area deficit | Internal short-circuit signatures |

### Verdict logic

Each cell becomes:

- **HIGH** if `composite_z > 2.0`, **or** `(n_methods_HIGH ≥ 2 AND composite_z ≥ 1.0)`
- **ELEVATED** if `composite_z > 1.0`, **or** `(n_methods_HIGH ≥ 1 AND composite_z ≥ 0.5)`
- **NORMAL** otherwise

Modules roll up as:

- **NOK** if any cell is HIGH
- **MARGINAL** if any cell is ELEVATED
- **OK** otherwise

Process exit code is `1` if any module is NOK, `0` otherwise.

## Install

```bash
git clone https://github.com/Ozzstein/stress-screen.git
cd stress-screen
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

Python ≥ 3.10 is required.

## Quick start

```bash
stress_screen DataLogging_C1_I01_P18052026_M6.csv
```

Default output is terse — only the per-module verdict lines:

```
M1: MARGINAL  [cells elevated: M1/G1 (ELEVATED)]
M2: OK
M3: OK
M4: MARGINAL  [cells elevated: M4/G8 (ELEVATED)]
M5: OK
M6: OK
```

HTML and PDF reports are still written to the input CSV's directory (use `--no-html` / `--no-pdf` to skip).

### Full output (pack header, progress, summary, report paths)

```bash
stress_screen <csv> --full
```

```
[1.0s] Loading CSV...
[71.6s] Loaded 238136 rows, 48 active channels
...
Pack: DataLogging_C1_I01_P18052026_M6.csv
Configuration: 6 modules, 4P8S (4 parallel × 8 series), 48 active cell-groups
Segments: 3 charge, 2 discharge, 1 rest (longest rest: 54.10 h)

M1: MARGINAL  [cells elevated: M1/G1 (ELEVATED)]
... (verdict lines)

Result: 2 of 6 modules MARGINAL
HTML report: /path/to/DataLogging_C1_I01_P18052026_M6_report.html
PDF report:  /path/to/DataLogging_C1_I01_P18052026_M6_report.pdf
```

### Verbose output (per-method z-scores for flagged cells)

```bash
stress_screen <csv> -v
```

Adds a block under each flagged cell showing every method's z-score and verdict.

## CLI reference

```
stress_screen [-h] [--chem {lfp,nmc,nca}] [--mapping PATH] [--out-dir DIR]
              [--no-html] [--no-pdf] [--downsample N] [-v] [--full] CSV
```

| Flag | Default | Description |
|---|---|---|
| `CSV` | — | Path to the tester CSV (semicolon-delimited, comma-decimal). Filename must contain `_M<n>` where `n` is the module count, e.g. `..._M6.csv`. |
| `--chem {lfp,nmc,nca}` | `lfp` | Sets OCV voltage bounds for the curve fit (LFP: 3.0–3.65 V, NMC/NCA: 3.0–4.25 V). |
| `--mapping PATH` | bundled `configs/temp_mapping.yaml` | Override the temperature sensor → group mapping (see *Temperature sensor layout* below). |
| `--out-dir DIR` | input CSV's directory | Where the HTML / PDF reports are written. |
| `--no-html` | off | Skip HTML report generation. |
| `--no-pdf` | off | Skip PDF report generation. |
| `--downsample N` | `1` (no downsampling) | Keep every Nth row. Use `1` for production runs — see *Downsampling caveats* below. |
| `-v`, `--verbose` | off | Show per-method z-scores under each flagged cell. |
| `--full` | off | Print pack header, progress messages, result summary, and report paths in addition to the verdict lines. |

## Input CSV format

The tester emits a semicolon-delimited file with European decimal commas:

```
TimeStamp;Current;Voltage;SOC %;Warning;Fault;Cell_1_Volt;Cell_1_Temp;Cell_2_Volt;Cell_2_Temp;...
26_05_21_10_36_51;0.0;26.45;100.0;OK;OK;3494;17;3497;17;...
```

- Voltage cells (`Cell_N_Volt`) are in millivolts.
- The number of active voltage channels divided by the `_M<n>` count from the filename determines the topology (4P8S, 2P16S, or 1P32S).
- Disconnected temperature sensors should read `0` — the loader treats those as NaN.

## Temperature sensor layout (4P8S)

Each 4P8S module has 7 physical sensors placed *between* the 8 series cell-groups, plus a structural 2-column gap between every pair of modules in the CSV. Sensor `k` of module `m` is read from CSV column:

```
Cell_C_Temp  where  C = ((m-1)//2)*16 + ((m-1)%2)*7 + k
```

Each group's temperature is the average of the two bracketing sensors (or the single neighbour for G1 and G8):

| Group | Sensors used | Formula |
|---|---|---|
| G1 | S1 | T(S1) |
| G2–G7 | S(k−1), Sk | avg of the two bracketing sensors |
| G8 | S7 | T(S7) |

When a sensor is dead (all-zero), the remaining valid neighbour is used; if all sensors for a group are dead, the nearest valid sensor in the same module is used as a last-resort fallback.

Single-sensor cells (G1, G8) automatically receive a √2 noise penalty on `thermal_corr` to correct for the higher per-sample noise relative to the two-sensor-averaged interior groups.

## Downsampling caveats

`--downsample 1` (the default) is the only setting that gives stable verdicts. Higher factors are faster but lose information:

| `--downsample` | Notes |
|---|---|
| `1` | Production / official verdicts. ~3–5 min per analysis. |
| `10` | ~3× faster. Borderline cells (composite_z ≈ 0.5) may flip a verdict bucket. |
| `60+` | **Not recommended.** Coarse time resolution breaks the segmenter; charge cycles can be truncated, causing false ISC NOK verdicts. |

## Output: the HTML report

The HTML report includes:

1. **Methodology section** — full description of all 8 methods, formulas, and verdict aggregation rules.
2. **Module summary table** — verdict, flagged cells, and methods fired per module.
3. **Pack overview** — composite z heatmap across all cells.
4. **Phase timeline** — charge / discharge / rest visualised on the pack-level current trace.
5. **Per-module detail** (6 charts each):
   - OCV fit overlay (rest phase)
   - dQ/dV incremental capacity (charge phase)
   - Voltage divergence from fleet median (`spread` method)
   - Voltage rank percentile over rest (`rank` method)
   - Temperature traces (rest + charge)
   - All-method z-score heatmap
6. **Per-cell method z-score table** — every cell, every method.
7. **Flagged-cell detail cards** with ISC sub-score breakdown.

## Development

```bash
uv sync
PYTHONPATH=src uv run pytest -q
```

The test suite contains 69+ tests covering:

- Topology derivation and temperature-sensor mapping
- Six rest-phase detection methods (synthetic + cross-mode)
- Li-plating (dQ/dV peaks + relaxation)
- ISC analysis (S1/S2/S3 sub-signals + thermal gating)
- Verdict aggregation (composite z, threshold gates)
- End-to-end CLI on synthetic packs
- Report generation (HTML / PDF)
- Cross-protocol confounds (thermal gradients, ambient drift)

## Project layout

```
src/stress_screen/
  cli.py                  # Entry point
  loader.py               # CSV parser + paired-layout temperature remap
  topology.py             # PackTopology derivation from active-channel count
  segmentation.py         # Charge / discharge / rest segment detection
  models.py               # Dataclasses (MethodResult, CellVerdict, ModuleVerdict, PackTopology)
  analysis/
    rest.py               # ocv_k, thermal_corr, spread, cusum, temp_k, rank
    li_plating.py         # dQ/dV peaks + relaxation fit
    short_circuit.py      # ISC composite (S1 + S2 + S3)
    aggregate.py          # Cell + module verdict roll-up
    util.py               # robust_z, CUSUM, Arrhenius correction, OCV model
  reports/
    charts.py             # Plotly chart builders
    html.py               # HTML report writer (Jinja2)
    pdf.py                # PDF report writer (ReportLab)
    templates/report.html.j2
configs/
  temp_mapping.yaml       # Group → sensor-list mapping per topology
tests/                    # 69+ unit and integration tests
```

## License

Private repository. All rights reserved.
