# stress-screen

Battery pack stress-test module screener. Given a tester CSV from a multi-module battery pack (4P8S / 2P16S / 1P32S topologies), `stress-screen` runs eight independent detection methods over the rest and conditioning-charge phases and produces a per-module **OK / OK - Marginal / NOK** verdict plus an HTML and PDF report.

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

### Composite: evidence clusters

The eight methods are not eight independent measurements — they share inputs
by construction, forming five evidence clusters:

| Cluster | Members | Physical signal |
|---|---|---|
| `self_discharge` | `ocv_k`, `temp_k` | Self-discharge rate (raw + T-corrected) |
| `residual_dynamics` | `thermal_corr`, `cusum` | Structure in the OCV-fit residuals |
| `fleet_divergence` | `spread`, `rank` | Drift away from the fleet |
| `li_plating` | `li_plating` | Plating signatures |
| `isc` | `isc` | Internal-short signatures |

Member z-scores average within their cluster (redundant noisy readings of one
signal), each cluster score is winsorized to ±8, and the composite is the
weighted mean of cluster scores (all weights 1.0 by default; tune them with
evidence via `stress_screen calibrate`). The pre-cluster plain-mean composite
is still computed on every run and stored as `composite_z_legacy` in the JSON
result; select it outright with `composite: {mode: legacy}` in a config file.

### Verdict logic

Each cell becomes:

- **HIGH** if `composite_z > 2.0`, **or** `(n_clusters_HIGH ≥ 2 AND composite_z ≥ 1.0)`
- **ELEVATED** if `composite_z > 1.0`, **or** `(n_clusters_HIGH ≥ 1 AND composite_z ≥ 0.5)`
- **NORMAL** otherwise

(`n_clusters_HIGH` counts clusters at z ≥ 2.0 — twin methods like `ocv_k` and
`temp_k` firing together count as one vote, not two.)

Modules roll up as:

- **NOK** if any cell is HIGH
- **OK - Marginal** if any cell is ELEVATED
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

## Download the pre-built binary (no Python install required)

Every release publishes stand-alone executables for Windows, macOS, and Linux to the [Releases page](https://github.com/Ozzstein/stress-screen/releases/latest). Nothing to install — download the file for your OS and run it.

| Platform | Asset | How to use |
|---|---|---|
| Windows 10 / 11 (x64) | `stress_screen-windows-x64.exe` | Drop into any folder, double-click Command Prompt in that folder, run `stress_screen-windows-x64.exe your-file.csv` |
| macOS (Apple Silicon) | `stress_screen-macos-arm64` | `chmod +x stress_screen-macos-arm64 && ./stress_screen-macos-arm64 your-file.csv` |
| Linux (x64) | `stress_screen-linux-x64` | `chmod +x stress_screen-linux-x64 && ./stress_screen-linux-x64 your-file.csv` |

### Windows quick start

1. Download `stress_screen-windows-x64.exe` from the Releases page.
2. (Optional but recommended) rename it to `stress_screen.exe` for shorter commands.
3. Place it in the folder that contains your tester CSV, e.g. `C:\Tests\`.
4. Open Command Prompt (`Win + R` → type `cmd` → Enter), then:
   ```
   cd C:\Tests
   stress_screen.exe DataLogging_C1_I01_P18052026_M6.csv
   ```
5. You'll see the per-module verdict lines. An HTML report and a PDF report are written next to your CSV file.

Common Windows commands:

```cmd
REM Only verdict lines, no reports:
stress_screen.exe my_test.csv --no-html --no-pdf

REM Full output (progress, header, summary):
stress_screen.exe my_test.csv --full

REM Verbose z-scores for every flagged cell:
stress_screen.exe my_test.csv -v

REM Send results to a specific output folder:
stress_screen.exe my_test.csv --out-dir C:\Reports\
```

### macOS / Linux quick start

```bash
# One-time: make the file executable
chmod +x stress_screen-macos-arm64        # or stress_screen-linux-x64

# Run it
./stress_screen-macos-arm64 DataLogging_C1_I01_P18052026_M6.csv
```

### First-time on Windows: SmartScreen warning

Because the binary isn't signed with a Microsoft-issued code-signing certificate, Windows SmartScreen may show *"Windows protected your PC"* the first time you run it. Click **More info → Run anyway**. This is standard for unsigned open-source binaries.

### First-time on macOS: Gatekeeper warning

macOS may block the binary with *"cannot be opened because the developer cannot be verified"*. Fix it once with:

```bash
xattr -d com.apple.quarantine stress_screen-macos-arm64
```

Or right-click the file → *Open* → *Open* to bypass the warning through the GUI.

## Quick start

```bash
stress_screen DataLogging_C1_I01_P18052026_M6.csv
```

Default output is terse — only the per-module verdict lines:

```
M1: OK - Marginal  [cells elevated: M1/G1 (ELEVATED)]
M2: OK
M3: OK
M4: OK - Marginal  [cells elevated: M4/G8 (ELEVATED)]
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

M1: OK - Marginal  [cells elevated: M1/G1 (ELEVATED)]
... (verdict lines)

Result: 2 of 6 modules OK - Marginal
HTML report: /path/to/DataLogging_C1_I01_P18052026_M6_report.html
PDF report:  /path/to/DataLogging_C1_I01_P18052026_M6_report.pdf
```

### Verbose output (per-method z-scores for flagged cells)

```bash
stress_screen <csv> -v
```

Adds a block under each flagged cell showing every method's z-score and verdict.

## CLI reference

The historical single-command form still works unchanged (`stress_screen
file.csv ...` is an implicit `run`). Subcommands: `run`, `calibrate`,
`trend`, `history`, `batch`.

### `stress_screen [run] CSV`

```
stress_screen [-h] [--chem {lfp,nmc,nca}] [--config PATH] [--c-rate X]
              [--capacity-ah X] [--mapping PATH] [--out-dir DIR]
              [--no-html] [--no-pdf] [--no-json] [--json-out PATH]
              [--history DIR] [--pack-id ID] [--downsample N] [-v] [--full] CSV
```

| Flag | Default | Description |
|---|---|---|
| `CSV` | — | Path to the tester CSV (semicolon-delimited, comma-decimal). Filename must contain `_M<n>` where `n` is the module count, e.g. `..._M6.csv`. |
| `--chem {lfp,nmc,nca}` | `lfp` | Chemistry preset (OCV voltage bounds and chemistry-specific parameters; see `configs/analysis_defaults.yaml`). |
| `--config PATH` | — | YAML file overriding any analysis parameter (thresholds, activation energies, verdict gates, composite weights). Every available key is documented in `configs/analysis_defaults.yaml`. Unknown keys are rejected. |
| `--c-rate X` | `0.5` | Cell-level charge C-rate of the test protocol; scales dQ/dV peak detection and thermal noise floors. |
| `--capacity-ah X` | `2.5` | Nominal cell capacity. |
| `--mapping PATH` | bundled `configs/temp_mapping.yaml` | Override the temperature sensor → group mapping (see *Temperature sensor layout* below). |
| `--out-dir DIR` | input CSV's directory | Where the HTML / PDF / JSON outputs are written. |
| `--no-html` / `--no-pdf` / `--no-json` | off | Skip that output. |
| `--json-out PATH` | `<stem>_result.json` | Write the JSON result to a specific path. |
| `--history DIR` | — | Also add this run's JSON result to a history store (see *Fleet tracking*). |
| `--pack-id ID` | derived from filename | Override the pack identifier recorded in the JSON result. |
| `--downsample N` | `1` (no downsampling) | Keep every Nth row. Use `1` for production runs — see *Downsampling caveats* below. |
| `-v`, `--verbose` | off | Show per-cluster and per-method z-scores under each flagged cell. |
| `--full` | off | Print pack header, progress messages, result summary, and report paths in addition to the verdict lines. |

### JSON result

Every run writes `<stem>_result.json` next to the reports: a versioned,
machine-readable record of the full verdict tree — per-cell composite and
cluster scores, every method's z-score and metadata (fitted k, tau, gate
values, ISC sub-scores), the resolved configuration, topology, and segments.
This is the substrate for calibration, trend tracking, and regression
testing; keep these files.

### `stress_screen calibrate`

```
stress_screen calibrate --results DIR --labels labels.csv [--sweep]
```

Scores past JSON results against known outcomes. The labels file is
semicolon-delimited: `pack_id;module_id;group;outcome` with outcome
`good`/`bad` (empty group = module-level label). Reports confusion matrices
at the current gates, rank-AUC separation power per method and per cluster,
and with `--sweep` a composite-z threshold sweep with a suggested operating
point. Record outcomes (teardowns, capacity tests, field returns) as they
arrive — the thresholds stay statistical defaults until this command says
otherwise.

### Fleet tracking: `trend`, `history`, `batch`

```
stress_screen batch DIR --history STORE [--no-html --no-pdf]
stress_screen history --history STORE [--pack ID] [--rebuild-index]
stress_screen trend --history STORE --pack ID [--module M --group G]
```

A history store is a plain directory of JSON results plus an append-only
`index.jsonl` (inspectable, diffable, safe on network shares; the index is
rebuildable). `trend` compares a pack across test dates on **raw physical
metrics** (self-discharge k, its 25 °C-corrected variant) rather than
z-scores — z is fleet-relative within one run, so a uniformly degrading pack
shows flat z but a climbing k. A cell is flagged WORSENING when its Theil–Sen
k-slope over ≥ 3 runs exceeds the floor, or when its verdict enters
ELEVATED/HIGH after a clean history (exit code 1 when anything is flagged).
`batch` analyses every `*_M<n>.csv` in a directory, continues on errors, and
exits with the worst per-file code.

## Input CSV format

The tester emits a semicolon-delimited file with European decimal commas:

```
TimeStamp;Current;Voltage;SOC %;Warning;Fault;Cell_1_Volt;Cell_1_Temp;Cell_2_Volt;Cell_2_Temp;...
26_05_21_10_36_51;0.0;26.45;100.0;OK;OK;3494;17;3497;17;...
```

- Voltage cells (`Cell_N_Volt`) are in millivolts.
- The number of active voltage channels divided by the `_M<n>` count from the filename determines the topology (4P8S, 2P16S, or 1P32S).
- Disconnected temperature sensors should read `0` — the loader treats those as NaN.

## Test protocol requirements

`stress_screen` enforces one hard precondition on the input data:

- **The test must contain a final rest period of at least 48 hours** during which `|current| < 0.5 A`.

This is a protocol requirement, not a tuneable parameter. Without 48 h of rest, OCV self-discharge slopes cannot be resolved above measurement noise and any verdict the tool produces would be unreliable. Files that fail this check are rejected with:

```
Error: Test invalidated: no rest segment >= 48 h found in the data. The
stress-test protocol requires a final rest period of at least 48 h for
OCV analysis.
```

The process exits with code `2`. Re-run the test with a longer rest cycle.

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
| `1` | Production / official verdicts. ~30–90 s per analysis (reports add ~1–2 min). |
| `10` | Faster still. Borderline cells (composite_z ≈ 0.5) may flip a verdict bucket. |
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

The test suite contains 130+ tests covering:

- Topology derivation and temperature-sensor mapping
- CSV loader (C-engine parsing, decimal commas, midnight rollover, legacy A/B)
- Six rest-phase detection methods (synthetic + cross-mode)
- Li-plating (dQ/dV peaks + relaxation)
- ISC analysis (S1/S2/S3 sub-signals + thermal gating)
- Verdict aggregation (cluster composite, legacy bit-compat, threshold gates)
- Config resolution (presets, user YAML, unknown-key rejection)
- Golden-output regression gate (committed JSON vs full pipeline run)
- End-to-end CLI on committed synthetic fixtures (no real CSVs needed)
- Calibration harness, history store, trend analysis, batch mode
- Report generation (HTML / PDF)
- Cross-protocol confounds (thermal gradients, ambient drift)

To intentionally re-baseline the golden file after a deliberate behavior
change: `STRESS_SCREEN_REGEN_GOLDEN=1 PYTHONPATH=src uv run pytest
tests/test_golden.py`, then review the diff and commit it.

## Project layout

```
src/stress_screen/
  cli.py                  # Entry point + subcommands (run/calibrate/trend/history/batch)
  config.py               # AnalysisConfig resolution (defaults ← preset ← --config ← flags)
  loader.py               # Fast C-engine CSV parser + paired-layout temperature remap
  serialize.py            # Versioned JSON result writer + filename date/pack-id parsing
  calibrate.py            # Verdicts vs labeled outcomes (confusion, AUC, sweep)
  history.py              # History store (JSON dir + index.jsonl) + trend analysis
  topology.py             # PackTopology derivation from active-channel count
  segmentation.py         # Charge / discharge / rest segment detection
  models.py               # Dataclasses (MethodResult, CellVerdict, ModuleVerdict, PackTopology)
  analysis/
    rest.py               # ocv_k, thermal_corr, spread, cusum, temp_k, rank
    li_plating.py         # dQ/dV peaks + relaxation fit
    short_circuit.py      # ISC composite (S1 + S2 + S3)
    aggregate.py          # Cluster composite + cell/module verdict roll-up
    protocol.py           # C-rate/chemistry-aware threshold scaling
    util.py               # robust_z, CUSUM, Arrhenius correction, OCV model
  reports/
    figures.py            # Build every Plotly figure once (shared HTML/PDF)
    charts.py             # Plotly chart builders
    html.py               # HTML report writer (Jinja2)
    pdf.py                # PDF report writer (ReportLab)
    templates/report.html.j2
configs/
  temp_mapping.yaml       # Group → sensor-list mapping per topology
  analysis_defaults.yaml  # Chemistry presets + reference of every tunable key
scripts/
  composite_ab.py         # Offline legacy-vs-clustered verdict A/B from JSONs
tests/                    # 130+ unit and integration tests + golden fixture
```

## License

Private repository. All rights reserved.
