# Detection Revision Design — Li-plating, Self-discharge, ISC

**Date:** 2026-05-25
**Scope:** Revise scientific validity of existing Li-plating and self-discharge methods; add a new soft/incipient internal short-circuit (ISC) detection module.
**Approach:** In-place revision (Approach A). No interface breaking changes.

---

## Context

The `stress_screen` tool screens battery-pack modules (LFP, 0.5–1 C charge rate) for three fault types:

| Fault | Current coverage | Gap |
|---|---|---|
| Li-plating | 5 sub-methods in `li_plating.py` | dV/dQ in wrong domain; T threshold too conservative |
| Self-discharge | M1–M6 in `rest.py` | M5 uses linear Arrhenius approx; M6 discards rank slope |
| Internal short (ISC) | None | Entire module missing |

Data available: pack-level `current` (top_df) + per-channel `voltage` and `temperature` (cell_df). No per-channel current.

---

## Fix 1 — dV/dQ domain correction (`li_plating.py`)

**Problem:** `_compute_dvdq_peak_sum` computes `np.gradient(voltage)` with respect to sample index, making the result sampling-rate-dependent and physically meaningless. True dV/dQ requires charge throughput Q = ∫|I|·dt as the x-axis.

**Solution:**

- `run_li_plating_analysis` gains two optional parameters: `top_charge_df: pd.DataFrame | None = None` (columns: `time_hours`, `current`) and `n_parallel: int = 1`.
- When `top_charge_df` is present, build a pack-level Q axis:
  ```
  Q = cumtrapz(|I_pack|, time_hours) / n_parallel
  ```
  The CLI passes `topology.parallel` as `n_parallel`; callers that omit it default to 1 (series-only, no division).
- `_compute_dvdq_peak_sum` gains an optional `q_axis: np.ndarray | None` parameter. When provided it passes `q_axis` as the second argument to `np.gradient`. When absent, falls back to index-based gradient with a `warnings.warn`.
- The CLI (`_run` in `cli.py`) slices `top_df` to the charge segment and passes it to `run_li_plating_analysis`.

**Backward compatibility:** existing callers that omit `top_charge_df` continue to work with index-based dV/dQ.

---

## Fix 2 — T_plating_threshold_c: 15 → 20 °C (`li_plating.py`)

**Problem:** The current default of 15 °C is appropriate for high-C fast charging but too conservative for LFP at 0.5–1 C, where plating can occur up to ~20 °C. Cells charging at 16–20 °C are incorrectly gated out.

**Solution:** Change the default in `LiPlatingParams`:
```python
T_plating_threshold_c: float = 20.0
```
No API change. Existing callers that set this explicitly are unaffected.

---

## Fix 3 — Late ΔT noise guard (`li_plating.py`)

**Problem:** When `|dT_late| < 0.3 °C` the signal is within sensor noise. Near-zero values pollute the heat_z z-score distribution, adding noise to the composite score.

**Solution:** After computing `dT_late_metrics[ch]`, apply:
```python
if abs(dT_late_metrics[ch]) < 0.3:
    dT_late_metrics[ch] = np.nan
```
The 0.3 °C guard is not exposed as a parameter (it is a sensor-noise floor, not a scientific tunable).

---

## Fix 4 — M5 proper Arrhenius correction (`rest.py`)

**Problem:** Current correction `k / (1 + 0.02 * (T − 25))` is a first-order Taylor approximation of the Arrhenius equation, accurate only within ±3 °C of 25 °C. It under-corrects cold cells and over-corrects warm cells.

**Solution:** Replace with the two-temperature Arrhenius form:
```
k_corrected = k_raw × exp(−Ea/R × (1/T_ref − 1/T_mean))
```
- `Ea` = activation energy in eV (default 0.5 eV for LFP self-discharge)
- `R` = 8.314 J/mol·K
- `T_ref` = 298.15 K (25 °C)
- Temperatures converted to Kelvin before use
- `Ea` is converted to J/mol internally: `Ea_J = Ea_eV × 96 485`

New `RestParams` field:
```python
arrhenius_ea_ev: float = 0.5
```

---

## Fix 5 — M6 combined rank score (`rest.py`)

**Problem:** `rank_slope` (linear regression of percentile rank over time) is computed but excluded from the M6 z-score. A cell that starts mid-pack but trends steadily downward is invisible until it crosses the 20th percentile.

**Solution:** Replace the single `frac_bot20`-based z-score with a composite scalar:
```
m6_score = frac_bot20 + w_slope × max(0, −rank_slope × T_span)
```
- `T_span` = rest duration in hours (from `t_set` range)
- `w_slope` = weighting factor (default 0.5); negative slope × positive span = positive penalty for downward-trending cells
- `max(0, ...)` ensures upward-trending cells receive no benefit (we are looking for deterioration)
- `w_slope` is exposed as `RestParams.m6_slope_weight: float = 0.5`

The robust z-score is then computed over the fleet's `m6_score` values as before.

---

## New Module — ISC Detection (`analysis/short_circuit.py`)

### Public API

```python
run_isc_analysis(
    rest_cell_df: pd.DataFrame,
    rest_results: dict[int, list[MethodResult]],  # M1 k extracted from metadata
    charge_cell_df: pd.DataFrame,
    params: ShortCircuitParams | None = None,
    top_charge_df: pd.DataFrame | None = None,
    n_parallel: int = 1,
) -> dict[int, MethodResult]
```

Returns `dict[int, MethodResult]` with `method_name="isc"`. The function extracts `m1_k` internally from `rest_results[ch][0].metadata["k"]` (M1 is always index 0), avoiding any change to `run_rest_analysis`'s return type.

### `ShortCircuitParams`

```python
@dataclass
class ShortCircuitParams:
    z_thresh: float = 2.0
    isc_k_sigma: float = 3.0    # S1 absolute excess multiplier (× MAD above median)
    settling_h: float = 2.0     # shared with RestParams
    min_points: int = 60
    dv_smooth_window: int = 11  # shared with LiPlatingParams for S3
```

### S1 — Excess self-discharge rate

Extracts per-channel `k` from `rest_results[ch][0].metadata["k"]`. Computes fleet median and MAD:
```
s1_excess[ch] = max(0, k[ch] − (median_k + isc_k_sigma × MAD_k))
```
Robust z-score over all `s1_excess` values gives `s1_z`. This flags cells whose absolute discharge rate places them well outside the fleet, not just relative outliers.

### S2 — Thermal anomaly during rest

For each channel, compute `dT/dt` via linear regression of temperature vs `time_hours` over the settled rest window. An ISC continuously dissipates `I_short² × R_cell` as heat, producing a small but persistent positive slope while healthy cells cool.

Robust z-score over all `dT/dt` values gives `s2_z`. Returns `nan` if temperature data is absent or constant.

### S3 — Charge-acceptance shape (dV/dQ area deficit)

Reuses the Q-domain dV/dQ from Fix 1. Computes the area under the smoothed dV/dQ curve via `np.trapz`. A cell with an active ISC during charge loses charge to the short, so its dV/dQ profile covers less Q — the area is smaller. Invert the robust z-score (low area = high z, suspicious).

Falls back to `nan` gracefully when `top_charge_df` is absent.

### Composite ISC z-score

```
z_isc = nanmean([s1_z, s2_z, s3_z])
```
Verdict uses the same `_verdict(z, z_thresh)` helper as the rest methods.

---

## Aggregate update (`aggregate.py`)

`aggregate` gains an optional parameter:
```python
def aggregate(
    rest_results,
    li_plating_results,
    topology,
    isc_results: dict[int, MethodResult] | None = None,
    z_thresh: float = 2.0,
) -> list[ModuleVerdict]:
```

When `isc_results` is provided, its z-score is appended to the per-channel list before computing `composite_z` and `n_methods_high`. Method count becomes 8. The `HIGH` threshold logic (`n_high >= 2 or composite_z > 2.0`) is unchanged.

---

## CLI update (`cli.py`)

In `_run`:
1. After step 4 (slicing), slice `top_df` to the charge segment window to produce `charge_top_df`.
2. Pass `charge_top_df` and `topology.parallel` to `run_li_plating_analysis`.
3. Add step 6b: call `run_isc_analysis(rest_cell_df, rest_results, charge_cell_df, top_charge_df=charge_top_df, n_parallel=topology.parallel)`. No change to `run_rest_analysis`'s return type.
4. Pass `isc_results` to `aggregate`.

---

## Testing

### Revised tests (`test_li_plating.py`)

| Test | Assertion |
|---|---|
| `test_dvdq_uses_q_domain` | With constant-current `top_charge_df`, peak prominence sum differs from index-based result |
| `test_dvdq_fallback_no_top_df` | Omitting `top_charge_df` returns without error |
| `test_t_threshold_20c` | Cell at 19 °C: gate ≥ 0.05; cell at 21 °C: gate ≈ 0 |
| `test_dt_late_noise_guard` | Cell with `|dT_late| < 0.3 °C` produces `heat_z = nan` |

### Revised tests (`test_rest_methods.py`)

| Test | Assertion |
|---|---|
| `test_m5_arrhenius_vs_linear` | At T = 35 °C, Arrhenius-corrected k differs from linear approx by > 1 % |
| `test_m6_slope_penalises_trending_cell` | Same `frac_bot20` as peers, negative `rank_slope` → higher M6 z-score |

### New tests (`tests/test_short_circuit.py`)

| Test | Assertion |
|---|---|
| `test_isc_returns_all_channels` | S1–S3 composite returned for every channel |
| `test_s1_high_k_cell_flagged` | Channel with 10× fleet-median k → HIGH S1 verdict |
| `test_s2_warming_cell_flagged` | Channel rising 0.05 °C/h vs flat peers → elevated S2 z |
| `test_s3_area_deficit_cell_flagged` | Channel with 20 % less dV/dQ area → higher S3 z than peers |
| `test_s3_fallback_no_top_df` | Absent `top_charge_df` → S3 returns `nan`, no error |
| `test_isc_aggregate_integration` | Full aggregate with ISC produces `CellVerdict` with 8 `method_results` |

---

## Files changed

| File | Change |
|---|---|
| `src/stress_screen/analysis/li_plating.py` | Fixes 1, 2, 3 |
| `src/stress_screen/analysis/rest.py` | Fixes 4, 5 |
| `src/stress_screen/analysis/short_circuit.py` | New file |
| `src/stress_screen/analysis/__init__.py` | Export `run_isc_analysis`, `ShortCircuitParams` |
| `src/stress_screen/analysis/aggregate.py` | Add optional `isc_results` parameter |
| `src/stress_screen/cli.py` | Pass `top_charge_df` to li_plating; wire up ISC analysis step |
| `tests/test_li_plating.py` | Add 4 new tests |
| `tests/test_rest_methods.py` | Add 2 new tests |
| `tests/test_short_circuit.py` | New file, 6 tests |
