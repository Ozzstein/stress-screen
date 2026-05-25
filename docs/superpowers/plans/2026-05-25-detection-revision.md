# Detection Method Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix five scientific errors in Li-plating and self-discharge detection, and add a new soft internal short-circuit (ISC) detection module — all in-place with full test coverage.

**Architecture:** Five targeted in-place fixes across `li_plating.py` and `rest.py`, plus a new `analysis/short_circuit.py` wired through `aggregate.py` and `cli.py`. Every fix follows TDD order: failing test → minimal implementation → passing test → commit.

**Tech Stack:** Python ≥3.10, NumPy, SciPy ≥1.11 (`cumulative_trapezoid`), pandas, pytest

---

## File Map

| File | Action | Scope |
|---|---|---|
| `src/stress_screen/analysis/li_plating.py` | Modify | Fixes 1 (dV/dQ domain), 2 (T threshold), 3 (ΔT noise guard) |
| `src/stress_screen/analysis/rest.py` | Modify | Fixes 4 (Arrhenius M5), 5 (M6 rank slope) |
| `src/stress_screen/analysis/short_circuit.py` | Create | ISC methods S1–S3 |
| `src/stress_screen/analysis/__init__.py` | Modify | Export ISC public API |
| `src/stress_screen/analysis/aggregate.py` | Modify | Optional `isc_results` parameter |
| `src/stress_screen/cli.py` | Modify | Pass `charge_top_df`; wire ISC analysis step |
| `tests/test_li_plating.py` | Modify | 4 new tests for Fixes 1–3 |
| `tests/test_rest_methods.py` | Modify | 2 new tests for Fixes 4–5 |
| `tests/test_short_circuit.py` | Create | 6 new tests for ISC module + aggregate |

---

### Task 1: Li-plating fixes — Q-domain dV/dQ, T threshold, ΔT noise guard

**Files:**
- Modify: `src/stress_screen/analysis/li_plating.py`
- Modify: `tests/test_li_plating.py`

- [ ] **Step 1.1: Write four failing tests**

Append to `tests/test_li_plating.py`:

```python
import warnings


def _make_top_charge_df(n_points=200, current=5.0):
    t = np.linspace(0, 2.0, n_points)
    return pd.DataFrame({"time_hours": t, "current": np.full(n_points, current)})


def test_dvdq_uses_q_domain():
    """Peak prominence sum must differ between Q-domain and index-domain calls."""
    charge = _make_charge_df(n_channels=5)
    rest = _make_rest_df(n_channels=5)
    top_charge = _make_top_charge_df(current=5.0)  # dq ≈ 0.05 Ah/sample → ~20× scale factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_idx = run_li_plating_analysis(charge, rest)
    results_q = run_li_plating_analysis(charge, rest, top_charge_df=top_charge, n_parallel=1)

    # Channel 0 has an injected bump → non-zero prominence in both calls
    idx_sum = results_idx[0].metadata["peak_prominence_sum"]
    q_sum = results_q[0].metadata["peak_prominence_sum"]
    assert idx_sum > 0, "Injected peak on ch0 should give non-zero index-based prominence"
    assert abs(q_sum - idx_sum) / (idx_sum + 1e-10) > 0.01, (
        f"Q-domain prominence {q_sum:.6f} too similar to index-based {idx_sum:.6f}"
    )


def test_dvdq_fallback_no_top_df():
    """Omitting top_charge_df issues a warning but returns normally."""
    charge = _make_charge_df()
    rest = _make_rest_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run_li_plating_analysis(charge, rest)
    assert len(results) == 5
    messages = " ".join(str(w.message) for w in caught).lower()
    assert "index" in messages or "top_charge_df" in messages, (
        f"Expected Q-domain fallback warning, got: {[str(w.message) for w in caught]}"
    )


def test_t_threshold_20c():
    """Default T_plating_threshold_c=20°C: 19°C cell gate ≥0.05; 21°C cell gate=0."""
    charge = _make_charge_df(n_channels=4)
    rest = _make_rest_df(n_channels=4)
    charge.loc[charge["channel_index"] == 0, "temperature"] = 19.0
    charge.loc[charge["channel_index"] == 1, "temperature"] = 21.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    assert results[0].metadata["temperature_gate"] >= 0.05, (
        f"19°C cell gate={results[0].metadata['temperature_gate']:.4f} must be ≥0.05 with 20°C threshold"
    )
    assert results[1].metadata["temperature_gate"] == 0.0, (
        f"21°C cell gate={results[1].metadata['temperature_gate']:.4f} must be 0.0"
    )


def test_dt_late_noise_guard():
    """dT_late < 0.3°C should produce heat_z = nan (sub-noise signal discarded)."""
    charge = _make_charge_df(n_channels=5)  # uniform T=25°C → dT_late=0.0 for all
    rest = _make_rest_df(n_channels=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_li_plating_analysis(charge, rest)
    for ch in range(5):
        assert np.isnan(results[ch].metadata["heat_z"]), (
            f"ch{ch}: expected heat_z=nan for |dT_late|<0.3°C, "
            f"got heat_z={results[ch].metadata['heat_z']}"
        )
```

- [ ] **Step 1.2: Run tests — confirm all 4 fail**

```bash
.venv/bin/pytest tests/test_li_plating.py::test_dvdq_uses_q_domain \
  tests/test_li_plating.py::test_dvdq_fallback_no_top_df \
  tests/test_li_plating.py::test_t_threshold_20c \
  tests/test_li_plating.py::test_dt_late_noise_guard -v
```

Expected output:
- `test_dvdq_*` — `TypeError: unexpected keyword argument 'top_charge_df'`
- `test_t_threshold_20c` — gate for 19°C cell = 0.0 (not ≥0.05), because default threshold is 15°C
- `test_dt_late_noise_guard` — heat_z = 0.0 (not nan), because noise guard not yet applied

- [ ] **Step 1.3: Add `cumulative_trapezoid` import to `li_plating.py`**

In `src/stress_screen/analysis/li_plating.py`, add after the existing scipy imports (after line 32 `from scipy.optimize import OptimizeWarning, curve_fit`):

```python
from scipy.integrate import cumulative_trapezoid
```

- [ ] **Step 1.4: Add `q_axis` parameter to `_compute_dvdq_peak_sum`**

In `src/stress_screen/analysis/li_plating.py`, change the function signature at line 96 from:

```python
def _compute_dvdq_peak_sum(
    voltage: np.ndarray,
    smooth_window: int,
    peak_prominence_pct: float,
) -> float:
```

to:

```python
def _compute_dvdq_peak_sum(
    voltage: np.ndarray,
    smooth_window: int,
    peak_prominence_pct: float,
    q_axis: np.ndarray | None = None,
) -> float:
```

Then change line 108 (`dv_dq = np.gradient(voltage)`) to:

```python
    if q_axis is not None:
        dv_dq = np.gradient(voltage, q_axis)
    else:
        dv_dq = np.gradient(voltage)
```

- [ ] **Step 1.5: Change `T_plating_threshold_c` default from 15 to 20**

In `LiPlatingParams` (line 65), change:

```python
    T_plating_threshold_c: float = 15.0
```

to:

```python
    T_plating_threshold_c: float = 20.0
```

- [ ] **Step 1.6: Update `run_li_plating_analysis` signature**

Change the signature at lines 189–193 from:

```python
def run_li_plating_analysis(
    charge_cell_df: pd.DataFrame,
    rest_cell_df: pd.DataFrame,
    params: LiPlatingParams | None = None,
) -> dict[int, MethodResult]:
```

to:

```python
def run_li_plating_analysis(
    charge_cell_df: pd.DataFrame,
    rest_cell_df: pd.DataFrame,
    params: LiPlatingParams | None = None,
    top_charge_df: pd.DataFrame | None = None,
    n_parallel: int = 1,
) -> dict[int, MethodResult]:
```

- [ ] **Step 1.7: Build Q axis and issue fallback warning inside the function**

After `if params is None: params = LiPlatingParams()` (line 213), insert:

```python
    # Build pack-level Q axis from charge current when available
    q_pack_time: np.ndarray | None = None
    q_pack_cumul: np.ndarray | None = None
    if top_charge_df is None:
        warnings.warn(
            "run_li_plating_analysis: top_charge_df not provided; dV/dQ computed vs "
            "sample index (physically incorrect). Pass top_charge_df for Q-domain analysis.",
            UserWarning,
            stacklevel=2,
        )
    elif len(top_charge_df) >= 2:
        _top = top_charge_df.sort_values("time_hours")
        q_pack_time = _top["time_hours"].values
        q_pack_cumul = np.concatenate([
            [0.0],
            cumulative_trapezoid(np.abs(_top["current"].values), q_pack_time),
        ]) / max(n_parallel, 1)
```

- [ ] **Step 1.8: Interpolate per-channel Q axis in the dV/dQ loop**

In the dV/dQ per-channel loop (around line 240), change:

```python
        else:
            dv_metrics[ch] = _compute_dvdq_peak_sum(
                ch_charge["voltage"].values,
                smooth_window=params.dv_smooth_window,
                peak_prominence_pct=params.peak_prominence_pct,
            )
```

to:

```python
        else:
            q_ch: np.ndarray | None = None
            if q_pack_time is not None:
                q_ch = np.interp(
                    ch_charge["time_hours"].values, q_pack_time, q_pack_cumul
                )
            dv_metrics[ch] = _compute_dvdq_peak_sum(
                ch_charge["voltage"].values,
                smooth_window=params.dv_smooth_window,
                peak_prominence_pct=params.peak_prominence_pct,
                q_axis=q_ch,
            )
```

- [ ] **Step 1.9: Add the ΔT noise guard**

In the late ΔT computation block (around line 337), change:

```python
            dT_late_metrics[ch] = T_late_mean - T_mid_mean
```

to:

```python
            dt_late = T_late_mean - T_mid_mean
            dT_late_metrics[ch] = dt_late if abs(dt_late) >= 0.3 else np.nan
```

- [ ] **Step 1.10: Run all li-plating tests — confirm all pass**

```bash
.venv/bin/pytest tests/test_li_plating.py -v
```

Expected: all tests PASS (4 new + all pre-existing).

- [ ] **Step 1.11: Commit**

```bash
git add src/stress_screen/analysis/li_plating.py tests/test_li_plating.py
git commit -m "fix(li_plating): Q-domain dV/dQ, T threshold 20°C, ΔT noise guard"
```

---

### Task 2: Rest method fixes — M5 Arrhenius, M6 rank slope

**Files:**
- Modify: `src/stress_screen/analysis/rest.py`
- Modify: `tests/test_rest_methods.py`

- [ ] **Step 2.1: Write two failing tests**

Append to `tests/test_rest_methods.py`:

```python
def test_m5_arrhenius_vs_linear():
    """M5 Arrhenius-corrected k at 35°C must differ from the old linear approx by >1%."""
    rng = np.random.default_rng(42)
    rest_df = _make_rest_cell_df(n_channels=8, rng=rng)
    rest_df["temperature"] = 35.0
    topo = derive_topology(8, 1)
    results = run_rest_analysis(rest_df, topo)

    checked = False
    for ch, mrs in results.items():
        m1 = next((mr for mr in mrs if mr.method_name == "M1_ocv_k"), None)
        m5 = next((mr for mr in mrs if mr.method_name == "M5_temp_k"), None)
        if m1 is None or m5 is None:
            continue
        k_raw = m1.metadata.get("k", float("nan"))
        k_corr = m5.metadata.get("k_corrected", float("nan"))
        if np.isnan(k_raw) or np.isnan(k_corr) or k_raw < 1e-10:
            continue
        k_linear = k_raw / (1.0 + 0.02 * (35.0 - 25.0))  # old formula: k / 1.2
        assert abs(k_corr - k_linear) / k_linear > 0.01, (
            f"ch{ch}: Arrhenius {k_corr:.8f} vs linear {k_linear:.8f} — "
            f"difference {100 * abs(k_corr - k_linear) / k_linear:.2f}% must be >1%"
        )
        checked = True
        break

    assert checked, "No channel produced valid M1+M5 data — check _make_rest_cell_df or min_points"


def test_m6_slope_penalises_trending_cell():
    """Channel with declining rank but frac_bot20=0 gets higher M6 z after slope fix."""
    rng = np.random.default_rng(0)
    n_channels = 8
    t = np.linspace(0, 10, 600)
    rows = []
    for ch in range(n_channels):
        if ch == 0:
            # starts highest (~87th pct), drifts toward 3rd rank — never below 20th pct
            V = 3.450 + 0.050 * np.exp(-t / 2.0) - 0.004 * t + rng.normal(0, 1e-4, len(t))
        elif ch in (1, 2):
            # permanently lowest (rank 1 and 2 → always below 20th pct with 8 channels)
            V = 3.395 + (ch - 1) * 0.002 + 0.050 * np.exp(-t / 2.0) + rng.normal(0, 1e-4, len(t))
        else:
            # stable middle channels, slope ≈ 0
            V = 3.420 + (ch - 3) * 0.003 + 0.050 * np.exp(-t / 2.0) + rng.normal(0, 1e-4, len(t))
        T = 25.0 + rng.normal(0, 0.05, len(t))
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest_df = pd.DataFrame(rows)
    topo = derive_topology(n_channels, 1)
    results = run_rest_analysis(rest_df, topo)

    m6_z_ch0 = next(mr.z_score for mr in results[0] if mr.method_name == "M6_rank")
    # Only compare against stable middle channels (ch3–ch7); they have near-zero slope
    m6_z_stable = [
        next(mr.z_score for mr in results[ch] if mr.method_name == "M6_rank")
        for ch in range(3, 8)
    ]
    assert m6_z_ch0 > float(np.nanmedian(m6_z_stable)) + 0.5, (
        f"Trending ch0 M6 z={m6_z_ch0:.3f} should be > stable median "
        f"{np.nanmedian(m6_z_stable):.3f} + 0.5"
    )
```

- [ ] **Step 2.2: Run tests — confirm both fail**

```bash
.venv/bin/pytest tests/test_rest_methods.py::test_m5_arrhenius_vs_linear \
  tests/test_rest_methods.py::test_m6_slope_penalises_trending_cell -v
```

Expected:
- `test_m5_arrhenius_vs_linear` — FAIL: k_corr equals k_linear (old linear formula still in use; difference is 0%)
- `test_m6_slope_penalises_trending_cell` — FAIL: m6_z_ch0 ≈ 0 (frac_bot20=0 for ch0; slope not used)

- [ ] **Step 2.3: Add two new fields to `RestParams`**

In `src/stress_screen/analysis/rest.py`, add after the `k_max` field (after line 62):

```python
    arrhenius_ea_ev: float = 0.5
    """Activation energy (eV) for Arrhenius self-discharge temperature correction (LFP default)."""

    m6_slope_weight: float = 0.5
    """Weight applied to the rank-slope penalty term in the M6 composite score."""
```

- [ ] **Step 2.4: Replace the M5 linear Arrhenius with the proper formula**

In the M5 block (around lines 348–356), replace:

```python
        if not np.isnan(T_mean):
            # Arrhenius approximation: k_corrected = k / (1 + 0.02*(T-25))
            denom = 1.0 + 0.02 * (T_mean - 25.0)
            if abs(denom) > 1e-6:
                m5_k_corr[ch] = k_raw / denom
            else:
                m5_k_corr[ch] = k_raw
        else:
            # No temperature data — fall back to raw k
            m5_k_corr[ch] = k_raw
```

with:

```python
        if not np.isnan(T_mean):
            T_K = T_mean + 273.15
            T_ref_K = 298.15
            Ea_J = params.arrhenius_ea_ev * 96_485.0  # eV → J/mol
            correction = np.exp(-Ea_J / 8.314 * (1.0 / T_ref_K - 1.0 / T_K))
            m5_k_corr[ch] = k_raw * correction
        else:
            m5_k_corr[ch] = k_raw
```

- [ ] **Step 2.5: Add `m6_t_span` tracking to the M6 per-channel loop**

Before the `if not pivot_v.empty:` block in the M6 section, add:

```python
    m6_t_span: dict[int, float] = {}
```

Inside the per-channel M6 loop, after computing `slope_r`, add:

```python
                t_span_ch = float(t_r[-1] - t_r[0]) if len(t_r) >= 2 else 0.0
                m6_t_span[ch] = t_span_ch
```

In the else branch (insufficient data or channel absent), add:

```python
                m6_t_span[ch] = 0.0
```

- [ ] **Step 2.6: Replace the M6 robust-z scoring with the composite score**

Replace the existing M6 z-score block (around lines 400–402):

```python
    frac_arr = np.array([m6_frac_bot20[ch] for ch in channels])
    frac_z_arr = robust_z(frac_arr)
    m6_z: dict[int, float] = {ch: float(frac_z_arr[i]) for i, ch in enumerate(channels)}
```

with:

```python
    m6_score: dict[int, float] = {}
    for ch in channels:
        fb = m6_frac_bot20.get(ch, np.nan)
        if np.isnan(fb):
            m6_score[ch] = np.nan
        else:
            slope = m6_rank_slope.get(ch, 0.0)
            t_span = m6_t_span.get(ch, 0.0)
            m6_score[ch] = fb + params.m6_slope_weight * max(0.0, -slope * t_span)

    score_arr = np.array([m6_score[ch] for ch in channels])
    score_z_arr = robust_z(score_arr)
    m6_z: dict[int, float] = {ch: float(score_z_arr[i]) for i, ch in enumerate(channels)}
```

- [ ] **Step 2.7: Run all rest-method tests — confirm all pass**

```bash
.venv/bin/pytest tests/test_rest_methods.py -v
```

Expected: all tests PASS.

- [ ] **Step 2.8: Commit**

```bash
git add src/stress_screen/analysis/rest.py tests/test_rest_methods.py
git commit -m "fix(rest): proper Arrhenius M5 correction, M6 rank-slope composite score"
```

---

### Task 3: ISC detection module

**Files:**
- Create: `src/stress_screen/analysis/short_circuit.py`
- Modify: `src/stress_screen/analysis/__init__.py`
- Create: `tests/test_short_circuit.py`

- [ ] **Step 3.1: Write five failing tests**

Create `tests/test_short_circuit.py`:

```python
import numpy as np
import pandas as pd
from stress_screen.models import MethodResult
from stress_screen.analysis.short_circuit import run_isc_analysis, ShortCircuitParams


def _make_isc_rest_df(n_channels=8, n_points=400):
    """Stable OCV rest data, 8 hours, all channels normal."""
    rows = []
    t = np.linspace(0, 8.0, n_points)
    for ch in range(n_channels):
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - 0.0001 * t
        T = 25.0 * np.ones(n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)


def _make_rest_results_with_k(n_channels=8, k_values=None):
    """Minimal rest_results dict — M1 at index 0 with only a 'k' metadata key."""
    k_values = k_values or {ch: 0.0001 for ch in range(n_channels)}
    results = {}
    for ch in range(n_channels):
        k = k_values.get(ch, 0.0001)
        results[ch] = [
            MethodResult(
                method_name="M1_ocv_k",
                z_score=0.0,
                verdict="NORMAL",
                metadata={"k": k, "V_ocv": 3.4, "tau": 1.0},
            )
        ]
    return results


def _make_charge_df_for_isc(n_channels=8, n_points=200):
    """Charge data for S3 area tests. ch4 has half the normal voltage rise."""
    rows = []
    for ch in range(n_channels):
        t = np.linspace(0, 2.0, n_points)
        if ch == 4:
            V = 3.2 + 0.2 * (t / t.max())  # half rise → smaller dV/dQ area
        else:
            V = 3.2 + 0.4 * (t / t.max())
        T = 25.0 * np.ones(n_points)
        for i in range(n_points):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    return pd.DataFrame(rows)


def test_isc_returns_all_channels():
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k()
    charge_df = pd.DataFrame(columns=["time_hours", "channel_index", "voltage", "temperature"])
    results = run_isc_analysis(rest_df, rest_results, charge_df)
    assert len(results) == 8
    assert all(r.method_name == "isc" for r in results.values())


def test_s1_high_k_cell_flagged():
    """Channel with 10× fleet-median k fires HIGH/ELEVATED on S1 and has higher S1 z."""
    k_values = {ch: 0.0001 for ch in range(8)}
    k_values[3] = 0.001  # 10× the fleet median
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k(k_values=k_values)
    charge_df = pd.DataFrame(columns=["time_hours", "channel_index", "voltage", "temperature"])
    results = run_isc_analysis(rest_df, rest_results, charge_df)
    assert results[3].metadata["s1_excess_k_z"] > results[0].metadata["s1_excess_k_z"], (
        "High-k channel should have higher S1 z than normal channel"
    )
    assert results[3].verdict in ("HIGH", "ELEVATED"), (
        f"High-k channel verdict expected HIGH/ELEVATED, got {results[3].verdict}"
    )


def test_s2_warming_cell_flagged():
    """Channel with rising temperature during rest gets elevated S2 z."""
    n_channels = 8
    rows = []
    t = np.linspace(0, 8.0, 400)
    for ch in range(n_channels):
        V = 3.4 + 0.05 * np.exp(-t / 2.0) - 0.0001 * t
        T = 25.0 + 0.05 * t if ch == 2 else 26.0 - 0.02 * t
        for i in range(len(t)):
            rows.append({"time_hours": t[i], "channel_index": ch,
                         "voltage": V[i], "temperature": T[i]})
    rest_df = pd.DataFrame(rows)
    rest_results = _make_rest_results_with_k(n_channels=n_channels)
    charge_df = pd.DataFrame(columns=["time_hours", "channel_index", "voltage", "temperature"])
    results = run_isc_analysis(rest_df, rest_results, charge_df)
    s2_z_ch2 = results[2].metadata["s2_dT_dt_z"]
    s2_z_others = [results[ch].metadata["s2_dT_dt_z"] for ch in range(n_channels) if ch != 2]
    assert s2_z_ch2 > float(np.nanmedian(s2_z_others)), (
        f"Warming ch2 S2 z={s2_z_ch2:.3f} should be above peers median "
        f"{np.nanmedian(s2_z_others):.3f}"
    )


def test_s3_area_deficit_cell_flagged():
    """Channel with half the dV/dQ area gets higher (inverted) S3 z than peers."""
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k()
    charge_df = _make_charge_df_for_isc()
    top_charge_df = pd.DataFrame({
        "time_hours": np.linspace(0, 2.0, 200),
        "current": np.full(200, 5.0),
    })
    results = run_isc_analysis(rest_df, rest_results, charge_df,
                               top_charge_df=top_charge_df)
    s3_z_ch4 = results[4].metadata["s3_area_deficit_z"]
    s3_z_others = [results[ch].metadata["s3_area_deficit_z"]
                   for ch in range(8) if ch != 4]
    assert s3_z_ch4 > float(np.nanmedian(s3_z_others)), (
        f"Area-deficit ch4 S3 z={s3_z_ch4:.3f} should be above peers median "
        f"{np.nanmedian(s3_z_others):.3f}"
    )


def test_s3_fallback_no_top_df():
    """Absent top_charge_df: all S3 area values are nan, no exception raised."""
    rest_df = _make_isc_rest_df()
    rest_results = _make_rest_results_with_k()
    charge_df = _make_charge_df_for_isc()
    results = run_isc_analysis(rest_df, rest_results, charge_df)  # no top_charge_df
    assert len(results) == 8
    for ch, mr in results.items():
        assert np.isnan(mr.metadata["s3_dvdq_area"]), (
            f"ch{ch} s3_dvdq_area should be nan when top_charge_df absent, "
            f"got {mr.metadata['s3_dvdq_area']}"
        )
```

- [ ] **Step 3.2: Run tests — confirm all 5 fail with ImportError**

```bash
.venv/bin/pytest tests/test_short_circuit.py -v
```

Expected: all 5 FAIL with `ModuleNotFoundError: No module named 'stress_screen.analysis.short_circuit'`.

- [ ] **Step 3.3: Create `src/stress_screen/analysis/short_circuit.py`**

```python
"""
analysis/short_circuit.py — Soft/incipient internal short-circuit (ISC) detection.

Three methods:
  S1: Excess self-discharge rate (k well above fleet median+MAD threshold)
  S2: Thermal anomaly during rest (positive dT/dt vs cooling peers)
  S3: Charge-acceptance shape deficit (reduced dV/dQ area in Q domain)

Public API
----------
run_isc_analysis(rest_cell_df, rest_results, charge_cell_df, params,
                 top_charge_df, n_parallel) -> dict[int, MethodResult]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats as _stats

from stress_screen.models import MethodResult
from stress_screen.analysis.util import robust_z


@dataclass
class ShortCircuitParams:
    """Tunable parameters for the ISC detection analysis."""

    z_thresh: float = 2.0
    isc_k_sigma: float = 3.0
    """Multiplier above (median + isc_k_sigma×MAD) to flag absolute self-discharge excess."""
    settling_h: float = 2.0
    min_points: int = 60
    dv_smooth_window: int = 11
    peak_prominence_pct: float = 0.05


def _verdict(z: float, z_thresh: float) -> str:
    if np.isnan(z):
        return "NORMAL"
    if z >= z_thresh:
        return "HIGH"
    if z >= 1.0:
        return "ELEVATED"
    return "NORMAL"


def run_isc_analysis(
    rest_cell_df: pd.DataFrame,
    rest_results: dict[int, list[MethodResult]],
    charge_cell_df: pd.DataFrame,
    params: ShortCircuitParams | None = None,
    top_charge_df: pd.DataFrame | None = None,
    n_parallel: int = 1,
) -> dict[int, MethodResult]:
    """Detect soft/incipient internal short-circuit (ISC) signatures.

    Parameters
    ----------
    rest_cell_df:
        Long-format rest-phase data with columns ``time_hours``, ``channel_index``,
        ``voltage``, ``temperature``.
    rest_results:
        Output of ``run_rest_analysis``. M1 (index 0) metadata must contain ``"k"``.
    charge_cell_df:
        Long-format charge-phase data (same schema). Used for S3.
    params:
        Tunable parameters; defaults used if None.
    top_charge_df:
        Pack-level charge DataFrame with ``time_hours`` and ``current``. S3 returns
        nan for all channels when absent.
    n_parallel:
        Parallel cell count per group (topology.parallel). Scales the Q axis.

    Returns
    -------
    dict mapping ``channel_index`` to a ``MethodResult`` with ``method_name="isc"``.
    """
    if params is None:
        params = ShortCircuitParams()

    channels = sorted(rest_cell_df["channel_index"].unique())
    t0 = rest_cell_df["time_hours"].min()

    # ------------------------------------------------------------------ #
    # Pre-process: settled per-channel data                               #
    # ------------------------------------------------------------------ #
    chan_data: dict[int, dict] = {}
    for ch in channels:
        cd = (
            rest_cell_df[rest_cell_df["channel_index"] == ch]
            .sort_values("time_hours")
            .copy()
        )
        cd["_t_rel"] = cd["time_hours"] - t0
        settled = cd[cd["_t_rel"] >= params.settling_h]
        has_temp = "temperature" in settled.columns
        chan_data[ch] = {
            "t_set": settled["_t_rel"].values,
            "temp_set": settled["temperature"].values if has_temp else np.array([]),
            "n_set": len(settled),
        }

    # ------------------------------------------------------------------ #
    # S1: Excess self-discharge rate                                      #
    # ------------------------------------------------------------------ #
    m1_k: dict[int, float] = {}
    for ch in channels:
        if ch in rest_results and rest_results[ch]:
            m1_k[ch] = float(rest_results[ch][0].metadata.get("k", np.nan))
        else:
            m1_k[ch] = np.nan

    k_arr = np.array([m1_k.get(ch, np.nan) for ch in channels])
    valid_k = k_arr[~np.isnan(k_arr)]

    s1_excess: dict[int, float] = {}
    if len(valid_k) >= 3:
        med_k = float(np.median(valid_k))
        mad_k = float(np.median(np.abs(valid_k - med_k)))
        threshold = med_k + params.isc_k_sigma * mad_k
        for i, ch in enumerate(channels):
            s1_excess[ch] = (
                max(0.0, float(k_arr[i]) - threshold)
                if not np.isnan(k_arr[i])
                else np.nan
            )
    else:
        for ch in channels:
            s1_excess[ch] = np.nan

    s1_arr = np.array([s1_excess[ch] for ch in channels])
    s1_z_arr = robust_z(s1_arr)
    s1_z: dict[int, float] = {ch: float(s1_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # S2: Thermal anomaly during rest (positive dT/dt slope)             #
    # ------------------------------------------------------------------ #
    s2_slope: dict[int, float] = {}
    for ch in channels:
        d = chan_data[ch]
        if d["n_set"] < params.min_points or len(d["temp_set"]) == 0:
            s2_slope[ch] = np.nan
            continue
        temp = d["temp_set"]
        t = d["t_set"]
        valid = ~np.isnan(temp)
        if valid.sum() < 5 or np.nanstd(temp[valid]) < 1e-6:
            s2_slope[ch] = np.nan
            continue
        slope, *_ = _stats.linregress(t[valid], temp[valid])
        s2_slope[ch] = float(slope)

    s2_arr = np.array([s2_slope[ch] for ch in channels])
    s2_z_arr = robust_z(s2_arr)
    s2_z: dict[int, float] = {ch: float(s2_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # S3: Charge-acceptance shape (dV/dQ area deficit)                   #
    # ------------------------------------------------------------------ #
    s3_area: dict[int, float] = {}

    if top_charge_df is not None and len(top_charge_df) >= 2 and not charge_cell_df.empty:
        from scipy.integrate import cumulative_trapezoid
        _top = top_charge_df.sort_values("time_hours")
        t_pack = _top["time_hours"].values
        i_pack = np.abs(_top["current"].values)
        Q_pack = np.concatenate([
            [0.0], cumulative_trapezoid(i_pack, t_pack)
        ]) / max(n_parallel, 1)

        for ch in channels:
            ch_charge = (
                charge_cell_df[charge_cell_df["channel_index"] == ch]
                .sort_values("time_hours")
            )
            if len(ch_charge) < params.min_points:
                s3_area[ch] = np.nan
                continue
            q_ch = np.interp(ch_charge["time_hours"].values, t_pack, Q_pack)
            voltage = ch_charge["voltage"].values
            dv_dq = np.gradient(voltage, q_ch)
            if len(dv_dq) >= params.dv_smooth_window:
                try:
                    dv_dq = signal.savgol_filter(dv_dq, params.dv_smooth_window, 2)
                except Exception:
                    pass
            s3_area[ch] = float(np.trapz(np.abs(dv_dq), q_ch))
    else:
        for ch in channels:
            s3_area[ch] = np.nan

    # Invert: low area = high z (area deficit is the suspicious direction)
    s3_arr = np.array([s3_area[ch] for ch in channels])
    s3_z_arr = -robust_z(s3_arr)
    s3_z: dict[int, float] = {ch: float(s3_z_arr[i]) for i, ch in enumerate(channels)}

    # ------------------------------------------------------------------ #
    # Assemble MethodResult per channel                                   #
    # ------------------------------------------------------------------ #
    results: dict[int, MethodResult] = {}
    for i, ch in enumerate(channels):
        z1, z2, z3 = s1_z[ch], s2_z[ch], s3_z[ch]
        valid_zs = [s for s in (z1, z2, z3) if not np.isnan(s)]
        z = float(np.mean(valid_zs)) if valid_zs else float("nan")
        results[ch] = MethodResult(
            method_name="isc",
            z_score=z,
            verdict=_verdict(z, params.z_thresh),
            metadata={
                "s1_excess_k_z": z1,
                "s2_dT_dt_z": z2,
                "s3_area_deficit_z": z3,
                "s1_excess_k": float(s1_excess[ch]),
                "s2_dT_dt_slope": float(s2_slope[ch]),
                "s3_dvdq_area": float(s3_area[ch]),
            },
        )
    return results
```

- [ ] **Step 3.4: Update `src/stress_screen/analysis/__init__.py`**

Replace the file content with:

```python
from stress_screen.analysis.short_circuit import run_isc_analysis, ShortCircuitParams

__all__ = ["run_isc_analysis", "ShortCircuitParams"]
```

- [ ] **Step 3.5: Run ISC tests — confirm all 5 pass**

```bash
.venv/bin/pytest tests/test_short_circuit.py::test_isc_returns_all_channels \
  tests/test_short_circuit.py::test_s1_high_k_cell_flagged \
  tests/test_short_circuit.py::test_s2_warming_cell_flagged \
  tests/test_short_circuit.py::test_s3_area_deficit_cell_flagged \
  tests/test_short_circuit.py::test_s3_fallback_no_top_df -v
```

Expected: all 5 PASS.

- [ ] **Step 3.6: Commit**

```bash
git add src/stress_screen/analysis/short_circuit.py \
        src/stress_screen/analysis/__init__.py \
        tests/test_short_circuit.py
git commit -m "feat(isc): add soft internal short-circuit detection module (S1-S3)"
```

---

### Task 4: Aggregate update + integration test

**Files:**
- Modify: `src/stress_screen/analysis/aggregate.py`
- Modify: `tests/test_short_circuit.py`

- [ ] **Step 4.1: Write failing integration test**

Append to `tests/test_short_circuit.py`:

```python
def test_isc_aggregate_integration():
    """Full aggregate with ISC produces CellVerdict with 8 method_results."""
    import warnings
    from stress_screen.analysis.rest import run_rest_analysis
    from stress_screen.analysis.li_plating import run_li_plating_analysis
    from stress_screen.analysis.aggregate import aggregate
    from stress_screen.topology import derive_topology

    n_channels = 8
    rest_df = _make_isc_rest_df()
    charge_df = _make_charge_df_for_isc()
    topo = derive_topology(n_channels, 1)

    full_rest_results = run_rest_analysis(rest_df, topo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        li_results = run_li_plating_analysis(charge_df, rest_df)
    isc_results = run_isc_analysis(rest_df, full_rest_results, charge_df)

    module_verdicts = aggregate(
        full_rest_results, li_results, topo, isc_results=isc_results
    )

    for mv in module_verdicts:
        for cv in mv.all_cells:
            assert len(cv.method_results) == 8, (
                f"{cv.label}: expected 8 method_results "
                f"(6 rest + 1 li_plating + 1 isc), got {len(cv.method_results)}"
            )
```

- [ ] **Step 4.2: Run test — confirm it fails**

```bash
.venv/bin/pytest tests/test_short_circuit.py::test_isc_aggregate_integration -v
```

Expected: FAIL with `TypeError: aggregate() got an unexpected keyword argument 'isc_results'`.

- [ ] **Step 4.3: Update `aggregate` signature**

In `src/stress_screen/analysis/aggregate.py`, change the function signature from:

```python
def aggregate(
    rest_results: dict[int, list[MethodResult]],
    li_plating_results: dict[int, MethodResult],
    topology: PackTopology,
    z_thresh: float = 2.0,
) -> list[ModuleVerdict]:
```

to:

```python
def aggregate(
    rest_results: dict[int, list[MethodResult]],
    li_plating_results: dict[int, MethodResult],
    topology: PackTopology,
    isc_results: dict[int, MethodResult] | None = None,
    z_thresh: float = 2.0,
) -> list[ModuleVerdict]:
```

- [ ] **Step 4.4: Append ISC z-score and method_result in the per-channel loop**

In the per-channel loop, change:

```python
        all_z = [mr.z_score for mr in rest_results[ch]] + [li_plating_results[ch].z_score]
```

to:

```python
        all_z = [mr.z_score for mr in rest_results[ch]] + [li_plating_results[ch].z_score]
        if isc_results and ch in isc_results:
            all_z.append(isc_results[ch].z_score)
```

And change:

```python
        cell_verdicts[ch] = CellVerdict(
            channel_index=ch,
            module_id=topology.module_for_channel(ch),
            group_in_module=topology.group_index_in_module(ch),
            composite_z=composite_z,
            n_methods_high=n_high,
            verdict=verdict,
            method_results=rest_results[ch] + [li_plating_results[ch]],
        )
```

to:

```python
        method_results_list = rest_results[ch] + [li_plating_results[ch]]
        if isc_results and ch in isc_results:
            method_results_list = method_results_list + [isc_results[ch]]
        cell_verdicts[ch] = CellVerdict(
            channel_index=ch,
            module_id=topology.module_for_channel(ch),
            group_in_module=topology.group_index_in_module(ch),
            composite_z=composite_z,
            n_methods_high=n_high,
            verdict=verdict,
            method_results=method_results_list,
        )
```

- [ ] **Step 4.5: Run all test_short_circuit tests — confirm all 6 pass**

```bash
.venv/bin/pytest tests/test_short_circuit.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 4.6: Run the full test suite — confirm no regressions**

```bash
.venv/bin/pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4.7: Commit**

```bash
git add src/stress_screen/analysis/aggregate.py tests/test_short_circuit.py
git commit -m "feat(aggregate): add optional isc_results parameter (8-method composite)"
```

---

### Task 5: CLI wiring

**Files:**
- Modify: `src/stress_screen/cli.py`

- [ ] **Step 5.1: Slice `charge_top_df` alongside `charge_cell_df`**

In `_run`, find the existing charge-slice block (around line 279):

```python
    if charge_segs:
        first_charge = charge_segs[0]
        charge_cell_df = cell_df[
            (cell_df["time_hours"] >= top_df.iloc[first_charge.start_row]["time_hours"])
            & (cell_df["time_hours"] <= top_df.iloc[first_charge.end_row]["time_hours"])
        ].copy()
    else:
        charge_cell_df = cell_df.iloc[0:0].copy()  # empty, same schema
```

Replace with:

```python
    if charge_segs:
        first_charge = charge_segs[0]
        charge_time_min = float(top_df.iloc[first_charge.start_row]["time_hours"])
        charge_time_max = float(top_df.iloc[first_charge.end_row]["time_hours"])
        charge_cell_df = cell_df[
            (cell_df["time_hours"] >= charge_time_min)
            & (cell_df["time_hours"] <= charge_time_max)
        ].copy()
        charge_top_df = top_df[
            (top_df["time_hours"] >= charge_time_min)
            & (top_df["time_hours"] <= charge_time_max)
        ].copy()
    else:
        charge_cell_df = cell_df.iloc[0:0].copy()
        charge_top_df = top_df.iloc[0:0].copy()
```

- [ ] **Step 5.2: Pass `top_charge_df` and `n_parallel` to `run_li_plating_analysis`**

Find the li-plating call (around line 304):

```python
    li_results = run_li_plating_analysis(charge_cell_df, li_rest_cell_df)
```

Replace with:

```python
    li_results = run_li_plating_analysis(
        charge_cell_df,
        li_rest_cell_df,
        top_charge_df=charge_top_df,
        n_parallel=topology.parallel,
    )
```

- [ ] **Step 5.3: Add ISC analysis step (step 6b)**

After step 6 (li-plating) and before step 7 (aggregate), insert:

```python
    # ------------------------------------------------------------------
    # 6b. ISC analysis
    # ------------------------------------------------------------------
    from stress_screen.analysis.short_circuit import run_isc_analysis
    prog.stage(f"Running ISC analysis on {n_active} channels...")
    isc_results = run_isc_analysis(
        rest_cell_df,
        rest_results,
        charge_cell_df,
        top_charge_df=charge_top_df,
        n_parallel=topology.parallel,
    )
```

- [ ] **Step 5.4: Pass `isc_results` to `aggregate`**

Find the aggregate call (around line 310):

```python
    module_verdicts = aggregate(rest_results, li_results, topology)
```

Replace with:

```python
    module_verdicts = aggregate(rest_results, li_results, topology, isc_results=isc_results)
```

- [ ] **Step 5.5: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests PASS. If `test_e2e_cli` is exercised (a `.csv` file is present in the project root), verify that the output still contains one `M<n>: OK|NOK` line per module.

- [ ] **Step 5.6: Commit**

```bash
git add src/stress_screen/cli.py
git commit -m "feat(cli): pass Q-domain data to li_plating; wire ISC analysis step"
```

---

## Spec-coverage self-review

| Spec requirement | Task/step that implements it |
|---|---|
| Fix 1 — dV/dQ Q-domain via `top_charge_df` + `n_parallel` | Task 1, steps 1.3–1.8 |
| Fix 1 — backward-compat fallback warning | Task 1, step 1.7 |
| Fix 2 — T_plating_threshold_c 15→20 °C | Task 1, step 1.5 |
| Fix 3 — ΔT noise guard 0.3 °C | Task 1, step 1.9 |
| Fix 4 — Arrhenius M5 with `arrhenius_ea_ev` param | Task 2, steps 2.3–2.4 |
| Fix 5 — M6 composite score with `m6_slope_weight` | Task 2, steps 2.5–2.6 |
| ISC S1 — fleet-median+MAD excess k | Task 3, step 3.3 |
| ISC S2 — dT/dt linear regression | Task 3, step 3.3 |
| ISC S3 — Q-domain dV/dQ area deficit (inverted z) | Task 3, step 3.3 |
| ISC S3 graceful fallback when `top_charge_df` absent | Task 3, step 3.3 |
| `aggregate` optional `isc_results` → 8 method_results | Task 4, steps 4.3–4.4 |
| CLI slices `charge_top_df`; wires ISC step | Task 5, steps 5.1–5.4 |
| All 12 new tests from spec | Tasks 1–4 |
