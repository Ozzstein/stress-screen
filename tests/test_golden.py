"""
Golden-output regression test.

Runs the full CLI pipeline on the deterministic synthetic fixture and compares
the JSON result against a committed golden file. This is the regression gate
for the loader rewrite and any statistics refactor: behavior-preserving
changes must reproduce the golden verdicts and z-scores exactly (within float
tolerance).

How tight "within float tolerance" is depends on where the golden file was
generated. Regenerating it records the platform (``<system>-<machine>``, for
example ``Linux-x86_64``) under a top-level ``_baseline`` key. Floats are
compared near-bit-exactly only on that platform; on any other platform scipy
curve fits differ slightly, so the test falls back to a loose tolerance and
emits a warning. ``STRESS_SCREEN_GOLDEN_LOOSE=1`` forces the loose tolerance
(CI sets it); a golden file without a ``_baseline`` marker is compared
strictly. Verdicts, labels and structure are always compared exactly.

To intentionally re-baseline after a deliberate behavior change (this makes
the current machine the baseline platform)::

    STRESS_SCREEN_REGEN_GOLDEN=1 PYTHONPATH=src uv run pytest tests/test_golden.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

from tests.synth import make_synthetic_csv

GOLDEN_PATH = Path(__file__).parent / "golden" / "synth_M2_result.json"

#: Subtrees compared against golden. Volatile sections (generated_at,
#: tool_version, absolute paths) and the config section (which grows as more
#: parameters become configurable) are excluded on purpose.
COMPARED_KEYS = ("topology", "segments", "verdict", "modules")

#: Float tolerance is tiered, as (rel_tol, abs_tol) for math.isclose. Strings
#: (verdicts, labels, phases) are always compared exactly. Which tier applies
#: is decided per run by _select_tolerance, from the platform recorded in the
#: golden file:
#:   - STRICT_TOL: near-bit-exact — the regression gate on the platform that
#:     generated the golden file (guards loader/statistics refactors).
#:   - LOOSE_TOL: scipy curve fits differ slightly across platforms/BLAS
#:     builds — up to ~10% on ill-conditioned fits of NORMAL cells — so on any
#:     other platform (and in CI, via STRESS_SCREEN_GOLDEN_LOOSE=1) structure
#:     and verdicts are checked exactly but floats only coarsely.
STRICT_TOL = (1e-5, 1e-8)
LOOSE_TOL = (0.25, 0.05)


def _platform_tag() -> str:
    """Platform identifier recorded in, and compared with, the golden file."""
    return f"{platform.system()}-{platform.machine()}"


def _select_tolerance(
    baseline_platform: str | None, current_platform: str, loose_env: str | None
) -> tuple[tuple[float, float], str]:
    """Return ((rel_tol, abs_tol), reason) for comparing floats to the golden.

    Pure: the caller passes the platform recorded in the golden file (None when
    it has no ``_baseline`` marker), this machine's platform tag and the raw
    STRESS_SCREEN_GOLDEN_LOOSE value. Precedence:

    1. loose_env == "1"                      -> LOOSE_TOL
    2. baseline_platform is None             -> STRICT_TOL
    3. baseline_platform != current_platform -> LOOSE_TOL
    4. otherwise                             -> STRICT_TOL

    Case 1 is the explicit override (CI sets it), case 2 a legacy golden file
    without a marker, case 3 a golden file generated on another platform.
    ``reason`` is a short human-readable string naming the deciding fact.
    """
    if loose_env == "1":
        return LOOSE_TOL, "STRESS_SCREEN_GOLDEN_LOOSE=1 is set"
    if baseline_platform is None:
        return STRICT_TOL, "golden has no _baseline marker"
    if baseline_platform != current_platform:
        return LOOSE_TOL, (
            f"golden baselined on {baseline_platform}, "
            f"running on {current_platform}"
        )
    return STRICT_TOL, (
        f"golden baselined on this platform ({current_platform})"
    )


def _run_pipeline(tmp_path: Path) -> dict:
    csv = make_synthetic_csv(tmp_path, leaky_channels={4: 0.8})
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "stress_screen", str(csv), "--no-html", "--no-pdf"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode in (0, 1), (
        f"Pipeline failed (exit {result.returncode}): {result.stderr}"
    )
    json_path = csv.parent / f"{csv.stem}_result.json"
    assert json_path.exists(), "JSON result file was not written"
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def _diff(a, b, rel_tol: float, abs_tol: float, path="$") -> list[str]:
    """Recursively diff two JSON trees; floats compared with tolerance."""
    problems: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                problems.append(f"{path}.{key}: missing in actual")
            elif key not in b:
                problems.append(f"{path}.{key}: unexpected in actual")
            else:
                problems.extend(
                    _diff(a[key], b[key], rel_tol, abs_tol, f"{path}.{key}")
                )
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(b)} != golden {len(a)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                problems.extend(_diff(x, y, rel_tol, abs_tol, f"{path}[{i}]"))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            problems.append(f"{path}: {b!r} != golden {a!r}")
    elif a != b:
        problems.append(f"{path}: {b!r} != golden {a!r}")
    return problems


def test_select_tolerance_env_override_forces_loose():
    # Precedence 1: the explicit override beats a matching platform, a foreign
    # platform and a missing marker alike.
    for baseline in ("Linux-x86_64", "Darwin-arm64", None):
        tol, reason = _select_tolerance(baseline, "Linux-x86_64", "1")
        assert tol == LOOSE_TOL, baseline
        assert "STRESS_SCREEN_GOLDEN_LOOSE=1" in reason, baseline


def test_select_tolerance_golden_without_marker_is_strict():
    # Precedence 2: a legacy golden file carries no _baseline marker.
    tol, reason = _select_tolerance(None, "Linux-x86_64", None)
    assert tol == STRICT_TOL
    assert "_baseline" in reason


def test_select_tolerance_foreign_baseline_platform_is_loose():
    # Precedence 3: the golden file was generated on another platform.
    tol, reason = _select_tolerance("Darwin-arm64", "Linux-x86_64", None)
    assert tol == LOOSE_TOL
    assert "Darwin-arm64" in reason
    assert "Linux-x86_64" in reason


def test_select_tolerance_matching_baseline_platform_is_strict():
    # Precedence 4: the golden file was generated on this very platform.
    tol, reason = _select_tolerance("Linux-x86_64", "Linux-x86_64", None)
    assert tol == STRICT_TOL
    assert "Linux-x86_64" in reason
    assert "this platform" in reason


def test_pipeline_matches_golden(tmp_path):
    actual = _run_pipeline(tmp_path)

    if os.environ.get("STRESS_SCREEN_REGEN_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        # The marker lives in the golden file only (the pipeline's own result
        # file is untouched) and is not part of COMPARED_KEYS.
        baselined = {"_baseline": {"platform": _platform_tag()}, **actual}
        with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
            json.dump(baselined, fh, indent=2)
            fh.write("\n")

    assert GOLDEN_PATH.exists(), (
        "Golden file missing — generate it with STRESS_SCREEN_REGEN_GOLDEN=1"
    )
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        golden = json.load(fh)

    baseline_platform = golden.get("_baseline", {}).get("platform")
    current_platform = _platform_tag()
    loose_env = os.environ.get("STRESS_SCREEN_GOLDEN_LOOSE")
    (rel_tol, abs_tol), reason = _select_tolerance(
        baseline_platform, current_platform, loose_env
    )
    if loose_env != "1" and (rel_tol, abs_tol) == LOOSE_TOL:
        # Loose without the explicit override means the platform differs, so
        # the strict gate is not running on this machine: say so.
        warnings.warn(
            f"golden file was baselined on {baseline_platform}, running on "
            f"{current_platform}: floats are compared loosely, so small "
            "numeric regressions are not caught on this machine. To make this "
            "machine the baseline: STRESS_SCREEN_REGEN_GOLDEN=1 uv run pytest "
            "tests/test_golden.py",
            UserWarning,
        )

    problems = []
    for key in COMPARED_KEYS:
        problems.extend(
            _diff(golden[key], actual[key], rel_tol, abs_tol, f"$.{key}")
        )
    assert not problems, (
        f"{len(problems)} deviations from golden output (first 20), using "
        f"rel_tol={rel_tol}, abs_tol={abs_tol} because {reason}:\n"
        + "\n".join(problems[:20])
    )


def test_golden_file_records_baseline_platform():
    """The committed golden file must say which platform generated it.

    Defined after test_pipeline_matches_golden on purpose: a regeneration run
    rewrites the golden there, and this check then sees the fresh marker.
    """
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        golden = json.load(fh)
    tag = golden["_baseline"]["platform"]
    assert isinstance(tag, str), tag
    system, sep, machine = tag.partition("-")
    assert sep and system and machine, (
        f"_baseline.platform must look like <system>-<machine>, got {tag!r}"
    )


def test_fixture_flags_injected_leak(tmp_path):
    """The injected 0.8 mV/h leak on channel 4 must be the top outlier."""
    actual = _run_pipeline(tmp_path)
    all_cells = [c for m in actual["modules"] for c in m["cells"]]
    worst = max(all_cells, key=lambda c: c["composite_z"] or 0.0)
    assert worst["channel_index"] == 4
    assert worst["verdict"] == "HIGH"
    assert actual["verdict"]["overall"] == "NOK"
    assert actual["verdict"]["exit_code"] == 1
