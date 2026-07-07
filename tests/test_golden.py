"""
Golden-output regression test.

Runs the full CLI pipeline on the deterministic synthetic fixture and compares
the JSON result against a committed golden file. This is the regression gate
for the loader rewrite and any statistics refactor: behavior-preserving
changes must reproduce the golden verdicts and z-scores exactly (within float
tolerance).

To intentionally re-baseline after a deliberate behavior change::

    STRESS_SCREEN_REGEN_GOLDEN=1 PYTHONPATH=src uv run pytest tests/test_golden.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from tests.synth import make_synthetic_csv

GOLDEN_PATH = Path(__file__).parent / "golden" / "synth_M2_result.json"

#: Subtrees compared against golden. Volatile sections (generated_at,
#: tool_version, absolute paths) and the config section (which grows as more
#: parameters become configurable) are excluded on purpose.
COMPARED_KEYS = ("topology", "segments", "verdict", "modules")

REL_TOL = 1e-5
ABS_TOL = 1e-8


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


def _diff(a, b, path="$") -> list[str]:
    """Recursively diff two JSON trees; floats compared with tolerance."""
    problems: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                problems.append(f"{path}.{key}: missing in actual")
            elif key not in b:
                problems.append(f"{path}.{key}: unexpected in actual")
            else:
                problems.extend(_diff(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(b)} != golden {len(a)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                problems.extend(_diff(x, y, f"{path}[{i}]"))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            problems.append(f"{path}: {b!r} != golden {a!r}")
    elif a != b:
        problems.append(f"{path}: {b!r} != golden {a!r}")
    return problems


def test_pipeline_matches_golden(tmp_path):
    actual = _run_pipeline(tmp_path)

    if os.environ.get("STRESS_SCREEN_REGEN_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
            json.dump(actual, fh, indent=2)
            fh.write("\n")

    assert GOLDEN_PATH.exists(), (
        "Golden file missing — generate it with STRESS_SCREEN_REGEN_GOLDEN=1"
    )
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        golden = json.load(fh)

    problems = []
    for key in COMPARED_KEYS:
        problems.extend(_diff(golden[key], actual[key], f"$.{key}"))
    assert not problems, (
        f"{len(problems)} deviations from golden output "
        f"(first 20):\n" + "\n".join(problems[:20])
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
