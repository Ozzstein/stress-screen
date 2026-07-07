"""End-to-end CLI tests on the synthetic fixture (and, opt-in, real CSVs).

Set ``STRESS_SCREEN_E2E_REAL_CSV=1`` to additionally run the CLI against any
real tester CSVs present in the project root (slow; skipped by default and in
CI, where no real CSVs are committed).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.synth import make_synthetic_csv


def _cli_env():
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(csv: Path, *extra: str):
    return subprocess.run(
        [sys.executable, "-m", "stress_screen", str(csv), *extra],
        capture_output=True,
        text=True,
        env=_cli_env(),
    )


def _assert_verdict_lines(stdout: str, module_count: int, name: str):
    for i in range(1, module_count + 1):
        assert re.search(rf"^M{i}:\s+(OK( - Marginal)?|NOK)", stdout, re.MULTILINE), \
            f"Missing M{i} verdict line in output for {name}"


def test_e2e_cli_synthetic(tmp_path):
    """Full CLI on the synthetic pack: verdicts, exit code, JSON sidecar."""
    csv = make_synthetic_csv(tmp_path, leaky_channels={4: 0.8})
    result = _run_cli(csv, "--no-html", "--no-pdf")
    # channel 4 is injected NOK → exit code 1
    assert result.returncode == 1, f"stderr: {result.stderr}"
    _assert_verdict_lines(result.stdout, 2, csv.name)
    assert (csv.parent / f"{csv.stem}_result.json").exists()


def test_e2e_cli_healthy_pack_exits_zero(tmp_path):
    csv = make_synthetic_csv(tmp_path, leaky_channels=None)
    result = _run_cli(csv, "--no-html", "--no-pdf", "--no-json")
    assert result.returncode == 0, (
        f"Healthy synthetic pack should be all-OK.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    _assert_verdict_lines(result.stdout, 2, csv.name)


def test_e2e_cli_short_rest_rejected(tmp_path):
    """A file without a >= 48 h rest must exit 2 with the protocol error."""
    csv = make_synthetic_csv(tmp_path, rest_hours=20.0)
    result = _run_cli(csv, "--no-html", "--no-pdf", "--no-json")
    assert result.returncode == 2
    assert "Test invalidated" in result.stderr


@pytest.mark.skipif(
    os.environ.get("STRESS_SCREEN_E2E_REAL_CSV") != "1",
    reason="real-CSV e2e is opt-in (STRESS_SCREEN_E2E_REAL_CSV=1)",
)
def test_e2e_cli_real_csvs():
    candidates = sorted(Path(".").glob("*.csv"))
    if not candidates:
        pytest.skip("No CSV file found in project root")
    ran_any = False
    for csv in candidates:
        result = _run_cli(csv, "--no-html", "--no-pdf", "--no-json")
        if result.returncode == 2:
            continue  # protocol-invalid CSV (e.g. < 48 h rest)
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode} on {csv.name}: {result.stderr}"
        )
        module_count = int(re.search(r"_M(\d+)\b", csv.name).group(1))
        _assert_verdict_lines(result.stdout, module_count, csv.name)
        ran_any = True
    if not ran_any:
        pytest.skip("All CSV files in project root were protocol-invalid")
