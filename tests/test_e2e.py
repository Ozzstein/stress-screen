import os
import subprocess
import sys
from pathlib import Path
import re

def test_e2e_cli():
    """Run the full CLI on the first valid sample CSV and verify output format.

    Iterates over CSV files in the project root and uses the first one that
    passes the protocol checks (>= 48 h rest segment). Files that exit with
    code 2 ("Test invalidated") are skipped — those represent legitimate
    short-rest rejections, not test failures.
    """
    import pytest
    candidates = sorted(Path(".").glob("*.csv"))
    if not candidates:
        pytest.skip("No CSV file found in project root")

    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    for csv in candidates:
        result = subprocess.run(
            [sys.executable, "-m", "stress_screen", str(csv), "--no-html", "--no-pdf"],
            capture_output=True,
            text=True,
            cwd=str(Path(".").resolve()),
            env=env,
        )
        if result.returncode == 2:
            continue  # protocol-invalid CSV (e.g. < 48 h rest) — try the next one
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode} on {csv.name}: {result.stderr}"
        )
        # Verify output contains module lines
        lines = result.stdout
        module_count = int(re.search(r"_M(\d+)\b", csv.name).group(1))
        for i in range(1, module_count + 1):
            assert re.search(rf"^M{i}:\s+(OK|MARGINAL|NOK)", lines, re.MULTILINE), \
                f"Missing M{i} verdict line in output for {csv.name}"
        return  # success

    pytest.skip("All CSV files in project root were protocol-invalid (no >= 48 h rest)")
