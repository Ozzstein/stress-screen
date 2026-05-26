import subprocess
import sys
from pathlib import Path
import re

def test_e2e_cli():
    """Run the full CLI on the sample CSV and verify output format."""
    csv = next(Path(".").glob("*.csv"), None)
    if csv is None:
        import pytest; pytest.skip("No CSV file found in project root")
    result = subprocess.run(
        [sys.executable, "-m", "stress_screen", str(csv), "--no-html", "--no-pdf"],
        capture_output=True,
        text=True,
        cwd=str(Path(".").resolve()),
    )
    assert result.returncode in (0, 1), f"Unexpected exit code {result.returncode}: {result.stderr}"
    # Verify output contains module lines
    lines = result.stdout
    module_count = int(re.search(r"_M(\d+)\b", csv.name).group(1))
    for i in range(1, module_count + 1):
        assert re.search(rf"^M{i}:\s+(OK|MARGINAL|NOK)", lines, re.MULTILINE), \
            f"Missing M{i} verdict line in output"
