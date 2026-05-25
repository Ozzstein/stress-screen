import pytest
from pathlib import Path

def test_html_and_pdf_generated(tmp_path):
    """Smoke test: run CLI with --out-dir and verify both files are created."""
    import subprocess, sys, re
    csv = next(Path(".").glob("*.csv"), None)
    if csv is None:
        pytest.skip("No CSV file found in project root")
    result = subprocess.run(
        [sys.executable, "-m", "stress_screen", str(csv), "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(Path(".").resolve()),
    )
    assert result.returncode in (0, 1), f"Exit code {result.returncode}: {result.stderr}"
    html_files = list(tmp_path.glob("*.html"))
    pdf_files = list(tmp_path.glob("*.pdf"))
    assert len(html_files) == 1, f"Expected 1 HTML file, got {html_files}"
    assert len(pdf_files) == 1, f"Expected 1 PDF file, got {pdf_files}"
    assert html_files[0].stat().st_size > 10_000, "HTML file suspiciously small"
    assert pdf_files[0].stat().st_size > 10_000, "PDF file suspiciously small"
