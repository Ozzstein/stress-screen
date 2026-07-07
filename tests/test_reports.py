"""Report-generation smoke test on the synthetic fixture."""

import os
import subprocess
import sys
from pathlib import Path

from tests.synth import make_synthetic_csv


def test_html_and_pdf_generated(tmp_path):
    """Run the CLI with --out-dir and verify HTML, PDF, and JSON are created."""
    csv = make_synthetic_csv(tmp_path / "input", leaky_channels={4: 0.8})
    out_dir = tmp_path / "out"

    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "stress_screen", str(csv), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode in (0, 1), f"Exit code {result.returncode}: {result.stderr}"

    html_files = list(out_dir.glob("*.html"))
    pdf_files = list(out_dir.glob("*.pdf"))
    json_files = list(out_dir.glob("*.json"))
    assert len(html_files) == 1, f"Expected 1 HTML file, got {html_files}"
    assert len(pdf_files) == 1, f"Expected 1 PDF file, got {pdf_files}"
    assert len(json_files) == 1, f"Expected 1 JSON file, got {json_files}"
    assert html_files[0].stat().st_size > 10_000, "HTML file suspiciously small"
    assert pdf_files[0].stat().st_size > 10_000, "PDF file suspiciously small"

    # The report must carry the real test date from the filename — never
    # today's date (the _D<DDMMYYYY>_ fix).
    html = html_files[0].read_text(encoding="utf-8", errors="replace")
    assert "2026-03-01" in html
