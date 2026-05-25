"""
cli.py — Main entry point for the stress_screen battery pack screener.

Orchestrates the full analysis pipeline:
    load_csv → derive_topology → segment → run_rest_analysis
             → run_li_plating_analysis → aggregate → print + reports
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemistry voltage presets
# ---------------------------------------------------------------------------

CHEM_VOLTAGE_BOUNDS: dict[str, tuple[float, float]] = {
    "lfp": (3.0, 3.65),
    "nmc": (3.0, 4.25),
    "nca": (3.0, 4.25),
}


# ---------------------------------------------------------------------------
# Helper: extract module count from filename
# ---------------------------------------------------------------------------

def _extract_module_count(csv_path: Path) -> int:
    """Parse the ``_M<n>`` component from the CSV filename.

    Raises
    ------
    ValueError
        If no ``_M<n>`` component is found.
    """
    m = re.search(r'_M(\d+)\b', csv_path.name)
    if not m:
        raise ValueError(
            f"Cannot determine module count from filename '{csv_path.name}'. "
            f"Expected a '_M<n>' component, e.g. '_M6.csv'."
        )
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

def _print_header(result_csv: Path, topo, segments) -> None:
    """Print the pack header block."""
    from stress_screen.segmentation import charge_segments, rest_segments

    n_charge = len(charge_segments(segments))
    n_discharge = sum(1 for s in segments if s.phase == "discharge")
    n_rest = len(rest_segments(segments))
    longest_rest = max(
        (s.duration_h for s in segments if s.phase == "rest"),
        default=0.0,
    )

    print(f"Pack: {result_csv.name}")
    print(
        f"Configuration: {topo.module_count} modules, "
        f"{topo.config_name} "
        f"({topo.parallel} parallel × {topo.series} series), "
        f"{topo.active_channels} active cell-groups"
    )
    print(
        f"Segments: {n_charge} charge, {n_discharge} discharge, "
        f"{n_rest} rest (longest rest: {longest_rest:.2f} h)"
    )


def _print_verdicts(module_verdicts, verbose: bool = False) -> None:
    """Print one line per module, and optionally per-method z-scores."""
    print()
    for mv in module_verdicts:
        # Build the flagged cells portion with method detail
        if mv.verdict == "NOK" and mv.flagged_cells:
            parts = []
            for fc in mv.flagged_cells:
                # Collect method names that fired HIGH on this cell
                high_methods = [
                    mr.method_name
                    for mr in fc.method_results
                    if mr.verdict == "HIGH"
                ]
                if high_methods:
                    methods_str = ", ".join(high_methods)
                    parts.append(f"{fc.label} ({fc.verdict} — methods: {methods_str})")
                else:
                    parts.append(f"{fc.label} ({fc.verdict})")
            flagged_str = ", ".join(parts)
            print(f"M{mv.module_id}: NOK  [cells flagged: {flagged_str}]")
        else:
            print(mv.summary_line)

        # Verbose: per-cell method z-scores for flagged cells
        if verbose and mv.flagged_cells:
            for fc in mv.flagged_cells:
                print(f"    {fc.label}  composite_z={fc.composite_z:.3f}")
                for mr in fc.method_results:
                    z_str = f"{mr.z_score:.3f}" if mr.z_score == mr.z_score else "NaN"
                    print(f"      {mr.method_name:20s}  z={z_str:>8s}  [{mr.verdict}]")

    print()
    nok_count = sum(1 for m in module_verdicts if m.verdict == "NOK")
    total = len(module_verdicts)
    if nok_count == 0:
        print(f"Result: all {total} modules OK")
    else:
        print(f"Result: {nok_count} of {total} modules NOK")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        prog="stress_screen",
        description="Battery pack stress-test module screener",
    )
    p.add_argument(
        "csv",
        type=Path,
        help="Path to the stress-test CSV file",
    )
    p.add_argument(
        "--chem",
        choices=["lfp", "nmc", "nca"],
        default="lfp",
        help="Battery chemistry (affects OCV-fit voltage bounds). Default: lfp",
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to custom temp_mapping.yaml (default: bundled configs/temp_mapping.yaml)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for HTML+PDF reports (default: same directory as input CSV)",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF report generation",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor for CSV loading (1=no downsampling, 60=1 per minute). Default: 1",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-method z-scores",
    )
    args = p.parse_args()

    try:
        _run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception:
        if args.verbose:
            traceback.print_exc()
        else:
            exc_type, exc_val, _ = sys.exc_info()
            print(
                f"Unexpected error ({exc_type.__name__}): {exc_val}\n"
                "Run with --verbose for full traceback.",
                file=sys.stderr,
            )
        sys.exit(2)


def _run(args: argparse.Namespace) -> None:
    """Execute the full pipeline; separated from main() for clean error wrapping."""
    from stress_screen.loader import load_csv, active_channel_count
    from stress_screen.topology import derive_topology
    from stress_screen.segmentation import segment, rest_segments, charge_segments
    from stress_screen.analysis.rest import run_rest_analysis, RestParams
    from stress_screen.analysis.li_plating import run_li_plating_analysis
    from stress_screen.analysis.aggregate import aggregate
    from stress_screen.models import AnalysisResult

    csv_path: Path = args.csv.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    top_df, cell_df = load_csv(csv_path, downsample=args.downsample)

    # ------------------------------------------------------------------
    # 2. Derive topology
    # ------------------------------------------------------------------
    module_count = _extract_module_count(csv_path)
    n_active = active_channel_count(cell_df)
    topology = derive_topology(
        active_channels=n_active,
        module_count=module_count,
        mapping_file=args.mapping,
    )

    # ------------------------------------------------------------------
    # 3. Segment
    # ------------------------------------------------------------------
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        segments = segment(top_df)

    for w in caught_warnings:
        print(f"Warning: {w.message}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 4. Slice data for analyses
    # ------------------------------------------------------------------
    rest_segs = rest_segments(segments)
    charge_segs = charge_segments(segments)

    if not rest_segs:
        raise ValueError("No rest segment found in data. Cannot run OCV analysis.")

    # Use the longest rest segment for OCV analysis
    longest_rest = rest_segs[0]
    rest_cell_df = cell_df[
        (cell_df["time_hours"] >= top_df.iloc[longest_rest.start_row]["time_hours"])
        & (cell_df["time_hours"] <= top_df.iloc[longest_rest.end_row]["time_hours"])
    ].copy()

    # Use the first charge segment (or empty frame if none)
    if charge_segs:
        first_charge = charge_segs[0]
        charge_cell_df = cell_df[
            (cell_df["time_hours"] >= top_df.iloc[first_charge.start_row]["time_hours"])
            & (cell_df["time_hours"] <= top_df.iloc[first_charge.end_row]["time_hours"])
        ].copy()
    else:
        charge_cell_df = cell_df.iloc[0:0].copy()  # empty, same schema

    # ------------------------------------------------------------------
    # 5. Rest analysis (M1–M6)
    # ------------------------------------------------------------------
    v_low, v_high = CHEM_VOLTAGE_BOUNDS[args.chem]
    rest_params = RestParams(voltage_bounds=(v_low, v_high))

    rest_results = run_rest_analysis(rest_cell_df, topology, params=rest_params)

    # ------------------------------------------------------------------
    # 6. Li-plating analysis
    # ------------------------------------------------------------------
    # For relaxation analysis use only the beginning of the rest window
    li_rest_cell_df = rest_cell_df
    li_results = run_li_plating_analysis(charge_cell_df, li_rest_cell_df)

    # ------------------------------------------------------------------
    # 7. Aggregate into module verdicts
    # ------------------------------------------------------------------
    module_verdicts = aggregate(rest_results, li_results, topology)

    result = AnalysisResult(
        csv_path=csv_path,
        topology=topology,
        segments=segments,
        module_verdicts=module_verdicts,
    )

    # ------------------------------------------------------------------
    # 8. Terminal output
    # ------------------------------------------------------------------
    _print_header(csv_path, topology, segments)
    _print_verdicts(module_verdicts, verbose=args.verbose)

    # ------------------------------------------------------------------
    # 9. Reports
    # ------------------------------------------------------------------
    out_dir = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    if not args.no_html:
        from stress_screen.reports.html import write_html_report
        html_path = out_dir / f"{stem}_report.html"
        write_html_report(result, rest_cell_df, charge_cell_df, top_df, html_path)
        print(f"HTML report: {html_path}")

    if not args.no_pdf:
        from stress_screen.reports.pdf import write_pdf_report
        pdf_path = out_dir / f"{stem}_report.pdf"
        write_pdf_report(result, rest_cell_df, charge_cell_df, top_df, pdf_path)
        print(f"PDF report:  {pdf_path}")

    # ------------------------------------------------------------------
    # 10. Exit code
    # ------------------------------------------------------------------
    sys.exit(1 if result.any_nok else 0)
