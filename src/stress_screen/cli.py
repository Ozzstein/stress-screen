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


def _print_verdicts(module_verdicts, verbose: bool = False, quiet: bool = False) -> None:
    """Print one line per module, and optionally per-method z-scores.

    When ``quiet`` is True, prints only the per-module verdict lines (no
    leading/trailing blank lines, no summary count).
    """
    if not quiet:
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

    if quiet:
        return

    print()
    nok_count = sum(1 for m in module_verdicts if m.verdict == "NOK")
    marginal_count = sum(1 for m in module_verdicts if m.verdict == "MARGINAL")
    total = len(module_verdicts)
    if nok_count == 0 and marginal_count == 0:
        print(f"Result: all {total} modules OK")
    elif nok_count == 0:
        print(f"Result: {marginal_count} of {total} modules MARGINAL")
    elif marginal_count == 0:
        print(f"Result: {nok_count} of {total} modules NOK")
    else:
        print(f"Result: {nok_count} of {total} modules NOK, {marginal_count} MARGINAL")


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
    p.add_argument(
        "--full",
        action="store_true",
        help="Print the full output (pack header, progress messages, "
             "result summary, and report-path lines). By default only the "
             "per-module verdict lines are printed.",
    )
    args = p.parse_args()

    # Default behaviour is terse: only per-module verdict lines. Pass --full
    # to opt into the legacy header + summary + report-path output.
    args.quiet = not args.full

    from stress_screen._progress import set_quiet
    set_quiet(args.quiet)

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
    from stress_screen._progress import get as get_progress
    from stress_screen.loader import load_csv, active_channel_count, remap_temperatures
    from stress_screen.topology import derive_topology
    from stress_screen.segmentation import segment, rest_segments, charge_segments
    from stress_screen.analysis.rest import run_rest_analysis, RestParams
    from stress_screen.analysis.li_plating import run_li_plating_analysis
    from stress_screen.analysis.aggregate import aggregate
    from stress_screen.analysis.short_circuit import run_isc_analysis
    from stress_screen.models import AnalysisResult

    prog = get_progress()

    csv_path: Path = args.csv.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    prog.stage("Loading CSV...")
    top_df, cell_df = load_csv(csv_path, downsample=args.downsample)
    n_active = active_channel_count(cell_df)
    prog.stage(f"Loaded {len(top_df)} rows, {n_active} active channels")

    # ------------------------------------------------------------------
    # 2. Derive topology
    # ------------------------------------------------------------------
    prog.stage("Deriving pack topology...")
    module_count = _extract_module_count(csv_path)
    topology = derive_topology(
        active_channels=n_active,
        module_count=module_count,
        mapping_file=args.mapping,
    )
    prog.stage(
        f"{topology.config_name} "
        f"({topology.parallel} parallel x {topology.series} series), "
        f"{topology.module_count} modules"
    )

    # Apply staggered sensor mapping: each group temperature is the average
    # of the sensors that bracket it (per temp_mapping.yaml), replacing the
    # raw 1:1 Cell_N_Temp assignment from the loader.
    cell_df = remap_temperatures(cell_df, topology)

    # ------------------------------------------------------------------
    # 3. Segment
    # ------------------------------------------------------------------
    prog.stage("Segmenting charge/discharge/rest phases...")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        segments = segment(top_df)

    for w in caught_warnings:
        print(f"Warning: {w.message}", file=sys.stderr)

    _n_charge = sum(1 for s in segments if s.phase == "charge")
    _n_discharge = sum(1 for s in segments if s.phase == "discharge")
    _rs = rest_segments(segments)
    _longest_rest_h = _rs[0].duration_h if _rs else 0.0
    prog.stage(
        f"{_n_charge} charge, {_n_discharge} discharge, "
        f"{len(_rs)} rest segment(s) (longest rest: {_longest_rest_h:.2f} h)"
    )

    # ------------------------------------------------------------------
    # 4. Slice data for analyses
    # ------------------------------------------------------------------
    rest_segs = rest_segments(segments)
    charge_segs = charge_segments(segments)

    if not rest_segs:
        raise ValueError(
            "Test invalidated: no rest segment >= 48 h found in the data. "
            "The stress-test protocol requires a final rest period of at "
            "least 48 h for OCV analysis."
        )

    # Use the longest rest segment for OCV analysis
    longest_rest = rest_segs[0]
    rest_cell_df = cell_df[
        (cell_df["time_hours"] >= top_df.iloc[longest_rest.start_row]["time_hours"])
        & (cell_df["time_hours"] <= top_df.iloc[longest_rest.end_row]["time_hours"])
    ].copy()

    # Use the last charge segment that ended before the longest rest began.
    # This is the conditioning charge whose signatures (dV/dQ, relaxation,
    # temperature) are most informative for plating and ISC detection.
    if charge_segs:
        pre_rest = [s for s in charge_segs if s.end_time_h <= longest_rest.start_time_h]
        target_charge = pre_rest[-1] if pre_rest else charge_segs[-1]
        charge_time_min = float(top_df.iloc[target_charge.start_row]["time_hours"])
        charge_time_max = float(top_df.iloc[target_charge.end_row]["time_hours"])
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

    if charge_segs:
        charge_idx = charge_segs.index(target_charge) + 1
        prog.stage(
            f"Using charge cycle {charge_idx}/{len(charge_segs)} "
            f"({charge_time_min:.2f}–{charge_time_max:.2f} h) "
            f"for Li-plating / ISC analysis"
        )

    # ------------------------------------------------------------------
    # 5. Rest analysis (six detection methods)
    # ------------------------------------------------------------------
    v_low, v_high = CHEM_VOLTAGE_BOUNDS[args.chem]
    rest_params = RestParams(voltage_bounds=(v_low, v_high))

    prog.stage(f"Running rest analysis (6 detection methods) on {n_active} channels...")
    rest_results = run_rest_analysis(rest_cell_df, topology, params=rest_params)

    # ------------------------------------------------------------------
    # 6. Li-plating analysis
    # ------------------------------------------------------------------
    # For relaxation analysis use only the beginning of the rest window
    li_rest_cell_df = rest_cell_df
    prog.stage(f"Running Li-plating analysis on {n_active} channels...")
    li_results = run_li_plating_analysis(
        charge_cell_df,
        li_rest_cell_df,
        top_charge_df=charge_top_df,
        n_parallel=topology.parallel,
    )

    # ------------------------------------------------------------------
    # 6b. ISC analysis
    # ------------------------------------------------------------------
    prog.stage(f"Running ISC analysis on {n_active} channels...")
    isc_results = run_isc_analysis(
        rest_cell_df,
        rest_results,
        charge_cell_df,
        top_charge_df=charge_top_df,
        n_parallel=topology.parallel,
        topology=topology,
    )

    # ------------------------------------------------------------------
    # 7. Aggregate into module verdicts
    # ------------------------------------------------------------------
    prog.stage("Aggregating verdicts...")
    module_verdicts = aggregate(rest_results, li_results, topology, isc_results=isc_results)

    result = AnalysisResult(
        csv_path=csv_path,
        topology=topology,
        segments=segments,
        module_verdicts=module_verdicts,
    )

    # ------------------------------------------------------------------
    # 8. Terminal output
    # ------------------------------------------------------------------
    if not args.quiet:
        _print_header(csv_path, topology, segments)
    _print_verdicts(module_verdicts, verbose=args.verbose, quiet=args.quiet)

    # ------------------------------------------------------------------
    # 9. Reports
    # ------------------------------------------------------------------
    out_dir = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    if not args.no_html:
        from stress_screen.reports.html import write_html_report
        html_path = out_dir / f"{stem}_report.html"
        prog.stage(f"Writing HTML report -> {html_path}")
        write_html_report(result, rest_cell_df, charge_cell_df, top_df, html_path,
                          top_charge_df=charge_top_df, n_parallel=topology.parallel)
        if not args.quiet:
            print(f"HTML report: {html_path}")

    if not args.no_pdf:
        from stress_screen.reports.pdf import write_pdf_report
        pdf_path = out_dir / f"{stem}_report.pdf"
        prog.stage(f"Writing PDF report -> {pdf_path}")
        write_pdf_report(result, rest_cell_df, charge_cell_df, top_df, pdf_path,
                         top_charge_df=charge_top_df, n_parallel=topology.parallel)
        if not args.quiet:
            print(f"PDF report:  {pdf_path}")

    prog.stage("Done.")

    # ------------------------------------------------------------------
    # 10. Exit code
    # ------------------------------------------------------------------
    sys.exit(1 if result.any_nok else 0)
