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
                if fc.cluster_scores:
                    clusters_str = "  ".join(
                        f"{name}={score:.2f}"
                        for name, score in fc.cluster_scores.items()
                    )
                    print(f"      clusters: {clusters_str}")
                for mr in fc.method_results:
                    z_str = f"{mr.z_score:.3f}" if mr.z_score == mr.z_score else "NaN"
                    print(f"      {mr.method_name:20s}  z={z_str:>8s}  [{mr.verdict}]")

    if quiet:
        return

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

#: Registered subcommands. Anything else as the first argument (a CSV path,
#: a flag, nothing at all) is treated as an implicit "run" — so the historical
#: `stress_screen file.csv --chem nmc` invocation works forever.
_SUBCOMMANDS = {"run", "calibrate", "trend", "history", "batch"}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in _SUBCOMMANDS:
        argv = ["run", *argv]

    parser = argparse.ArgumentParser(
        prog="stress_screen",
        description="Battery pack stress-test module screener",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Analyse a tester CSV and produce verdicts + reports (default)",
        description="Battery pack stress-test module screener",
    )
    _add_run_arguments(run_p)

    cal_p = sub.add_parser(
        "calibrate",
        help="Score past JSON results against labeled ground-truth outcomes",
        description=(
            "Join *_result.json files to a labels CSV "
            "(pack_id;module_id;group;outcome) and report confusion matrices, "
            "per-method/per-cluster separation power, and an optional "
            "threshold sweep."
        ),
    )
    cal_p.add_argument("--results", type=Path, required=True,
                       help="Directory containing *_result.json files")
    cal_p.add_argument("--labels", type=Path, required=True,
                       help="Labels CSV: pack_id;module_id;group;outcome")
    cal_p.add_argument("--sweep", action="store_true",
                       help="Sweep composite_z thresholds and suggest an operating point")

    trend_p = sub.add_parser(
        "trend",
        help="Track a pack across tests over time (raw k trajectories)",
        description=(
            "Compare a pack's runs in a history store over time. Trends are "
            "computed on raw physical metrics (self-discharge k) — z-scores "
            "are fleet-relative within one run and would hide uniform "
            "degradation. Exit code 1 when any cell is flagged as worsening."
        ),
    )
    trend_p.add_argument("--history", type=Path, required=True,
                         help="History store directory")
    trend_p.add_argument("--pack", required=True, help="Pack id to trend")
    trend_p.add_argument("--module", type=int, default=None, help="Module number")
    trend_p.add_argument("--group", type=int, default=None,
                         help="Cell-group number (requires --module)")
    trend_p.add_argument("--k-slope-floor", type=float, default=1e-6,
                         help="Theil-Sen k-slope (1/h per run) above which a "
                              "cell counts as worsening. Default: 1e-6")

    hist_p = sub.add_parser(
        "history",
        help="List runs in a history store",
    )
    hist_p.add_argument("--history", type=Path, required=True,
                        help="History store directory")
    hist_p.add_argument("--pack", default=None, help="Filter by pack id")
    hist_p.add_argument("--rebuild-index", action="store_true",
                        help="Rebuild index.jsonl from the stored JSON files")

    batch_p = sub.add_parser(
        "batch",
        help="Analyse every tester CSV in a directory (continue on error)",
    )
    batch_p.add_argument("directory", type=Path, help="Directory of *_M<n>.csv files")
    batch_p.add_argument("--chem", choices=["lfp", "nmc", "nca"], default="lfp")
    batch_p.add_argument("--config", type=Path, default=None)
    batch_p.add_argument("--out-dir", type=Path, default=None)
    batch_p.add_argument("--no-html", action="store_true")
    batch_p.add_argument("--no-pdf", action="store_true")
    batch_p.add_argument("--downsample", type=int, default=1)
    batch_p.add_argument("--history", type=Path, default=None,
                         help="Add each run's JSON result to this history store")

    args = parser.parse_args(argv)

    if args.command == "calibrate":
        _cmd_calibrate(args)
    elif args.command == "trend":
        _cmd_trend(args)
    elif args.command == "history":
        _cmd_history(args)
    elif args.command == "batch":
        _cmd_batch(args)
    else:
        _cmd_run(args)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from stress_screen.calibrate import run_calibration

    try:
        print(run_calibration(args.results, args.labels, sweep=args.sweep))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


def _add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "csv",
        type=Path,
        help="Path to the stress-test CSV file",
    )
    p.add_argument(
        "--chem",
        choices=["lfp", "nmc", "nca"],
        default="lfp",
        help="Battery chemistry preset (selects OCV-fit voltage bounds and "
             "chemistry-specific parameters). Default: lfp",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML analysis-config file overriding preset parameters "
             "(see configs/analysis_defaults.yaml for every available key)",
    )
    p.add_argument(
        "--c-rate",
        type=float,
        default=None,
        help="Cell-level charge C-rate of the test protocol (default: 0.5). "
             "Scales dQ/dV peak-detection and thermal noise floors.",
    )
    p.add_argument(
        "--capacity-ah",
        type=float,
        default=None,
        help="Nominal cell capacity in Ah (default: 2.5)",
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
        "--no-json",
        action="store_true",
        help="Skip JSON result generation",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the JSON result to this path (default: <csv stem>_result.json "
             "in the output directory)",
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
    p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Add this run's JSON result to the given history store directory "
             "(implies JSON generation)",
    )
    p.add_argument(
        "--pack-id",
        default=None,
        help="Override the pack identifier recorded in the JSON result "
             "(default: derived from the filename)",
    )


def _cmd_run(args: argparse.Namespace) -> None:
    # Default behaviour is terse: only per-module verdict lines. Pass --full
    # to opt into the legacy header + summary + report-path output.
    args.quiet = not args.full

    from stress_screen._progress import set_quiet
    set_quiet(args.quiet)

    try:
        sys.exit(_run(args))
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


def _cmd_trend(args: argparse.Namespace) -> None:
    from stress_screen.history import HistoryStore, analyze_cell_trend

    store = HistoryStore(args.history)
    entries = store.entries(args.pack)
    if not entries:
        known = ", ".join(store.packs()) or "(store is empty)"
        print(f"Error: no runs for pack '{args.pack}' in {args.history}. "
              f"Known packs: {known}", file=sys.stderr)
        sys.exit(2)

    print(f"Pack {args.pack}: {len(entries)} run(s)")
    for e in entries:
        mods = "  ".join(f"M{m}:{v}" for m, v in sorted(e.modules.items()))
        print(f"  {e.test_date or 'unknown':10s}  {e.overall:8s}  {mods}")

    def _cell_labels() -> list[str]:
        labels: set[str] = set()
        for e in entries:
            labels.update(e.cells.keys())
        return sorted(labels, key=lambda s: (int(s.split('/')[0][1:]), int(s.split('/')[1][1:])))

    if args.group is not None and args.module is None:
        print("Error: --group requires --module", file=sys.stderr)
        sys.exit(2)

    if args.module is not None and args.group is not None:
        labels = [f"M{args.module}/G{args.group}"]
    elif args.module is not None:
        labels = [lb for lb in _cell_labels() if lb.startswith(f"M{args.module}/")]
    else:
        labels = _cell_labels()

    any_worsening = False
    detail = args.module is not None and args.group is not None
    print()
    for label in labels:
        module_id = int(label.split("/")[0][1:])
        group = int(label.split("/")[1][1:])
        series = store.cell_series(args.pack, module_id, group)
        if not series:
            continue
        trend = analyze_cell_trend(series, k_slope_floor=args.k_slope_floor)
        if detail:
            print(f"{label}: per-run history")
            for s in series:
                k = f"{s['k']:.3e}" if s.get("k") is not None else "—"
                kc = f"{s['k_corrected']:.3e}" if s.get("k_corrected") is not None else "—"
                cz = f"{s['composite_z']:.2f}" if s.get("composite_z") is not None else "—"
                print(f"  {s['test_date'] or 'unknown':10s}  k={k}  k25={kc}  "
                      f"composite_z={cz}  [{s.get('verdict', '?')}]")
        if trend["worsening"]:
            any_worsening = True
            reasons = []
            if trend["k_worsening"]:
                reasons.append(f"k slope {trend['k_slope_per_run']:.2e}/run")
            if trend["entered_flagged"]:
                reasons.append(f"entered {series[-1].get('verdict')}")
            print(f"  WORSENING {label}: {', '.join(reasons)} "
                  f"(over {trend['n_runs']} runs)")

    if not any_worsening:
        print("No worsening cells flagged.")
    sys.exit(1 if any_worsening else 0)


def _cmd_history(args: argparse.Namespace) -> None:
    from stress_screen.history import HistoryStore

    store = HistoryStore(args.history)
    if args.rebuild_index:
        n = store.rebuild_index()
        print(f"Index rebuilt: {n} run(s)")
    entries = store.entries(args.pack)
    if not entries:
        print("No runs in store." if args.pack is None
              else f"No runs for pack '{args.pack}'.")
        sys.exit(0)
    for e in entries:
        mods = "  ".join(f"M{m}:{v}" for m, v in sorted(e.modules.items()))
        print(f"{e.pack_id:24s} {e.test_date or 'unknown':10s} {e.overall:8s} {mods}")
    sys.exit(0)


def _cmd_batch(args: argparse.Namespace) -> None:
    from stress_screen._progress import set_quiet
    set_quiet(True)

    csvs = sorted(p for p in args.directory.glob("*_M*.csv"))
    if not csvs:
        print(f"Error: no *_M<n>.csv files found in {args.directory}", file=sys.stderr)
        sys.exit(2)

    worst = 0
    summary: list[tuple[str, str]] = []
    for csv in csvs:
        run_args = argparse.Namespace(
            csv=csv, chem=args.chem, config=args.config, c_rate=None,
            capacity_ah=None, mapping=None, out_dir=args.out_dir,
            no_html=args.no_html, no_pdf=args.no_pdf, no_json=False,
            json_out=None, downsample=args.downsample, verbose=False,
            full=False, quiet=True, history=args.history, pack_id=None,
        )
        try:
            code = _run(run_args)
            status = {0: "OK", 1: "NOK"}.get(code, f"exit {code}")
        except (ValueError, FileNotFoundError) as exc:
            code, status = 2, f"ERROR: {exc}"
        except Exception as exc:  # continue-on-error is the point of batch
            code, status = 2, f"ERROR ({type(exc).__name__}): {exc}"
        worst = max(worst, code)
        summary.append((csv.name, status))

    print()
    print(f"Batch: {len(csvs)} file(s)")
    for name, status in summary:
        print(f"  {name}: {status}")
    sys.exit(worst)


def _run(args: argparse.Namespace) -> int:
    """Execute the full pipeline and return the exit code (0 ok / 1 any NOK);
    separated from the command wrappers for clean error handling and reuse
    by the batch command."""
    from stress_screen._progress import get as get_progress
    from stress_screen.config import load_config
    from stress_screen.loader import load_csv, active_channel_count, remap_temperatures
    from stress_screen.topology import derive_topology
    from stress_screen.segmentation import segment, rest_segments, charge_segments
    from stress_screen.analysis.rest import run_rest_analysis
    from stress_screen.analysis.li_plating import run_li_plating_analysis
    from stress_screen.analysis.aggregate import aggregate
    from stress_screen.analysis.short_circuit import run_isc_analysis
    from stress_screen.models import AnalysisResult

    prog = get_progress()

    csv_path: Path = args.csv.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # ------------------------------------------------------------------
    # 0. Resolve analysis configuration
    #    (defaults ← chemistry preset ← --config file ← explicit CLI flags)
    # ------------------------------------------------------------------
    protocol_overrides = {}
    if args.c_rate is not None:
        protocol_overrides["c_rate"] = args.c_rate
    if args.capacity_ah is not None:
        protocol_overrides["nominal_capacity_ah"] = args.capacity_ah
    config = load_config(
        config_path=args.config,
        chem=args.chem,
        cli_overrides={"protocol": protocol_overrides} if protocol_overrides else None,
    )

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
    prog.stage(f"Running rest analysis (6 detection methods) on {n_active} channels...")
    rest_results = run_rest_analysis(rest_cell_df, topology, params=config.rest)

    # ------------------------------------------------------------------
    # 6. Li-plating analysis
    # ------------------------------------------------------------------
    # For relaxation analysis use only the beginning of the rest window
    li_rest_cell_df = rest_cell_df
    prog.stage(f"Running Li-plating analysis on {n_active} channels...")
    li_results = run_li_plating_analysis(
        charge_cell_df,
        li_rest_cell_df,
        params=config.li_plating,
        top_charge_df=charge_top_df,
        n_parallel=topology.parallel,
        protocol=config.protocol,
    )

    # ------------------------------------------------------------------
    # 6b. ISC analysis
    # ------------------------------------------------------------------
    prog.stage(f"Running ISC analysis on {n_active} channels...")
    isc_results = run_isc_analysis(
        rest_cell_df,
        rest_results,
        charge_cell_df,
        params=config.short_circuit,
        top_charge_df=charge_top_df,
        n_parallel=topology.parallel,
        topology=topology,
    )

    # ------------------------------------------------------------------
    # 7. Aggregate into module verdicts
    # ------------------------------------------------------------------
    prog.stage("Aggregating verdicts...")
    module_verdicts = aggregate(
        rest_results, li_results, topology,
        isc_results=isc_results, params=config.aggregate,
        composite=config.composite,
    )

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

    # Build every Plotly figure and the findings exactly once; both report
    # writers share them.
    figures = None
    findings = None
    if not args.no_html or not args.no_pdf:
        from stress_screen.reports.figures import build_figures
        from stress_screen.reports.findings import build_findings
        prog.stage("Building report figures...")
        figures = build_figures(
            result, rest_cell_df, charge_cell_df, top_df,
            top_charge_df=charge_top_df, n_parallel=topology.parallel,
        )
        findings = build_findings(result)

    if not args.no_html:
        from stress_screen.reports.html import write_html_report
        html_path = out_dir / f"{stem}_report.html"
        prog.stage(f"Writing HTML report -> {html_path}")
        write_html_report(result, rest_cell_df, charge_cell_df, top_df, html_path,
                          top_charge_df=charge_top_df, n_parallel=topology.parallel,
                          figures=figures, findings=findings)
        if not args.quiet:
            print(f"HTML report: {html_path}")

    if not args.no_pdf:
        from stress_screen.reports.pdf import write_pdf_report
        pdf_path = out_dir / f"{stem}_report.pdf"
        prog.stage(f"Writing PDF report -> {pdf_path}")
        write_pdf_report(result, rest_cell_df, charge_cell_df, top_df, pdf_path,
                         top_charge_df=charge_top_df, n_parallel=topology.parallel,
                         figures=figures, findings=findings)
        if not args.quiet:
            print(f"PDF report:  {pdf_path}")

    history_dir = getattr(args, "history", None)
    if not args.no_json or history_dir is not None:
        from stress_screen.serialize import write_json_result
        json_path = args.json_out or (out_dir / f"{stem}_result.json")
        prog.stage(f"Writing JSON result -> {json_path}")
        run_info: dict = {"downsample": args.downsample}
        if getattr(args, "pack_id", None):
            run_info["pack_id"] = args.pack_id
        write_json_result(
            result,
            json_path,
            config=config.to_dict(),
            run_info=run_info,
        )
        if not args.quiet:
            print(f"JSON result: {json_path}")

        if history_dir is not None:
            from stress_screen.history import HistoryStore
            summary = HistoryStore(history_dir).add(json_path)
            prog.stage(f"Added to history store ({summary.pack_id})")

    prog.stage("Done.")

    # ------------------------------------------------------------------
    # 10. Exit code
    # ------------------------------------------------------------------
    return 1 if result.any_nok else 0
