"""
calibrate.py — score past verdicts against labeled ground-truth outcomes.

The screener's thresholds are statistical defaults; this command turns
accumulated outcomes (teardowns, capacity tests, field returns) into
evidence: a confusion matrix at the current gates, per-method and
per-cluster separation power, and optionally a threshold sweep suggesting
a better operating point.

Labels file (semicolon-delimited, header required)::

    pack_id;module_id;group;outcome
    DataLogging_C1_I01;1;7;bad
    DataLogging_C1_I01;2;;good        # empty group = module-level label

``pack_id`` must match the ``input.pack_id`` of the JSON results (the CSV
filename stem with the date/_M<n> components stripped). ``outcome`` is
``good`` or ``bad``. Cell-level rows are preferred; module-level rows are
matched against the module verdict and, for score-based metrics, the
module's worst cell.

Usage::

    stress_screen calibrate --results DIR --labels labels.csv [--sweep]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabelRow:
    pack_id: str
    module_id: int
    group: int | None          # None = module-level label
    outcome: str                # "good" | "bad"


@dataclass
class ScoredItem:
    """One labeled item joined to its analysis result."""

    label: LabelRow
    verdict: str                # cell verdict or module verdict
    composite_z: float | None
    method_z: dict[str, float]          # method name -> z
    cluster_scores: dict[str, float]    # cluster name -> score


def load_labels(path: Path) -> list[LabelRow]:
    rows: list[LabelRow] = []
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"Labels file {path} is empty")
    header = [c.strip().lower() for c in lines[0].split(";")]
    expected = ["pack_id", "module_id", "group", "outcome"]
    if header != expected:
        raise ValueError(
            f"Labels file header must be '{';'.join(expected)}', got {lines[0]!r}"
        )
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [c.strip() for c in line.split(";")]
        if len(parts) < 4:
            raise ValueError(f"Labels file line {i}: expected 4 fields, got {line!r}")
        pack_id, module_s, group_s, outcome = parts[0], parts[1], parts[2], parts[3]
        outcome = outcome.lower()
        if outcome not in ("good", "bad"):
            raise ValueError(f"Labels file line {i}: outcome must be good|bad, got {outcome!r}")
        rows.append(LabelRow(
            pack_id=pack_id,
            module_id=int(module_s),
            group=int(group_s) if group_s else None,
            outcome=outcome,
        ))
    return rows


def _load_results(results_dir: Path) -> list[dict]:
    files = sorted(Path(results_dir).glob("*_result.json"))
    if not files:
        raise FileNotFoundError(f"No *_result.json files found in {results_dir}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _join(labels: list[LabelRow], results: list[dict]) -> tuple[list[ScoredItem], list[LabelRow]]:
    """Join labels to results. Returns (matched items, unmatched labels)."""
    # pack_id -> list of result dicts (a pack can be tested multiple times;
    # every matching run is scored)
    by_pack: dict[str, list[dict]] = {}
    for r in results:
        by_pack.setdefault(r.get("input", {}).get("pack_id", ""), []).append(r)

    items: list[ScoredItem] = []
    unmatched: list[LabelRow] = []
    for lab in labels:
        found = False
        for r in by_pack.get(lab.pack_id, []):
            module = next(
                (m for m in r.get("modules", []) if m["module_id"] == lab.module_id),
                None,
            )
            if module is None:
                continue
            if lab.group is not None:
                cell = next(
                    (c for c in module["cells"] if c["group_in_module"] == lab.group),
                    None,
                )
                if cell is None:
                    continue
                items.append(ScoredItem(
                    label=lab,
                    verdict=cell["verdict"],
                    composite_z=cell.get("composite_z"),
                    method_z={
                        m["name"]: m["z"] for m in cell.get("methods", [])
                        if m.get("z") is not None
                    },
                    cluster_scores=cell.get("cluster_scores") or {},
                ))
            else:
                # Module-level label: verdict from the module, scores from
                # its worst cell (max composite).
                cells = module.get("cells", [])
                worst = max(
                    cells,
                    key=lambda c: c.get("composite_z") if c.get("composite_z") is not None else float("-inf"),
                    default=None,
                )
                items.append(ScoredItem(
                    label=lab,
                    verdict=module["verdict"],
                    composite_z=worst.get("composite_z") if worst else None,
                    method_z={
                        m["name"]: m["z"] for m in (worst or {}).get("methods", [])
                        if m.get("z") is not None
                    },
                    cluster_scores=(worst or {}).get("cluster_scores") or {},
                ))
            found = True
        if not found:
            unmatched.append(lab)
    return items, unmatched


def _rank_auc(good: list[float], bad: list[float]) -> float | None:
    """P(score_bad > score_good) — rank-based AUC, no dependencies."""
    if not good or not bad:
        return None
    wins = ties = 0
    for b in bad:
        for g in good:
            if b > g:
                wins += 1
            elif b == g:
                ties += 1
    return (wins + 0.5 * ties) / (len(good) * len(bad))


def _confusion(items: list[ScoredItem], bad_verdicts: set[str]) -> dict[str, int]:
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for it in items:
        predicted_bad = it.verdict in bad_verdicts
        actually_bad = it.label.outcome == "bad"
        if predicted_bad and actually_bad:
            c["tp"] += 1
        elif predicted_bad:
            c["fp"] += 1
        elif actually_bad:
            c["fn"] += 1
        else:
            c["tn"] += 1
    return c


def _fmt_confusion(name: str, c: dict[str, int]) -> str:
    n = sum(c.values())
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan")
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
    return (
        f"{name} (n={n}):\n"
        f"    TP={c['tp']:3d}  FP={c['fp']:3d}\n"
        f"    FN={c['fn']:3d}  TN={c['tn']:3d}\n"
        f"    precision={precision:.2f}  recall={recall:.2f}"
    )


def run_calibration(
    results_dir: Path,
    labels_path: Path,
    sweep: bool = False,
) -> str:
    """Run the calibration analysis and return the text report."""
    labels = load_labels(labels_path)
    results = _load_results(results_dir)
    items, unmatched = _join(labels, results)

    lines: list[str] = []
    lines.append(f"Results: {len(results)} runs from {results_dir}")
    lines.append(f"Labels:  {len(labels)} rows, {len(items)} matched to results")
    if unmatched:
        lines.append(f"WARNING: {len(unmatched)} label rows had no matching result:")
        for lab in unmatched[:10]:
            grp = f"/G{lab.group}" if lab.group is not None else ""
            lines.append(f"  {lab.pack_id} M{lab.module_id}{grp}")
    if not items:
        lines.append("Nothing to calibrate — no labels matched any result JSON.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Confusion matrices at the current gates
    # ------------------------------------------------------------------
    cell_items = [it for it in items if it.label.group is not None]
    module_items = [it for it in items if it.label.group is None]
    lines.append("")
    lines.append("== Confusion at current gates ==")
    if cell_items:
        lines.append(_fmt_confusion(
            "Cells, strict (HIGH = bad)", _confusion(cell_items, {"HIGH"})))
        lines.append(_fmt_confusion(
            "Cells, lenient (HIGH or ELEVATED = bad)",
            _confusion(cell_items, {"HIGH", "ELEVATED"})))
    if module_items:
        lines.append(_fmt_confusion(
            "Modules (NOK = bad)", _confusion(module_items, {"NOK"})))

    # ------------------------------------------------------------------
    # Separation power (rank AUC) per score
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("== Separation power (rank AUC; 0.5 = chance, 1.0 = perfect) ==")

    def _auc_line(name: str, extract) -> None:
        good = [v for it in items if it.label.outcome == "good"
                and (v := extract(it)) is not None]
        bad = [v for it in items if it.label.outcome == "bad"
               and (v := extract(it)) is not None]
        auc = _rank_auc(good, bad)
        if auc is not None:
            lines.append(f"  {name:24s} AUC={auc:.3f}  (n_good={len(good)}, n_bad={len(bad)})")

    _auc_line("composite_z", lambda it: it.composite_z)
    all_clusters = sorted({c for it in items for c in it.cluster_scores})
    for cluster in all_clusters:
        _auc_line(f"cluster:{cluster}", lambda it, c=cluster: it.cluster_scores.get(c))
    all_methods = sorted({m for it in items for m in it.method_z})
    for method in all_methods:
        _auc_line(f"method:{method}", lambda it, m=method: it.method_z.get(m))

    # ------------------------------------------------------------------
    # Threshold sweep over composite_z
    # ------------------------------------------------------------------
    if sweep:
        lines.append("")
        lines.append("== Threshold sweep (composite_z as sole HIGH gate) ==")
        scored = [(it.composite_z, it.label.outcome == "bad")
                  for it in items if it.composite_z is not None]
        thresholds = sorted({round(z, 3) for z, _ in scored})
        best = None
        lines.append(f"  {'threshold':>10s} {'precision':>10s} {'recall':>8s} {'F1':>6s}")
        for thr in thresholds:
            tp = sum(1 for z, bad in scored if z >= thr and bad)
            fp = sum(1 for z, bad in scored if z >= thr and not bad)
            fn = sum(1 for z, bad in scored if z < thr and bad)
            if tp + fp == 0 or tp + fn == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            lines.append(f"  {thr:>10.3f} {precision:>10.2f} {recall:>8.2f} {f1:>6.2f}")
            if best is None or f1 > best[1]:
                best = (thr, f1)
        if best:
            lines.append(f"  Suggested operating point (max F1): composite_z >= {best[0]:.3f}")

    return "\n".join(lines)
