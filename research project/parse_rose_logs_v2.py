#!/usr/bin/env python3
"""
parse_rose_logs_v2.py
=====================
Generalized parser for Rose solver logs. Walks ALL configuration directories
under ``Rose Logs/`` (e.g. ``auto_config_basic.json``, ``default_config 2.json``)
plus the original Ramsey-only ``rose_logs/`` directory from the midpoint
checkpoint, and consolidates every solver run into a single CSV
(``rose_logs_parsed_v2.csv``).

For each run folder the script reads:
  - status.json           - solution status, termination reason, objective, bounds
  - basic_metrics.json    - timelines, node counts, LP iterations, cuts
  - problem.json          - problem dimensions, constraint/variable types
  - detailed_metrics.json - cut-type breakdowns, heuristic stats, dive/branch
                            node classifications, time-slice breakdowns
  - bounds.json           - bound trajectory (summarised as count + final values)
  - stdout.txt[.gz]       - raw solver log (ANSI codes stripped)

Key differences from the original ``parse_rose_logs.py``:
  * Handles the new directory layout ``<root>/<config>.json/1/<problem>.mps/``
  * Reads gzipped ``stdout.txt.gz`` transparently
  * Adds ``config_name``, ``problem_name``, and ``run_id`` columns so paired
    runs (same problem under different configs) are joinable
  * Backward-compatible with the flat ``rose_logs/<run>/`` layout used for
    the Ramsey runs
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# --- helpers -----------------------------------------------------------------

# Map raw directory names to readable display names that capture the actual
# behavioral difference between the two solver configurations:
#   * auto_config_basic (staging build, +44% nodes)        -> branching_heavy
#   * default_config 2  (release build, +13% incumbents)   -> heuristic_heavy
# The ``rose_logs`` flat-layout subdirectory holds the 8 Ramsey runs from the
# midpoint checkpoint; we relabel it ``ramsey`` to distinguish problem source
# from solver configuration. Directory names are preserved on disk for
# provenance.
CONFIG_NAME_MAP = {
    "auto_config_basic": "branching_heavy",
    "default_config 2": "heuristic_heavy",
    "rose_logs": "ramsey",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from solver log text."""
    return ANSI_RE.sub("", text)


def safe_load_json(path: Path) -> dict:
    """Load a JSON file; return {} on missing/invalid content."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # missing files are common (e.g. detailed_metrics.json on quick runs)
        if not isinstance(e, FileNotFoundError):
            print(f"  warning: could not load {path}: {e}", file=sys.stderr)
        return {}


def read_stdout(run_dir: Path) -> str:
    """Read stdout.txt or stdout.txt.gz (whichever exists). Returns ''."""
    plain = run_dir / "stdout.txt"
    gz = run_dir / "stdout.txt.gz"
    try:
        if plain.exists():
            return plain.read_text(errors="replace")
        if gz.exists():
            with gzip.open(gz, "rt", errors="replace") as f:
                return f.read()
    except OSError as e:
        print(f"  warning: could not read stdout in {run_dir}: {e}", file=sys.stderr)
    return ""


# --- per-run extraction ------------------------------------------------------

def extract_stdout_features(stdout_text: str) -> dict:
    """Parse stdout text for features not in the JSON files."""
    features: dict = {"stdout_clean": ""}
    if not stdout_text:
        return features

    clean = strip_ansi(stdout_text)
    features["stdout_clean"] = clean
    features["stdout_num_lines"] = clean.count("\n") + 1
    features["stdout_num_chars"] = len(clean)

    # Root LP objective (e.g. "Root LP contest winner: highs ... objective: 0")
    m = re.search(r"Root LP contest.*?objective:\s*([\d.eE+\-inf]+)", clean)
    features["root_lp_objective"] = (
        float(m.group(1)) if m and "inf" not in m.group(1) else None
    )

    # Root LP winner method
    m = re.search(r"Root LP contest.*?winner:\s*(\w+)", clean)
    features["root_lp_winner"] = m.group(1) if m else None

    # Stages observed (contest, cuts, strongb, dive, branch)
    stages_found = set(re.findall(r"\b(contest|cuts|strongb|dive|branch)\b", clean))
    features["stages_observed"] = ",".join(sorted(stages_found)) if stages_found else None
    features["num_stages_observed"] = len(stages_found)

    # Original problem dimensions from head_client line
    m = re.search(r"Original:\s*(\d+)\s*x\s*(\d+),\s*nonz:\s*(\d+)", clean)
    if m:
        features["original_constraints"] = int(m.group(1))
        features["original_variables"] = int(m.group(2))
        features["original_nonzeros"] = int(m.group(3))

    # Presolved dimensions
    m = re.search(r"Presolved:\s*(\d+)\s*x\s*(\d+),\s*nonz:\s*(\d+)", clean)
    if m:
        features["presolved_constraints_log"] = int(m.group(1))
        features["presolved_variables_log"] = int(m.group(2))
        features["presolved_nonzeros_log"] = int(m.group(3))

    # Final status line
    m = re.search(r"Final\s+objective:\s*([\d.eE+\-naninf]+),\s*status:\s*(\w+)", clean)
    if m:
        features["final_status_log"] = m.group(2)
        try:
            features["final_objective_log"] = float(m.group(1))
        except ValueError:
            features["final_objective_log"] = None

    # Termination message
    m = re.search(r"Termination reason:\s*(\w+),\s*Message:\s*(.+)", clean)
    if m:
        features["termination_message"] = m.group(2).strip()

    # Solved-in-presolve marker
    features["solved_in_presolve"] = int(
        "Problem solved at the Head" in clean or "solved the problem locally" in clean
    )

    return features


def extract_bounds_summary(bounds_path: Path) -> dict:
    """Summarise the bound trajectory from bounds.json."""
    data = safe_load_json(bounds_path)
    summary: dict = {}
    if not data:
        return summary

    dual = data.get("dual_bound", []) or []
    primal = data.get("primal_bound", []) or []
    timestamps = data.get("timestamp", []) or []
    origins = data.get("origin", []) or []

    summary["bounds_num_updates"] = len(timestamps)

    dual_vals = [v for v in dual if v is not None]
    primal_vals = [v for v in primal if v is not None]
    summary["bounds_final_dual"] = dual_vals[-1] if dual_vals else None
    summary["bounds_final_primal"] = primal_vals[-1] if primal_vals else None
    summary["bounds_first_dual"] = dual_vals[0] if dual_vals else None
    summary["bounds_first_primal"] = primal_vals[0] if primal_vals else None

    # Final relative gap = |primal - dual| / max(|primal|, 1e-10)
    if dual_vals and primal_vals:
        p, d = primal_vals[-1], dual_vals[-1]
        denom = max(abs(p), 1e-10)
        summary["bounds_final_rel_gap"] = abs(p - d) / denom

    ts_vals = [t for t in timestamps if t is not None]
    if ts_vals:
        summary["bounds_first_timestamp"] = ts_vals[0]
        summary["bounds_last_timestamp"] = ts_vals[-1]
        summary["bounds_duration"] = ts_vals[-1] - ts_vals[0]

    if origins:
        origin_counts: dict = {}
        for o in origins:
            origin_counts[o] = origin_counts.get(o, 0) + 1
        for origin, count in origin_counts.items():
            summary[f"bounds_origin_{origin}_count"] = count

    return summary


def extract_detailed_metrics_subset(detailed_path: Path) -> dict:
    """Pull the most analytically useful fields from detailed_metrics.json."""
    data = safe_load_json(detailed_path)
    if not data:
        return {}

    keep_prefixes = [
        "cut_gomory_", "cut_cover_", "cut_mir_", "cut_strong_cg_",
        "heuristic_",
        "dive_",
        "domainprop_",
        "nodes_branched", "nodes_cut",
        "nodes_pre_lp_", "nodes_post_lp_",
        "first_time_feasible_seconds",
        "first_time_gap_1_percent", "first_time_gap_2_percent",
        "first_time_gap_5_percent", "first_time_gap_10_percent",
        "timeslice_burl_branch_and_bound",
        "timeslice_burl_cuts_seconds",
        "timeslice_burl_dive_seconds",
        "timeslice_burl_heuristics_seconds",
        "timeslice_burl_solve_seconds",
        "timeslice_burl_strong_branch_seconds",
        "timeslice_burl_total_seconds",
    ]

    subset = {}
    for k, v in data.items():
        if any(k.startswith(p) or k == p for p in keep_prefixes):
            if "relaxed_validation" in k:
                continue
            # Skip lists - they would need their own column expansion.
            if isinstance(v, list):
                subset[f"detail_{k}_len"] = len(v)
                continue
            if isinstance(v, dict):
                continue
            subset[f"detail_{k}"] = v
    return subset


def parse_single_run(run_dir: Path, config_name: str) -> dict:
    """Parse all data files for a single solver run into a flat dict."""
    row: dict = {}
    run_name = run_dir.name

    # naming
    problem_name = run_name[:-4] if run_name.endswith(".mps") else run_name
    row["config_name"] = config_name
    row["problem_name"] = problem_name
    row["run_name"] = run_name
    row["run_id"] = f"{config_name}::{problem_name}"

    # Ramsey-style parameters (only present for ramsey_* runs)
    m = re.match(r"ramsey_r(\d+)_s(\d+)_n(\d+)", run_name)
    if m:
        row["ramsey_r"] = int(m.group(1))
        row["ramsey_s"] = int(m.group(2))
        row["ramsey_n"] = int(m.group(3))

    # 1. status.json
    status = safe_load_json(run_dir / "status.json")
    for k, v in status.items():
        row[f"status_{k}"] = v

    # 2. basic_metrics.json
    metrics = safe_load_json(run_dir / "basic_metrics.json")
    for k, v in metrics.items():
        row[f"metric_{k}"] = v

    # 3. problem.json
    problem = safe_load_json(run_dir / "problem.json")
    for k, v in problem.items():
        row[f"problem_{k}"] = v

    # 4. detailed_metrics.json (curated subset)
    row.update(extract_detailed_metrics_subset(run_dir / "detailed_metrics.json"))

    # 5. bounds.json (summary)
    row.update(extract_bounds_summary(run_dir / "bounds.json"))

    # 6. stdout.txt[.gz] (parsed features + cleaned text)
    stdout_text = read_stdout(run_dir)
    row.update(extract_stdout_features(stdout_text))

    # ---- derived features useful for ML / labeling ------------------------
    total_time = metrics.get("timeline_total_seconds", 0) or 0
    solve_time = metrics.get("timeline_solve_seconds", 0) or 0
    presolve_time = metrics.get("timeline_presolve_seconds", 0) or 0
    nodes_total = metrics.get("num_nodes_total", 0) or 0
    cuts_gen = metrics.get("num_cuts_generated", 0) or 0
    cuts_app = metrics.get("num_cuts_applied", 0) or 0
    n_vars = problem.get("num_variables", 0) or 0
    n_cons = problem.get("num_constraints", 0) or 0
    n_pre_vars = problem.get("num_presolved_variables", 0) or 0
    n_pre_cons = problem.get("num_presolved_constraints", 0) or 0

    row["derived_solve_fraction"] = solve_time / total_time if total_time > 0 else None
    row["derived_presolve_fraction"] = presolve_time / total_time if total_time > 0 else None
    row["derived_cut_efficiency"] = cuts_app / cuts_gen if cuts_gen > 0 else None
    row["derived_presolve_var_reduction"] = (
        1.0 - n_pre_vars / n_vars if n_vars > 0 else None
    )
    row["derived_presolve_con_reduction"] = (
        1.0 - n_pre_cons / n_cons if n_cons > 0 else None
    )
    row["derived_nodes_per_second"] = (
        nodes_total / solve_time if solve_time > 0 else None
    )
    row["derived_lp_iter_per_var"] = (
        (metrics.get("num_lp_iterations") or 0) / n_vars if n_vars > 0 else None
    )
    row["derived_is_optimal"] = int(status.get("solution_status") == "optimal")
    row["derived_has_solution"] = int(status.get("solution_status") != "no_solution")
    row["derived_timed_out"] = int(status.get("termination_reason") == "timeout")

    return row


# --- discovery ---------------------------------------------------------------

def discover_runs(rose_logs_root: Path) -> Iterable[tuple[str, Path]]:
    """
    Walk a Rose Logs root directory and yield (config_name, run_dir) pairs.

    Supports two layouts:
      * <root>/<config>.json/1/<problem>.mps/   (new MIPLIB layout)
      * <root>/<run_name>/                       (flat Ramsey layout)
    """
    if not rose_logs_root.is_dir():
        return

    for entry in sorted(rose_logs_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        # Detect new layout: <config>.json/<id>/<problem>.mps
        # The config dir name typically ends with ".json" but we also accept
        # any directory whose immediate children are numeric run-id folders
        # containing problem.mps directories.
        nested_ids = [d for d in entry.iterdir() if d.is_dir()]
        looks_like_config = (
            entry.name.endswith(".json")
            or all(re.fullmatch(r"\d+", d.name) for d in nested_ids if not d.name.startswith("."))
        )

        if looks_like_config and nested_ids:
            raw_name = entry.name.removesuffix(".json").strip()
            config_name = CONFIG_NAME_MAP.get(raw_name, raw_name)
            for run_id_dir in sorted(nested_ids):
                if run_id_dir.name.startswith("."):
                    continue
                for problem_dir in sorted(run_id_dir.iterdir()):
                    if problem_dir.is_dir() and not problem_dir.name.startswith("."):
                        # require at least one of the JSONs to exist
                        if (problem_dir / "status.json").exists():
                            yield config_name, problem_dir
            continue

        # Flat layout: assume each subdir is a run, and the parent dir name
        # is the "config" (use a generic 'flat' tag if the parent looks like
        # the rose_logs root itself).
        if (entry / "status.json").exists():
            raw_name = rose_logs_root.name.removesuffix(".json").strip() or "flat"
            config_name = CONFIG_NAME_MAP.get(raw_name, raw_name)
            yield config_name, entry


# --- main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more directories to walk for Rose log runs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Drop stdout_clean column from output to keep the CSV small.",
    )
    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.roots]
    output_csv = Path(args.output).resolve()

    runs: list[tuple[str, Path]] = []
    for root in roots:
        if not root.exists():
            print(f"warning: root not found: {root}", file=sys.stderr)
            continue
        n_before = len(runs)
        runs.extend(discover_runs(root))
        print(f"Discovered {len(runs) - n_before} run(s) under {root}")

    if not runs:
        print("Error: no runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nParsing {len(runs)} runs total ...")
    rows = []
    for i, (config_name, run_dir) in enumerate(runs, 1):
        if i == 1 or i % 25 == 0 or i == len(runs):
            print(f"  [{i:>3}/{len(runs)}] {config_name} :: {run_dir.name}")
        try:
            row = parse_single_run(run_dir, config_name)
            if args.no_stdout:
                row.pop("stdout_clean", None)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR parsing {run_dir}: {e}", file=sys.stderr)

    # Stable column order: identifying columns first, then everything else
    leading = [
        "run_id", "config_name", "problem_name", "run_name",
        "ramsey_r", "ramsey_s", "ramsey_n",
    ]
    seen = set(leading)
    all_keys = list(leading)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows x {len(all_keys)} columns to {output_csv}")

    # Summarise column groups for sanity
    groups: dict = {}
    for k in all_keys:
        prefix = k.split("_")[0]
        groups.setdefault(prefix, []).append(k)
    print("\nColumn groups:")
    for prefix, cols in sorted(groups.items()):
        print(f"  {prefix:14s}: {len(cols):>3d}")


if __name__ == "__main__":
    main()
