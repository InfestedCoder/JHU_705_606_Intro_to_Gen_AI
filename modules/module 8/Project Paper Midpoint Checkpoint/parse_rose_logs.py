#!/usr/bin/env python3
"""
parse_rose_logs.py
==================
Walks through the rose_logs/ directory and consolidates solver run data into
a single CSV file (rose_logs_parsed.csv) suitable for analysis in the
Ginn-Project-Notebook.

For each run folder the script reads:
  - status.json       – solution status, termination reason, objective, bounds
  - basic_metrics.json – timelines, node counts, LP iterations, cuts
  - problem.json       – problem dimensions, constraint/variable types
  - detailed_metrics.json – cut-type breakdowns, heuristic stats, dive/branch
                            node classifications, time-slice breakdowns
  - bounds.json        – bound trajectory (summarised as count + final values)
  - stdout.txt         – raw solver log (ANSI codes stripped)

Output: one row per run, all fields flattened.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from solver log text."""
    return ANSI_RE.sub("", text)


def safe_load_json(path: Path) -> dict:
    """Load a JSON file; return empty dict on any error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: could not load {path.name}: {e}", file=sys.stderr)
        return {}


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a dict, prepending *prefix* to every key."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=f"{key}_"))
        elif isinstance(v, list):
            # Store list length; skip embedding full arrays (except bounds)
            pass
        else:
            out[key] = v
    return out


# ── per-run extraction ───────────────────────────────────────────────────────

def extract_stdout_features(stdout_path: Path) -> dict:
    """
    Parse stdout.txt for features not available in the JSON files:
      - cleaned log text (ANSI stripped)
      - root LP objective (from contest winner line)
      - solver stages observed
      - final objective and status from the last log line
      - problem dimensions from the 'Original:' line
    """
    features: dict = {}
    if not stdout_path.exists():
        features["stdout_clean"] = ""
        return features

    raw = stdout_path.read_text(errors="replace")
    clean = strip_ansi(raw)
    features["stdout_clean"] = clean

    # Root LP objective  (e.g. "Root LP contest winner: highs, ... objective: 0")
    m = re.search(r"Root LP contest.*?objective:\s*([\d.eE+\-inf]+)", clean)
    features["root_lp_objective"] = float(m.group(1)) if m and "inf" not in m.group(1) else None

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
        obj_str = m.group(1)
        try:
            features["final_objective_log"] = float(obj_str)
        except ValueError:
            features["final_objective_log"] = None

    # Termination message
    m = re.search(r"Termination reason:\s*(\w+),\s*Message:\s*(.+)", clean)
    if m:
        features["termination_message"] = m.group(2).strip()

    # Whether problem was solved in presolve (head)
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

    dual = data.get("dual_bound", [])
    primal = data.get("primal_bound", [])
    timestamps = data.get("timestamp", [])
    origins = data.get("origin", [])

    summary["bounds_num_updates"] = len(timestamps)

    # Final non-null dual/primal bounds
    dual_vals = [v for v in dual if v is not None]
    primal_vals = [v for v in primal if v is not None]
    summary["bounds_final_dual"] = dual_vals[-1] if dual_vals else None
    summary["bounds_final_primal"] = primal_vals[-1] if primal_vals else None

    # Time span of bound updates
    ts_vals = [t for t in timestamps if t is not None]
    if ts_vals:
        summary["bounds_first_timestamp"] = ts_vals[0]
        summary["bounds_last_timestamp"] = ts_vals[-1]
        summary["bounds_duration"] = ts_vals[-1] - ts_vals[0]

    # Origins summary
    if origins:
        origin_counts = {}
        for o in origins:
            origin_counts[o] = origin_counts.get(o, 0) + 1
        for origin, count in origin_counts.items():
            summary[f"bounds_origin_{origin}_count"] = count

    return summary


def extract_detailed_metrics_subset(detailed_path: Path) -> dict:
    """
    Pull the most analytically useful fields from detailed_metrics.json.
    We skip the low-level comm/load-balance fields and focus on:
      - cut type breakdowns (gomory, cover, mir, strong_cg)
      - heuristic stats
      - dive stats
      - domain propagation
      - node classification
      - time-to-gap milestones
      - key timeslice breakdowns
    """
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
            # Skip the relaxed_validation fields to reduce noise
            if "relaxed_validation" in k:
                continue
            subset[f"detail_{k}"] = v

    return subset


def parse_single_run(run_dir: Path) -> dict:
    """Parse all data files for a single solver run into a flat dict."""
    row: dict = {}
    run_name = run_dir.name
    row["run_name"] = run_name

    # Parse Ramsey parameters from folder name (ramsey_r{r}_s{s}_n{n})
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
    detailed = extract_detailed_metrics_subset(run_dir / "detailed_metrics.json")
    row.update(detailed)

    # 5. bounds.json (summary)
    bounds = extract_bounds_summary(run_dir / "bounds.json")
    row.update(bounds)

    # 6. stdout.txt (parsed features + cleaned text)
    stdout_features = extract_stdout_features(run_dir / "stdout.txt")
    row.update(stdout_features)

    # Derived features useful for ML
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
    row["derived_is_optimal"] = int(status.get("solution_status") == "optimal")
    row["derived_has_solution"] = int(status.get("solution_status") != "no_solution")

    return row


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Determine paths
    script_dir = Path(__file__).resolve().parent
    rose_logs_dir = script_dir / "rose_logs"

    if not rose_logs_dir.is_dir():
        print(f"Error: rose_logs directory not found at {rose_logs_dir}", file=sys.stderr)
        sys.exit(1)

    output_csv = script_dir / "rose_logs_parsed.csv"

    # Discover run folders (sorted for reproducibility)
    run_dirs = sorted([
        d for d in rose_logs_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not run_dirs:
        print("Error: no run folders found in rose_logs/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run folder(s) in {rose_logs_dir}")

    # Parse all runs
    rows = []
    for run_dir in run_dirs:
        print(f"  Parsing {run_dir.name} ...")
        row = parse_single_run(run_dir)
        rows.append(row)

    # Collect all column names (union of all rows) preserving insertion order
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows x {len(all_keys)} columns to {output_csv}")
    print(f"Column groups:")

    # Summarize column groups
    groups = {}
    for k in all_keys:
        prefix = k.split("_")[0]
        groups.setdefault(prefix, []).append(k)
    for prefix, cols in sorted(groups.items()):
        print(f"  {prefix:12s}: {len(cols):3d} columns")


if __name__ == "__main__":
    main()
