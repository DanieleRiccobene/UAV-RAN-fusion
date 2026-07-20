#!/usr/bin/env python3
"""Regenerate UAV campaign plots (time-series + CDFs) from a results folder's CSVs.

Usage:
  # explicit folder(s)
  python3 scripts/plot_uav_results.py outputs/uav/split_20_80_2026-07-06_16-13-02
  # no args: auto-pick the latest folder per scenario under outputs/uav/
  python3 scripts/plot_uav_results.py
"""

import csv
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl.plots import plot_campaign

SCENARIOS = ("full_rate", "split_half", "split_20_80")
TOKEN_FAMILY = {
    "clara": "MAB", "clara_diff": "MAB", "ucb": "MAB", "random": "MAB",
    "oracle": "ORACLE", "dqn": "DRL", "ppo": "DRL",
    "el_amine": "RL", "xu": "RL", "rezaei": "DRL", "masrur": "DRL", "wang": "DRL",
}
# Longest tokens first so 'clara_diff' matches before 'clara'.
TOKENS = sorted(TOKEN_FAMILY, key=len, reverse=True)


def parse_csv_name(basename):
    """N{n}_{scenario}_{token}.csv -> (num_uavs, scenario, token) or None."""
    if not basename.startswith("N") or not basename.endswith(".csv"):
        return None
    stem = basename[1:-4]  # drop leading 'N' and '.csv'
    n_str, _, rest = stem.partition("_")
    try:
        num_uavs = int(n_str)
    except ValueError:
        return None
    for token in TOKENS:
        if rest.endswith("_" + token):
            scenario = rest[: -(len(token) + 1)]
            return num_uavs, scenario, token
    return None


def load_history(path):
    history = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for k, v in row.items():
                if k == "episode":
                    rec[k] = int(float(v))
                else:
                    try:
                        rec[k] = float(v)
                    except (TypeError, ValueError):
                        rec[k] = v
            history.append(rec)
    history.sort(key=lambda r: r["episode"])
    return history


def plot_folder(folder):
    by_scenario = {}       # scenario -> {num_uavs: {token: (history, family)}}
    steps_by_scenario = {}  # scenario -> {num_uavs: {token: [per-step rows]}}
    for path in glob.glob(os.path.join(folder, "N*_*.csv")):
        bn = os.path.basename(path)
        if bn.endswith("_steps.csv"):
            parsed = parse_csv_name(bn[: -len("_steps.csv")] + ".csv")
            if parsed is None:
                continue
            num_uavs, scenario, token = parsed
            rows = load_history(path)  # generic numeric loader
            if rows:
                steps_by_scenario.setdefault(scenario, {}).setdefault(num_uavs, {})[token] = rows
            continue
        parsed = parse_csv_name(bn)
        if parsed is None:
            continue
        num_uavs, scenario, token = parsed
        history = load_history(path)
        if not history:
            continue
        by_scenario.setdefault(scenario, {}).setdefault(num_uavs, {})[token] = (
            history, TOKEN_FAMILY.get(token, "MAB")
        )
    if not by_scenario:
        print(f"[skip] no campaign CSVs found in {folder}", flush=True)
        return
    for scenario, results in by_scenario.items():
        step_results = steps_by_scenario.get(scenario)
        print(f"[plot] {folder}  scenario={scenario}  "
              f"N={sorted(results)}  agents={sorted({t for pt in results.values() for t in pt})}"
              f"{'  (per-step CDFs)' if step_results else ''}",
              flush=True)
        plot_campaign(folder, results, scenario, step_results=step_results)


def latest_per_scenario(root):
    picks = []
    for scenario in SCENARIOS:
        cands = sorted(glob.glob(os.path.join(root, f"{scenario}_*")), reverse=True)
        cands = [c for c in cands if os.path.isdir(c)]
        if cands:
            picks.append(cands[0])
    return picks


def main():
    folders = sys.argv[1:]
    if not folders:
        folders = latest_per_scenario(os.path.join(REPO_ROOT, "outputs", "uav"))
        if not folders:
            raise SystemExit("No results folders found under outputs/uav/; pass a folder explicitly.")
    for folder in folders:
        plot_folder(folder)


if __name__ == "__main__":
    main()
