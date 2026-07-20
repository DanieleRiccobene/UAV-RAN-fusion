#!/usr/bin/env python3
"""UAV xApp testbed — episodic training + N-campaign + comparison plots.

Replaces the ns-3 / fidelity / sliding-window orchestrator. See docs/REFACTOR_PLAN.md.

Examples:
  # single run
  python3 scripts/train_uav.py --agents clara --num-uavs 5 --num-episodes 100
  # full campaign, all agents (dqn/ppo need torch)
  python3 scripts/train_uav.py --agents clara,clara_diff,ucb,dqn,ppo --campaign
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from rl.uav_env import UAVEnv
from rl.agents.registry import build_agent
from rl.train_loop import run_training
from rl.plots import plot_campaign

DEFAULT_CSV = os.path.join(REPO_ROOT, "mobility_data", "ind_mob", "individual_peak_20users_1sec_15min.csv")

# Agent token -> (registry name, env credit scheme override or None), family
AGENT_SPECS = {
    "clara":      ("clara", "shared_off_penalty", "MAB"),
    "clara_diff": ("clara", "difference",         "MAB"),
    "ucb":        ("ucb",   None,                 "MAB"),
    "random":     ("random", None,                "MAB"),
    "oracle":     ("oracle", None,                "ORACLE"),
    "dqn":        ("dqn",   None,                 "DRL"),
    "ppo":        ("ppo",   None,                 "DRL"),
    # OREO Table 1 baselines (RU-activation-only, no user association).
    "el_amine":   ("el_amine", None,              "RL"),   # [20] tabular Q-learning
    "xu":         ("xu",       None,              "RL"),   # [18] hierarchical RL
    "rezaei":     ("rezaei",   None,              "DRL"),  # [19] multi-agent DQN
    "masrur":     ("masrur",   None,              "DRL"),  # [16] MARL Double-DQN
    "wang":       ("wang",     None,              "DRL"),  # [12] DQN xApp (paper hyperparams)
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", default="clara", help="comma list: " + ",".join(AGENT_SPECS))
    p.add_argument("--num-uavs", type=int, default=5)
    p.add_argument("--campaign", action="store_true", help="loop N in 3,5,7,9")
    p.add_argument("--campaign-uavs", default="3,5,7,9")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--max-ues-per-uav", type=int, default=10)
    p.add_argument("--max-users", type=int, default=20)
    p.add_argument("--mobility-csv", default=DEFAULT_CSV)
    p.add_argument("--per-ue-throughput-mbps", type=float, default=10.0)
    # Traffic / QoS scenario (per-UE downlink demand). Choose one.
    p.add_argument("--traffic-scenario", choices=["full_rate", "split_half", "split_20_80"],
                   default="full_rate")
    p.add_argument("--full-rate", dest="traffic_scenario", action="store_const", const="full_rate",
                   help="all UEs demand 10 Mbps DL (default)")
    p.add_argument("--split-half", dest="traffic_scenario", action="store_const", const="split_half",
                   help="50%% at 14 / 50%% at 6 Mbps (avg 10 — same total load as full_rate)")
    p.add_argument("--split-2080", dest="traffic_scenario", action="store_const", const="split_20_80",
                   help="20%% at 20 / 80%% at 7.5 Mbps (avg 10 — same total load as full_rate)")
    p.add_argument("--uav-capacity-mbps", type=float, default=None,
                   help="UAV total serving capacity (Mbps); default = K * per-ue rate")
    p.add_argument("--off-penalty", type=float, default=50.0, help="gamma' (CLARA scheme-B)")
    p.add_argument("--w-energy", type=float, default=0.001)
    p.add_argument("--w-switch", type=float, default=0.1)
    p.add_argument("--w-rlf", type=float, default=1.0)
    p.add_argument("--time-factor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "uav"))
    p.add_argument("--append-to", default=None,
                   help="existing results folder: write these agents' CSVs into it and "
                        "re-plot ALL agents (previous + new) together. Match the scenario "
                        "and config of that folder's earlier run.")
    p.add_argument("--record-steps", dest="record_steps", action="store_true", default=True,
                   help="write per-step CSVs; CDFs use instantaneous per-step values (default)")
    p.add_argument("--no-record-steps", dest="record_steps", action="store_false",
                   help="skip per-step CSVs; CDFs fall back to per-episode values")
    return p.parse_args()


def build_env(args, num_uavs, credit_scheme):
    return UAVEnv(
        mobility_csv=args.mobility_csv,
        num_uavs=num_uavs,
        max_ues_per_uav=args.max_ues_per_uav,
        max_users=args.max_users,
        per_ue_throughput_mbps=args.per_ue_throughput_mbps,
        traffic_scenario=args.traffic_scenario,
        uav_capacity_mbps=args.uav_capacity_mbps,
        off_penalty=args.off_penalty,
        w_energy=args.w_energy,
        w_switch=args.w_switch,
        w_rlf=args.w_rlf,
        time_factor=args.time_factor,
        credit_scheme=credit_scheme or "shared_off_penalty",
        seed=args.seed,
    )


def write_history_csv(path, history):
    if not history:
        return
    fieldnames = ["episode"] + [k for k in history[0] if k != "episode"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in history:
            w.writerow(row)


def write_steps_csv(path, step_rows):
    if not step_rows:
        return
    fieldnames = ["episode", "step"] + [k for k in step_rows[0] if k not in ("episode", "step")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in step_rows:
            w.writerow(row)


def run_one(args, num_uavs, token):
    name, credit, family = AGENT_SPECS[token]
    env = build_env(args, num_uavs, credit)
    agent = build_agent(name, env, seed=args.seed)
    print(f"  [{token}] N={num_uavs} agent={name} family={family} "
          f"episodes={args.num_episodes} steps/ep={len(env)} ...", flush=True)
    history, steps = run_training(env, agent, args.num_episodes, collect_steps=args.record_steps)
    return history, steps, family




def main():
    args = parse_args()
    tokens = [t.strip() for t in args.agents.split(",") if t.strip()]
    for t in tokens:
        if t not in AGENT_SPECS:
            raise SystemExit(f"Unknown agent {t!r}; choose from {list(AGENT_SPECS)}")
    uav_counts = ([int(x) for x in args.campaign_uavs.split(",")] if args.campaign else [args.num_uavs])

    scenario = args.traffic_scenario
    if args.append_to:
        out_dir = args.append_to
        if not os.path.isdir(out_dir):
            raise SystemExit(f"--append-to folder not found: {out_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        # Scenario in the path so parallel runs (one per scenario) never collide.
        out_dir = os.path.join(args.output_dir, f"{scenario}_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
    print(f"output: {out_dir}\nscenario: {scenario}\nagents: {tokens}\nN: {uav_counts}"
          f"{'  (append mode)' if args.append_to else ''}", flush=True)

    results = {}
    step_results = {}
    for num_uavs in uav_counts:
        results[num_uavs] = {}
        step_results[num_uavs] = {}
        for token in tokens:
            try:
                history, steps, family = run_one(args, num_uavs, token)
            except Exception as exc:
                print(f"  [{token}] N={num_uavs} FAILED: {exc}", flush=True)
                continue
            write_history_csv(os.path.join(out_dir, f"N{num_uavs}_{scenario}_{token}.csv"), history)
            if steps:
                write_steps_csv(os.path.join(out_dir, f"N{num_uavs}_{scenario}_{token}_steps.csv"), steps)
                step_results[num_uavs][token] = steps
            results[num_uavs][token] = (history, family)
            last = history[-1]
            print(f"    done: return_cmp={last['return_cmp']:.1f} "
                  f"throughput={last['mean_throughput_mbps']:.1f} "
                  f"rlf={last['mean_rlf']:.2f} mean_active={last['mean_active_uavs']:.2f}", flush=True)

    if args.append_to:
        # Re-plot the WHOLE folder from CSVs so previous + new agents appear together.
        from plot_uav_results import plot_folder
        plot_folder(out_dir)
    else:
        plot_campaign(out_dir, results, scenario, step_results=step_results)
    print("Campaign complete.", flush=True)


if __name__ == "__main__":
    main()
