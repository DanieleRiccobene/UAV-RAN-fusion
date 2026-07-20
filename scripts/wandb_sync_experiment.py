#!/usr/bin/env python3
import argparse
import csv
import json
import os
import math


PLOT_FILENAMES = [
    "episodic_reward.png",
    "mean_throughput.png",
    "mean_outage_count.png",
    "mean_energy_cost.png",
    "mean_active_gnb_count.png",
    "mean_current_power_w.png",
    "final_cumulative_energy_kwh.png",
    "final_topology_60s.png",
]

MORABITO_PLOT_FILENAMES = [
    "morabito_reward.png",
    "morabito_qosflow.png",
    "morabito_tb_totnbrdl.png",
    "morabito_rlf.png",
    "morabito_es_on_cost.png",
    "morabito_zero_count.png",
    "morabito_crashed.png",
    "morabito_done.png",
    "morabito_aggregate_throughput_mbps.png",
    "morabito_normalized_throughput.png",
    "morabito_connected_ues.png",
    "morabito_disconnected_ues.png",
    "morabito_reward_components_stacked.png",
    "reward_internal_vs_morabito.png",
    "morabito_throughput_cdf.png",
    "morabito_rlf_cdf.png",
    "morabito_activation_cost_cdf.png",
]

CONTROL_PLOT_FILENAMES = [
    "gnb_active_count_timeline.png",
    "gnb_activation_heatmap.png",
    "mab_external_reward_timeline.png",
    "mab_selected_arm_timeline.png",
    "mab_arm_mean_reward_timeline.png",
]


def log_csv_artifact(wandb, run, name, path):
    if not path or not os.path.exists(path):
        return
    artifact = wandb.Artifact(name=name, type="dataset")
    artifact.add_file(path)
    run.log_artifact(artifact)


def maybe_log_images(wandb, paths, *, step, prefix):
    images = {}
    for path in paths:
        if path and os.path.exists(path):
            images[f"{prefix}/{os.path.splitext(os.path.basename(path))[0]}"] = wandb.Image(path)
    if images:
        wandb.log(images, step=step, commit=False)


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "nan", "null"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def compact_numeric_metrics(metrics):
    compacted = {}
    for key, value in metrics.items():
        parsed = parse_optional_float(value)
        if parsed is not None:
            compacted[key] = parsed
    return compacted


def _find_matching_row(rows, *, window_index, request_id=None):
    if not rows:
        return None
    window_text = str(int(window_index))
    request_text = str(request_id) if request_id is not None else None
    for row in reversed(rows):
        if request_text and str(row.get("request_id") or "") == request_text:
            return row
    for row in reversed(rows):
        row_window = row.get("window_index")
        if row_window is None:
            continue
        try:
            if str(int(float(row_window))) == window_text:
                return row
        except (TypeError, ValueError):
            continue
    return rows[-1]


def _weighted_reward_components_from_feedback_row(row):
    reward_mode = str(row.get("reward_mode") or "throughput_active_gnb")
    alpha = parse_optional_float(row.get("reward_alpha")) or 0.0
    beta = parse_optional_float(row.get("reward_beta")) or 0.0
    gamma = parse_optional_float(row.get("reward_gamma")) or 0.0
    weighted_throughput = parse_optional_float(row.get("reward_component_throughput_weighted"))
    weighted_active = parse_optional_float(row.get("reward_component_active_gnb_weighted"))
    weighted_disconnected = parse_optional_float(row.get("reward_component_disconnected_ues_weighted"))
    if weighted_throughput is not None and weighted_active is not None and weighted_disconnected is not None:
        return {
            "throughput": weighted_throughput,
            "active_gnb_count": weighted_active,
            "disconnected_ues": weighted_disconnected,
        }
    if reward_mode == "raw_mbps":
        weighted_throughput = alpha * (parse_optional_float(row.get("aggregate_throughput_mbps")) or 0.0)
        weighted_active = beta * (parse_optional_float(row.get("active_gnb_count")) or 0.0)
        weighted_disconnected = gamma * (parse_optional_float(row.get("disconnected_ues")) or 0.0)
    else:
        weighted_throughput = alpha * (parse_optional_float(row.get("normalized_throughput")) or 0.0)
        weighted_active = beta * (parse_optional_float(row.get("normalized_active_gnb_count")) or 0.0)
        weighted_disconnected = gamma * (parse_optional_float(row.get("normalized_disconnected_ues")) or 0.0)
    return {
        "throughput": weighted_throughput,
        "active_gnb_count": weighted_active,
        "disconnected_ues": weighted_disconnected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-json", required=True)
    args = parser.parse_args()

    with open(args.payload_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(payload["wandb_state_json"], encoding="utf-8") as handle:
        state = json.load(handle)

    import wandb

    window = payload["window"]
    window_index = int(window["window_index"])
    request_id = f"window-{window_index:04d}"

    run = wandb.init(
        project=state["project"],
        entity=state["entity"],
        id=state["run_id"],
        name=state["run_name"],
        resume="allow",
        reinit=True,
        mode=state.get("mode", "online"),
        config={
            "experiment_root": state.get("experiment_root"),
        },
    )
    run.define_metric("external/ns3_step")
    run.define_metric("external/*", step_metric="external/ns3_step")
    run.define_metric("mab/*", step_metric="external/ns3_step")
    run.define_metric("control/*", step_metric="external/ns3_step")
    # Per-fidelity metrics use their own step axis so HIGH/MEDIUM/LOW charts
    # are directly comparable without gaps from other arms being interleaved.
    for _fid in ("high", "medium", "low"):
        run.define_metric(f"internal/{_fid}/step")
        run.define_metric(f"internal/{_fid}/*", step_metric=f"internal/{_fid}/step")

    fidelity_level = str(window.get("fidelity_level") or "unknown").lower()

    training_csv = payload.get("training_csv")
    next_internal_step = int(state.get("next_internal_step", 0))
    # Per-fidelity step counters stored in state so each arm's x-axis is contiguous.
    fid_step_key = f"next_internal_step_{fidelity_level}"
    next_fidelity_step = int(state.get(fid_step_key, 0))
    if training_csv and os.path.exists(training_csv):
        with open(training_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            episode = int(float(row["episode"]))
            global_step = next_internal_step
            next_internal_step += 1
            fidelity_step = next_fidelity_step
            next_fidelity_step += 1

            # Combined view (all fidelities on the same timeline).
            combined_metrics = compact_numeric_metrics(
                {
                    "internal/window_index": window_index,
                    "internal/episode": episode,
                    "internal/episode_reward": row.get("episode_reward"),
                    "internal/mean_throughput": row.get("mean_throughput_mbps", row.get("mean_throughput")),
                    "internal/mean_throughput_bps": row.get("mean_throughput"),
                    "internal/mean_throughput_mbps": row.get("mean_throughput_mbps"),
                    "internal/normalized_mean_throughput": row.get("normalized_mean_throughput"),
                    "internal/mean_outage_count": row.get("mean_outage_count"),
                    "internal/mean_energy_cost": row.get("mean_energy_cost"),
                    "internal/mean_active_gnb_count": row.get("mean_active_gnb_count"),
                    "internal/mean_current_power_w": row.get("mean_current_power_w"),
                    "internal/final_cumulative_energy_kwh": row.get("final_cumulative_energy_kwh"),
                    "internal/epsilon": row.get("epsilon"),
                }
            )
            if combined_metrics:
                wandb.log(combined_metrics, step=global_step, commit=False)

            # Per-fidelity view (contiguous x-axis, no gaps from other arms).
            fid_metrics = compact_numeric_metrics(
                {
                    f"internal/{fidelity_level}/step": fidelity_step,
                    f"internal/{fidelity_level}/window_index": window_index,
                    f"internal/{fidelity_level}/episode": episode,
                    f"internal/{fidelity_level}/episode_reward": row.get("episode_reward"),
                    f"internal/{fidelity_level}/mean_throughput_mbps": row.get("mean_throughput_mbps", row.get("mean_throughput")),
                    f"internal/{fidelity_level}/normalized_mean_throughput": row.get("normalized_mean_throughput"),
                    f"internal/{fidelity_level}/mean_outage_count": row.get("mean_outage_count"),
                    f"internal/{fidelity_level}/mean_energy_cost": row.get("mean_energy_cost"),
                    f"internal/{fidelity_level}/mean_active_gnb_count": row.get("mean_active_gnb_count"),
                    f"internal/{fidelity_level}/epsilon": row.get("epsilon"),
                }
            )
            if fid_metrics:
                wandb.log(fid_metrics, step=global_step, commit=False)

        log_csv_artifact(wandb, run, f"training_metrics_window_{window_index:04d}", training_csv)
        state["next_internal_step"] = next_internal_step
        state[fid_step_key] = next_fidelity_step
    window_log_step = max(int(state.get("next_window_step", 0)), next_internal_step)

    internal_plot_paths = payload.get("internal_plot_paths") or []
    maybe_log_images(wandb, internal_plot_paths, step=window_log_step, prefix="internal/plots")
    maybe_log_images(wandb, internal_plot_paths, step=window_log_step, prefix=f"internal/{fidelity_level}/plots")
    final_topology_plot_path = payload.get("final_topology_plot_path")
    if final_topology_plot_path and os.path.exists(final_topology_plot_path):
        wandb.log({"internal/plots/final_topology": wandb.Image(final_topology_plot_path)}, step=window_log_step, commit=False)
        wandb.log({f"internal/{fidelity_level}/plots/final_topology": wandb.Image(final_topology_plot_path)}, step=window_log_step, commit=False)

    gnb_timeline_csv = payload.get("gnb_timeline_csv")
    if gnb_timeline_csv and os.path.exists(gnb_timeline_csv):
        with open(gnb_timeline_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = _find_matching_row(rows, window_index=window_index)
            ns3_step = parse_optional_float(row.get("window_index"))
            if ns3_step is None:
                ns3_step = window_index
            gnb_metrics = compact_numeric_metrics(
                {
                    "external/ns3_step": ns3_step,
                    "control/window_index": row.get("window_index"),
                    "control/active_gnb_count": row.get("active_gnb_count"),
                }
            )
            for key, value in row.items():
                if key.startswith("BS_"):
                    parsed = parse_optional_float(value)
                    if parsed is not None:
                        gnb_metrics[f"control/gnb_state/{key}"] = parsed
            if gnb_metrics:
                wandb.log(gnb_metrics, step=window_log_step, commit=False)
        log_csv_artifact(wandb, run, "gnb_activation_timeline", gnb_timeline_csv)
        plot_dir = payload.get("gnb_timeline_plot_dir")
        if plot_dir:
            maybe_log_images(
                wandb,
                [os.path.join(plot_dir, name) for name in CONTROL_PLOT_FILENAMES if name.startswith("gnb_")],
                step=window_log_step,
                prefix="control/plots",
            )

    morabito_feedback_csv = payload.get("morabito_feedback_csv")
    if morabito_feedback_csv and os.path.exists(morabito_feedback_csv):
        with open(morabito_feedback_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = _find_matching_row(rows, window_index=window_index, request_id=request_id)
            ns3_step = parse_optional_float(row.get("window_index"))
            if ns3_step is None:
                ns3_step = window_index
            weighted_components = _weighted_reward_components_from_feedback_row(row)
            metrics = compact_numeric_metrics(
                {
                    "external/window_index": row.get("window_index"),
                    "external/ns3_step": ns3_step,
                    "external/reward": row.get("reward"),
                    "external/step_reward": row.get("reward"),
                    "external/internal_policy_evaluation_reward": row.get("internal_policy_evaluation_reward"),
                    "external/internal_best_episode_reward": row.get("internal_best_episode_reward"),
                    "external/aggregate_throughput_mbps": row.get("aggregate_throughput_mbps"),
                    "external/normalized_throughput": row.get("normalized_throughput"),
                    "external/active_gnb_count": row.get("active_gnb_count"),
                    "external/connected_ues": row.get("connected_ues"),
                    "external/disconnected_ues": row.get("disconnected_ues"),
                    "external/done": row.get("done"),
                    "external/terminated": row.get("terminated"),
                    "external/truncated": row.get("truncated"),
                    "external/crashed": row.get("crashed"),
                    "external/sum_qosflow_pdcp_pdu_volume_dl_filter": row.get("sum_qosflow_pdcp_pdu_volume_dl_filter"),
                    "external/sum_tb_totnbrdl_1": row.get("sum_tb_totnbrdl_1"),
                    "external/sum_rlf_value": row.get("sum_rlf_value"),
                    "external/sum_es_on_cost": row.get("sum_es_on_cost"),
                    "external/zero_count": row.get("zero_count"),
                    "external/reward_component_weighted/throughput": weighted_components["throughput"],
                    "external/reward_component_weighted/active_gnb_count": weighted_components["active_gnb_count"],
                    "external/reward_component_weighted/disconnected_ues": weighted_components["disconnected_ues"],
                }
            )
            if metrics:
                wandb.log(metrics, step=window_log_step, commit=False)
                for key, value in metrics.items():
                    run.summary[f"latest/{key}"] = value
            for key in [
                "throughput_source",
                "load_source",
                "selected_fidelity",
            ]:
                if row.get(key):
                    run.summary[f"latest/external/{key}"] = row.get(key)

            # Log the exported/applied topology only when the matching external ns-3 row
            # for this same window is available. This gives the user a topology plot that
            # is semantically aligned with external/* metrics like active_gnb_count.
            if final_topology_plot_path and os.path.exists(final_topology_plot_path):
                topology_metrics = {
                    "external/plots/final_topology_applied": wandb.Image(final_topology_plot_path),
                }
                wandb.log(topology_metrics, step=window_log_step, commit=False)

            # [FIX 2b] Medie di fine simulazione (equivalente a avg_qos_mbps / avg_rlf / avg_activation_cost di DQN/PPO)
            ext_window_count = int(state.get("_ext_window_count", 0)) + 1
            state["_ext_window_count"] = ext_window_count
            _mean_fields = [
                ("reward", "mean/external_reward"),
                ("aggregate_throughput_mbps", "mean/external_aggregate_throughput_mbps"),
                ("normalized_throughput", "mean/external_normalized_throughput"),
                ("active_gnb_count", "mean/external_active_gnb_count"),
                ("connected_ues", "mean/external_connected_ues"),
                ("disconnected_ues", "mean/external_disconnected_ues"),
                ("sum_rlf_value", "mean/external_avg_rlf"),
                ("sum_es_on_cost", "mean/external_avg_activation_cost"),
            ]
            for csv_key, summary_key in _mean_fields:
                val = parse_optional_float(row.get(csv_key))
                if val is not None:
                    cumsum_key = f"_cumsum_{csv_key}"
                    state[cumsum_key] = state.get(cumsum_key, 0.0) + val
                    run.summary[summary_key] = state[cumsum_key] / ext_window_count
            cur_reward = parse_optional_float(row.get("reward"))
            if cur_reward is not None:
                if cur_reward > state.get("_best_ext_reward", float("-inf")):
                    state["_best_ext_reward"] = cur_reward
                run.summary["best/external_reward"] = state["_best_ext_reward"]

        log_csv_artifact(wandb, run, "morabito_feedback", morabito_feedback_csv)
    morabito_plot_data_csv = payload.get("morabito_plot_data_csv")
    if morabito_plot_data_csv and os.path.exists(morabito_plot_data_csv):
        log_csv_artifact(wandb, run, "morabito_plot_data", morabito_plot_data_csv)
    morabito_plot_dir = payload.get("morabito_plot_dir")
    if morabito_plot_dir:
        maybe_log_images(
            wandb,
            [os.path.join(morabito_plot_dir, name) for name in MORABITO_PLOT_FILENAMES],
            step=window_log_step,
            prefix="external/plots",
        )

    mab_history_csv = payload.get("mab_history_csv")
    if mab_history_csv and os.path.exists(mab_history_csv):
        with open(mab_history_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = _find_matching_row(rows, window_index=window_index)
            ns3_step = parse_optional_float(row.get("window_index"))
            if ns3_step is None:
                ns3_step = window_index
            def _fidelity(arm_idx):
                return str(row.get(f"arm_{arm_idx}_fidelity") or f"arm_{arm_idx}")

            def _arm_mean(arm_idx):
                count = float(row.get(f"arm_{arm_idx}_count") or 0)
                return row.get(f"arm_{arm_idx}_mean_reward") if count > 0 else None

            mab_metrics = {
                "external/ns3_step": ns3_step,
                "mab/window_index": row.get("window_index"),
                "mab/selected_arm": row.get("selected_arm"),
                "mab/external_reward": row.get("external_reward"),
            }
            # Per-arm metrics keyed by fidelity name (high/medium/low) instead of arm index.
            for arm_idx in range(3):
                fid = _fidelity(arm_idx)
                count = float(row.get(f"arm_{arm_idx}_count") or 0)
                mab_metrics[f"mab/{fid}_count"] = row.get(f"arm_{arm_idx}_count")
                mab_metrics[f"mab/{fid}_mean_reward"] = _arm_mean(arm_idx)

            metrics = compact_numeric_metrics(mab_metrics)
            if metrics:
                wandb.log(metrics, step=window_log_step, commit=False)
            run.summary["mab/latest_selected_fidelity"] = row.get("selected_fidelity")
            run.summary["mab/latest_selected_arm"] = row.get("selected_arm")
        log_csv_artifact(wandb, run, "mab_history", mab_history_csv)
    mab_plot_dir = payload.get("mab_plot_dir")
    if mab_plot_dir:
        maybe_log_images(
            wandb,
            [os.path.join(mab_plot_dir, name) for name in CONTROL_PLOT_FILENAMES if name.startswith("mab_")],
            step=window_log_step,
            prefix="mab/plots",
        )

    wandb.log({}, step=window_log_step, commit=True)
    state["next_window_step"] = window_log_step + 1
    with open(payload["wandb_state_json"], "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    run.finish()


if __name__ == "__main__":
    main()
