#!/usr/bin/env python3
import argparse
import csv
import os


def plot_gnb_activation_timeline(plt, output_dir, csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    gnb_columns = [
        column
        for column in rows[0].keys()
        if column.startswith("BS_")
    ]
    gnb_columns.sort(key=lambda value: int(value.split("_")[1]))
    window_index = [int(float(row["window_index"])) for row in rows]
    output_paths = []

    active_count_path = os.path.join(output_dir, "gnb_active_count_timeline.png")
    plt.figure(figsize=(10, 4))
    plt.plot(window_index, [float(row["active_gnb_count"]) for row in rows], marker="o", linewidth=1.3)
    plt.xlabel("Window Index")
    plt.ylabel("Active gNB Count")
    plt.title("Active gNB Count by Window")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(active_count_path, dpi=200, bbox_inches="tight")
    plt.close()
    output_paths.append(active_count_path)

    heatmap_path = os.path.join(output_dir, "gnb_activation_heatmap.png")
    matrix = [[int(float(row[column])) for column in gnb_columns] for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(3, len(gnb_columns) * 0.45)))
    image = ax.imshow(list(zip(*matrix)), aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("gNB")
    ax.set_title("gNB ON/OFF Timeline")
    ax.set_yticks(range(len(gnb_columns)))
    ax.set_yticklabels(gnb_columns)
    ax.set_xticks(range(0, len(window_index), max(1, len(window_index) // 10)))
    ax.set_xticklabels([window_index[index] for index in range(0, len(window_index), max(1, len(window_index) // 10))])
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["OFF", "ON"])
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(heatmap_path)
    return output_paths


def plot_mab_timeline(plt, output_dir, csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    window_index = [int(float(row["window_index"])) for row in rows]
    reward = [float(row["external_reward"]) for row in rows]
    selected_arm = [int(float(row["selected_arm"])) for row in rows]
    output_paths = []

    reward_path = os.path.join(output_dir, "mab_external_reward_timeline.png")
    plt.figure(figsize=(10, 4))
    plt.plot(window_index, reward, marker="o", linewidth=1.3, color="#d62728")
    plt.xlabel("Window Index")
    plt.ylabel("External Reward")
    plt.title("MAB External Reward by Window")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reward_path, dpi=200, bbox_inches="tight")
    plt.close()
    output_paths.append(reward_path)

    arm_path = os.path.join(output_dir, "mab_selected_arm_timeline.png")
    plt.figure(figsize=(10, 4))
    plt.step(window_index, selected_arm, where="post", linewidth=1.4, color="#1f77b4")
    plt.yticks([0, 1, 2], ["high", "medium", "low"])
    plt.xlabel("Window Index")
    plt.ylabel("Selected Fidelity")
    plt.title("MAB Selected Fidelity by Window")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(arm_path, dpi=200, bbox_inches="tight")
    plt.close()
    output_paths.append(arm_path)

    mean_reward_columns = sorted(
        [column for column in rows[0].keys() if column.endswith("_mean_reward") and column.startswith("arm_")],
        key=lambda value: int(value.split("_")[1]),
    )
    mean_reward_path = os.path.join(output_dir, "mab_arm_mean_reward_timeline.png")
    plt.figure(figsize=(10, 4))
    for column in mean_reward_columns:
        arm_index = int(column.split("_")[1])
        plt.plot(window_index, [float(row[column]) for row in rows], marker="o", linewidth=1.2, label=f"arm {arm_index}")
    plt.xlabel("Window Index")
    plt.ylabel("Mean Reward")
    plt.title("MAB Mean Reward per Arm")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(mean_reward_path, dpi=200, bbox_inches="tight")
    plt.close()
    output_paths.append(mean_reward_path)
    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnb-csv")
    parser.add_argument("--mab-csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    mpl_config_dir = os.path.join(args.output_dir, ".matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.gnb_csv:
        plot_gnb_activation_timeline(plt, args.output_dir, args.gnb_csv)
    if args.mab_csv:
        plot_mab_timeline(plt, args.output_dir, args.mab_csv)


if __name__ == "__main__":
    main()
