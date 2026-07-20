#!/usr/bin/env python3
import argparse
import csv
import os


FIGURE_DPI = 300
SINGLE_FIGSIZE = (7.0, 3.8)
WIDE_FIGSIZE = (8.4, 4.4)
STACKED_FIGSIZE = (8.8, 4.8)
LINEWIDTH = 1.6
MARKER_SIZE = 4.5
GRID_ALPHA = 0.22
LEGEND_FONTSIZE = 9

COLOR_REWARD = "#1b4965"
COLOR_INTERNAL = "#1b4965"
COLOR_EXTERNAL = "#d1495b"
COLOR_QOS = "#2a9d8f"
COLOR_TB = "#4c78a8"
COLOR_RLF = "#b22222"
COLOR_COST = "#f4a261"
COLOR_ZERO = "#7a7a7a"
COLOR_DONE = "#3c6e71"
COLOR_CRASHED = "#6d597a"
COLOR_CONN = "#588157"
COLOR_DISC = "#bc4749"


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row, key, default=0.0):
    return float(row.get(key, default) or default)


def configure_plot_style(matplotlib):
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_current_figure(plt, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def write_empirical_cdf(plt, values, *, path, title, xlabel, color):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    n = len(ordered)
    y = [(index + 1) / n for index in range(n)]
    plt.figure(figsize=SINGLE_FIGSIZE)
    plt.step(ordered, y, where="post", linewidth=LINEWIDTH, color=color)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)
    save_current_figure(plt, path)
    return path


def generate_plots(csv_path, output_dir):
    mpl_config_dir = os.path.join(output_dir, ".matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    configure_plot_style(matplotlib)
    import matplotlib.pyplot as plt

    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    x = [int(float(row["window_index"])) for row in rows]
    reward = [to_float(row, "reward") for row in rows]
    internal_reward = [to_float(row, "internal_policy_evaluation_reward") for row in rows]
    aggregate_throughput_mbps = [to_float(row, "aggregate_throughput_mbps") for row in rows]
    normalized_throughput = [to_float(row, "normalized_throughput") for row in rows]
    connected_ues = [to_float(row, "connected_ues") for row in rows]
    disconnected_ues = [to_float(row, "disconnected_ues") for row in rows]
    qos = [to_float(row, "throughput_proxy_qosflow") for row in rows]
    tb = [to_float(row, "tb_totnbrdl_1") for row in rows]
    rlf = [to_float(row, "radio_link_failure") for row in rows]
    on_cost = [to_float(row, "activation_cost") for row in rows]
    zero_count = [to_float(row, "zero_count") for row in rows]
    crashed = [to_float(row, "crashed") for row in rows]
    done = [to_float(row, "done") for row in rows]

    plot_specs = [
        ("morabito_reward.png", reward, "Morabito/ns-3 Reward", "reward", COLOR_REWARD),
        ("morabito_aggregate_throughput_mbps.png", aggregate_throughput_mbps, "Morabito compatible aggregate throughput (Mbps)", "aggregate_throughput_mbps", COLOR_QOS),
        ("morabito_normalized_throughput.png", normalized_throughput, "Morabito compatible normalized throughput", "normalized_throughput", COLOR_TB),
        ("morabito_connected_ues.png", connected_ues, "Morabito compatible connected UEs", "connected_ues", COLOR_CONN),
        ("morabito_disconnected_ues.png", disconnected_ues, "Morabito compatible disconnected UEs", "disconnected_ues", COLOR_DISC),
        ("morabito_qosflow.png", qos, "ns-3 SUM_QosFlow.PdcpPduVolumeDL_Filter", "throughput_proxy_qosflow", COLOR_QOS),
        ("morabito_tb_totnbrdl.png", tb, "ns-3 SUM_TB.TotNbrDl.1", "tb_totnbrdl_1", COLOR_TB),
        ("morabito_rlf.png", rlf, "ns-3 SUM_RLF_VALUE", "radio_link_failure", COLOR_RLF),
        ("morabito_es_on_cost.png", on_cost, "ns-3 SUM_ES_ON_COST", "activation_cost", COLOR_COST),
        ("morabito_zero_count.png", zero_count, "ns-3 ZERO_COUNT", "zero_count", COLOR_ZERO),
        ("morabito_crashed.png", crashed, "ns-3 crashed flag", "crashed", COLOR_CRASHED),
        ("morabito_done.png", done, "ns-3 done flag", "done", COLOR_DONE),
    ]
    generated = []
    for filename, y_values, title, ylabel, color in plot_specs:
        path = os.path.join(output_dir, filename)
        plt.figure(figsize=SINGLE_FIGSIZE)
        plt.plot(
            x,
            y_values,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=color,
        )
        plt.xlabel("Window Index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)
        save_current_figure(plt, path)
        generated.append(path)

    stacked_path = os.path.join(output_dir, "morabito_reward_components_stacked.png")
    plt.figure(figsize=STACKED_FIGSIZE)
    plt.stackplot(
        x,
        qos,
        tb,
        rlf,
        on_cost,
        zero_count,
        labels=[
            "SUM_QosFlow.PdcpPduVolumeDL_Filter",
            "SUM_TB.TotNbrDl.1",
            "SUM_RLF_VALUE",
            "SUM_ES_ON_COST",
            "ZERO_COUNT",
        ],
        colors=[COLOR_QOS, COLOR_TB, COLOR_RLF, COLOR_COST, COLOR_ZERO],
        alpha=0.7,
    )
    plt.xlabel("Window Index")
    plt.ylabel("Component value")
    plt.title("Morabito/ns-3 Reward Components")
    plt.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)
    save_current_figure(plt, stacked_path)
    generated.append(stacked_path)

    comparison_path = os.path.join(output_dir, "reward_internal_vs_morabito.png")
    plt.figure(figsize=WIDE_FIGSIZE)
    plt.plot(
        x,
        internal_reward,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINEWIDTH,
        color=COLOR_INTERNAL,
        label="RAN FUSION internal policy reward",
    )
    plt.plot(
        x,
        reward,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINEWIDTH,
        color=COLOR_EXTERNAL,
        label="Morabito/ns-3 external reward",
    )
    plt.xlabel("Window Index")
    plt.ylabel("Reward")
    plt.title("Internal vs External Reward by Window")
    plt.legend(loc="best")
    plt.grid(True, alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)
    save_current_figure(plt, comparison_path)
    generated.append(comparison_path)

    cdf_specs = [
        (
            "morabito_throughput_cdf.png",
            aggregate_throughput_mbps,
            "Morabito compatible aggregate throughput CDF",
            "Aggregate throughput (Mbps)",
            COLOR_QOS,
        ),
        (
            "morabito_rlf_cdf.png",
            rlf,
            "Morabito/ns-3 Radio Link Failure CDF",
            "SUM_RLF_VALUE",
            COLOR_RLF,
        ),
        (
            "morabito_activation_cost_cdf.png",
            on_cost,
            "Morabito/ns-3 Activation Cost CDF",
            "SUM_ES_ON_COST",
            COLOR_COST,
        ),
    ]
    for filename, values, title, xlabel, color in cdf_specs:
        path = write_empirical_cdf(
            plt,
            values,
            path=os.path.join(output_dir, filename),
            title=title,
            xlabel=xlabel,
            color=color,
        )
        if path:
            generated.append(path)
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate JSAC figures from Morabito/ns-3 plot CSV data.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to morabito_plot_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where plots will be written. Defaults to the CSV parent directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)
    generated = generate_plots(os.path.abspath(args.csv), output_dir)
    print(f"plot_csv: {os.path.abspath(args.csv)}")
    print(f"plot_output_dir: {output_dir}")
    for path in generated:
        print(f"plot: {path}")


if __name__ == "__main__":
    main()
