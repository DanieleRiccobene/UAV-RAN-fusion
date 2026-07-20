#!/usr/bin/env python3
"""
Academic plotting utilities for Morabito DT / ns-3 external evaluation CSV exports.

The script expects Weights & Biases-style wide CSV files, where each run has columns like:
    <run_name> - external/<metric_name>
    <run_name> - external/<metric_name>__MIN
    <run_name> - external/<metric_name>__MAX
    <run_name> - _step

Only the actual metric columns are used. The __MIN/__MAX and _step columns are ignored.

Generated figures:
  1. Instantaneous curves: one curve per algorithm, averaged across runs when multiple runs exist.
     The x-axis is automatically truncated to the minimum available curve length.
  2. Mean-value bars: mean value per run, then mean/std across runs of the same algorithm.
  3. CDF curves: empirical CDF over all instantaneous samples of each algorithm.

Usage from the folder containing the CSV files:
    python plot_academic_metrics.py --input-dir . --output-dir figures

Typical usage with explicit paths:
    python plot_academic_metrics.py \
        --input-dir /path/to/csvs \
        --output-dir /path/to/figures \
        --formats pdf png svg
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Metric configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricConfig:
    filename: str
    metric_key: str
    ylabel: str
    slug: str
    title: str


METRICS: Tuple[MetricConfig, ...] = (
    MetricConfig(
        filename="aggregate_throughput.csv",
        metric_key="aggregate_throughput_mbps",
        ylabel="Aggregate throughput [Mbps]",
        slug="aggregate_throughput",
        title="Aggregate throughput",
    ),
    MetricConfig(
        filename="step_reward.csv",
        metric_key="step_reward",
        ylabel="Step reward",
        slug="step_reward",
        title="Step reward",
    ),
    MetricConfig(
        filename="active_gnb.csv",
        metric_key="active_gnb_count",
        ylabel="Active gNBs",
        slug="active_gnb",
        title="Active gNB count",
    ),
    MetricConfig(
        filename="discconnected_ues.csv",  # keep the uploaded filename spelling
        metric_key="disconnected_ues",
        ylabel="Disconnected UEs",
        slug="disconnected_ues",
        title="Disconnected UEs",
    ),
)


# You can edit this mapping if you want a different label for the date-only run.
RUN_LABEL_OVERRIDES: Mapping[str, str] = {
    "2026-05-08_13-32": "External MAB",
}

ALGORITHM_ORDER: Tuple[str, ...] = (
    "External MAB",
    "PPO",
    "A2C",
    "DQN",
    "Dueling DQN",
)


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

def set_academic_style() -> None:
    """Configure a clean academic matplotlib style without requiring LaTeX."""
    plt.rcParams.update({
        "figure.figsize": (6.6, 3.8),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


MARKERS: Tuple[str, ...] = ("o", "s", "D", "^", "v", "P", "X", "*")
LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def infer_algorithm(run_name: str) -> str:
    """Infer the algorithm label from a W&B run name."""
    if run_name in RUN_LABEL_OVERRIDES:
        return RUN_LABEL_OVERRIDES[run_name]

    low = run_name.lower()
    if "dueling_dqn" in low or "dueling-dqn" in low or "dueling dqn" in low:
        return "Dueling DQN"
    if re.search(r"(^|[_\-])dqn([_\-]|$)", low):
        return "DQN"
    if re.search(r"(^|[_\-])ppo([_\-]|$)", low):
        return "PPO"
    if re.search(r"(^|[_\-])a2c([_\-]|$)", low):
        return "A2C"
    if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", run_name):
        return "External MAB"
    return run_name


def algorithm_sort_key(label: str) -> Tuple[int, str]:
    if label in ALGORITHM_ORDER:
        return (ALGORITHM_ORDER.index(label), label)
    return (len(ALGORITHM_ORDER), label)


def metric_columns(df: pd.DataFrame, metric_key: str) -> List[str]:
    """Return only the real metric columns, excluding __MIN/__MAX and _step columns."""
    target_suffix = f" - external/{metric_key}"
    cols = [c for c in df.columns if c.endswith(target_suffix)]
    if not cols:
        available = [c for c in df.columns if " - external/" in c and "__" not in c]
        raise ValueError(
            f"No columns found for metric '{metric_key}'.\n"
            f"Expected suffix: {target_suffix}\n"
            f"Available metric-like columns include: {available[:10]}"
        )
    return cols


def run_name_from_metric_column(col: str, metric_key: str) -> str:
    suffix = f" - external/{metric_key}"
    return col[: -len(suffix)]


def read_metric_series(csv_path: Path, metric_key: str) -> Dict[str, List[np.ndarray]]:
    """
    Read a CSV and return {algorithm_label: [series_per_run, ...]}.
    Each series is a 1-D float array with NaNs removed.
    """
    df = pd.read_csv(csv_path)
    groups: Dict[str, List[np.ndarray]] = {}

    for col in metric_columns(df, metric_key):
        run_name = run_name_from_metric_column(col, metric_key)
        label = infer_algorithm(run_name)
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        groups.setdefault(label, []).append(values)

    if not groups:
        raise ValueError(f"No non-empty series found in {csv_path} for {metric_key}")
    return groups


def ordered_groups(groups: Mapping[str, List[np.ndarray]]) -> List[Tuple[str, List[np.ndarray]]]:
    return sorted(groups.items(), key=lambda item: algorithm_sort_key(item[0]))


def global_min_length(groups: Mapping[str, List[np.ndarray]]) -> int:
    lengths = [len(s) for series_list in groups.values() for s in series_list]
    if not lengths:
        raise ValueError("Cannot compute minimum length: no series available")
    return min(lengths)


def stack_truncated(series_list: Sequence[np.ndarray], n: int) -> np.ndarray:
    valid = [s[:n] for s in series_list if len(s) >= n]
    if not valid:
        raise ValueError("No valid series after truncation")
    return np.vstack(valid)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    window = int(window)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_dir: Path, basename: str, formats: Iterable[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{basename}.{fmt}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_instantaneous(
    groups: Mapping[str, List[np.ndarray]],
    metric: MetricConfig,
    output_dir: Path,
    formats: Sequence[str],
    smooth_window: int = 1,
    show_std: bool = True,
) -> None:
    """
    Plot instantaneous curves.

    Important: the x-axis is truncated to the minimum number of samples across all
    available curves. This is the behaviour requested for fair instantaneous plots.
    """
    n = global_min_length(groups)
    x = np.arange(n)

    fig, ax = plt.subplots()

    for idx, (label, series_list) in enumerate(ordered_groups(groups)):
        data = stack_truncated(series_list, n)
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)

        if smooth_window > 1:
            mean_to_plot = moving_average(mean, smooth_window)
            std_to_plot = moving_average(std, smooth_window)
        else:
            mean_to_plot = mean
            std_to_plot = std

        markevery = max(1, n // 12)
        line, = ax.plot(
            x,
            mean_to_plot,
            label=label,
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markevery=markevery,
            markersize=4.2,
        )
        if show_std and data.shape[0] > 1:
            ax.fill_between(
                x,
                mean_to_plot - std_to_plot,
                mean_to_plot + std_to_plot,
                alpha=0.12,
                linewidth=0,
                color=line.get_color(),
            )

    ax.set_xlabel("Step index")
    ax.set_ylabel(metric.ylabel)
    ax.set_title(metric.title)
    ax.set_xlim(0, n - 1)
    ax.legend(frameon=True, ncol=1)
    fig.tight_layout()
    save_figure(fig, output_dir, f"instantaneous_{metric.slug}", formats)


def plot_mean_values(
    groups: Mapping[str, List[np.ndarray]],
    metric: MetricConfig,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot mean values per algorithm with run-to-run standard deviation."""
    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []

    for label, series_list in ordered_groups(groups):
        run_means = np.array([np.mean(s) for s in series_list], dtype=float)
        labels.append(label)
        means.append(float(np.mean(run_means)))
        stds.append(float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(x, means, yerr=stds, capsize=4, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.ylabel)
    ax.set_title(f"Mean {metric.title.lower()}")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    save_figure(fig, output_dir, f"mean_{metric.slug}", formats)


def plot_cdf(
    groups: Mapping[str, List[np.ndarray]],
    metric: MetricConfig,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot empirical CDF for each algorithm using all instantaneous samples."""
    fig, ax = plt.subplots()

    for idx, (label, series_list) in enumerate(ordered_groups(groups)):
        samples = np.concatenate([s for s in series_list if len(s) > 0])
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        xs = np.sort(samples)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        markevery = max(1, len(xs) // 14)
        ax.plot(
            xs,
            ys,
            label=label,
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markevery=markevery,
            markersize=4.0,
        )

    ax.set_xlabel(metric.ylabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"CDF of {metric.title.lower()}")
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=True, ncol=1)
    fig.tight_layout()
    save_figure(fig, output_dir, f"cdf_{metric.slug}", formats)


# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

def append_summary_rows(summary_rows: List[dict], metric: MetricConfig, groups: Mapping[str, List[np.ndarray]]) -> None:
    for label, series_list in ordered_groups(groups):
        run_means = np.array([np.mean(s) for s in series_list], dtype=float)
        all_samples = np.concatenate(series_list)
        summary_rows.append({
            "metric": metric.metric_key,
            "algorithm": label,
            "n_runs": len(series_list),
            "n_samples_total": int(sum(len(s) for s in series_list)),
            "min_curve_length": int(min(len(s) for s in series_list)),
            "mean_of_run_means": float(np.mean(run_means)),
            "std_of_run_means": float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0,
            "median_all_samples": float(np.median(all_samples)),
            "p05_all_samples": float(np.percentile(all_samples, 5)),
            "p95_all_samples": float(np.percentile(all_samples, 95)),
        })


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate academic instantaneous, mean, and CDF plots from W&B CSV exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Folder containing active_gnb.csv, step_reward.csv, discconnected_ues.csv, aggregate_throughput.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Folder where figures and summary CSV will be saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
        choices=["pdf", "png", "svg", "eps"],
        help="Output figure formats.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["instantaneous", "mean", "cdf"],
        choices=["instantaneous", "mean", "cdf"],
        help="Which plot types to generate.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average window for instantaneous curves. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Disable standard-deviation shaded bands in instantaneous plots.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_academic_style()
    ensure_dir(args.output_dir)

    summary_rows: List[dict] = []

    for metric in METRICS:
        csv_path = args.input_dir / metric.filename
        if not csv_path.exists():
            print(f"[WARNING] Missing file: {csv_path}. Skipping {metric.metric_key}.")
            continue

        groups = read_metric_series(csv_path, metric.metric_key)
        append_summary_rows(summary_rows, metric, groups)

        min_len = global_min_length(groups)
        print(f"[INFO] {metric.metric_key}: algorithms={list(groups.keys())}, min_curve_length={min_len}")

        if "instantaneous" in args.plots:
            plot_instantaneous(
                groups,
                metric,
                args.output_dir,
                args.formats,
                smooth_window=args.smooth_window,
                show_std=not args.no_std_band,
            )
        if "mean" in args.plots:
            plot_mean_values(groups, metric, args.output_dir, args.formats)
        if "cdf" in args.plots:
            plot_cdf(groups, metric, args.output_dir, args.formats)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = args.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Summary statistics saved to: {summary_path}")
    else:
        print("[WARNING] No figures generated: no valid input files found.")


if __name__ == "__main__":
    main()