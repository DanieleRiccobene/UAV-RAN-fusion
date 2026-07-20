"""Plotting for UAV campaigns — time-series + CDF charts, all into a plots/ folder.

Shared by scripts/train_uav.py (live) and scripts/plot_uav_results.py (from CSVs).
"""

from __future__ import annotations

import os

# Time-series charts (one line per approach): (suffix, title, metric_key).
METRIC_CHARTS = [
    ("rlf", "Radio link failures (mean disconnected UEs/step)", "mean_rlf"),
    ("active_rus", "Radio units in ON status (mean active UAVs)", "mean_active_uavs"),
    ("cumulative_reward", "Cumulative reward (episode return)", "return_base"),
    ("throughput", "Throughput (mean Mbps/step)", "mean_throughput_mbps"),
    ("activation_cost", "Activation cost per episode", "switch_cost_total"),
    ("comparable_reward", "Comparable reward (fair MAB vs DRL)", "return_cmp"),
]

# CDF charts over the per-episode distribution (one CDF curve per approach).
CDF_CHARTS = [
    ("cdf_num_ru", "CDF — number of RUs ON", "mean_active_uavs"),
    ("cdf_throughput", "CDF — throughput (Mbps)", "mean_throughput_mbps"),
    ("cdf_reward", "CDF — reward (episode return)", "return_base"),
    ("cdf_activation_cost", "CDF — activation (switching) cost", "switch_cost_total"),
    ("cdf_rlf", "CDF — radio link failures", "mean_rlf"),
]

# CDF over INSTANTANEOUS per-step values pooled across episodes (used when per-step
# data is available). Keys match the columns written to the *_steps.csv files.
CDF_STEP_CHARTS = [
    ("cdf_num_ru", "CDF — number of RUs ON (per-step)", "active_uavs"),
    ("cdf_throughput", "CDF — throughput (Mbps, per-step)", "throughput_mbps"),
    ("cdf_reward", "CDF — reward (per-step)", "reward"),
    ("cdf_activation_cost", "CDF — activation (switching) cost (per-step)", "switch_cost"),
    ("cdf_rlf", "CDF — radio link failures (per-step)", "rlf"),
]


# --- Per-approach visual identity: FIXED colour + line style + marker, identical
# in every figure. With >8 approaches we use composite encoding (data-viz method):
# colour is grouped by algorithm family, line style + marker disambiguate within it.
# Colours are the validated 8-hue categorical palette (+ gray/black for baselines).
TOKEN_COLORS = {
    "clara":      "#0072b2",   # blue        (flagship)
    "clara_diff": "#56b4e9",   # sky blue    (clara's twin — distinct shade + dashed)
    "ppo":        "#e69f00",   # orange
    "dqn":        "#d55e00",   # vermillion
    "wang":       "#8c3b00",   # brown       (dqn's twin — dashed)
    "ucb":        "#cc79a7",   # mauve
    "random":     "#999999",   # gray        (sanity baseline)
    "oracle":     "#000000",   # black       (genie reference)
    "el_amine":   "#009e73",   # green
    "xu":         "#117777",   # teal
    "rezaei":     "#e7298a",   # magenta
    "masrur":     "#7b3294",   # purple      (rezaei's twin — dashed)
}
TOKEN_LINESTYLES = {
    "clara": "solid", "clara_diff": (0, (5, 2)),    # dashed twin of clara
    "dqn": "solid",   "wang": (0, (5, 2)),          # dashed twin of dqn
    "rezaei": "solid", "masrur": (0, (5, 2)),        # dashed twin of rezaei
    "ppo": "solid", "ucb": "solid", "el_amine": "solid", "xu": "solid",
    "random": (0, (3, 1, 1, 1)),                     # dash-dot
    "oracle": (0, (1, 1)),                           # dotted
}
# Stable, distinct marker per approach (consistent across every chart).
TOKEN_MARKERS = {
    "clara": "o", "clara_diff": "s", "ucb": "^", "random": "D", "dqn": "v", "ppo": "P",
    "oracle": "*",
    "el_amine": "X", "xu": "<", "rezaei": ">", "masrur": "p", "wang": "h",
}
# Fixed order → identical marker phase (and fallback colour/marker slot) everywhere.
TOKEN_ORDER = ["clara", "clara_diff", "oracle", "ucb", "random", "dqn", "ppo",
               "wang", "el_amine", "xu", "rezaei", "masrur"]
_COLOR_CYCLE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
                "#e34948", "#e87ba4", "#eb6834"]
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
LEGEND_LOC = "upper right"
LINE_WIDTH = 1.6
# Approaches omitted from every figure (kept in the CSVs, just not drawn).
EXCLUDE_TOKENS = {"random", "ucb"}
# Legend display names (labels only — CSV tokens / filenames are unchanged).
DISPLAY_NAMES = {
    "clara": "CLARA",
    "clara_diff": "CLARA_DIFF",
    "dqn": "DQN",
    "ppo": "PPO",
    "wang": "ES-xApp",
    "el_amine": "Tabular-El_Amine",
    "rezaei": "MADQN-Rezaei",
    "masrur": "MARL-DDQN-Masrur",
    "xu": "GAMA",
}


def _label(token):
    return DISPLAY_NAMES.get(token, token)


def _phase(token):
    """Stable per-token index → marker phase offset (identical across all figures)."""
    return TOKEN_ORDER.index(token) if token in TOKEN_ORDER else 0


def _marker(token, index):
    return TOKEN_MARKERS.get(token, _MARKER_CYCLE[index % len(_MARKER_CYCLE)])


def _color(token, index):
    return TOKEN_COLORS.get(token, _COLOR_CYCLE[index % len(_COLOR_CYCLE)])


def _linestyle(token):
    return TOKEN_LINESTYLES.get(token, "solid")


def _style(token, index, n_points):
    """Per-series kwargs: fixed colour, per-approach line style, phase-offset markers."""
    me = max(1, n_points // 12)
    return dict(color=_color(token, index), linestyle=_linestyle(token),
                marker=_marker(token, index), markevery=(_phase(token) % me, me),
                markersize=6, linewidth=LINE_WIDTH)


def _cdf(values):
    vals = sorted(values)
    n = len(vals)
    return vals, [(i + 1) / n for i in range(n)]


def _legend_below_title(ax, title):
    """Horizontal legend just below the title, above the plot area."""
    n_lab = len(ax.get_legend_handles_labels()[1])
    ncol = min(4, max(1, n_lab))
    rows = -(-n_lab // ncol)  # ceil
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
              fontsize=8, frameon=False, columnspacing=1.2, handletextpad=0.4)
    ax.set_title(title, pad=14 * rows + 12)


def plot_campaign(out_dir, results, scenario, exclude=EXCLUDE_TOKENS, step_results=None):
    """results: {num_uavs: {token: (history, family)}}.

    step_results (optional): {num_uavs: {token: [per-step dicts]}}. When present,
    the CDF charts are built over the INSTANTANEOUS per-step values pooled across
    episodes (not per-episode averages).

    Writes, into <out_dir>/plots/: the time-series metric charts + CDF charts per
    N (one curve per approach), plus a cross-N summary. Approaches in `exclude`
    are dropped from every figure (their CSVs are left untouched).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots skipped] matplotlib unavailable ({exc}); CSVs written.", flush=True)
        return None

    if exclude:
        results = {n: {t: v for t, v in per.items() if t not in exclude}
                   for n, per in results.items()}
        if step_results:
            step_results = {n: {t: v for t, v in per.items() if t not in exclude}
                            for n, per in step_results.items()}

    plots_dir = os.path.join(out_dir, "plots")
    cdf_dir = os.path.join(plots_dir, "cdf")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(cdf_dir, exist_ok=True)

    for num_uavs, per_token in sorted(results.items()):
        # Time-series (x = episode).
        for suffix, title, key in METRIC_CHARTS:
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted = False
            for idx, (token, (history, _f)) in enumerate(sorted(per_token.items())):
                if not history or key not in history[0]:
                    continue
                ax.plot([r["episode"] for r in history], [r[key] for r in history],
                        label=_label(token), **_style(token, idx, len(history)))
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel("episode")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
            _legend_below_title(ax, f"{title}\nN={num_uavs} UAVs · scenario={scenario}")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"N{num_uavs}_{scenario}_{suffix}.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

        # CDF: over instantaneous per-step values if available, else per-episode.
        step_per_token = (step_results or {}).get(num_uavs, {})
        use_steps = bool(step_per_token)
        cdf_spec = CDF_STEP_CHARTS if use_steps else CDF_CHARTS
        ylabel = ("CDF (fraction of steps ≤ x)" if use_steps
                  else "CDF (fraction of episodes ≤ x)")
        for suffix, title, key in cdf_spec:
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted = False
            for idx, (token, (history, _f)) in enumerate(sorted(per_token.items())):
                if use_steps:
                    rows = step_per_token.get(token)
                    if not rows or key not in rows[0]:
                        continue
                    values = [r[key] for r in rows]
                else:
                    if not history or key not in history[0]:
                        continue
                    values = [r[key] for r in history]
                xs, ys = _cdf(values)
                ax.plot(xs, ys, label=_label(token), **_style(token, idx, len(xs)))
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel(key)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            _legend_below_title(ax, f"{title}\nN={num_uavs} UAVs · scenario={scenario}")
            fig.tight_layout()
            fig.savefig(os.path.join(cdf_dir, f"N{num_uavs}_{scenario}_{suffix}.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

    # Cross-N summary: final comparable reward vs N (mean of last 10 episodes).
    fig, ax = plt.subplots(figsize=(8, 5))
    tokens = sorted({t for pt in results.values() for t in pt})
    for idx, token in enumerate(tokens):
        xs, ys = [], []
        for num_uavs in sorted(results):
            if token in results[num_uavs]:
                history = results[num_uavs][token][0]
                tail = history[-10:]
                if tail:
                    xs.append(num_uavs)
                    ys.append(sum(r["return_cmp"] for r in tail) / len(tail))
        if xs:
            ax.plot(xs, ys, label=_label(token), color=_color(token, idx),
                    linestyle=_linestyle(token), marker=_marker(token, idx),
                    markersize=7, linewidth=LINE_WIDTH)
    ax.set_title(f"Final comparable reward vs N (mean last 10 ep) · scenario={scenario}")
    ax.set_xlabel("number of UAVs")
    ax.set_ylabel("return_cmp")
    ax.grid(True, alpha=0.3)
    ax.legend(loc=LEGEND_LOC)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"summary_{scenario}_return_cmp_vs_N.png"), dpi=140)
    plt.close(fig)
    print(f"[plots] written to {plots_dir}", flush=True)
    return plots_dir
