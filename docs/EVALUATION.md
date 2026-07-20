# Evaluation & Fair Comparison Spec

**Purpose:** define the metrics and plots that let us compare MAB (CLARA) and DRL
(DQN/PPO) **fairly**, despite them training on slightly different reward
functions (see [REWARD_SPACE.md](REWARD_SPACE.md) §4).

## 1. The comparability problem

- **MAB / CLARA** trains on `r_base` (switching cost only biases *action
  selection*, not reward → stays a bandit).
- **DRL / DQN, PPO** trains on `r_drl = r_base − w·switch_cost` (switching cost is
  *in the reward*, as in the papers those methods come from).

Plotting each agent's *native training reward* on one axis would be **unfair** —
DRL's curve is penalized by a term MAB's curve never sees. So we log an
apples-to-apples **comparable reward** in addition to each native reward.

## 2. Comparable reward (evaluation only)

For **every** agent, every step, compute:
```
r_cmp(t) = r_base(t) − w · switch_cost(t)
```
- For DRL this **equals** its training reward.
- For MAB the switch cost is subtracted **only here, for display** — MAB is still
  *trained* purely on `r_base`.
- `switch_cost(t)` is computed identically for all agents from the same formula
  and params, so the axis is common.

This is a *measurement*, never fed back into any learner.

## 3. Required charts (per the user)

Per UAV count `N ∈ {3,5,7,9}` (and per capacity `K` if swept):

1. **DRL/PPO reward chart** — native training reward of DQN and PPO
   (`= r_cmp`), episode return over the 100 episodes.
2. **MAB reward chart** — native training reward of CLARA (and UCB/tabular
   baselines) = `r_base`, episode return over 100 episodes.
3. **Unified comparison chart** — **all** approaches on the **comparable reward
   `r_cmp`**: MAB curves are still trained on `r_base` but *plotted* with the
   switch cost subtracted, so CLARA, DQN, PPO sit on one fair axis.

Each chart: x = episode (1..100), y = episode return (mean ± std band if seeds
are repeated); one line per agent.

## 4. Per-episode metrics to log (all agents)

Logged every episode so any of the above can be reconstructed:

| Metric | Notes |
|---|---|
| `return_base` | Σ_t `r_base` (MAB native) |
| `return_cmp` | Σ_t `r_cmp` (= DRL native; comparable axis) |
| `switch_cost_total` | Σ_t `switch_cost(t)` |
| `mean_throughput_mbps` | avg aggregate throughput |
| `mean_active_uavs` | avg # UAVs ON |
| `mean_commanded_active_uavs` | avg # commanded ON (pre-constraint) |
| `outage_rate` | mean `disconnected_ue_count / ue_count` |
| `mean_capacity_utilization` | mean served/K across active UAVs |
| `total_energy_kwh` | from energy model (placeholder → UAV model later) |
| per-UAV activation cost | mean `switch_cost_k` per UAV |

(Names mirror the user's `energy_saving_drl.py` wandb keys where possible:
"cumulative reward", "avg active RUs", "avg commanded active RUs",
"activation cost", "average throughput Mbps", "avg unconnected UEs".)

## 4b. CLARA credit-assignment comparison
CLARA runs under two selectable credit schemes (see
[REWARD_SPACE.md](REWARD_SPACE.md) §3): **B** `shared_off_penalty` (default) and
**C** `difference`. Treat them as two agent variants (`clara-B`, `clara-C`) in the
same charts so we can see which credit rule learns better. Both are still plotted
on the system-level `r_cmp`, so they remain comparable to each other and to DRL.

## 5. Cross-N / cross-K summary
After all campaigns: summary plots of best/final `return_cmp` (and throughput,
outage, activation cost) **vs N** (and vs K), one line per agent — the headline
"which xApp scales best" result.

## 6. W&B
Keep W&B logging (already in the repo). Mirror the metric names above so the new
runs line up with the user's existing dashboards. Local CSV/PNG fallback stays
for offline runs.

## 7. Alignment checklist
- [ ] `switch_cost` computed once, identically, for all agents (same Cf, λf,
      time_factor).
- [ ] MAB training never sees `switch_cost`; DRL training does.
- [ ] `r_cmp = r_base − w·switch_cost` logged for every agent every episode.
- [ ] Three charts produced per (N[,K]); unified chart uses `r_cmp` for all.
- [ ] Same seeds / trace / env across agents within a comparison.
