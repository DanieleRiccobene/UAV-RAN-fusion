# State Space Specification

**Purpose:** define exactly what each control solution (MAB / tabular / DRL,
single / multi-agent) observes, so we can check alignment before coding.
Source of truth for the raw observation is `RANTraceEnv.collect_observation`
in [rl/ran_trace_env.py](../rl/ran_trace_env.py).

Fleet size `N ∈ {3, 5, 7, 9}` is fixed within a campaign.

## 1. Raw observation (what the env emits every step)

`env.step()` returns `obs` (dict). Unchanged by this refactor:

```
obs = {
  "timestamp": <trace time>,
  "metrics": { total_throughput, aggregate_throughput_mbps, ue_count,
               outage_ue_count, active_gnb_count, total_gnb_count,
               current_power_w, normalized_throughput,
               normalized_active_gnb_count, ... },
  "gnbs": { UAV_k: { is_active: bool, tx_power: float } },     # N entries
  "ues":  { ue_id: { x_m, y_m, lat, lon, serving_gnb_id,
                     serving_distance_m, estimated_sinr_db,
                     spectral_efficiency_bps_hz, in_outage,
                     throughput, ... } },
}
```

Controllers do **not** consume this dict directly for learning; each projects it
into its own state representation below. All projections are pure functions of
`obs` so every agent sees a consistent world.

## 2. Per-solution state representation

| Solution | Single/Multi | State | Dim |
|---|---|---|---|
| `clara` (game-theoretic cMAB) | multi | **other UAVs' ON/OFF** (context) | N−1 per agent |
| `ucb_stateless` (UCB bandit) | single | **∅ (stateless)** | 0 |
| `tabular_single` (joint Q) | single | active-mask tuple | 2^N keys |
| `tabular_multi` (per-UAV Q) | multi | per-UAV **discretized local load** | small int per agent |
| `dqn` / `ppo` | single | **global feature vector** (below) | `5N + 4` |
| (`dqn_multi`, future) | multi | **per-UAV local vector** (below) | `10` per agent |

### 2.0 CLARA context (multi-agent, our method)
Each UAV agent's "state" is the **vector of the other UAVs' current ON/OFF
decisions** (length N−1), used as the contextual key in its Q-table during
Best-Response negotiation. It deliberately does **not** consume UE-level features
— coordination emerges from the joint-action context. This is native to
`MultiAgentGameMAB` (see [AGENTS.md](AGENTS.md) §3.1).

### 2.1 Stateless (MAB single-agent)
No state. The bandit only sees rewards per arm. Matches `ucb_stateless` today.

### 2.2 Tabular single-agent
State key = current ON/OFF mask as a sorted tuple `((UAV_1, bool), ...)`.
(Current `state_key`.) Tabular over ≤ 2^N states — fine to N=9 (512).

### 2.3 Tabular multi-agent
Each UAV agent's state = **its local served-UE count**, clipped/bucketed to keep
the table small. Current `local_user_load_state` returns the raw integer count;
we bucket it: `{0, 1-2, 3-5, 6-9, 10+}` → 5 states per agent (tunable).

### 2.4 DRL global state vector (`dqn`/`ppo`) — length `5N + 4`
Deterministic ordering `UAV_1..UAV_N`, features normalized to ~[0, 1].
**Implemented** in `rl/agents/base.py:global_state_vector`.

Per UAV `k` (5 features) — **capacity-centric** under the LoS model:
1. `is_active` ∈ {0,1}
2. **capacity utilization** = served-UE count / `K`  (K = `max_ues_per_uav`)
3. **free-slot fraction** = (K − served) / K  (0 ⇒ full)
4. UAV throughput (Mbps) / (`per_ue_throughput_mbps · K`)
5. **off-duration** of UAV k, normalized (0 if ON) — lets DRL learn the
   switching-cost term (see [REWARD_SPACE.md](REWARD_SPACE.md) §4). MAB does
   **not** need this; it gets the cost via `cost_vector`.

Global tail (4 features):
6. episode progress = `current_index / len(trace)`
7. `disconnected_ue_count / ue_count`
8. `active_uav_count / N`
9. `aggregate_throughput_mbps / (per_ue_throughput_mbps · ue_count)`

Vector length = **`5N + 4`**. Mean-distance was **dropped** — under LoS distance
doesn't drive service, so capacity utilization + free slots + off-duration are
what matter for the ON/OFF decision.

### 2.5 DRL per-UAV local state (`dqn_multi`, future) — length 9 per agent
Features 1–5 above for the agent's own UAV, plus global 6–9. Enables independent
learners with a shared or per-agent network. Optionally append a 1-hot agent id
(off by default).

## 3. Alignment checklist
- [ ] Every feature is derivable from `obs` with no ns-3 / fidelity fields.
- [ ] Fixed `N` per campaign ⇒ fixed vector length ⇒ no ragged tensors.
- [ ] Normalization constants (`throughput_reference_mbps`,
      `max_coverage_distance_m` (~150 m in 400×400), `sinr_min/max_db`) are the
      env's, single-sourced. Distances are in the rescaled 400×400 frame
      (see [TOPOLOGY.md](TOPOLOGY.md)).
- [ ] Ordering `UAV_1..UAV_N` is stable across reset/steps/episodes.
- [ ] Stateless MAB and rich DRL both read the **same** `obs`.
- [ ] No positional-control features yet (UAVs fixed) — leave room for
      `(x, y, z, battery_soc)` per UAV when repositioning/energy land.

## 4. Not in the state yet (future hooks)
- UAV position / velocity / altitude (repositioning phase).
- Battery State-of-Charge (UAV energy model phase).
- Time-of-day / handover history.
