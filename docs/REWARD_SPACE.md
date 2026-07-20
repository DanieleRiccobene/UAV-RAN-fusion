# Reward Space Specification

**Purpose:** define the reward signal, taken **directly from the user's ns-3
`EnergySavingEnv`** (both the DQN/PPO variant `es_env_dqn` and the CLARA variant
`es_env_GTcmab`), mapped onto our LoS RAN-Fusion sim. Computed in
`RANTraceEnv.compute_reward_components` in [rl/ran_trace_env.py](../rl/ran_trace_env.py).

## 0. ns-3 KPM → our-sim quantity mapping

Their reward uses ns-3 KPMs we don't produce; we map to the LoS-sim equivalents:

| ns-3 term (their code) | weight | our-sim quantity |
|---|---|---|
| `throughput_mbps` (from `SUM_QosFlow.PdcpPduVolumeDL`) | `+1` | `aggregate_throughput_mbps` |
| `SUM_RLF_VALUE` (RLF counter **+ unconnected UEs**) | `−1` | `disconnected_ue_count` (outage) |
| `consumed_power_watts` (from `remainingEnergy` Joules Δ) | `−0.001` | per-UAV `current_power_w` (our 120–260 W model) |
| `ES_ON_COST` | `×0.1` | `switch_cost` (§4) |
| `SUM_TB.TotNbrDl.1` (transport blocks) | `−0.1`/`−0.5` | **no analog → dropped** (see §5) |
| `ZERO_COUNT` (active-cell count) | `−0.1` (DQN only) | `active_uav_count` — **optional**, off by default |

## 1. Extracted parameters (all from your code; only γ′ is new)

| param | value | where in your code |
|---|---|---|
| `α` (throughput coeff) | **1.0** | `throughput_mbps` term, both envs |
| `w_rlf` (disconnected penalty) | **1.0** | `−1·SUM_RLF_VALUE`, both envs |
| `w_e` (energy/power weight) | **0.001** | `−0.001·consumed_power_watts` |
| `w_s` (switching weight) | **0.1** | `0.1·ES_ON_COST` (CLARA cost) / `−0.1·SUM_ES_ON_COST` (DQN) |
| `Cf` | **1.0** | `self.Cf = 1` |
| `λf` | **0.1** | `self.lambdaf = 0.1` |
| `time_factor` | **0.01** | `self.time_factor = 0.01` |
| `K` (UEs per UAV) | **10** | `maxUsersPerGnb` default |
| **`γ′`** (scheme-B OFF penalty) | **NEW — propose 50** | not in code; see §3 |

## 2. The three reward pieces (our unified sim)

Per control step, from env state:

```
G          = α·tput_mbps − w_rlf·disconnected_ue        # shared performance (system objective)
energy_k   = −w_e·power_w_k                              # per-UAV, own battery drain (120–260 W model)
switch_k   = w_s·switch_cost_k                           # per-UAV, activation cost (§4)
```

- `G` is **identical for MAB and DRL** so the comparison is clean. (Your ns-3
  code had minor secondary-weight drift between the two envs — TB weight 0.1 vs
  0.5, an active-count term only in DQN — which I've normalized away; the *only*
  intended MAB↔DRL difference is where the switching cost lives, §4. Flag if you
  want the exact per-env weights preserved instead.)
- `power_w_k` comes straight from the existing `LoadDependentEnergyModel`
  (idle 120 W → max 260 W), so **the energy term is real now**, not a placeholder.

## 3. CLARA per-agent reward & credit-assignment schemes

CLARA consumes `rewards_dict[UAV_k] = [global_k, individual_k]`:
```
individual_k = energy_k = −w_e·power_w_k          # own choice only (both schemes)
```
`global_k` depends on the **credit scheme** (both implemented, `--clara-credit`):

| Scheme | `global_k` | Default? |
|---|---|---|
| **B — shared + OFF penalty** | `G + (off_k ? −γ′·disconnected_ue : 0)` | **yes** |
| **C — difference reward** | `D_k = G(a) − G(a with UAV_k flipped)` | selectable |

- **γ′ is the only new parameter.** You want it **high** ("no disconnected UEs
  unless absolutely necessary"). It must exceed the throughput a single UE
  contributes, so an agent never turns off if that strands a UE it could serve.
  Given per-UE throughput is O(10–20 Mbps) in our sim, I propose **γ′ = 50**
  (configurable `--clara-off-penalty`), i.e. one stranded UE ≳ 2–5 UEs' worth of
  served throughput. Confirm or set your own.
- Switching cost is **never** in these terms — CLARA sees it via `cost_vector`
  (§4). Scheme B's γ′ term is a *disconnection* penalty, not the switch cost.
- Scheme A (your current code: pure shared `G`, no OFF penalty) = scheme B with
  γ′ = 0, so it's reachable too.

## 4. Switching (activation) cost — in-reward for DRL, action-selection for MAB

Per UAV k, exactly your `es_on_cost_calculation`:
```
switch_cost_k = Cf · (1 − λf) ** (off_duration_k · time_factor)   (0 while the UAV is ON)
```
- Charged **while a UAV is OFF**, and **decays** the longer it stays off: just
  turned off ⇒ ≈`Cf` (so flipping it back on immediately is discouraged →
  anti-ping-pong); off a long time ⇒ →0 (cheap to reactivate). Matches "the
  longer it stayed off, the less the cost to reactivate."
- `off_duration_k`: seconds UAV k has been continuously OFF. (Your code uses ns-3
  ms timestamps with a `+100` offset and `time_factor=0.01`; in our 1 s-step
  sim I keep `Cf=1, λf=0.1` and will **rescale `time_factor` to seconds** so the
  decay spans ~tens of seconds — a small calibration, flagged.)

| Family | Training reward | Switch cost acts… |
|---|---|---|
| **MAB / CLARA** | `Σ_k(global_k + individual_k)` (no switch term) | **in action selection** — `cost_vector[k] = switch_cost_k`, passed to `negotiate` (your `_get_info()['agent_costs']` = `0.1·es_on_cost`). Keeps CLARA a bandit. |
| **DRL / DQN, PPO** | `r_drl = G + Σ_k energy_k − Σ_k switch_k` | **in the reward** (as in the DRL papers; DRL state carries off-duration → Markovian). |

So `r_drl = r_mab − Σ w_s·switch_cost` — your "DRL reward = MAB reward + switching
cost."

## 5. Dropped / optional terms
- **`SUM_TB.TotNbrDl.1`** (downlink transport blocks) and **`ZERO_COUNT`**: ns-3
  scheduler metrics with no clean LoS-sim analog. **Dropped by default.**
  `ZERO_COUNT ≈ active_uav_count` can be re-enabled as `−w_active·active_uav` if
  you want an explicit "fewer active UAVs" pressure beyond the energy term.
- If you'd rather I reproduce each ns-3 env's exact reward (including these), say
  so — I defaulted to the clean unified `G` for fair MAB↔DRL comparison.

## 6. Comparable reward (evaluation only)
For **every** agent, log `r_cmp = G + Σ_k energy_k − Σ_k switch_k`. For DRL this
equals its training reward; for MAB the switch cost is subtracted **only for
display** (MAB is trained without it). This is the single fair axis for the
cross-agent chart. Never fed to any learner. See [EVALUATION.md](EVALUATION.md).

## 7. Episode return & cadence
Return `= Σ_t r_t` over the full trace; primary campaign metric (mean/std over
100 episodes), per `N`. If `control_step_seconds > 1 s`, the held-interval reward
is the **mean** of per-second sub-step rewards. At 1 s cadence: plain per-second.

## 8. Alignment checklist
- [ ] Same `α, w_rlf, w_e, w_s, Cf, λf, time_factor, K` across all agents.
- [ ] `G` identical for MAB and DRL; only the switch-cost placement differs.
- [ ] Energy term = real per-UAV `power_w` from the 120–260 W model.
- [ ] CLARA: `[global_k, individual_k]` per selected scheme; switch cost only in
      `cost_vector`.
- [ ] `r_cmp` logged for every agent for the unified chart.
- [ ] Reward deterministic given actions + trace + seed.
