# RAN Fusion → UAV xApp Testbed: Refactor Plan

**Status:** proposal for review · **Date:** 2026-07-06

## 1. Goal

Turn RAN Fusion from a *digital-twin driver for ns-3* into a **self-contained,
fast simulator for benchmarking xApps** that control a fleet of **UAV-mounted 5G
gNBs**. No ns-3, no multi-fidelity. The simulator must let us drop in and compare
different **MAB** and **DRL** control solutions, in both **single-agent** and
**multi-agent** flavours.

### Locked design decisions (from planning session)

| Decision | Choice |
|---|---|
| ns-3 / external reward | **Removed.** Reward comes only from the internal env. |
| Fidelity levels + fidelity MAB | **Removed.** Single fidelity = full per-UE trace. |
| Training structure | **Episodic.** 1 episode = full ~10-min trace replay. |
| Episodes per run | **100** (configurable). |
| Control cadence | **Configurable, default 1 s** (== trace step). |
| Action space | Per-UAV **ON/OFF** (sleep/wake). Positions fixed. |
| Fleet size | **Campaigns at N = 3, 5, 7, 9 UAVs**, 100 episodes each. |
| Placement | **Dice-face patterns** on a 3×3 grid in a **400×400 m** area; center UAV always present. |
| UE trace | **Rescaled** into the 400×400 area (isotropic, centered). |
| `min_active_uavs` | **1**, and the mandatory-active UAV is the **center**. |
| Bottleneck | **LoS** → per-UAV **capacity K UEs** (default 10) is the binding constraint + a **sweep knob**. |
| Reward | From your ns-3 envs: `G = tput_mbps − w_rlf·disconnected`; per-UAV energy `−w_e·power_w`; switch cost `w_s·es_on_cost`. Params extracted from code ([REWARD_SPACE.md](REWARD_SPACE.md) §1); only **γ′** (scheme-B OFF penalty) is new. |
| Energy | **Real now** — per-UAV power from the 120–260 W `LoadDependentEnergyModel` (`−0.001·W`); UAV-specific propulsion model swaps in later. |
| Agents | Integrate **CLARA** (flagship, multi-agent cMAB) + **DQN** + **PPO** (user-supplied) behind one interface; keep UCB/tabular baselines. |

The contracts are specified in the sibling docs:
[STATE_SPACE.md](STATE_SPACE.md) · [ACTION_SPACE.md](ACTION_SPACE.md) ·
[REWARD_SPACE.md](REWARD_SPACE.md) · [TOPOLOGY.md](TOPOLOGY.md) ·
[AGENTS.md](AGENTS.md) · [EVALUATION.md](EVALUATION.md).

**Switching-cost asymmetry (key):** MAB (CLARA) keeps the activation/switching
cost *out* of its reward (it biases action selection only → stays a bandit); DRL
(DQN/PPO) puts it *in* the reward (as in their source papers). So
`r_drl = r_mab − w·switch_cost`. For fair comparison we log a **comparable
reward** `r_cmp = r_base − w·switch_cost` for every agent and produce three charts
(DRL-native, MAB-native, unified-on-`r_cmp`). See
[REWARD_SPACE.md](REWARD_SPACE.md) §4 + [EVALUATION.md](EVALUATION.md).

## 2. What stays, what goes

### Keep (core of the simulator)
- `rl/ran_trace_env.py` — the env (stepping, association, SINR/AMC throughput,
  energy, reward). This is already self-contained and computes a full reward
  every step. **This is our reward source now.**
- `network/` — gNB/cell/sector/UE managers, `energy_model.py` (120–260 W
  load-dependent model, becomes the bridge to the UAV energy model).
- `rl/trace_mobility.py`, `rl/trace_serving.py`, `rl/bs_mapping.py` — trace
  parsing and UE→UAV association.
- The three existing controllers as our first **MAB / tabular** baselines:
  - `ucb_stateless` → **MAB, single-agent** (bandit over ON/OFF masks)
  - `single_agent` → tabular Q, joint action
  - `multi_agent` → tabular Q, per-UAV agent

### Remove / gut
- `rl/fidelity_mab.py` — **delete** (fidelity MAB, unrelated to control MAB).
- `rl/reward_listener.py` — **delete** (ns-3 reward socket).
- `rl/fidelity_provider.py` — **collapse** to a single full-trace provider
  (keep `IndividualMobilityProvider`, drop medium/low + `FidelityLevel`).
- `scripts/train_rl_agent.py` — major surgery (see Phase 3–4). Remove:
  socket export, reward listener wiring, `morabito_*` feedback/plots, fidelity
  MAB orchestration, sliding-window loop.
- Socket protocol, ports 5001/5002, retry logic — gone.

### Terminology
Internal classes stay named `gNodeB*` (mass-rename is high-risk and low-value);
**UAV terminology is exposed at the CLI, config, docs, and output layer**
(`--num-uavs`, `UAV_k` labels). Flag this if you'd rather do a full rename.

## 3. Phased implementation

Each phase is independently runnable/reviewable.

### Phase 0 — Specs (this PR)
Land these four docs. Review that state/action/reward match intent. No code.

### Phase 1 — Cut ns-3 and the external reward path
- Delete `reward_listener.py`; remove `RewardSocketListener`,
  `send_configuration_and_wait_for_external_reward`, all `socket_export`,
  `send_best_configuration_*`, and `morabito_*` code paths from the orchestrator.
- Remove CLI: `--socket-host/-port`, `--reward-port`, `--reward-timeout-sec`,
  `--enable/disable-socket-export`, `--ignore-first-external-reward`,
  `--socket-timeout-ms`.
- Reward used for learning = `info["reward"]` from `env.step()`.
- **Exit check:** a window still trains, now scored purely by internal reward.

### Phase 2 — Cut fidelity
- Delete `fidelity_mab.py`; strip `FidelityLevel`, `build_fidelity_provider`
  branching down to the single full provider.
- Remove CLI: `--fidelity-level`, `--enable-fidelity-mab`, `--mab-algorithm`
  (fidelity one), `--mab-*`, `--medium-snapshot-sec`,
  `--real-time-budget-{high,medium,low}-sec`.
- `main()` loses the two-branch (manual vs MAB) structure → one path.

### Phase 3 — Episodic loop
- Replace `build_window_start_times` / sliding-window loop with:
  ```
  for episode in range(num_episodes):        # default 100
      obs = env.reset()
      agent.begin_episode(obs)
      done = False
      while not done:                        # full trace
          action = agent.act(obs)            # every control_step_seconds
          next_obs, reward, done, info = env.step(action)
          agent.observe(obs, action, reward, next_obs, done)
          obs = next_obs
      agent.end_episode()
  ```
- New CLI: `--num-episodes` (100), `--control-step-seconds` (1.0),
  `--min-active-uavs` (replaces `--min-active-gnbs`).
- `control_step_seconds > trace_step` ⇒ hold action, accumulate reward across
  held steps (mean or sum — see [REWARD_SPACE.md](REWARD_SPACE.md)).
- Per-episode metrics: return, mean throughput, mean active UAVs, outage rate,
  energy. Learning curves over 100 episodes replace per-window plots.
- **Exit check:** 100-episode run produces a monotone-ish learning curve for a
  tabular agent on N fixed UAVs.

### Phase 4 — Spatial model + UAV fleet parametrization (N = 3/5/7/9)
See [TOPOLOGY.md](TOPOLOGY.md) for the full geometry.
- **400×400 m service area.** Add a one-time UE-trace rescaling transform
  (isotropic, centered) applied at env init; UAV positions become synthetic
  dice-face anchors, **not** the BS lat/lon.
- `--num-uavs N`: env places N UAVs on the dice-face pattern (center always
  included). UE→UAV association already uses `nearest_active`, so N just changes
  how many anchors are instantiated.
- **LoS + capacity:** set `max_coverage_distance_m` ≥ ~285 m (range never binds)
  and expose `--max-ues-per-uav K` (default 10) as the binding constraint and a
  sweep knob. Association already spills to the next-nearest free UAV
  (`nearest_active`), so this needs config plumbing, not new logic.
- Optional `los` throughput mode (capacity shared over admitted UEs) — see
  [TOPOLOGY.md](TOPOLOGY.md) §4.
- `--min-active-uavs 1` with the **center** UAV as the forced-on anchor.
- Campaign runner (`scripts/run_campaign.py` or a `--campaign` flag): loops
  `N ∈ {3,5,7,9}`, 100 episodes each, writes one output subtree per N, plus a
  cross-N comparison summary.
- **Exit check:** four campaigns run end-to-end; comparison plot of best return
  vs N; a topology plot shows UAVs on the dice grid with rescaled UEs.

### Phase 5 — Agent interface + integrate CLARA / DQN / PPO
See [AGENTS.md](AGENTS.md) for the full roster and adapters.
- Define one `Agent` protocol (`begin_episode / act / observe / end_episode`)
  with a **decomposed** `reward_info` so single- and multi-agent share it.
- `--agent {clara, dqn, ppo, ucb_stateless, tabular_single, tabular_multi}`
  (supersedes `--control-approach`).
- Vendor the user-supplied scripts under `rl/agents/` and write thin adapters:
  - **CLARA** (`MultiAgentGameMAB`): map `negotiate`/`update`; fix the all-off
    guard to force the **center** UAV; feed per-UAV `[global_k, individual_k]`
    reward under a selectable credit scheme `--clara-credit
    {shared_off_penalty(default), difference}` — both implemented so campaigns can
    compare them ([REWARD_SPACE.md](REWARD_SPACE.md) §3, [EVALUATION.md](EVALUATION.md) §4b).
  - **DQN** (Double DQN) and **PPO** (PPO-Clip): global state vector (5N+4),
    enumerated valid-mask `action_list`, scalar `total` reward.
- Add `torch` to `requirements.txt`.
- **Exit check:** CLARA, DQN, PPO each run a full N-campaign with no env changes;
  learning curves + cross-agent comparison plot.

### Phase 6 — UAV energy model (separate task, not now)
- Replace the 120–260 W `LoadDependentEnergyModel` (which currently supplies
  `power_w_k`) with a UAV power model (propulsion/hover + comms + battery/SoC).
  Reward stays structurally the same — only `power_w_k` changes source. Hook
  points marked in [REWARD_SPACE.md](REWARD_SPACE.md) §2.

## 4. Decisions — resolved (2026-07-06)
1. **Placement:** dice-face patterns on a 3×3 grid in a 400×400 m area; center
   UAV always present; UE trace rescaled into that area. See
   [TOPOLOGY.md](TOPOLOGY.md).
2. **`min_active_uavs = 1`**, forced-on UAV = the **center** anchor.
3. **Keep internal `gNodeB*` names**, expose UAV only at CLI/config/output.
4. **DRL:** implement a basic DQN (hand-rolled, torch) + a basic ε-greedy agent
   now; user provides more agent scripts later.
5. **Reward over held steps** (only relevant when `control_step_seconds > 1`):
   **mean** of the per-second rewards during the held interval (keeps reward
   scale independent of cadence). At the default 1 s cadence this is a no-op.

### Also resolved (2nd round)
6. **Placement:** N=3 = **diagonal** (dice-3); N=7 = left/right side mid-points.
   **Confirmed.**
7. **Physics:** LoS ⇒ per-UAV **capacity K UEs** (default 10) is the bottleneck
   and a sweep knob; range non-binding. Uses existing `nearest_active` +
   `max_ues_per_gnb`.
8. **Agents:** integrate user-supplied **CLARA / DQN / PPO** (not hand-rolled).

### Also resolved (3rd round)
9. **PPO cadence:** from the user's `energy_saving_drl.py` loop — **PPO updates
   once per episode**, **DQN every step**. Instantiation uses `batch_size=64`,
   `lr=1e-5`, `gamma=0.99`, `gae_lambda=0.95` (not the class defaults).
10. **Switching cost = activation cost** `Cf·(1−λf)^(off_dur·time_factor)`:
    in-reward for DRL, action-selection-only for CLARA; comparable-reward plots
    for fairness (§ above, [EVALUATION.md](EVALUATION.md)).

### Also resolved (4th round)
11. **CLARA credit assignment:** implement **both** scheme B (`shared_off_penalty`,
    **default**) and scheme C (`difference` reward); selectable via `--clara-credit`
    so campaigns compare them. `individual_k = β·energy_k` (own energy) in both;
    switching cost stays out of reward. Per-UAV reward = `[global_k, individual_k]`.
12. **Reward shape:** `system G = α·tput + γ·disconnected`; single-agent
    `total = G + Σ β·energy_k`; DRL subtracts `w·switch_cost`.

### Still to confirm (values, not structure)
- **`Cf, λf, time_factor`** (switching cost) from the paper; meaning of
  `off_duration`/`diff = ∞`.
- **`γ'`** OFF-penalty weight for CLARA scheme B.
- **`w`** switching-cost weight in DRL reward / comparable reward (default 1).
- **`β` / energy term:** active-count proxy now; real per-UAV power later.
