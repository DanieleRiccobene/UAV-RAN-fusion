# Agent Roster & Integration Spec

**Purpose:** map the control solutions we will test onto one env interface, so
they are swappable and comparable. Covers the three user-supplied scripts plus
the existing baselines. Flagship = **CLARA**.

## 1. Roster (first batch)

| id | Script | Family | Single/Multi | Action encoding | State it needs |
|---|---|---|---|---|---|
| `clara` | `GTcmab.py` (`MultiAgentGameMAB`) | Game-theoretic contextual MAB | **multi** | per-UAV {0,1} via Best-Response negotiation | context = other UAVs' ON/OFF (len N−1) |
| `dqn` | `discrete_dqn.py` (`DQNAgent`, Double DQN) | DRL value-based | single | index into enumerated valid masks | global vector (5N+4) |
| `ppo` | `ppo_discrete.py` (`PPOAgent`, PPO-Clip) | DRL policy-gradient | single | index into enumerated valid masks | global vector (5N+4) |
| `ucb_stateless` | existing | MAB (UCB) | single | index into masks | ∅ |
| `tabular_single` / `tabular_multi` | existing | tabular Q | single / multi | mask / per-UAV | mask / bucketed load |

CLARA is our method; DQN, PPO, UCB, and tabular Q are baselines.

## 2. The env-facing Agent interface

All agents are driven by the same episodic loop via a thin adapter:

```python
class Agent(Protocol):
    def begin_episode(self, obs) -> None: ...
    def act(self, obs) -> dict:               # -> { UAV_k: "ON"|"OFF" }
    def observe(self, obs, action, reward_info, next_obs, done) -> None: ...
    def end_episode(self) -> None: ...
```

`reward_info` is the decomposed reward (see [REWARD_SPACE.md](REWARD_SPACE.md)):
`{ total, global, per_uav: {UAV_k: individual_k} }`. Scalar agents read
`reward_info.total`; CLARA reads `global` + `per_uav`.

## 3. Per-agent adapters

### 3.1 CLARA (`MultiAgentGameMAB`) — flagship, multi-agent
Native API: `negotiate(previous_actions, cost_vector, max_steps) -> (actions, converged)`
and `update(rewards_dict, cell_list)`.
- `num_agents = N`. UAV order fixed (`UAV_1..UAV_N`), index ↔ UAV stable.
- **`begin_episode`**: `previous_actions = center-on vector` (all others off, or
  all-on — choose one, keep constant). Store `cell_list = [UAV_1..UAV_N]`.
- **`act`**: build `cost_vector` (per-UAV activation cost, §4), call `negotiate`,
  convert the 0/1 vector to `{UAV_k: ON/OFF}`. Cache it as `previous_actions`.
- **Min-active fix:** the script forces `current_actions[0]=1` when all off — must
  be changed to force the **center** UAV's index instead (topology invariant).
- **`observe`**: build `rewards_dict[UAV_k] = [global_k, individual_k]` per the
  selected **credit-assignment scheme** and call `update(rewards_dict, cell_list)`:
  - **B `shared_off_penalty` (default):** `global_k = G + (off_k ? −γ′·disconnected : 0)`.
  - **C `difference`:** `global_k = D_k = G(a) − G(a with UAV_k flipped)`, computed
    by re-running the env association with UAV k's bit flipped (cheap, N passes).
  - `individual_k = −w_e·power_w_k` (own energy, w_e=0.001) in both. Flags
    `--clara-credit`, `--clara-off-penalty` (γ′).
- Contextual: state = the *other* agents' actions; it does **not** use UE-level
  features. That's by design. Credit scheme only changes the reward it's fed, not
  its negotiation logic.

### 3.2 DQN (`DQNAgent`) — single-agent, Double DQN
Native API: `select_action(state)->idx`, `decode_action(idx)`, `replay_buffer.push(...)`, `update()`.
- Build `action_list = build_action_space(N, min_active_uavs=1)` (enumerated
  valid masks). `state_dim = 5N+4`. `n_actions = len(action_list)`.
- **`act`**: `state = global_vector(obs)`; `idx = select_action(state)`;
  `mask = decode_action(idx)` → `{UAV_k: ON/OFF}`.
- **`observe`**: `push(state, idx, reward_drl, next_state, done)`; call
  `update()` **every step** (guarded by batch size internally). Matches the
  user's `energy_saving_drl.py` loop (DQN updates each step). `reward_drl`
  includes the switching cost — see [REWARD_SPACE.md](REWARD_SPACE.md) §4.
- Invalid-mask handling: enumerated list already contains only valid masks, so no
  masking needed at the Q-head for the discrete list. (If we later switch to a
  per-UAV bit head, add −∞ masking.)

### 3.3 PPO (`PPOAgent`) — single-agent, PPO-Clip
Native API: `select_action(state)->idx`, `store(reward, done, next_state)`, `update()`.
- Same `action_list` / `state_dim` / `n_actions` as DQN.
- **`act`**: `idx = select_action(global_vector(obs))` → decode to mask.
- **`observe`**: `store(reward_drl, done, next_state)` (switching cost included).
- **`end_episode`**: call `update()`. **Confirmed from the user's
  `energy_saving_drl.py` loop: PPO updates once per episode** (after all steps),
  DQN every step. With our full-trace episodes (~600 steps, minibatch 64,
  n_epochs=10) that's ~9k gradient steps over 100 episodes — plenty. If learning
  stalls we can add intra-episode updates, but default = per-episode for fidelity
  to the paper.

## 4. `cost_vector` for CLARA = the activation (switching) cost
`cost_vector[k]` is fed into `negotiate` each step; keeping the previous action
is free and switching costs `cost_vector[k]`, so it resists toggling. Value =
`w_s · switch_cost_k` with `w_s = 0.1`, matching your `_get_info()['agent_costs']
= 0.1·es_on_cost`:
```
switch_cost_k = Cf · (1 − λf) ** (off_duration_k · time_factor)   (0 while UAV k is ON)
Cf = 1.0,  λf = 0.1,  time_factor = 0.01  (time_factor rescaled to our 1 s steps)
```
Charged **while OFF** and **decaying** with off-duration: just turned off ⇒ ≈`Cf`
(hard to flip back on → anti-ping-pong); off a long time ⇒ →0 (cheap to
reactivate).

- **For CLARA this cost is in *action selection only*, never in the reward** —
  keeps it a bandit (your explicit design). DRL puts the same cost *in* the reward.
  See [REWARD_SPACE.md](REWARD_SPACE.md) §4 and [EVALUATION.md](EVALUATION.md).

## 5. Fairness rules (so comparisons are valid)
- [ ] All agents see the **same** env, trace, rescaling, N, K (capacity), seeds.
- [ ] Discrete single-agent agents share **one** `action_list` (same index→mask).
- [ ] Same `α, w_rlf, w_e, w_s, Cf, λf, time_factor` and reward decomposition for all.
- [ ] `min_active_uavs=1` (center) enforced identically (env-side projection is
      the source of truth; CLARA's internal all-off guard mirrors it).
- [ ] Report identical per-episode metrics (return, throughput, active UAVs,
      outage rate, energy) across agents.

## 6. Where the scripts live (Phase 5)
Vendored under `rl/agents/` (`clara.py`, `dqn.py`, `ppo.py`) with a small
`rl/agents/adapters.py` implementing the `Agent` protocol around each. The
`--agent {clara,dqn,ppo,ucb_stateless,tabular_single,tabular_multi}` flag selects
one. torch is added to `requirements.txt` for DQN/PPO.
