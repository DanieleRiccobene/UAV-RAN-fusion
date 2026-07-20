# Baselines from OREO (Qazzaz et al.) Table 1 — "no user association" rows

Source: M. M. H. Qazzaz et al., "OREO: Open RAN Energy Optimization via Deep
Reinforcement Learning for 6G Networks," IEEE OJ-COMS, 2026.

## Why these rows

OREO's Table 1 compares energy-management works across several axes. The rows
with **✗ in the "User Association" column** solve *exactly our problem*: **RU/BS
ON/OFF activation only, with association left to a fixed rule** — which is our
env (`UAVEnv` fixes association to nearest-active-UAV-with-a-free-slot). So these
are the most apt learning baselines for the UAV testbed.

| Ref | RL Algorithm | User Assoc. | Status in this repo | Token |
|-----|--------------|-------------|---------------------|-------|
| Bordin **[10]** | PPO, DQN | ✗ | **Already present** = `ppo` + `dqn` | `ppo`, `dqn` |
| Wang **[12]** | DQN | ✗ | **New** — `rl/agents/wang_dqn.py` (original DQN state/action, paper's training hyperparameters) | `wang` |
| Masrur **[16]** | MARL-DDQN | ✗ | **New** — `rl/agents/marl_dqn.py` (double=True) | `masrur` |
| Xu **[18]** | Hierarchical RL | ✗ | **New** — `rl/agents/hierarchical.py` | `xu` |
| Rezaei **[19]** | Multi-agent DQN | ✗ | **New** — `rl/agents/marl_dqn.py` (double=False) | `rezaei` |
| El Amine **[20]** | Tabular Q-learning | ✗ | **New** — `rl/agents/tabular_q.py` | `el_amine` |

The other Table 1 rows (Liang [11], Ntassah [13], Akman [14], Sun [15],
Dang [17], Marzuk [21], OREO itself) all do user association (✓ or partial), so
they don't map cleanly onto our association-free env and are **not** implemented
here.

## What each new baseline does (adapted to our binary ON/OFF UAV env)

All four are standalone agents implementing the common Agent protocol
(`begin_episode` / `act` / `observe` / `end_episode`); none touch CLARA.

- **`el_amine` [20] — tabular Q-learning (multi-sleeping).** Single-agent,
  state-aware Q-learning over the joint ON/OFF mask. Multi-sleep collapses to two
  levels (active/sleep) in our env. Coarse discrete state = (per-UAV active bits,
  UE-load bin); ε-greedy; TD bootstrapping on `reward_drl` (incl. switching cost).
  Torch-free. Distinct from `ucb` (contextless bandit, no bootstrapping).

- **`xu` [18] — hierarchical RL.** HIGH level: tabular Q chooses an activation
  budget k ∈ [min_active, N] from a coarse state. LOW level: activate the k UAVs
  with the highest **recent** served load (previous observation only — no peeking;
  center-optional). Torch-free. A lightweight, faithful-in-spirit adaptation of the
  hierarchical structure (not the deep GAMA/QMIX variant). Note: `xu`'s reachable
  masks are a **structured subset** (budget + load ranking) — inherent to the
  hierarchical method, not the center pin.

- **`rezaei` [19] — multi-agent DQN.** One independent DQN per UAV over the binary
  {OFF, ON} action, all observing the shared global state (5N+4) and trained on
  the shared team reward. Vanilla DQN target. Torch (lazy import).

- **`masrur` [16] — MARL Double-DQN.** Same multi-agent structure as `rezaei` but
  with a Double-DQN target. Torch (lazy import). Shares
  `rl/agents/marl_dqn.MARLDQNAgent` (`double=True`).

- **`wang` [12] — DQN xApp (paper hyperparameters).** Subclasses the base
  `rl/agents/dqn.py` (identical 5N+4 state, enumerated-mask action, double-DQN
  update) but swaps in Wang's reported training hyperparameters: AdamW, lr=1e-4,
  γ=0.99, batch=64, replay=300000, target update every 50 steps, ε 0.9→0.05.
  Distinct curve from `dqn` (which keeps our default hyperparameters). Torch.

## Fidelity notes (be honest in the paper)

- `el_amine`, `xu` are **torch-free** and smoke-tested here. `xu` is a *simplified*
  hierarchical adaptation; label it as "hierarchical-RL-style" rather than a
  reproduction of Xu's deep framework.
- `rezaei`, `masrur` need **torch** (like `dqn`/`ppo`); they are written in the
  same lazy-import style and are **not** smoke-tested on machines without torch.
- All four are association-free by construction (matching the ✗ column), which is
  the reason they map cleanly onto `UAVEnv`.
- **Action space (fairness):** every agent shares CLARA's action set — an ON/OFF
  mask with ≥ `min_active` active, **center-optional** (center is only filled as
  the min-active fallback, exactly as `env._enforce_min_active`). `oracle`,
  `rezaei`, `masrur`, `xu` were relaxed from center-forced to center-optional so
  none is more constrained than CLARA. `xu` still explores a structured subset of
  masks by design (hierarchical budget + load ranking).

## Run

```bash
# torch-free pair
python3 scripts/train_uav.py --agents el_amine,xu --campaign
# full OREO-Table-1 baseline set alongside CLARA (dqn/ppo/rezaei/masrur need torch)
python3 scripts/train_uav.py \
  --agents clara,clara_diff,dqn,ppo,el_amine,xu,rezaei,masrur --campaign
```

Bordin [10] is represented by running both `dqn` and `ppo`. Wang [12] is now its
own agent (`wang`) with the paper's hyperparameters, distinct from `dqn`.
