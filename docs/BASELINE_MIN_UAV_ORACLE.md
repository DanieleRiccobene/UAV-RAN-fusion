# Implementation Plan — [5] Min-UAV Oracle Baseline

Baseline reference: **[5]** — "minimize the number of active UAVs subject to
covering demand." Added as an **offline optimality reference** (genie), not a
runnable xApp.

## What it is, in our env

At each control step the oracle is allowed to peek at the **current** frame's UE
positions and pick the **smallest active set that serves everyone** (zero
outage), falling back to max-coverage when the frame is oversubscribed. It
becomes the "you cannot do better than this many RUs at zero RLF" reference line
in every plot — the lower envelope of the active-RU curves at iso-outage.

It is explicitly a **genie**: it sees current UE positions before deciding and
is switching-cost-blind. Label it as such in the paper.

## Core algorithm (per step)

Given the current frame (UE positions), find the active mask `S` that:

1. satisfies `|S| ≥ min_active_uavs` (center-optional — same action set as CLARA);
2. is evaluated with the env's **own** greedy association (`env._associate`), so
   feasibility reflects reality, not an idealized assignment;
3. minimizes lexicographically:
   **(a) outage → (b) number of active UAVs → (c) −throughput → (d) churn**
   (Hamming distance to the currently-active set, to break ties without needless
   toggling).

Because `N ≤ 9`, brute-force over all `2^N` masks (≤ 512) is trivial — reuse
`enumerate_masks(env.num_uavs, env.min_active_uavs)` (already used by UCB);
center-optional, so no filtering.

```python
best = None  # (key, mask); key = (outage, active_count, -throughput, churn)
for mask in enumerate_masks(N, min_active):  # center-optional
    assignment, counts = env._associate(mask, frame)
    outage = len(frame) - sum(counts.values())
    tput, _ = env._throughput(assignment)
    churn = hamming(mask, env.active)
    key = (outage, sum(mask), -tput, churn)
    if best is None or key < best[0]:
        best = (key, mask)
return mask_to_action(best[1], env.uav_ids)
```

This yields **min UAVs at zero outage** when the frame is coverable, and
**max coverage with fewest UAVs** when it is not (oversubscribed frames with
> N·K UEs).

## Key integration details

- **Peeking**: in `act(obs)`, read the upcoming frame directly —
  `frame = env._ue_by_index[env.current_index]` (`step()` reads that index, then
  increments). `env._associate` / `env._throughput` are side-effect-free, so
  evaluating candidates does not perturb env state
  (see `rl/uav_env.py` `_associate` / `_throughput`).
- **Determinism + speed**: the oracle's choice depends only on the frame index
  (trace fixed, association deterministic), so **memoize `frame_index → mask`**.
  Episode 1 fills the cache (≤ 512 × ~60 × 9 ops/frame × 900 frames ≈ 250M once);
  episodes 2–100 are cache hits. Per-episode metrics are flat lines — correct and
  expected for an oracle.
- **No learning**: `begin_episode` / `observe` / `end_episode` are no-ops; reward
  metrics (`return_base`, `return_cmp`, RLF, active RUs, switching) are recorded
  by the training loop from the env exactly like every other agent, so
  plots/CSVs need zero special-casing.

## Files to touch

1. **`rl/agents/oracle.py`** (new): `MinUavOracleAgent` implementing the Agent
   protocol above, with the per-frame cache.
2. **`rl/agents/registry.py`**: add `"oracle"` → `MinUavOracleAgent(env)` in
   `build_agent`.
3. **`scripts/train_uav.py`**: add `AGENT_SPECS["oracle"] = ("oracle", None, "ORACLE")`.
4. **`rl/plots.py`** + **`scripts/plot_uav_results.py`**: add `"oracle"` to
   `TOKEN_MARKERS` (e.g. `"*"`) and `TOKEN_FAMILY`.
5. **`docs/AGENTS.md` / `docs/EVALUATION.md`**: document it as an offline
   optimality reference (genie — peeks at current positions, switching-blind).

## Design choices worth flagging

- **Switching-blind by default** (canonical [5]): it re-optimizes coverage each
  frame and will toggle UAVs as UEs move → it shows a *high activation cost*.
  That is a feature, not a bug — the talking point: "the coverage-optimal oracle
  needs the fewest RUs but churns; CLARA reaches near-oracle RU counts with far
  lower switching cost." The tertiary churn tie-break only trims *free* toggles,
  so it stays a true min-UAV oracle.
- **Optional stable variant** (`oracle_hyst`): add hysteresis (turn a UAV OFF
  only after it has been idle for `h` frames). Gives a second reference line that
  is fairer on switching cost. Add only if wanted — the pure oracle is the [5]
  requested here.

## Validation

- **N=5, full_rate, 20 UEs / K=10** → oracle holds **2 active UAVs, 0 outage**
  (matches CLARA's converged optimum) with **higher switching cost** than CLARA.
- **60 UEs on N=5** (oversubscription) → oracle keeps all 5 on, outage ≈ 60−50 = 10,
  confirming graceful fallback.
- **Cross-N summary** → oracle's active-RU line is the **lower envelope** of all
  learners at iso-outage.
