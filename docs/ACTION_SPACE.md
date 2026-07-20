# Action Space Specification

**Purpose:** define the control action and how each solution encodes it, so MAB
and DRL (single/multi) stay interchangeable behind one env.
Env application is `RANTraceEnv.apply_action` in
[rl/ran_trace_env.py](../rl/ran_trace_env.py).

Fleet size `N ∈ {3, 5, 7, 9}` fixed per campaign.

## 1. The physical action

For each control step (default every 1 s), the agent sets the **power state of
each UAV**:

```
action = { UAV_1: ON|OFF, UAV_2: ON|OFF, ..., UAV_N: ON|OFF }
```

- **ON** = UAV radio serving UEs (idle→max power per energy model).
- **OFF** = UAV sleeping (no service; UEs re-associate to nearest active UAV,
  or go into outage if none within `max_coverage_distance_m`).
- Positions are **fixed** (hover points). No movement action in this version.

The env consumes the canonical form (unchanged):
`[{"gnb_id": UAV_k, "state": "ON"|"OFF"}, ...]`.

## 2. Constraints

- `min_active_uavs` (**default 1**): actions with fewer active UAVs are
  projected up to the constraint. **The forced-on UAV is the geometric center**
  ((200,200) anchor, present in every N — see [TOPOLOGY.md](TOPOLOGY.md)), not
  the highest-load one. `enforce_multi_agent_constraints` must be updated to
  prefer the center anchor when forcing the minimum active set.
- `allow_all_off` (default false): with `min_active_uavs=1` the all-OFF action
  is illegal by construction; set true only to study the degenerate no-coverage
  case.
- Invalid/constrained actions are **masked**, not penalized, so learning signals
  stay clean.

## 3. Per-solution encoding

| Solution | Encoding | # actions |
|---|---|---|
| `mab_single` (UCB bandit) | arm = index into enumerated valid masks | ≤ 2^N |
| `tabular_single` (joint Q) | discrete index over valid masks | ≤ 2^N |
| `tabular_multi` (per-UAV Q) | each agent picks {OFF=0, ON=1}; combined then constraint-projected | 2 per agent |
| `dqn_single` | Q-head over enumerated valid masks, **invalid actions masked to −∞** | ≤ 2^N |
| `dqn_multi` | per-agent 2-way head (OFF/ON); joint action = concat + projection | 2 per agent |

Valid-mask enumeration (already implemented as `build_action_space`) drops masks
violating `min_active_uavs` / `allow_all_off`. With `min_active_uavs=1` (only the
all-OFF mask excluded): N=3→7, N=5→31, N=7→127, N=9→511 — all tractable for a
discrete Q-head.

## 4. Unifying Agent interface (Phase 5)

Every solution implements one protocol so the episodic loop is agent-agnostic:

```python
class Agent(Protocol):
    def begin_episode(self, obs) -> None: ...
    def act(self, obs) -> dict:            # returns { UAV_k: "ON"|"OFF" }
        ...
    def observe(self, obs, action, reward, next_obs, done) -> None: ...
    def end_episode(self) -> None: ...
    # single vs multi-agent is internal to the implementation
```

- MAB agents ignore `obs` in `act` (or use bucketed state), update arm stats in
  `observe`.
- DRL agents build the state vector ([STATE_SPACE.md](STATE_SPACE.md)) inside
  `act`/`observe`.
- The loop always receives a full `{UAV_k: state}` mask and passes it to the env.

## 5. Alignment checklist
- [ ] Action = exactly one ON/OFF bit per UAV; length `N`.
- [ ] `min_active_uavs` + `allow_all_off` applied **identically** for all agents
      (shared `enforce_*` / `build_action_space`).
- [ ] Discrete agents share one canonical valid-mask list (same index → same
      mask) so results are comparable across agents.
- [ ] Multi-agent joint action passes through the same constraint projection as
      single-agent.
- [ ] No movement/positioning dimension yet (documented future extension).

## 6. Future extension (not now)
Repositioning would extend each UAV's action to `(power, move)` where `move`
is a discrete step or continuous `(Δx, Δy, Δz)`. State/action docs reserve room;
env geometry + mobility for UAVs would be added then.
