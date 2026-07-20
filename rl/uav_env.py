"""Lean UAV-gNB environment: LoS + per-UAV capacity, dice placement, episodic.

See docs/TOPOLOGY.md, STATE_SPACE.md, REWARD_SPACE.md. This replaces the
ns-3 / SINR / sliding-window machinery with a self-contained simulator:

- N UAVs at fixed dice-face positions in a 400x400 m area (center always on).
- UE positions come from the mobility trace, rescaled once into the area.
- Association: nearest active UAV with a free slot (capacity K); else outage.
- Throughput: LoS best-effort, per served UE = `per_ue_throughput_mbps`.
- Energy: load-dependent power (idle 120 W -> max 260 W).
- Reward: decomposed (system G, per-UAV energy) + switching cost; CLARA credit
  schemes B (shared+OFF-penalty) / C (difference). See compute_reward_info().
"""

from __future__ import annotations

import math
from collections import defaultdict

from rl.trace_mobility import MobilityTrace
from rl.uav_placement import AreaRescaler, dice_uav_positions

IDLE_POWER_W = 120.0
MAX_POWER_W = 260.0


def _distance(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


class UAVEnv:
    def __init__(
        self,
        *,
        mobility_csv,
        num_uavs,
        max_ues_per_uav=10,
        max_users=None,
        per_ue_throughput_mbps=10.0,
        traffic_scenario="full_rate",
        uav_capacity_mbps=None,
        max_coverage_distance_m=600.0,
        # reward weights (extracted from the user's ns-3 EnergySavingEnv)
        alpha=1.0,            # throughput
        w_rlf=1.0,            # disconnected-UE penalty in G
        w_energy=0.001,       # per-Watt energy penalty
        w_switch=0.1,         # switching-cost weight
        off_penalty=50.0,     # gamma': CLARA scheme-B extra OFF penalty per disconnected UE
        # switching (activation) cost: Cf * (1-lambdaf)**(off_duration*time_factor)
        cf=1.0,
        lambdaf=0.1,
        time_factor=0.5,      # rescaled for 1 s control steps
        min_active_uavs=1,
        credit_scheme="shared_off_penalty",
        seed=None,
    ):
        self.credit_scheme = credit_scheme
        self.num_uavs = int(num_uavs)
        self.max_ues_per_uav = int(max_ues_per_uav)
        self.max_users = max_users
        self.per_ue_throughput_mbps = float(per_ue_throughput_mbps)
        self.traffic_scenario = str(traffic_scenario)
        self.max_coverage_distance_m = float(max_coverage_distance_m)
        self.alpha = float(alpha)
        self.w_rlf = float(w_rlf)
        self.w_energy = float(w_energy)
        self.w_switch = float(w_switch)
        self.off_penalty = float(off_penalty)
        self.cf = float(cf)
        self.lambdaf = float(lambdaf)
        self.time_factor = float(time_factor)
        self.min_active_uavs = int(min_active_uavs)
        self.seed = seed

        # UAV identity / geometry.
        positions, center_index = dice_uav_positions(self.num_uavs)
        self.uav_positions = positions
        self.center_index = center_index
        self.uav_ids = [f"UAV_{i + 1}" for i in range(self.num_uavs)]

        # Load trace and rescale UE coordinates into the area (fit once).
        self.trace = MobilityTrace.from_csv(mobility_csv)
        self.rescaler = self._fit_rescaler()
        self.timestamps = list(self.trace.timestamps)
        # Precompute rescaled UE positions per timestamp: [(uid, x, y), ...].
        self._ue_by_index = [self._rescaled_ues(ts) for ts in self.timestamps]
        self.total_ue_ids = sorted({uid for frame in self._ue_by_index for uid, _, _ in frame})

        # Per-UE downlink throughput demand (Mbps) per traffic scenario.
        self.demand = self._assign_demands(self.traffic_scenario)
        # UAV total serving capacity (Mbps). Default: K * per_ue rate (so K UEs at
        # the nominal rate exactly fill a UAV).
        self.uav_capacity_mbps = (
            float(uav_capacity_mbps) if uav_capacity_mbps is not None
            else self.max_ues_per_uav * self.per_ue_throughput_mbps
        )

        self.current_index = 0
        self.active = [True] * self.num_uavs
        self.off_duration = [0] * self.num_uavs  # steps a UAV has been continuously OFF
        self._last_delivered = {}

    # ------------------------------------------------------------------ setup
    def _fit_rescaler(self):
        points = []
        for samples in self.trace.samples_by_timestamp.values():
            for s in samples:
                if s.x_m is not None and s.y_m is not None:
                    points.append((s.x_m, s.y_m))
        return AreaRescaler.fit(points)

    def _rescaled_ues(self, timestamp):
        samples = self.trace.samples_by_timestamp[timestamp]
        if self.max_users is not None:
            samples = samples[: self.max_users]
        out = []
        for s in samples:
            x, y = self.rescaler.apply(s.x_m, s.y_m)
            if x is None:
                continue
            out.append((str(s.user_id), x, y))
        return out

    def _assign_demands(self, scenario):
        """Assign each UE a downlink throughput demand (Mbps) per traffic scenario.

        All three mixes average `per_ue_throughput_mbps` (r, default 10) Mbps/UE, so
        every scenario carries the SAME total offered load — they differ only in
        heterogeneity (a clean QoS control):
        - full_rate    : every UE at r          (avg r).
        - split_half   : 50% at 1.4r, 50% at 0.6r        (avg r).
        - split_20_80  : 20% at 2.0r, 80% at 0.75r       (avg r).
        With r=10: full=10, split_half=14/6, split_20_80=20/7.5.
        Assignment is deterministic (sorted UE ids) so it is stable across episodes.
        """
        ids = sorted(self.total_ue_ids)
        n = len(ids)
        demand = {}
        r = self.per_ue_throughput_mbps
        scenario = scenario.lower()
        if scenario == "full_rate":
            for u in ids:
                demand[u] = r
        elif scenario == "split_half":
            half = n // 2
            for i, u in enumerate(ids):
                demand[u] = 1.4 * r if i < half else 0.6 * r   # avg r
        elif scenario == "split_20_80":
            k = round(0.2 * n)
            for i, u in enumerate(ids):
                demand[u] = 2.0 * r if i < k else 0.75 * r      # avg r
        else:
            raise ValueError(
                f"Unknown traffic_scenario {scenario!r}; "
                "use full_rate | split_half | split_20_80"
            )
        return demand

    # --------------------------------------------------------------- dynamics
    def reset(self):
        self.current_index = 0
        self.active = [True] * self.num_uavs
        self.off_duration = [0] * self.num_uavs
        self._last_delivered = {}
        return self._observation()

    def _enforce_min_active(self, active):
        """Force at least min_active_uavs ON; the center UAV is the mandatory anchor."""
        active = list(active)
        if sum(active) >= self.min_active_uavs:
            return active
        # Turn on the center first, then remaining UAVs by index, until satisfied.
        order = [self.center_index] + [i for i in range(self.num_uavs) if i != self.center_index]
        for i in order:
            if sum(active) >= self.min_active_uavs:
                break
            active[i] = True
        return active

    def apply_action(self, action):
        """action: dict {UAV_k: 'ON'|'OFF'|bool} or list[bool] of length N."""
        if isinstance(action, dict):
            desired = [self._is_on(action.get(self.uav_ids[i], True)) for i in range(self.num_uavs)]
        else:
            desired = [self._is_on(a) for a in action]
        desired = self._enforce_min_active(desired)
        # Update off-duration BEFORE the step: a UAV freshly OFF starts at 0.
        for i in range(self.num_uavs):
            if desired[i]:
                self.off_duration[i] = 0
            else:
                self.off_duration[i] = (self.off_duration[i] + 1) if not self.active[i] else 0
        self.active = desired

    @staticmethod
    def _is_on(value):
        if isinstance(value, str):
            return value.strip().upper() == "ON"
        return bool(value)

    def step(self, action):
        self.apply_action(action)
        frame = self._ue_by_index[self.current_index]
        assignment, counts = self._associate(self.active, frame)
        reward_info = self.compute_reward_info(
            frame, assignment, counts, credit_scheme=self.credit_scheme
        )
        obs = self._observation(assignment=assignment, counts=counts)
        info = {
            "timestamp": self.timestamps[self.current_index],
            "assignment": assignment,
            "served_counts": counts,
            "reward_info": reward_info,
        }
        self.current_index += 1
        done = self.current_index >= len(self.timestamps)
        return obs, reward_info, done, info

    # ------------------------------------------------------------ association
    def _active_indices(self, active):
        return [i for i in range(self.num_uavs) if active[i]]

    def _associate(self, active, frame):
        """Nearest active UAV with a free slot; else outage. Returns (assignment, counts)."""
        counts = {i: 0 for i in range(self.num_uavs)}
        assignment = {}
        active_idx = self._active_indices(active)
        for uid, x, y in sorted(frame, key=lambda t: t[0]):
            best = None
            best_d = None
            for i in active_idx:
                if counts[i] >= self.max_ues_per_uav:
                    continue
                d = _distance(x, y, *self.uav_positions[i])
                if d > self.max_coverage_distance_m:
                    continue
                if best_d is None or d < best_d:
                    best_d, best = d, i
            if best is None:
                assignment[uid] = None
            else:
                assignment[uid] = best
                counts[best] += 1
        return assignment, counts

    def _served_outage(self, counts, frame):
        served = sum(counts.values())
        outage = len(frame) - served
        return served, outage

    def _throughput(self, assignment):
        """Demand-based delivered throughput (Mbps): each admitted UE gets its
        demand up to the UAV's total capacity (Mbps). Returns (total, per_uav)."""
        per_uav_demand = defaultdict(float)
        for uid, uav_idx in assignment.items():
            if uav_idx is None:
                continue
            per_uav_demand[uav_idx] += self.demand.get(uid, self.per_ue_throughput_mbps)
        per_uav_delivered = {i: min(d, self.uav_capacity_mbps) for i, d in per_uav_demand.items()}
        return sum(per_uav_delivered.values()), per_uav_delivered

    def _system_G(self, active, frame):
        """Evaluate the system objective G for a given active set (side-effect free)."""
        assignment, counts = self._associate(active, frame)
        served, outage = self._served_outage(counts, frame)
        tput_mbps, _ = self._throughput(assignment)
        return self.alpha * tput_mbps - self.w_rlf * outage, served, outage, counts

    # ---------------------------------------------------------------- energy
    def _uav_power_w(self, uav_index, counts):
        if not self.active[uav_index]:
            return 0.0
        load = min(1.0, counts.get(uav_index, 0) / max(1, self.max_ues_per_uav))
        return IDLE_POWER_W + (MAX_POWER_W - IDLE_POWER_W) * load

    def _switch_cost(self, uav_index):
        if self.active[uav_index]:
            return 0.0
        return self.cf * ((1.0 - self.lambdaf) ** (self.off_duration[uav_index] * self.time_factor))

    def cost_vector(self):
        """Per-UAV switching cost for CLARA's negotiation (action selection only)."""
        return [self.w_switch * self._switch_cost(i) for i in range(self.num_uavs)]

    # ---------------------------------------------------------------- reward
    def compute_reward_info(self, frame, assignment, counts, credit_scheme="shared_off_penalty"):
        served, outage = self._served_outage(counts, frame)
        tput_mbps, self._last_delivered = self._throughput(assignment)
        G = self.alpha * tput_mbps - self.w_rlf * outage

        per_uav = {}
        switch_costs = {}
        energy_total = 0.0
        switch_total = 0.0
        for i, uav_id in enumerate(self.uav_ids):
            power_w = self._uav_power_w(i, counts)
            energy_k = -self.w_energy * power_w
            energy_total += energy_k
            switch_k = self.w_switch * self._switch_cost(i)
            switch_costs[uav_id] = switch_k
            switch_total += switch_k

            if credit_scheme == "difference":
                flipped = list(self.active)
                flipped[i] = not flipped[i]
                g_flip, _, _, _ = self._system_G(flipped, frame)
                global_k = G - g_flip
            else:  # shared_off_penalty (default, scheme B)
                off_pen = 0.0 if self.active[i] else self.off_penalty * outage
                global_k = G - off_pen
            per_uav[uav_id] = {"global_k": global_k, "individual_k": energy_k}

        total = G + energy_total  # single-agent scalar (MAB base, no switch cost)
        return {
            "system_G": G,
            "per_uav": per_uav,
            "switch_costs": switch_costs,
            "switch_total": switch_total,
            "energy_total": energy_total,
            "total": total,                          # MAB / base scalar
            "reward_drl": total - switch_total,      # DRL training reward
            "reward_cmp": total - switch_total,      # comparable reward (plots)
            "metrics": {
                "aggregate_throughput_mbps": tput_mbps,
                "served_ue_count": served,
                "disconnected_ue_count": outage,
                "ue_count": len(frame),
                "active_uav_count": sum(self.active),
            },
        }

    # ----------------------------------------------------------- observation
    def _observation(self, assignment=None, counts=None):
        if counts is None:
            counts = {i: 0 for i in range(self.num_uavs)}
        delivered = getattr(self, "_last_delivered", {}) if assignment is not None else {}
        frame = self._ue_by_index[min(self.current_index, len(self.timestamps) - 1)]
        served = sum(counts.values())
        uavs = {}
        for i, uav_id in enumerate(self.uav_ids):
            uavs[uav_id] = {
                "index": i,
                "is_active": bool(self.active[i]),
                "position": self.uav_positions[i],
                "served": int(counts.get(i, 0)),
                "capacity": self.max_ues_per_uav,
                "off_duration": int(self.off_duration[i]),
                "power_w": self._uav_power_w(i, counts),
                "throughput_mbps": float(delivered.get(i, 0.0)),
                "is_center": i == self.center_index,
            }
        return {
            "timestamp": self.timestamps[min(self.current_index, len(self.timestamps) - 1)],
            "episode_progress": self.current_index / max(1, len(self.timestamps)),
            "uavs": uavs,
            "ue_count": len(frame),
            "served_ue_count": served,
            "disconnected_ue_count": len(frame) - served,
            "active_uav_count": sum(self.active),
            "aggregate_throughput_mbps": float(sum(delivered.values())),
        }

    def __len__(self):
        return len(self.timestamps)
