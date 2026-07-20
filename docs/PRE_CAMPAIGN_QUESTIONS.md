# Questions to resolve BEFORE the experimental campaign

Raised by the user 2026-07-06. Implementation proceeds now; **ask these before
launching the real N∈{3,5,7,9} × 100-episode campaign.**

1. **Initial UE distribution scenario.** Do we want a scenario with a *specified*
   initial UE distribution (e.g. seed all UEs in a given spatial layout at t=0)
   rather than whatever the trace starts with? Affects env reset + placement.

2. **Random UE movement instead of fixed traces.** Option to drive UE mobility
   with a random-motion model instead of the recorded CSV traces — this would let
   us drop the control step below 1 s (`time_factor < 1s`), since we'd no longer
   be locked to the 1 s trace granularity. Affects the mobility provider + cadence.

3. **Additional state-of-the-art baselines.** Which extra SOTA methods (beyond
   CLARA / DQN / PPO / UCB / tabular) should be added to the comparison?

4. **Per-UE QoS throughput requirements.** ✅ **IMPLEMENTED** (2026-07-06). Each
   UE now carries a downlink demand; delivered throughput = Σ admitted demands,
   capped by per-UAV capacity (Mbps). Three scenarios via flags:
   `--full-rate` (all 10 Mbps), `--split-half` (50% 14 / 50% 6),
   `--split-2080` (20% 20 / 80% 7.5). All three average 10 Mbps/UE, i.e. the
   SAME total offered load, differing only in heterogeneity.
   `--uav-capacity-mbps` sets the cap
   (default K × per-ue rate). Verified: policy adapts (2→3 active UAVs from
   full_rate→split_20_80). Remaining question if wanted: penalize *admitted but
   unsatisfied* demand as a distinct QoS-violation term (today unmet demand just
   lowers throughput; only unadmitted UEs count as RLF/outage).
