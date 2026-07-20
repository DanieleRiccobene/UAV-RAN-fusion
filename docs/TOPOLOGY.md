# Topology & Spatial Model Specification

**Purpose:** define the physical playground — a 400×400 m area, deterministic
UAV placement (dice-face patterns), and the rescaling of the UE mobility trace
into that area. This replaces the old "UAVs sit at the Colosseo BS coordinates"
assumption entirely.

## 1. Service area

- Square area **400 m × 400 m**, local coordinates `(x, y) ∈ [0, 400] × [0, 400]`.
- All geometry (UAV positions, UE positions, distances, SINR pathloss, coverage)
  is computed in these local meters. The original lat/lon / EPSG `x_m,y_m` of the
  trace are **only** used to derive the rescaling transform (§3), never directly.

## 2. UAV placement — dice faces

UAVs are placed on a fixed **3×3 anchor grid** with an 80 m edge margin:

```
x, y ∈ {80, 200, 320}          # fractions 0.2 / 0.5 / 0.8 of 400 m

(80,320) (200,320) (320,320)     top row
(80,200) (200,200) (320,200)     middle row   ← (200,200) = CENTER
(80, 80) (200, 80) (320, 80)     bottom row
```

The N-UAV configuration is a symmetric subset of these anchors, always including
the **center**:

| N | Pattern | Anchor points |
|---|---|---|
| 3 | **diagonal** line (dice-3) | (80,80), **(200,200)**, (320,320) |
| 5 | quincunx (4 corners + center) — dice 5 | 4 corners + **(200,200)** |
| 7 | corners + center + two side mid-points | 4 corners, (80,200), (320,200), **(200,200)** |
| 9 | full 3×3 grid — dice 9 | all 9 anchors |

- **Center UAV `(200,200)` is the anchor** and is present in every N. It is the
  mandatory-active UAV when the minimum-active constraint kicks in
  (see [ACTION_SPACE.md](ACTION_SPACE.md)).
- Margin (80 m) keeps UAVs off the exact corners ("enough space from the
  corners"). Margin and the {0.2,0.5,0.8} fractions are tunable constants.
- **N=3** is a true diagonal (dice-3): bottom-left, center, top-right.
- **N=7** uses the left/right side mid-points (confirmed).

UAV positions are synthetic and fixed for the whole run; the `gNodeB` objects are
assigned these `(x_m, y_m)` at env init instead of the config lat/lon.

## 3. UE trace rescaling into 400×400

The UE trace (`x_m, y_m`, per UE per second) spans the real Colosseo extent.
We map it linearly into the 400×400 box, computed **once** from the full trace:

1. Bounding box over all UE samples: `[x_min, x_max] × [y_min, y_max]`.
2. **Isotropic scale** (preserve aspect ratio, no distortion):
   `s = 400 / max(x_max − x_min, y_max − y_min)`.
3. Center the scaled cloud in the box (equal padding on the short axis).
4. Apply the same affine `(x,y) → (s·(x−x_min)+ox, s·(y−y_min)+oy)` to every UE
   sample, every episode. Deterministic and stationary.

Rationale for isotropic + centered: keeps UE spatial density realistic and makes
distances comparable across UAV configs. A stretch-to-fill variant is possible
but distorts geometry — not recommended.

## 4. LoS + capacity model (the binding constraint is capacity, not range)

UAVs are treated as **line-of-sight** to the whole area, so radio range is *not*
the bottleneck — **per-UAV admission capacity is**. Each UAV admits up to **K
UEs** (default K=10); beyond that it rejects further UEs.

- **`max_ues_per_uav = K`** (currently `max_ues_per_gnb`, default 10) is the
  primary constraint **and a study knob** — we sweep it to see how the xApps
  behave as capacity changes.
- **Association = nearest active UAV with a free slot.** Already implemented in
  `TraceServingController._apply_global_nearest_active_assignment`
  ([rl/trace_serving.py](../rl/trace_serving.py)): UEs are assigned to the
  nearest active UAV that isn't full; if every reachable active UAV is full → the
  UE is in **outage**. Overflow spills to the next-nearest automatically.
- **`max_coverage_distance_m`**: set **large enough to cover the area** (≥ ~285 m,
  the center-to-corner distance) so range never binds under LoS; capacity does.
  If we later want partial coverage, this becomes a second knob.
- **SINR/pathloss:** under LoS the SINR model saturates to high spectral
  efficiency, so per-UE rate is driven by **capacity sharing among admitted
  UEs**, not distance. Options: (a) keep `ns_oran_compatible` (it will report
  near-max SE at short LoS distances — acceptable), or (b) add a simpler `los`
  throughput mode where an active UAV's capacity C is shared over its admitted
  UEs (`rate_per_ue = C / n_admitted`, `n_admitted ≤ K`). **Decision: default (b)
  `los`** for now, but keep the throughput model **pluggable** (a
  `--throughput-mode` selector) so a more complex link/capacity model can be
  swapped in later without touching the rest of the env.

Distances for reference (400×400, anchors at 80/200/320):
- center → corner UE (0,0): √(200²+200²) ≈ **283 m**
- adjacent anchors: **120 m**; center → side anchor: **120 m**.

Placement still matters even with area-wide coverage: UEs cluster around demand
hotspots, so a UAV whose slots sit where UEs actually are fills usefully, while a
badly placed UAV wastes its K slots. That is exactly what the xApps must learn.

## 5. Alignment checklist
- [ ] UAV count N ∈ {3,5,7,9}; positions from the table above; center always in.
- [ ] Center UAV is the min-active anchor.
- [ ] UE positions rescaled by a single trace-wide isotropic affine into [0,400]².
- [ ] `max_ues_per_uav = K` (default 10) is the binding constraint + a sweep knob.
- [ ] `max_coverage_distance_m` ≥ ~285 m so range never binds under LoS.
- [ ] UE→UAV = nearest active UAV with a free slot; else outage (already coded).
- [ ] Old BS lat/lon used only to build the rescale transform, not for placement.

## 6. Future extension
When UAV repositioning lands, the fixed anchors become the *initial* positions
and `(x,y[,z])` enters the action/state space (reserved in the sibling specs).
