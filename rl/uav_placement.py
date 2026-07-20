"""UAV placement (dice-face patterns) and UE-trace rescaling into a fixed area.

See docs/TOPOLOGY.md. UAVs are placed on a 3x3 anchor grid inside a square
service area; the center anchor is always included (it is the min-active anchor).
UE trace coordinates are rescaled once (isotropic, centered) into the same area.
"""

from __future__ import annotations

AREA_SIZE_M = 400.0
MARGIN_M = 80.0  # 0.2 * AREA_SIZE_M -> anchors at 80 / 200 / 320

# 3x3 anchor grid keyed by a stable label.
_ANCHORS = {
    "BL": (MARGIN_M, MARGIN_M),
    "BC": (AREA_SIZE_M / 2.0, MARGIN_M),
    "BR": (AREA_SIZE_M - MARGIN_M, MARGIN_M),
    "ML": (MARGIN_M, AREA_SIZE_M / 2.0),
    "C": (AREA_SIZE_M / 2.0, AREA_SIZE_M / 2.0),
    "MR": (AREA_SIZE_M - MARGIN_M, AREA_SIZE_M / 2.0),
    "TL": (MARGIN_M, AREA_SIZE_M - MARGIN_M),
    "TC": (AREA_SIZE_M / 2.0, AREA_SIZE_M - MARGIN_M),
    "TR": (AREA_SIZE_M - MARGIN_M, AREA_SIZE_M - MARGIN_M),
}

# Dice-face subsets. Center "C" is always present. Order is stable (index == UAV k).
_PATTERNS = {
    3: ["BL", "C", "TR"],                            # diagonal (dice-3)
    5: ["BL", "BR", "TL", "TR", "C"],                # quincunx (dice-5)
    7: ["BL", "BR", "TL", "TR", "ML", "MR", "C"],    # corners + left/right mids + center
    9: ["BL", "BC", "BR", "ML", "C", "MR", "TL", "TC", "TR"],  # full 3x3 (dice-9)
}

SUPPORTED_UAV_COUNTS = tuple(sorted(_PATTERNS))


def dice_uav_positions(num_uavs):
    """Return (positions, center_index) for a dice-face fleet of ``num_uavs``.

    positions: list of (x_m, y_m) in the 400x400 area, index == UAV k (0-based).
    center_index: index of the mandatory-active center UAV in that list.
    """
    if num_uavs not in _PATTERNS:
        raise ValueError(
            f"Unsupported num_uavs={num_uavs!r}; supported: {SUPPORTED_UAV_COUNTS}"
        )
    labels = _PATTERNS[num_uavs]
    positions = [_ANCHORS[label] for label in labels]
    center_index = labels.index("C")
    return positions, center_index


class AreaRescaler:
    """Isotropic, centered affine mapping of trace (x, y) into [0, AREA] x [0, AREA].

    Fit once over every UE sample in the trace so distances are stationary across
    episodes. Preserves aspect ratio (no distortion); centers the shorter axis.
    """

    def __init__(self, x_min, x_max, y_min, y_max, area_size_m=AREA_SIZE_M):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.area_size_m = float(area_size_m)
        span = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.scale = self.area_size_m / span if span > 0 else 1.0
        self.offset_x = (self.area_size_m - (self.x_max - self.x_min) * self.scale) / 2.0
        self.offset_y = (self.area_size_m - (self.y_max - self.y_min) * self.scale) / 2.0

    @classmethod
    def fit(cls, xy_points, area_size_m=AREA_SIZE_M):
        xs = [float(x) for x, _ in xy_points if x is not None]
        ys = [float(y) for _, y in xy_points if y is not None]
        if not xs or not ys:
            raise ValueError("AreaRescaler.fit requires at least one (x, y) point")
        return cls(min(xs), max(xs), min(ys), max(ys), area_size_m=area_size_m)

    def apply(self, x, y):
        if x is None or y is None:
            return None, None
        return (
            self.scale * (float(x) - self.x_min) + self.offset_x,
            self.scale * (float(y) - self.y_min) + self.offset_y,
        )

    def as_dict(self):
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "scale": self.scale,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "area_size_m": self.area_size_m,
        }
