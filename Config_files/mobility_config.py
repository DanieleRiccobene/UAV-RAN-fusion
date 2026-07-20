import copy
import csv
import math
import os
from pathlib import Path


DEFAULT_MOBILITY_PROFILE = "peak"
VALID_MOBILITY_PROFILES = ("night", "peak")
MOBILITY_DATA_DIRNAME = "mobility_data"
IND_MOB_DIRNAME = "ind_mob"
GNB_POSITIONS_FILENAME = "colosseo_poi_bs_coordinates_table.csv"
PREFERRED_TRACE_USER_COUNT = "20users"
FALLBACK_TRACE_USER_COUNT = "100users"
PROFILE_TO_TRACE_FILENAME = {
    "night": "individual_night_20users_1sec_15min.csv",
    "peak": "individual_peak_20users_1sec_15min.csv",
}
BS_TYPES = {"BS", "GNB", "GNODEB"}
EARTH_RADIUS_M = 6371000.0


def determine_base_dir(base_dir=None):
    if base_dir is not None:
        return base_dir
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_mobility_profile(profile=None):
    normalized = (profile or DEFAULT_MOBILITY_PROFILE).strip().lower()
    if normalized not in VALID_MOBILITY_PROFILES:
        raise ValueError(
            f"Unsupported mobility profile '{profile}'. Expected one of {', '.join(VALID_MOBILITY_PROFILES)}."
        )
    return normalized


def resolve_mobility_data_dir(base_dir=None):
    return os.path.join(determine_base_dir(base_dir), MOBILITY_DATA_DIRNAME)


def resolve_gnb_positions_csv(base_dir=None):
    mobility_data_dir = Path(resolve_mobility_data_dir(base_dir))
    preferred_path = mobility_data_dir / GNB_POSITIONS_FILENAME
    if preferred_path.exists():
        return str(preferred_path)

    candidates = sorted(mobility_data_dir.glob("*bs*coordinates*.csv"))
    filtered = [
        candidate for candidate in candidates
        if "outerring" not in candidate.name.lower()
    ]
    if filtered:
        return str(filtered[0])
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        f"Could not find any gNB positions CSV inside '{mobility_data_dir}'."
    )


def resolve_mobility_trace_csv(profile=None, base_dir=None):
    selected_profile = resolve_mobility_profile(profile)
    ind_mob_dir = Path(resolve_mobility_data_dir(base_dir)) / IND_MOB_DIRNAME
    preferred_path = ind_mob_dir / PROFILE_TO_TRACE_FILENAME[selected_profile]
    if preferred_path.exists():
        return str(preferred_path)

    candidates = []
    for candidate in sorted(ind_mob_dir.glob(f"individual_{selected_profile}*.csv")):
        name = candidate.name.lower()
        if name.endswith("_summary.csv"):
            continue
        candidates.append(candidate)
    if candidates:
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                PREFERRED_TRACE_USER_COUNT not in candidate.name.lower(),
                FALLBACK_TRACE_USER_COUNT not in candidate.name.lower(),
                "hour" not in candidate.name.lower(),
                candidate.name.lower(),
            ),
        )
        return str(ranked_candidates[0])
    raise FileNotFoundError(
        f"Could not find any individual mobility CSV for profile '{selected_profile}' in '{ind_mob_dir}'."
    )


def resolve_zero_mobility_matrix_csv(profile=None, base_dir=None):
    selected_profile = resolve_mobility_profile(profile)
    zero_mob_dir = Path(resolve_mobility_data_dir(base_dir)) / "zero_mob"
    # Prefer the 20-user matrix (matches HIGH/MEDIUM fidelity UE count) over the 100-user one.
    preferred_path = zero_mob_dir / f"zero_{selected_profile}_{PREFERRED_TRACE_USER_COUNT}_1sec_15min_bs_matrix.csv"
    if preferred_path.exists():
        return str(preferred_path)
    fallback_path = zero_mob_dir / f"zero_{selected_profile}_hour_{FALLBACK_TRACE_USER_COUNT}_1sec_15min_bs_matrix.csv"
    if fallback_path.exists():
        return str(fallback_path)

    candidates = []
    for candidate in sorted(zero_mob_dir.glob(f"zero_{selected_profile}*_bs_matrix.csv")):
        candidates.append(candidate)
    if candidates:
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                PREFERRED_TRACE_USER_COUNT not in candidate.name.lower(),
                "hour" not in candidate.name.lower(),
                candidate.name.lower(),
            ),
        )
        return str(ranked_candidates[0])
    raise FileNotFoundError(
        f"Could not find any zero-mobility BS matrix CSV for profile '{selected_profile}' in '{zero_mob_dir}'."
    )


def load_bs_position_map(base_dir=None):
    bs_positions = {}
    csv_path = resolve_gnb_positions_csv(base_dir)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_type = (row.get("type") or "").strip().upper()
            if row_type not in BS_TYPES:
                continue
            bs_id = (row.get("id") or "").strip()
            if not bs_id:
                continue
            latitude = float(row["lat"])
            longitude = float(row["lon"])
            bs_positions[bs_id] = {
                "latitude": latitude,
                "longitude": longitude,
                "location": [latitude, longitude],
                "name": row.get("name"),
            }
    if not bs_positions:
        raise ValueError(f"No BS/gNB rows were found in {csv_path}")
    return bs_positions


def _extract_numeric_suffix(label):
    digits = "".join(character for character in str(label) if character.isdigit())
    return int(digits) if digits else 0


def _project_lat_lon_to_local_xy_m(lat, lon, origin_lat, origin_lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)
    x = (lon_rad - origin_lon_rad) * math.cos((lat_rad + origin_lat_rad) / 2.0) * EARTH_RADIUS_M
    y = (lat_rad - origin_lat_rad) * EARTH_RADIUS_M
    return x, y


def _compute_xy_from_anchor(latitude, longitude, anchor_template, anchor_position):
    anchor_x = float(anchor_template.get("x_m", 0.0))
    anchor_y = float(anchor_template.get("y_m", 0.0))
    dx, dy = _project_lat_lon_to_local_xy_m(
        latitude,
        longitude,
        anchor_position["latitude"],
        anchor_position["longitude"],
    )
    return round(anchor_x + dx, 3), round(anchor_y + dy, 3)


def _build_generated_gnodeb(bs_id, position, template_gnodeb, anchor_template, anchor_position):
    generated = copy.deepcopy(template_gnodeb)
    numeric_suffix = _extract_numeric_suffix(bs_id)
    generated["gnodeb_id"] = bs_id
    generated["latitude"] = position["latitude"]
    generated["longitude"] = position["longitude"]
    generated["location"] = list(position["location"])
    generated["cellIds"] = [f"{bs_id}_C1"]
    generated["sectorCount"] = max(1, int(generated.get("sectorCount", 1)))
    generated["cellCount"] = max(1, int(generated.get("cellCount", 1)))
    generated["x_m"], generated["y_m"] = _compute_xy_from_anchor(
        position["latitude"], position["longitude"], anchor_template, anchor_position
    )

    generated_sector_ids = []
    for sector_index in range(generated["sectorCount"]):
        generated_sector_ids.append(
            {
                "sectorId": f"S{sector_index + 1}",
                "pci": 100 + numeric_suffix + sector_index,
                "bandwidth": generated.get("bandwidth", template_gnodeb.get("bandwidth")),
                "power": generated.get("power", template_gnodeb.get("power")),
                "frequency": generated.get("frequency", template_gnodeb.get("frequency")),
            }
        )
    generated["sectorIds"] = generated_sector_ids
    return generated


def overlay_gnodeb_positions(config, base_dir=None):
    updated = copy.deepcopy(config)
    bs_positions = load_bs_position_map(base_dir)
    gnodebs = updated.get("gNodeBs", [])
    if not gnodebs:
        raise ValueError("gNodeB configuration is empty")

    gnodebs.sort(key=lambda item: _extract_numeric_suffix(item.get("gnodeb_id", "")))
    existing_ids = {gnodeb.get("gnodeb_id") for gnodeb in gnodebs}

    anchor_template = None
    anchor_position = None
    for gnodeb in gnodebs:
        gnodeb_id = gnodeb.get("gnodeb_id")
        if gnodeb_id in bs_positions:
            anchor_template = gnodeb
            anchor_position = bs_positions[gnodeb_id]
            break
    if anchor_template is None or anchor_position is None:
        raise ValueError(
            f"No overlapping gNodeB ids were found between the config and {resolve_gnb_positions_csv(base_dir)}"
        )

    for gnodeb in gnodebs:
        gnodeb_id = gnodeb.get("gnodeb_id")
        if gnodeb_id not in bs_positions:
            raise ValueError(f"Missing BS position for gNodeB '{gnodeb_id}' in {resolve_gnb_positions_csv(base_dir)}")
        position = bs_positions[gnodeb_id]
        gnodeb["latitude"] = position["latitude"]
        gnodeb["longitude"] = position["longitude"]
        gnodeb["location"] = list(position["location"])
        if "x_m" in gnodeb and "y_m" in gnodeb:
            continue
        gnodeb["x_m"], gnodeb["y_m"] = _compute_xy_from_anchor(
            position["latitude"], position["longitude"], anchor_template, anchor_position
        )

    template_gnodeb = copy.deepcopy(gnodebs[-1])
    missing_bs_ids = sorted(
        [bs_id for bs_id in bs_positions if bs_id not in existing_ids],
        key=_extract_numeric_suffix,
    )
    for bs_id in missing_bs_ids:
        generated = _build_generated_gnodeb(
            bs_id,
            bs_positions[bs_id],
            template_gnodeb,
            anchor_template,
            anchor_position,
        )
        gnodebs.append(generated)

    gnodebs.sort(key=lambda item: _extract_numeric_suffix(item.get("gnodeb_id", "")))
    updated["gNodeBs"] = gnodebs
    return updated
