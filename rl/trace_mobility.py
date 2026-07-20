import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MOBILITY_BASE_REQUIRED_COLUMNS = {
    "user_id",
    "serving_bs_id",
    "covered",
    "x_m",
    "y_m",
    "dist_to_serving_bs_m",
    "lon",
    "lat",
}
MOBILITY_REQUIRED_COLUMNS_HOUR = MOBILITY_BASE_REQUIRED_COLUMNS | {"hour"}
MOBILITY_REQUIRED_COLUMNS_OFFSET = MOBILITY_BASE_REQUIRED_COLUMNS | {"time_offset_sec", "timestamp"}
MOBILITY_REQUIRED_COLUMNS_PERIOD = MOBILITY_BASE_REQUIRED_COLUMNS | {"period_hour", "time_offset_sec", "timestamp"}

STATIC_REQUIRED_COLUMNS = {
    "user_id",
    "home_bs_idx",
    "bs_id",
    "x_m",
    "y_m",
    "lon",
    "lat",
}


def _to_float(value, default=None):
    if value in (None, "", "NA"):
        return default
    return float(value)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() in {"TRUE", "1", "YES", "Y"}


def _timestamp_key(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _first_present(row, *keys, default=None):
    for key in keys:
        if key in row and row[key] not in (None, "", "NA"):
            return row[key]
    return default


def _pick_timestamp_value(row):
    if "time_offset_sec" in row and row["time_offset_sec"] not in (None, "", "NA"):
        return str(row["time_offset_sec"]), "seconds"
    if "hour" in row and row["hour"] not in (None, "", "NA"):
        return str(row["hour"]), "hours"
    if "period_hour" in row and row["period_hour"] not in (None, "", "NA"):
        return str(row["period_hour"]), "hours"
    if "timestamp" in row and row["timestamp"] not in (None, "", "NA"):
        return str(row["timestamp"]), "datetime"
    raise ValueError("Mobility row does not contain a usable timestamp column")


def _parse_timestamp_to_seconds(value, unit):
    if value in (None, "", "NA"):
        return None
    if unit == "seconds":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if unit == "hours":
        try:
            return float(value) * 3600.0
        except (TypeError, ValueError):
            return None
    if unit == "datetime":
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MobilitySample:
    user_id: str
    timestamp: str
    prev_bs_id: str
    chosen_bs_id: str
    serving_bs_id: str
    covered: bool
    prev_x_m: float
    prev_y_m: float
    x_m: float
    y_m: float
    move_distance_m: float
    dist_to_serving_bs_m: float
    lon: float
    lat: float
    synthetic: bool = False
    has_precise_position: bool = True

    @classmethod
    def from_row(cls, row):
        timestamp_value, _timestamp_unit = _pick_timestamp_value(row)
        serving_bs_id = str(row["serving_bs_id"])
        x_m = _to_float(row["x_m"])
        y_m = _to_float(row["y_m"])
        return cls(
            user_id=str(row["user_id"]),
            timestamp=timestamp_value,
            prev_bs_id=str(_first_present(row, "prev_bs_id", default=serving_bs_id)),
            chosen_bs_id=str(_first_present(row, "chosen_bs_id", default=serving_bs_id)),
            serving_bs_id=serving_bs_id,
            covered=_to_bool(row["covered"]),
            prev_x_m=_to_float(_first_present(row, "prev_x_m", default=x_m), x_m),
            prev_y_m=_to_float(_first_present(row, "prev_y_m", default=y_m), y_m),
            x_m=x_m,
            y_m=y_m,
            move_distance_m=_to_float(_first_present(row, "move_distance_m", default=0.0), 0.0),
            dist_to_serving_bs_m=_to_float(row["dist_to_serving_bs_m"]),
            lon=_to_float(row["lon"]),
            lat=_to_float(row["lat"]),
            synthetic=False,
            has_precise_position=True,
        )


@dataclass(frozen=True)
class StaticUserPosition:
    user_id: str
    home_bs_idx: str
    bs_id: str
    x_m: float
    y_m: float
    lon: float
    lat: float

    @classmethod
    def from_row(cls, row):
        return cls(
            user_id=str(row["user_id"]),
            home_bs_idx=str(row["home_bs_idx"]),
            bs_id=str(row["bs_id"]),
            x_m=_to_float(row["x_m"]),
            y_m=_to_float(row["y_m"]),
            lon=_to_float(row["lon"]),
            lat=_to_float(row["lat"]),
        )


class MobilityTrace:
    def __init__(self, samples_by_timestamp, timestamp_unit="unknown", source_path=None):
        self.samples_by_timestamp = samples_by_timestamp
        self.timestamps = sorted(samples_by_timestamp, key=_timestamp_key)
        self.user_ids = sorted({sample.user_id for samples in samples_by_timestamp.values() for sample in samples}, key=_timestamp_key)
        self.serving_bs_ids = sorted(
            {sample.serving_bs_id for samples in samples_by_timestamp.values() for sample in samples if sample.serving_bs_id != "NA"},
            key=_timestamp_key,
        )
        self.timestamp_unit = timestamp_unit
        self.source_path = str(source_path) if source_path is not None else None
        self.relative_timestamp_seconds = self._build_relative_timestamp_seconds()

    @classmethod
    def from_csv(cls, path):
        path = Path(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_hour = MOBILITY_REQUIRED_COLUMNS_HOUR - fieldnames
            missing_offset = MOBILITY_REQUIRED_COLUMNS_OFFSET - fieldnames
            missing_period = MOBILITY_REQUIRED_COLUMNS_PERIOD - fieldnames
            if missing_hour and missing_offset and missing_period:
                raise ValueError(
                    f"Mobility trace {path} is missing columns for all supported schemas: "
                    f"hour_schema_missing={sorted(missing_hour)} "
                    f"offset_schema_missing={sorted(missing_offset)} "
                    f"period_schema_missing={sorted(missing_period)}"
                )
            samples_by_timestamp = {}
            timestamp_unit = None
            for row in reader:
                _timestamp_value, row_timestamp_unit = _pick_timestamp_value(row)
                if timestamp_unit is None:
                    timestamp_unit = row_timestamp_unit
                sample = MobilitySample.from_row(row)
                samples_by_timestamp.setdefault(sample.timestamp, []).append(sample)
        return cls(samples_by_timestamp, timestamp_unit=timestamp_unit or "unknown", source_path=path)

    def samples_at_index(self, index):
        timestamp = self.timestamps[index]
        return timestamp, self.samples_by_timestamp[timestamp]

    def __len__(self):
        return len(self.timestamps)

    def _build_relative_timestamp_seconds(self):
        absolute_values = []
        for timestamp in self.timestamps:
            seconds = _parse_timestamp_to_seconds(timestamp, self.timestamp_unit)
            absolute_values.append(seconds)
        valid_values = [value for value in absolute_values if value is not None]
        if not valid_values:
            return {timestamp: float(index) for index, timestamp in enumerate(self.timestamps)}
        origin_seconds = valid_values[0]
        return {
            timestamp: max(0.0, absolute_values[index] - origin_seconds)
            if absolute_values[index] is not None else float(index)
            for index, timestamp in enumerate(self.timestamps)
        }

    def total_duration_seconds(self):
        if not self.timestamps:
            return 0.0
        if len(self.timestamps) == 1:
            return 1.0
        deltas = []
        for index in range(1, len(self.timestamps)):
            previous_seconds = self.relative_timestamp_seconds[self.timestamps[index - 1]]
            current_seconds = self.relative_timestamp_seconds[self.timestamps[index]]
            delta = current_seconds - previous_seconds
            if delta > 0:
                deltas.append(delta)
        step_seconds = deltas[-1] if deltas else 1.0
        return float(self.relative_timestamp_seconds[self.timestamps[-1]] + step_seconds)

    def windowed(self, start_time_seconds, window_size_seconds):
        start_time_seconds = float(start_time_seconds)
        window_size_seconds = float(window_size_seconds)
        end_time_seconds = start_time_seconds + window_size_seconds
        windowed_samples = {
            timestamp: list(self.samples_by_timestamp[timestamp])
            for timestamp in self.timestamps
            if start_time_seconds <= self.relative_timestamp_seconds[timestamp] < end_time_seconds
        }
        return MobilityTrace(
            windowed_samples,
            timestamp_unit=self.timestamp_unit,
            source_path=self.source_path,
        )


class StaticUserTrace:
    def __init__(self, positions):
        self.positions = positions
        self.by_user_id = {position.user_id: position for position in positions}
        self.bs_ids = sorted({position.bs_id for position in positions}, key=_timestamp_key)

    @classmethod
    def from_csv(cls, path):
        path = Path(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = STATIC_REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Static user position trace {path} is missing columns: {sorted(missing)}")
            return cls([StaticUserPosition.from_row(row) for row in reader])
