import csv
import logging
from collections import Counter
from dataclasses import replace
from enum import Enum
from pathlib import Path

from Config_files.mobility_config import (
    DEFAULT_MOBILITY_PROFILE,
    load_bs_position_map,
    resolve_mobility_profile,
    resolve_mobility_trace_csv,
    resolve_zero_mobility_matrix_csv as resolve_zero_mobility_matrix_csv_from_config,
)
from rl.trace_mobility import MobilitySample, MobilityTrace


LOGGER = logging.getLogger(__name__)

class FidelityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.HIGH.value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported fidelity level '{value}'. Expected one of: high, medium, low."
            ) from exc


def resolve_zero_mobility_matrix_csv(profile=None, base_dir=None):
    return Path(resolve_zero_mobility_matrix_csv_from_config(profile, base_dir))


def _seconds_key(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _total_duration_from_relative_map(timestamps, relative_seconds):
    if not timestamps:
        return 0.0
    if len(timestamps) == 1:
        return 1.0
    deltas = []
    for index in range(1, len(timestamps)):
        previous_seconds = relative_seconds[timestamps[index - 1]]
        current_seconds = relative_seconds[timestamps[index]]
        delta = current_seconds - previous_seconds
        if delta > 0:
            deltas.append(delta)
    step_seconds = deltas[-1] if deltas else 1.0
    return float(relative_seconds[timestamps[-1]] + step_seconds)


class BaseFidelityProvider:
    def __init__(self, fidelity_level, source_path):
        self.fidelity_level = FidelityLevel.normalize(fidelity_level)
        self.source_path = str(source_path)

    def __len__(self):
        return len(self.timestamps)

    def samples_at_index(self, index):
        timestamp = self.timestamps[index]
        return timestamp, self.get_users_at_time(timestamp)

    def get_time_range(self):
        if not self.timestamps:
            return None, None
        return self.timestamps[0], self.timestamps[-1]

    def total_duration_seconds(self):
        return _total_duration_from_relative_map(self.timestamps, self.relative_timestamp_seconds)

    def has_user_positions(self):
        raise NotImplementedError

    def get_users_at_time(self, timestamp):
        raise NotImplementedError

    def get_bs_load_at_time(self, timestamp):
        raise NotImplementedError

    def windowed(self, start_time_seconds, window_size_seconds):
        raise NotImplementedError

    def fidelity_log_message(self):
        return f"[FIDELITY] mode={self.fidelity_level.value} source={self.source_path}"


class IndividualMobilityProvider(BaseFidelityProvider):
    def __init__(self, mobility_trace, source_path=None):
        super().__init__(FidelityLevel.HIGH, source_path or mobility_trace.source_path or "<mobility-trace>")
        self.mobility_trace = mobility_trace
        self.timestamps = list(mobility_trace.timestamps)
        self.timestamp_unit = mobility_trace.timestamp_unit
        self.user_ids = list(mobility_trace.user_ids)
        self.serving_bs_ids = list(mobility_trace.serving_bs_ids)
        self.relative_timestamp_seconds = dict(mobility_trace.relative_timestamp_seconds)

    def has_user_positions(self):
        return True

    def get_users_at_time(self, timestamp):
        return list(self.mobility_trace.samples_by_timestamp.get(timestamp, []))

    def get_bs_load_at_time(self, timestamp):
        return dict(
            sorted(
                Counter(
                    sample.serving_bs_id
                    for sample in self.get_users_at_time(timestamp)
                    if sample.covered and sample.serving_bs_id not in ("NA", None)
                ).items()
            )
        )

    def windowed(self, start_time_seconds, window_size_seconds):
        return IndividualMobilityProvider(
            self.mobility_trace.windowed(start_time_seconds, window_size_seconds),
            source_path=self.source_path,
        )


class FrozenSnapshotProvider(BaseFidelityProvider):
    def __init__(self, timeline_trace, snapshot_samples, snapshot_sec, source_path=None):
        super().__init__(FidelityLevel.MEDIUM, source_path or timeline_trace.source_path or "<mobility-trace>")
        self.timeline_trace = timeline_trace
        self.snapshot_samples = list(snapshot_samples)
        self.snapshot_sec = float(snapshot_sec)
        self.timestamps = list(timeline_trace.timestamps)
        self.timestamp_unit = timeline_trace.timestamp_unit
        self.user_ids = sorted({sample.user_id for sample in self.snapshot_samples}, key=_seconds_key)
        self.serving_bs_ids = sorted(
            {sample.serving_bs_id for sample in self.snapshot_samples if sample.serving_bs_id not in ("NA", None)},
            key=_seconds_key,
        )
        self.relative_timestamp_seconds = dict(timeline_trace.relative_timestamp_seconds)

    @classmethod
    def from_trace(cls, mobility_trace, snapshot_sec):
        snapshot_sec = float(snapshot_sec)
        snapshot_timestamp = None
        for timestamp in mobility_trace.timestamps:
            relative_seconds = mobility_trace.relative_timestamp_seconds[timestamp]
            if abs(relative_seconds - snapshot_sec) <= 1e-9:
                snapshot_timestamp = timestamp
                break
        if snapshot_timestamp is None:
            raise ValueError(
                f"Snapshot second {snapshot_sec} was not found in the mobility trace."
            )
        return cls(
            timeline_trace=mobility_trace,
            snapshot_samples=mobility_trace.samples_by_timestamp[snapshot_timestamp],
            snapshot_sec=snapshot_sec,
            source_path=mobility_trace.source_path,
        )

    def has_user_positions(self):
        return True

    def get_users_at_time(self, timestamp):
        return [replace(sample, timestamp=str(timestamp)) for sample in self.snapshot_samples]

    def get_bs_load_at_time(self, timestamp):
        return dict(
            sorted(
                Counter(
                    sample.serving_bs_id
                    for sample in self.get_users_at_time(timestamp)
                    if sample.covered and sample.serving_bs_id not in ("NA", None)
                ).items()
            )
        )

    def windowed(self, start_time_seconds, window_size_seconds):
        return FrozenSnapshotProvider(
            timeline_trace=self.timeline_trace.windowed(start_time_seconds, window_size_seconds),
            snapshot_samples=self.snapshot_samples,
            snapshot_sec=self.snapshot_sec,
            source_path=self.source_path,
        )

    def fidelity_log_message(self):
        return (
            f"[FIDELITY] mode={self.fidelity_level.value} "
            f"snapshot_sec={self.snapshot_sec} source={self.source_path}"
        )


class BsAggregateProvider(BaseFidelityProvider):
    def __init__(self, *, source_path, timestamps, bs_load_by_timestamp, relative_timestamp_seconds, bs_position_map):
        super().__init__(FidelityLevel.LOW, source_path)
        self.timestamps = list(timestamps)
        self.timestamp_unit = "seconds"
        self.bs_load_by_timestamp = {
            timestamp: dict(sorted(loads.items()))
            for timestamp, loads in bs_load_by_timestamp.items()
        }
        self.relative_timestamp_seconds = dict(relative_timestamp_seconds)
        self.bs_position_map = dict(bs_position_map)
        self.serving_bs_ids = sorted(bs_position_map.keys(), key=_seconds_key)
        self.user_ids = sorted(
            {
                f"SYN_{bs_id}_{index:03d}"
                for bs_id, max_count in self._max_users_per_bs().items()
                for index in range(max_count)
            },
            key=_seconds_key,
        )

    @classmethod
    def from_csv(cls, path, *, base_dir=None):
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if "time_offset_sec" not in fieldnames:
                raise ValueError(f"Aggregate BS trace {path} is missing time_offset_sec")
            bs_columns = [field for field in fieldnames if field != "time_offset_sec"]
            rows = list(reader)
        timestamps = []
        bs_load_by_timestamp = {}
        relative_timestamp_seconds = {}
        origin_seconds = None
        for row in rows:
            timestamp = str(row["time_offset_sec"])
            seconds = float(row["time_offset_sec"])
            if origin_seconds is None:
                origin_seconds = seconds
            timestamps.append(timestamp)
            relative_timestamp_seconds[timestamp] = seconds - origin_seconds
            bs_load_by_timestamp[timestamp] = {
                bs_id: int(float(row[bs_id]))
                for bs_id in bs_columns
                if int(float(row[bs_id])) > 0
            }
        return cls(
            source_path=path,
            timestamps=timestamps,
            bs_load_by_timestamp=bs_load_by_timestamp,
            relative_timestamp_seconds=relative_timestamp_seconds,
            bs_position_map=load_bs_position_map(base_dir),
        )

    def _max_users_per_bs(self):
        max_users = Counter()
        for loads in self.bs_load_by_timestamp.values():
            for bs_id, count in loads.items():
                max_users[bs_id] = max(max_users.get(bs_id, 0), int(count))
        return dict(max_users)

    def has_user_positions(self):
        return False

    def get_users_at_time(self, timestamp):
        loads = self.get_bs_load_at_time(timestamp)
        samples = []
        for bs_id, count in sorted(loads.items(), key=lambda item: _seconds_key(item[0])):
            bs_pos = self.bs_position_map.get(bs_id, {})
            bs_lat = bs_pos.get("latitude")
            bs_lon = bs_pos.get("longitude")
            for index in range(int(count)):
                samples.append(
                    MobilitySample(
                        user_id=f"SYN_{bs_id}_{index:03d}",
                        timestamp=str(timestamp),
                        prev_bs_id=bs_id,
                        chosen_bs_id=bs_id,
                        serving_bs_id=bs_id,
                        covered=True,
                        prev_x_m=None,
                        prev_y_m=None,
                        x_m=None,
                        y_m=None,
                        move_distance_m=0.0,
                        dist_to_serving_bs_m=None,
                        lon=bs_lon,
                        lat=bs_lat,
                        synthetic=True,
                        has_precise_position=False,
                    )
                )
        return samples

    def get_bs_load_at_time(self, timestamp):
        return dict(self.bs_load_by_timestamp.get(timestamp, {}))

    def windowed(self, start_time_seconds, window_size_seconds):
        start_time_seconds = float(start_time_seconds)
        end_time_seconds = start_time_seconds + float(window_size_seconds)
        timestamps = [
            timestamp
            for timestamp in self.timestamps
            if start_time_seconds <= self.relative_timestamp_seconds[timestamp] < end_time_seconds
        ]
        relative_seconds = {
            timestamp: self.relative_timestamp_seconds[timestamp] - start_time_seconds
            for timestamp in timestamps
        }
        bs_load_by_timestamp = {
            timestamp: self.bs_load_by_timestamp[timestamp]
            for timestamp in timestamps
        }
        return BsAggregateProvider(
            source_path=self.source_path,
            timestamps=timestamps,
            bs_load_by_timestamp=bs_load_by_timestamp,
            relative_timestamp_seconds=relative_seconds,
            bs_position_map=self.bs_position_map,
        )


def build_fidelity_provider(
    *,
    fidelity_level,
    mobility_profile=DEFAULT_MOBILITY_PROFILE,
    base_dir=None,
    mobility_trace_csv=None,
    medium_snapshot_sec=0.0,
    provider_override=None,
):
    if provider_override is not None:
        return provider_override
    normalized_level = FidelityLevel.normalize(fidelity_level)
    normalized_profile = resolve_mobility_profile(mobility_profile)
    if normalized_level == FidelityLevel.HIGH:
        trace_path = mobility_trace_csv or resolve_mobility_trace_csv(normalized_profile, base_dir)
        return IndividualMobilityProvider(MobilityTrace.from_csv(trace_path), source_path=trace_path)
    if normalized_level == FidelityLevel.MEDIUM:
        trace_path = mobility_trace_csv or resolve_mobility_trace_csv(normalized_profile, base_dir)
        return FrozenSnapshotProvider.from_trace(
            MobilityTrace.from_csv(trace_path),
            snapshot_sec=medium_snapshot_sec,
        )
    matrix_path = resolve_zero_mobility_matrix_csv(normalized_profile, base_dir)
    return BsAggregateProvider.from_csv(matrix_path, base_dir=base_dir)


def log_fidelity_provider(provider):
    message = provider.fidelity_log_message()
    LOGGER.info(message)
    print(message)
