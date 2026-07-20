#!/usr/bin/env python3
import argparse
import contextlib
import csv
import json
import logging
import math
import os
import pickle
import random
import shutil
import socket
import subprocess
import sys
import time
import threading
import types
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Config_files.mobility_config import DEFAULT_MOBILITY_PROFILE, resolve_mobility_trace_csv
from rl.default_paths import DEFAULT_STATIC_CSV
from rl.fidelity_mab import FidelityMabController
from rl.fidelity_provider import FidelityLevel, build_fidelity_provider
from rl.reward_listener import RewardSocketListener


ORCH_LOGGER = logging.getLogger("ranfusion.orchestrator")
MAB_LOGGER = logging.getLogger("ranfusion.mab")
RL_LOGGER = logging.getLogger("ranfusion.rl_internal")

PLOT_HELPER_PYTHON_CANDIDATES = [
    os.environ.get("RANFUSION_PLOT_PYTHON"),
    "/home/raoulraft/.venvs/ranfusion_ns3/bin/python",
    shutil.which("python3"),
    shutil.which("python"),
    "/home/raoulraft/miniconda3/bin/python3",
]


def _normalize_log_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def log_event(logger, event, **fields):
    field_parts = [f"{key}={_normalize_log_value(value)}" for key, value in fields.items()]
    message = f"event={event}"
    if field_parts:
        message += " " + " ".join(field_parts)
    logger.info(message)


def _configure_named_logger(logger, path, *, console=False):
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def resolve_plot_helper_python():
    current_executable = os.path.realpath(sys.executable)
    seen = set()
    for candidate in PLOT_HELPER_PYTHON_CANDIDATES:
        if not candidate:
            continue
        real_candidate = os.path.realpath(candidate)
        if real_candidate in seen:
            continue
        seen.add(real_candidate)
        if not os.path.exists(real_candidate):
            continue
        if real_candidate != current_executable or "miniconda3" in real_candidate:
            return candidate
    return None


def invoke_plot_helper(script_name, arguments, *, logger, event_prefix):
    helper_python = resolve_plot_helper_python()
    if not helper_python:
        log_event(logger, f"{event_prefix}_unavailable", reason="no_helper_python")
        return False
    helper_path = os.path.join(REPO_ROOT, "scripts", script_name)
    command = [helper_python, helper_path, *arguments]
    helper_env = os.environ.copy()
    helper_env["PYTHONNOUSERSITE"] = "1"
    helper_env.pop("PYTHONHOME", None)
    helper_env["PYTHONPATH"] = REPO_ROOT
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=helper_env,
        )
        log_event(logger, f"{event_prefix}_succeeded", helper_python=helper_python, helper_script=helper_path)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        log_event(
            logger,
            f"{event_prefix}_failed",
            helper_python=helper_python,
            helper_script=helper_path,
            return_code=exc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        return False


def install_external_stubs():
    os.environ.setdefault("INFLUXDB_TOKEN", "test-token")
    os.environ.setdefault("INFLUXDB_URL", "http://localhost:8086")
    os.environ.setdefault("INFLUXDB_ORG", "ranfusion")
    os.environ.setdefault("INFLUXDB_BUCKET", "RAN_metrics")

    class FakePoint:
        def __init__(self, measurement):
            self.measurement = measurement
            self._fields = {}
            self._tags = {}
        def tag(self, *args, **kwargs):
            if len(args) >= 2:
                self._tags[args[0]] = args[1]
            return self
        def field(self, *args, **kwargs):
            if len(args) >= 2:
                self._fields[args[0]] = args[1]
            return self
        def time(self, *args, **kwargs):
            return self
        def to_line_protocol(self):
            return self.measurement

    class FakeWriteApi:
        def write(self, *args, **kwargs):
            return None

    class FakeClient:
        url = "http://localhost:8086"
        def write_api(self, *args, **kwargs):
            return FakeWriteApi()
        def query_api(self):
            return types.SimpleNamespace(query=lambda *args, **kwargs: [])
        def delete_api(self):
            return types.SimpleNamespace(delete=lambda *args, **kwargs: None)
        def ping(self):
            return True
        def close(self):
            return None

    influxdb_client = types.ModuleType("influxdb_client")
    influxdb_client.InfluxDBClient = lambda *args, **kwargs: FakeClient()
    influxdb_client.Point = FakePoint
    influxdb_client.QueryApi = object
    influxdb_client.WritePrecision = types.SimpleNamespace(S="s", NS="ns")
    client_pkg = types.ModuleType("influxdb_client.client")
    write_api = types.ModuleType("influxdb_client.client.write_api")
    write_api.SYNCHRONOUS = object()
    write_api.WritePrecision = influxdb_client.WritePrecision
    write_api.WriteOptions = object
    delete_api = types.ModuleType("influxdb_client.client.delete_api")
    delete_api.DeleteApi = object
    sys.modules.setdefault("influxdb_client", influxdb_client)
    sys.modules.setdefault("influxdb_client.client", client_pkg)
    sys.modules.setdefault("influxdb_client.client.write_api", write_api)
    sys.modules.setdefault("influxdb_client.client.delete_api", delete_api)

    ntplib = types.ModuleType("ntplib")
    ntplib.NTPException = Exception
    ntplib.NTPClient = lambda: types.SimpleNamespace(request=lambda *args, **kwargs: types.SimpleNamespace(tx_time=time.time()))
    sys.modules.setdefault("ntplib", ntplib)


def quiet_simulator_loggers():
    for logger_name in ("gnodeb_logger", "cell_logger", "sector_logger", "ue_logger"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def maybe_suppress_stdout(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        yield


def monotonic_utc_now():
    return datetime.now(timezone.utc).isoformat()


def launch_timestamp_label():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "inf", "infinite"}:
        return None
    return float(value)


def configure_logging():
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)


def configure_experiment_logging(output_layout):
    logs_dir = output_layout["logs"]
    os.makedirs(logs_dir, exist_ok=True)
    _configure_named_logger(ORCH_LOGGER, os.path.join(logs_dir, "orchestrator.log"), console=True)
    _configure_named_logger(MAB_LOGGER, os.path.join(logs_dir, "mab.log"), console=True)
    _configure_named_logger(RL_LOGGER, os.path.join(logs_dir, "rl_internal.log"), console=False)


class RealTimeBudgetController:
    def __init__(self, budget_sec, time_check_interval_ms):
        self.budget_sec = float(budget_sec)
        self.time_check_interval_sec = max(0.0, float(time_check_interval_ms) / 1000.0)
        self.start_monotonic = time.monotonic()
        self.start_wall_utc = monotonic_utc_now()
        self.last_check_monotonic = self.start_monotonic
        self.cached_elapsed = 0.0
        self.expired = False

    def elapsed_real_sec(self):
        now = time.monotonic()
        if self.expired:
            self.cached_elapsed = now - self.start_monotonic
            return self.cached_elapsed
        if (now - self.last_check_monotonic) >= self.time_check_interval_sec:
            self.cached_elapsed = now - self.start_monotonic
            self.last_check_monotonic = now
            if self.cached_elapsed >= self.budget_sec:
                self.expired = True
        else:
            self.cached_elapsed = now - self.start_monotonic
            if self.cached_elapsed >= self.budget_sec:
                self.expired = True
        return self.cached_elapsed

    def should_stop(self):
        return self.elapsed_real_sec() >= self.budget_sec

    def finalize(self):
        end_monotonic = time.monotonic()
        end_wall_utc = monotonic_utc_now()
        elapsed = end_monotonic - self.start_monotonic
        return {
            "real_start_wall_utc": self.start_wall_utc,
            "real_end_wall_utc": end_wall_utc,
            "actual_elapsed_real_sec": elapsed,
            "real_time_budget_sec": self.budget_sec,
            "budget_expired": elapsed >= self.budget_sec,
        }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_evaluation_row(path, fieldnames, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_final_socket_configuration(final_payload):
    best_configuration = final_payload.get("best_configuration") or {}
    desired_states = best_configuration.get("desired_gnb_states") or {}
    kpis = final_payload.get("kpis") or {}
    final_socket_summary = {
        "socket_event": final_payload.get("event"),
        "best_reward": final_payload.get("best_reward"),
        "best_found_at_real_sec": final_payload.get("best_found_at_real_sec"),
        "best_found_at_sim_time": final_payload.get("best_found_at_sim_time"),
        "best_action_label": best_configuration.get("label"),
        "best_action_mask": best_configuration.get("mask"),
    }
    if desired_states:
        ordered_states = ", ".join(
            f"{gnb_id}={'ON' if is_on else 'OFF'}"
            for gnb_id, is_on in sorted(desired_states.items())
        )
        final_socket_summary["gnb_states"] = ordered_states
    else:
        final_socket_summary["gnb_states"] = None
    final_socket_summary["kpis"] = {
        "throughput": kpis.get("throughput"),
        "power_consumption": kpis.get("power_consumption"),
        "energy_cost": kpis.get("energy_cost"),
        "energy_saving": kpis.get("energy_saving"),
        "total_energy_kwh": kpis.get("total_energy_kwh"),
        "outage_ue_count": kpis.get("outage_ue_count"),
        "active_gnb_count": kpis.get("active_gnb_count"),
        "ue_associations": kpis.get("ue_associations"),
    }
    log_event(ORCH_LOGGER, "final_socket_configuration", **final_socket_summary)


def emit_terminal_window_summary(
    *,
    window_metadata,
    training_csv,
    plot_output_dir,
    final_payload,
    plot_paths,
    final_topology_plot,
):
    key_plots = []
    for path in plot_paths or []:
        name = os.path.basename(path)
        if name == "episodic_reward.png":
            key_plots.append(path)
    if final_topology_plot and final_topology_plot.get("path"):
        key_plots.append(final_topology_plot["path"])
    print(
        f"window_index: {window_metadata['window_index']}\n"
        f"window_interval: [{window_metadata['window_start_seconds']}, {window_metadata['window_end_seconds']})\n"
        f"training_csv: {training_csv}\n"
        f"plots_output_dir: {plot_output_dir}"
    )
    for path in key_plots:
        print(f"plot: {path}")
    print(
        "socket_export_configuration:",
        json.dumps(final_payload.get("best_configuration", {}), sort_keys=True),
    )
    best_reward = final_payload.get("best_reward")
    print(f"final_internal_best_reward: {best_reward}")
    print_final_socket_configuration(final_payload)


def describe_mab_arm(arm, fidelity_level):
    if arm is None:
        return f"manual fidelity={fidelity_level}"
    return (
        f"MAB arm={arm} -> fidelity={fidelity_level} "
        f"(0=high real per-user mobility, 1=medium frozen snapshot, 2=low BS aggregate only)"
    )


def effective_real_time_budget_sec(args, fidelity_level):
    normalized = FidelityLevel.normalize(fidelity_level).value
    budget_by_fidelity = {
        FidelityLevel.HIGH.value: getattr(args, "real_time_budget_high_sec", None),
        FidelityLevel.MEDIUM.value: getattr(args, "real_time_budget_medium_sec", None),
        FidelityLevel.LOW.value: getattr(args, "real_time_budget_low_sec", None),
    }
    selected = budget_by_fidelity.get(normalized)
    if selected is None:
        selected = getattr(args, "real_time_budget_sec", 30.0)
    return float(selected)


def build_window_start_times(mobility_trace, window_size_seconds, step_seconds):
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be positive")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    total_duration_seconds = mobility_trace.total_duration_seconds()
    last_valid_start = total_duration_seconds - float(window_size_seconds)
    if last_valid_start < 0:
        return [], total_duration_seconds, last_valid_start
    starts = []
    current_start = 0.0
    tolerance = 1e-9
    while current_start <= last_valid_start + tolerance:
        starts.append(round(current_start, 6))
        current_start += float(step_seconds)
    return starts, total_duration_seconds, last_valid_start


def build_window_output_dir(base_output_dir, window_index, window_start_seconds, window_end_seconds):
    label = (
        f"window_{window_index:04d}_"
        f"{int(round(window_start_seconds)):04d}s_"
        f"{int(round(window_end_seconds)):04d}s"
    )
    return os.path.join(base_output_dir, label)


def build_trial_output_dir(base_output_dir, trial_index, fidelity_level):
    label = f"trial_{trial_index:04d}_{FidelityLevel.normalize(fidelity_level).value}"
    return os.path.join(base_output_dir, label)


def build_experiment_output_dirs(base_output_dir):
    experiment_root = os.path.join(base_output_dir, launch_timestamp_label())
    ran_fusion_dir = os.path.join(experiment_root, "ran_fusion_internal")
    morabito_dir = os.path.join(experiment_root, "morabito_ns3")
    logs_dir = os.path.join(experiment_root, "logs")
    os.makedirs(ran_fusion_dir, exist_ok=True)
    os.makedirs(morabito_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return {
        "experiment_root": experiment_root,
        "ran_fusion_internal": ran_fusion_dir,
        "morabito_ns3": morabito_dir,
        "logs": logs_dir,
    }


def build_wandb_state(output_layout, args):
    run_name = getattr(args, "wandb_run_name", None) or os.path.basename(output_layout["experiment_root"])
    state = {
        "run_id": uuid.uuid4().hex,
        "run_name": run_name,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "mode": args.wandb_mode,
        "experiment_root": output_layout["experiment_root"],
        "next_internal_step": 0,
        "next_window_step": 0,
    }
    state_path = os.path.join(output_layout["experiment_root"], "wandb_state.json")
    write_json(state_path, state)
    return state_path, state


def sync_window_to_wandb(
    args,
    *,
    wandb_state_path,
    window_metadata,
    training_csv=None,
    internal_plot_paths=None,
    final_topology_plot_path=None,
    gnb_timeline_csv=None,
    gnb_timeline_plot_dir=None,
    morabito_feedback_csv=None,
    morabito_plot_data_csv=None,
    morabito_plot_dir=None,
    mab_history_csv=None,
    mab_plot_dir=None,
):
    if not getattr(args, "enable_wandb", False):
        return False
    payload = {
        "wandb_state_json": wandb_state_path,
        "window": window_metadata,
        "training_csv": training_csv,
        "internal_plot_paths": internal_plot_paths or [],
        "final_topology_plot_path": final_topology_plot_path,
        "gnb_timeline_csv": gnb_timeline_csv,
        "gnb_timeline_plot_dir": gnb_timeline_plot_dir,
        "morabito_feedback_csv": morabito_feedback_csv,
        "morabito_plot_data_csv": morabito_plot_data_csv,
        "morabito_plot_dir": morabito_plot_dir,
        "mab_history_csv": mab_history_csv,
        "mab_plot_dir": mab_plot_dir,
    }
    payload_path = os.path.join(
        os.path.dirname(training_csv or gnb_timeline_csv or morabito_feedback_csv or wandb_state_path),
        f"wandb_sync_window_{int(window_metadata['window_index']):04d}.json",
    )
    write_json(payload_path, payload)
    return invoke_plot_helper(
        "wandb_sync_experiment.py",
        ["--payload-json", payload_path],
        logger=ORCH_LOGGER,
        event_prefix="wandb_sync",
    )


def clone_args(args, **overrides):
    payload = dict(vars(args))
    payload.update(overrides)
    return argparse.Namespace(**payload)


def reset_network_singletons():
    from network.NetworkLoadManager import NetworkLoadManager
    from network.cell import cell_instances
    from network.cell_manager import CellManager
    from network.gNodeB import gNodeB_instances
    from network.gNodeB_manager import gNodeBManager
    from network.loadbalancer import LoadBalancer
    from network.sector import all_sectors, global_ue_ids, sector_instances
    from network.sector_manager import SectorManager
    from network.ue import UE
    from network.ue_manager import UEManager

    gNodeBManager._instance = None
    CellManager._instance = None
    SectorManager._instance = None
    UEManager._instance = None
    NetworkLoadManager._instance = None
    LoadBalancer._instance = None

    gNodeB_instances.clear()
    cell_instances.clear()
    sector_instances.clear()
    all_sectors.clear()
    global_ue_ids.clear()
    UE.existing_ue_ids.clear()
    UE.ue_instances.clear()


def normalize_coverage_radius_to_meters(value):
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return 0.0
    if radius <= 0:
        return 0.0
    if radius < 1.0:
        return radius * 1000.0
    return radius


def project_lat_lon_to_local_xy_m(lat, lon, origin_lat, origin_lon):
    earth_radius_m = 6371000.0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)
    x = (lon_rad - origin_lon_rad) * math.cos((lat_rad + origin_lat_rad) / 2.0) * earth_radius_m
    y = (lat_rad - origin_lat_rad) * earth_radius_m
    return x, y


def build_final_topology_snapshot(env, observation, info):
    timestamp = None
    if info:
        timestamp = info.get("timestamp")
    if timestamp is None and observation:
        timestamp = observation.get("timestamp")
    gnb_records = []
    for gnb_id, gnb in sorted(env.gnodeb_manager.gNodeBs.items()):
        gnb_records.append(
            {
                "gnb_id": gnb_id,
                "lat": getattr(gnb, "Latitude", None),
                "lon": getattr(gnb, "Longitude", None),
                "x_m": getattr(gnb, "x_m", None),
                "y_m": getattr(gnb, "y_m", None),
                "coverage_radius_m": normalize_coverage_radius_to_meters(getattr(gnb, "CoverageRadius", 0.0)),
                "is_active": bool(getattr(gnb, "is_active", True)),
                "tx_power": getattr(gnb, "TransmissionPower", None),
            }
        )
    ue_records = []
    for ue_id, ue_state in sorted((observation or {}).get("ues", {}).items()):
        ue_records.append(
            {
                "ue_id": ue_id,
                "lat": ue_state.get("lat"),
                "lon": ue_state.get("lon"),
                "x_m": ue_state.get("x_m"),
                "y_m": ue_state.get("y_m"),
                "nominal_serving_bs_id": ue_state.get("nominal_serving_bs_id"),
                "serving_gnb_id": ue_state.get("serving_gnb_id"),
                "serving_distance_m": ue_state.get("serving_distance_m"),
                "distance_throughput_factor": ue_state.get("distance_throughput_factor"),
                "distance_throughput_tier": ue_state.get("distance_throughput_tier"),
                "throughput": ue_state.get("throughput"),
                "in_outage": ue_state.get("in_outage"),
            }
        )
    return {
        "timestamp": timestamp,
        "gnbs": gnb_records,
        "ues": ue_records,
        "distance_thresholds": dict(env.distance_throughput_config),
        "metrics": dict((observation or {}).get("metrics", {})),
    }


def apply_socket_configuration_to_topology_snapshot(snapshot, final_payload):
    best_configuration = (final_payload or {}).get("best_configuration") or {}
    desired_states = best_configuration.get("desired_gnb_states") or {}
    observed_states = best_configuration.get("gnb_state_after_step") or {}
    active_count = 0
    inactive_count = 0
    for record in snapshot.get("gnbs", []):
        gnb_id = record["gnb_id"]
        if gnb_id in desired_states:
            configured_on = bool(desired_states[gnb_id])
            state_source = "socket_desired_gnb_states"
        else:
            observed_state = observed_states.get(gnb_id, {})
            if isinstance(observed_state, dict) and "is_active" in observed_state:
                configured_on = bool(observed_state["is_active"])
                state_source = "socket_observed_gnb_state_after_step"
            else:
                configured_on = bool(record.get("is_active", True))
                state_source = "runtime_snapshot"
        record["socket_config_is_active"] = configured_on
        record["socket_config_state_source"] = state_source
        if configured_on:
            active_count += 1
        else:
            inactive_count += 1
    snapshot["socket_configuration"] = {
        "active_gnb_count": active_count,
        "inactive_gnb_count": inactive_count,
        "best_action_label": best_configuration.get("label"),
        "best_action_mask": best_configuration.get("mask"),
    }
    return snapshot


def generate_final_topology_plot(snapshot, *, output_dir, output_filename, config):
    mpl_config_dir = os.path.join(output_dir, ".matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    path = os.path.join(output_dir, output_filename)
    gnb_records = list(snapshot.get("gnbs", []))
    ue_records = list(snapshot.get("ues", []))
    if not gnb_records or not ue_records:
        raise ValueError("Final topology plot requires at least one gNB and one UE")

    ue_with_xy = [item for item in ue_records if item.get("x_m") is not None and item.get("y_m") is not None]
    use_trace_xy = bool(ue_with_xy)
    origin_lat = None
    origin_lon = None
    if not use_trace_xy:
        valid_latlon = [item for item in gnb_records + ue_records if item.get("lat") is not None and item.get("lon") is not None]
        if not valid_latlon:
            raise ValueError("Final topology plot requires UE/gNB coordinates; neither x/y nor lat/lon were available")
        origin_lat = sum(item["lat"] for item in valid_latlon) / len(valid_latlon)
        origin_lon = sum(item["lon"] for item in valid_latlon) / len(valid_latlon)

    for record in ue_records:
        if use_trace_xy and record.get("x_m") is not None and record.get("y_m") is not None:
            record["plot_x_m"] = float(record["x_m"])
            record["plot_y_m"] = float(record["y_m"])
        elif record.get("lat") is not None and record.get("lon") is not None:
            ux, uy = project_lat_lon_to_local_xy_m(record["lat"], record["lon"], origin_lat, origin_lon)
            record["plot_x_m"] = ux
            record["plot_y_m"] = uy
        else:
            record["plot_x_m"] = None
            record["plot_y_m"] = None

    gnb_positions = {}
    for record in gnb_records:
        if record.get("x_m") is not None and record.get("y_m") is not None:
            gx = float(record["x_m"])
            gy = float(record["y_m"])
        else:
            candidate_ues = [
                ue for ue in ue_records
                if ue.get("plot_x_m") is not None
                and ue.get("plot_y_m") is not None
                and ue.get("serving_gnb_id") == record["gnb_id"]
            ]
            if candidate_ues:
                gx = sum(ue["plot_x_m"] for ue in candidate_ues) / len(candidate_ues)
                gy = sum(ue["plot_y_m"] for ue in candidate_ues) / len(candidate_ues)
            elif record.get("lat") is not None and record.get("lon") is not None and origin_lat is not None and origin_lon is not None:
                gx, gy = project_lat_lon_to_local_xy_m(record["lat"], record["lon"], origin_lat, origin_lon)
            else:
                gx = gy = None
        record["plot_x_m"] = gx
        record["plot_y_m"] = gy
        if gx is not None and gy is not None:
            gnb_positions[record["gnb_id"]] = (gx, gy)

    tier_colors = {
        "near": "#1b9e77",
        "mid": "#d95f02",
        "far": "#7570b3",
        "edge": "#e7298a",
        "unknown": "#888888",
        "outage": "#cc0000",
        "sinr_efficiency": "#2a9d8f",
        "pdcp_compatible": "#4c78a8",
    }
    gnb_colors = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for index, record in enumerate(gnb_records):
        gnb_colors[record["gnb_id"]] = palette[index % len(palette)]

    fig, ax = plt.subplots(
        figsize=(float(config["plot_image_width"]), float(config["plot_image_height"])),
        dpi=int(config["dpi"]),
    )

    socket_configuration = snapshot.get("socket_configuration", {})

    # Build a per-gNB active map using socket_config_is_active (best policy) when available,
    # falling back to the runtime is_active.  This resolves the temporal mismatch where the
    # gNB runtime state reflects a post-training policy evaluation while UE observations come
    # from the last training episode (where the gNB might still have been on).
    gnb_active_map: dict = {
        r["gnb_id"]: bool(r.get("socket_config_is_active", r.get("is_active", True)))
        for r in gnb_records
    }

    for record in gnb_records:
        if record.get("plot_x_m") is None or record.get("plot_y_m") is None:
            continue
        gx = record["plot_x_m"]
        gy = record["plot_y_m"]
        config_is_active = gnb_active_map[record["gnb_id"]]
        marker = "^" if config_is_active else "X"
        marker_size = 180 if config_is_active else 220
        facecolor = gnb_colors[record["gnb_id"]] if config_is_active else "white"
        edgecolor = "black" if config_is_active else "#b22222"
        linewidth = 0.8 if config_is_active else 1.6
        ax.scatter(gx, gy, marker=marker, s=marker_size, color=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=6)
        if config["show_gnb_labels"]:
            state_label = "ON" if config_is_active else "OFF"
            ax.text(gx + 8.0, gy + 8.0, f"{record['gnb_id']} [{state_label}]", fontsize=9, weight="bold", color=gnb_colors[record["gnb_id"]])

    show_ue_labels = bool(config["show_ue_labels"]) and len(ue_records) <= 30
    draw_serving_lines = bool(config["draw_serving_lines"])
    for record in ue_records:
        if record.get("plot_x_m") is None or record.get("plot_y_m") is None:
            continue
        ux = record["plot_x_m"]
        uy = record["plot_y_m"]
        tier = record.get("distance_throughput_tier") or "unknown"
        serving_gnb_id = record.get("serving_gnb_id")
        ax.scatter(
            ux,
            uy,
            marker="o",
            s=55 if not record.get("in_outage") else 70,
            color=tier_colors.get(tier, "#888888"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
            zorder=5,
        )
        if (
            draw_serving_lines
            and serving_gnb_id in gnb_positions
            and not record.get("in_outage")
            and gnb_active_map.get(serving_gnb_id, True)
        ):
            gx, gy = gnb_positions[serving_gnb_id]
            ax.plot([gx, ux], [gy, uy], color=gnb_colors.get(serving_gnb_id, "#999999"), linewidth=0.4, alpha=0.35, zorder=2)
        if show_ue_labels:
            ax.text(ux + 3.0, uy + 3.0, record["ue_id"], fontsize=6, alpha=0.8)

    handles = [
        Line2D([0], [0], marker="^", color="w", label="gNB ON", markerfacecolor="#333333", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="X", color="w", label="gNB OFF", markerfacecolor="white", markeredgecolor="#b22222", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="UE (connected)", markerfacecolor="#2a9d8f", markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="UE (outage)", markerfacecolor="#cc0000", markeredgecolor="white", markersize=7),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)

    runtime_on  = sum(1 for gnb_id, active in gnb_active_map.items() if active)
    runtime_off = len(gnb_active_map) - runtime_on
    best_label = socket_configuration.get("best_action_label") or "n/a"
    ax.set_title(
        "Final Network Topology (last sim step)\n"
        f"UEs={len(ue_records)}  gNBs={len(gnb_records)}  "
        f"ON={runtime_on}  OFF={runtime_off}  "
        f"t={snapshot.get('timestamp')}  best_policy={best_label}",
        fontsize=11,
    )
    ax.set_xlabel("Local X [m]")
    ax.set_ylabel("Local Y [m]")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return path


def snapshot_best_result(action, env_action, observation, info, reward, episode, step_index, iteration_index, elapsed_real_sec):
    metrics = dict(info.get("metrics", {}))
    summary = dict(info.get("summary", {}))
    return {
        "reward": reward,
        "episode": episode,
        "step_in_episode": step_index,
        "iteration": iteration_index,
        "elapsed_real_sec": elapsed_real_sec,
        "sim_timestamp": info.get("timestamp"),
        "action_index": action.get("mask"),
        "action_label": action.get("label"),
        "action_mask": action.get("mask"),
        "best_configuration": {
            "action": env_action,
            "label": action.get("label"),
            "mask": action.get("mask"),
            "desired_gnb_states": dict(sorted(action.get("desired", {}).items())),
            "gnb_state_after_step": observation.get("gnbs", {}),
        },
        "kpis": {
            "throughput": metrics.get("total_throughput"),
            "delay": metrics.get("ue_delay"),
            "packet_loss": metrics.get("packet_loss"),
            "power_consumption": metrics.get("current_power_w"),
            "energy_cost": metrics.get("energy_cost"),
            "energy_saving": metrics.get("energy_saving"),
            "step_energy_wh": metrics.get("step_energy_wh"),
            "step_energy_kwh": metrics.get("step_energy_kwh"),
            "total_energy_wh": metrics.get("total_energy_wh"),
            "total_energy_kwh": metrics.get("total_energy_kwh"),
            "outage_ue_count": metrics.get("outage_ue_count"),
            "active_gnb_count": metrics.get("active_gnb_count"),
            "ue_associations": summary.get("serving_gnb_counts"),
        },
        "metrics": metrics,
        "summary": summary,
    }


def build_final_socket_payload(
    runtime_result,
    best_result,
    final_iteration,
    agent_id,
    window_metadata=None,
    *,
    socket_export_policy="best",
    policy_warm_started=False,
):
    payload = {
        "event": "real_time_budget_expired" if runtime_result["budget_expired"] else "run_completed",
        "real_time_budget_sec": runtime_result["real_time_budget_sec"],
        "actual_elapsed_real_sec": runtime_result["actual_elapsed_real_sec"],
        "best_configuration": best_result["best_configuration"] if best_result else None,
        "best_reward": (
            best_result.get("checkpoint_episode_reward", best_result["reward"])
            if best_result else None
        ),
        "best_found_at_real_sec": best_result["elapsed_real_sec"] if best_result else None,
        "best_found_at_sim_time": (
            best_result.get("checkpoint_sim_timestamp", best_result["sim_timestamp"])
            if best_result else None
        ),
        "final_iteration": final_iteration,
        "agent_id": agent_id,
        "kpis": best_result["kpis"] if best_result else {},
        "real_start_wall_utc": runtime_result["real_start_wall_utc"],
        "real_end_wall_utc": runtime_result["real_end_wall_utc"],
        "socket_export_policy": socket_export_policy,
        "policy_warm_started": bool(policy_warm_started),
    }
    if window_metadata is not None:
        payload["window"] = dict(window_metadata)
    if best_result:
        payload["best_episode"] = best_result["episode"]
        payload["best_step_in_episode"] = best_result["step_in_episode"]
        payload["best_action_label"] = best_result["action_label"]
        payload["best_policy_source"] = best_result.get("policy_source")
        payload["best_policy_state_origin"] = best_result.get("policy_state_origin")
        payload["policy_evaluation_reward"] = best_result.get("reward")
    return payload


def select_greedy_action_from_checkpoint(checkpoint, observation):
    if not checkpoint or not observation:
        return None
    control_approach = checkpoint.get("control_approach", "single_agent")
    if control_approach == "multi_agent":
        gnb_ids = checkpoint.get("gnb_ids") or sorted((observation.get("gnbs") or {}).keys())
        agent_q_tables = checkpoint.get("agent_q_tables") or {}
        local_state = local_user_load_state(observation)
        desired = {}
        for gnb_id in gnb_ids:
            state_value = int(local_state.get(gnb_id, 0))
            state_values = (agent_q_tables.get(gnb_id) or {}).get(state_value)
            if not state_values:
                desired[gnb_id] = True
                continue
            best_value = max(state_values)
            best_indices = [idx for idx, value in enumerate(state_values) if value == best_value]
            desired[gnb_id] = bool(min(best_indices))
        desired = enforce_multi_agent_constraints(
            desired,
            local_state,
            gnb_ids,
            allow_all_off=bool(checkpoint.get("allow_all_off", False)),
            min_active_gnbs=int(checkpoint.get("min_active_gnbs", 1)),
        )
        return desired_map_to_action(desired, gnb_ids)
    actions = checkpoint.get("actions") or []
    if not actions:
        return None
    if control_approach == "ucb_stateless":
        action_counts = checkpoint.get("ucb_action_counts") or []
        action_total_rewards = checkpoint.get("ucb_action_total_rewards") or []
        if len(action_counts) == len(actions) and len(action_total_rewards) == len(actions):
            tried_indices = [idx for idx, count in enumerate(action_counts) if float(count) > 0.0]
            if tried_indices:
                best_index = max(
                    tried_indices,
                    key=lambda idx: (
                        float(action_total_rewards[idx]) / max(float(action_counts[idx]), 1.0),
                        -idx,
                    ),
                )
                return actions[int(best_index)]
        return actions[0]

    state = state_key(None, observation)
    q_table = checkpoint.get("q_table") or {}
    values = q_table.get(state)
    if values:
        best_value = max(values)
        best_indices = [idx for idx, value in enumerate(values) if value == best_value]
        return actions[min(best_indices)]

    # Fallback for unseen states: pick the action with the best average Q-value across known states.
    if q_table:
        aggregate = [0.0 for _ in actions]
        count = 0
        for state_values in q_table.values():
            if len(state_values) != len(actions):
                continue
            count += 1
            for idx, value in enumerate(state_values):
                aggregate[idx] += value
        if count:
            averaged = [value / count for value in aggregate]
            best_value = max(averaged)
            best_indices = [idx for idx, value in enumerate(averaged) if value == best_value]
            return actions[min(best_indices)]

    return actions[0]


def build_policy_result_from_checkpoint(env, args, checkpoint, *, policy_source_label):
    if checkpoint is None:
        return None

    suppress_simulator_output = args.quiet_simulator_logs and not args.verbose_simulator_logs
    with maybe_suppress_stdout(suppress_simulator_output):
        observation, _ = env.reset()
    action = select_greedy_action_from_checkpoint(checkpoint, observation)
    if action is None:
        return None
    env_action = action_to_env_action(action)
    with maybe_suppress_stdout(suppress_simulator_output):
        next_observation, reward, done, info = env.step(env_action)

    result = snapshot_best_result(
        action=action,
        env_action=env_action,
        observation=next_observation,
        info=info,
        reward=reward,
        episode=checkpoint.get("episode"),
        step_index=0,
        iteration_index=checkpoint.get("iteration"),
        elapsed_real_sec=checkpoint.get("elapsed_real_sec"),
    )
    result["policy_source"] = policy_source_label
    result["policy_state_origin"] = "initial_observation_reset"
    result["checkpoint_episode_reward"] = checkpoint.get("episode_reward")
    result["checkpoint_sim_timestamp"] = checkpoint.get("sim_timestamp")
    result["checkpoint_episode_length"] = checkpoint.get("episode_length")
    result["done_after_policy_step"] = done
    return result


def send_best_configuration_over_socket(host, port, payload, socket_timeout_ms):
    message = json.dumps(payload).encode("utf-8")
    timeout_seconds = None if socket_timeout_ms is None or socket_timeout_ms <= 0 else max(
        0.001, socket_timeout_ms / 1000.0
    )
    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        sock.sendall(message)
        sock.shutdown(socket.SHUT_WR)


def send_best_configuration_over_socket_until_success(
    host,
    port,
    payload,
    socket_timeout_ms,
    retry_delay_seconds=1.0,
):
    attempt = 0
    started_at = time.monotonic()
    retry_started = False
    while True:
        attempt += 1
        try:
            send_best_configuration_over_socket(
                host=host,
                port=port,
                payload=payload,
                socket_timeout_ms=socket_timeout_ms,
            )
            log_event(
                ORCH_LOGGER,
                "socket_export_succeeded",
                host=host,
                port=port,
                attempts=attempt,
                elapsed_sec=time.monotonic() - started_at,
            )
            return
        except Exception as exc:
            if not retry_started:
                retry_started = True
                log_event(
                    ORCH_LOGGER,
                    "socket_export_retry_started",
                    host=host,
                    port=port,
                    attempt=attempt,
                    error=exc,
                )
            elif attempt == 30:
                log_event(
                    ORCH_LOGGER,
                    "energy_saving_not_ready_yet",
                    host=host,
                    port=port,
                    attempts=attempt,
                    elapsed_sec=time.monotonic() - started_at,
                )
            elif attempt % 10 == 0:
                log_event(
                    ORCH_LOGGER,
                    "socket_export_retry_progress",
                    host=host,
                    port=port,
                    attempt=attempt,
                    elapsed_sec=time.monotonic() - started_at,
                    last_error=exc,
                )
            time.sleep(retry_delay_seconds)


def send_configuration_and_wait_for_external_reward(
    *,
    host,
    port,
    payload,
    socket_timeout_ms,
    reward_listener,
):
    server_ready = threading.Event()
    payload_holder = {}
    error_holder = {}

    def _wait_for_reward():
        try:
            payload_holder["payload"] = reward_listener.wait_for_payload(
                server_ready_event=server_ready,
                expected_request_id=payload.get("request_id"),
            )
        except Exception as exc:
            error_holder["error"] = exc

    listener_thread = threading.Thread(
        target=_wait_for_reward,
        name="reward-listener",
        daemon=True,
    )
    listener_thread.start()
    deadline = time.time() + 5.0
    while not server_ready.is_set():
        if "error" in error_holder:
            listener_thread.join(timeout=0)
            raise error_holder["error"]
        if not listener_thread.is_alive():
            raise RuntimeError(
                f"Reward listener thread for port {reward_listener.port} exited before becoming ready."
            )
        if time.time() >= deadline:
            raise TimeoutError(
                f"Reward listener on port {reward_listener.port} was not ready before socket export."
            )
        time.sleep(0.01)

    send_best_configuration_over_socket_until_success(
        host=host,
        port=port,
        payload=payload,
        socket_timeout_ms=socket_timeout_ms,
    )
    listener_thread.join()
    if "error" in error_holder:
        raise error_holder["error"]
    payload = payload_holder["payload"]
    payload["reward"] = float(payload["reward"])
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple RL agent for trace-driven gNB ON/OFF control.")
    parser.add_argument("--mobility-profile", choices=["night", "peak"], default=DEFAULT_MOBILITY_PROFILE)
    parser.add_argument("--mobility-csv", default=None)
    parser.add_argument("--fidelity-level", choices=[level.value for level in FidelityLevel], default=FidelityLevel.HIGH.value)
    parser.add_argument("--medium-snapshot-sec", type=float, default=0.0)
    parser.add_argument(
        "--enable-fidelity-mab",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mab-algorithm", choices=["epsilon_greedy", "ucb1"], default="ucb1")
    parser.add_argument("--mab-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--mab-min-initial-pulls-per-arm",
        type=int,
        default=5,
        help="Minimum number of pulls to force for each fidelity arm before the external MAB switches to UCB/epsilon-greedy selection.",
    )
    parser.add_argument(
        "--mab-num-trials",
        type=int,
        default=10,
        help="Deprecated in per-window MAB mode; kept only for backward CLI compatibility and currently ignored.",
    )
    parser.add_argument("--reward-port", type=int, default=5002)
    parser.add_argument(
        "--ignore-first-external-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat the first external ns-3 reward as warm-up: keep the action export, but do not update the MAB with that first reward.",
    )
    parser.add_argument(
        "--reward-timeout-sec",
        default=None,
        help="Timeout in seconds while waiting for the external MAB reward on port 5002. Use 'none' to wait indefinitely.",
    )
    parser.add_argument("--enable-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="ranfusion-nsorangym")
    parser.add_argument("--wandb-entity", default="xraulz")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--static-csv", default=DEFAULT_STATIC_CSV)
    parser.add_argument("--bs-mapping", default=os.path.join(REPO_ROOT, "Config_files", "bs_gnb_mapping.json"))
    parser.add_argument("--allow-unmapped-bs", action="store_true")
    parser.add_argument("--throughput-mode", choices=["full_dl_reference", "legacy", "deterministic", "ns_oran_compatible"], default="ns_oran_compatible")
    parser.add_argument("--reward-mode", choices=["throughput_active_gnb", "raw_mbps", "legacy"], default="raw_mbps")
    parser.add_argument("--throughput-normalization-mode", choices=["fixed_reference"], default="fixed_reference")
    parser.add_argument("--throughput-reference-mbps", type=float, default=100.0)
    parser.add_argument("--reward-alpha", type=float, default=1.0)
    parser.add_argument("--reward-beta", type=float, default=-2.0)
    parser.add_argument("--reward-gamma", type=float, default=-5.0)
    parser.add_argument("--max-ues-per-gnb", type=int, default=10)
    parser.add_argument("--max-coverage-distance-m", type=float, default=500.0,
                        help="Max distance (m) within which a UE can associate to a gNB. "
                             "UEs farther from all active gNBs go into outage (default: 500).")
    parser.add_argument(
        "--control-approach",
        choices=["multi_agent", "single_agent", "ucb_stateless"],
        default="ucb_stateless",
        help=(
            "Internal controller used inside RAN FUSION. "
            "multi_agent uses one epsilon-greedy RU/gNB agent each; "
            "single_agent uses the legacy joint action space; "
            "ucb_stateless uses a single no-state UCB bandit over full gNB ON/OFF masks."
        ),
    )
    parser.add_argument("--max-users", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=1_000_000)
    parser.add_argument("--max-steps-per-episode", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ucb-exploration-coef", type=float, default=2.0)
    parser.add_argument(
        "--ucb-window-memory-mode",
        choices=["reset_stats_prioritize_previous", "carry_stats"],
        default="reset_stats_prioritize_previous",
        help=(
            "Across windows for ucb_stateless: reset_stats_prioritize_previous resets UCB counts/rewards "
            "every window but retries previously promising actions first; carry_stats preserves the current "
            "behavior and keeps the full UCB statistics across windows."
        ),
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay", type=float, default=0.9901)
    parser.add_argument(
        "--carry-policy-across-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the internal Q-values/policy across sliding windows. Exploration can still be reset independently.",
    )
    parser.add_argument(
        "--socket-export-policy",
        choices=["best", "last"],
        default="best",
        help="Which learned policy snapshot to export to ns-3: the best checkpoint seen during the window or the final checkpoint at budget end.",
    )
    parser.add_argument("--window-size-seconds", type=float, default=30.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--reset-exploration-each-window", dest="reset_exploration_each_window", action="store_true", default=True)
    parser.add_argument("--no-reset-exploration-each-window", dest="reset_exploration_each_window", action="store_false")
    parser.add_argument("--exploration-reset-value", type=float, default=1)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "test_mab"))
    parser.add_argument("--real-time-budget-sec", type=float, default=30.0)
    parser.add_argument(
        "--real-time-budget-high-sec",
        type=float,
        default=5.0,
        help="Internal training budget in seconds for HIGH fidelity windows.",
    )
    parser.add_argument(
        "--real-time-budget-medium-sec",
        type=float,
        default=10.0,
        help="Internal training budget in seconds for MEDIUM fidelity windows.",
    )
    parser.add_argument(
        "--real-time-budget-low-sec",
        type=float,
        default=20.0,
        help="Internal training budget in seconds for LOW fidelity windows.",
    )
    parser.add_argument("--time-check-interval-ms", type=int, default=100)
    parser.add_argument("--socket-host", default="127.0.0.1")
    parser.add_argument("--socket-port", type=int, default=5001)
    parser.add_argument(
        "--socket-timeout-ms",
        type=int,
        default=0,
        help="Socket timeout in milliseconds for the final export. Use 0 or a negative value to disable the timeout.",
    )
    parser.add_argument("--enable-socket-export", action="store_true", default=True)
    parser.add_argument("--disable-socket-export", dest="enable_socket_export", action="store_false")
    parser.add_argument("--enable-final-topology-plot", action="store_true", default=True)
    parser.add_argument("--disable-final-topology-plot", dest="enable_final_topology_plot", action="store_false")
    parser.add_argument("--plot-output-path", default=None)
    parser.add_argument("--plot-output-filename", default="final_topology_60s.png")
    parser.add_argument("--show-ue-labels", action="store_true", default=False)
    parser.add_argument("--show-gnb-labels", action="store_true", default=True)
    parser.add_argument("--draw-serving-lines", action="store_true", default=True)
    parser.add_argument("--plot-image-width", type=float, default=12.0)
    parser.add_argument("--plot-image-height", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--deterministic-sector-capacity-bps", type=float, default=100_000_000.0)
    parser.add_argument("--full-dl-packet-size-bytes", type=int, default=1280)
    parser.add_argument("--full-dl-inter-packet-interval-seconds", type=float, default=0.0005)
    parser.add_argument("--full-dl-sector-capacity-bps", type=float, default=100_000_000.0)
    parser.add_argument(
        "--sinr-pathloss-model",
        choices=["log_distance", "fspl", "threegpp_umi_nlos"],
        default=None,
        help="Override the internal ns_oran_compatible pathloss model.",
    )
    parser.add_argument("--sinr-pathloss-exponent", type=float, default=None)
    parser.add_argument("--sinr-additional-pathloss-db", type=float, default=None)
    parser.add_argument("--sinr-spectral-efficiency-scale", type=float, default=None)
    parser.add_argument("--sinr-spectral-efficiency-max-bps-hz", type=float, default=None)
    parser.add_argument("--sinr-synthetic-db", type=float, default=None)
    parser.add_argument("--min-active-gnbs", type=int, default=2, help="Minimum active gNBs allowed in learned action masks.")
    parser.add_argument("--allow-all-off", action="store_true", help="Include the all-gNBs-OFF action in the action set.")
    parser.add_argument("--quiet-simulator-logs", action="store_true", help="Suppress noisy simulator print/log output during training.")
    parser.add_argument("--verbose-simulator-logs", action="store_true", help="Compatibility flag; simulator output is visible by default.")
    parser.add_argument("--run-baselines", action="store_true", default=True)
    parser.add_argument("--no-baselines", dest="run_baselines", action="store_false")
    parser.add_argument(
        "--stub-external",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local stubs for external dependencies such as InfluxDB/NTP. Enabled by default for local simulator runs.",
    )
    args = parser.parse_args()
    if args.window_size_seconds <= 0:
        parser.error("--window-size-seconds must be positive")
    if args.step_seconds <= 0:
        parser.error("--step-seconds must be positive")
    if args.mab_num_trials <= 0:
        parser.error("--mab-num-trials must be positive")
    if args.mab_epsilon < 0:
        parser.error("--mab-epsilon must be non-negative")
    if args.mab_min_initial_pulls_per_arm <= 0:
        parser.error("--mab-min-initial-pulls-per-arm must be positive")
    if args.real_time_budget_sec <= 0:
        parser.error("--real-time-budget-sec must be positive")
    if args.real_time_budget_high_sec <= 0:
        parser.error("--real-time-budget-high-sec must be positive")
    if args.real_time_budget_medium_sec <= 0:
        parser.error("--real-time-budget-medium-sec must be positive")
    if args.real_time_budget_low_sec <= 0:
        parser.error("--real-time-budget-low-sec must be positive")
    args.reward_timeout_sec = parse_optional_float(args.reward_timeout_sec)
    if args.mobility_csv is None:
        args.mobility_csv = resolve_mobility_trace_csv(args.mobility_profile, REPO_ROOT)
    return args


def make_env(args, seed_offset=0, mobility_provider=None):
    from rl.ran_trace_env import RANTraceEnv

    reset_network_singletons()
    ns_oran_sinr_config_overrides = {}
    for arg_name, config_key in (
        ("sinr_pathloss_model", "pathloss_model"),
        ("sinr_pathloss_exponent", "pathloss_exponent"),
        ("sinr_additional_pathloss_db", "additional_pathloss_db"),
        ("sinr_spectral_efficiency_scale", "spectral_efficiency_scale"),
        ("sinr_spectral_efficiency_max_bps_hz", "spectral_efficiency_max_bps_hz"),
        ("sinr_synthetic_db", "sinr_synthetic_db"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            ns_oran_sinr_config_overrides[config_key] = value
    return RANTraceEnv(
        base_dir=REPO_ROOT,
        mobility_trace_csv=args.mobility_csv,
        mobility_provider=mobility_provider,
        static_user_csv=args.static_csv,
        bs_mapping_path=args.bs_mapping,
        strict_bs_mapping=not args.allow_unmapped_bs,
        sleep_behavior="reattach",
        max_users=args.max_users,
        fidelity_level=args.fidelity_level,
        mobility_profile=args.mobility_profile,
        medium_snapshot_sec=args.medium_snapshot_sec,
        throughput_mode=args.throughput_mode,
        reward_mode=args.reward_mode,
        throughput_normalization_mode=args.throughput_normalization_mode,
        throughput_reference_mbps=args.throughput_reference_mbps,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        max_ues_per_gnb=args.max_ues_per_gnb,
        max_coverage_distance_m=args.max_coverage_distance_m,
        deterministic_sector_capacity_bps=args.deterministic_sector_capacity_bps,
        full_dl_packet_size_bytes=args.full_dl_packet_size_bytes,
        full_dl_inter_packet_interval_seconds=args.full_dl_inter_packet_interval_seconds,
        full_dl_sector_capacity_bps=args.full_dl_sector_capacity_bps,
        ns_oran_sinr_config_overrides=ns_oran_sinr_config_overrides,
        seed=args.seed + seed_offset,
    )


def build_action_space(gnb_ids, allow_all_off=False, min_active_gnbs=1):
    actions = []
    start = 0 if allow_all_off else 1
    for mask in range(start, 1 << len(gnb_ids)):
        desired = {gnb_id: bool(mask & (1 << index)) for index, gnb_id in enumerate(gnb_ids)}
        if sum(desired.values()) < min_active_gnbs:
            continue
        actions.append(desired_map_to_action(desired, gnb_ids))
    return actions


def desired_map_to_action(desired, gnb_ids):
    ordered_gnb_ids = list(gnb_ids)
    mask = 0
    for index, gnb_id in enumerate(ordered_gnb_ids):
        if bool(desired.get(gnb_id, False)):
            mask |= 1 << index
    return {
        "mask": mask,
        "desired": {gnb_id: bool(desired.get(gnb_id, False)) for gnb_id in ordered_gnb_ids},
        "label": "+".join(gnb_id for gnb_id in ordered_gnb_ids if desired.get(gnb_id, False)) or "ALL_OFF",
    }


def action_to_env_action(action):
    return [
        {"gnb_id": gnb_id, "state": "ON" if is_on else "OFF"}
        for gnb_id, is_on in sorted(action["desired"].items())
    ]


def state_key(env, observation):
    active_mask = tuple(
        (gnb_id, bool(state["is_active"]))
        for gnb_id, state in sorted(observation["gnbs"].items())
    )
    return active_mask


def local_user_load_state(observation):
    serving_counts = Counter()
    for ue_state in (observation.get("ues") or {}).values():
        serving_gnb_id = ue_state.get("serving_gnb_id")
        if not serving_gnb_id or ue_state.get("in_outage"):
            continue
        serving_counts[serving_gnb_id] += 1
    return {
        gnb_id: int(serving_counts.get(gnb_id, 0))
        for gnb_id in sorted((observation.get("gnbs") or {}).keys())
    }


def choose_binary_action(values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(2)
    best_value = max(values)
    best_indices = [idx for idx, value in enumerate(values) if value == best_value]
    return rng.choice(best_indices)


def enforce_multi_agent_constraints(desired, local_state, gnb_ids, *, allow_all_off, min_active_gnbs):
    adjusted = {gnb_id: bool(desired.get(gnb_id, False)) for gnb_id in gnb_ids}
    active_count = sum(adjusted.values())
    required_active = 0 if allow_all_off else max(1, int(min_active_gnbs))
    required_active = max(required_active, int(min_active_gnbs))
    if active_count >= required_active:
        return adjusted
    ranked_gnbs = sorted(gnb_ids, key=lambda gnb_id: (-int(local_state.get(gnb_id, 0)), gnb_id))
    for gnb_id in ranked_gnbs:
        if adjusted[gnb_id]:
            continue
        adjusted[gnb_id] = True
        active_count += 1
        if active_count >= required_active:
            break
    return adjusted


def build_multi_agent_action(agent_q_tables, observation, epsilon, rng, gnb_ids, args):
    local_state = local_user_load_state(observation)
    desired = {}
    action_indices = {}
    for gnb_id in gnb_ids:
        state_value = int(local_state.get(gnb_id, 0))
        values = agent_q_tables[gnb_id][state_value]
        action_index = choose_binary_action(values, epsilon, rng)
        action_indices[gnb_id] = action_index
        desired[gnb_id] = bool(action_index)
    desired = enforce_multi_agent_constraints(
        desired,
        local_state,
        gnb_ids,
        allow_all_off=args.allow_all_off,
        min_active_gnbs=args.min_active_gnbs,
    )
    return desired_map_to_action(desired, gnb_ids), local_state, action_indices


def choose_action(q_table, state, actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(len(actions))
    values = q_table[state]
    best_value = max(values)
    best_indices = [idx for idx, value in enumerate(values) if value == best_value]
    return rng.choice(best_indices)


def choose_ucb_action(action_counts, action_total_rewards, total_pulls, exploration_coef, rng):
    untried = [idx for idx, count in enumerate(action_counts) if int(count) <= 0]
    if untried:
        return rng.choice(untried)
    log_total = math.log(max(float(total_pulls), 1.0))
    scores = []
    for idx, count in enumerate(action_counts):
        mean_reward = float(action_total_rewards[idx]) / float(count)
        bonus = float(exploration_coef) * math.sqrt(log_total / float(count))
        scores.append(mean_reward + bonus)
    best_value = max(scores)
    best_indices = [idx for idx, value in enumerate(scores) if value == best_value]
    return rng.choice(best_indices)


def rank_ucb_actions_from_stats(action_counts, action_total_rewards):
    ranked = []
    for idx, count in enumerate(action_counts):
        count_int = int(count)
        total_reward = float(action_total_rewards[idx])
        mean_reward = (total_reward / float(count_int)) if count_int > 0 else float("-inf")
        ranked.append((mean_reward, count_int, -idx, idx))
    ranked.sort(reverse=True)
    return [idx for _mean_reward, _count, _neg_idx, idx in ranked]


def choose_ucb_action_with_priority(
    action_counts,
    action_total_rewards,
    total_pulls,
    exploration_coef,
    rng,
    priority_order=None,
):
    untried = [idx for idx, count in enumerate(action_counts) if int(count) <= 0]
    if untried:
        if priority_order:
            for idx in priority_order:
                if idx in untried:
                    return idx
        return rng.choice(untried)
    return choose_ucb_action(
        action_counts,
        action_total_rewards,
        total_pulls,
        exploration_coef,
        rng,
    )


def controller_policy_name(control_approach):
    if control_approach == "multi_agent":
        return "q_learning_multi_agent"
    if control_approach == "single_agent":
        return "q_learning_single_agent"
    if control_approach == "ucb_stateless":
        return "ucb_stateless"
    return str(control_approach)


def empty_episode_stats(episode, policy_name, throughput_reference_mbps=100.0):
    return {
        "episode": episode,
        "policy": policy_name,
        "policy_warm_started": 0,
        "episode_completed": 0,
        "budget_interrupted": 0,
        "episode_reward": 0.0,
        "mean_step_reward": 0.0,
        "total_throughput": 0.0,
        "mean_throughput": 0.0,
        "mean_throughput_mbps": 0.0,
        "normalized_mean_throughput": 0.0,
        "total_outage_count": 0.0,
        "mean_outage_count": 0.0,
        "total_energy_cost": 0.0,
        "mean_energy_cost": 0.0,
        "total_energy_saving": 0.0,
        "mean_energy_saving": 0.0,
        "total_step_energy_wh": 0.0,
        "mean_step_energy_wh": 0.0,
        "total_step_energy_kwh": 0.0,
        "mean_step_energy_kwh": 0.0,
        "mean_current_power_w": 0.0,
        "final_cumulative_energy_wh": 0.0,
        "final_cumulative_energy_kwh": 0.0,
        "mean_active_gnb_count": 0.0,
        "fallback_ue_count": 0.0,
        "nominal_serving_ue_count": 0.0,
        "episode_length": 0,
        "epsilon": "",
        "throughput_reference_mbps": float(throughput_reference_mbps),
    }


def update_episode_stats(stats, reward, info):
    summary = info["summary"]
    metrics = info["metrics"]
    stats["episode_reward"] += reward
    stats["total_throughput"] += metrics["total_throughput"]
    stats["total_outage_count"] += metrics["outage_ue_count"]
    stats["total_energy_cost"] += metrics["energy_cost"]
    stats["total_energy_saving"] += metrics["energy_saving"]
    stats["total_step_energy_wh"] += metrics.get("step_energy_wh", 0.0)
    stats["total_step_energy_kwh"] += metrics.get("step_energy_kwh", 0.0)
    stats["mean_current_power_w"] += metrics.get("current_power_w", 0.0)
    stats["final_cumulative_energy_wh"] = metrics.get("total_energy_wh", stats["final_cumulative_energy_wh"])
    stats["final_cumulative_energy_kwh"] = metrics.get("total_energy_kwh", stats["final_cumulative_energy_kwh"])
    stats["mean_active_gnb_count"] += metrics["active_gnb_count"]
    stats["fallback_ue_count"] += summary["fallback_reassigned_ues"]
    stats["nominal_serving_ue_count"] += summary["nominal_serving_ues"]
    stats["episode_length"] += 1


def finalize_episode_stats(stats, epsilon=None):
    length = max(1, stats["episode_length"])
    stats["mean_step_reward"] = stats["episode_reward"] / length
    stats["mean_throughput"] = stats["total_throughput"] / length
    stats["mean_throughput_mbps"] = stats["mean_throughput"] / 1_000_000.0
    throughput_reference_mbps = max(float(stats.get("throughput_reference_mbps", 100.0)), 1e-9)
    stats["normalized_mean_throughput"] = stats["mean_throughput_mbps"] / throughput_reference_mbps
    stats["mean_outage_count"] = stats["total_outage_count"] / length
    stats["mean_energy_cost"] = stats["total_energy_cost"] / length
    stats["mean_energy_saving"] = stats["total_energy_saving"] / length
    stats["mean_step_energy_wh"] = stats["total_step_energy_wh"] / length
    stats["mean_step_energy_kwh"] = stats["total_step_energy_kwh"] / length
    stats["mean_current_power_w"] = stats["mean_current_power_w"] / length
    stats["mean_active_gnb_count"] = stats["mean_active_gnb_count"] / length
    stats["fallback_ue_count"] = stats["fallback_ue_count"] / length
    stats["nominal_serving_ue_count"] = stats["nominal_serving_ue_count"] / length
    if epsilon is not None:
        stats["epsilon"] = epsilon
    return stats


def train_q_learning(
    args,
    *,
    mobility_provider=None,
    output_dir=None,
    initial_epsilon=None,
    window_metadata=None,
    initial_checkpoint=None,
):
    rng = random.Random(args.seed)
    suppress_simulator_output = args.quiet_simulator_logs and not args.verbose_simulator_logs
    effective_output_dir = output_dir or args.output_dir
    selected_fidelity = window_metadata.get("fidelity_level") if window_metadata else args.fidelity_level
    effective_budget_sec = effective_real_time_budget_sec(args, selected_fidelity)
    with maybe_suppress_stdout(suppress_simulator_output):
        env = make_env(args, mobility_provider=mobility_provider)
        initial_observation, _ = env.reset()
    runtime_controller = RealTimeBudgetController(effective_budget_sec, args.time_check_interval_ms)
    gnb_ids = sorted(env.gnodeb_manager.gNodeBs)
    control_approach = args.control_approach
    policy_name = controller_policy_name(control_approach)
    actions = []
    if control_approach in {"single_agent", "ucb_stateless"}:
        actions = build_action_space(gnb_ids, allow_all_off=args.allow_all_off, min_active_gnbs=args.min_active_gnbs)
        if not actions:
            raise ValueError("Action space is empty; reduce --min-active-gnbs or enable fewer constraints.")
    q_table = defaultdict(lambda: [0.0 for _ in actions]) if control_approach == "single_agent" else None
    ucb_action_counts = [0 for _ in actions] if control_approach == "ucb_stateless" else None
    ucb_action_total_rewards = [0.0 for _ in actions] if control_approach == "ucb_stateless" else None
    ucb_priority_order = None
    agent_q_tables = (
        {gnb_id: defaultdict(lambda: [0.0, 0.0]) for gnb_id in gnb_ids}
        if control_approach == "multi_agent"
        else None
    )
    checkpoint_loaded = False
    if initial_checkpoint and initial_checkpoint.get("control_approach") == control_approach:
        if control_approach == "single_agent" and q_table is not None:
            for state, values in (initial_checkpoint.get("q_table") or {}).items():
                if isinstance(values, list) and len(values) == len(actions):
                    q_table[state] = [float(value) for value in values]
                    checkpoint_loaded = True
        elif control_approach == "ucb_stateless" and ucb_action_counts is not None and ucb_action_total_rewards is not None:
            initial_counts = initial_checkpoint.get("ucb_action_counts") or []
            initial_totals = initial_checkpoint.get("ucb_action_total_rewards") or []
            if len(initial_counts) == len(actions) and len(initial_totals) == len(actions):
                checkpoint_loaded = True
                if args.ucb_window_memory_mode == "carry_stats":
                    ucb_action_counts = [int(count) for count in initial_counts]
                    ucb_action_total_rewards = [float(total) for total in initial_totals]
                else:
                    ucb_priority_order = rank_ucb_actions_from_stats(initial_counts, initial_totals)
        elif control_approach == "multi_agent" and agent_q_tables is not None:
            initial_agent_tables = initial_checkpoint.get("agent_q_tables") or {}
            for gnb_id in gnb_ids:
                local_table = initial_agent_tables.get(gnb_id) or {}
                for state_value, values in local_table.items():
                    if isinstance(values, list) and len(values) == 2:
                        agent_q_tables[gnb_id][int(state_value)] = [float(value) for value in values]
                        checkpoint_loaded = True
    log_event(
        RL_LOGGER,
        "controller_ready",
        window_index=window_metadata.get("window_index") if window_metadata else None,
        fidelity=window_metadata.get("fidelity_level") if window_metadata else args.fidelity_level,
        gnb_count=len(gnb_ids),
        action_count=len(actions) if control_approach in {"single_agent", "ucb_stateless"} else 2 * len(gnb_ids),
        action_space=(
            "gnb_on_off_masks"
            if control_approach == "single_agent"
            else "state_free_gnb_on_off_bandit"
            if control_approach == "ucb_stateless"
            else "independent_binary_ru_agents"
        ),
        control_approach=control_approach,
        ucb_window_memory_mode=args.ucb_window_memory_mode if control_approach == "ucb_stateless" else None,
        epsilon_start=(
            None
            if control_approach == "ucb_stateless"
            else initial_epsilon if initial_epsilon is not None else args.epsilon_start
        ),
        checkpoint_loaded=checkpoint_loaded,
        ucb_exploration_coef=args.ucb_exploration_coef if control_approach == "ucb_stateless" else None,
        real_time_budget_sec=effective_budget_sec,
    )
    epsilon = (
        float(args.epsilon_start if initial_epsilon is None else initial_epsilon)
        if control_approach != "ucb_stateless"
        else None
    )
    rows = []
    best_reward = -math.inf
    best_episode_reward = -math.inf
    best_checkpoint = None
    best_result = None
    iteration_index = 0
    last_observation = initial_observation
    last_info = None

    max_steps = args.max_steps_per_episode or len(env.mobility_trace.timestamps)
    budget_expired = False

    for episode in range(args.max_episodes):
        if runtime_controller.should_stop():
            budget_expired = True
            break
        if episode == 0:
            observation = initial_observation
        else:
            with maybe_suppress_stdout(suppress_simulator_output):
                observation, _ = env.reset()
        state = state_key(env, observation) if control_approach == "single_agent" else None
        stats = empty_episode_stats(episode, policy_name, throughput_reference_mbps=args.throughput_reference_mbps)
        stats["policy_warm_started"] = int(checkpoint_loaded)
        done = False
        episode_interrupted_by_budget = False

        for step_in_episode in range(max_steps):
            if runtime_controller.should_stop():
                budget_expired = True
                break
            if control_approach == "single_agent":
                action_index = choose_action(q_table, state, actions, epsilon, rng)
                action = actions[action_index]
                local_state = None
                action_indices = None
            elif control_approach == "ucb_stateless":
                total_pulls = sum(int(count) for count in ucb_action_counts)
                action_index = choose_ucb_action_with_priority(
                    ucb_action_counts,
                    ucb_action_total_rewards,
                    total_pulls,
                    args.ucb_exploration_coef,
                    rng,
                    priority_order=ucb_priority_order,
                )
                action = actions[action_index]
                local_state = None
                action_indices = None
            else:
                action, local_state, action_indices = build_multi_agent_action(
                    agent_q_tables,
                    observation,
                    epsilon,
                    rng,
                    gnb_ids,
                    args,
                )
                action_index = action.get("mask")
            env_action = action_to_env_action(action)
            with maybe_suppress_stdout(suppress_simulator_output):
                next_observation, reward, done, info = env.step(env_action)
            next_state = state_key(env, next_observation) if control_approach == "single_agent" else None
            iteration_index += 1
            elapsed_real_sec = runtime_controller.elapsed_real_sec()
            last_observation = next_observation
            last_info = info

            if control_approach == "single_agent":
                current = q_table[state][action_index]
                next_best = 0.0 if done else max(q_table[next_state])
                q_table[state][action_index] = current + args.learning_rate * (reward + args.gamma * next_best - current)
            elif control_approach == "ucb_stateless":
                ucb_action_counts[action_index] += 1
                ucb_action_total_rewards[action_index] += float(reward)
            else:
                next_local_state = local_user_load_state(next_observation)
                for gnb_id in gnb_ids:
                    state_value = int(local_state.get(gnb_id, 0))
                    next_state_value = int(next_local_state.get(gnb_id, 0))
                    local_action_index = int(action_indices[gnb_id])
                    current = agent_q_tables[gnb_id][state_value][local_action_index]
                    next_best = 0.0 if done else max(agent_q_tables[gnb_id][next_state_value])
                    agent_q_tables[gnb_id][state_value][local_action_index] = current + args.learning_rate * (
                        reward + args.gamma * next_best - current
                    )

            if reward > best_reward:
                best_reward = reward
                best_result = snapshot_best_result(
                    action=action,
                    env_action=env_action,
                    observation=next_observation,
                    info=info,
                    reward=reward,
                    episode=episode,
                    step_index=step_in_episode,
                    iteration_index=iteration_index,
                    elapsed_real_sec=elapsed_real_sec,
                )

            update_episode_stats(stats, reward, info)
            observation = next_observation
            state = next_state
            if done or runtime_controller.should_stop():
                episode_interrupted_by_budget = runtime_controller.should_stop()
                budget_expired = budget_expired or episode_interrupted_by_budget
                break

        finalize_episode_stats(stats, epsilon=epsilon if control_approach != "ucb_stateless" else None)
        stats["episode_completed"] = int(done)
        stats["budget_interrupted"] = int(episode_interrupted_by_budget)
        if not episode_interrupted_by_budget:
            rows.append(stats)
        else:
            log_event(
                RL_LOGGER,
                "partial_episode_discarded",
                window_index=(window_metadata or {}).get("window_index"),
                episode=episode,
                episode_reward=stats["episode_reward"],
                episode_length=stats["episode_length"],
            )

        checkpoint = {
            "q_table": dict(q_table) if q_table is not None else None,
            "ucb_action_counts": list(ucb_action_counts) if ucb_action_counts is not None else None,
            "ucb_action_total_rewards": list(ucb_action_total_rewards) if ucb_action_total_rewards is not None else None,
            "agent_q_tables": {
                gnb_id: dict(local_q_table)
                for gnb_id, local_q_table in (agent_q_tables or {}).items()
            } if agent_q_tables is not None else None,
            "actions": actions,
            "gnb_ids": gnb_ids,
            "args": vars(args),
            "control_approach": control_approach,
            "ucb_window_memory_mode": args.ucb_window_memory_mode if control_approach == "ucb_stateless" else None,
            "allow_all_off": bool(args.allow_all_off),
            "min_active_gnbs": int(args.min_active_gnbs),
            "episode": episode,
            "episode_reward": stats["episode_reward"],
            "episode_length": stats["episode_length"],
            "elapsed_real_sec": runtime_controller.elapsed_real_sec(),
            "sim_timestamp": last_info.get("timestamp") if last_info else None,
            "iteration": iteration_index,
            "best_result": best_result,
            "window_metadata": dict(window_metadata) if window_metadata else None,
            "policy_warm_started": bool(checkpoint_loaded),
        }
        if not episode_interrupted_by_budget and stats["episode_reward"] > best_episode_reward:
            best_episode_reward = stats["episode_reward"]
            best_checkpoint = checkpoint

        log_event(
            RL_LOGGER,
            "episode_completed",
            window_index=(window_metadata or {}).get("window_index"),
            episode=episode,
            episode_reward=stats["episode_reward"],
            mean_throughput_bps=stats["mean_throughput"],
            mean_throughput_mbps=stats["mean_throughput_mbps"],
            mean_outage=stats["mean_outage_count"],
            mean_energy=stats["mean_energy_cost"],
            mean_power=stats["mean_current_power_w"],
            final_energy_kwh=stats["final_cumulative_energy_kwh"],
            epsilon=epsilon if control_approach != "ucb_stateless" else None,
        )
        epsilon_display = f"{epsilon:.4f}" if control_approach != "ucb_stateless" and epsilon is not None else "n/a"
        print(
            f"episode={episode} "
            f"window={(window_metadata or {}).get('window_index')} "
            f"reward={stats['episode_reward']:.6f} "
            f"mean_throughput_bps={stats['mean_throughput']:.3f} "
            f"mean_throughput_mbps={stats['mean_throughput_mbps']:.3f} "
            f"mean_outage={stats['mean_outage_count']:.3f} "
            f"mean_energy={stats['mean_energy_cost']:.3f} "
            f"mean_power={stats['mean_current_power_w']:.3f} "
            f"final_energy_kwh={stats['final_cumulative_energy_kwh']:.6f} "
            f"epsilon={epsilon_display}"
        )
        if control_approach != "ucb_stateless":
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        if budget_expired:
            break

    final_checkpoint = {
        "q_table": dict(q_table) if q_table is not None else None,
        "ucb_action_counts": list(ucb_action_counts) if ucb_action_counts is not None else None,
        "ucb_action_total_rewards": list(ucb_action_total_rewards) if ucb_action_total_rewards is not None else None,
        "agent_q_tables": {
            gnb_id: dict(local_q_table)
            for gnb_id, local_q_table in (agent_q_tables or {}).items()
        } if agent_q_tables is not None else None,
        "actions": actions,
        "gnb_ids": gnb_ids,
        "args": vars(args),
        "control_approach": control_approach,
        "ucb_window_memory_mode": args.ucb_window_memory_mode if control_approach == "ucb_stateless" else None,
        "allow_all_off": bool(args.allow_all_off),
        "min_active_gnbs": int(args.min_active_gnbs),
        "episode": rows[-1]["episode"] if rows else None,
        "episode_reward": rows[-1]["episode_reward"] if rows else None,
        "episode_length": rows[-1]["episode_length"] if rows else None,
        "elapsed_real_sec": runtime_controller.elapsed_real_sec(),
        "sim_timestamp": last_info.get("timestamp") if last_info else None,
        "iteration": iteration_index,
        "best_result": best_result,
        "window_metadata": dict(window_metadata) if window_metadata else None,
        "policy_warm_started": bool(checkpoint_loaded),
    }
    runtime_result = runtime_controller.finalize()
    runtime_result["budget_expired"] = budget_expired or runtime_result["budget_expired"]
    return (
        rows,
        final_checkpoint,
        best_checkpoint,
        env,
        runtime_result,
        best_result,
        iteration_index,
        last_observation,
        last_info,
        epsilon,
    )


def evaluate_policy(args, env, policy_name, policy_fn):
    suppress_simulator_output = args.quiet_simulator_logs and not args.verbose_simulator_logs
    with maybe_suppress_stdout(suppress_simulator_output):
        observation, _ = env.reset()
    stats = empty_episode_stats(0, policy_name, throughput_reference_mbps=args.throughput_reference_mbps)
    done = False
    max_steps = args.max_steps_per_episode or len(env.mobility_trace.timestamps)

    for _ in range(max_steps):
        action = policy_fn(env, observation)
        with maybe_suppress_stdout(suppress_simulator_output):
            observation, reward, done, info = env.step(action)
        update_episode_stats(stats, reward, info)
        if done:
            break
    return finalize_episode_stats(stats)


def all_on_policy(env, _observation):
    return [{"gnb_id": gnb_id, "state": "ON"} for gnb_id in sorted(env.gnodeb_manager.gNodeBs)]


def heuristic_energy_policy(env, _observation):
    timestamp, samples = env._current_samples()
    nominal_counts = Counter()
    for sample in samples:
        mapped = env.bs_id_map.get(sample.serving_bs_id)
        if sample.covered and mapped:
            nominal_counts[mapped] += 1
    if not nominal_counts:
        keep_on = {next(iter(sorted(env.gnodeb_manager.gNodeBs)))}
    else:
        keep_on = set(nominal_counts)
    return [
        {"gnb_id": gnb_id, "state": "ON" if gnb_id in keep_on else "OFF"}
        for gnb_id in sorted(env.gnodeb_manager.gNodeBs)
    ]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def plot_training(output_dir, training_rows, training_csv_path=None):
    try:
        mpl_config_dir = os.path.join(output_dir, ".matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        log_event(RL_LOGGER, "plots_skipped", reason="matplotlib_import_error", error=exc)
        if training_csv_path:
            helper_args = ["--training-csv", training_csv_path, "--output-dir", output_dir]
            if invoke_plot_helper(
                "generate_ranfusion_internal_plots.py",
                helper_args,
                logger=RL_LOGGER,
                event_prefix="plots_helper",
            ):
                expected_paths = [
                    os.path.join(output_dir, name)
                    for name in [
                        "episodic_reward.png",
                        "mean_throughput.png",
                        "mean_outage_count.png",
                        "mean_energy_cost.png",
                        "mean_active_gnb_count.png",
                        "mean_current_power_w.png",
                        "final_cumulative_energy_kwh.png",
                    ]
                ]
                return [path for path in expected_paths if os.path.exists(path)]
        return []

    plot_specs = [
        ("episodic_reward.png", "episode_reward", "Episode Reward"),
        ("mean_throughput.png", "mean_throughput_mbps", "Mean Throughput [Mbps]"),
        ("mean_outage_count.png", "mean_outage_count", "Mean Outage Count"),
        ("mean_energy_cost.png", "mean_energy_cost", "Mean Energy Cost"),
        ("mean_active_gnb_count.png", "mean_active_gnb_count", "Mean Active gNB Count"),
        ("mean_current_power_w.png", "mean_current_power_w", "Mean Current Power [W]"),
        ("final_cumulative_energy_kwh.png", "final_cumulative_energy_kwh", "Final Cumulative Energy [kWh]"),
    ]
    paths = []
    episodes = [row["episode"] for row in training_rows]
    for filename, column, title in plot_specs:
        path = os.path.join(output_dir, filename)
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, [row[column] for row in training_rows], marker="o", linewidth=1)
        plt.xlabel("Episode")
        plt.ylabel(column)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_morabito_feedback_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_csv_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def compute_weighted_external_reward_components(
    *,
    reward_mode,
    reward_alpha,
    reward_beta,
    reward_gamma,
    aggregate_throughput_mbps,
    normalized_throughput,
    active_gnb_count,
    normalized_active_gnb_count,
    disconnected_ues,
    normalized_disconnected_ues,
):
    reward_mode = str(reward_mode or "throughput_active_gnb")
    reward_alpha = float(reward_alpha or 0.0)
    reward_beta = float(reward_beta or 0.0)
    reward_gamma = float(reward_gamma or 0.0)
    if reward_mode == "raw_mbps":
        throughput_component = reward_alpha * float(aggregate_throughput_mbps or 0.0)
        active_component = reward_beta * float(active_gnb_count or 0.0)
        disconnected_component = reward_gamma * float(disconnected_ues or 0.0)
    else:
        throughput_component = reward_alpha * float(normalized_throughput or 0.0)
        active_component = reward_beta * float(normalized_active_gnb_count or 0.0)
        disconnected_component = reward_gamma * float(normalized_disconnected_ues or 0.0)
    return {
        "reward_component_throughput_weighted": float(throughput_component),
        "reward_component_active_gnb_weighted": float(active_component),
        "reward_component_disconnected_ues_weighted": float(disconnected_component),
    }


def build_gnb_activation_row(window_metadata, final_payload):
    best_configuration = (final_payload or {}).get("best_configuration") or {}
    desired_states = best_configuration.get("desired_gnb_states") or {}
    sorted_gnbs = sorted(desired_states)
    row = {
        "window_index": int(window_metadata["window_index"]),
        "window_start_seconds": float(window_metadata["window_start_seconds"]),
        "window_end_seconds": float(window_metadata["window_end_seconds"]),
        "fidelity_level": window_metadata.get("fidelity_level"),
        "selected_arm": window_metadata.get("selected_arm"),
        "action_label": best_configuration.get("label"),
        "action_mask": best_configuration.get("mask"),
        "active_gnb_count": int(sum(1 for state in desired_states.values() if state)),
    }
    for gnb_id in sorted_gnbs:
        row[gnb_id] = int(bool(desired_states.get(gnb_id)))
    return row


def append_mab_history_row(path, *, window_metadata, selected_arm, selected_fidelity, reward, mab_statistics):
    arms = mab_statistics.get("arms", {})
    row = {
        "window_index": int(window_metadata["window_index"]),
        "window_start_seconds": float(window_metadata["window_start_seconds"]),
        "window_end_seconds": float(window_metadata["window_end_seconds"]),
        "selected_arm": int(selected_arm),
        "selected_fidelity": selected_fidelity,
        "external_reward": float(reward),
        "algorithm": mab_statistics.get("algorithm"),
        "epsilon": float(mab_statistics.get("epsilon", 0.0) or 0.0),
        "total_pulls": int(mab_statistics.get("total_pulls", 0)),
    }
    for arm_key, arm_stats in sorted(arms.items(), key=lambda item: int(item[0])):
        arm_index = int(arm_key)
        row[f"arm_{arm_index}_fidelity"] = arm_stats.get("fidelity")
        row[f"arm_{arm_index}_count"] = int(arm_stats.get("count", 0))
        row[f"arm_{arm_index}_mean_reward"] = float(arm_stats.get("mean_reward", 0.0) or 0.0)
        row[f"arm_{arm_index}_total_reward"] = float(arm_stats.get("total_reward", 0.0) or 0.0)
    append_csv_row(path, row)


def optional_float_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "nan"}:
            return ""
        return stripped
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(numeric):
        return ""
    return numeric


def write_morabito_plot_data_csv(output_dir, feedback_rows):
    path = os.path.join(output_dir, "morabito_plot_data.csv")
    rows = []
    for row in feedback_rows:
        rows.append(
            {
                "window_index": int(row["window_index"]),
                "window_start_seconds": optional_float_csv_value(row.get("window_start_seconds")),
                "window_end_seconds": optional_float_csv_value(row.get("window_end_seconds")),
                "reward": optional_float_csv_value(row.get("reward")),
                "internal_policy_evaluation_reward": optional_float_csv_value(row.get("internal_policy_evaluation_reward")),
                "internal_best_episode_reward": optional_float_csv_value(row.get("internal_best_episode_reward")),
                "aggregate_throughput_mbps": optional_float_csv_value(row.get("aggregate_throughput_mbps")),
                "normalized_throughput": optional_float_csv_value(row.get("normalized_throughput")),
                "active_gnb_count": optional_float_csv_value(row.get("active_gnb_count")),
                "connected_ues": optional_float_csv_value(row.get("connected_ues")),
                "disconnected_ues": optional_float_csv_value(row.get("disconnected_ues")),
                "throughput_proxy_qosflow": optional_float_csv_value(row.get("sum_qosflow_pdcp_pdu_volume_dl_filter")),
                "tb_totnbrdl_1": optional_float_csv_value(row.get("sum_tb_totnbrdl_1")),
                "radio_link_failure": optional_float_csv_value(row.get("sum_rlf_value")),
                "activation_cost": optional_float_csv_value(row.get("sum_es_on_cost")),
                "zero_count": optional_float_csv_value(row.get("zero_count")),
                "done": optional_float_csv_value(row.get("done")),
                "terminated": optional_float_csv_value(row.get("terminated")),
                "truncated": optional_float_csv_value(row.get("truncated")),
                "crashed": optional_float_csv_value(row.get("crashed")),
            }
        )
    write_csv(path, rows)
    return path


def _plot_empirical_cdf(plt, values, *, path, title, xlabel):
    filtered = sorted(float(value) for value in values)
    if not filtered:
        return None
    n = len(filtered)
    y = [(index + 1) / n for index in range(n)]
    plt.figure(figsize=(8, 4))
    plt.step(filtered, y, where="post", linewidth=1.4)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_morabito_feedback(output_dir, feedback_rows, plot_data_csv_path=None):
    if not feedback_rows:
        return []
    try:
        mpl_config_dir = os.path.join(output_dir, ".matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        log_event(ORCH_LOGGER, "morabito_plots_skipped", reason="matplotlib_import_error", error=exc)
        if plot_data_csv_path and invoke_plot_helper(
            "plot_figures_jsac.py",
            ["--csv", plot_data_csv_path, "--output-dir", output_dir],
            logger=ORCH_LOGGER,
            event_prefix="morabito_plots_helper",
        ):
            return [
                os.path.join(output_dir, name)
                for name in [
                    "morabito_reward.png",
                    "morabito_aggregate_throughput_mbps.png",
                    "morabito_normalized_throughput.png",
                    "morabito_connected_ues.png",
                    "morabito_disconnected_ues.png",
                    "morabito_qosflow.png",
                    "morabito_tb_totnbrdl.png",
                    "morabito_rlf.png",
                    "morabito_es_on_cost.png",
                    "morabito_zero_count.png",
                    "morabito_crashed.png",
                    "morabito_done.png",
                    "morabito_reward_components_stacked.png",
                    "reward_internal_vs_morabito.png",
                    "morabito_throughput_cdf.png",
                    "morabito_rlf_cdf.png",
                    "morabito_activation_cost_cdf.png",
                ]
                if os.path.exists(os.path.join(output_dir, name))
            ]
        return []

    x = [int(row["window_index"]) for row in feedback_rows]

    def _series(column):
        values = []
        for row in feedback_rows:
            value = optional_float_csv_value(row.get(column))
            values.append(float("nan") if value == "" else float(value))
        return values

    plot_specs = [
        ("morabito_reward.png", "reward", "Morabito/ns-3 Reward"),
        ("morabito_aggregate_throughput_mbps.png", "aggregate_throughput_mbps", "Morabito compatible aggregate throughput (Mbps)"),
        ("morabito_normalized_throughput.png", "normalized_throughput", "Morabito compatible normalized throughput"),
        ("morabito_connected_ues.png", "connected_ues", "Morabito compatible connected UEs"),
        ("morabito_disconnected_ues.png", "disconnected_ues", "Morabito compatible disconnected UEs"),
        ("morabito_qosflow.png", "sum_qosflow_pdcp_pdu_volume_dl_filter", "ns-3 SUM_QosFlow.PdcpPduVolumeDL_Filter"),
        ("morabito_tb_totnbrdl.png", "sum_tb_totnbrdl_1", "ns-3 SUM_TB.TotNbrDl.1"),
        ("morabito_rlf.png", "sum_rlf_value", "ns-3 SUM_RLF_VALUE"),
        ("morabito_es_on_cost.png", "sum_es_on_cost", "ns-3 SUM_ES_ON_COST"),
        ("morabito_zero_count.png", "zero_count", "ns-3 ZERO_COUNT"),
        ("morabito_crashed.png", "crashed", "ns-3 crashed flag"),
        ("morabito_done.png", "done", "ns-3 done flag"),
    ]
    paths = []
    for filename, column, title in plot_specs:
        y = _series(column)
        path = os.path.join(output_dir, filename)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, marker="o", linewidth=1)
        plt.xlabel("Window Index")
        plt.ylabel(column)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)

    stacked_path = os.path.join(output_dir, "morabito_reward_components_stacked.png")
    qos = _series("sum_qosflow_pdcp_pdu_volume_dl_filter")
    tb = _series("sum_tb_totnbrdl_1")
    rlf = _series("sum_rlf_value")
    on_cost = _series("sum_es_on_cost")
    zero_count = _series("zero_count")
    plt.figure(figsize=(10, 5))
    plt.stackplot(
        x,
        qos,
        tb,
        rlf,
        on_cost,
        zero_count,
        labels=[
            "SUM_QosFlow.PdcpPduVolumeDL_Filter",
            "SUM_TB.TotNbrDl.1",
            "SUM_RLF_VALUE",
            "SUM_ES_ON_COST",
            "ZERO_COUNT",
        ],
        alpha=0.7,
    )
    plt.xlabel("Window Index")
    plt.ylabel("Component value")
    plt.title("Morabito/ns-3 Reward Components")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(stacked_path)
    plt.close()
    paths.append(stacked_path)

    comparison_path = os.path.join(output_dir, "reward_internal_vs_morabito.png")
    internal_reward = _series("internal_policy_evaluation_reward")
    external_reward = _series("reward")
    plt.figure(figsize=(9, 4))
    plt.plot(x, internal_reward, marker="o", linewidth=1.2, label="RAN FUSION internal policy reward")
    plt.plot(x, external_reward, marker="s", linewidth=1.2, label="Morabito/ns-3 external reward")
    plt.xlabel("Window Index")
    plt.ylabel("Reward")
    plt.title("Internal vs External Reward by Window")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    paths.append(comparison_path)

    cdf_specs = [
        (
            "morabito_throughput_cdf.png",
            _series("aggregate_throughput_mbps"),
            "Morabito compatible aggregate throughput CDF",
            "Aggregate throughput (Mbps)",
        ),
        (
            "morabito_rlf_cdf.png",
            rlf,
            "Morabito/ns-3 Radio Link Failure CDF",
            "SUM_RLF_VALUE",
        ),
        (
            "morabito_activation_cost_cdf.png",
            on_cost,
            "Morabito/ns-3 Activation Cost CDF",
            "SUM_ES_ON_COST",
        ),
    ]
    for filename, values, title, xlabel in cdf_specs:
        cdf_path = _plot_empirical_cdf(
            plt,
            values,
            path=os.path.join(output_dir, filename),
            title=title,
            xlabel=xlabel,
        )
        if cdf_path:
            paths.append(cdf_path)
    return paths


def run_training_window(
    args,
    *,
    window_provider,
    window_metadata,
    window_output_dir,
    initial_epsilon,
    initial_checkpoint=None,
    mab_controller=None,
    mab_arm=None,
    reward_listener=None,
    external_feedback_dir=None,
    wandb_state_path=None,
):
    os.makedirs(window_output_dir, exist_ok=True)
    (
        training_rows,
        final_checkpoint,
        best_checkpoint,
        env,
        runtime_result,
        best_result,
        final_iteration,
        last_observation,
        last_info,
        final_epsilon,
    ) = train_q_learning(
        args,
        mobility_provider=window_provider,
        output_dir=window_output_dir,
        initial_epsilon=initial_epsilon,
        window_metadata=window_metadata,
        initial_checkpoint=initial_checkpoint,
    )

    training_csv = os.path.join(window_output_dir, "training_metrics.csv")
    write_csv(training_csv, training_rows)

    best_model = os.path.join(window_output_dir, "best_q_learning_model.pkl")
    if best_checkpoint is not None:
        save_checkpoint(best_model, best_checkpoint)

    baseline_csv = os.path.join(window_output_dir, "baseline_metrics.csv")
    baseline_rows = []
    if args.run_baselines and not runtime_result["budget_expired"]:
        baseline_rows.append(evaluate_policy(args, env, "all_on", all_on_policy))
        baseline_rows.append(evaluate_policy(args, env, "heuristic_energy", heuristic_energy_policy))
        write_csv(baseline_csv, baseline_rows)

    export_checkpoint = best_checkpoint if args.socket_export_policy == "best" else final_checkpoint
    export_policy_source_label = (
        "best_checkpoint_initial_state_greedy"
        if args.socket_export_policy == "best"
        else "final_checkpoint_initial_state_greedy"
    )
    final_payload = build_final_socket_payload(
        runtime_result=runtime_result,
        best_result=build_policy_result_from_checkpoint(
            env,
            args,
            export_checkpoint,
            policy_source_label=export_policy_source_label,
        ),
        final_iteration=final_iteration,
        agent_id=f"q_learning_agent_window_{window_metadata['window_index']}",
        window_metadata=window_metadata,
        socket_export_policy=args.socket_export_policy,
        policy_warm_started=bool(final_checkpoint.get("policy_warm_started")) if final_checkpoint else False,
    )
    final_topology_snapshot = build_final_topology_snapshot(env, last_observation, last_info)
    final_topology_snapshot = apply_socket_configuration_to_topology_snapshot(final_topology_snapshot, final_payload)
    final_topology_plot = {
        "enabled": bool(args.enable_final_topology_plot),
        "path": None,
        "error": None,
        "ue_count": len(final_topology_snapshot.get("ues", [])),
        "gnb_count": len(final_topology_snapshot.get("gnbs", [])),
        "thresholds": final_topology_snapshot.get("distance_thresholds", {}),
        "timestamp": final_topology_snapshot.get("timestamp"),
        "socket_configuration": final_topology_snapshot.get("socket_configuration", {}),
    }
    topology_snapshot_path = os.path.join(window_output_dir, "final_topology_snapshot.json")
    write_json(topology_snapshot_path, final_topology_snapshot)
    plot_config = dict(env.distance_throughput_config)
    plot_config.update(
        {
            "show_ue_labels": bool(args.show_ue_labels),
            "show_gnb_labels": bool(args.show_gnb_labels),
            "draw-serving_lines": bool(args.draw_serving_lines),
            "draw_serving_lines": bool(args.draw_serving_lines),
            "plot_image_width": float(args.plot_image_width),
            "plot_image_height": float(args.plot_image_height),
            "dpi": int(args.dpi),
        }
    )
    topology_config_path = os.path.join(window_output_dir, "final_topology_plot_config.json")
    write_json(topology_config_path, plot_config)

    plot_paths = plot_training(window_output_dir, training_rows, training_csv) if training_rows else []
    if plot_paths:
        log_event(
            RL_LOGGER,
            "plots_generated",
            window_index=window_metadata["window_index"],
            plot_count=len(plot_paths),
            output_dir=window_output_dir,
        )
    if args.enable_final_topology_plot:
        try:
            plot_output_dir = args.plot_output_path or window_output_dir
            os.makedirs(plot_output_dir, exist_ok=True)
            final_topology_plot["path"] = generate_final_topology_plot(
                final_topology_snapshot,
                output_dir=plot_output_dir,
                output_filename=args.plot_output_filename,
                config=plot_config,
            )
            log_event(
                RL_LOGGER,
                "final_topology_plot_generated",
                window_index=window_metadata["window_index"],
                path=final_topology_plot["path"],
            )
        except Exception as exc:
            final_topology_plot["error"] = str(exc)
            helper_plot_output_dir = args.plot_output_path or window_output_dir
            helper_args = [
                "--output-dir",
                helper_plot_output_dir,
                "--topology-json",
                topology_snapshot_path,
                "--topology-config-json",
                topology_config_path,
                "--topology-output-filename",
                args.plot_output_filename,
            ]
            if invoke_plot_helper(
                "generate_ranfusion_internal_plots.py",
                helper_args,
                logger=RL_LOGGER,
                event_prefix="final_topology_plot_helper",
            ):
                helper_plot_path = os.path.join(helper_plot_output_dir, args.plot_output_filename)
                if os.path.exists(helper_plot_path):
                    final_topology_plot["path"] = helper_plot_path
                    final_topology_plot["error"] = None
                    log_event(
                        RL_LOGGER,
                        "final_topology_plot_generated",
                        window_index=window_metadata["window_index"],
                        path=helper_plot_path,
                        source="helper",
                    )
                else:
                    log_event(
                        RL_LOGGER,
                        "final_topology_plot_failed",
                        window_index=window_metadata["window_index"],
                        error=exc,
                    )
            else:
                log_event(
                    RL_LOGGER,
                    "final_topology_plot_failed",
                    window_index=window_metadata["window_index"],
                    error=exc,
                )

    final_payload["window_index"] = int(window_metadata["window_index"])
    final_payload["request_id"] = f"window-{int(window_metadata['window_index']):04d}"
    payload_path = os.path.join(window_output_dir, "socket_export_payload.json")
    write_json(payload_path, final_payload)

    experiment_root_dir = os.path.dirname(window_output_dir)
    gnb_timeline_csv = os.path.join(experiment_root_dir, "gnb_activation_timeline.csv")
    gnb_timeline_row = build_gnb_activation_row(window_metadata, final_payload)
    append_csv_row(gnb_timeline_csv, gnb_timeline_row)
    invoke_plot_helper(
        "plot_control_timelines.py",
        ["--gnb-csv", gnb_timeline_csv, "--output-dir", experiment_root_dir],
        logger=ORCH_LOGGER,
        event_prefix="gnb_timeline_plots",
    )

    sync_window_to_wandb(
        args,
        wandb_state_path=wandb_state_path,
        window_metadata=window_metadata,
        training_csv=training_csv,
        internal_plot_paths=plot_paths,
        final_topology_plot_path=final_topology_plot.get("path"),
        gnb_timeline_csv=gnb_timeline_csv,
        gnb_timeline_plot_dir=experiment_root_dir,
    )

    plot_output_dir = args.plot_output_path or window_output_dir

    log_event(
        ORCH_LOGGER,
        "window_artifacts_ready",
        window_index=window_metadata["window_index"],
        interval=f"[{window_metadata['window_start_seconds']},{window_metadata['window_end_seconds']})",
        training_csv=training_csv,
        baseline_csv=baseline_csv if baseline_rows else None,
        best_model=best_model if best_checkpoint is not None else None,
        plots_output_dir=plot_output_dir,
        plot_count=len(plot_paths),
        socket_export_payload=payload_path,
    )
    try:
        emit_terminal_window_summary(
            window_metadata=window_metadata,
            training_csv=training_csv,
            plot_output_dir=plot_output_dir,
            final_payload=final_payload,
            plot_paths=plot_paths,
            final_topology_plot=final_topology_plot,
        )
    except Exception as exc:
        log_event(
            ORCH_LOGGER,
            "terminal_summary_failed",
            window_index=window_metadata["window_index"],
            error=exc,
        )

    socket_export = {
        "enabled": bool(args.enable_socket_export),
        "host": args.socket_host,
        "port": args.socket_port,
        "success": False,
        "error": None,
        "external_reward": None,
        "external_feedback_payload": None,
    }
    if args.enable_socket_export:
        log_event(
            ORCH_LOGGER,
            "socket_export_starting",
            window_index=window_metadata["window_index"],
            host=args.socket_host,
            port=args.socket_port,
            mab_mode=bool(mab_controller is not None),
        )
        if reward_listener is not None:
            if mab_controller is not None:
                log_event(
                    MAB_LOGGER,
                    "waiting_for_external_step_reward",
                    window_index=window_metadata["window_index"],
                    host=args.socket_host,
                    port=args.socket_port,
                    reward_port=reward_listener.port,
                    selection=describe_mab_arm(mab_arm, window_metadata.get("fidelity_level")),
                )
            feedback_payload = send_configuration_and_wait_for_external_reward(
                host=args.socket_host,
                port=args.socket_port,
                payload=final_payload,
                socket_timeout_ms=args.socket_timeout_ms,
                reward_listener=reward_listener,
            )
            reward = float(feedback_payload["reward"])
            socket_export["external_reward"] = reward
            socket_export["external_feedback_payload"] = feedback_payload
            warmup_reward_ignored = bool(
                args.ignore_first_external_reward
                and int(window_metadata["window_index"]) == 0
            )
            if mab_controller is not None:
                log_event(
                    MAB_LOGGER,
                    "external_reward_received",
                    window_index=window_metadata["window_index"],
                    reward=reward,
                    selection=describe_mab_arm(mab_arm, window_metadata.get("fidelity_level")),
                    warmup_ignored=warmup_reward_ignored,
                )
                if warmup_reward_ignored:
                    log_event(
                        MAB_LOGGER,
                        "warmup_reward_ignored",
                        window_index=window_metadata["window_index"],
                        reward=reward,
                        selection=describe_mab_arm(mab_arm, window_metadata.get("fidelity_level")),
                    )
                else:
                    mab_controller.update(mab_arm, reward)
                mab_statistics = mab_controller.get_statistics()
                arm_stats = mab_statistics["arms"][str(mab_arm)]
                if not warmup_reward_ignored:
                    log_event(
                        MAB_LOGGER,
                        "arm_updated",
                        selection=describe_mab_arm(mab_arm, window_metadata.get("fidelity_level")),
                        count=arm_stats["count"],
                        mean_reward=arm_stats["mean_reward"],
                        total_reward=arm_stats["total_reward"],
                    )
                log_event(MAB_LOGGER, "stats_snapshot", stats=mab_statistics)
                mab_history_csv = os.path.join(external_feedback_dir or experiment_root_dir, "mab_history.csv")
                append_mab_history_row(
                    mab_history_csv,
                    window_metadata=window_metadata,
                    selected_arm=mab_arm,
                    selected_fidelity=window_metadata.get("fidelity_level"),
                    reward=reward,
                    mab_statistics=mab_statistics,
                )
                invoke_plot_helper(
                    "plot_control_timelines.py",
                    ["--mab-csv", mab_history_csv, "--output-dir", external_feedback_dir or experiment_root_dir],
                    logger=MAB_LOGGER,
                    event_prefix="mab_timeline_plots",
                )
            else:
                log_event(
                    ORCH_LOGGER,
                    "external_reward_received",
                    window_index=window_metadata["window_index"],
                    reward=reward,
                    warmup_ignored=warmup_reward_ignored,
                )
            if external_feedback_dir:
                os.makedirs(external_feedback_dir, exist_ok=True)
                jsonl_path = os.path.join(external_feedback_dir, "morabito_feedback.jsonl")
                csv_path = os.path.join(external_feedback_dir, "morabito_feedback.csv")
                append_jsonl(
                    jsonl_path,
                    {
                        "window": window_metadata,
                        "feedback": feedback_payload,
                    },
                )
                reward_components = feedback_payload.get("reward_components") or {}
                compatibility_metrics = (feedback_payload.get("compatibility_metrics") or {})
                reward_mode = compatibility_metrics.get("reward_mode")
                reward_alpha = optional_float_csv_value(compatibility_metrics.get("reward_alpha"))
                reward_beta = optional_float_csv_value(compatibility_metrics.get("reward_beta"))
                reward_gamma = optional_float_csv_value(compatibility_metrics.get("reward_gamma"))
                aggregate_throughput_mbps = optional_float_csv_value(compatibility_metrics.get("aggregate_throughput_mbps"))
                normalized_throughput = optional_float_csv_value(compatibility_metrics.get("normalized_throughput"))
                active_gnb_count = optional_float_csv_value(compatibility_metrics.get("active_gnb_count"))
                normalized_active_gnb_count = optional_float_csv_value(compatibility_metrics.get("normalized_active_gnb_count"))
                connected_ues = optional_float_csv_value(compatibility_metrics.get("connected_ue_count"))
                disconnected_ues = optional_float_csv_value(compatibility_metrics.get("disconnected_ue_count"))
                normalized_disconnected_ues = optional_float_csv_value(compatibility_metrics.get("normalized_disconnected_ues"))
                weighted_components = compute_weighted_external_reward_components(
                    reward_mode=reward_mode,
                    reward_alpha=reward_alpha,
                    reward_beta=reward_beta,
                    reward_gamma=reward_gamma,
                    aggregate_throughput_mbps=aggregate_throughput_mbps,
                    normalized_throughput=normalized_throughput,
                    active_gnb_count=active_gnb_count,
                    normalized_active_gnb_count=normalized_active_gnb_count,
                    disconnected_ues=disconnected_ues,
                    normalized_disconnected_ues=normalized_disconnected_ues,
                )
                row = {
                        "window_index": window_metadata["window_index"],
                        "request_id": feedback_payload.get("request_id", final_payload.get("request_id")),
                        "window_start_seconds": window_metadata["window_start_seconds"],
                        "window_end_seconds": window_metadata["window_end_seconds"],
                        "selected_arm": mab_arm if mab_controller is not None else None,
                        "fidelity_level": window_metadata.get("fidelity_level"),
                        "warmup_reward_ignored": int(warmup_reward_ignored),
                    "sim_timestamp": feedback_payload.get("sim_timestamp"),
                    "reward": reward,
                    "internal_policy_evaluation_reward": optional_float_csv_value(final_payload.get("policy_evaluation_reward")),
                    "internal_best_episode_reward": optional_float_csv_value(final_payload.get("best_reward")),
                    "done": int(bool(feedback_payload.get("done"))),
                    "terminated": int(bool(feedback_payload.get("terminated"))),
                    "truncated": int(bool(feedback_payload.get("truncated"))),
                    "crashed": int(bool(feedback_payload.get("crashed"))),
                    "sum_qosflow_pdcp_pdu_volume_dl_filter": optional_float_csv_value(reward_components.get("sum_qosflow_pdcp_pdu_volume_dl_filter")),
                    "throughput_source": str(reward_components.get("throughput_source", "SUM_QosFlow.PdcpPduVolumeDL_Filter")),
                    "sum_tb_totnbrdl_1": optional_float_csv_value(reward_components.get("sum_tb_totnbrdl_1")),
                    "load_source": str(reward_components.get("load_source", "SUM_TB.TotNbrDl.1")),
                    "sum_rlf_value": optional_float_csv_value(reward_components.get("sum_rlf_value")),
                    "sum_es_on_cost": optional_float_csv_value(reward_components.get("sum_es_on_cost")),
                    "zero_count": optional_float_csv_value(reward_components.get("zero_count")),
                    "aggregate_throughput_mbps": aggregate_throughput_mbps,
                    "normalized_throughput": normalized_throughput,
                    "active_gnb_count": active_gnb_count,
                    "connected_ues": connected_ues,
                    "disconnected_ues": disconnected_ues,
                    "reward_mode": reward_mode,
                    "reward_alpha": reward_alpha,
                    "reward_beta": reward_beta,
                    "reward_gamma": reward_gamma,
                    "throughput_reference_mbps": optional_float_csv_value(compatibility_metrics.get("throughput_reference_mbps")),
                    "normalized_active_gnb_count": normalized_active_gnb_count,
                    "normalized_disconnected_ues": normalized_disconnected_ues,
                    "reward_component_throughput_weighted": weighted_components["reward_component_throughput_weighted"],
                    "reward_component_active_gnb_weighted": weighted_components["reward_component_active_gnb_weighted"],
                    "reward_component_disconnected_ues_weighted": weighted_components["reward_component_disconnected_ues_weighted"],
                }
                append_morabito_feedback_row(csv_path, row)
                with open(csv_path, newline="", encoding="utf-8") as handle:
                    feedback_rows = list(csv.DictReader(handle))
                plot_data_csv_path = write_morabito_plot_data_csv(external_feedback_dir, feedback_rows)
                plot_morabito_feedback(external_feedback_dir, feedback_rows, plot_data_csv_path=plot_data_csv_path)
                sync_window_to_wandb(
                    args,
                    wandb_state_path=wandb_state_path,
                    window_metadata=window_metadata,
                    final_topology_plot_path=final_topology_plot.get("path"),
                    morabito_feedback_csv=csv_path,
                    morabito_plot_data_csv=plot_data_csv_path,
                    morabito_plot_dir=external_feedback_dir,
                    mab_history_csv=mab_history_csv if mab_controller is not None else None,
                    mab_plot_dir=external_feedback_dir or experiment_root_dir,
                )
                log_event(
                    ORCH_LOGGER,
                    "morabito_plot_data_ready",
                    output_dir=external_feedback_dir,
                    feedback_csv=csv_path,
                    plot_data_csv=plot_data_csv_path,
                    row_count=len(feedback_rows),
                )
        else:
            send_best_configuration_over_socket_until_success(
                host=args.socket_host,
                port=args.socket_port,
                payload=final_payload,
                socket_timeout_ms=args.socket_timeout_ms,
            )
        socket_export["success"] = True

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "window": window_metadata,
        "runtime": runtime_result,
        "policy_warm_started": bool(final_checkpoint.get("policy_warm_started")) if final_checkpoint else False,
        "socket_export_policy": args.socket_export_policy,
        "training_csv": training_csv,
        "baseline_csv": baseline_csv if baseline_rows else None,
        "best_model": best_model if best_checkpoint is not None else None,
        "plots": plot_paths,
        "best_result": best_result,
        "final_iteration": final_iteration,
        "evaluation_log_csv": None,
        "final_topology_snapshot": final_topology_snapshot,
        "final_topology_plot": final_topology_plot,
        "socket_export_payload": payload_path,
        "socket_export": socket_export,
        "final_epsilon": final_epsilon,
    }
    manifest_path = os.path.join(window_output_dir, "run_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    budget_result_path = os.path.join(window_output_dir, "real_time_budget_result.json")
    write_json(
        budget_result_path,
        {
            "window": window_metadata,
            "runtime": runtime_result,
            "best_result": best_result,
            "final_topology_snapshot": final_topology_snapshot,
            "final_topology_plot": final_topology_plot,
            "final_payload": final_payload,
            "socket_export": socket_export,
        },
    )

    log_event(
        ORCH_LOGGER,
        "window_completed",
        window_index=window_metadata["window_index"],
        budget_result=budget_result_path,
        manifest=manifest_path,
        policy_warm_started=bool(final_checkpoint.get("policy_warm_started")) if final_checkpoint else False,
        socket_export_policy=args.socket_export_policy,
        socket_export_success=socket_export["success"],
        external_reward=socket_export["external_reward"],
    )
    return {
        "window_metadata": window_metadata,
        "runtime_result": runtime_result,
        "best_result": best_result,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "final_iteration": final_iteration,
        "socket_export": socket_export,
        "manifest_path": manifest_path,
        "budget_result_path": budget_result_path,
        "payload_path": payload_path,
        "training_csv": training_csv,
        "plot_paths": plot_paths,
        "final_epsilon": final_epsilon,
        "policy_warm_started": bool(final_checkpoint.get("policy_warm_started")) if final_checkpoint else False,
        "window_output_dir": window_output_dir,
        "timestamp_count": len(window_provider.timestamps),
        "external_reward": socket_export["external_reward"],
    }


def run_single_experiment(
    args,
    *,
    output_dir=None,
    trial_metadata=None,
    mab_controller=None,
    reward_listener=None,
    fidelity_selector=None,
    external_feedback_dir=None,
    wandb_state_path=None,
):
    effective_output_dir = output_dir or args.output_dir
    os.makedirs(effective_output_dir, exist_ok=True)
    timeline_provider = build_fidelity_provider(
        fidelity_level=FidelityLevel.HIGH.value,
        mobility_profile=args.mobility_profile,
        base_dir=REPO_ROOT,
        mobility_trace_csv=args.mobility_csv,
        medium_snapshot_sec=args.medium_snapshot_sec,
    )
    window_starts, total_duration_seconds, last_valid_start = build_window_start_times(
        timeline_provider,
        args.window_size_seconds,
        args.step_seconds,
    )
    if not window_starts:
        raise ValueError(
            "The mobility trace is shorter than the requested sliding window. "
            f"total_duration_seconds={total_duration_seconds}, "
            f"window_size_seconds={args.window_size_seconds}"
        )

    log_event(
        ORCH_LOGGER,
        "sliding_window_plan",
        fidelity_level=args.fidelity_level if fidelity_selector is None else "mab_dynamic",
        fidelity_source=timeline_provider.source_path,
        total_duration_seconds=total_duration_seconds,
        window_size_seconds=args.window_size_seconds,
        step_seconds=args.step_seconds,
        last_valid_start_seconds=last_valid_start,
        window_count=len(window_starts),
    )

    initial_epsilon = (
        float(args.exploration_reset_value)
        if args.reset_exploration_each_window
        else float(args.epsilon_start)
    )
    # Per-fidelity state: Q-tables and epsilon are kept separate for each fidelity
    # level so that switching arms does not corrupt the other level's learned values.
    carried_checkpoint_per_fidelity: dict = {}
    carried_epsilon_per_fidelity: dict = {}

    window_results = []
    provider_cache = {}
    for window_index, window_start_seconds in enumerate(window_starts):
        window_end_seconds = window_start_seconds + float(args.window_size_seconds)
        if fidelity_selector is None:
            selected_arm = None
            selected_fidelity = FidelityLevel.normalize(args.fidelity_level)
        else:
            selected_arm, selected_fidelity = fidelity_selector(
                window_index=window_index,
                window_start_seconds=window_start_seconds,
                window_end_seconds=window_end_seconds,
            )
            selected_fidelity = FidelityLevel.normalize(selected_fidelity)
        provider_cache_key = selected_fidelity.value
        if provider_cache_key not in provider_cache:
            provider_cache[provider_cache_key] = build_fidelity_provider(
                fidelity_level=selected_fidelity.value,
                mobility_profile=args.mobility_profile,
                base_dir=REPO_ROOT,
                mobility_trace_csv=args.mobility_csv,
                medium_snapshot_sec=args.medium_snapshot_sec,
            )
        full_window_provider = provider_cache[provider_cache_key]

        # Look up per-fidelity checkpoint and epsilon.  Each fidelity level has its
        # own Q-table lineage so that switching arms does not corrupt learned values.
        fidelity_key = selected_fidelity.value
        carried_checkpoint = carried_checkpoint_per_fidelity.get(fidelity_key)
        carried_epsilon = carried_epsilon_per_fidelity.get(fidelity_key, initial_epsilon)
        if args.reset_exploration_each_window:
            # Reset exploration only when this fidelity has no warm-started checkpoint.
            if not (args.carry_policy_across_windows and carried_checkpoint is not None):
                carried_epsilon = float(args.exploration_reset_value)

        if fidelity_selector is None:
            log_event(
                ORCH_LOGGER,
                "window_started",
                window_index=window_index,
                interval=f"[{window_start_seconds},{window_end_seconds})",
                fidelity_level=selected_fidelity.value,
                source=full_window_provider.source_path,
            )
        else:
            log_event(
                MAB_LOGGER,
                "arm_selected",
                window_index=window_index,
                interval=f"[{window_start_seconds},{window_end_seconds})",
                selected_arm=selected_arm,
                fidelity_level=selected_fidelity.value,
                source=full_window_provider.source_path,
            )
        window_provider = full_window_provider.windowed(window_start_seconds, args.window_size_seconds)
        window_metadata = {
            "window_index": window_index,
            "window_start_seconds": window_start_seconds,
            "window_end_seconds": window_end_seconds,
            "window_size_seconds": float(args.window_size_seconds),
            "step_seconds": float(args.step_seconds),
            "last_valid_start_seconds": last_valid_start,
            "total_duration_seconds": total_duration_seconds,
            "trace_timestamp_count": len(window_provider.timestamps),
            "fidelity_level": selected_fidelity.value,
            "real_time_budget_sec": effective_real_time_budget_sec(args, selected_fidelity.value),
            "selected_arm": selected_arm,
            "fidelity_source": full_window_provider.source_path,
        }
        if len(window_provider.timestamps) == 0:
            break
        window_output_dir = build_window_output_dir(
            effective_output_dir,
            window_index,
            window_start_seconds,
            window_end_seconds,
        )
        result = run_training_window(
            args,
            window_provider=window_provider,
            window_metadata=window_metadata,
            window_output_dir=window_output_dir,
            initial_epsilon=carried_epsilon,
            initial_checkpoint=carried_checkpoint,
            mab_controller=mab_controller,
            mab_arm=selected_arm,
            reward_listener=reward_listener,
            external_feedback_dir=external_feedback_dir,
            wandb_state_path=wandb_state_path,
        )
        # Store results back into the per-fidelity state.
        carried_epsilon_per_fidelity[fidelity_key] = result["final_epsilon"]
        if args.carry_policy_across_windows:
            carried_checkpoint_per_fidelity[fidelity_key] = result["final_checkpoint"]
        window_results.append(result)

    global_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "trial": trial_metadata,
        "mobility_csv": args.mobility_csv,
        "window_size_seconds": float(args.window_size_seconds),
        "step_seconds": float(args.step_seconds),
        "total_duration_seconds": total_duration_seconds,
        "last_valid_start_seconds": last_valid_start,
        "window_count": len(window_results),
        "windows": [
            {
                "window": result["window_metadata"],
                "runtime": result["runtime_result"],
                "socket_export": result["socket_export"],
                "external_reward": result["external_reward"],
                "selected_arm": result["window_metadata"].get("selected_arm"),
                "selected_fidelity": result["window_metadata"].get("fidelity_level"),
                "policy_warm_started": result.get("policy_warm_started", False),
                "manifest_path": result["manifest_path"],
                "budget_result_path": result["budget_result_path"],
                "payload_path": result["payload_path"],
                "training_csv": result["training_csv"],
                "window_output_dir": result["window_output_dir"],
            }
            for result in window_results
        ],
    }
    global_manifest_path = os.path.join(effective_output_dir, "sliding_window_manifest.json")
    write_json(global_manifest_path, global_manifest)
    log_event(
        ORCH_LOGGER,
        "sliding_window_training_completed",
        manifest=global_manifest_path,
        window_count=len(window_results),
        output_dir=effective_output_dir,
    )
    return {
        "trial": trial_metadata,
        "fidelity_level": args.fidelity_level if fidelity_selector is None else "mab_dynamic",
        "fidelity_source": timeline_provider.source_path,
        "total_duration_seconds": total_duration_seconds,
        "last_valid_start_seconds": last_valid_start,
        "window_count": len(window_results),
        "window_results": window_results,
        "sliding_window_manifest": global_manifest_path,
        "output_dir": effective_output_dir,
    }


def main():
    args = parse_args()
    if args.wandb_mode == "disabled":
        args.enable_wandb = False
    configure_logging()
    if args.stub_external:
        install_external_stubs()
    if args.quiet_simulator_logs and not args.verbose_simulator_logs:
        os.environ.setdefault("RANFUSION_SUPPRESS_EMPTY_QUERY_LOGS", "1")
        os.environ.setdefault("RANFUSION_SUPPRESS_TRAFFIC_STOP_LOGS", "1")
        quiet_simulator_loggers()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    output_layout = build_experiment_output_dirs(args.output_dir)
    configure_experiment_logging(output_layout)
    wandb_state_path = None
    if args.enable_wandb:
        wandb_state_path, _wandb_state = build_wandb_state(output_layout, args)
    log_event(
        ORCH_LOGGER,
        "experiment_started",
        experiment_root=output_layout["experiment_root"],
        ran_fusion_internal=output_layout["ran_fusion_internal"],
        morabito_ns3=output_layout["morabito_ns3"],
        logs=output_layout["logs"],
        mobility_profile=args.mobility_profile,
        socket_export=bool(args.enable_socket_export),
        fidelity_mab=bool(args.enable_fidelity_mab),
        wandb_enabled=bool(args.enable_wandb),
        wandb_entity=args.wandb_entity if args.enable_wandb else None,
        wandb_project=args.wandb_project if args.enable_wandb else None,
    )

    if not args.enable_fidelity_mab:
        log_event(MAB_LOGGER, "controller_disabled", reason="enable_fidelity_mab_false")
        reward_listener = (
            RewardSocketListener(
                port=args.reward_port,
                timeout_sec=args.reward_timeout_sec,
            )
            if args.enable_socket_export
            else None
        )
        run_single_experiment(
            args,
            output_dir=output_layout["ran_fusion_internal"],
            external_feedback_dir=output_layout["morabito_ns3"],
            wandb_state_path=wandb_state_path,
            reward_listener=reward_listener,
        )
        log_event(
            ORCH_LOGGER,
            "experiment_completed",
            mode="manual_fidelity",
            experiment_root=output_layout["experiment_root"],
        )
        return

    if not args.enable_socket_export:
        raise ValueError("MAB mode requires --enable-socket-export because rewards arrive from energy_saving feedback.")

    log_event(
        MAB_LOGGER,
        "controller_enabled",
        algorithm=args.mab_algorithm,
        min_initial_pulls_per_arm=args.mab_min_initial_pulls_per_arm,
        reward_port=args.reward_port,
        reward_timeout_sec=args.reward_timeout_sec,
        arms={0: "high", 1: "medium", 2: "low"},
    )
    if args.mab_num_trials != 10:
        log_event(MAB_LOGGER, "deprecated_argument_ignored", mab_num_trials=args.mab_num_trials)
    mab = FidelityMabController(
        algorithm=args.mab_algorithm,
        epsilon=args.mab_epsilon,
        seed=args.seed,
        min_initial_pulls_per_arm=args.mab_min_initial_pulls_per_arm,
    )
    reward_listener = RewardSocketListener(
        port=args.reward_port,
        timeout_sec=args.reward_timeout_sec,
    )
    selection_history = []

    def select_fidelity_for_window(*, window_index, window_start_seconds, window_end_seconds):
        arm = mab.select_arm()
        fidelity = mab.arm_to_fidelity(arm)
        selection_history.append(
            {
                "window_index": window_index,
                "window_start_seconds": window_start_seconds,
                "window_end_seconds": window_end_seconds,
                "selected_arm": arm,
                "selected_fidelity": fidelity.value,
            }
        )
        return arm, fidelity

    run_result = run_single_experiment(
        args,
        output_dir=output_layout["ran_fusion_internal"],
        trial_metadata={
            "mode": "fidelity_mab_per_window",
            "mab_algorithm": args.mab_algorithm,
        },
        mab_controller=mab,
        reward_listener=reward_listener,
        fidelity_selector=select_fidelity_for_window,
        external_feedback_dir=output_layout["morabito_ns3"],
        wandb_state_path=wandb_state_path,
    )

    fidelity_mab_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "mab_statistics": mab.get_statistics(),
        "selection_history": selection_history,
        "output_layout": output_layout,
        "run": {
            "output_dir": run_result["output_dir"],
            "sliding_window_manifest": run_result["sliding_window_manifest"],
            "window_count": run_result["window_count"],
            "fidelity_source": run_result["fidelity_source"],
        },
    }
    mab_manifest_path = os.path.join(output_layout["experiment_root"], "fidelity_mab_manifest.json")
    write_json(mab_manifest_path, fidelity_mab_manifest)
    log_event(
        ORCH_LOGGER,
        "experiment_completed",
        mode="fidelity_mab_per_window",
        experiment_root=output_layout["experiment_root"],
        mab_manifest=mab_manifest_path,
        ran_fusion_plots_dir=output_layout["ran_fusion_internal"],
        morabito_ns3_plots_dir=output_layout["morabito_ns3"],
    )


if __name__ == "__main__":
    main()
