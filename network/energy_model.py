import csv
import os
from typing import Dict


class LoadDependentEnergyModel:
    """Simple load-dependent BS/gNB power and cumulative energy model."""

    DEFAULT_CONFIG = {
        "energy_model_enabled": True,
        "idle_power_w": 120.0,
        "max_power_w": 260.0,
        "load_metric_type": "throughput_ratio",
        "load_smoothing_enabled": True,
        "load_smoothing_factor": 0.2,
        "logging_interval": 60.0,
    }

    def __init__(self, gnodeb_id, output_dir):
        self.gnodeb_id = gnodeb_id
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, f"gnb_energy_{gnodeb_id}.csv")
        self.initialize({})

    def initialize(self, config):
        merged = dict(self.DEFAULT_CONFIG)
        merged.update(config or {})
        self.config = merged
        self.energy_model_enabled = bool(merged["energy_model_enabled"])
        self.idle_power_w = float(merged["idle_power_w"])
        self.max_power_w = float(merged["max_power_w"])
        self.load_metric_type = str(merged["load_metric_type"])
        self.load_smoothing_enabled = bool(merged["load_smoothing_enabled"])
        self.load_smoothing_factor = float(merged["load_smoothing_factor"])
        self.logging_interval = float(merged["logging_interval"])
        self.current_power_w = 0.0
        self.last_raw_load = 0.0
        self.last_smoothed_load = 0.0
        self.total_energy_wh = 0.0
        self.last_step_energy_wh = 0.0
        self.elapsed_since_log_s = 0.0
        self._has_smoothed_value = False
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "timestamp",
                    "gnodeb_id",
                    "raw_load",
                    "smoothed_load",
                    "current_power_w",
                    "cumulative_energy_wh",
                    "cumulative_energy_kwh",
                ])

    def compute_load(self, sim_state):
        if not self.energy_model_enabled or not sim_state.get("is_active", True):
            raw_load = 0.0
        elif self.load_metric_type == "throughput_ratio":
            current_throughput_bps = float(sim_state.get("current_throughput_bps", 0.0))
            max_throughput_bps = float(sim_state.get("max_throughput_bps", 0.0))
            raw_load = current_throughput_bps / max_throughput_bps if max_throughput_bps > 0 else 0.0
        elif self.load_metric_type == "active_ue_ratio":
            active_ue_count = float(sim_state.get("active_ue_count", 0.0))
            max_active_ue_count = float(sim_state.get("max_active_ue_count", 0.0))
            raw_load = active_ue_count / max_active_ue_count if max_active_ue_count > 0 else 0.0
        elif self.load_metric_type == "provided_load":
            raw_load = float(sim_state.get("load_ratio", 0.0))
        else:
            raise ValueError(f"Unsupported load_metric_type: {self.load_metric_type}")

        raw_load = max(0.0, min(1.0, raw_load))
        if not self.load_smoothing_enabled:
            smoothed_load = raw_load
        elif not self._has_smoothed_value:
            smoothed_load = raw_load
            self._has_smoothed_value = True
        else:
            beta = max(0.0, min(1.0, self.load_smoothing_factor))
            smoothed_load = beta * raw_load + (1.0 - beta) * self.last_smoothed_load

        self.last_raw_load = raw_load
        self.last_smoothed_load = smoothed_load
        return raw_load, smoothed_load

    def compute_power(self, load):
        if not self.energy_model_enabled:
            self.current_power_w = 0.0
            return 0.0
        clipped_load = max(0.0, min(1.0, float(load)))
        self.current_power_w = self.idle_power_w + (self.max_power_w - self.idle_power_w) * clipped_load
        return self.current_power_w

    def update_energy(self, delta_t, sim_state):
        delta_t = max(0.0, float(delta_t))
        raw_load, smoothed_load = self.compute_load(sim_state)
        if not sim_state.get("is_active", True):
            self.current_power_w = 0.0
        else:
            self.compute_power(smoothed_load)

        self.last_step_energy_wh = (self.current_power_w * delta_t) / 3600.0
        self.total_energy_wh += self.last_step_energy_wh
        self.elapsed_since_log_s += delta_t
        if self.logging_interval <= 0 or self.elapsed_since_log_s >= self.logging_interval:
            self._append_log(sim_state.get("timestamp", "unknown"))
            self.elapsed_since_log_s = 0.0
        return {
            "raw_load": raw_load,
            "smoothed_load": smoothed_load,
            "current_power_w": self.current_power_w,
            "step_energy_wh": self.last_step_energy_wh,
            "total_energy_wh": self.total_energy_wh,
            "total_energy_kwh": self.get_total_energy_kwh(),
        }

    def _append_log(self, timestamp):
        with open(self.log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                timestamp,
                self.gnodeb_id,
                self.last_raw_load,
                self.last_smoothed_load,
                self.current_power_w,
                self.total_energy_wh,
                self.get_total_energy_kwh(),
            ])

    def get_current_power_w(self):
        return self.current_power_w

    def get_total_energy_wh(self):
        return self.total_energy_wh

    def get_total_energy_kwh(self):
        return self.total_energy_wh / 1000.0


def _normalize_capacity_to_bps(value):
    numeric = float(value or 0.0)
    if numeric <= 0:
        return 0.0
    if numeric <= 10000:
        return numeric * 1_000_000.0
    return numeric


def build_energy_models(base_dir, gnodeb_manager, energy_config):
    output_dir = os.path.join(base_dir, "outputs", "energy_logs")
    models = {}
    for gnodeb_id in sorted(gnodeb_manager.gNodeBs):
        model = LoadDependentEnergyModel(gnodeb_id, output_dir)
        model.initialize(energy_config)
        models[gnodeb_id] = model
    return models


def build_gnodeb_energy_state(gnodeb):
    served_ues = []
    max_throughput_bps = 0.0
    max_active_ue_count = 0.0
    for cell in getattr(gnodeb, "Cells", []):
        for sector in getattr(cell, "sectors", []):
            if not getattr(sector, "is_active", True):
                continue
            max_throughput_bps += _normalize_capacity_to_bps(getattr(sector, "max_throughput", 0.0))
            max_active_ue_count += float(getattr(sector, "capacity", 0))
            for ue in sector.ues.values():
                if getattr(ue, "in_outage", False) or getattr(ue, "is_connected", True) is False:
                    continue
                served_ues.append(ue)
    return {
        "is_active": bool(getattr(gnodeb, "is_active", True)),
        "current_throughput_bps": sum(float(getattr(ue, "throughput", 0.0)) for ue in served_ues),
        "max_throughput_bps": max_throughput_bps,
        "active_ue_count": float(len(served_ues)),
        "max_active_ue_count": max_active_ue_count,
    }
