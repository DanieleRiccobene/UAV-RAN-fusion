import os
import random
import logging
import math
from collections import Counter
from datetime import datetime

from Config_files.config import Config
from Config_files.mobility_config import DEFAULT_MOBILITY_PROFILE
from network.initialize_network import initialize_network
from network.gNodeB_manager import gNodeBManager
from network.sector_manager import SectorManager
from network.ue_manager import UEManager
from network.NetworkLoadManager import NetworkLoadManager
from traffic.traffic_generator import TrafficController

from rl.trace_mobility import MobilityTrace, StaticUserTrace
from rl.trace_serving import TraceServingController
from rl.bs_mapping import load_bs_gnb_mapping, validate_bs_gnb_mapping
from rl.fidelity_provider import build_fidelity_provider, log_fidelity_provider, FidelityLevel
from network.energy_model import build_energy_models, build_gnodeb_energy_state
from network.utils import calculate_distance


LOGGER = logging.getLogger(__name__)


class RANTraceEnv:
    """Minimal in-process RL scaffold for trace-driven RU/gNB ON-OFF control."""

    OBSERVATION_SCHEMA = {
        "timestamp": "current trace timestamp from the mobility CSV",
        "metrics": "aggregate reward-oriented metrics",
        "gnbs": {"<gnb_id>": {"is_active": "bool", "tx_power": "float"}},
        "ues": {
            "<ue_id>": {
                "lat": "float latitude from trace",
                "lon": "float longitude from trace",
                "x_m": "float projected x coordinate from trace",
                "y_m": "float projected y coordinate from trace",
                "nominal_serving_bs_id": "trace serving_bs_id",
                "serving_gnb_id": "actual simulator serving gNB after fallback/outage policy",
                "serving_cell_id": "actual simulator serving cell",
                "serving_sector_id": "actual simulator serving sector",
                "serving_distance_m": "distance in meters to the serving gNB",
                "distance_throughput_factor": "distance-tier throughput multiplier",
                "distance_throughput_tier": "near|mid|far|edge|unknown|outage",
                "estimated_sinr_db": "continuous SINR estimate used by the ns_oran_compatible throughput model",
                "spectral_efficiency_bps_hz": "spectral efficiency derived from the SINR estimate",
                "in_outage": "bool",
                "throughput": "bits/s",
            }
        },
    }
    ACTION_SCHEMA = {
        "none": "no-op",
        "dict": {"gnb_id": "<simulator gNB ID>", "state": "ON|OFF"},
        "tuple": ("<simulator gNB ID>", "ON|OFF"),
        "list": "list of dict or tuple actions applied in order",
    }
    REWARD_SCHEMA = {
        "formula": "legacy: throughput_weight * total_throughput - energy_weight * energy_cost - outage_penalty * outage_ue_count; throughput_active_gnb: reward_alpha * normalized_throughput + reward_beta * normalized_active_gnb_count + reward_gamma * normalized_disconnected_ues; raw_mbps: reward_alpha * tput_mbps + reward_beta * active_gnb_count + reward_gamma * disconnected_ue_count",
        "components": ["throughput_component", "energy_component", "outage_component"],
        "done": "True after the final timestamp available in the mobility trace has been stepped",
    }
    DISTANCE_THROUGHPUT_DEFAULTS = {
        "enabled": True,
        "distance_threshold_1": 100.0,
        "distance_threshold_2": 200.0,
        "distance_threshold_3": 300.0,
        "throughput_factor_near": 1.0,
        "throughput_factor_mid": 0.7,
        "throughput_factor_far": 0.4,
        "throughput_factor_edge": 0.1,
    }
    NS_ORAN_SINR_MODEL_DEFAULTS = {
        "enabled": True,
        "pathloss_model": "threegpp_umi_nlos",
        "reference_distance_m": 1.0,
        "pathloss_reference_db": 43.3,
        "pathloss_exponent": 2.1,
        "additional_pathloss_db": 8.0,
        "noise_figure_db": 7.0,
        "interference_weight": 1.0,
        "interference_density_exponent": 3.0,
        "spectral_efficiency_mode": "amc_table",
        "spectral_efficiency_scale": 0.30,
        "spectral_efficiency_max_bps_hz": 5.0,
        "sinr_min_db": -5.0,
        "sinr_max_db": 30.0,
        "capacity_sharing_mode": "proportional_fair",
    }
    # SINR thresholds and spectral efficiencies from the standard LTE CQI table
    # (3GPP TS 36.213 Table 7.2.3-1), capped at spectral_efficiency_max_bps_hz.
    # Each entry: (sinr_lower_bound_db, se_bps_hz).
    # CQI 0 (below first threshold) maps to sinr_min_db → outage/minimum efficiency.
    _LTE_AMC_TABLE = (
        (-6.5,  0.1523),
        (-4.5,  0.2344),
        (-2.6,  0.3770),
        (-0.5,  0.6016),
        ( 2.0,  0.8770),
        ( 4.0,  1.1758),
        ( 6.5,  1.4766),
        ( 8.6,  1.9141),
        (10.4,  2.4063),
        (12.4,  2.7305),
        (14.8,  3.3223),
        (16.9,  3.9023),
        (19.2,  4.5234),
        (21.7,  5.1152),
        (25.0,  5.5547),
    )
    FULL_DL_PACKET_SIZE_BYTES = 1280
    FULL_DL_INTER_PACKET_INTERVAL_SECONDS = 0.0005
    DETERMINISTIC_SERVICE_DEMAND_BPS = {
        "voice": 64_000.0,
        "video": 5_000_000.0,
        "game": 256_000.0,
        "data": 2_000_000.0,
        "iot": 16_000.0,
    }

    def __init__(
        self,
        *,
        base_dir,
        mobility_trace_csv,
        mobility_trace=None,
        mobility_provider=None,
        static_user_csv=None,
        bs_id_map=None,
        bs_mapping_path=None,
        strict_bs_mapping=True,
        sleep_behavior="reattach",
        energy_cost_per_active_gnb=1.0,
        throughput_weight=1e-6,
        energy_weight=1.0,
        outage_penalty=10.0,
        throughput_mode="full_dl_reference",
        reward_mode="throughput_active_gnb",
        throughput_normalization_mode="fixed_reference",
        throughput_reference_mbps=1000.0,
        reward_alpha=1.0,
        reward_beta=-0.1,
        reward_gamma=-0.3,
        max_ues_per_gnb=10,
        max_coverage_distance_m=500.0,
        deterministic_sector_capacity_bps=100_000_000.0,
        full_dl_packet_size_bytes=FULL_DL_PACKET_SIZE_BYTES,
        full_dl_inter_packet_interval_seconds=FULL_DL_INTER_PACKET_INTERVAL_SECONDS,
        full_dl_sector_capacity_bps=100_000_000.0,
        fidelity_level=FidelityLevel.HIGH.value,
        mobility_profile=DEFAULT_MOBILITY_PROFILE,
        medium_snapshot_sec=0.0,
        ns_oran_sinr_config_overrides=None,
        seed=None,
        max_users=None,
    ):
        self.base_dir = base_dir
        self.seed = seed
        self._seed_random_generators(seed)
        self.config = Config(base_dir)
        self.fidelity_level = FidelityLevel.normalize(fidelity_level)
        provider_override = mobility_provider
        if provider_override is None and mobility_trace is not None:
            from rl.fidelity_provider import IndividualMobilityProvider
            provider_override = IndividualMobilityProvider(mobility_trace, source_path=mobility_trace_csv)
        self.mobility_provider = build_fidelity_provider(
            fidelity_level=self.fidelity_level.value,
            mobility_profile=mobility_profile,
            base_dir=base_dir,
            mobility_trace_csv=mobility_trace_csv,
            medium_snapshot_sec=medium_snapshot_sec,
            provider_override=provider_override,
        )
        self.mobility_trace = self.mobility_provider
        log_fidelity_provider(self.mobility_provider)
        self.static_user_trace = StaticUserTrace.from_csv(static_user_csv) if static_user_csv else None
        self.sleep_behavior = sleep_behavior
        self.strict_bs_mapping = strict_bs_mapping
        self.energy_cost_per_active_gnb = energy_cost_per_active_gnb
        self.throughput_weight = throughput_weight
        self.energy_weight = energy_weight
        self.outage_penalty = outage_penalty
        self.throughput_mode = throughput_mode
        self.reward_mode = reward_mode
        self.throughput_normalization_mode = throughput_normalization_mode
        self.throughput_reference_mbps = float(throughput_reference_mbps)
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)
        self.max_ues_per_gnb = int(max_ues_per_gnb)
        self.max_coverage_distance_m = float(max_coverage_distance_m)
        self.deterministic_sector_capacity_bps = float(deterministic_sector_capacity_bps)
        self.full_dl_packet_size_bytes = int(full_dl_packet_size_bytes)
        self.full_dl_inter_packet_interval_seconds = float(full_dl_inter_packet_interval_seconds)
        self.full_dl_sector_capacity_bps = float(full_dl_sector_capacity_bps)
        self.max_users = max_users
        self.current_index = 0
        self.last_serving_results = {}
        self.last_reward_components = {}
        self.last_energy_update = {
            "step_energy_wh": 0.0,
            "step_energy_kwh": 0.0,
            "total_energy_wh": 0.0,
            "total_energy_kwh": 0.0,
            "total_power_w": 0.0,
            "per_gnb": {},
            "delta_t_seconds": 0.0,
        }
        self.last_bs_load = {}

        self.gnodeb_manager = gNodeBManager.get_instance(base_dir, sleep_behavior=sleep_behavior)
        self.gNodeBs, self.cells, self.sectors, self.initial_ues, self.cell_manager = initialize_network(base_dir, num_ues_to_launch=0)
        self.gnodeb_manager.set_sleep_behavior(sleep_behavior)
        self.sector_manager = SectorManager.get_instance()
        self.ue_manager = UEManager.get_instance(base_dir)
        self.network_load_manager = NetworkLoadManager.get_instance(self.cell_manager, self.sector_manager, self.gnodeb_manager)
        self.traffic_controller = TrafficController(base_dir)

        self.bs_id_map = dict(bs_id_map) if bs_id_map is not None else load_bs_gnb_mapping(base_dir, bs_mapping_path)
        self.bs_mapping_validation = validate_bs_gnb_mapping(
            self.mobility_trace.serving_bs_ids,
            self.gnodeb_manager.gNodeBs.keys(),
            self.bs_id_map,
            strict=strict_bs_mapping,
        )
        self.serving_controller = TraceServingController(
            gnodeb_manager=self.gnodeb_manager,
            ue_manager=self.ue_manager,
            ue_config=self.config.ue_config,
            bs_id_map=self.bs_id_map,
            association_mode="nearest_active" if self.throughput_mode == "ns_oran_compatible" else "trace_nominal_fallback",
            max_ues_per_gnb=self.max_ues_per_gnb,
            max_coverage_distance_m=self.max_coverage_distance_m,
        )
        self.energy_config = self.config.core_config.get("energy_model", {})
        distance_config = dict(self.DISTANCE_THROUGHPUT_DEFAULTS)
        distance_config.update(self.config.core_config.get("distance_throughput_model", {}))
        self.distance_throughput_config = distance_config
        ns_oran_sinr_config = dict(self.NS_ORAN_SINR_MODEL_DEFAULTS)
        ns_oran_sinr_config.update(self.config.core_config.get("ns_oran_sinr_model", {}))
        if ns_oran_sinr_config_overrides:
            ns_oran_sinr_config.update(ns_oran_sinr_config_overrides)
        self.ns_oran_sinr_config = ns_oran_sinr_config
        self.energy_models = build_energy_models(base_dir, self.gnodeb_manager, self.energy_config)
        if self.throughput_mode not in {"full_dl_reference", "deterministic", "legacy", "ns_oran_compatible"}:
            raise ValueError("throughput_mode must be 'full_dl_reference', 'deterministic', 'legacy', or 'ns_oran_compatible'")
        if self.full_dl_packet_size_bytes <= 0:
            raise ValueError("full_dl_packet_size_bytes must be positive")
        if self.full_dl_inter_packet_interval_seconds <= 0:
            raise ValueError("full_dl_inter_packet_interval_seconds must be positive")
        if self.full_dl_sector_capacity_bps <= 0:
            raise ValueError("full_dl_sector_capacity_bps must be positive")

    @classmethod
    def observation_schema(cls):
        return cls.OBSERVATION_SCHEMA

    @classmethod
    def action_schema(cls):
        return cls.ACTION_SCHEMA

    @classmethod
    def reward_schema(cls):
        return cls.REWARD_SCHEMA

    def _seed_random_generators(self, seed):
        if seed is None:
            return
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

    def reset(self):
        self._seed_random_generators(self.seed)
        self.current_index = 0
        self.last_reward_components = {}
        for model in self.energy_models.values():
            model.initialize(self.energy_config)
        self.last_energy_update = {
            "step_energy_wh": 0.0,
            "step_energy_kwh": 0.0,
            "total_energy_wh": 0.0,
            "total_energy_kwh": 0.0,
            "total_power_w": 0.0,
            "per_gnb": {},
            "delta_t_seconds": 0.0,
        }
        for gnb_id in list(self.gnodeb_manager.gNodeBs):
            self.gnodeb_manager.turn_on(gnb_id)
        timestamp, samples = self._current_samples()
        self.last_bs_load = self.mobility_provider.get_bs_load_at_time(timestamp)
        self.last_serving_results = self.serving_controller.apply_samples(samples)
        observation = self.collect_observation(timestamp)
        return observation, {
            "timestamp": timestamp,
            "bs_id_map": self.bs_id_map,
            "bs_mapping_validation": self.bs_mapping_validation,
            "summary": self.build_step_summary(observation["metrics"]),
            "observation_schema": self.observation_schema(),
            "action_schema": self.action_schema(),
            "reward_schema": self.reward_schema(),
        }

    def step(self, action=None):
        self.apply_action(action)
        timestamp, samples = self._current_samples()
        self.last_bs_load = self.mobility_provider.get_bs_load_at_time(timestamp)
        self.last_serving_results = self.serving_controller.apply_samples(samples)
        throughput = self._update_throughput()
        delta_t_seconds = self._current_step_delta_seconds()
        self.last_energy_update = self._update_energy_models(timestamp, delta_t_seconds)
        metrics = self.collect_metrics(total_throughput=throughput)
        reward_components = self.compute_reward_components(metrics)
        reward = reward_components["total"]
        self.last_reward_components = reward_components
        observation = self.collect_observation(timestamp, metrics=metrics)
        info = {
            "timestamp": timestamp,
            "serving_results": self.last_serving_results,
            "metrics": metrics,
            "reward": reward,
            "reward_components": reward_components,
            "summary": self.build_step_summary(metrics),
            "bs_id_map": self.bs_id_map,
            "bs_mapping_validation": self.bs_mapping_validation,
            "observation_schema": self.observation_schema(),
            "action_schema": self.action_schema(),
            "reward_schema": self.reward_schema(),
            "throughput_mode": self.throughput_mode,
            "energy_update": self.last_energy_update,
        }
        self.current_index += 1
        done = self.current_index >= len(self.mobility_trace)
        return observation, reward, done, info

    def apply_action(self, action):
        if action is None:
            return
        actions = action if isinstance(action, list) else [action]
        for item in actions:
            if isinstance(item, tuple):
                gnb_id, state = item
            else:
                gnb_id = item.get("gnb_id") or item.get("gNodeB_ID") or item.get("id")
                state = item.get("state") or item.get("target_state")
            state = str(state).upper()
            if state == "OFF":
                self.gnodeb_manager.turn_off(gnb_id, sleep_behavior=self.sleep_behavior)
            elif state == "ON":
                self.gnodeb_manager.turn_on(gnb_id)
            else:
                raise ValueError(f"Unsupported action state {state!r}; expected ON or OFF")

    def _current_samples(self):
        timestamp, samples = self.mobility_trace.samples_at_index(self.current_index)
        if self.max_users is not None:
            samples = samples[: self.max_users]
        return timestamp, samples

    def _update_throughput(self):
        if self.throughput_mode == "ns_oran_compatible":
            return self._update_ns_oran_compatible_throughput()
        if self.throughput_mode == "full_dl_reference":
            return self._update_full_dl_reference_throughput()
        if self.throughput_mode == "deterministic":
            return self._update_deterministic_throughput()
        return self._update_legacy_throughput()

    def _update_legacy_throughput(self):
        total = 0.0
        for ue in self.ue_manager.ues.values():
            result = self.traffic_controller.calculate_throughput(ue)
            throughput = self._apply_distance_throughput_factor(ue, float(result["throughput"]))
            ue.throughput = throughput
            total += throughput
        try:
            self.network_load_manager.network_measurement()
        except Exception:
            pass
        return total

    def _is_served_on_active_path(self, ue):
        if getattr(ue, "in_outage", False) or getattr(ue, "is_connected", True) is False:
            return False, None
        if not ue.ConnectedSector:
            return False, None
        sector = self.sector_manager.get_sector_by_id(ue.ConnectedSector)
        if not sector:
            return False, None
        gnodeb = self.gnodeb_manager.get_gNodeB(sector.cell.gNodeB_ID)
        active = (
            gnodeb is not None
            and getattr(gnodeb, "is_active", True)
            and getattr(sector.cell, "IsActive", True)
            and getattr(sector, "is_active", True)
        )
        return active, sector if active else None

    def _update_full_dl_reference_throughput(self):
        total = 0.0
        offered_load_bps = (
            self.full_dl_packet_size_bytes
            * 8.0
            / self.full_dl_inter_packet_interval_seconds
        )
        served_sector_ids = []
        ue_sector = {}
        for ue in self.ue_manager.ues.values():
            served, sector = self._is_served_on_active_path(ue)
            if served:
                ue_sector[ue.ID] = sector
                served_sector_ids.append(sector.sector_id)
            else:
                ue.throughput = 0.0
                ue.ue_delay = 0.0
                ue.ue_jitter = 0.0
                ue.ue_packet_loss_rate = 1.0 if getattr(ue, "in_outage", False) else 0.0

        sector_counts = Counter(served_sector_ids)
        for ue in self.ue_manager.ues.values():
            sector = ue_sector.get(ue.ID)
            if not sector:
                continue
            sector_count = max(1, sector_counts.get(sector.sector_id, 1))
            capacity_share = self.full_dl_sector_capacity_bps / sector_count
            base_throughput = min(offered_load_bps, capacity_share)
            throughput = self._apply_distance_throughput_factor(ue, base_throughput, sector=sector)
            ue.throughput = float(throughput)
            ue.ue_delay = 0.0
            ue.ue_jitter = 0.0
            ue.ue_packet_loss_rate = 0.0
            total += ue.throughput

        try:
            self.network_load_manager.network_measurement()
        except Exception:
            pass
        return total

    def _update_ns_oran_compatible_throughput(self):
        total = 0.0
        offered_load_bps = (
            self.full_dl_packet_size_bytes
            * 8.0
            / self.full_dl_inter_packet_interval_seconds
        )
        served_sector_ids = []
        ue_sector = {}
        for ue in self.ue_manager.ues.values():
            served, sector = self._is_served_on_active_path(ue)
            if served:
                ue_sector[ue.ID] = sector
                served_sector_ids.append(sector.sector_id)
                setattr(ue, "serving_distance_m", self._compute_ue_serving_distance_m(ue, sector=sector))
                setattr(ue, "distance_throughput_factor", 1.0)
                setattr(ue, "distance_throughput_tier", "pdcp_compatible")
            else:
                ue.throughput = 0.0
                ue.ue_delay = 0.0
                ue.ue_jitter = 0.0
                ue.ue_packet_loss_rate = 1.0 if getattr(ue, "in_outage", False) else 0.0
                setattr(ue, "serving_distance_m", self._compute_ue_serving_distance_m(ue, sector=None))
                setattr(ue, "distance_throughput_factor", 0.0)
                setattr(ue, "distance_throughput_tier", "outage")
                setattr(ue, "estimated_sinr_db", None)
                setattr(ue, "spectral_efficiency_bps_hz", None)
                setattr(ue, "estimated_pathloss_db", None)

        # Compute SINR and efficiency for every served UE first.
        ue_sinr: dict = {}
        ue_eff: dict = {}
        ue_se: dict = {}
        for ue_id, sector in ue_sector.items():
            ue = self.ue_manager.ues[ue_id]
            sinr_db = self._estimate_ue_sinr_db(ue, sector=sector)
            eff, se = self._sinr_to_efficiency_factor(sinr_db, sector=sector)
            ue_sinr[ue_id] = sinr_db
            ue_eff[ue_id] = eff
            ue_se[ue_id] = se

        # Per-sector sum of efficiencies (for proportional-fair sharing).
        sector_eff_sum: dict = {}
        for ue_id, sector in ue_sector.items():
            sid = sector.sector_id
            sector_eff_sum[sid] = sector_eff_sum.get(sid, 0.0) + ue_eff[ue_id]

        sharing_mode = str(
            self.ns_oran_sinr_config.get("capacity_sharing_mode", "equal")
        ).lower()

        sector_counts = Counter(served_sector_ids)
        for ue in self.ue_manager.ues.values():
            sector = ue_sector.get(ue.ID)
            if not sector:
                continue
            sinr_db = ue_sinr[ue.ID]
            efficiency_factor = ue_eff[ue.ID]
            spectral_efficiency_bps_hz = ue_se[ue.ID]

            if sharing_mode == "proportional_fair":
                # Proportional-Fair capacity sharing with multi-user diversity.
                # Time allocation ∝ efficiency → capacity ∝ eff²/Σ(eff).
                # Reduces to equal time sharing when all UEs have the same SINR;
                # gives a diversity gain when SINRs are heterogeneous (closer UEs
                # get more PRBs, distant ones less, cell total > equal sharing).
                eff_sum = sector_eff_sum.get(sector.sector_id, efficiency_factor)
                if eff_sum > 0.0:
                    capacity_share_bps = (
                        self.full_dl_sector_capacity_bps
                        * (efficiency_factor ** 2)
                        / eff_sum
                    )
                else:
                    capacity_share_bps = 0.0
                throughput_bps = min(offered_load_bps, capacity_share_bps)
            else:
                sector_count = max(1, sector_counts.get(sector.sector_id, 1))
                capacity_share_bps = self.full_dl_sector_capacity_bps / sector_count
                throughput_bps = min(offered_load_bps, capacity_share_bps * efficiency_factor)

            ue.throughput = float(throughput_bps)
            ue.ue_delay = 0.0
            ue.ue_jitter = 0.0
            ue.ue_packet_loss_rate = 0.0
            setattr(ue, "estimated_sinr_db", sinr_db)
            setattr(ue, "spectral_efficiency_bps_hz", spectral_efficiency_bps_hz)
            setattr(ue, "distance_throughput_factor", efficiency_factor)
            setattr(ue, "distance_throughput_tier", "sinr_efficiency")
            total += ue.throughput

        try:
            self.network_load_manager.network_measurement()
        except Exception:
            pass
        return total

    def _sector_bandwidth_hz(self, sector):
        bandwidth_mhz = getattr(sector, "bandwidth", None)
        if bandwidth_mhz is None:
            return None
        try:
            return float(bandwidth_mhz) * 1_000_000.0
        except (TypeError, ValueError):
            return None

    def _compute_log_distance_pathloss_db(self, distance_m, frequency_hz):
        config = self.ns_oran_sinr_config
        reference_distance_m = max(float(config.get("reference_distance_m", 1.0)), 1e-6)
        pathloss_reference_db = float(config.get("pathloss_reference_db", 43.3))
        pathloss_exponent = float(config.get("pathloss_exponent", 2.1))
        additional_pathloss_db = float(config.get("additional_pathloss_db", 0.0))
        effective_distance_m = max(float(distance_m), reference_distance_m)
        pathloss_model = str(config.get("pathloss_model", "log_distance")).lower()
        if frequency_hz and pathloss_model == "fspl":
            frequency_mhz = max(float(frequency_hz) / 1_000_000.0, 1e-9)
            return 32.4 + 20.0 * math.log10(effective_distance_m / 1000.0) + 20.0 * math.log10(frequency_mhz)
        if frequency_hz and pathloss_model == "threegpp_umi_nlos":
            frequency_ghz = max(float(frequency_hz) / 1_000_000_000.0, 1e-9)
            # 3GPP TR 38.901 UMi Street Canyon NLoS. This remains a configurable
            # internal approximation; additional_pathloss_db can still be used to
            # calibrate toward the observed external ns-3 throughput.
            return (
                35.3 * math.log10(effective_distance_m)
                + 22.4
                + 21.3 * math.log10(frequency_ghz)
                + additional_pathloss_db
            )
        return (
            pathloss_reference_db
            + 10.0 * pathloss_exponent * math.log10(effective_distance_m / reference_distance_m)
            + additional_pathloss_db
        )

    def _estimate_received_power_dbm(self, ue, sector):
        serving_distance_m = self._compute_ue_serving_distance_m(ue, sector=sector)
        if serving_distance_m is None:
            return None
        frequency_hz = getattr(sector, "frequency", None)
        pathloss_db = self._compute_log_distance_pathloss_db(serving_distance_m, frequency_hz)
        tx_power_dbm = float(getattr(sector, "tx_power", getattr(sector, "nominal_tx_power", 20.0)))
        rx_power_dbm = tx_power_dbm - pathloss_db
        setattr(ue, "estimated_pathloss_db", pathloss_db)
        return rx_power_dbm

    @staticmethod
    def _dbm_to_mw(value_dbm):
        return 10.0 ** (float(value_dbm) / 10.0)

    @staticmethod
    def _mw_to_dbm(value_mw):
        return 10.0 * math.log10(max(float(value_mw), 1e-15))

    def _estimate_ue_sinr_db(self, ue, sector):
        # Synthetic UEs (LOW fidelity) carry a gNB-position lat/lon for association
        # purposes but have no precise individual position.  Skip the pathloss
        # computation entirely and return a fixed nominal SINR so that throughput
        # reflects load-based sharing rather than an artificially perfect distance-0 link.
        if not getattr(ue, "has_precise_position", True) or getattr(ue, "is_synthetic", False):
            return float(self.ns_oran_sinr_config.get("sinr_synthetic_db", -2.0))
        serving_rx_dbm = self._estimate_received_power_dbm(ue, sector)
        if serving_rx_dbm is None:
            return float(self.ns_oran_sinr_config.get("sinr_min_db", -10.0))

        interference_mw = 0.0
        interference_weight = float(self.ns_oran_sinr_config.get("interference_weight", 1.0))
        for other_sector in self.sectors:
            if other_sector.sector_id == sector.sector_id:
                continue
            if not getattr(other_sector, "is_active", True) or not getattr(other_sector.cell, "IsActive", True):
                continue
            other_gnb = self.gnodeb_manager.get_gNodeB(other_sector.cell.gNodeB_ID)
            if other_gnb is None or not getattr(other_gnb, "is_active", True):
                continue
            other_distance_m = self._compute_ue_serving_distance_m(ue, sector=other_sector)
            if other_distance_m is None:
                continue
            other_pathloss_db = self._compute_log_distance_pathloss_db(
                other_distance_m,
                getattr(other_sector, "frequency", None),
            )
            other_tx_power_dbm = float(
                getattr(other_sector, "tx_power", getattr(other_sector, "nominal_tx_power", 20.0))
            )
            interference_mw += self._dbm_to_mw(other_tx_power_dbm - other_pathloss_db) * interference_weight

        # Density correction: in a real dense deployment, when many gNBs are active
        # simultaneously on the same frequency, near-field interference grows faster
        # than linearly due to the proximity of co-channel transmitters.
        # (n_active/n_total)^k < 1 reduces interference when few gNBs are ON,
        # capturing the ns-3 observed SINR collapse at full activation.
        density_exp = float(self.ns_oran_sinr_config.get("interference_density_exponent", 0.0))
        if density_exp > 0.0 and interference_mw > 0.0:
            n_active = sum(
                1 for gnb in self.gnodeb_manager.gNodeBs.values()
                if getattr(gnb, "is_active", True)
            )
            n_total = max(1, len(self.gnodeb_manager.gNodeBs))
            density_factor = (n_active / n_total) ** density_exp
            interference_mw *= density_factor

        bandwidth_hz = self._sector_bandwidth_hz(sector) or 20_000_000.0
        noise_figure_db = float(self.ns_oran_sinr_config.get("noise_figure_db", 7.0))
        thermal_noise_dbm = -174.0 + 10.0 * math.log10(max(bandwidth_hz, 1.0)) + noise_figure_db
        noise_mw = self._dbm_to_mw(thermal_noise_dbm)

        sinr_linear = self._dbm_to_mw(serving_rx_dbm) / max(noise_mw + interference_mw, 1e-15)
        sinr_db = 10.0 * math.log10(max(sinr_linear, 1e-15))
        sinr_min_db = float(self.ns_oran_sinr_config.get("sinr_min_db", -10.0))
        sinr_max_db = float(self.ns_oran_sinr_config.get("sinr_max_db", 30.0))
        return max(sinr_min_db, min(sinr_max_db, sinr_db))

    def _sinr_to_efficiency_factor(self, sinr_db, sector):
        config = self.ns_oran_sinr_config
        max_se = max(float(config.get("spectral_efficiency_max_bps_hz", 5.0)), 1e-9)
        mode = str(config.get("spectral_efficiency_mode", "shannon")).lower()
        sinr_db_f = float(sinr_db)

        if mode == "amc_table":
            table = self._LTE_AMC_TABLE
            if sinr_db_f < table[0][0]:
                se = 0.0
            elif sinr_db_f >= table[-1][0]:
                se = table[-1][1]
            else:
                # Linear interpolation between adjacent CQI thresholds.
                se = table[0][1]
                for i in range(1, len(table)):
                    lo_thr, lo_se = table[i - 1]
                    hi_thr, hi_se = table[i]
                    if lo_thr <= sinr_db_f < hi_thr:
                        frac = (sinr_db_f - lo_thr) / (hi_thr - lo_thr)
                        se = lo_se + frac * (hi_se - lo_se)
                        break
            spectral_efficiency_bps_hz = min(max_se, se)
        else:
            scale = max(float(config.get("spectral_efficiency_scale", 0.30)), 1e-9)
            sinr_linear = 10.0 ** (sinr_db_f / 10.0)
            spectral_efficiency_bps_hz = min(max_se, scale * math.log2(1.0 + sinr_linear))

        efficiency_factor = max(0.0, min(1.0, spectral_efficiency_bps_hz / max_se))
        return efficiency_factor, spectral_efficiency_bps_hz

    def _resolve_serving_sector_and_gnb(self, ue, sector=None):
        serving_sector = sector
        if serving_sector is None and getattr(ue, "ConnectedSector", None):
            serving_sector = self.sector_manager.get_sector_by_id(ue.ConnectedSector)
        if not serving_sector:
            return None, None
        gnodeb = self.gnodeb_manager.get_gNodeB(serving_sector.cell.gNodeB_ID)
        return serving_sector, gnodeb

    def _compute_ue_serving_distance_m(self, ue, sector=None):
        _serving_sector, gnodeb = self._resolve_serving_sector_and_gnb(ue, sector=sector)
        if not gnodeb:
            return None
        nominal_serving_bs_id = getattr(ue, "nominal_serving_bs_id", None)
        trace_distance_m = getattr(ue, "dist_to_serving_bs_m", None)
        mapped_nominal_gnb_id = self.bs_id_map.get(nominal_serving_bs_id) if nominal_serving_bs_id else None
        if trace_distance_m is not None and mapped_nominal_gnb_id == getattr(gnodeb, "ID", None):
            return float(trace_distance_m)
        ue_x_m = getattr(ue, "x_m", None)
        ue_y_m = getattr(ue, "y_m", None)
        gnb_x_m = getattr(gnodeb, "x_m", None)
        gnb_y_m = getattr(gnodeb, "y_m", None)
        if None not in (ue_x_m, ue_y_m, gnb_x_m, gnb_y_m):
            return float(((float(ue_x_m) - float(gnb_x_m)) ** 2 + (float(ue_y_m) - float(gnb_y_m)) ** 2) ** 0.5)
        ue_lat = getattr(ue, "Latitude", None)
        ue_lon = getattr(ue, "Longitude", None)
        gnb_lat = getattr(gnodeb, "Latitude", None)
        gnb_lon = getattr(gnodeb, "Longitude", None)
        if None in (ue_lat, ue_lon, gnb_lat, gnb_lon):
            return None
        return float(calculate_distance(ue_lat, ue_lon, gnb_lat, gnb_lon) * 1000.0)

    def _get_distance_tier_factor(self, distance_m):
        if not self.distance_throughput_config.get("enabled", True):
            return 1.0, "disabled"
        if distance_m is None:
            return 1.0, "unknown"
        d1 = float(self.distance_throughput_config["distance_threshold_1"])
        d2 = float(self.distance_throughput_config["distance_threshold_2"])
        d3 = float(self.distance_throughput_config["distance_threshold_3"])
        if distance_m <= d1:
            return float(self.distance_throughput_config["throughput_factor_near"]), "near"
        if distance_m <= d2:
            return float(self.distance_throughput_config["throughput_factor_mid"]), "mid"
        if distance_m <= d3:
            return float(self.distance_throughput_config["throughput_factor_far"]), "far"
        return float(self.distance_throughput_config["throughput_factor_edge"]), "edge"

    def _apply_distance_throughput_factor(self, ue, base_throughput, sector=None):
        if base_throughput <= 0:
            setattr(ue, "serving_distance_m", self._compute_ue_serving_distance_m(ue, sector=sector))
            setattr(ue, "distance_throughput_factor", 0.0)
            setattr(ue, "distance_throughput_tier", "outage")
            return 0.0
        distance_m = self._compute_ue_serving_distance_m(ue, sector=sector)
        factor, tier = self._get_distance_tier_factor(distance_m)
        setattr(ue, "serving_distance_m", distance_m)
        setattr(ue, "distance_throughput_factor", factor)
        setattr(ue, "distance_throughput_tier", tier)
        return float(base_throughput) * factor

    def _update_deterministic_throughput(self):
        total = 0.0
        sector_counts = Counter()
        served_ue_sector = {}
        for ue in self.ue_manager.ues.values():
            served, sector = self._is_served_on_active_path(ue)
            if served:
                served_ue_sector[ue.ID] = sector
                sector_counts[sector.sector_id] += 1
        for ue in self.ue_manager.ues.values():
            sector = served_ue_sector.get(ue.ID)
            if not sector:
                ue.throughput = 0.0
                ue.ue_delay = 0.0
                ue.ue_jitter = 0.0
                ue.ue_packet_loss_rate = 1.0
                continue
            sector_count = max(1, sector_counts.get(sector.sector_id, 1))
            capacity_share = self.deterministic_sector_capacity_bps / sector_count
            service_demand = self.DETERMINISTIC_SERVICE_DEMAND_BPS.get(str(ue.ServiceType).lower(), 1_000_000.0)
            base_throughput = min(service_demand * float(getattr(ue, "traffic_factor", 1.0)), capacity_share)
            throughput = self._apply_distance_throughput_factor(ue, base_throughput, sector=sector)
            ue.throughput = float(throughput)
            ue.ue_delay = 0.0
            ue.ue_jitter = 0.0
            ue.ue_packet_loss_rate = 0.0
            total += ue.throughput
        try:
            self.network_load_manager.network_measurement()
        except Exception:
            pass
        return total

    def _timestamp_seconds(self, value):
        unit = getattr(self.mobility_trace, "timestamp_unit", "unknown")
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

    def _current_step_delta_seconds(self):
        timestamps = self.mobility_trace.timestamps
        if len(timestamps) <= 1:
            return 1.0
        current_value = self._timestamp_seconds(timestamps[self.current_index])
        if current_value is None:
            return 1.0
        if self.current_index + 1 < len(timestamps):
            next_value = self._timestamp_seconds(timestamps[self.current_index + 1])
            if next_value is not None:
                return max(0.0, next_value - current_value)
        if self.current_index > 0:
            prev_value = self._timestamp_seconds(timestamps[self.current_index - 1])
            if prev_value is not None:
                return max(0.0, current_value - prev_value)
        return 1.0

    def _update_energy_models(self, timestamp, delta_t_seconds):
        per_gnb = {}
        step_energy_wh = 0.0
        total_energy_wh = 0.0
        total_power_w = 0.0
        for gnodeb_id, gnodeb in sorted(self.gnodeb_manager.gNodeBs.items()):
            sim_state = build_gnodeb_energy_state(gnodeb)
            sim_state["timestamp"] = timestamp
            update = self.energy_models[gnodeb_id].update_energy(delta_t_seconds, sim_state)
            per_gnb[gnodeb_id] = update
            step_energy_wh += update["step_energy_wh"]
            total_energy_wh += update["total_energy_wh"]
            total_power_w += update["current_power_w"]
        return {
            "step_energy_wh": step_energy_wh,
            "step_energy_kwh": step_energy_wh / 1000.0,
            "total_energy_wh": total_energy_wh,
            "total_energy_kwh": total_energy_wh / 1000.0,
            "total_power_w": total_power_w,
            "per_gnb": per_gnb,
            "delta_t_seconds": delta_t_seconds,
        }

    def collect_metrics(self, total_throughput=None):
        active_gnbs = [gnb for gnb in self.gnodeb_manager.gNodeBs.values() if getattr(gnb, "is_active", True)]
        outage_ues = [ue for ue in self.ue_manager.ues.values() if getattr(ue, "in_outage", False) or getattr(ue, "is_connected", True) is False]
        if total_throughput is None:
            total_throughput = sum(float(getattr(ue, "throughput", 0.0)) for ue in self.ue_manager.ues.values())
        if self.energy_config.get("energy_model_enabled", True):
            energy_cost = self.last_energy_update["step_energy_wh"]
            reference_total_power_w = sum(model.max_power_w for model in self.energy_models.values())
            reference_step_energy_wh = (reference_total_power_w * self.last_energy_update["delta_t_seconds"]) / 3600.0
            energy_saving = max(0.0, reference_step_energy_wh - energy_cost)
        else:
            energy_cost = len(active_gnbs) * self.energy_cost_per_active_gnb
            max_energy_cost = max(1, len(self.gnodeb_manager.gNodeBs)) * self.energy_cost_per_active_gnb
            energy_saving = max_energy_cost - energy_cost
        return {
            "total_throughput": float(total_throughput),
            "aggregate_throughput_mbps": float(total_throughput) / 1_000_000.0,
            "ue_count": len(self.ue_manager.ues),
            "outage_ue_count": len(outage_ues),
            "active_gnb_count": len(active_gnbs),
            "total_gnb_count": len(self.gnodeb_manager.gNodeBs),
            "trace_bs_load": dict(sorted(self.last_bs_load.items())),
            "energy_cost": energy_cost,
            "energy_saving": energy_saving,
            "current_power_w": self.last_energy_update["total_power_w"],
            "step_energy_wh": self.last_energy_update["step_energy_wh"],
            "step_energy_kwh": self.last_energy_update["step_energy_kwh"],
            "total_energy_wh": self.last_energy_update["total_energy_wh"],
            "total_energy_kwh": self.last_energy_update["total_energy_kwh"],
            "energy_delta_t_seconds": self.last_energy_update["delta_t_seconds"],
            "normalized_throughput": float(total_throughput) / 1_000_000.0 / max(self.throughput_reference_mbps, 1e-9),
            "normalized_active_gnb_count": len(active_gnbs) / max(float(len(self.gnodeb_manager.gNodeBs)), 1.0),
        }

    def build_step_summary(self, metrics=None):
        if metrics is None:
            metrics = self.collect_metrics()
        status_counts = Counter(result.get("status", "unknown") for result in self.last_serving_results.values())
        nominal_bs_counts = Counter(
            getattr(ue, "nominal_serving_bs_id", None) or "UNKNOWN"
            for ue in self.ue_manager.ues.values()
        )
        serving_gnb_counts = Counter(
            ue.gNodeB_ID if not getattr(ue, "in_outage", False) and getattr(ue, "is_connected", True) else "OUTAGE"
            for ue in self.ue_manager.ues.values()
        )
        per_gnb_throughput_mbps = Counter()
        for ue in self.ue_manager.ues.values():
            if getattr(ue, "in_outage", False) or getattr(ue, "is_connected", True) is False:
                continue
            if not getattr(ue, "gNodeB_ID", None):
                continue
            per_gnb_throughput_mbps[ue.gNodeB_ID] += float(getattr(ue, "throughput", 0.0)) / 1_000_000.0
        unmapped_bs_counts = Counter(
            result.get("nominal_bs_id")
            for result in self.last_serving_results.values()
            if result.get("status") == "unmapped_outage"
        )
        return {
            "total_ues": len(self.ue_manager.ues),
            "nominal_serving_ues": status_counts.get("nominal", 0),
            "fallback_reassigned_ues": status_counts.get("fallback", 0),
            "outage_ues": metrics["outage_ue_count"],
            "active_gnb_count": metrics["active_gnb_count"],
            "unmapped_bs_ues": status_counts.get("unmapped_outage", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "serving_gnb_counts": dict(sorted(serving_gnb_counts.items())),
            "per_gnb_ue_counts": dict(sorted((key, value) for key, value in serving_gnb_counts.items() if key != "OUTAGE")),
            "disconnected_ues": metrics["outage_ue_count"],
            "per_gnb_throughput_mbps": dict(sorted(per_gnb_throughput_mbps.items())),
            "nominal_trace_bs_counts": dict(sorted(nominal_bs_counts.items())),
            "unmapped_trace_bs_counts": dict(sorted((k, v) for k, v in unmapped_bs_counts.items() if k)),
            "reward_total": self.last_reward_components.get("total"),
            "trace_bs_load": dict(sorted(self.last_bs_load.items())),
            "reward_throughput_component": self.last_reward_components.get("throughput_component"),
            "reward_energy_component": self.last_reward_components.get("energy_component"),
            "reward_outage_component": self.last_reward_components.get("outage_component"),
            "normalized_throughput": metrics["normalized_throughput"],
            "aggregate_throughput_mbps": metrics["aggregate_throughput_mbps"],
            "normalized_active_gnb_count": metrics["normalized_active_gnb_count"],
            "reward_alpha": self.reward_alpha,
            "reward_beta": self.reward_beta,
            "reward_gamma": self.reward_gamma,
            "max_ues_per_gnb": self.max_ues_per_gnb,
            "max_coverage_distance_m": self.max_coverage_distance_m,
            "energy_cost": metrics["energy_cost"],
            "energy_saving": metrics["energy_saving"],
            "total_throughput": metrics["total_throughput"],
            "current_power_w": metrics["current_power_w"],
            "step_energy_wh": metrics["step_energy_wh"],
            "step_energy_kwh": metrics["step_energy_kwh"],
            "total_energy_wh": metrics["total_energy_wh"],
            "total_energy_kwh": metrics["total_energy_kwh"],
        }

    def compute_reward_components(self, metrics):
        ue_count = max(int(metrics.get("ue_count", 0)), 1)
        outage_count = int(metrics.get("outage_ue_count", 0))
        normalized_disconnected = float(outage_count) / float(ue_count)
        if self.reward_mode == "raw_mbps":
            # Raw Mbps: alpha*tput_mbps + beta*active_gnb_count + gamma*disconnected_count
            throughput_component = self.reward_alpha * metrics["aggregate_throughput_mbps"]
            energy_component = self.reward_beta * metrics["active_gnb_count"]
            outage_component = self.reward_gamma * outage_count
            return {
                "throughput_component": throughput_component,
                "energy_component": energy_component,
                "outage_component": outage_component,
                "normalized_throughput": metrics["normalized_throughput"],
                "normalized_active_gnb_count": metrics["normalized_active_gnb_count"],
                "normalized_disconnected_ues": normalized_disconnected,
                "total": throughput_component + energy_component + outage_component,
            }
        if self.reward_mode == "throughput_active_gnb":
            throughput_component = self.reward_alpha * metrics["normalized_throughput"]
            energy_component = self.reward_beta * metrics["normalized_active_gnb_count"]
            outage_component = self.reward_gamma * normalized_disconnected
            return {
                "throughput_component": throughput_component,
                "energy_component": energy_component,
                "outage_component": outage_component,
                "normalized_throughput": metrics["normalized_throughput"],
                "normalized_active_gnb_count": metrics["normalized_active_gnb_count"],
                "normalized_disconnected_ues": normalized_disconnected,
                "total": throughput_component + energy_component + outage_component,
            }
        throughput_component = self.throughput_weight * metrics["total_throughput"]
        energy_component = -self.energy_weight * metrics["energy_cost"]
        outage_component = -self.outage_penalty * metrics["outage_ue_count"]
        return {
            "throughput_component": throughput_component,
            "energy_component": energy_component,
            "outage_component": outage_component,
            "total": throughput_component + energy_component + outage_component,
        }

    def compute_reward(self, metrics):
        return self.compute_reward_components(metrics)["total"]

    def collect_observation(self, timestamp, metrics=None):
        if metrics is None:
            metrics = self.collect_metrics()
        gnb_state = {
            gnb_id: {
                "is_active": bool(getattr(gnb, "is_active", True)),
                "tx_power": gnb.TransmissionPower,
            }
            for gnb_id, gnb in sorted(self.gnodeb_manager.gNodeBs.items())
        }
        ue_state = {
            ue_id: {
                "lat": getattr(ue, "Latitude", None),
                "lon": getattr(ue, "Longitude", None),
                "x_m": getattr(ue, "X", None),
                "y_m": getattr(ue, "Y", None),
                "nominal_serving_bs_id": getattr(ue, "nominal_serving_bs_id", None),
                "serving_gnb_id": ue.gNodeB_ID,
                "serving_cell_id": ue.ConnectedCellID,
                "serving_sector_id": ue.ConnectedSector,
                "serving_distance_m": getattr(ue, "serving_distance_m", None),
                "distance_throughput_factor": getattr(ue, "distance_throughput_factor", None),
                "distance_throughput_tier": getattr(ue, "distance_throughput_tier", None),
                "estimated_sinr_db": getattr(ue, "estimated_sinr_db", None),
                "spectral_efficiency_bps_hz": getattr(ue, "spectral_efficiency_bps_hz", None),
                "in_outage": getattr(ue, "in_outage", False),
                "throughput": getattr(ue, "throughput", 0.0),
                "is_synthetic": getattr(ue, "is_synthetic", False),
                "has_precise_position": getattr(ue, "has_precise_position", True),
            }
            for ue_id, ue in sorted(self.ue_manager.ues.items())
        }
        return {"timestamp": timestamp, "metrics": metrics, "gnbs": gnb_state, "ues": ue_state}
