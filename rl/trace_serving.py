from network.ue import UE
from network.sector_manager import SectorManager
from network.utils import calculate_distance


def ue_id_from_trace_user(user_id):
    user_id = str(user_id).strip().lower()
    return user_id if user_id.startswith("ue") else f"ue{user_id}"


class TraceServingController:
    def __init__(
        self,
        *,
        gnodeb_manager,
        ue_manager,
        ue_config,
        bs_id_map=None,
        association_mode="trace_nominal_fallback",
        max_ues_per_gnb=10,
        max_coverage_distance_m=500.0,
    ):
        self.gnodeb_manager = gnodeb_manager
        self.ue_manager = ue_manager
        self.ue_config = ue_config
        self.bs_id_map = bs_id_map or {}
        self.association_mode = association_mode
        self.max_ues_per_gnb = int(max_ues_per_gnb)
        self.max_coverage_distance_m = float(max_coverage_distance_m)

    def set_bs_id_map(self, bs_id_map):
        self.bs_id_map = dict(bs_id_map or {})

    def ensure_ue(self, sample):
        ue_id = ue_id_from_trace_user(sample.user_id)
        ue = self.ue_manager.ues.get(ue_id)
        if ue:
            return ue, False
        location = None
        if getattr(sample, "lat", None) is not None and getattr(sample, "lon", None) is not None:
            location = [sample.lat, sample.lon]
        ue = UE(
            config=self.ue_config,
            ue_id=ue_id,
            location=location,
            latitude=sample.lat,
            longitude=sample.lon,
            x_m=sample.x_m,
            y_m=sample.y_m,
            nominal_serving_bs_id=sample.serving_bs_id,
            is_synthetic=getattr(sample, "synthetic", False),
            has_precise_position=getattr(sample, "has_precise_position", True),
        )
        self.ue_manager.ues[ue.ID] = ue
        return ue, True

    def apply_samples(self, samples):
        active_ue_ids = {ue_id_from_trace_user(sample.user_id) for sample in samples}
        stale_ue_ids = [ue_id for ue_id in list(self.ue_manager.ues.keys()) if ue_id not in active_ue_ids]
        for ue_id in stale_ue_ids:
            self._remove_stale_trace_ue(ue_id)
        if self.association_mode == "nearest_active":
            return self._apply_global_nearest_active_assignment(samples)
        results = {}
        for sample in samples:
            ue, created = self.ensure_ue(sample)
            ue.update_trace_state(sample)
            results[ue.ID] = self.apply_serving_policy(ue, sample, created=created)
        return results

    def _max_ues_for_gnb(self, gnb_id):
        return int(self.max_ues_per_gnb)

    def _gnb_current_load(self, gnb_id, exclude_ue_id=None):
        count = 0
        for ue in self.ue_manager.ues.values():
            if exclude_ue_id and ue.ID == exclude_ue_id:
                continue
            if getattr(ue, "in_outage", False) or getattr(ue, "is_connected", True) is False:
                continue
            if getattr(ue, "gNodeB_ID", None) == gnb_id:
                count += 1
        return count

    def _distance_to_gnb(self, sample, gnodeb):
        sample_x = getattr(sample, "x_m", None)
        sample_y = getattr(sample, "y_m", None)
        gnb_x = getattr(gnodeb, "x_m", None)
        gnb_y = getattr(gnodeb, "y_m", None)
        if None not in (sample_x, sample_y, gnb_x, gnb_y):
            dx = float(sample_x) - float(gnb_x)
            dy = float(sample_y) - float(gnb_y)
            return (dx * dx + dy * dy) ** 0.5
        sample_lat = getattr(sample, "lat", None)
        sample_lon = getattr(sample, "lon", None)
        gnb_lat = getattr(gnodeb, "Latitude", None)
        gnb_lon = getattr(gnodeb, "Longitude", None)
        if None in (sample_lat, sample_lon, gnb_lat, gnb_lon):
            return float("inf")
        return calculate_distance(sample_lat, sample_lon, gnb_lat, gnb_lon) * 1000.0

    def _nearest_active_gnb_ids(self, sample):
        # Only apply distance filter for UEs with real positions.
        # Synthetic UEs (LOW fidelity, no x_m/y_m) have no position data:
        # _distance_to_gnb returns inf, which would incorrectly filter them all out.
        has_position = (
            getattr(sample, "has_precise_position", True)
            and getattr(sample, "x_m", None) is not None
        )
        candidates = []
        for gnb_id, gnodeb in self.gnodeb_manager.gNodeBs.items():
            if not getattr(gnodeb, "is_active", True):
                continue
            load = self._gnb_current_load(gnb_id)
            if load >= self._max_ues_for_gnb(gnb_id):
                continue
            dist = self._distance_to_gnb(sample, gnodeb)
            if has_position and dist > self.max_coverage_distance_m:
                continue
            candidates.append((dist, gnb_id))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [gnb_id for _, gnb_id in candidates]

    def _apply_global_nearest_active_assignment(self, samples):
        results = {}
        ordered_samples = sorted(samples, key=lambda sample: str(getattr(sample, "user_id", "")))
        per_gnb_ue_counts = {}
        assignments = {}

        for sample in ordered_samples:
            ue, created = self.ensure_ue(sample)
            ue.update_trace_state(sample)
            ue_id = ue.ID

            # Synthetic UEs (LOW fidelity) carry their target gNB in serving_bs_id.
            # Honor that label directly instead of falling back to distance-based sorting
            # (distance is inf for all gNBs since synthetic UEs have no coordinates).
            has_position = (
                getattr(sample, "has_precise_position", True)
                and getattr(sample, "x_m", None) is not None
            )
            nominal_bs_id = getattr(sample, "serving_bs_id", None)
            nominal_gnb_id = self.bs_id_map.get(nominal_bs_id) if nominal_bs_id and nominal_bs_id != "NA" else None
            if not has_position and nominal_gnb_id:
                nominal_gnb = self.gnodeb_manager.gNodeBs.get(nominal_gnb_id)
                if nominal_gnb and getattr(nominal_gnb, "is_active", True) and \
                        per_gnb_ue_counts.get(nominal_gnb_id, 0) < self._max_ues_for_gnb(nominal_gnb_id):
                    per_gnb_ue_counts[nominal_gnb_id] = per_gnb_ue_counts.get(nominal_gnb_id, 0) + 1
                    assignments[ue_id] = nominal_gnb_id
                    results[ue_id] = {"created": created, "status": "nominal", "nominal_gnb_id": nominal_gnb_id}
                    continue
                # Nominal gNB off or full: fall through to distance-based candidates
            ranked_candidates = []
            for gnb_id, gnodeb in self.gnodeb_manager.gNodeBs.items():
                if not getattr(gnodeb, "is_active", True):
                    continue
                if per_gnb_ue_counts.get(gnb_id, 0) >= self._max_ues_for_gnb(gnb_id):
                    continue
                dist = self._distance_to_gnb(sample, gnodeb)
                if has_position and dist > self.max_coverage_distance_m:
                    continue
                ranked_candidates.append((dist, gnb_id))
            ranked_candidates.sort(key=lambda item: (item[0], item[1]))

            if not ranked_candidates:
                assignments[ue_id] = None
                results[ue_id] = {
                    "created": created,
                    "status": "outage",
                    "reason": "all active gNodeBs are full or unavailable",
                }
                continue

            selected_gnb_id = ranked_candidates[0][1]
            per_gnb_ue_counts[selected_gnb_id] = per_gnb_ue_counts.get(selected_gnb_id, 0) + 1
            assignments[ue_id] = selected_gnb_id
            results[ue_id] = {
                "created": created,
                "status": "nominal",
                "nominal_gnb_id": selected_gnb_id,
            }

        for ue_id, target_gnb_id in assignments.items():
            ue = self.ue_manager.ues[ue_id]
            if target_gnb_id is None:
                self.gnodeb_manager.mark_ue_outage(ue, results[ue_id]["reason"])
                continue
            target_sector = self.gnodeb_manager.get_best_sector_for_gnodeb(target_gnb_id)
            if not target_sector:
                reason = f"no active target sector available for {target_gnb_id}"
                self.gnodeb_manager.mark_ue_outage(ue, reason)
                results[ue_id] = {
                    **results[ue_id],
                    "status": "outage",
                    "reason": reason,
                }
                continue
            ok, message = self.gnodeb_manager.move_ue_to_sector(ue, target_sector)
            if not ok:
                self.gnodeb_manager.mark_ue_outage(ue, message)
                results[ue_id] = {
                    **results[ue_id],
                    "status": "outage",
                    "reason": message,
                }
                continue
            results[ue_id]["sector"] = target_sector.sector_id
            results[ue_id]["message"] = message

        return results

    def _apply_nearest_active_serving_policy(self, ue, sample, created=False):
        for gnb_id in self._nearest_active_gnb_ids(sample):
            target_sector = self.gnodeb_manager.get_best_sector_for_gnodeb(gnb_id)
            if not target_sector:
                continue
            if ue.gNodeB_ID == gnb_id and not getattr(ue, "in_outage", False):
                return {
                    "created": created,
                    "status": "nominal",
                    "nominal_gnb_id": gnb_id,
                    "sector": ue.ConnectedSector,
                }
            ok, message = self.gnodeb_manager.move_ue_to_sector(ue, target_sector)
            if ok:
                return {
                    "created": created,
                    "status": "nominal",
                    "nominal_gnb_id": gnb_id,
                    "sector": target_sector.sector_id,
                    "message": message,
                }
        reason = "all active gNodeBs are full or unavailable"
        self.gnodeb_manager.mark_ue_outage(ue, reason)
        return {
            "created": created,
            "status": "outage",
            "reason": reason,
        }

    def _remove_stale_trace_ue(self, ue_id):
        ue = self.ue_manager.ues.get(ue_id)
        if ue is None:
            return
        sector_id = getattr(ue, "ConnectedSector", None)
        if sector_id:
            try:
                SectorManager.get_instance().remove_ue_from_sector(sector_id, ue_id)
            except Exception:
                pass
        self.ue_manager.ues.pop(ue_id, None)
        UE.existing_ue_ids.discard(ue_id)
        UE.ue_instances.pop(ue_id, None)

    def apply_serving_policy(self, ue, sample, created=False):
        if self.association_mode == "nearest_active":
            return self._apply_nearest_active_serving_policy(ue, sample, created=created)
        nominal_gnb_id = self.bs_id_map.get(sample.serving_bs_id) if sample.serving_bs_id != "NA" else None
        if not nominal_gnb_id:
            reason = f"unmapped trace serving BS: {sample.serving_bs_id}"
            self.gnodeb_manager.mark_ue_outage(ue, reason)
            return {
                "created": created,
                "status": "unmapped_outage",
                "nominal_bs_id": sample.serving_bs_id,
                "nominal_gnb_id": None,
                "reason": reason,
            }

        nominal_sector = self.gnodeb_manager.get_best_sector_for_gnodeb(nominal_gnb_id)
        if nominal_sector:
            if ue.gNodeB_ID == nominal_gnb_id and not getattr(ue, "in_outage", False):
                return {
                    "created": created,
                    "status": "nominal",
                    "nominal_bs_id": sample.serving_bs_id,
                    "nominal_gnb_id": nominal_gnb_id,
                    "sector": ue.ConnectedSector,
                }
            ok, message = self.gnodeb_manager.move_ue_to_sector(ue, nominal_sector)
            if ok:
                return {
                    "created": created,
                    "status": "nominal",
                    "nominal_bs_id": sample.serving_bs_id,
                    "nominal_gnb_id": nominal_gnb_id,
                    "sector": nominal_sector.sector_id,
                    "message": message,
                }

        for fallback_gnb_id in self._nearest_active_gnb_ids(sample):
            fallback_sector = self.gnodeb_manager.get_best_sector_for_gnodeb(fallback_gnb_id)
            if not fallback_sector:
                continue
            ok, message = self.gnodeb_manager.move_ue_to_sector(ue, fallback_sector)
            if ok:
                return {
                    "created": created,
                    "status": "fallback",
                    "nominal_bs_id": sample.serving_bs_id,
                    "nominal_gnb_id": nominal_gnb_id,
                    "fallback_gnb_id": fallback_gnb_id,
                    "sector": fallback_sector.sector_id,
                    "message": message,
                }

        reason = f"nominal BS {sample.serving_bs_id} mapped to {nominal_gnb_id} inactive or full; no fallback within {self.max_coverage_distance_m:.0f}m"
        self.gnodeb_manager.mark_ue_outage(ue, reason)
        return {
            "created": created,
            "status": "outage",
            "nominal_bs_id": sample.serving_bs_id,
            "nominal_gnb_id": nominal_gnb_id,
            "reason": reason,
        }
