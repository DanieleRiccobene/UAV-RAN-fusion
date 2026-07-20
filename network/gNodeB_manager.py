###################################################################################################################
# gNodeBManager.py inside the network folder .#The gNodeBManager class is responsible for managing gNodeB (gNB)   #
# instances within the network simulation. It provides functionalities to create, update, delete, and manage gNBs #
# across the network. This class follows the Singleton design pattern to ensure that only one instance of the     #
# gNodeBManager exists throughout the application lifecycle.                                                      #
###################################################################################################################
import os
from network.gNodeB import gNodeB, load_gNodeB_config
from database.database_manager import DatabaseManager
from logs.logger_config import cell_logger, gnodeb_logger
from network.utils import calculate_distance
import threading

VALID_SLEEP_BEHAVIORS = {"outage", "reattach"}

class gNodeBManager:
    _instance = None
    _lock = threading.Lock()
    _call_count = 0  # Add a class variable to count calls

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(gNodeBManager, cls).__new__(cls)
            # Initialize the object here, if necessary
        return cls._instance
    
    @classmethod
    def get_instance(cls, base_dir=None, sleep_behavior=None):
        cls._call_count += 1
        gnodeb_logger.debug(f"gNodeBManager get_instance called {cls._call_count} times.")
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(base_dir, sleep_behavior=sleep_behavior or "outage")
            elif sleep_behavior is not None:
                cls._instance.set_sleep_behavior(sleep_behavior)
        return cls._instance
    
    def __init__(self, base_dir, sleep_behavior="outage"):
        if not hasattr(self, 'initialized'):
            self.gNodeBs = {}
            self.db_manager = DatabaseManager.get_instance()
            self.base_dir = base_dir
            self.runtime_lock = threading.RLock()
            self.sleep_behavior = self._normalize_sleep_behavior(sleep_behavior)
            self.gNodeBs_config = load_gNodeB_config()

            # Check if gNodeBs_config contains gNodeBs data
            if 'gNodeBs' not in self.gNodeBs_config or not self.gNodeBs_config['gNodeBs']:
                gnodeb_logger.error("gNodeBs configuration is missing or empty.")
                raise ValueError("gNodeBs configuration is missing or empty.")
            self.initialized = True

    @staticmethod
    def _normalize_sleep_behavior(sleep_behavior):
        if sleep_behavior not in VALID_SLEEP_BEHAVIORS:
            raise ValueError(f"Unsupported sleep_behavior '{sleep_behavior}'. Expected one of {sorted(VALID_SLEEP_BEHAVIORS)}.")
        return sleep_behavior

    def set_sleep_behavior(self, sleep_behavior):
        self.sleep_behavior = self._normalize_sleep_behavior(sleep_behavior)
        gnodeb_logger.info(f"gNodeB sleep behavior set to {self.sleep_behavior}.")

    def initialize_gNodeBs(self):
        """
        Initialize gNodeBs based on the loaded configuration and insert them into the database.
        """
        for gNodeB_data in self.gNodeBs_config['gNodeBs']:
            if gNodeB_data['gnodeb_id'] in self.gNodeBs:
                raise ValueError(f"Duplicate gNodeB ID {gNodeB_data['gnodeb_id']} found during initialization.")
            
            gnodeb = gNodeB(**gNodeB_data)
            self.gNodeBs[gnodeb.ID] = gnodeb
            point = gnodeb.serialize_for_influxdb()  # Serialize for InfluxDB
            self.db_manager.insert_data(point)  # Insert the Point object directly
        return self.gNodeBs

    def turn_off(self, gnodeb_id, sleep_behavior=None):
        """
        Put a gNodeB into simulator sleep mode without deleting runtime objects.
        sleep_behavior='outage' keeps served UEs on the old association with zero service.
        sleep_behavior='reattach' tries deterministic reassignment before falling back to outage.
        """
        behavior = self._normalize_sleep_behavior(sleep_behavior or self.sleep_behavior)
        with self.runtime_lock:
            gnodeb = self.get_gNodeB(gnodeb_id)
            if not gnodeb:
                return False, f"gNodeB {gnodeb_id} not found"

            affected_ues = self._get_served_ues(gnodeb)
            self._set_gnodeb_power_state(gnodeb, is_active=False)

            results = {}
            if behavior == "reattach":
                for ue in affected_ues:
                    success, message = self._reattach_ue(ue, source_gnodeb_id=gnodeb.ID)
                    results[ue.ID] = {"reattached": success, "message": message}
                    if not success:
                        self._mark_ue_outage(ue, reason=message)
            else:
                for ue in affected_ues:
                    self._mark_ue_outage(ue, reason=f"serving gNodeB {gnodeb.ID} is OFF")
                    results[ue.ID] = {"reattached": False, "message": "outage mode"}

            self._serialize_power_state(gnodeb)
            return True, {
                "gnodeb_id": gnodeb.ID,
                "sleep_behavior": behavior,
                "affected_ues": results,
            }

    def turn_on(self, gnodeb_id):
        """Wake a sleeping gNodeB and its child cells/sectors without recreating objects."""
        with self.runtime_lock:
            gnodeb = self.get_gNodeB(gnodeb_id)
            if not gnodeb:
                return False, f"gNodeB {gnodeb_id} not found"

            self._set_gnodeb_power_state(gnodeb, is_active=True)
            self._restore_served_ues(gnodeb)
            self._serialize_power_state(gnodeb)
            return True, self.get_power_state(gnodeb_id)

    def get_power_state(self, gnodeb_id):
        gnodeb = self.get_gNodeB(gnodeb_id)
        if not gnodeb:
            return None
        return {
            "gnodeb_id": gnodeb.ID,
            "is_active": bool(getattr(gnodeb, "is_active", True)),
            "power_state": getattr(gnodeb, "power_state", "ON"),
            "transmission_power": getattr(gnodeb, "TransmissionPower", None),
            "sleep_behavior": self.sleep_behavior,
        }

    def _set_gnodeb_power_state(self, gnodeb, is_active):
        gnodeb.is_active = is_active
        gnodeb.power_state = "ON" if is_active else "OFF"
        gnodeb.TransmissionPower = gnodeb.NominalTransmissionPower if is_active else 0

        for cell in gnodeb.Cells:
            if not hasattr(cell, "NominalTxPower"):
                cell.NominalTxPower = cell.TxPower
            cell.IsActive = is_active
            cell.TxPower = cell.NominalTxPower if is_active else 0
            for sector in cell.sectors:
                if not hasattr(sector, "nominal_tx_power"):
                    sector.nominal_tx_power = sector.tx_power
                sector.is_active = is_active
                sector.tx_power = sector.nominal_tx_power if is_active else 0

    def _serialize_power_state(self, gnodeb):
        try:
            self.db_manager.insert_data(gnodeb.serialize_for_influxdb())
            for cell in gnodeb.Cells:
                self.db_manager.insert_data(cell.serialize_for_influxdb(cell.cell_load))
                for sector in cell.sectors:
                    self.db_manager.insert_data(sector.serialize_for_influxdb())
        except Exception as exc:
            gnodeb_logger.error(f"Failed to serialize power state for gNodeB {gnodeb.ID}: {exc}")

    def _get_served_ues(self, gnodeb):
        served_ues = []
        seen = set()
        for cell in gnodeb.Cells:
            for sector in cell.sectors:
                for ue in list(sector.ues.values()):
                    if ue.ID not in seen:
                        served_ues.append(ue)
                        seen.add(ue.ID)
        return served_ues

    def _is_sector_feasible(self, sector, source_gnodeb_id=None):
        target_gnodeb = getattr(sector.cell, "gNodeB", None)
        if not target_gnodeb:
            return False
        if source_gnodeb_id and target_gnodeb.ID == source_gnodeb_id:
            return False
        if not getattr(target_gnodeb, "is_active", True):
            return False
        if not getattr(sector.cell, "IsActive", True):
            return False
        if not getattr(sector, "is_active", True):
            return False
        if sector.remaining_capacity <= 0:
            return False
        return True

    def _candidate_sectors(self, source_gnodeb_id=None):
        candidates = []
        for gnodeb in self.gNodeBs.values():
            for cell in gnodeb.Cells:
                for sector in cell.sectors:
                    if self._is_sector_feasible(sector, source_gnodeb_id=source_gnodeb_id):
                        candidates.append(sector)
        return sorted(
            candidates,
            key=lambda sector: (
                -sector.remaining_capacity,
                len(sector.connected_ues),
                str(sector.cell.gNodeB.ID),
                str(sector.cell.ID),
                str(sector.sector_id),
            ),
        )

    def get_best_sector_for_gnodeb(self, gnodeb_id):
        gnodeb = self.get_gNodeB(gnodeb_id)
        if not gnodeb or not getattr(gnodeb, "is_active", True):
            return None
        candidates = []
        for cell in gnodeb.Cells:
            for sector in cell.sectors:
                if self._is_sector_feasible(sector):
                    candidates.append(sector)
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda sector: (
                -sector.remaining_capacity,
                len(sector.connected_ues),
                str(sector.cell.ID),
                str(sector.sector_id),
            ),
        )[0]

    def get_best_active_sector(self):
        candidates = self._candidate_sectors(source_gnodeb_id=None)
        return candidates[0] if candidates else None

    def move_ue_to_sector(self, ue, target_sector):
        old_sector = self._find_sector_by_ue(ue)
        if not target_sector:
            return False, "target sector not found"
        if not self._move_ue_to_sector(ue, old_sector, target_sector):
            return False, f"failed to move UE to sector {target_sector.sector_id}"
        return True, f"moved to gNodeB {target_sector.cell.gNodeB.ID}, cell {target_sector.cell.ID}, sector {target_sector.sector_id}"

    def mark_ue_outage(self, ue, reason):
        self._mark_ue_outage(ue, reason)

    def _reattach_ue(self, ue, source_gnodeb_id):
        old_sector = self._find_sector_by_ue(ue)
        candidates = self._candidate_sectors(source_gnodeb_id=source_gnodeb_id)
        if not candidates:
            return False, "no active target sector with free capacity"

        target_sector = candidates[0]
        if not self._move_ue_to_sector(ue, old_sector, target_sector):
            return False, f"failed to move UE to sector {target_sector.sector_id}"
        return True, f"reattached to gNodeB {target_sector.cell.gNodeB.ID}, cell {target_sector.cell.ID}, sector {target_sector.sector_id}"

    def _find_sector_by_ue(self, ue):
        for gnodeb in self.gNodeBs.values():
            for cell in gnodeb.Cells:
                for sector in cell.sectors:
                    if ue.ID in sector.connected_ues or ue.ID in sector.ues:
                        return sector
        return None

    def _move_ue_to_sector(self, ue, old_sector, target_sector):
        if not self._is_sector_feasible(target_sector):
            return False

        if old_sector:
            if ue.ID in old_sector.connected_ues:
                old_sector.connected_ues.remove(ue.ID)
            old_sector.ues.pop(ue.ID, None)
            old_sector.current_load = max(0, old_sector.current_load - 1)
            old_sector.remaining_capacity = old_sector.capacity - len(old_sector.connected_ues)
            old_sector.cell.update_ue_lists()

        target_sector.connected_ues.append(ue.ID)
        target_sector.ues[ue.ID] = ue
        target_sector.current_load += 1
        target_sector.remaining_capacity = target_sector.capacity - len(target_sector.connected_ues)
        target_sector.cell.update_ue_lists()

        ue.ConnectedSector = target_sector.sector_id
        ue.connected_sector = target_sector.sector_id
        ue.ConnectedCellID = target_sector.cell.ID
        ue.gNodeB_ID = target_sector.cell.gNodeB.ID
        ue.is_connected = True
        ue.in_outage = False
        ue.outage_reason = None
        self._serialize_reassignment(ue, old_sector, target_sector)
        return True

    def _mark_ue_outage(self, ue, reason):
        ue.is_connected = False
        ue.in_outage = True
        ue.outage_reason = reason
        ue.throughput = 0.0
        ue.ue_packet_loss_rate = 1.0
        gnodeb_logger.debug(f"UE {ue.ID} in outage: {reason}")

    def _restore_served_ues(self, gnodeb):
        for ue in self._get_served_ues(gnodeb):
            if getattr(ue, "in_outage", False):
                ue.is_connected = True
                ue.in_outage = False
                ue.outage_reason = None
                ue.ue_packet_loss_rate = 0.0

    def _serialize_reassignment(self, ue, old_sector, target_sector):
        try:
            if old_sector:
                self.db_manager.insert_data(old_sector.serialize_for_influxdb())
                self.db_manager.insert_data(old_sector.cell.serialize_for_influxdb(old_sector.cell.cell_load))
            self.db_manager.insert_data(target_sector.serialize_for_influxdb())
            self.db_manager.insert_data(target_sector.cell.serialize_for_influxdb(target_sector.cell.cell_load))
            self.db_manager.insert_data(ue.serialize_for_influxdb())
        except Exception as exc:
            gnodeb_logger.error(f"Failed to serialize reassignment for UE {ue.ID}: {exc}")
    
    def list_all_gNodeBs_detailed(self):
        """List all gNodeBs managed by this manager with detailed information."""
        gNodeBs_detailed_list = []
        for gnodeb_id, gnodeb in self.gNodeBs.items():
            gNodeBs_detailed_list.append({
                'id': gnodeb.ID,
                'latitude': gnodeb.Latitude,
                'longitude': gnodeb.Longitude,
                'coverage_radius': gnodeb.CoverageRadius,
                'transmission_power': gnodeb.TransmissionPower,
                'frequency': gnodeb.Frequency,
                'bandwidth': gnodeb.Bandwidth,
                'max_ues': gnodeb.MaxUEs,
                'cell_count': gnodeb.CellCount,
                'sector_count': gnodeb.SectorCount,
                'is_active': bool(getattr(gnodeb, 'is_active', True)),
                'power_state': getattr(gnodeb, 'power_state', 'ON'),
                # Add more fields as needed
            })
        return gNodeBs_detailed_list

    def add_gNodeB(self, gNodeB_data):
        """
        Add a single gNodeB instance to the manager and the database.
        
        :param gNodeB_data: Dictionary containing the data for the gNodeB to be added.
        """
        if gNodeB_data['gnodeb_id'] in self.gNodeBs:
            raise ValueError(f"Duplicate gNodeB ID {gNodeB_data['gnodeb_id']} found.")
        
        gnodeb = gNodeB(**gNodeB_data)
        self.gNodeBs[gnodeb.ID] = gnodeb
        point = gnodeb.serialize_for_influxdb()
        self.db_manager.insert_data(point)

    def remove_gNodeB(self, gnodeb_id):
        """
        Remove a gNodeB instance from the manager and the database.
        
        :param gnodeb_id: ID of the gNodeB to be removed.
        """
        if gnodeb_id in self.gNodeBs:
            del self.gNodeBs[gnodeb_id]
            # Assuming there's a method in DBManager to remove data
            self.db_manager.remove_data(gnodeb_id)
        else:
            print(f"gNodeB ID {gnodeb_id} not found.")

    def get_gNodeB(self, gnodeb_id):
        """
        Retrieve a gNodeB instance by its ID.
        
        :param gnodeb_id: ID of the gNodeB to retrieve.
        :return: The gNodeB instance, if found; None otherwise.
        """
        return self.gNodeBs.get(gnodeb_id)
    
        
    def get_neighbor_gNodeBs(self, gnodeb_id):
        """
        Find neighboring gNodeBs based on coverage radius overlap.

        :param gnodeb_id: ID of the gNodeB to find neighbors for.
        :return: A list of gNodeB IDs that are neighbors based on coverage overlap.
        """
        target_gNodeB = self.get_gNodeB(gnodeb_id)
        if not target_gNodeB:
            return []  # Target gNodeB not found

        neighbors = []
        for gnb_id, gnb in self.gNodeBs.items():
            if gnb_id != gnodeb_id:  # Don't include the target gNodeB itself
                distance = calculate_distance(target_gNodeB.Latitude, target_gNodeB.Longitude, gnb.Latitude, gnb.Longitude)
                # Check if the distance is less than the sum of their coverage radii
                if distance <= (target_gNodeB.CoverageRadius + gnb.CoverageRadius) / 1000:  # Convert meters to kilometers
                    neighbors.append(gnb_id)

        return neighbors
