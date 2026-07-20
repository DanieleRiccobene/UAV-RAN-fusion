import os

from Config_files.mobility_config import (
    DEFAULT_MOBILITY_PROFILE,
    resolve_gnb_positions_csv,
    resolve_mobility_trace_csv,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MOBILITY_PROFILE = DEFAULT_MOBILITY_PROFILE
DEFAULT_MOBILITY_CSV = resolve_mobility_trace_csv(DEFAULT_MOBILITY_PROFILE, REPO_ROOT)
DEFAULT_GNB_POSITIONS_CSV = resolve_gnb_positions_csv(REPO_ROOT)
DEFAULT_STATIC_CSV = None
