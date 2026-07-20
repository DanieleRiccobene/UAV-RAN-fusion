import json
import os


DEFAULT_MAPPING_FILENAME = "bs_gnb_mapping.json"


def load_bs_gnb_mapping(base_dir, mapping_path=None):
    """Load an explicit trace-BS to simulator-gNB mapping from JSON."""
    if mapping_path is None:
        mapping_path = os.path.join(base_dir, "Config_files", DEFAULT_MAPPING_FILENAME)
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"BS-to-gNB mapping file not found: {mapping_path}. "
            f"Create Config_files/{DEFAULT_MAPPING_FILENAME} or pass bs_id_map explicitly."
        )
    with open(mapping_path, "r") as handle:
        data = json.load(handle)
    mapping = data.get("bs_to_gnb", data)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"BS-to-gNB mapping file {mapping_path} must contain a non-empty 'bs_to_gnb' object")
    return {str(bs_id): str(gnb_id) for bs_id, gnb_id in mapping.items()}


def validate_bs_gnb_mapping(trace_bs_ids, gnodeb_ids, bs_id_map, *, strict=True):
    """Validate trace serving BS coverage and mapped simulator gNB IDs."""
    normalized_trace_bs_ids = {str(bs_id) for bs_id in trace_bs_ids if str(bs_id).strip() and str(bs_id) != "NA"}
    mapped_bs_ids = set(bs_id_map)
    valid_gnodeb_ids = {str(gnb_id) for gnb_id in gnodeb_ids}

    missing_bs_ids = sorted(normalized_trace_bs_ids - mapped_bs_ids)
    invalid_targets = {
        bs_id: gnb_id
        for bs_id, gnb_id in sorted(bs_id_map.items())
        if str(gnb_id) not in valid_gnodeb_ids
    }

    if invalid_targets:
        raise ValueError(
            "BS-to-gNB mapping contains target gNB IDs that are not initialized: "
            f"{invalid_targets}. Valid gNB IDs: {sorted(valid_gnodeb_ids)}"
        )
    if strict and missing_bs_ids:
        raise ValueError(
            "BS-to-gNB mapping is missing serving_bs_id values from the mobility trace: "
            f"{missing_bs_ids}. Add them to Config_files/{DEFAULT_MAPPING_FILENAME} "
            "or run with strict_bs_mapping=False to mark unmapped BSs as outage."
        )

    return {
        "missing_bs_ids": missing_bs_ids,
        "invalid_targets": invalid_targets,
        "mapped_bs_count": len(mapped_bs_ids),
        "trace_bs_count": len(normalized_trace_bs_ids),
    }
