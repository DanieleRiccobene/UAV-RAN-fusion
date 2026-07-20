import logging
from datetime import datetime, timezone
from time import monotonic as default_time
from zoneinfo import ZoneInfo

try:
    import ntplib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ntplib = None


logger = logging.getLogger(__name__)

CACHED_TIME = None
CACHE_EXPIRY = 30 * 60
RETRY_LIMIT = 3
DEFAULT_TIMEZONE = ZoneInfo("US/Eastern")

server_pools = [
    "129.6.15.28",
    "132.163.96.3",
    "132.163.96.1",
    "216.239.35.0",
    "216.239.35.4",
    "129.6.15.30",
    "64.6.64.6",
    "38.229.71.1",
    "194.153.171.2",
]


def is_time_valid(time_to_check):
    return time_to_check is not None


def _format_eastern(dt):
    return dt.astimezone(DEFAULT_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _system_time_string():
    return datetime.now(DEFAULT_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def get_current_time_ntp(cache_expiry=CACHE_EXPIRY):
    global CACHED_TIME

    if CACHED_TIME and isinstance(CACHED_TIME, tuple) and isinstance(CACHED_TIME[1], float):
        if (default_time() - CACHED_TIME[1]) < cache_expiry:
            return CACHED_TIME[0]

    if ntplib is not None:
        ntp_client = ntplib.NTPClient()
        for server in server_pools:
            for _ in range(RETRY_LIMIT):
                try:
                    response = ntp_client.request(server, timeout=5, version=3)
                    utc_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
                    formatted_time = _format_eastern(utc_time)

                    if is_time_valid(formatted_time):
                        CACHED_TIME = (formatted_time, default_time())
                        return formatted_time
                except Exception as exc:  # pragma: no cover - network failure path
                    logger.debug("Attempt to fetch time from NTP server '%s' failed: %s", server, exc)
                    continue
        logger.warning("All NTP pools failed; falling back to system time")
    else:
        logger.info("ntplib not available; using system time fallback")

    formatted_time = _system_time_string()
    CACHED_TIME = (formatted_time, default_time())
    return formatted_time


if __name__ == "__main__":
    current_time = get_current_time_ntp()
    print(f"Current Time: {current_time}")
