import json
import socket
import time


class RewardSocketListener:
    def __init__(self, *, port=5002, timeout_sec=None, host="0.0.0.0", backlog=1):
        self.host = host
        self.port = int(port)
        self.timeout_sec = None if timeout_sec is None else float(timeout_sec)
        self.backlog = int(backlog)

    def wait_for_payload(self, server_ready_event=None, expected_request_id=None):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self.backlog)
            if server_ready_event is not None:
                server_ready_event.set()
            deadline = None if self.timeout_sec is None else (time.time() + self.timeout_sec)
            while True:
                remaining_timeout = None
                if deadline is not None:
                    remaining_timeout = max(0.0, deadline - time.time())
                    server.settimeout(remaining_timeout)
                try:
                    connection, _address = server.accept()
                except socket.timeout as exc:
                    raise TimeoutError(
                        f"Timed out while waiting for external reward on port {self.port} after {self.timeout_sec} seconds."
                    ) from exc
                with connection:
                    if deadline is not None:
                        connection.settimeout(max(0.0, deadline - time.time()))
                    elif self.timeout_sec is not None:
                        connection.settimeout(self.timeout_sec)
                    payload = self._read_all(connection)
                parsed = self._parse_payload(payload)
                if expected_request_id is None:
                    return parsed
                if str(parsed.get("request_id")) == str(expected_request_id):
                    return parsed

    def wait_for_reward(self, server_ready_event=None):
        payload = self.wait_for_payload(server_ready_event=server_ready_event)
        return self._extract_reward(payload)

    def _read_all(self, connection):
        chunks = []
        while True:
            try:
                chunk = connection.recv(4096)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"Timed out while reading external reward on port {self.port} after {self.timeout_sec} seconds."
                ) from exc
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            raise ValueError("Received an empty reward payload.")
        return b"".join(chunks).decode("utf-8").strip()

    def _parse_payload(self, payload):
        try:
            return {"reward": float(payload)}
        except ValueError:
            pass
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed reward payload '{payload}'. Expected a numeric scalar or a JSON object containing 'reward'."
            ) from exc
        if isinstance(decoded, (int, float)):
            return {"reward": float(decoded)}
        if isinstance(decoded, dict):
            for key in ("reward", "value", "external_reward"):
                if key in decoded:
                    if decoded[key] is None:
                        break
                    normalized = dict(decoded)
                    normalized["reward"] = float(decoded[key])
                    return normalized
            if decoded.get("error"):
                raise ValueError(
                    f"External reward payload reported an error instead of a reward: {decoded['error']}"
                )
            if "reward" in decoded and decoded.get("reward") is None:
                raise ValueError(
                    f"External reward payload contains reward=null: {payload}"
                )
        raise ValueError(
            f"Unsupported reward payload '{payload}'. Expected a numeric scalar or a JSON object containing 'reward'."
        )

    def _extract_reward(self, payload):
        try:
            return float(payload["reward"])
        except Exception as exc:
            raise ValueError(f"Reward payload does not contain a usable 'reward' field: {payload!r}") from exc
