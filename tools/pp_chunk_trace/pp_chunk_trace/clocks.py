"""Low-frequency host clock calibration for offline pipeline traces.

Run one reference server before starting workers::

    python -m pp_chunk_trace.clocks serve --host 0.0.0.0 --port 29681

The returned anchor maps ``host_ns`` to ``reference_ns``.  It is a clock
estimate, not a synchronization barrier.  Call only at capture boundaries;
keep all timestamps and uncertainty in logs for offline validation.
"""

import argparse
import functools
import hashlib
import json
import math
import pathlib
import platform
import socket
import socketserver
import subprocess
import time
import uuid
from urllib.parse import urlsplit


MAX_MESSAGE_BYTES = 4096
PROTOCOL_VERSION = 1


class ClockCalibrationError(RuntimeError):
    """No trustworthy common clock sample could be obtained."""


def host_monotonic_ns():
    """Shared host timestamp; use this for recorder timestamps and anchors.

    Python < 3.10 on macOS gives monotonic_ns/perf_counter_ns a per-process
    origin.  The POSIX clock avoids falsely treating those as one host clock.
    """
    if hasattr(time, "clock_gettime_ns") and hasattr(time, "CLOCK_MONOTONIC"):
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    return time.monotonic_ns()


def _read_line(sock, timeout=None):
    data = bytearray()
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClockCalibrationError("Clock message deadline exceeded")
            sock.settimeout(remaining)
        chunk = sock.recv(min(512, MAX_MESSAGE_BYTES + 1 - len(data)))
        if not chunk:
            raise ClockCalibrationError("Clock peer closed before a complete message")
        data.extend(chunk)
        if len(data) > MAX_MESSAGE_BYTES:
            raise ClockCalibrationError("Clock message exceeds size limit")
        if b"\n" in data:
            line, remainder = bytes(data).split(b"\n", 1)
            if remainder:
                raise ClockCalibrationError("Unexpected data after clock message")
            return line


def _decode_message(line):
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ClockCalibrationError("Invalid clock JSON message") from exc
    if not isinstance(message, dict) or type(message.get("v")) is not int:
        raise ClockCalibrationError("Invalid clock message envelope")
    if message["v"] != PROTOCOL_VERSION:
        raise ClockCalibrationError("Unsupported clock protocol version")
    return message


def _encode_message(message):
    data = (json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ClockCalibrationError("Clock message exceeds size limit")
    return data


def _address_parts(address):
    if not isinstance(address, str) or not address.strip():
        raise ValueError("Clock address must be HOST:PORT or [IPv6]:PORT")
    try:
        parsed = urlsplit(address if "://" in address else "tcp://" + address)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError("Invalid clock server address") from exc
    if (
        parsed.scheme != "tcp"
        or not host
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Clock address must be HOST:PORT or [IPv6]:PORT")
    return host, port


def _read_text(path):
    return pathlib.Path(path).read_text(encoding="ascii").strip()


def _sysctl(key):
    return subprocess.run(
        ["/usr/sbin/sysctl", "-n", key],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    ).stdout.strip()


@functools.lru_cache(maxsize=1)
def _local_clock_id():
    """Identify an OS monotonic clock domain without process-local IDs."""
    system = platform.system()
    identity = None
    if system == "Linux":
        try:
            boot_id = _read_text("/proc/sys/kernel/random/boot_id")
            identity = "linux-boot:" + str(uuid.UUID(boot_id))
        except (OSError, ValueError):
            try:
                for line in _read_text("/proc/stat").splitlines():
                    if line.startswith("btime "):
                        identity = "linux-boot-time:" + platform.node() + ":" + str(
                            int(line.split()[1])
                        )
                        break
            except (OSError, ValueError, IndexError):
                pass
    elif system == "Darwin":
        try:
            identity = "darwin-boot:" + str(uuid.UUID(_sysctl("kern.bootsessionuuid")))
        except (OSError, ValueError, subprocess.SubprocessError):
            try:
                boot_time = _sysctl("kern.boottime")
                if boot_time:
                    identity = "darwin-boot-time:" + platform.node() + ":" + boot_time
            except (OSError, subprocess.SubprocessError):
                pass
    if identity is None:
        raise ClockCalibrationError(
            "Cannot identify the local host boot clock; configure a clock server"
        )
    return "local-" + hashlib.sha256(identity.encode()).hexdigest()[:32]


def _sample_to_anchor(t0, t1, t2, t3, clock_id):
    if any(type(value) is not int or value < 0 for value in (t0, t1, t2, t3)):
        raise ClockCalibrationError("Clock timestamps must be nonnegative integers")
    if not isinstance(clock_id, str) or not 1 <= len(clock_id) <= 128:
        raise ClockCalibrationError("Invalid reference clock identity")
    rtt = t3 - t0
    processing = t2 - t1
    if rtt < 0 or processing < 0 or processing > rtt:
        raise ClockCalibrationError("Inconsistent clock timestamp ordering")
    return {
        "host_ns": (t0 + t3) // 2,
        "reference_ns": (t1 + t2) // 2,
        # Conservative bound: retain processing time in the budget rather
        # than claim a precise one-way delay from an asymmetric TCP path.
        "uncertainty_ns": (rtt + 1) // 2,
        "clock_id": clock_id,
        "rtt_ns": rtt,
        "server_processing_ns": processing,
        "network_delay_ns": rtt - processing,
    }


def _sample_once(address, timeout):
    request = _encode_message({"v": PROTOCOL_VERSION, "op": "sample"})
    with socket.create_connection(address, timeout=timeout) as sock:
        sock.settimeout(timeout)
        t0 = host_monotonic_ns()
        sock.sendall(request)
        raw = _read_line(sock, timeout=timeout)
        t3 = host_monotonic_ns()
    response = _decode_message(raw)
    if response.get("op") != "sample":
        raise ClockCalibrationError("Unexpected clock response")
    return _sample_to_anchor(
        t0, response.get("recv_ns"), response.get("send_ns"), t3,
        response.get("clock_id"),
    )


def sample_host_clock(address=None, samples=5, timeout=2.0):
    """Return the lowest-RTT host/reference anchor from a bounded sample set.

    ``address=None`` uses the local OS boot's shared monotonic domain.
    Distinct hosts require the same reference server.  Failures never silently
    fall back to a different domain.  Restarting the server changes clock_id.
    """
    if type(samples) is not int or not 1 <= samples <= 100:
        raise ValueError("samples must be an integer from 1 to 100")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 30
    ):
        raise ValueError("timeout must be finite and in (0, 30] seconds")
    if address is None:
        clock_id = _local_clock_id()
        now = host_monotonic_ns()
        return {
            "host_ns": now,
            "reference_ns": now,
            "uncertainty_ns": 0,
            "clock_id": clock_id,
            "rtt_ns": 0,
            "server_processing_ns": 0,
            "network_delay_ns": 0,
        }
    endpoint = _address_parts(address)
    results = []
    try:
        for _ in range(samples):
            result = _sample_once(endpoint, timeout)
            if results and result["clock_id"] != results[0]["clock_id"]:
                raise ClockCalibrationError("Reference clock changed during calibration")
            results.append(result)
    except (OSError, TimeoutError) as exc:
        raise ClockCalibrationError("Clock server sample failed: " + str(exc)) from exc
    return min(results, key=lambda item: item["rtt_ns"])


def _handle_sample(sock, clock_id):
    raw = _read_line(sock, timeout=2.0)
    received = host_monotonic_ns()
    request = _decode_message(raw)
    if set(request) != {"v", "op"} or request.get("op") != "sample":
        raise ClockCalibrationError("Unsupported clock request")
    sent = host_monotonic_ns()
    sock.sendall(_encode_message({
        "v": PROTOCOL_VERSION,
        "op": "sample",
        "clock_id": clock_id,
        "recv_ns": received,
        "send_ns": sent,
    }))


class _ClockHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(2.0)
        try:
            _handle_sample(self.request, self.server.clock_id)
        except (OSError, ClockCalibrationError):
            # A malformed/slow peer affects only this connection; no evaluation
            # or filesystem operations are selected by protocol data.
            pass


class _ClockServer(socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, bind_and_activate=True):
        self.clock_id = "reference-" + uuid.uuid4().hex
        super().__init__(address, _ClockHandler, bind_and_activate=bind_and_activate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="serve a shared monotonic clock")
    serve.add_argument("--host", required=True)
    serve.add_argument("--port", type=int, default=29681)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    server_type = _ClockServer
    if ":" in args.host:
        class IPv6ClockServer(_ClockServer):
            address_family = socket.AF_INET6
        server_type = IPv6ClockServer
    with server_type((args.host, args.port)) as server:
        print(json.dumps({"listen": list(server.server_address),
                          "clock_id": server.clock_id}), flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
