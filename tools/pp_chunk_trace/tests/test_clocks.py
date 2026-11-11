import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import unittest
from unittest import mock

from pp_chunk_trace import clocks


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.sent = []

    def recv(self, limit):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > limit:
            self.chunks.insert(0, chunk[limit:])
        return chunk[:limit]

    def sendall(self, data):
        self.sent.append(data)

    def settimeout(self, timeout):
        self.timeout = timeout


class ClockTests(unittest.TestCase):
    def tearDown(self):
        clocks._local_clock_id.cache_clear()

    def test_shared_posix_clock_not_process_relative_python_clock(self):
        with mock.patch.object(clocks.time, "clock_gettime_ns", return_value=1234567), \
             mock.patch.object(clocks.time, "monotonic_ns", return_value=10):
            self.assertEqual(clocks.host_monotonic_ns(), 1234567)

    def test_host_clock_matches_independent_process(self):
        before = clocks.host_monotonic_ns()
        child = int(subprocess.check_output(
            [sys.executable, "-B", "-c",
             "from pp_chunk_trace.clocks import host_monotonic_ns; print(host_monotonic_ns())"],
            cwd=str(Path(clocks.__file__).resolve().parent.parent), timeout=5,
            text=True,
        ))
        after = clocks.host_monotonic_ns()
        self.assertLessEqual(before, child)
        self.assertLessEqual(child, after)

    def test_local_anchor_never_uses_wall_time(self):
        with mock.patch.object(clocks, "_local_clock_id", return_value="local-test"), \
             mock.patch.object(clocks, "host_monotonic_ns", return_value=123), \
             mock.patch.object(clocks.time, "time", side_effect=AssertionError("wall clock")), \
             mock.patch.object(clocks.time, "time_ns", side_effect=AssertionError("wall clock")):
            anchor = clocks.sample_host_clock()
        self.assertEqual((anchor["host_ns"], anchor["reference_ns"]), (123, 123))

    def test_four_timestamp_arithmetic_and_conservative_budget(self):
        # Host 100..150; server offset +1000, forward 10, processing 20,
        # return 20. Midpoint offset is +995; true +1000 lies in budget.
        result = clocks._sample_to_anchor(100, 1110, 1130, 150, "reference-a")
        self.assertEqual(result["host_ns"], 125)
        self.assertEqual(result["reference_ns"], 1120)
        self.assertEqual(result["uncertainty_ns"], 25)
        self.assertEqual(result["network_delay_ns"], 30)
        self.assertEqual(result["server_processing_ns"], 20)

    def test_round_up_odd_rtt(self):
        self.assertEqual(clocks._sample_to_anchor(0, 100, 100, 3, "x")[
            "uncertainty_ns"], 2)

    def test_invalid_timestamps(self):
        for values in ((10, 20, 21, 9), (0, 20, 19, 10),
                       (0, 20, 40, 10), (True, 1, 1, 2), (0, None, 1, 2)):
            with self.subTest(values=values), self.assertRaises(clocks.ClockCalibrationError):
                clocks._sample_to_anchor(*values, "x")

    def test_reference_identity_validation(self):
        for value in (None, "", "a" * 129, 7):
            with self.subTest(value=value), self.assertRaises(clocks.ClockCalibrationError):
                clocks._sample_to_anchor(0, 100, 100, 2, value)

    def test_min_rtt_sample_selected(self):
        samples = [clocks._sample_to_anchor(0, 100, 101, n, "x")
                   for n in (10, 4, 8)]
        with mock.patch.object(clocks, "_sample_once", side_effect=samples) as sample:
            result = clocks.sample_host_clock("localhost:1234", samples=3)
        self.assertEqual(result, samples[1])
        self.assertEqual(sample.call_count, 3)

    def test_server_restart_during_sampling_rejected(self):
        samples = [clocks._sample_to_anchor(0, 100, 101, 10, x) for x in ("a", "b")]
        with mock.patch.object(clocks, "_sample_once", side_effect=samples):
            with self.assertRaisesRegex(clocks.ClockCalibrationError, "changed"):
                clocks.sample_host_clock("localhost:1234", samples=2)

    def test_network_failure_does_not_fall_back(self):
        with mock.patch.object(clocks, "_sample_once", side_effect=socket.timeout("timeout")):
            with self.assertRaises(clocks.ClockCalibrationError):
                clocks.sample_host_clock("localhost:1234")

    def test_local_clock_uses_one_shared_domain(self):
        boot_id = "26bb47f1-280f-46ed-8b32-b6d727f1a063"
        with mock.patch.object(clocks.platform, "system", return_value="Linux"), \
             mock.patch.object(clocks, "_read_text", return_value=boot_id), \
             mock.patch.object(clocks, "host_monotonic_ns", side_effect=[123, 456]):
            first, second = clocks.sample_host_clock(), clocks.sample_host_clock()
        self.assertEqual(first["clock_id"], second["clock_id"])
        self.assertEqual(first["host_ns"], first["reference_ns"])
        self.assertEqual(second["host_ns"], 456)
        self.assertEqual(first["uncertainty_ns"], 0)

    def test_boot_change_changes_local_domain(self):
        with mock.patch.object(clocks.platform, "system", return_value="Linux"), \
             mock.patch.object(clocks, "_read_text", side_effect=[
                 "26bb47f1-280f-46ed-8b32-b6d727f1a063",
                 "26bb47f1-280f-46ed-8b32-b6d727f1a064"]):
            first = clocks._local_clock_id()
            clocks._local_clock_id.cache_clear()
            self.assertNotEqual(first, clocks._local_clock_id())

    def test_mac_fallback_uses_host_and_boot(self):
        with mock.patch.object(clocks.platform, "system", return_value="Darwin"), \
             mock.patch.object(clocks.platform, "node", return_value="mac-a"), \
             mock.patch.object(clocks, "_sysctl", side_effect=[ValueError(), "{ sec = 123; }"]):
            first = clocks._local_clock_id()
        clocks._local_clock_id.cache_clear()
        with mock.patch.object(clocks.platform, "system", return_value="Darwin"), \
             mock.patch.object(clocks.platform, "node", return_value="mac-b"), \
             mock.patch.object(clocks, "_sysctl", side_effect=[ValueError(), "{ sec = 123; }"]):
            self.assertNotEqual(first, clocks._local_clock_id())

    def test_unknown_local_clock_does_not_invent_process_identity(self):
        with mock.patch.object(clocks.platform, "system", return_value="Unknown"):
            with self.assertRaises(clocks.ClockCalibrationError):
                clocks.sample_host_clock()

    def test_fragmented_protocol_request(self):
        sock = FakeSocket([b'{"v":1,', b'"op":"sample"}', b'\n'])
        with mock.patch.object(clocks, "host_monotonic_ns", side_effect=[10, 12]):
            clocks._handle_sample(sock, "reference-test")
        response = json.loads(sock.sent[0])
        self.assertEqual(response["recv_ns"], 10)
        self.assertEqual(response["send_ns"], 12)
        self.assertEqual(response["clock_id"], "reference-test")

    def test_bounded_and_corrupt_messages(self):
        for payload in (b"x" * (clocks.MAX_MESSAGE_BYTES + 1), b"{}\n{}\n", b"partial"):
            with self.subTest(payload=payload[:20]), self.assertRaises(clocks.ClockCalibrationError):
                clocks._read_line(FakeSocket([payload]))
        for payload in (b"not-json", b"[]", b'{"v":true}', b'{"v":2}', b"\xff"):
            with self.subTest(payload=payload), self.assertRaises(clocks.ClockCalibrationError):
                clocks._decode_message(payload)

    def test_total_message_deadline(self):
        sock = FakeSocket([b'{"v":', b'1}\n'])
        with mock.patch.object(clocks.time, "monotonic", side_effect=[0.0, 0.1, 2.1]):
            with self.assertRaisesRegex(clocks.ClockCalibrationError, "deadline"):
                clocks._read_line(sock, timeout=2.0)

    def test_server_rejects_unknown_operation_and_fields(self):
        for message in ({"v": 1, "op": "execute"},
                        {"v": 1, "op": "sample", "extra": "x"}):
            sock = FakeSocket([clocks._encode_message(message)])
            with self.assertRaises(clocks.ClockCalibrationError):
                clocks._handle_sample(sock, "x")
            self.assertEqual(sock.sent, [])

    def test_address_and_argument_validation(self):
        self.assertEqual(clocks._address_parts("tcp://server:29681"), ("server", 29681))
        self.assertEqual(clocks._address_parts("[::1]:29681"), ("::1", 29681))
        for address in ("", "host", "http://host:1", "user:pw@host:1", "host:0", "host:1/x"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                clocks._address_parts(address)
        for kwargs in ({"samples": 0}, {"samples": True}, {"samples": 101},
                       {"timeout": 0}, {"timeout": float("nan")}, {"timeout": 31}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                clocks.sample_host_clock(**kwargs)

    def test_server_lifetime_id_is_unique(self):
        with clocks._ClockServer(("127.0.0.1", 0), bind_and_activate=False) as a, \
             clocks._ClockServer(("127.0.0.1", 0), bind_and_activate=False) as b:
            self.assertNotEqual(a.clock_id, b.clock_id)

    def test_local_tcp_round_trip(self):
        try:
            server = clocks._ClockServer(("127.0.0.1", 0))
        except OSError as exc:
            self.skipTest("Local socket unavailable: " + str(exc))
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            anchor = clocks.sample_host_clock("127.0.0.1:" + str(server.server_address[1]), samples=3)
            self.assertEqual(anchor["clock_id"], server.clock_id)
            self.assertLessEqual(abs(anchor["reference_ns"] - anchor["host_ns"]),
                                 anchor["uncertainty_ns"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
