"""CPU-only contract tests for the Qwen3.5 two-worker pipeline."""

import array
import json
import socket
import struct
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.layers.qwen3_5 import pipeline

MODEL_DIR = Path("/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")
REFERENCE_PYTHON = Path("/home/yaozhenyang/downloads/yes/envs/minisglang/bin/python")


class _ScriptedWorker:
    """CPU-only stand-in which records the controller-side worker contract."""

    def __init__(
        self,
        role,
        *,
        tokens=(17, 18, 19),
        fail_command=None,
        validation_route_delta=None,
    ):
        self.role = role
        self.tokens = iter(tokens)
        self.fail_command = fail_command
        self.validation_route_delta = validation_route_delta
        self.calls = []
        self.closed = False
        self.ready = None
        self.last_response = None
        self.consumed_len = 0
        self._replay_last_sample = False
        self._last_sample_token = None

    @staticmethod
    def _hidden(rows):
        return bytes(rows * 2048 * 2)

    def _router_capture(self, command):
        layer_ids = (
            list(pipeline.FRONT_LAYER_IDS)
            if self.role == "front"
            else list(pipeline.BACK_LAYER_IDS)
        )
        expert_ids = [
            [(layer_id + slot) % 256 for slot in range(pipeline.ROUTER_TOP_K)]
            for layer_id in layer_ids
        ]
        probabilities = [[0.125] * pipeline.ROUTER_TOP_K for _ in layer_ids]
        if (
            command.startswith("VALIDATE")
            and self.validation_route_delta
            and self.validation_route_delta[0] in layer_ids
        ):
            layer_id, slot, expert_id, probability = self.validation_route_delta
            row = layer_ids.index(layer_id)
            expert_ids[row][slot] = expert_id
            probabilities[row][slot] = probability
            # Keep the fake response normalized while making the probability
            # discrepancy independently observable by the controller.
            probabilities[row][(slot + 1) % pipeline.ROUTER_TOP_K] = (
                1.0 - probability - 0.125 * (pipeline.ROUTER_TOP_K - 2)
            )
        return {
            pipeline.ROUTER_CAPTURE_KEY: {
                "layer_ids": layer_ids,
                "expert_ids": expert_ids,
                "probabilities": probabilities,
            }
        }

    def request(self, header, payload=b""):
        self.calls.append((dict(header), payload))
        command = header["command"]
        if command == self.fail_command:
            raise pipeline.PipelineWorkerError(
                f"injected {self.role} {command} failure"
            )
        if command == "BEGIN":
            self.consumed_len = 0
            response, body = {
                "kind": "OK",
                "role": self.role,
                "epoch": header["epoch"],
                "step_id": -1,
                "consumed_len": 0,
            }, b""
        elif command == "RESET":
            self.consumed_len = 0
            response, body = {
                "kind": "OK",
                "role": self.role,
                "epoch": None,
                "step_id": -1,
                "consumed_len": 0,
            }, b""
        elif command.startswith("PREFILL") or command.startswith("VALIDATE"):
            if command.startswith("PREFILL"):
                self.consumed_len = header["token_count"]
            else:
                self._replay_last_sample = True
            rows = header["token_count"] if self.role == "front" else 1
            response, body = {
                "kind": "HIDDEN",
                "role": self.role,
                "epoch": header["epoch"],
                "step_id": header["step_id"],
                "consumed_len": self.consumed_len,
                "shape": [rows, 2048],
                "dtype": "float16",
            }, self._hidden(rows)
            if header.get("capture_router"):
                response.update(self._router_capture(command))
        elif command.startswith("DECODE"):
            self.consumed_len += 1
            response, body = {
                "kind": "HIDDEN",
                "role": self.role,
                "epoch": header["epoch"],
                "step_id": header["step_id"],
                "consumed_len": self.consumed_len,
                "shape": [1, 2048],
                "dtype": "float16",
            }, self._hidden(1)
            if header.get("capture_router"):
                response.update(self._router_capture(command))
        elif command == "SAMPLE":
            if self._replay_last_sample:
                token_id = self._last_sample_token
                self._replay_last_sample = False
            else:
                token_id = next(self.tokens)
                self._last_sample_token = token_id
            response, body = {
                "kind": "TOKEN",
                "role": self.role,
                "epoch": header["epoch"],
                "step_id": header["step_id"],
                "consumed_len": self.consumed_len,
                "token_id": token_id,
            }, (
                bytes(pipeline.MODEL_VOCAB_SIZE * 2)
                if header.get("return_logits")
                else b""
            )
            if header.get("return_logits"):
                response.update(
                    {
                        "logits_shape": [1, pipeline.MODEL_VOCAB_SIZE],
                        "logits_dtype": "float16",
                    }
                )
        else:
            raise AssertionError(f"unexpected controller command: {command}")
        self.last_response = response
        return response, body

    def close(self):
        self.closed = True


class TestPipelineFrames(unittest.TestCase):
    @staticmethod
    def _valid_router_header(layer_ids):
        return {
            pipeline.ROUTER_CAPTURE_KEY: {
                "layer_ids": list(layer_ids),
                "expert_ids": [
                    [slot for slot in range(pipeline.ROUTER_TOP_K)] for _ in layer_ids
                ],
                "probabilities": [[0.125] * pipeline.ROUTER_TOP_K for _ in layer_ids],
            }
        }

    def test_socket_frame_round_trip_sets_the_current_version(self):
        sender, receiver = socket.socketpair()
        try:
            pipeline.send_frame(
                sender,
                {"kind": "PING", "version": -1, "epoch": 7},
                b"fp16-bytes",
            )
            header, payload = pipeline.recv_frame(receiver)
        finally:
            sender.close()
            receiver.close()
        self.assertEqual(
            header,
            {"epoch": 7, "kind": "PING", "version": pipeline.PROTOCOL_VERSION},
        )
        self.assertEqual(payload, b"fp16-bytes")

    def test_socket_frame_rejects_unsupported_version(self):
        sender, receiver = socket.socketpair()
        try:
            encoded = json.dumps({"version": pipeline.PROTOCOL_VERSION + 1}).encode()
            sender.sendall(
                struct.pack(pipeline._FRAME.format, pipeline._MAGIC, len(encoded), 0)
                + encoded
            )
            with self.assertRaisesRegex(
                pipeline.PipelineProtocolError, "unsupported.*version"
            ):
                pipeline.recv_frame(receiver)
        finally:
            sender.close()
            receiver.close()

    def test_socket_frame_rejects_bool_and_float_version_impostors(self):
        for version in (True, 1.0):
            with self.subTest(version=version):
                sender, receiver = socket.socketpair()
                try:
                    encoded = json.dumps({"version": version}).encode()
                    sender.sendall(
                        struct.pack(
                            pipeline._FRAME.format,
                            pipeline._MAGIC,
                            len(encoded),
                            0,
                        )
                        + encoded
                    )
                    with self.assertRaisesRegex(
                        pipeline.PipelineProtocolError, "protocol version.*integer"
                    ):
                        pipeline.recv_frame(receiver)
                finally:
                    sender.close()
                    receiver.close()

    def test_socket_frame_rejects_announced_oversize_before_body_read(self):
        sender, receiver = socket.socketpair()
        try:
            sender.sendall(
                struct.pack(
                    pipeline._FRAME.format,
                    pipeline._MAGIC,
                    pipeline._MAX_HEADER_BYTES + 1,
                    0,
                )
            )
            with self.assertRaisesRegex(pipeline.PipelineProtocolError, "size limit"):
                pipeline.recv_frame(receiver)
        finally:
            sender.close()
            receiver.close()

    def test_send_rejects_configured_header_and_payload_limits(self):
        sender, receiver = socket.socketpair()
        try:
            with mock.patch.object(pipeline, "_MAX_HEADER_BYTES", 16):
                with self.assertRaisesRegex(pipeline.PipelineProtocolError, "header"):
                    pipeline.send_frame(sender, {"long": "x" * 32})
            with mock.patch.object(pipeline, "_MAX_PAYLOAD_BYTES", 2):
                with self.assertRaisesRegex(pipeline.PipelineProtocolError, "payload"):
                    pipeline.send_frame(sender, {"kind": "X"}, b"123")
        finally:
            sender.close()
            receiver.close()

    def test_router_capture_requires_exact_json_shape_and_scalar_types(self):
        header = self._valid_router_header(pipeline.FRONT_LAYER_IDS)
        parsed = pipeline._parse_router_capture(header, pipeline.FRONT_LAYER_IDS)
        self.assertEqual(tuple(parsed), pipeline.FRONT_LAYER_IDS)
        cases = (
            ("layer_ids", 0, False, "integer"),
            ("expert_ids", 0, 0.0, "integer"),
            ("probabilities", 0, 1, "float"),
        )
        for field, row, value, message in cases:
            with self.subTest(field=field, value=value):
                invalid = self._valid_router_header(pipeline.FRONT_LAYER_IDS)
                if field == "layer_ids":
                    invalid[pipeline.ROUTER_CAPTURE_KEY][field][row] = value
                else:
                    invalid[pipeline.ROUTER_CAPTURE_KEY][field][row][0] = value
                with self.assertRaisesRegex(pipeline.PipelineProtocolError, message):
                    pipeline._parse_router_capture(invalid, pipeline.FRONT_LAYER_IDS)

    def test_router_request_flag_rejects_truthy_non_booleans(self):
        self.assertFalse(pipeline._capture_router_requested({}))
        self.assertTrue(pipeline._capture_router_requested({"capture_router": True}))
        with self.assertRaisesRegex(pipeline.PipelineProtocolError, "must be bool"):
            pipeline._capture_router_requested({"capture_router": 1})

    def test_frame_integer_shapes_reject_bool_float_and_tuple_coercion(self):
        pipeline._require_exact_shape([1, 2048], [1, 2048], "shape")
        cases = ([True, 2048], [1.0, 2048], (1, 2048))
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(pipeline.PipelineProtocolError):
                    pipeline._require_exact_shape(value, [1, 2048], "shape")

    def test_hidden_shape_and_progress_require_exact_json_integers(self):
        with self.assertRaises(pipeline.PipelineProtocolError):
            pipeline.Qwen35Pipeline._expect_hidden(
                {"kind": "HIDDEN", "dtype": "float16", "shape": [1.0, 2048]},
                bytes(2048 * 2),
                1,
            )
        for invalid in (True, 1.0):
            with self.subTest(consumed_len=invalid):
                with self.assertRaisesRegex(pipeline.PipelineProtocolError, "integer"):
                    pipeline.Qwen35Pipeline._check_progress(
                        {"consumed_len": invalid}, {"consumed_len": 1}, 1
                    )


class TestPipelineInputValidation(unittest.TestCase):
    @staticmethod
    def _unstarted_pipeline(capacity):
        # Bypass __init__ so these tests prove that validation occurs before a
        # worker request.  Accessing either fake worker is an immediate error.
        instance = object.__new__(pipeline.Qwen35Pipeline)
        instance.capacity = capacity
        instance._closed = False
        instance._epoch = 0
        instance.front = mock.Mock()
        instance.back = mock.Mock()
        return instance

    @classmethod
    def _scripted_pipeline(cls, capacity, **worker_kwargs):
        instance = cls._unstarted_pipeline(capacity)
        instance.front = _ScriptedWorker("front", **worker_kwargs)
        instance.back = _ScriptedWorker("back", **worker_kwargs)
        instance.last_validation = []
        return instance

    def test_zero_generation_needs_no_worker_and_keeps_request_slot_idle(self):
        instance = self._unstarted_pipeline(capacity=3)
        self.assertEqual(instance.generate_ids([4, 5, 6], max_new_tokens=0), [])
        instance.front.request.assert_not_called()
        instance.back.request.assert_not_called()
        self.assertEqual(instance._epoch, 0)

    def test_bad_prompt_and_cache_overflow_are_rejected_before_worker_start(self):
        cases = (
            ([], 1, "nonempty"),
            ([pipeline.TOKENIZER_VOCAB_SIZE], 1, "valid token"),
            ([1, 2, 3], 2, "exceeds cache capacity"),
        )
        for ids, max_new_tokens, message in cases:
            with self.subTest(ids=ids, max_new_tokens=max_new_tokens):
                instance = self._unstarted_pipeline(capacity=4)
                with self.assertRaisesRegex(ValueError, message):
                    instance.generate_ids(ids, max_new_tokens=max_new_tokens)
                instance.front.request.assert_not_called()
                instance.back.request.assert_not_called()

    def test_constructor_rejects_bad_capacity_before_opening_workers(self):
        with mock.patch.object(pipeline, "_WorkerClient") as worker:
            with self.assertRaisesRegex(ValueError, "capacity"):
                pipeline.Qwen35Pipeline("/does/not/matter", capacity=0)
        worker.assert_not_called()

    def test_prompt_ids_reject_lossy_integer_coercion(self):
        instance = self._unstarted_pipeline(capacity=4)
        with self.assertRaisesRegex(TypeError, "Python ints"):
            instance.generate_ids([1.5], max_new_tokens=1)
        with self.assertRaisesRegex(TypeError, "Python ints"):
            instance.generate_ids([True], max_new_tokens=1)
        instance.front.request.assert_not_called()
        instance.back.request.assert_not_called()

    def test_one_sided_begin_failure_resets_both_workers(self):
        instance = self._unstarted_pipeline(capacity=4)
        instance.front.request.side_effect = (
            (
                {
                    "kind": "OK",
                    "epoch": 1,
                    "step_id": -1,
                    "consumed_len": 0,
                },
                b"",
            ),
            (
                {"kind": "OK", "epoch": None, "step_id": -1, "consumed_len": 0},
                b"",
            ),
        )
        instance.back.request.side_effect = (
            pipeline.PipelineWorkerError("injected BEGIN failure"),
            (
                {"kind": "OK", "epoch": None, "step_id": -1, "consumed_len": 0},
                b"",
            ),
        )
        with self.assertRaisesRegex(
            pipeline.PipelineWorkerError, "injected BEGIN failure"
        ):
            instance.generate_ids([1], max_new_tokens=1)
        self.assertEqual(
            [call.args[0]["command"] for call in instance.front.request.call_args_list],
            ["BEGIN", "RESET"],
        )
        self.assertEqual(
            [call.args[0]["command"] for call in instance.back.request.call_args_list],
            ["BEGIN", "RESET"],
        )

    def test_generation_sends_r_minus_one_decodes_and_resets_after_last_token(self):
        instance = self._scripted_pipeline(capacity=6, tokens=(91, 92, 93))

        self.assertEqual(
            instance.generate_ids([4, 5, 6], max_new_tokens=3), [91, 92, 93]
        )

        front_commands = [header["command"] for header, _ in instance.front.calls]
        back_commands = [header["command"] for header, _ in instance.back.calls]
        self.assertEqual(
            front_commands,
            [
                "BEGIN",
                "PREFILL_IDS",
                "SAMPLE",
                "DECODE_ID",
                "SAMPLE",
                "DECODE_ID",
                "SAMPLE",
                "RESET",
            ],
        )
        self.assertEqual(
            back_commands,
            ["BEGIN", "PREFILL_HIDDEN", "DECODE_HIDDEN", "DECODE_HIDDEN", "RESET"],
        )
        decode_headers = [
            header
            for header, _ in instance.front.calls
            if header["command"] == "DECODE_ID"
        ]
        self.assertEqual([header["step_id"] for header in decode_headers], [1, 2])
        self.assertEqual(
            [header["expected_prefix_len"] for header in decode_headers], [3, 4]
        )
        self.assertEqual({header["epoch"] for header in decode_headers}, {1})
        decode_ids = []
        for _, payload in [
            call for call in instance.front.calls if call[0]["command"] == "DECODE_ID"
        ]:
            values = array.array("i")
            values.frombytes(payload)
            decode_ids.append(list(values))
        self.assertEqual(decode_ids, [[91], [92]])
        self.assertEqual(instance.front.calls[-1][0]["command"], "RESET")
        self.assertEqual(instance.back.calls[-1][0]["command"], "RESET")

    def test_total_context_capacity_is_stricter_than_decode_count(self):
        instance = self._scripted_pipeline(capacity=4, tokens=(91,))
        self.assertEqual(instance.generate_ids([1, 2, 3], max_new_tokens=1), [91])
        self.assertEqual(
            [header["command"] for header, _ in instance.front.calls],
            ["BEGIN", "PREFILL_IDS", "SAMPLE", "RESET"],
        )
        overflow = self._scripted_pipeline(capacity=4)
        with self.assertRaisesRegex(ValueError, "exceeds cache capacity"):
            overflow.generate_ids([1, 2, 3], max_new_tokens=2)
        self.assertEqual(overflow.front.calls, [])
        self.assertEqual(overflow.back.calls, [])

    def test_full_prefill_cannot_return_a_token_beyond_total_context_limit(self):
        instance = self._scripted_pipeline(capacity=4)
        with self.assertRaisesRegex(ValueError, "exceeds cache capacity"):
            instance.generate_ids([1, 2, 3, 4], max_new_tokens=1)
        self.assertEqual(instance.front.calls, [])
        self.assertEqual(instance.back.calls, [])

    def test_sample_rejects_bool_token_and_coerced_logits_shape(self):
        instance = self._unstarted_pipeline(capacity=4)
        instance.front.request.return_value = (
            {"kind": "TOKEN", "token_id": True, "consumed_len": 1},
            b"",
        )
        with self.assertRaisesRegex(pipeline.PipelineProtocolError, "integer"):
            instance._sample(bytes(2048 * 2), 1, 0, 1)

        instance.front.request.return_value = (
            {
                "kind": "TOKEN",
                "token_id": 1,
                "consumed_len": 1,
                "logits_shape": [1.0, pipeline.MODEL_VOCAB_SIZE],
                "logits_dtype": "float16",
            },
            bytes(pipeline.MODEL_VOCAB_SIZE * 2),
        )
        with self.assertRaises(pipeline.PipelineProtocolError):
            instance._sample(bytes(2048 * 2), 1, 0, 1, return_logits=True)

        for invalid in (True, 1.0, 0):
            with self.subTest(consumed_len=invalid):
                instance.front.request.return_value = (
                    {
                        "kind": "TOKEN",
                        "token_id": 1,
                        "consumed_len": invalid,
                    },
                    b"",
                )
                with self.assertRaises(pipeline.PipelineProtocolError):
                    instance._sample(bytes(2048 * 2), 1, 0, 1)

    def test_reset_requires_exact_zero_consumed_len(self):
        for invalid in (True, 0.0, 1):
            with self.subTest(consumed_len=invalid):
                instance = self._unstarted_pipeline(capacity=4)
                invalid_ack = {
                    "kind": "OK",
                    "epoch": None,
                    "step_id": -1,
                    "consumed_len": invalid,
                }
                valid_ack = {
                    "kind": "OK",
                    "epoch": None,
                    "step_id": -1,
                    "consumed_len": 0,
                }
                instance.front.request.return_value = (invalid_ack, b"")
                instance.back.request.return_value = (valid_ack, b"")
                with self.assertRaisesRegex(
                    pipeline.PipelineWorkerError, "failed to reset"
                ):
                    instance._reset()
                instance.front.request.assert_called_once_with({"command": "RESET"})
                instance.back.request.assert_called_once_with({"command": "RESET"})

    def test_stateless_validate_requires_matching_exact_consumed_len(self):
        instance = self._scripted_pipeline(capacity=4, tokens=(91,))
        original_request = instance.back.request

        def wrong_validate_progress(header, payload=b""):
            response, body = original_request(header, payload)
            if header["command"] == "VALIDATE_HIDDEN":
                response = dict(response, consumed_len=1.0)
            return response, body

        instance.back.request = wrong_validate_progress
        with self.assertRaisesRegex(pipeline.PipelineProtocolError, "integer"):
            instance.generate_ids([1, 2], max_new_tokens=1, validate_stateless=True)
        self.assertEqual(instance.front.calls[-1][0]["command"], "RESET")
        self.assertEqual(instance.back.calls[-1][0]["command"], "RESET")

    def test_back_decode_failure_resets_both_halves_of_the_request(self):
        instance = self._scripted_pipeline(capacity=5)
        instance.back.fail_command = "DECODE_HIDDEN"

        with self.assertRaisesRegex(
            pipeline.PipelineWorkerError, "injected back DECODE_HIDDEN failure"
        ):
            instance.generate_ids([1, 2], max_new_tokens=2)

        self.assertEqual(
            [header["command"] for header, _ in instance.front.calls],
            ["BEGIN", "PREFILL_IDS", "SAMPLE", "DECODE_ID", "RESET"],
        )
        self.assertEqual(
            [header["command"] for header, _ in instance.back.calls],
            ["BEGIN", "PREFILL_HIDDEN", "DECODE_HIDDEN", "RESET"],
        )

    def test_divergent_decode_ack_is_rejected_and_resets_both_workers(self):
        instance = self._scripted_pipeline(capacity=5)
        original_request = instance.back.request

        def wrong_progress(header, payload=b""):
            response, body = original_request(header, payload)
            if header["command"] == "DECODE_HIDDEN":
                response = dict(response, consumed_len=response["consumed_len"] - 1)
            return response, body

        instance.back.request = wrong_progress
        with self.assertRaisesRegex(pipeline.PipelineProtocolError, "lengths diverged"):
            instance.generate_ids([1, 2], max_new_tokens=2)
        self.assertEqual(instance.front.calls[-1][0]["command"], "RESET")
        self.assertEqual(instance.back.calls[-1][0]["command"], "RESET")

    def test_router_capture_is_opt_in_and_compares_every_layer(self):
        instance = self._scripted_pipeline(capacity=5, tokens=(91, 92))

        self.assertEqual(
            instance.generate_ids([1, 2], max_new_tokens=2, validate_stateless=True),
            [91, 92],
        )

        captured_commands = {
            "PREFILL_IDS",
            "PREFILL_HIDDEN",
            "DECODE_ID",
            "DECODE_HIDDEN",
            "VALIDATE_IDS",
            "VALIDATE_HIDDEN",
        }
        for worker in (instance.front, instance.back):
            for header, _ in worker.calls:
                if header["command"] in captured_commands:
                    self.assertIs(header["capture_router"], True)
        self.assertEqual(len(instance.last_validation), 2)
        for validation in instance.last_validation:
            routes = validation["router"]
            self.assertEqual(routes["mismatch_layers"], [])
            self.assertEqual(routes["id_mismatch_slot_count"], 0)
            self.assertEqual(routes["max_prob_abs"], 0.0)
            self.assertEqual(len(routes["layers"]), 40)
            self.assertEqual(routes["layers"][0]["layer_id"], 0)
            self.assertEqual(routes["layers"][-1]["layer_id"], 39)

    def test_production_generation_never_requests_or_receives_router_capture(self):
        instance = self._scripted_pipeline(capacity=4, tokens=(91,))

        self.assertEqual(instance.generate_ids([1, 2], max_new_tokens=1), [91])

        stateful = {
            "PREFILL_IDS",
            "PREFILL_HIDDEN",
            "DECODE_ID",
            "DECODE_HIDDEN",
        }
        for worker in (instance.front, instance.back):
            for header, _ in worker.calls:
                if header["command"] in stateful:
                    self.assertIs(header["capture_router"], False)
        self.assertFalse(instance.last_validation)

    def test_router_summary_reports_ids_slots_and_probability_delta(self):
        instance = self._scripted_pipeline(
            capacity=4,
            tokens=(91,),
            validation_route_delta=(39, 2, 99, 0.2),
        )

        self.assertEqual(
            instance.generate_ids([1, 2], max_new_tokens=1, validate_stateless=True),
            [91],
        )

        routes = instance.last_validation[0]["router"]
        self.assertEqual(routes["mismatch_layers"], [39])
        self.assertEqual(routes["id_mismatch_slot_count"], 1)
        self.assertAlmostEqual(routes["max_prob_abs"], 0.075)
        layer = routes["layers"][39]
        self.assertEqual(layer["cached_ids"][2], 41)
        self.assertEqual(layer["stateless_ids"][2], 99)
        self.assertEqual(layer["id_mismatch_slots"], [2])
        self.assertAlmostEqual(layer["max_prob_abs"], 0.075)

    def test_malformed_router_capture_is_rejected_before_it_can_be_compared(self):
        instance = self._scripted_pipeline(capacity=4, tokens=(91,))
        original_request = instance.back.request

        def malformed_routes(header, payload=b""):
            response, body = original_request(header, payload)
            if header["command"] == "PREFILL_HIDDEN":
                response = dict(response)
                response[pipeline.ROUTER_CAPTURE_KEY] = {
                    "layer_ids": list(pipeline.BACK_LAYER_IDS),
                    "expert_ids": [[20] * pipeline.ROUTER_TOP_K] * 20,
                    # IDs and probabilities must have matching Top-8 shape.
                    "probabilities": [[1.0]] * 20,
                }
            return response, body

        instance.back.request = malformed_routes
        with self.assertRaisesRegex(pipeline.PipelineProtocolError, "Top-8"):
            instance.generate_ids([1, 2], max_new_tokens=1, validate_stateless=True)
        self.assertEqual(instance.front.calls[-1][0]["command"], "RESET")
        self.assertEqual(instance.back.calls[-1][0]["command"], "RESET")


class TestWorkerClientValidation(unittest.TestCase):
    def _client(self):
        client = object.__new__(pipeline._WorkerClient)
        client.role = "front"
        client.sock = mock.Mock()
        client.closed = False
        return client

    def test_response_role_epoch_and_step_are_authenticated(self):
        request = {"command": "BEGIN", "epoch": 7, "step_id": -1}
        invalid = (
            {"kind": "OK", "role": "back", "epoch": 7, "step_id": -1},
            {"kind": "OK", "role": "front", "epoch": 8, "step_id": -1},
            {"kind": "OK", "role": "front", "epoch": 7, "step_id": 0},
        )
        for response in invalid:
            with self.subTest(response=response):
                client = self._client()
                with mock.patch.object(pipeline, "send_frame"), mock.patch.object(
                    pipeline, "recv_frame", return_value=(response, b"")
                ):
                    with self.assertRaises(pipeline.PipelineProtocolError):
                        client.request(request)

    def test_response_identity_rejects_bool_and_float_integer_impostors(self):
        request = {"command": "SAMPLE", "epoch": 1, "step_id": 0}
        invalid = (
            {"kind": "TOKEN", "role": "front", "epoch": True, "step_id": 0},
            {"kind": "TOKEN", "role": "front", "epoch": 1, "step_id": 0.0},
        )
        for response in invalid:
            with self.subTest(response=response):
                client = self._client()
                with mock.patch.object(pipeline, "send_frame"), mock.patch.object(
                    pipeline, "recv_frame", return_value=(response, b"")
                ):
                    with self.assertRaisesRegex(
                        pipeline.PipelineProtocolError, "integer"
                    ):
                        client.request(request)

        client = self._client()
        reset_response = {
            "kind": "OK",
            "role": "front",
            "epoch": None,
            "step_id": -1.0,
        }
        with mock.patch.object(pipeline, "send_frame"), mock.patch.object(
            pipeline, "recv_frame", return_value=(reset_response, b"")
        ):
            with self.assertRaisesRegex(pipeline.PipelineProtocolError, "integer"):
                client.request({"command": "RESET"})


@unittest.skipUnless(
    MODEL_DIR.is_dir(), "Qwen-AgentWorld tokenizer fixture unavailable"
)
class TestTokenizerCompatibility(unittest.TestCase):
    def test_legacy_loader_matches_transformers_4573_per_id_for_raw_and_chat(self):
        tokenizer = pipeline.load_tokenizer_compat(MODEL_DIR)
        prompts = ("Hello, Qwen!", "请用一句话解释缓存。")
        actual = {
            "raw": [
                tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
            ],
            "chat": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ],
        }
        self.assertEqual(len(tokenizer), pipeline.TOKENIZER_VOCAB_SIZE)

        if not REFERENCE_PYTHON.is_file():
            self.skipTest("Transformers 4.57.3 reference environment unavailable")
        source = """
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
prompts = ('Hello, Qwen!', '请用一句话解释缓存。')
print(json.dumps({
    'raw': [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts],
    'chat': [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True) for prompt in prompts],
}, ensure_ascii=False))
"""
        completed = subprocess.run(
            [str(REFERENCE_PYTHON), "-c", source, str(MODEL_DIR)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(actual, json.loads(completed.stdout))
