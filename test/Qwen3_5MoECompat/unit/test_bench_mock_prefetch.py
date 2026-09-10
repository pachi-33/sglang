import asyncio
import unittest
from unittest import mock

from sglang import bench_serving


class TestBenchMockPrefetch(unittest.TestCase):
    def test_record_is_nonstreaming_and_replay_is_the_only_returned_timing(self):
        record = bench_serving.RequestFuncOutput(
            success=True,
            latency=9.0,
            request_id="record-id",
            completion_token_ids=[1, 2],
        )
        replay = bench_serving.RequestFuncOutput(
            success=True,
            latency=1.5,
            request_id="replay-id",
            completion_token_ids=[1, 2],
            mock_expert_prefetch_metrics={"pair_id": "pair"},
        )
        request = bench_serving.RequestFuncInput(
            prompt=[10, 11],
            api_url="http://127.0.0.1/v1/completions",
            prompt_len=2,
            output_len=2,
            model="model",
            mock_expert_prefetch={
                "pair_id": "pair",
                "route_recall": 0.5,
                "top_k": 8,
                "lead_layers": 2,
                "seed": 0,
            },
        )
        with mock.patch.object(
            bench_serving,
            "async_request_openai_completions",
            new=mock.AsyncMock(side_effect=(record, replay)),
        ) as send:
            output = asyncio.run(
                bench_serving.async_request_mock_expert_prefetch(request)
            )
        first, second = [call.args[0] for call in send.call_args_list]
        self.assertFalse(first.stream)
        self.assertEqual(
            first.mock_expert_prefetch, {"phase": "record", "pair_id": "pair"}
        )
        self.assertTrue(second.stream)
        self.assertEqual(second.mock_expert_prefetch["phase"], "replay")
        self.assertEqual(output.latency, 1.5)
        self.assertEqual(output.record_latency, 9.0)
        self.assertEqual(output.record_request_id, "record-id")

    def test_token_divergence_marks_pair_failed(self):
        record = bench_serving.RequestFuncOutput(success=True, completion_token_ids=[1])
        replay = bench_serving.RequestFuncOutput(
            success=True,
            completion_token_ids=[2],
            mock_expert_prefetch_metrics={"pair_id": "p"},
        )
        request = bench_serving.RequestFuncInput(
            prompt=[10],
            api_url="http://127.0.0.1/v1/completions",
            prompt_len=1,
            output_len=1,
            model="model",
            mock_expert_prefetch={
                "pair_id": "p",
                "route_recall": 0.0,
                "top_k": 0,
                "lead_layers": 1,
            },
        )
        with mock.patch.object(
            bench_serving,
            "async_request_openai_completions",
            new=mock.AsyncMock(side_effect=(record, replay)),
        ):
            output = asyncio.run(
                bench_serving.async_request_mock_expert_prefetch(request)
            )
        self.assertFalse(output.success)
        self.assertIn("token IDs differ", output.error)


if __name__ == "__main__":
    unittest.main()
