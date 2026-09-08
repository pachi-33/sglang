import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.bench_serving import (
    RequestFuncInput,
    RequestFuncOutput,
    get_tokenizer,
    positive_int,
    request_with_concurrency_limit,
)


class TestBenchServingConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_semaphore_limits_in_flight_requests(self):
        active = 0
        peak = 0

        async def request_func(request_func_input, pbar=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return RequestFuncOutput(success=True)

        request = RequestFuncInput("prompt", "url", 1, 1, "model")
        semaphore = asyncio.Semaphore(2)
        outputs = await asyncio.gather(
            *(
                request_with_concurrency_limit(
                    request_func,
                    request,
                    None,
                    semaphore,
                )
                for _ in range(6)
            )
        )

        self.assertEqual(peak, 2)
        self.assertTrue(all(output.success for output in outputs))

    async def test_none_keeps_requests_unlimited(self):
        started = 0
        all_started = asyncio.Event()

        async def request_func(request_func_input, pbar=None):
            nonlocal started
            started += 1
            if started == 4:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return RequestFuncOutput(success=True)

        request = RequestFuncInput("prompt", "url", 1, 1, "model")
        await asyncio.gather(
            *(
                request_with_concurrency_limit(request_func, request, None, None)
                for _ in range(4)
            )
        )

        self.assertEqual(started, 4)


class TestPositiveInt(unittest.TestCase):
    def test_accepts_positive_integer(self):
        self.assertEqual(positive_int("3"), 3)

    def test_rejects_non_positive_or_non_integer(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_int(value)


class TestTokenizerCompatibility(unittest.TestCase):
    def test_qwen35_local_checkpoint_uses_compat_loader(self):
        tokenizer = object()
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5_moe"}), encoding="utf-8"
            )
            with mock.patch(
                "sglang.srt.layers.qwen3_5.pipeline.load_tokenizer_compat",
                return_value=tokenizer,
            ) as loader:
                self.assertIs(get_tokenizer(str(model_dir)), tokenizer)
        loader.assert_called_once_with(str(model_dir))


if __name__ == "__main__":
    unittest.main()
