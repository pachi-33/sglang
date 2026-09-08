"""CPU-only launch wiring tests for the single-GPU HTTP module."""

import sys
import unittest
from unittest import mock

from sglang.srt.layers.qwen3_5 import single_gpu_api


class TestSingleGPUAPI(unittest.TestCase):
    def test_cli_exposes_fixed_expert_runtime_controls(self):
        defaults = single_gpu_api._parse_args([])
        self.assertEqual(defaults.expert_cache_mib, 7168)
        self.assertEqual(defaults.expert_stage_slots, 16)
        self.assertEqual(defaults.expert_io_workers, 2)
        self.assertEqual(defaults.capacity, 2048)
        self.assertIsNone(defaults.expert_trace_dir)
        self.assertEqual(defaults.host, "127.0.0.1")
        self.assertEqual(defaults.port, 30000)

        configured = single_gpu_api._parse_args(
            [
                "--expert-pack-manifest",
                "/pack/manifest.json",
                "--expert-cache-mib",
                "6144",
                "--expert-stage-slots",
                "4",
                "--expert-io-workers",
                "1",
                "--capacity",
                "1024",
                "--served-model-name",
                "agent-world",
                "--expert-trace-dir",
                "/tmp/expert-traces",
            ]
        )
        self.assertEqual(configured.expert_pack_manifest, "/pack/manifest.json")
        self.assertEqual(configured.expert_cache_mib, 6144)
        self.assertEqual(configured.expert_stage_slots, 4)
        self.assertEqual(configured.expert_io_workers, 1)
        self.assertEqual(configured.capacity, 1024)
        self.assertEqual(configured.served_model_name, "agent-world")
        self.assertEqual(configured.expert_trace_dir, "/tmp/expert-traces")

    def test_main_constructs_one_backend_and_one_uvicorn_worker(self):
        tokenizer = object()
        backend = object()
        engine = mock.Mock()
        app = object()
        uvicorn = mock.Mock()
        with mock.patch.object(
            single_gpu_api, "load_tokenizer_compat", return_value=tokenizer
        ) as load_tokenizer, mock.patch.object(
            single_gpu_api, "Qwen35SingleGPU", return_value=backend
        ) as backend_type, mock.patch.object(
            single_gpu_api, "Qwen35SingleGPUAPIEngine", return_value=engine
        ) as engine_type, mock.patch.object(
            single_gpu_api, "create_app", return_value=app
        ) as create_app, mock.patch.dict(
            sys.modules, {"uvicorn": uvicorn}
        ):
            result = single_gpu_api.main(
                [
                    "--model-dir",
                    "/models/agent-world",
                    "--expert-pack-manifest",
                    "/pack/manifest.json",
                    "--expert-cache-mib",
                    "6144",
                    "--expert-stage-slots",
                    "8",
                    "--expert-io-workers",
                    "2",
                    "--capacity",
                    "1024",
                    "--stats-path",
                    "/tmp/stats.json",
                    "--expert-trace-dir",
                    "/tmp/expert-traces",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "31000",
                    "--api-key",
                    "secret",
                ]
            )
        self.assertEqual(result, 0)
        load_tokenizer.assert_called_once()
        _, backend_kwargs = backend_type.call_args
        self.assertEqual(backend_kwargs["expert_cache_mib"], 6144)
        self.assertEqual(backend_kwargs["capacity"], 1024)
        self.assertEqual(backend_kwargs["expert_trace_dir"], "/tmp/expert-traces")
        engine_type.assert_called_once_with(backend, tokenizer, model_id="agent-world")
        create_app.assert_called_once_with(engine, api_key="secret")
        uvicorn.run.assert_called_once_with(
            app,
            host="0.0.0.0",
            port=31000,
            log_level="info",
            workers=1,
        )
        engine.close.assert_called_once_with()

    def test_main_rejects_invalid_port_before_loading_model(self):
        with mock.patch.object(single_gpu_api, "load_tokenizer_compat") as loader:
            with self.assertRaisesRegex(SystemExit, "port"):
                single_gpu_api.main(["--port", "0"])
        loader.assert_not_called()

    def test_main_rejects_empty_api_key_before_loading_model(self):
        with mock.patch.object(single_gpu_api, "load_tokenizer_compat") as loader:
            with self.assertRaisesRegex(SystemExit, "api-key"):
                single_gpu_api.main(["--api-key", ""])
        loader.assert_not_called()

    def test_engine_construction_failure_closes_loaded_backend(self):
        backend = mock.Mock()
        with mock.patch.object(
            single_gpu_api, "load_tokenizer_compat", return_value=object()
        ), mock.patch.object(
            single_gpu_api, "Qwen35SingleGPU", return_value=backend
        ), mock.patch.object(
            single_gpu_api,
            "Qwen35SingleGPUAPIEngine",
            side_effect=RuntimeError("engine failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "engine failed"):
                single_gpu_api.main([])
        backend.close.assert_called_once_with()

    def test_app_construction_failure_closes_engine(self):
        engine = mock.Mock()
        with mock.patch.object(
            single_gpu_api, "load_tokenizer_compat", return_value=object()
        ), mock.patch.object(
            single_gpu_api, "Qwen35SingleGPU", return_value=object()
        ), mock.patch.object(
            single_gpu_api, "Qwen35SingleGPUAPIEngine", return_value=engine
        ), mock.patch.object(
            single_gpu_api,
            "create_app",
            side_effect=RuntimeError("app failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "app failed"):
                single_gpu_api.main([])
        engine.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
