"""CPU-only checks for the independent two-node prefill launcher."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path("scripts/moe_offload/serve_glm52_prefill_2node_pp4_cp8_ep8.sh")
BASH = shutil.which("bash")


class TestTwoNodePrefillLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "standalone checkout"
        self.launcher = self.checkout / SCRIPT
        # No other launcher is present. Preflight must not need accelerator imports.
        for relative in (
            SCRIPT,
            Path("python/sglang/srt/offloading/moe/npu/options.py"),
        ):
            target = self.checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("dirname", "date", "cat"):
            (self.bin / name).symlink_to(shutil.which(name))
        self.model = self.root / "checkpoint with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "glm_moe_dsa",
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "num_hidden_layers": 78,
                    "first_k_dense_replace": 3,
                    "moe_layer_freq": 1,
                    "n_routed_experts": 256,
                    "hidden_size": 6144,
                    "moe_intermediate_size": 2048,
                }
            )
        )
        self.env = dict(os.environ)
        self.env.update(
            PATH=str(self.bin),
            MODEL_PATH=str(self.model),
            PYTHON_BIN=sys.executable,
            P0_IP="192.0.2.10",
            P1_IP="192.0.2.11",
            NODE_RANK="0",
            COMM_IFACE="eth0",
            RUNTIME_DIR=str(self.root / "runtime"),
            LOG_DIR=str(self.root / "logs"),
            HOST_POOL_GIB="",
            PREFILL_OFFLOAD_RATIO="",
            PREFILL_RESIDENT_RATIO="",
            EXECUTION_SLOTS="",
            CONTEXT_LENGTH="",
            CHUNKED_PREFILL_SIZE="",
            MAX_PREFILL_TOKENS="",
            MAX_RUNNING_REQUESTS="",
            MEM_FRACTION_STATIC="",
            CP_STRATEGY="",
            P_MOE_A2A_BACKEND="",
            FAKE_PREFILL_TEST="",
            ENABLE_MTP="0",
            SGLANG_PP_LAYER_PARTITION="18,20,24,16",
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(16))),
            SERVER_HOST="0.0.0.0",
            SERVER_PORT="8000",
            DIST_PORT="5000",
            BOOTSTRAP_PORT="8998",
            MF_STORE_PORT="24707",
            CANN_ENV="/missing/cann.sh",
            ASCEND_MF_STORE_URL="tcp://stale.example:1",
            MF_CONFIG_STORE_URL="tcp://stale.example:2",
        )

    def _run(self, **changes):
        result = subprocess.run(
            [BASH, str(self.launcher), "--dry-run"],
            env={**self.env, **changes},
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertFalse((self.root / "runtime").exists())
        self.assertFalse((self.root / "logs").exists())
        return result

    def _success(self, **changes):
        result = self._run(**changes)
        self.assertEqual(result.returncode, 0, result.stderr)
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        command = shlex.split(result.stdout.split("Command:", 1)[1])
        signature = result.stdout.split(
            "Common configuration SHA256 (compare both nodes): ", 1
        )[1].splitlines()[0]
        return result, config, command, signature

    def test_both_nodes_share_config_and_launch_two_complete_stages(self):
        signatures = []
        for rank in (0, 1):
            result, config, command, signature = self._success(NODE_RANK=str(rank))
            signatures.append(signature)
            for flag, expected in {
                "--model-path": str(self.model),
                "--nnodes": "2",
                "--node-rank": str(rank),
                "--pp-size": "4",
                "--tp-size": "8",
                "--ep-size": "8",
                "--dp-size": "1",
                "--attn-dp-size": "1",
                "--attn-cp-size": "8",
                "--cp-strategy": "interleave",
                "--dist-init-addr": "192.0.2.10:5000",
                "--moe-a2a-backend": "deepep",
                "--deepep-mode": "normal",
                "--deepep-dispatcher-output-dtype": "int8",
                "--chunked-prefill-size": "4096",
                "--max-running-requests": "64",
                "--disaggregation-mode": "prefill",
            }.items():
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertIn("--enable-dp-lm-head", command)
            self.assertIn("--disable-overlap-schedule", command)
            self.assertNotIn("--speculative-algorithm", command)
            self.assertIn(
                f"Node {rank}: PP stages {[2 * rank, 2 * rank + 1]}", result.stdout
            )
            self.assertIn(f"Local IP: 192.0.2.{10 + rank}", result.stdout)
            self.assertIn("KV store: tcp://192.0.2.10:24707", result.stdout)
            self.assertEqual(config["host_pool_gib"], 32)
            self.assertEqual(config["prefill_resident_ratio"], 0.25)
            self.assertEqual(config["mode"], "prefill_only")
            self.assertNotIn("execution_slots", config)
            self.assertIn(
                "HBM slots/rank by stage: [168, 208, 240, 176]", result.stdout
            )
            self.assertIn("512 GiB/node", result.stdout)
        self.assertEqual(signatures[0], signatures[1])

    def test_ratio_aliases_have_same_signature_and_report_rounded_experts(self):
        first = self._success(PREFILL_OFFLOAD_RATIO="0.5")
        second = self._success(PREFILL_RESIDENT_RATIO="0.5")
        self.assertEqual(first[3], second[3])
        self.assertIn("resident=16, cold=16, total=32", first[0].stdout)
        self.assertIn("[272, 352, 416, 288]", first[0].stdout)
        rounded, *_ = self._success(PREFILL_OFFLOAD_RATIO="0.9")
        self.assertIn("resident=3, cold=29, total=32", rounded.stdout)
        self.assertIn("Effective offload ratio: 0.906250", rounded.stdout)

    def test_full_offload_uses_two_layer_banks(self):
        result, config, *_ = self._success(PREFILL_OFFLOAD_RATIO="1")
        self.assertEqual(config["prefill_resident_ratio"], 0)
        self.assertEqual(config["execution_slots"], 64)
        self.assertIn("cold H2D/layer/rank: 1.126831 GiB", result.stdout)

    def test_configuration_fingerprint_catches_common_setting_changes(self):
        base = self._success()[3]
        for changes in (
            dict(PREFILL_OFFLOAD_RATIO="0.5"),
            dict(CHUNKED_PREFILL_SIZE="16384"),
            dict(CP_STRATEGY="zigzag"),
            dict(P_MOE_A2A_BACKEND="ascend_ops"),
            dict(SGLANG_PP_LAYER_PARTITION="20,20,22,16"),
            dict(FAKE_PREFILL_TEST="0"),
        ):
            with self.subTest(changes=changes):
                self.assertNotEqual(base, self._success(**changes)[3])

    def test_rejects_bad_ratio_and_capacity_before_startup(self):
        for changes, message in (
            (dict(PREFILL_OFFLOAD_RATIO="0"), "PREFILL_OFFLOAD_RATIO must be"),
            (dict(PREFILL_OFFLOAD_RATIO="1.01"), "PREFILL_OFFLOAD_RATIO must be"),
            (dict(PREFILL_OFFLOAD_RATIO="nan"), "PREFILL_OFFLOAD_RATIO must be"),
            (dict(PREFILL_RESIDENT_RATIO="1"), "prefill_resident_ratio must be"),
            (
                dict(PREFILL_OFFLOAD_RATIO="0.75", PREFILL_RESIDENT_RATIO="0.25"),
                "Set only one",
            ),
            (dict(EXECUTION_SLOTS="239"), "at least 240"),
            (dict(HOST_POOL_GIB="27"), "HOST_POOL_GIB leaves no room"),
        ):
            with self.subTest(changes=changes):
                result = self._run(**changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_rejects_invalid_node_network_and_topology_settings(self):
        for changes, message in (
            (dict(NODE_RANK=""), "NODE_RANK"),
            (dict(NODE_RANK="2"), "NODE_RANK"),
            (dict(P1_IP="192.0.2.10"), "different nodes"),
            (dict(P0_IP="127.0.0.1"), "non-loopback"),
            (dict(P0_IP=""), "Set both P0_IP and P1_IP"),
            (dict(COMM_IFACE="lo"), "non-loopback"),
            (dict(DIST_PORT="8000"), "must be distinct"),
            (dict(ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"), "16 distinct"),
            (dict(CHUNKED_PREFILL_SIZE="4097"), "divisible by 128"),
            (dict(CP_STRATEGY="invalid"), "CP_STRATEGY"),
            (dict(ENABLE_MTP="1"), "ENABLE_MTP=0"),
            (dict(SGLANG_PP_LAYER_PARTITION="18,20,24,15"), "summing to 78"),
        ):
            with self.subTest(changes=changes):
                result = self._run(**changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
