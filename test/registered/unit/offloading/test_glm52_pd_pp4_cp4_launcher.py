"""Contract tests for the 1P-offload / 2D launcher without accelerator dependencies."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts/moe_offload/serve_glm52_pd_pp4_cp4.sh"
)
BASH = shutil.which("bash")
IPS = ("10.20.30.10", "10.20.30.11", "10.20.30.12")


class TestGlm52PdPP4Launcher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glm52-pd-pp4-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # No ip / ifconfig / hostname or accelerator programs in PATH.
        for name in ("dirname", "date", "cat", "mkdir", "tee"):
            (self.bin / name).symlink_to(shutil.which(name))
        python = self.bin / "python3"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if len(sys.argv) > 1 and sys.argv[1].endswith('/pd_node.py'):\n"
            "    sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))\n"
            "    import pd_node\n"
            "    from unittest.mock import patch\n"
            "    local = {ip: {'bond0'} for ip in os.environ.get('STUB_LOCAL_IPS', '').split(',') if ip}\n"
            "    with patch.object(pd_node, 'local_ipv4_interfaces', return_value=local):\n"
            "        pd_node.main(sys.argv[2:])\n"
            "    sys.exit(0)\n"
            "if sys.argv[1:4] in (['-u', '-m', 'sglang.launch_server'], ['-u', '-m', 'sglang_router.launch_router']):\n"
            "    keys = ['SGLANG_HOST_IP', 'SGLANG_PP_LAYER_PARTITION', 'HCCL_SOCKET_IFNAME',\n"
            "            'ASCEND_MF_STORE_URL', 'SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE', 'PYTHONPATH',\n"
            "            'HCCL_BUFFSIZE', 'TASK_QUEUE_ENABLE', 'SGLANG_ENABLE_OVERLAP_PLAN_STREAM',\n"
            "            'SGLANG_NPU_USE_MULTI_STREAM', 'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']\n"
            "    data = {'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in keys}}\n"
            "    pathlib.Path(os.environ['STUB_RECORD']).write_text(json.dumps(data))\n"
            "    sys.exit(0)\n"
            "if sys.argv[1:3] == ['-m', 'sglang.srt.offloading.moe.npu.native_acl']:\n"
            "    sys.exit(0)\n"
            "if len(sys.argv) > 2 and sys.argv[1] == '-' and sys.argv[2] == os.environ['RUNTIME_DIR'] + '/python':\n"
            "    pathlib.Path(sys.argv[2], 'memfabric_hybrid', 'lib').mkdir(parents=True, exist_ok=True)\n"
            "    sys.exit(0)\n"
            "if len(sys.argv) > 2 and sys.argv[1] == '-' and sys.argv[2].endswith('/memfabric_hybrid/lib'):\n"
            "    sys.exit(0)\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        self.config = {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "num_hidden_layers": 78,
            "first_k_dense_replace": 3,
            "moe_layer_freq": 1,
            "n_routed_experts": 256,
            "hidden_size": 6144,
            "moe_intermediate_size": 2048,
        }
        (self.model / "config.json").write_text(json.dumps(self.config))
        env_file = self.root / "ascend-env.sh"
        env_file.write_text("# mock Ascend environment\n")
        self.env = {
            "PATH": str(self.bin),
            "LC_ALL": "C",
            "MODEL_PATH": str(self.model),
            "P_IP": IPS[0],
            "D0_IP": IPS[1],
            "D1_IP": IPS[2],
            "STUB_LOCAL_IPS": IPS[0],
            "STUB_RECORD": str(self.root / "launched.json"),
            "RUNTIME_DIR": str(self.root / "runtime"),
            "LOG_DIR": str(self.root / "logs"),
            "CANN_ENV": str(env_file),
            "ATB_ENV": str(env_file),
            "CUSTOMIZE_ENV": str(env_file),
            "CUSTOM_TRANSFORMER_ENV": str(env_file),
            # Test that these pre-existing values do not leak to the wrong role.
            "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
            "SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE": "1",
        }

    def run_launcher(self, *args, dry=True, overrides=None):
        command = [BASH, str(SCRIPT), *args]
        if dry:
            command.append("--dry-run")
        result = subprocess.run(
            command,
            env={**self.env, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=20,
        )
        if dry:
            self.assertFalse((self.root / "runtime").exists())
            self.assertFalse((self.root / "logs").exists())
            self.assertFalse((self.root / "launched.json").exists())
        return result

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("Command:")
        )
        return shlex.split(line[len("Command:") :])

    def option(self, command, name, value):
        self.assertEqual(command[command.index(name) + 1], str(value))

    def test_prefill_auto_uses_stage_ep_cp_and_offload(self):
        result = self.run_launcher()
        command = self.command(result)
        for key, value in {
            "--disaggregation-mode": "prefill",
            "--disaggregation-transfer-backend": "ascend",
            "--dist-init-addr": f"{IPS[0]}:5000",
            "--nnodes": 1,
            "--node-rank": 0,
            "--pp-size": 4,
            "--tp-size": 4,
            "--dp-size": 1,
            "--attn-dp-size": 1,
            "--attn-cp-size": 4,
            "--cp-strategy": "zigzag",
            "--ep-size": 4,
            "--moe-a2a-backend": "ascend_ops",
            "--served-model-name": "glm-5",
            "--chunked-prefill-size": 8192,
            "--port": 8000,
            "--disaggregation-bootstrap-port": 8998,
        }.items():
            self.option(command, key, value)
        self.assertIn("--enable-prefill-cp", command)
        self.assertIn("--disable-overlap-schedule", command)
        self.assertNotIn("--speculative-algorithm", command)
        self.assertIn("interface=bond0", result.stdout)
        config = json.JSONDecoder().raw_decode(
            result.stdout[result.stdout.index('{\n  "mode"') :]
        )[0]
        self.assertEqual(config["mode"], "prefill_only")
        self.assertEqual(config["execution_slots"], 128)
        self.assertEqual(config["cache_ratio"], 0)
        self.assertEqual(config["host_pool_gib"], 56)
        self.assertIn("896 GiB/node", result.stdout)

    def test_decode_uses_two_node_baseline_without_offload_or_cp(self):
        for rank, address in enumerate(IPS[1:]):
            with self.subTest(rank=rank):
                result = self.run_launcher(overrides={"STUB_LOCAL_IPS": address})
                command = self.command(result)
                for key, value in {
                    "--disaggregation-mode": "decode",
                    "--nnodes": 2,
                    "--node-rank": rank,
                    "--dist-init-addr": f"{IPS[1]}:5000",
                    "--tp-size": 32,
                    "--attn-dp-size": 32,
                    "--ep-size": 32,
                    "--moe-a2a-backend": "deepep",
                    "--deepep-mode": "low_latency",
                    "--cuda-graph-backend-decode": "full",
                    "--cuda-graph-max-bs-decode": 4,
                    "--served-model-name": "glm-5",
                    "--port": 8003,
                }.items():
                    self.option(command, key, value)
                for absent in (
                    "--moe-offload-config",
                    "--pp-size",
                    "--attn-cp-size",
                    "--speculative-algorithm",
                ):
                    self.assertNotIn(absent, command)
                self.assertIn(
                    f"IP={address} role=decode node_rank={rank}", result.stdout
                )

    def test_router_needs_no_model_or_node_enumeration(self):
        result = self.run_launcher(
            "router",
            overrides={
                "MODEL_PATH": "/missing",
                "STUB_LOCAL_IPS": "",
                "D1_IP": "unused",
            },
        )
        command = self.command(result)
        self.assertIn("sglang_router.launch_router", command)
        self.option(command, "--prefill", f"http://{IPS[0]}:8000")
        self.option(command, "--decode", f"http://{IPS[1]}:8003")
        self.option(command, "--port", 6688)
        self.assertEqual(command.count("--prefill"), 1)
        self.assertEqual(command.count("--decode"), 1)

    def test_invalid_settings_fail_before_start(self):
        cases = (
            ({"D0_IP": IPS[0]}, "distinct node addresses"),
            ({"STUB_LOCAL_IPS": ""}, "exactly one local node IP match"),
            ({"NODE_RANK": "1"}, "conflicts with detected P_IP"),
            ({"ENABLE_MTP": "1"}, "requires ENABLE_MTP=0"),
            ({"HOST_POOL_GIB": "48"}, "HOST_POOL_GIB leaves no room"),
            ({"EXECUTION_SLOTS": "127"}, "at least 128"),
            ({"SGLANG_PP_LAYER_PARTITION": "18,20,24,15"}, "summing to 78"),
            ({"CHUNKED_PREFILL_SIZE": "8193"}, "divisible by page size 128"),
            ({"ASCEND_RT_VISIBLE_DEVICES": "0,1"}, "16 distinct"),
            ({"P_PORT": "0"}, "Invalid TCP port"),
            ({"MEM_FRACTION_STATIC": "nan"}, "MEM_FRACTION_STATIC"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                result = self.run_launcher(overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Command:", result.stdout)

    def test_actual_launch_generates_only_prefill_config_and_correct_environment(self):
        for index, address in enumerate(IPS):
            with self.subTest(index=index):
                logs = self.root / f"logs-{index}"
                record = self.root / f"launched-{index}.json"
                result = self.run_launcher(
                    dry=False,
                    overrides={
                        "STUB_LOCAL_IPS": address,
                        "STUB_RECORD": str(record),
                        "LOG_DIR": str(logs),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = json.loads(record.read_text())
                command, env = data["args"], data["env"]
                self.assertEqual(env["SGLANG_HOST_IP"], address)
                self.assertEqual(env["HCCL_SOCKET_IFNAME"], "bond0")
                self.assertEqual(env["HCCL_BUFFSIZE"], "1024")
                self.assertEqual(env["ASCEND_MF_STORE_URL"], f"tcp://{IPS[0]}:24707")
                self.assertIsNone(env["SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE"])
                if index == 0:
                    self.assertEqual(env["SGLANG_PP_LAYER_PARTITION"], "18,20,24,16")
                    self.assertEqual(env["TASK_QUEUE_ENABLE"], "1")
                    self.assertEqual(env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"], "0")
                    config = json.loads((logs / "offload.json").read_text())
                    self.assertEqual(config["mode"], "prefill_only")
                    self.option(command, "--moe-offload-config", logs / "offload.json")
                    self.assertIn(str(self.root / "runtime/python"), env["PYTHONPATH"])
                else:
                    self.assertIsNone(env["SGLANG_PP_LAYER_PARTITION"])
                    self.assertEqual(env["TASK_QUEUE_ENABLE"], "0")
                    self.assertEqual(env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"], "1")
                    self.assertEqual(env["SGLANG_NPU_USE_MULTI_STREAM"], "1")
                    self.assertEqual(
                        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"], "32"
                    )
                    self.assertFalse((logs / "offload.json").exists())
                    self.assertNotIn("--moe-offload-config", command)
                    self.assertNotIn(
                        str(self.root / "runtime/python"), env["PYTHONPATH"]
                    )


if __name__ == "__main__":
    unittest.main()
