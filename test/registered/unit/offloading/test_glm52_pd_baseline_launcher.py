"""Check the resident 2P/2D launcher without Ascend libraries or a real network."""

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts/moe_offload/serve_glm52_pd_baseline.sh"
)
_IPS = ("10.20.30.10", "10.20.30.11", "10.20.30.12", "10.20.30.13")
_BASH = shutil.which("bash")


def _interface(name, *addresses):
    return {
        "ifname": name,
        "addr_info": [{"family": "inet", "local": address} for address in addresses],
    }


class TestGlm52PdBaselineLauncher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glm52-pd-baseline-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("dirname", "date", "cat", "tee", "mkdir"):
            (self.bin / name).symlink_to(shutil.which(name))
        # Run inline metadata/detection Python normally. Replace only the final
        # server process, so non-dry-run exercises CANN sourcing and logging too.
        python = self.bin / "python3"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if sys.argv[1:4] in (['-u', '-m', 'sglang.launch_server'], "
            "['-u', '-m', 'sglang_router.launch_router']):\n"
            "    pathlib.Path(os.environ['STUB_LAUNCH']).write_text(json.dumps({\n"
            "        'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
            "    sys.exit(0)\n"
            "if len(sys.argv) > 1 and sys.argv[1].endswith('/glm52_mtp_config.py') "
            "and os.environ.get('STUB_MTP_OK') == '1':\n"
            "    pathlib.Path(os.environ['STUB_MTP_CALL']).write_text(json.dumps(sys.argv[1:]))\n"
            "    sys.exit(0)\n"
            "if len(sys.argv) > 1 and sys.argv[1].endswith('/pd_node.py'):\n"
            "    sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))\n"
            "    import pd_node\n"
            "    from unittest.mock import patch\n"
            "    def enumerate_interfaces():\n"
            "        pathlib.Path(os.environ['STUB_ENUM_CALLS']).touch()\n"
            "        if os.environ.get('STUB_ENUM_ERROR'):\n"
            "            raise OSError(os.environ['STUB_ENUM_ERROR'])\n"
            "        local = {}\n"
            "        for interface in json.loads(os.environ['STUB_INTERFACES']):\n"
            "            for info in interface.get('addr_info', []):\n"
            "                if info.get('family') == 'inet':\n"
            "                    local.setdefault(info['local'], set()).add(interface['ifname'])\n"
            "        return local\n"
            "    with patch.object(pd_node, 'local_ipv4_interfaces', side_effect=enumerate_interfaces):\n"
            "        pd_node.main(sys.argv[2:])\n"
            "    sys.exit(0)\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "glm_moe_dsa",
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "n_routed_experts": 256,
                }
            ),
            encoding="utf-8",
        )
        environment = self.root / "ascend_env.sh"
        environment.write_text("export STUB_ASCEND_READY=1\n", encoding="utf-8")
        self.env = {
            "PATH": str(self.bin),
            "LC_ALL": "C",
            "MODEL_PATH": str(self.model),
            # Routing/capacity cases use a checkpoint without bundled MTP.
            "ENABLE_MTP": "0",
            **dict(zip(("P0_IP", "P1_IP", "D0_IP", "D1_IP"), _IPS)),
            "CANN_ENV": str(environment),
            "ATB_ENV": str(environment),
            "STUB_ENUM_CALLS": str(self.root / "enum-called"),
            "STUB_MTP_CALL": str(self.root / "mtp-called"),
            "RUNTIME_DIR": str(self.root / "unused-offload-runtime"),
        }
        self.invocation = 0
        self._local(_IPS[0])

    def _local(self, *addresses):
        self.env["STUB_INTERFACES"] = json.dumps(
            [
                _interface("lo", "127.0.0.1"),
                _interface("management0", "192.168.99.20"),
                _interface("bond0", "172.20.1.2", *addresses),
            ]
        )

    def _run(self, *args, dry=True, overrides=None):
        self.invocation += 1
        self.logs = self.root / f"logs {self.invocation}"
        self.capture = self.root / f"launch-{self.invocation}.json"
        result = subprocess.run(
            [_BASH, str(_SCRIPT), *args, *(["--dry-run"] if dry else [])],
            env={
                **self.env,
                "LOG_DIR": str(self.logs),
                "STUB_LAUNCH": str(self.capture),
                **(overrides or {}),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertFalse((self.root / "unused-offload-runtime").exists())
        self.assertFalse(list(self.root.rglob("offload.json")))
        if dry:
            self.assertFalse(self.logs.exists())
            self.assertFalse(self.capture.exists())
        return result

    def _command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(s for s in result.stdout.splitlines() if s.startswith("Command:"))
        return shlex.split(line[len("Command:") :])

    def _option(self, command, name, value):
        self.assertEqual(command[command.index(name) + 1], str(value))

    def test_four_nodes_share_script_and_form_two_resident_instances(self):
        for index, address in enumerate(_IPS):
            for dry in (True, False):
                with self.subTest(index=index, dry=dry):
                    self._local(address)
                    command = self._command(self._run(dry=dry))
                    prefill = index < 2
                    role = "prefill" if prefill else "decode"
                    self._option(command, "--disaggregation-mode", role)
                    self._option(command, "--nnodes", 2)
                    self._option(command, "--node-rank", index % 2)
                    self._option(command, "--tp-size", 32)
                    self._option(command, "--ep-size", 32)
                    self._option(command, "--attn-dp-size", 8 if prefill else 32)
                    master = _IPS[0] if prefill else _IPS[2]
                    self._option(command, "--dist-init-addr", f"{master}:5000")
                    self._option(command, "--model-path", self.model)
                    self._option(command, "--moe-a2a-backend", "deepep")
                    self._option(
                        command, "--deepep-mode", "normal" if prefill else "low_latency"
                    )
                    self._option(
                        command, "--max-running-requests", 64 if prefill else 128
                    )
                    self._option(command, "--context-length", 180000)
                    self.assertNotIn("--speculative-algorithm", command)
                    self.assertNotIn("--moe-offload-config", command)
                    self.assertNotIn("--cpu-offload-gb", command)
                    self.assertNotIn("--disable-overlap-schedule", command)
                    if prefill:
                        self._option(command, "--chunked-prefill-size", 524288)
                        self._option(command, "--max-prefill-tokens", 180000)
                        self._option(command, "--cuda-graph-backend-decode", "disabled")
                        self._option(
                            command, "--cuda-graph-backend-prefill", "disabled"
                        )
                    else:
                        self._option(command, "--cuda-graph-backend-decode", "full")
                        self._option(command, "--cuda-graph-max-bs-decode", 4)
                    if not dry:
                        launch = json.loads(self.capture.read_text())
                        self.assertEqual(launch["args"], command[1:])
                        env = launch["env"]
                        self.assertEqual(env["STUB_ASCEND_READY"], "1")
                        self.assertEqual(env["HCCL_SOCKET_IFNAME"], "bond0")
                        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "bond0")
                        self.assertEqual(
                            env["ASCEND_MF_STORE_URL"], f"tcp://{_IPS[0]}:24707"
                        )
                        self.assertEqual(
                            env["TASK_QUEUE_ENABLE"], "2" if prefill else "0"
                        )
                        self.assertEqual(
                            env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"],
                            "0" if prefill else "1",
                        )
                        if prefill:
                            self.assertEqual(
                                env["DEEP_NORMAL_MODE_USE_INT8_QUANT"], "1"
                            )
                        else:
                            self.assertEqual(env["HCCL_BUFFSIZE"], "650")

    def test_auto_alias_and_role_specific_overrides(self):
        for index, setting, value in (
            (0, "P_MAX_RUNNING_REQUESTS", "12"),
            (3, "D_MAX_RUNNING_REQUESTS", "48"),
        ):
            with self.subTest(index=index):
                self._local(_IPS[index])
                command = self._command(self._run("auto", overrides={setting: value}))
                self._option(command, "--max-running-requests", value)

    def test_invalid_identity_and_capacity_fail_before_launch(self):
        cases = (
            ((_IPS[0],), {"P1_IP": "invalid"}, "Set P1_IP"),
            ((_IPS[0],), {"P1_IP": _IPS[0]}, "distinct node addresses"),
            (("192.168.99.21",), {}, "exactly one local node IP match"),
            ((_IPS[0], _IPS[2]), {}, "exactly one local node IP match"),
            ((_IPS[1],), {"NODE_RANK": "0"}, "conflicts with detected P1_IP"),
            ((_IPS[0],), {"COMM_IFACE": "management0"}, "does not own detected IP"),
            ((_IPS[0],), {"CHUNKED_PREFILL_SIZE": "8193"}, "divisible by DP8"),
            ((_IPS[0],), {"MEM_FRACTION_STATIC": "nan"}, "MEM_FRACTION_STATIC"),
            ((_IPS[0],), {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}, "16 distinct"),
            ((_IPS[0],), {"P_DIST_PORT": "0"}, "Invalid TCP port"),
        )
        for local, overrides, message in cases:
            with self.subTest(overrides=overrides, local=local):
                self._local(*local)
                result = self._run(overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Command:", result.stdout)

    def test_mtp_is_checked_and_preserves_official_role_parameters(self):
        for index in (0, 2):
            with self.subTest(index=index):
                self._local(_IPS[index])
                command = self._command(
                    self._run(overrides={"ENABLE_MTP": "", "STUB_MTP_OK": "1"})
                )
                self._option(command, "--speculative-algorithm", "NEXTN")
                self._option(command, "--speculative-num-steps", 1 if index == 0 else 3)
                self._option(
                    command, "--speculative-num-draft-tokens", 2 if index == 0 else 4
                )
                self._option(
                    command, "--speculative-draft-model-quantization", "unquant"
                )
                checked = json.loads((self.root / "mtp-called").read_text())
                self.assertEqual(
                    checked[1:], [str(self.model), "--draft-quantization", "unquant"]
                )
        result = self._run(overrides={"ENABLE_MTP": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Command:", result.stdout)

    def test_router_registers_only_instance_leaders(self):
        for dry in (True, False):
            with self.subTest(dry=dry):
                command = self._command(
                    self._run(
                        "router",
                        dry=dry,
                        overrides={
                            "MODEL_PATH": "",
                            "P1_IP": "",
                            "D1_IP": "",
                            "CANN_ENV": "/missing",
                        },
                    )
                )
                self.assertIn("sglang_router.launch_router", command)
                self._option(command, "--prefill", f"http://{_IPS[0]}:8000")
                self._option(command, "--decode", f"http://{_IPS[2]}:8003")
                self.assertEqual(command.count("--prefill"), 1)
                self.assertEqual(command.count("--decode"), 1)
                self.assertFalse((self.root / "enum-called").exists())


if __name__ == "__main__":
    unittest.main()
