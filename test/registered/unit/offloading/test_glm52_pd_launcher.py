"""Exercise the real PD launcher with only stdlib, a fake ip command and dry-run."""

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts/moe_offload/serve_glm52_pd.sh"
_BASH = shutil.which("bash")
_IPS = ("10.20.30.10", "10.20.30.11", "10.20.30.12")


def _interface(name, *addresses):
    return {
        "ifname": name,
        "addr_info": [{"family": "inet", "local": address} for address in addresses],
    }


class TestGlm52PdLauncher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glm52-pd-launcher-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # A closed PATH makes missing-ip tests independent of the host OS and
        # prevents inherited model/fabric/Ascend configuration affecting results.
        for name, executable in (
            ("python3", sys.executable),
            ("dirname", shutil.which("dirname")),
            ("date", shutil.which("date")),
            ("cat", shutil.which("cat")),
        ):
            (self.bin / name).symlink_to(executable)
        self.ip = self.bin / "ip"
        self.ip.write_text(
            f"#!{sys.executable}\n"
            "import os, pathlib, sys\n"
            "assert sys.argv[1:] == ['-j', '-4', 'addr', 'show', 'up']\n"
            "pathlib.Path(os.environ['STUB_IP_CALLS']).write_text('called')\n"
            "print(os.environ['STUB_IP_RESULT'])\n"
            "sys.exit(int(os.environ.get('STUB_IP_EXIT', '0')))\n",
            encoding="utf-8",
        )
        self.ip.chmod(0o755)
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
        self.calls = self.root / "ip-calls"
        self.env = {
            "PATH": str(self.bin),
            "LC_ALL": "C",
            "MODEL_PATH": str(self.model),
            "P_IP": _IPS[0],
            "D0_IP": _IPS[1],
            "D1_IP": _IPS[2],
            "RUNTIME_DIR": str(self.root / "runtime"),
            "LOG_DIR": str(self.root / "logs"),
            "STUB_IP_CALLS": str(self.calls),
        }
        self._set_local(_IPS[0])

    def _set_local(self, *addresses):
        self.env["STUB_IP_RESULT"] = json.dumps(
            [
                _interface("lo", "127.0.0.1"),
                _interface("management0", "192.168.99.20"),
                _interface("bond0", "172.20.1.2", *addresses),
            ]
        )

    def _run(self, *args, overrides=None):
        result = subprocess.run(
            [_BASH, str(_SCRIPT), *args, "--dry-run"],
            env={**self.env, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=15,
        )
        # This also catches regressions that create config/build/log files
        # before dry-run finishes, including rejected configurations.
        self.assertFalse((self.root / "runtime").exists())
        self.assertFalse((self.root / "logs").exists())
        self.assertFalse(list(self.model.rglob("__pycache__")))
        return result

    def _command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("Command:")
        )
        return shlex.split(line[len("Command:") :])

    def _assert_option(self, command, name, expected):
        self.assertEqual(command[command.index(name) + 1], str(expected))

    def test_auto_roles_use_all_local_addresses_and_infer_interface(self):
        for node, ip in enumerate(_IPS):
            for args in ((), ("auto",)):
                with self.subTest(node=node, args=args):
                    self._set_local(ip)
                    result = self._run(*args)
                    command = self._command(result)
                    role = "prefill" if node == 0 else "decode"
                    self.assertIn(f"IP={ip} role={role}", result.stdout)
                    self.assertIn("communication interface: bond0", result.stdout)
                    self._assert_option(command, "--model-path", self.model)
                    self._assert_option(command, "--disaggregation-mode", role)
                    self._assert_option(command, "--node-rank", max(node - 1, 0))
                    self._assert_option(command, "--nnodes", 1 if node == 0 else 2)
                    self._assert_option(command, "--tp-size", 16 if node == 0 else 32)
                    self._assert_option(
                        command, "--attn-dp-size", 4 if node == 0 else 32
                    )
                    self._assert_option(command, "--ep-size", 16 if node == 0 else 32)
                    self._assert_option(
                        command, "--max-running-requests", 64 if node == 0 else 128
                    )
                    if node == 0:
                        self._assert_option(command, "--context-length", 180000)
                        self._assert_option(command, "--chunked-prefill-size", 65536)
                        self.assertIn("--moe-offload-config", command)
                        self.assertIn('"mode": "prefill_only"', result.stdout)
                        self.assertIn('"cache_ratio": 0', result.stdout)
                    else:
                        self.assertNotIn("--moe-offload-config", command)
                        self._assert_option(
                            command, "--dist-init-addr", f"{_IPS[1]}:5000"
                        )

    def test_invalid_auto_configuration_fails_before_launch(self):
        cases = (
            (("192.168.99.21",), {}, "exactly one local node IP match"),
            ((_IPS[0], _IPS[1]), {}, "exactly one local node IP match"),
            ((_IPS[0],), {"D0_IP": _IPS[0]}, "three distinct node addresses"),
            ((_IPS[2],), {"NODE_RANK": "0"}, "conflicts with detected D1_IP"),
            ((_IPS[0],), {"COMM_IFACE": "management0"}, "does not own detected IP"),
            ((_IPS[0],), {"D1_IP": "invalid"}, "Set D1_IP to a valid node IPv4"),
            ((_IPS[0],), {"D1_IP": "127.0.0.1"}, "non-loopback unicast IPv4"),
        )
        for local_ips, overrides, message in cases:
            with self.subTest(overrides=overrides, local_ips=local_ips):
                self._set_local(*local_ips)
                result = self._run(overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Command:", result.stdout)

    def test_automatic_matching_accepts_consistent_overrides(self):
        self._set_local(_IPS[2])
        command = self._command(
            self._run(overrides={"NODE_RANK": "1", "COMM_IFACE": "bond0"})
        )
        self._assert_option(command, "--node-rank", 1)

    def test_missing_ip_reports_explicit_role_fallback(self):
        self.ip.unlink()
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Linux iproute2", result.stderr)
        self.assertIn("explicit role with COMM_IFACE", result.stderr)

    def test_failed_or_invalid_ip_output_is_rejected(self):
        for overrides in ({"STUB_IP_EXIT": "1"}, {"STUB_IP_RESULT": "not JSON"}):
            with self.subTest(overrides=overrides):
                result = self._run(overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Cannot read local IPv4 addresses", result.stderr)

    def test_explicit_roles_do_not_require_d1_or_scan_addresses(self):
        for role, rank in (("prefill", "0"), ("decode", "0"), ("decode", "1")):
            with self.subTest(role=role, rank=rank):
                command = self._command(
                    self._run(
                        role,
                        overrides={
                            "D1_IP": "",
                            "COMM_IFACE": "explicit0",
                            "NODE_RANK": rank,
                            "STUB_IP_EXIT": "99",
                        },
                    )
                )
                self._assert_option(command, "--disaggregation-mode", role)
                self._assert_option(command, "--node-rank", rank)
                self.assertFalse(self.calls.exists())

    def test_router_needs_neither_model_nor_local_ip_scan(self):
        command = self._command(
            self._run(
                "router",
                overrides={"MODEL_PATH": "", "D1_IP": "", "STUB_IP_EXIT": "99"},
            )
        )
        self.assertIn("sglang_router.launch_router", command)
        self._assert_option(command, "--prefill", f"http://{_IPS[0]}:8000")
        self._assert_option(command, "--decode", f"http://{_IPS[1]}:8003")
        self.assertFalse(self.calls.exists())

    def test_unknown_arguments_and_multiple_roles_are_rejected(self):
        for args in (("--typo",), ("auto", "prefill")):
            with self.subTest(args=args):
                result = self._run(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ERROR:", result.stderr)
                self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
