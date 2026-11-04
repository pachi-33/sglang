"""Exercise the real PD launcher with only stdlib, mocked interfaces and no network commands and dry-run."""

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
        # A closed PATH excludes ip/ifconfig/hostname on every test host and
        # prevents inherited model/fabric/Ascend configuration affecting results.
        for name, executable in (
            ("dirname", shutil.which("dirname")),
            ("date", shutil.which("date")),
            ("cat", shutil.which("cat")),
        ):
            (self.bin / name).symlink_to(executable)
        python = self.bin / "python3"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
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
            "STUB_ENUM_CALLS": str(self.calls),
        }
        self._set_local(_IPS[0])

    def _set_local(self, *addresses):
        self.env["STUB_INTERFACES"] = json.dumps(
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
            ((_IPS[0],), {"D0_IP": _IPS[0]}, "distinct node addresses"),
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

    def test_native_enumeration_failure_is_reported(self):
        result = self._run(overrides={"STUB_ENUM_ERROR": "simulated failure"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot identify local PD node", result.stderr)
        self.assertIn("simulated failure", result.stderr)

    def test_same_ip_on_multiple_interfaces_needs_override(self):
        interfaces = json.loads(self.env["STUB_INTERFACES"])
        interfaces.append(_interface("other0", _IPS[0]))
        self.env["STUB_INTERFACES"] = json.dumps(interfaces)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("multiple interfaces", result.stderr)
        self._command(self._run(overrides={"COMM_IFACE": "bond0"}))

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
                            "STUB_ENUM_ERROR": "must not enumerate",
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
                overrides={
                    "MODEL_PATH": "",
                    "D1_IP": "",
                    "STUB_ENUM_ERROR": "must not enumerate",
                },
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
