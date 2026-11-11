import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trace_installer_boot", ROOT / "install.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.site = self.base / "fake site"
        installer.install(self.site)
        package = self.base / "packages"
        self.package = package
        for relative in ("sglang", "sglang/srt", "sglang/srt/managers"):
            directory = package / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("")
        (package / "sglang/srt/managers/scheduler.py").write_text(
            "loaded = True\nclass Scheduler: pass\n"
        )

    def run_script(self, body, enabled="1", check=True):
        env = dict(os.environ, PP_CHUNK_TRACE_ENABLE=enabled, PYTHONDONTWRITEBYTECODE="1")
        # -S isolates the test from real site-packages and installed SGLang.
        code = (
            "import sys, site, types, json\n"
            f"sys.path.insert(0, {str(self.package)!r})\n"
            f"site.addsitedir({str(self.site)!r})\n" + textwrap.dedent(body)
        )
        return subprocess.run(
            [sys.executable, "-B", "-S", "-c", code],
            env=env, text=True, capture_output=True, check=check,
        )

    def test_disabled_does_not_import_bootstrap_torch_or_sglang(self):
        result = self.run_script(
            "print(json.dumps([name for name in sys.modules if "
            "name.startswith(('torch', 'sglang', 'pp_chunk_trace'))]))",
            enabled="0",
        )
        self.assertEqual(json.loads(result.stdout), [])

    def test_enabled_is_lazy_and_runs_after_scheduler_import(self):
        result = self.run_script("""
            assert 'sglang' not in sys.modules
            assert 'torch' not in sys.modules
            import pp_chunk_trace.bootstrap as bootstrap
            assert bootstrap.activate()
            assert sum(isinstance(x, bootstrap._Finder) for x in sys.meta_path) == 1
            patch = types.ModuleType('pp_chunk_trace.patch')
            calls = []
            def install(module):
                assert module.loaded
                assert module.Scheduler
                calls.append(module.__name__)
            patch.install = install
            sys.modules[patch.__name__] = patch
            import sglang.srt.managers.scheduler
            import sglang.srt.managers.scheduler
            assert bootstrap.activate()
            assert bootstrap._finder is None
            assert not any(isinstance(x, bootstrap._Finder) for x in sys.meta_path)
            print(json.dumps(calls))
        """)
        self.assertEqual(json.loads(result.stdout), ["sglang.srt.managers.scheduler"])

    def test_patch_failure_prevents_scheduler_start(self):
        result = self.run_script("""
            patch = types.ModuleType('pp_chunk_trace.patch')
            def install(module):
                raise ValueError('unsupported scheduler signature')
            patch.install = install
            sys.modules[patch.__name__] = patch
            import sglang.srt.managers.scheduler
            print('incorrectly continued')
        """, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported scheduler signature", result.stderr)
        self.assertIn("PP chunk tracing could not be enabled", result.stderr)
        self.assertNotIn("incorrectly continued", result.stdout)

    def test_deactivate_restores_import_behavior(self):
        result = self.run_script("""
            import pp_chunk_trace.bootstrap as bootstrap
            bootstrap.deactivate()
            import sglang.srt.managers.scheduler as scheduler
            assert scheduler.loaded
            assert 'pp_chunk_trace.patch' not in sys.modules
            print('ok')
        """)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_existing_scheduler_can_be_patched_explicitly(self):
        result = self.run_script("""
            import sglang.srt.managers.scheduler as scheduler
            import os
            os.environ['PP_CHUNK_TRACE_ENABLE'] = '1'
            patch = types.ModuleType('pp_chunk_trace.patch')
            calls = []
            patch.install = lambda module: calls.append(module.__name__)
            sys.modules[patch.__name__] = patch
            import pp_chunk_trace.bootstrap as bootstrap
            assert bootstrap.activate()
            assert bootstrap.activate()
            print(json.dumps(calls))
        """, enabled="0")
        self.assertEqual(json.loads(result.stdout), ["sglang.srt.managers.scheduler"])


if __name__ == "__main__":
    unittest.main()
