import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trace_installer", ROOT / "install.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / "site packages"

    def test_round_trip_is_idempotent_and_preserves_source(self):
        result = installer.install(self.target)
        self.assertTrue(result["installed"])
        self.assertTrue(result["source_exists"])
        before = (self.target / installer.PTH_NAME).read_bytes()
        self.assertEqual(installer.install(self.target), result)
        self.assertEqual((self.target / installer.PTH_NAME).read_bytes(), before)
        self.assertFalse(installer.uninstall(self.target)["installed"])
        self.assertFalse(installer.uninstall(self.target)["installed"])
        self.assertTrue((ROOT / "pp_chunk_trace" / "bootstrap.py").is_file())

    def test_preserves_unrecognized_registration(self):
        self.target.mkdir()
        path = self.target / installer.PTH_NAME
        path.write_text("import unrelated\n", encoding="utf-8")
        for action in (installer.install, installer.uninstall, installer.status):
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                action(self.target)
        self.assertEqual(path.read_text(), "import unrelated\n")

    def test_preserves_symlink(self):
        self.target.mkdir()
        other = self.target / "other"
        other.write_text("do not edit")
        (self.target / installer.PTH_NAME).symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlink"):
            installer.uninstall(self.target)
        self.assertEqual(other.read_text(), "do not edit")

    def test_source_can_contain_spaces_and_unicode(self):
        source = Path(self.tmp.name) / "新 package directory"
        (source / "pp_chunk_trace").mkdir(parents=True)
        (source / "pp_chunk_trace" / "bootstrap.py").write_text("# placeholder\n")
        result = installer.install(self.target, source=source)
        self.assertEqual(result["source"], str(source.resolve()))
        with self.assertRaisesRegex(ValueError, "Already installed"):
            installer.install(self.target)
        self.assertEqual(installer.status(self.target)["source"], str(source.resolve()))

    def test_newline_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "newlines"):
            installer._payload("/tmp/a\nb")

    def test_install_race_does_not_replace_new_file(self):
        from unittest.mock import patch

        real_link = os.link

        def race(source, destination):
            Path(destination).write_text("another process owns this\n")
            return real_link(source, destination)

        with patch.object(installer.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                installer.install(self.target)
        self.assertEqual(
            (self.target / installer.PTH_NAME).read_text(), "another process owns this\n"
        )
        self.assertFalse(list(self.target.glob(".pp_chunk_trace-*")))

    def test_cli_with_explicit_target(self):
        for action, expected in (("install", True), ("status", True), ("uninstall", False)):
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "install.py"), action,
                 "--target", str(self.target)],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(result.stdout)["installed"], expected)


if __name__ == "__main__":
    unittest.main()
