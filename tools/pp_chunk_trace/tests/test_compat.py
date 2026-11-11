import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pp_chunk_trace import compat


class CompatibilityTests(unittest.TestCase):
    def test_ignores_layout_and_comments_but_not_behavior(self):
        a = "def f(x):\n    return x + 1\n"
        self.assertEqual(compat.fingerprint(a), compat.fingerprint("def f(x): # hi\n\n    return x+1"))
        self.assertNotEqual(compat.fingerprint(a), compat.fingerprint("def f(x):\n    return x + 2"))

    def test_python312_empty_type_params_do_not_change_hash(self):
        source = "def f(x):\n    return x\n"
        expected = compat.fingerprint(source)
        tree = ast.parse(source)
        function = tree.body[0]
        function._fields = function._fields + ("type_params",)
        function.type_params = []
        with mock.patch.object(compat.ast, "parse", return_value=tree):
            self.assertEqual(expected, compat.fingerprint(source))

    def test_static_boundary_check_does_not_import_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "class S:\n    def f(self):\n        return 1\n"
            (root / "source.py").write_text(source)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(reference_commit="test", methods=[dict(
                path="source.py", **{"class": "S", "method": "f"},
                sha256=compat.fingerprint("def f(self):\n    return 1"))])))
            with mock.patch.object(compat, "MANIFEST", manifest):
                self.assertTrue(compat.check(root)["compatible"])
                (root / "source.py").write_text(source.replace("return 1", "return 2"))
                self.assertFalse(compat.check(root)["compatible"])


if __name__ == "__main__":
    unittest.main()
