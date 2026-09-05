import importlib.metadata
import unittest

from pyairports.airports import AIRPORT_LIST


class PyAirportsCompatibilityTest(unittest.TestCase):
    def test_local_distribution_is_installed(self):
        self.assertEqual(
            importlib.metadata.version("pyairports"), "0.0.1+sglang.v100.1"
        )

    def test_legacy_airport_rows(self):
        codes = [airport[3] for airport in AIRPORT_LIST]
        self.assertGreater(len(codes), 1_000)
        self.assertEqual(codes, sorted(set(codes)))
        self.assertIn("SIN", codes)
        self.assertIn("SFO", codes)

    def test_outlines_airport_enum_imports(self):
        from outlines.types.airports import IATA

        self.assertEqual(IATA.SIN.value, "SIN")


if __name__ == "__main__":
    unittest.main()
