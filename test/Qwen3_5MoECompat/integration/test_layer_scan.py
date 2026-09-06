"""Unittest discovery entry point for the optional real-layer M3 smoke."""

import unittest

import torch

from .layer_scan import TestLayerScanSmoke, _per_expert_route_report


class TestRouteReport(unittest.TestCase):
    def test_one_bad_route_cannot_hide_in_aggregate(self):
        actual = torch.zeros((32, 8, 2048))
        reference = actual.clone()
        reference[31, 7, 0] = 1.0
        report = _per_expert_route_report(actual, reference)
        self.assertFalse(report["pass"])
        self.assertGreater(report["nrmse"][255], 0.005)

    def test_nan_route_fails_even_when_other_experts_match(self):
        actual = torch.zeros((32, 8, 2048))
        reference = actual.clone()
        actual[22, 4, 17] = float("nan")
        report = _per_expert_route_report(actual, reference)
        self.assertFalse(report["finite"])
        self.assertFalse(report["pass"])


__all__ = ["TestLayerScanSmoke", "TestRouteReport"]
