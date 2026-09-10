import unittest
from unittest import mock

import torch

from sglang.srt.layers.qwen3_5.mock_prefetch import (
    DeterministicMockPredictionProvider,
    MockPrefetchConfig,
    MockPrefetchRun,
    MockPrefetchScheduler,
    RecordedRouteOracle,
)


def _oracle(rows=5):
    routes = torch.empty((rows, 40, 8), dtype=torch.uint8)
    for row in range(rows):
        for layer in range(40):
            routes[row, layer] = torch.tensor(
                [(row * 11 + layer * 7 + rank) % 256 for rank in range(8)],
                dtype=torch.uint8,
            )
    return RecordedRouteOracle(routes, tuple(range(rows)))


class TestMockPrefetchConfig(unittest.TestCase):
    def test_recall_is_bounded_by_candidate_budget(self):
        self.assertEqual(MockPrefetchConfig(0.5, 4, 2).top_k, 4)
        with self.assertRaisesRegex(ValueError, "possible|must be in"):
            MockPrefetchConfig(0.51, 4, 2)
        self.assertEqual(MockPrefetchConfig(0.0, 0, 1).top_k, 0)

    def test_record_and_replay_shapes_are_distinct(self):
        self.assertEqual(MockPrefetchRun("record", "p").phase, "record")
        with self.assertRaisesRegex(ValueError, "requires"):
            MockPrefetchRun("replay", "p")


class TestDeterministicProvider(unittest.TestCase):
    def test_exact_recall_unique_candidates_and_wrong_exclusion(self):
        oracle = _oracle()
        config = MockPrefetchConfig(0.5, 8, 2, seed=17)
        provider = DeterministicMockPredictionProvider(oracle, config, pair_id="pair-a")
        true_count = 0
        candidate_count = 0
        for row in range(1, oracle.rows):
            for layer in range(2, 39):
                candidates = provider.candidates(row, layer)
                actual = set(int(x) for x in oracle.expert_ids[row, layer])
                self.assertEqual(len(candidates), 8)
                self.assertEqual(len(set(candidates)), 8)
                true_count += len(set(candidates).intersection(actual))
                candidate_count += len(candidates)
        self.assertEqual(true_count, provider.true_candidates)
        self.assertAlmostEqual(provider.achieved_recall, 0.5)
        self.assertEqual(candidate_count, provider.eligible_groups * 8)

    def test_seed_and_pair_are_stable_inputs(self):
        oracle = _oracle()
        config = MockPrefetchConfig(0.25, 8, 1, seed=3)
        first = DeterministicMockPredictionProvider(oracle, config, pair_id="a")
        same = DeterministicMockPredictionProvider(oracle, config, pair_id="a")
        other = DeterministicMockPredictionProvider(oracle, config, pair_id="b")
        self.assertEqual(first.candidates(1, 1), same.candidates(1, 1))
        self.assertNotEqual(first.candidates(1, 1), other.candidates(1, 1))

    def test_prefill_row_and_unreachable_early_layers_have_no_candidates(self):
        provider = DeterministicMockPredictionProvider(
            _oracle(), MockPrefetchConfig(0.5, 8, 3), pair_id="p"
        )
        self.assertEqual(provider.candidates(0, 3), ())
        self.assertEqual(provider.candidates(1, 2), ())


class TestScheduler(unittest.TestCase):
    def test_router_boundary_enqueues_target_without_synchronizing(self):
        provider = DeterministicMockPredictionProvider(
            _oracle(), MockPrefetchConfig(0.5, 8, 2), pair_id="p"
        )
        store = mock.Mock()
        scheduler = MockPrefetchScheduler(provider, store, request_epoch=9)
        scheduler.begin_decode_row(1)
        event = mock.Mock()
        with mock.patch(
            "sglang.srt.layers.qwen3_5.mock_prefetch.torch.cuda.Event",
            return_value=event,
        ), mock.patch(
            "sglang.srt.layers.qwen3_5.mock_prefetch.torch.cuda.current_stream",
            return_value=mock.sentinel.stream,
        ):
            scheduler.on_router_ready(5)
        event.record.assert_called_once_with(mock.sentinel.stream)
        command, trigger = store.prefetch.call_args.args
        self.assertEqual(command.target_layer, 7)
        self.assertEqual(command.output_row, 1)
        self.assertEqual(command.request_epoch, 9)
        self.assertIs(trigger, event)


if __name__ == "__main__":
    unittest.main()
