import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import utils


class _FakeTensor:
    def __init__(self, device_type: str):
        self.device = SimpleNamespace(type=device_type)
        # Simulates torch_npu's compatibility alias without requiring torch_npu.
        self.is_cuda = True
        self.shape = (2, 3)
        self.dtype = torch.float32
        self.to = Mock()
        self.record_stream = Mock()


class TestAsyncD2HDeviceDispatch(unittest.TestCase):
    def test_npu_alias_uses_plain_copy_without_cuda_apis(self):
        source = _FakeTensor("npu")
        host = object()
        source.to.return_value = host

        with (
            patch.object(utils.torch, "empty") as empty,
            patch.object(utils.torch.cuda, "current_stream") as current_stream,
        ):
            result = utils._async_d2h(source)

        self.assertIs(result, host)
        source.to.assert_called_once_with("cpu", non_blocking=True)
        empty.assert_not_called()
        current_stream.assert_not_called()
        source.record_stream.assert_not_called()

    def test_cuda_keeps_pinned_copy_and_stream_lifetime(self):
        source = _FakeTensor("cuda")
        host = Mock()
        stream = object()

        with (
            patch.object(utils.torch, "empty", return_value=host) as empty,
            patch.object(
                utils.torch.cuda, "current_stream", return_value=stream
            ) as current_stream,
        ):
            result = utils._async_d2h(source)

        self.assertIs(result, host)
        empty.assert_called_once_with(source.shape, dtype=source.dtype, pin_memory=True)
        host.copy_.assert_called_once_with(source, non_blocking=True)
        current_stream.assert_called_once_with(source.device)
        source.record_stream.assert_called_once_with(stream)
        source.to.assert_not_called()
